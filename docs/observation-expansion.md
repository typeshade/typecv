# Image-based expansion of sparse observations

Status: this bounded batch stage is implemented and locally validated. Baseline: `e4e5046`.

The existing `ImageFeatures.keypoints` accepts continuous measured coordinates;
the core mapper does not require independent detector output. Existing KLT has
pyramids, forward/backward checking, NCC, spatial fill-in and fresh SIFT seeds at
joins. Its current join emits pairs only when a track reaches an existing SIFT
keypoint. Fill-in therefore does not by itself register detector-missed coordinates
as SfM observations. The pair LK API also starts its search at zero displacement.

This stage reuses those kernels and the mapper's geometric verification. It adds
an initial-target option and an explicit batch adapter that starts tracks from
multiple input frames, searches both earlier and later neighbours, and appends
image-measured coordinates while preserving all original feature indices.
Estimated rotations or finite geometry may initialize a search. Predictions are
never emitted directly as observations. Patch/round-trip checks remain mandatory;
the mapper must subsequently verify new pair candidates before accepting structure.

The first adapter uses a bounded frame neighbourhood, existing per-frame seeds,
and the existing translation-patch LK model. General affine patch warping,
occlusion inference, learned tracking, arbitrary track merging, and streaming
retroactive session updates are separate work. This does not implement Dense-SfM
or claim equivalent results. PR #148 owns photo pair selection and remains separate.

## Tasks

- [x] E1: audit existing KLT, join and mapper ingestion; compare numerical-only
  refinement with missing-observation recovery and identify the gap.
- [x] E2: add predicted initial coordinates to existing CPU/GPU LK, preserve the
  default path, and test that image evidence overrides or rejects predictions.
- [x] E3: add multi-source, forward/backward observation expansion with append-only
  indices, ownership/duplicate checks, provenance and explicit candidate geometry.
- [x] E4: compare original/new observation counts, track lengths, distinct landmarks,
  coverage and accuracy on RGB-derived input; test occlusion, textureless regions,
  wrong predictions, rotation and insufficient parallax on CPU and WebGPU.
- [x] E5: finish documentation, execution/memory evidence and task/PR updates.

No source pose/depth truth is supplied to RGB-only reconstruction. Known-K and
estimated-K experiments must remain distinct. A longer track is not a new landmark,
and a point propagated under pure rotation still has unresolved depth.

E2 validation: 15 scalar-reference tests, 15 TypeShade CPU tests (including default
bit equality), and one SwiftShader WebGPU prediction/chunk-tail test pass. A one-level
tracker recovers a 30 px translation from an inexact prediction; flat pixels and
out-of-frame predictions are rejected. Forward/backward buffers also hold the
initial guesses, so no additional resident point array is allocated.

E3 validation: three TypeShade CPU tests and one SwiftShader WebGPU test pass.
The adapter recovers 24 coordinates from early and late seeds without changing
original indices. Flat/occluded pixels, local repeated texture, zero valid masks
and known shot boundaries produce no coordinates. Tests also cover inexact finite
map predictions, invalid depth projections, ownership conflicts and append limits.
All new rows remain geometric candidates until mapper verification.

Follow-up: [patch geometry, resident pyramids and cohort evaluation](observation-patch-cache.md).

## Reused code and the supported boundary

| Area | Baseline evidence | This stage |
| --- | --- | --- |
| Continuous coordinates | `src/reconstruction/types.ts::ImageFeatures`; `src/reconstruction/tracks.ts::validateFeatures` | Reused; original rows remain an exact prefix, new rows have no invented descriptors. |
| Pyramidal image tracking | `src/gpu/klt.ts::trackKltReference`; `src/gpu/klt.shade.ts::kltTrack`; `src/gpu/klt-device.ts::trackKltGpu` | Optional absolute target initialization; the existing NCC, structure, convergence and round-trip tests remain active. |
| Streaming fill-in | `src/gpu/klt-tracker.ts::createKltTracker`; `website/src/lib/klt-join.ts` | Existing detector joins remain unchanged. Batch expansion can append missing coordinates; retroactive streaming mutation is not implemented. |
| Ownership and geometry | `src/reconstruction/tracks.ts::buildTracks`; mapper pair verification, triangulation, continuation and BA | Verified input edges establish owners. New edges are candidates until normal mapper verification. Conflicting established owners are deferred. |
| Rotation without structure | `src/reconstruction/tracking.ts::PairRotationEvidence`; `src/geometry/rotation.ts::estimateRelativeRotation` | Existing relative bearing rotations initialize searches before a finite map exists. They do not supply a world camera centre or depth. |
| Refinement of an accepted map | `src/reconstruction/mapper.ts::incrementalSteps` resume path | `refineIncremental` / `refineIncrementalGpu` retain its seed gauge and add valid observations to already registered views before triangulation/BA. This completion is explicit and does not change streaming defaults. |

No library or trained model was added. The adapter's seeds are the original points
of **each** input frame, so later-discovered points can be searched in earlier
frames. It prioritizes missing observations, distributes the finite seed budget
across image cells, and searches each anchor directly instead of recursively
turning projections into observations. It does not select a globally best patch,
infer arbitrary occlusions, or merge independently established track owners.

## API and evidence

```ts
import { reconstructIncremental, refineIncremental } from 'typecv';
import { expandObservationsGpu } from 'typecv/gpu';

// frames retain RGB-derived gray, dimensions, K and original keypoints.
const initial = reconstructIncremental(frames, descriptorCandidates, controls);
const expanded = await expandObservationsGpu(device, frames, descriptorCandidates, {
  verifiedPairs: initial.verifiedPairs,
  initialModel: initial.reconstruction,
  rotations: initial.tracking?.rotations,
  radius: 1,
});
const refined = refineIncremental(
  expanded.images, expanded.candidatePairs, initial, controls,
);
```

`refineIncrementalGpu(device, images, candidates, initial, controls)` runs numerical
geometry on the selected TypeShade backend. The RGB comparison isolates tracking
by using CPU geometry for both runs; it is not a full GPU pipeline benchmark.
If initialization is deferred, retain the mapper error's `tracking` evidence and
supply its `rotations` to expansion, then attempt normal initialization again.
Never replace that deferred state with a fabricated finite map.

`newObservations` records the source frame/feature, measured subpixel coordinate,
NCC, round-trip error, local peak margin and prediction type. Its geometry status
is always `candidate`. A candidate can be rejected by pair verification, ownership,
cheirality, parallax, triangulation or final filtering. Counts of candidates,
verified tracks and finite landmarks therefore remain separate.

Coordinates use the existing top-left image axes (x right, y down), integer pixel
centres and row-major rotations. Poses obey `q_camera = R X_world + t`; monocular
world scale remains arbitrary. Frame dimensions, gray pixels, keypoints, masks and
K must describe the same analysis image. Different-sized neighbours are deferred;
this adapter does not silently resize or crop them. `validMask` is caller-supplied
and only admits patches fully inside the valid region; it does not classify motion.
The caller must also filter original candidates or use the mapper's observation
masks. Existing original measurements and raw images are retained.

## Quality gates and remaining limits

- NCC, source structure, convergence and forward/backward checking measure image
  evidence. A local second-peak NCC test also rejects competing matches separated
  by at least four pixels within a configurable radius. It cannot rule out
  repetition outside that radius or repeated assets elsewhere in the scene.
- Rotation/K/finite-point projections initialize **centres**, not whole patch
  warps. In a synthetic 12-degree roll, nine of twelve seeds pass and the largest
  accepted coordinate error is about 0.45 px. This is evidence for that case, not
  affine, zoom or large-viewpoint invariance. Affine/plane-aware patches and
  epipolar interval search remain follow-up work.
- Known shot labels prevent crossing any intervening boundary. Automatic cuts,
  dissolves, relocalization and cross-shot search are not added here.
- The accepted model must be one connected coordinate system. Refinement retains
  every registered view, the original seed gauge and original measurement indices;
  stale indices, removed registered views and a requested new gauge are rejected.
- A new map can still gain or lose landmarks during geometric filtering. More
  observations alone do not establish better geometry. Unknown K, dynamic objects,
  reflection, repeated textures and nonrigid surfaces still need their own policies.
- No MVS, dynamic object shape, Gaussian rendering or original game mesh recovery
  is claimed. Output is sparse geometry plus retained unresolved 2D evidence.

## Execution and next work

LK pyramids/gradients, tracking and the round-trip kernel use TypeShade CPU or
WebGPU. The host manages frame/track ownership, mask checks and local NCC ambiguity.
Predictions reuse the forward/backward resident arrays. The first batch adapter
rebuilds the two pyramids for each ordered neighbour pair and reads three four-word
record arrays (48 bytes per attempted seed); it does not read all descriptors or
images back from the GPU. The RGB images are already retained on the host.

Reported transfers are application counters, not physical bus measurements.
Device peaks exclude browser images, CPU geometry, staging and driver memory.
Software WebGPU measurements identify Google SwiftShader; no physical GPU result
or WebGL2 execution is claimed. Reusing resident frame pyramids, reducing record
readbacks, affine patch warping, geometry-conditioned intervals, and a bounded
streaming image cache are the next implementation tasks. They need independent
quality and memory measurements before becoming defaults.


## RGB comparisons (2026-10-24)

Numerical implementation: `f12bdd4`, baseline `e4e5046`, Node 24.19.0,
Chromium 154.0.8037.57 and TypeShade pin
`d36ac5027356b96e87a9ad900dfd8e6ea4ab55dc`. All WebGPU results below use
Google SwiftShader (`fallback=true`). CPU geometry is identical in both sides of
each comparison. The RGB renderer's poses/depths/point identities stay outside the
reconstruction process; only RGB and declared calibration enter it. Graham Hall
uses cached RGB-derived features/matches and supplied COLMAP calibration. The
video window uses fresh RGB features, manual crop/mask and **assumed**, unverified
calibration. There is no object/depth prior or trained tracker.

The table gives the WebGPU tracking run. A landmark is an accepted mapper track,
not proof of a new physical point under repeated or disconnected appearances.

| Input | Registered frames | Finite landmarks | Finite observations | Pending tracks | New measured coordinates |
| --- | --- | --- | --- | --- | --- |
| Rendered translation, known K | 4 → 4 | 141 → 255 | 331 → 606 | 0 → 0 | 125 |
| Same RGB, controlled detector omissions | 2 → 4 | 21 → 170 | 42 → 399 | 6 → 0 | 160 |
| Rendered rotation, known K | 0 → 0 | 0 → 0 | 0 → 0 | 191 → 345 | 177 |
| Rendered low parallax, known K | 0 → 0 | 0 → 0 | 0 → 0 | 210 → 371 | 192 |
| Graham Hall, 12 existing photos | 12 → 12 | 2423 → 2471 | 5571 → 5899 | 365 → 342 | 322 |
| Uploaded Shorts window, assumed K + manual mask | 6 → 6 | 202 → 471 | 488 → 1253 | 9 → 79 | 526 |

Normal rendered translation gains 114 landmark tracks with no baseline landmark
observation. Graham Hall gains 52 such tracks and loses four previously finite
tracks during filtering, giving a net gain of 48. The video window gains 269
tracks, retaining its original 202. This distinguishes added coordinates,
longer/connected tracks, and newly triangulated landmark tracks. It does not claim
a globally unique physical identity for every point.

- Rendered translation: fixed-seed camera-centre RMSE 0.01003 → 0.00661 scene
  units; maximum rotation error 0.1312° → 0.1481°. Median relative
  depth error **1.55% → 2.18%**; fraction above 10% error 0.71% → 0.39%.
  Additional weak points increase coverage while median depth precision worsens.
- Controlled omissions: two → four registered frames; median relative depth error
  4.08% → 2.31%. Camera RMSE 0.00226 over the original two frames versus 0.00644
  over all four is **not** an accuracy gain over the same population.
- Pure rotation and low parallax: verified 2D tracks 191 → 345 and 210 → 371,
  respectively; both retain zero finite landmarks and deferred initialization.
- Graham Hall: median rotation difference to COLMAP 0.2111° → 0.2038°;
  median centre error / reference trajectory radius **0.154258 → 0.154657**.
  Four-by-four image-cell coverage is unchanged. This is more sparse support,
  not a demonstrated position-accuracy gain.
- The video is only six resampled 10 fps frames near 8 seconds of the 41-second
  upload, at 360×480 after an explicit crop. A manual background rectangle excludes
  titles, HUD and the vehicle; no automatic segmentation or cut inference is used.
  Reprojection error 0.321 → 0.336 px is not geometric ground truth. No calibrated
  pose/depth labels exist for this clip, so its geometric accuracy is unmeasured.
  The remaining 79 tracks are unresolved rather than forced into the map.

Both CPU and WebGPU tracking produce 255 landmarks on rendered translation and
2,471 on cached Graham Hall. The controlled-omission case produces 172 versus 170;
rotation starts with 190 versus 191 tracks. These threshold-sensitive differences
are recorded, not hidden by claiming exact RGB pipeline parity. Low-level
tracking has separate CPU-reference and WebGPU pixel-error/chunk tests.

All camera/depth evaluation uses one fixed seed gauge per connected synthetic map.
Graham Hall uses one whole-map Sim(3) alignment to the COLMAP reference. No per-frame
alignment is applied. Rendered depth uses camera +Z and the nearest source pixel;
quantization and surface-boundary errors are included, and source point identity
is not provided. Actual-video geometry is not scored from reprojection alone.

**Defaults remain unchanged.** The observed gains are substantial on missing
observations, but position/depth precision is not consistently better, compute
cost increases, and general gameplay/zoom/dynamics remain unvalidated.

## Timing and memory

These are single cold browser runs, not a hardware speedup or steady-state rate.
Totals include browser-side image decode/gray conversion, fresh features/matches
where applicable, baseline reconstruction, expansion and refinement. They exclude
browser startup, transport of prepared inputs and synthetic RGB generation.
Cached Graham Hall excludes its original detector/matcher time. The video receipt
separately records 430 ms of ffmpeg crop/resampling preparation, including decode.

| SwiftShader input | Baseline pipeline | With expansion/refinement | Expansion alone | Upload / readback | Reads |
| --- | --- | --- | --- | --- | --- |
| Rendered translation, known K | 0.989 s | 2.558 s | 1.478 s | 4,932,864 / 36,864 B | 18 |
| Graham Hall, 12 existing photos | 3.405 s | 9.899 s | 4.783 s | 64,052,752 / 129,552 B | 66 |
| Uploaded Shorts window, assumed K + manual mask | 2.676 s | 5.459 s | 2.451 s | 18,412,848 / 52,848 B | 30 |

Graham Hall's TypeShade CPU expansion takes 16.85 s versus 4.78 s on SwiftShader
in these runs; this is software-backend evidence only. Device-managed peak active
storage is 8,724,992 B for the five-case run and 5,516,192 B for the video window;
all live storage returns to zero. Host/browser/driver/staging peak memory is not
measured. The dominant measured expansion cost is tracking (4.69 of 4.78 s for
Graham Hall); repeated uploads and 66 read calls remain concrete optimization work.

## Reproduction and checks

```sh
npm run build
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --backend=webgpu
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --backend=cpu
TYPESHADE_CHROMIUM=/usr/bin/chromium node --test tests/typeshade/observation-expansion-rgb-webgpu.test.mjs
```

The optional video path is local and is never published. To reproduce that window,
pass `--video=/path/to/input.mp4 --cases=user-video-window` (defaults: seek 8 s,
`--video-crop=1080:1440:0:480`, `--video-valid=0,32,360,230`). It requires ffmpeg.
The receipt records per-frame hashes, coordinate transforms, calibration and mask
provenance; no frame pixels or uploaded video are committed.

Evidence files: `docs/experiments/general-video/observation-expansion-{cpu,webgpu,user-video}.json`.
The RGB WebGPU regression runs the four small rendered cases and asserts camera,
depth, registration and deferred-initialization criteria, not just point counts.

A control on the **same** Graham Hall expanded candidates allows rebuilding from
scratch (`--refinement=rebuild`). It changes the seed [2,3] → [1,2], gives 2,422
landmarks / 5,768 observations, and raises median centre/radius error to 0.158755.
Carrying the accepted map keeps [2,3], gives 2,471 / 5,899 and 0.154657. This isolates
why refinement reuses the accepted map instead of blindly restarting SfM. The
control is recorded in `observation-expansion-rebuild-webgpu.json`.


Final local checks: 304 core tests; 63 affected TypeShade CPU tests; six KLT /
expansion / refinement WebGPU tests; one four-case RGB WebGPU regression; and four
baseline CPU MVS smoke tests pass. One optional MVS-defaults test is intentionally
skipped unless `TYPECV_MVS_DEFAULTS=1`. Formatting, core typecheck, GPU declaration
build, TypeShade runtime audit and the 89-page website build pass. The validation
receipt records commands, source/report hashes and execution limits in
`observation-expansion-validation.json`.

The latest combined heads of #222/#223 are not claimed as tested here. This stage
retains the existing sparse schema and tests baseline MVS compatibility. No merge,
deployment or lab-default activation is part of this implementation report.
