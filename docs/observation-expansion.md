# Image-based expansion of sparse observations

Status: implementation in progress. Baseline: `e4e5046`.

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
- [ ] E4: compare original/new observation counts, track lengths, distinct landmarks,
  coverage and accuracy on RGB-derived input; test occlusion, textureless regions,
  wrong predictions, rotation and insufficient parallax on CPU and WebGPU.
- [ ] E5: finish documentation, execution/memory evidence and task/PR updates.

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
