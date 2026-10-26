# Estimated surfaces and independent image corroboration

For the subsequent execution optimization with unchanged observations and geometry,
see [klt-performance.md](klt-performance.md). Timings below describe this earlier stage.

This follow-up extends PR224's bounded observation-expansion API. The comparison
head is `0ec468f924a52b360cc440c8e39f734736287624`. It preserves continuous feature
indices, original pixels, accepted-map gauge and the existing TypeShade LK kernels,
mapper verification, triangulation and BA. It adds no model/library dependency.

## Scope and tasks

- [x] V1: Audit translated-surface patch mismatch and missing independent image corroboration.
- [x] V2: Add conservative estimated-map surface patch proposals with degeneracy fallback.
- [x] V3: Corroborate candidates from another frozen original observation, including canonical joins.
- [x] V4: Compare controlled tracking and ordinary RGB using fixed map cohorts and alignment.
- [x] V5: Record measured limits, default decisions and final checks.

## Design

The previous affine proposal differentiates rotation and camera calibration. A
finite landmark guides its centre but does not supply the local surface's depth
gradient. Forward camera motion and a tilted surface can therefore leave a biased
image measurement even with a good centre initializer.

`surfacePatchWarp` proposes a local inverse-depth plane from the estimated sparse
map. In the source camera, a plane has the form `1/Z = a*x + b*y + c`, using
undistorted normalized rays. At most twelve nearby source-observed finite points
within 48 pixels fit a three-variable system using existing camera and linear
algebra functions. The query itself is held out. At least six samples, a
well-conditioned spatial distribution, support surrounding the entire source
patch and a maximum 2% relative inverse-depth residual are required. A finite
query point must agree with the predicted depth and source projection. Invalid,
behind-camera or unsupported proposals fall back to rotation/identity patches.

The plane and estimated relative pose predict a centre and a local patch
Jacobian, including translation. Original images must still supply the LK
measurement, forward/backward check, NCC and competing-peak test. Neither inferred
depth nor projected coordinates become observations or finite landmarks. Source
surface grids are bounded to four cached pairs. Full patch mask checks remain in
effect. Source-view holes, depth boundaries, curved surfaces, occlusion and sparse
support may prevent a proposal; an apparently planar wrong surface can still
mislead it, which is why subsequent image and mapper checks remain necessary.

`multiViewConsistency` freezes trusted track membership before any new coordinate
is appended. For each candidate, when possible it selects one other ORIGINAL
track member in a compatible continuous shot/coordinate system. Selection prefers a comparable estimated point-view angle, magnification and
camera orientation. Without finite geometry it selects the nearest target
playback index. These scores order the search; they do not establish visibility,
point identity or a measured observation.
The alternate original patch is independently tracked into the target, initialized
at the primary IMAGE measurement. It must pass its own tracking, mask, NCC and
ambiguity checks, then agree within `maxConsistencyErrorPx` (default 0.5 pixels).
Newly propagated descendants cannot corroborate themselves. Searches are batched
by anchor/target pair and use the same resident/stateless execution paths.

Disagreement and failed secondary image measurement reject the candidate. With
`multiViewConsistency: true`, when there is no eligible alternate original observation, a two-view candidate remains
available with `corroboration.status = 'unavailable'`. A successful witness records
its original feature, measured endpoint and quality. These are image consistency
states; even verified candidates still require mapper geometry checks. Repeated
identical patterns can fool both anchors. This is not proof of physical identity.

If deduplication selects an existing target keypoint, the FINAL stored coordinate
must also agree with both measurements within the same tolerance. Otherwise the
edge is rejected, and the existing coordinate is preserved.

`multiViewConsistency: 'required'` instead defers a candidate unless an alternate
original patch independently measures an agreeing endpoint. Seeds without another
trusted member outside their source and target views are deferred before the
bounded tracking/spacing budget is spent. A later call may retry them with more
verified input evidence. This changes neither the default `false` nor the legacy
`true` policy. "Original" means trusted at this invocation's start, not a guarantee
of detector origin or independence across earlier invocations. The frozen membership
prevents coordinates propagated in this call from acting as their own witnesses.

`newObservations` continues to list newly appended coordinates only. `newEdges`
records every newly added pair edge, including joins to existing target features.
Each row includes `createdFeature`, the KLT `measuredPixel`, and `targetPixel`, the
preserved or appended feature coordinate that geometric verification will use.
These coordinates may differ for a join; neither the original feature nor its ID
is overwritten. The row retains the witness endpoint and quality and remains
`geometry: 'candidate'`. No descriptor, visibility or accepted 3D geometry is
fabricated. All supplied raw pair candidates remain in `candidatePairs`.

In required mode, `rejected.multiViewUnavailable` and `corroboration.unavailable`
also count seeds deferred by the early membership check, before a KLT attempt.
They can therefore exceed `diagnostics.attempts`; this prebudget deferral does
not report a failed image-tracking measurement.
The existing witness disagreement/failure counters and post-measurement ownership,
join and capacity guards retain their meanings. More corroborated edges do not
establish better depth, pose accuracy or Dense coverage; those require separate
controlled reconstruction evaluation. The mode remains opt-in.

### Appending measurements without solving the map

`appendExpandedObservations` is a separate, synchronous host helper for callers
that want to retain the supplied finite geometry. It does not invoke the full
refiner, BA, triangulation, track merging, registration or filtering:

```ts
import { expandObservationsGpu, appendExpandedObservations } from 'typecv/gpu';

const expansion = await expandObservationsGpu(device, frames, rawPairs, {
  initialModel: model,
  verifiedPairs,
  multiViewConsistency: 'required',
});
const appended = appendExpandedObservations(model, frames, expansion, {
  maxErrorPx: 3, // In these frames' pixels; scale explicitly when resizing inputs.
});
const nextModel = appended.reconstruction;
```

Retain the immutable `frames`, model and expansion result from the same call.
Every source and distinct witness must already belong to the same finite point
in that original model. An original feature that obtained ownership through a
new join cannot establish that authority for another append. The target must be
registered, retain its actual `targetPixel`, project in front of the fixed camera
and pass the caller's reprojection gate. Both original source and witness must
also project in front of their fixed cameras and pass that same gate against
their original observation pixels; ownership alone does not establish this.
An existing-feature join keeps that
feature's original pixel, even when the KLT endpoint differs.

The helper validates original image identities, cameras, complete feature
prefixes and original observation coordinates. Malformed lineage throws; valid
but unqualified measurements are deferred with a rejection reason. Competing
eligible claims for a target feature or a point's target view are all deferred,
and exact duplicate claims are counted once. `admitted` records each accepted
edge's index in `expansion.newEdges`, track ID and reprojection error.

All original views, poses, XYZ, track IDs and observations are copied unchanged.
The helper does not alter feature coordinates, descriptors or raw pair matches,
and does not verify `candidatePairs` for another mapper. Its image evidence and
fixed-model reprojection are conditional on the supplied geometry; repeated
patterns and a self-consistent wrong map can still pass. New observations can
change Dense seeds, depth ranges, source selection and fusion, so geometry
preservation alone does not establish a Dense quality improvement. This path is
explicitly opt-in and leaves existing mapper and source-selection defaults intact.

## Validation plan

Controlled tests isolate translated tilted-plane tracking with a deliberately
inexact supplied map; those are API/tracking experiments, not RGB-only SfM claims.
Separate three-frame tests cover consistent anchors, corrupted patches, displaced
original features, canonical-coordinate joins and two-view unavailable evidence.
Core tests cover surface degeneracy, distortion, coordinate conventions, gauge
invariance, frozen ownership, known cuts and masks. CPU/WebGPU run the same inputs.

RGB comparisons reconstruct their initial map from images and declared camera
calibration. Renderer poses/depth stay outside reconstruction. Existing fixed
cohort and whole-map alignment metrics distinguish retained landmarks, new tracks,
losses and merges. Pure rotation/low parallax must retain deferred finite depth.
Graham Hall and the existing manually masked Shorts window provide ordinary
photo/video comparisons, with no geometry truth claimed for the uploaded clip.

The new options remain opt-in while comparisons run. `elapsedMs` and the whole
pipeline timer include all control/geometry work. `geometryPlanningMs` covers
surface construction/fit and initializers; `trackingMs` includes primary and
witness LK/pyramid execution; `hostValidationMs` covers post-tracking patch and
ownership checks. These are selected stages, not a complete partition of CPU
time. Transfer counters include witness searches. `surfaceProposals` includes
later-rejected and witness proposals; corroboration counters precede ownership,
capacity and canonical-join admission. Driver memory, retained host RGB and host
surface-grid peaks are not GPU working-storage measurements.

## Calling the bounded pass

```ts
const expanded = await expandObservationsGpu(device, frames, rawPairs, {
  initialModel: initial.reconstruction,
  verifiedPairs: initial.verifiedPairs,
  rotations: initial.tracking?.rotations,
  surfacePatchWarp: true,
  multiViewConsistency: true,
  maxConsistencyErrorPx: 0.5,
});
const refined = await refineIncrementalGpu(
  device,
  expanded.images,
  expanded.candidatePairs,
  initial,
  mapperOptions,
);
```

Frames must retain the original grayscale and original feature-index prefix for
every registered view in the accepted connected map. Use the accepted calibration
in the same pixel coordinate system; automatic focal/crop changes are not a reason
to reuse stale K. Caller-supplied masks and known shot boundaries retain their
existing meaning. Surface patch warping implies affine tracking without requiring
`geometryPatchWarp:true`; the previous rotation/identity fallbacks still apply.

This is not connected to Lab's streaming/default path. The current worker releases
most pixel/feature evidence and its session keeps a bounded window. Whole-map
postprocessing cannot reconstruct that missing evidence from the final 3D points.
A future Lab integration must explicitly bound retained RGB/features/pairs, record
skip/cancel/failure states and run before releasing evidence and starting MVS.
No full-clip observation archive, automatic mask/cut estimation, object-local
geometry, nonrigid tracking or combined latest PR222/223 validation is added here.

## Measured results

The implementation is `ad19b563edfdf10a2dfcf4a6f1fefc9dab82d79d`. Eight reports
under `docs/experiments/general-video/observation-surface-*` and
`observation-consensus-only.json` record the controlled experiment, four RGB
ablations, two video runs and a cache comparison. Reports include source hashes.
The comparison reproduces the preceding `0ec468f` rotation/K affine results.
Every run used Node 24.19.0, Chromium 154.0.8037.57 and the pinned TypeShade
`d36ac5027356b96e87a9ad900dfd8e6ea4ab55dc`. WebGPU identifies `google`,
`swiftshader`, `fallback=true`. Physical hardware was not measured.

In the controlled tilted-plane tracking experiment, a slightly inexact supplied
map initializes both methods. On the **same 27 retained seeds**, rotation/K-only
mean coordinate error is 0.470929px; local-surface patches give 0.017964px, with
maximum 0.036327px. CPU and software WebGPU agree. This is a 96.2% mean tracking
error reduction under the declared experiment, not an RGB-only SfM accuracy gain.
All target coordinates move away from the deliberately perturbed projection.

In the following RGB experiments, each initial sparse map is estimated from RGB
features/matches and declared K. Renderer truth never enters the browser pipeline.
All four modes use the same original images and cached/fresh front end, mapper
settings and accepted seed. `reference` already includes rotation/K patch warps;
the surface and corroboration options are varied independently. Resident pyramid
reuse is off in this ablation.

| RGB input                       | Reference points / observations | Surface only | Corroboration only | Both        |
| ------------------------------- | ------------------------------- | ------------ | ------------------ | ----------- |
| render-translation              | 250 / 597                       | 251 / 599    | 250 / 588          | 251 / 590   |
| render-translation-omission     | 169 / 397                       | 169 / 397    | 168 / 394          | 168 / 394   |
| render-rotation                 | 0 / 0                           | 0 / 0        | 0 / 0              | 0 / 0       |
| render-low-parallax             | 0 / 0                           | 0 / 0        | 0 / 0              | 0 / 0       |
| graham-hall-cached-rgb-features | 2471 / 5893                     | 2468 / 5904  | 2457 / 5693        | 2457 / 5694 |

Both translation cases register four frames; Graham Hall registers all twelve.
Pure rotation and low parallax keep their 2D/pending tracks and deferred finite
initialization. Zero finite landmarks in these cases are a required result.

| Fixed evaluation population                                        | Reference affine expansion | Both options |
| ------------------------------------------------------------------ | -------------------------- | ------------ |
| Translation, same 141 retained points: median relative depth error | 1.72510%                   | 1.65285%     |
| Translation, same points: p90 relative depth error                 | 4.98491%                   | 5.09148%     |
| Translation, same 4 cameras: centre RMSE                           | 0.00636953                 | 0.00610710   |
| Omission, same 21 retained points: median relative depth error     | 1.28846%                   | 1.27850%     |
| Omission, same 2 initially registered cameras: centre RMSE         | 0.00179058                 | 0.00177153   |
| Graham Hall, same 12 cameras, frozen baseline Sim(3): centre RMSE  | 0.57503950                 | 0.57597733   |

Translation improves retained median depth by 4.19% and centre RMSE by 4.12% relative
to prior affine expansion. Its depth p90 regresses, and the unexpanded seed map's
median 1.56371% remains lower. Surface-only depth median 1.85276% also regresses;
the controlled patch result does not establish general surface-map accuracy.
The omission case improves only slightly over prior expansion. Camera/depth units
and a single seed gauge follow the existing evaluator; no per-frame alignment is
used. New landmark cohorts are reported separately (109 versus 110 new translation
tracks), so they do not silently alter the retained accuracy population.

Graham Hall accuracy does not improve. The baseline map has 2423 landmarks; the
reference expansion retains 2420 and loses 3, while both options retain 2419 and lose 4.
These ownership losses, new cohorts and zero ownership merges are reported, not
hidden by a new alignment or nearest-XYZ correspondence. They do not measure the
true physical false-merge rate in a scene with no landmark identity truth.

The same six-frame Shorts window uses a manual background rectangle, explicit
crop/resize and assumed focal length. Reference expansion gives 470 landmarks /
1252 observations, versus 468 /1176 with both options. **All 202 baseline tracks
remain** in both runs; new-without-baseline tracks change 268→266 and newly measured
coordinates 531→455. Six frames remain registered. No reference geometry exists,
so rejecting more measurements is not evidence of improved 3D accuracy. Surface
support is insufficient here (zero proposals); the measured change is corroboration.

## Cost and defaults

These are single cold process runs on software WebGPU, not stable speedup claims.
Tracking includes pyramid preparation, execution, synchronization and the consumed
readbacks. Whole browser pipeline time includes decode/grayscale, fresh front end
where applicable, the initial map, expansion and resumed refinement. Browser
startup, transport and fixture generation are excluded. Graham Hall's cached
RGB-derived features/matches exclude original detector/matcher cost. For the clip,
ffmpeg decode/crop preparation is separately 0.449s/0.460s; browser decode is 0.065s/
0.063s, feature extraction 1.391s/1.440s and matching 0.261s/0.257s.

| Mode / input                        | Expansion | Browser pipeline | Application upload bytes | Full Resident snapshot bytes |
| ----------------------------------- | --------- | ---------------- | ------------------------ | ---------------------------- |
| Reference / Graham Hall             | 4.884s    | 10.023s          | 64,139,120               | 129,552                      |
| Both / Graham Hall                  | 22.021s   | 26.993s          | 255,936,000              | 145,920                      |
| Both + resident cache / Graham Hall | 21.491s   | 26.687s          | 122,278,400              | 145,920                      |
| Reference / clip window             | 2.379s    | 4.807s           | 18,448,080               | 52,848                       |
| Both / clip window                  | 8.402s    | 10.870s          | 60,689,920               | 61,152                       |
| Both + resident cache / clip window | 8.132s    | 10.508s          | 28,559,920               | 61,152                       |

The cache comparison has exactly identical evaluated cohorts, point/observation
counts and new-coordinate counts to its stateless counterpart. It reduces logical
uploads by 52.2%/52.9%, but retains a third pyramid and does not offset the added
corroboration work. Graham Hall builds 84 pyramids instead of 176; the clip builds 31
instead of 66. Cache entries are disposed automatically (frames/residentBytes0).
Peak active device storage is 8,729,088B in the stateless RGB suite and 13,087,488B in
the cached comparison. Every report ends at liveBytes0, pendingReads0. These are
allocator/application counters, not driver memory, physical bus bytes or total
host memory. The measured extra cost is predominantly in tracking, not surface
fitting (Graham Hall planning 0.118s versus tracking 21.777s with both options).

**`surfacePatchWarp` and `multiViewConsistency` default to false.** Existing
`geometryPatchWarp` and `pyramidCache` also remain opt-in. Declared conditions
improve controlled tracking and some synthetic geometry metrics, but ordinary
photos do not show an accuracy gain and multiview verification adds substantial
latency. No Lab/streaming default is changed on this evidence.

Reproduction:

```sh
npm run build
node --test tests/typeshade/observation-surface.test.mjs
TYPESHADE_CHROMIUM=/usr/bin/chromium node --test tests/typeshade/observation-surface-webgpu.test.mjs
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --patch-warp=1 --surface-warp=0 --multi-view=0 --pyramid-cache=0 --output=/path/reference.json
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --patch-warp=1 --surface-warp=1 --multi-view=1 --pyramid-cache=0 --output=/path/combined.json
```

Use each flag separately for the two intermediate ablations, and `--pyramid-cache=1`
for the resident comparison. The existing `--video=/local/clip.mp4
--cases=user-video-window` flags reproduce the explicitly masked/cropped window;
uploaded image/video content is not committed. The validation receipt records
commands, source/report hashes, scenario conditions and local check counts.

Final local checks pass: 329 core tests, 29 affected TypeShade CPU tests and ten
WebGPU tests (including the four-case RGB regression), formatting, typecheck,
build/declarations, TypeShade runtime audit and a 95-page website build. The
validation receipt records the source/report/log hashes. Remote CI is separate;
these results do not claim that every repository integration or physical GPU was tested.
