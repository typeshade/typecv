# Estimated surfaces and independent image corroboration

This follow-up extends PR224's bounded observation-expansion API. The comparison
head is `0ec468f924a52b360cc440c8e39f734736287624`. It preserves continuous feature
indices, original pixels, accepted-map gauge and the existing TypeShade LK kernels,
mapper verification, triangulation and BA. It adds no model/library dependency.

## Scope and tasks

- [x] V1: Audit translated-surface patch mismatch and missing independent image corroboration.
- [x] V2: Add conservative estimated-map surface patch proposals with degeneracy fallback.
- [x] V3: Corroborate candidates from another frozen original observation, including canonical joins.
- [ ] V4: Compare controlled tracking and ordinary RGB using fixed map cohorts and alignment.
- [ ] V5: Record measured limits, default decisions and final checks.

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

Disagreement and failed secondary image measurement reject the candidate. When
there is no eligible alternate original observation, a two-view candidate remains
available with `corroboration.status = 'unavailable'`. A successful witness records
its original feature, measured endpoint and quality. These are image consistency
states; even verified candidates still require mapper geometry checks. Repeated
identical patterns can fool both anchors. This is not proof of physical identity.

If deduplication selects an existing target keypoint, the FINAL stored coordinate
must also agree with both measurements within the same tolerance. Otherwise the
edge is rejected, and the existing coordinate is preserved.

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
