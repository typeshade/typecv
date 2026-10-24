# Directional focal and persistent-depth completion

This follow-up integrates PR #223 at `cc4a5d5` with main `dd086b44` (PR #224).
The PR already contains the merged #225 persistent inverse-depth tracker and
pipeline work. Those implementations and their historical measurements are
preserved; they are not newly invented or remeasured by this follow-up.

- [x] U1: integrate current main while preserving directional BA, persistent depth,
  the measured-observation refinement API and both sets of tests (`4c91823`).
- [x] U2: preserve admission, stable identity and ownership across callbacks,
  descriptor adapters, sessions and measured-observation refinement.
- [ ] U3: compare current main and this revision with supplied and estimated
  calibration, enforce default parity and record the measured limits.
- [ ] U4: complete final local/remote checks on the delivered commit and merge.

## Reproduced publication failures

The tracker deliberately admits no XYZ when its required relative uncertainty
cannot be met. Native triangulation/PnP seeds may still be useful inside the
mapper. They must not be published as admitted persistent-depth results.
Four new regressions fail before the correction on the TypeShade CPU tier:

| Path | Before | Required result |
| --- | --- | --- |
| Direct mapper, 25 native points and zero admitted points | Callback point counts `[25,25,25,25,0]`; four callbacks even when final refinement fails | One callback containing the completed zero-point map; none on failure |
| Accepted-map refinement | `[25,25,0]`; two callbacks on failure | One admitted callback; none on failure |
| Float-descriptor adapter | No final callback | One admitted callback; borrowed descriptors stay owned by the caller |
| Sliding session, track 0 pending | A second archive renumbers admitted points and callback/result IDs disagree with depth snapshots | Persistent IDs, observation ownership and admission agree through updates and finish |

The direct CPU and scene protocols also previously passed the returned map by
reference to their admitted callback. A callback could mutate its pose/XYZ and
corrupt the returned geometry. A separate mutation regression reproduces this.
The new tests mutate callback poses/cameras/points and verify that returned maps
and tracker snapshots remain independent.

## Retained implementation and state boundaries

The common GPU driver suppresses native callbacks only when depth tracking is
requested. It forwards one snapshot after the tracker finishes. The float
adapter preserves the caller callback so the common driver can publish it.
The CPU reconstruction and accepted-map refinement use the same driver boundary,
including deferred initialization and completed depth publication. CPU, scene
and GPU final callbacks receive independently owned geometry snapshots.

A depth-tracked session uses the tracker's existing full-history reconstruction
as its archive. Sending that result through the native window archive again
would create a second identity/admission authority. Native sessions keep their
existing archive behavior. The tracker still owns late exclusions, demotion,
reanchoring and archived-view uncertainty; no new persistence layer is added.

The session regression uses ten views and an eight-view window. It keeps track 0
pending while 24 others are finite, exercises archived cameras, revokes one
previously finite track including its archived observations, and finishes with
23 finite tracks and 232 unique observations. Callback, returned landmark and
snapshot IDs/observation keys must agree. Failures publish no geometry and leave
the previous tracker snapshot intact. The real numerical mapper/matcher runs;
only the tested refinement failure is injected.

The CPU composition fixture appends measured coordinates and correspondences
to already registered views, extending 30 finite tracks to 60 and unique
observations from 110 to 240. Replaying those observations keeps their IDs and
does not artificially shrink covariance. The initial accepted input remains
unchanged. With joint depth optimization, the first camera and seed baseline
length fix the world gauge; the second camera's orientation may still improve.
This is a supplied-correspondence geometry regression, not an RGB detector test.

The existing uncertainty model, geometric thresholds, PnP requirements, pixel
conventions and numerical shader arithmetic are preserved. TypeShade still
linearizes inverse-depth measurements on the selected backend; the disclosed
host f64 fitting, Schur solve and covariance remain host work. Device failures
propagate without a CPU retry. No new upload/readback stage is added.

## Comparison contract

The updated sample comparison runs main and the candidate with directional
handling OFF and ON on the same seven cached-feature inputs. Five rows supply
COLMAP K/distortion; Pyramid and the additional Graham row estimate shared focal
from a dimensions-only starting camera. The latter assumes square pixels,
centered principal point and zero distortion; reference calibration is not sent
to reconstruction. Cached guided matches can carry prior geometric selection,
so this does not establish fully blind RGB calibration.

Candidate/refinement bearing support, final BA support, registration/deferral,
input/source hashes, finite geometry and pending tracks are reported separately.
The default-OFF comparison is an explicit failure gate. COLMAP pose agreement is
agreement with an estimated reference, using one whole-map alignment, not ground
truth accuracy. Reconstruction-only wall times are diagnostic, with no speedup
claim. Historical #225 and #224 performance ratios cannot be multiplied by these
measurements.

## Defaults and limits

Directional focal handling stays experimental and defaults to false. Persistent
depth requires an explicitly supplied tracker; it is not enabled in the Lab UI.
The known high-noise directional classification failures and HUD/low-texture
limitations still require their own evidence. This completion stage does not
claim automatic sky/reflection detection, unknown zoom, object deformation,
cut/map merging, whole-Shorts reconstruction, dynamic Radiance integration or
physical-GPU performance.

Historical design and measurements remain in [directional focal BA](focal-directions.md),
[persistent inverse depth](persistent-inverse-depth.md) and
[pipeline performance](pipeline-performance.md). The final current-main comparison
and validation receipt are recorded after execution completes.
