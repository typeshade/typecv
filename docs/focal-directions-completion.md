# Directional focal and persistent-depth completion

This follow-up integrates PR #223 at `cc4a5d5` with main `dd086b44` (PR #224).
The PR already contains the merged #225 persistent inverse-depth tracker and
pipeline work. Those implementations and their historical measurements are
preserved; they are not newly invented or remeasured by this follow-up.

- [x] U1: integrate current main while preserving directional BA, persistent depth,
  the measured-observation refinement API and both sets of tests (`4c91823`).
- [x] U2: preserve admission, stable identity and ownership across callbacks,
  descriptor adapters, sessions and measured-observation refinement.
- [x] U3: compare current main and this revision with supplied and estimated
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

Without a depth tracker, directional focal handling stays experimental and
defaults to false. Supplying `depthTracking` implicitly enables separation unless
the caller explicitly sets `separateDirectionalObservations: false`; this existing
conditional policy is preserved. Persistent depth requires an explicitly supplied
tracker; it is not enabled in the Lab UI.
The known high-noise directional classification failures and HUD/low-texture
limitations still require their own evidence. This completion stage does not
claim automatic sky/reflection detection, unknown zoom, object deformation,
cut/map merging, whole-Shorts reconstruction, dynamic Radiance integration or
physical-GPU performance.

Historical design and measurements remain in [directional focal BA](focal-directions.md),
[persistent inverse depth](persistent-inverse-depth.md) and
[pipeline performance](pipeline-performance.md). The final current-main comparison
and validation receipt are recorded after execution completes.

## Current-main comparison results

[The seven-case report](experiments/general-video/focal-directions-completion-samples.json)
pins baseline `dd086b44` and measured source `9a0f1da`. All 28 mode runs finish
and pass input immutability checks. Default-OFF geometry **and tracking** hashes
match main for all seven inputs. Opt-in finite geometry matches in six inputs;
tracking hashes match in five. The additional tracking output is deliberate
preservation of pending evidence, not an accuracy certificate.

| Input | Opt-in main → candidate finite points | Pending tracks in candidate |
| --- | --- | --- |
| South building, supplied K | 5,785 → 5,785 | Same tracking as main |
| Gerrard Hall, supplied K | 7,775 → 7,775 | Same tracking as main |
| Person Hall, supplied K | 12,217 → 12,217 | Same tracking as main |
| Graham Hall COLMAP matches, supplied K | 5,935 → 5,935 | Same tracking as main |
| Graham Hall Lab matches, supplied K | 2,423 → 2,423 | Same tracking as main |
| Pyramid eight frames, estimated focal | 689 → 685 | 32, previously not published |
| Graham Hall Lab matches, estimated focal | 2,405 → 2,405 | 386, previously not published |

Every row retains all 8 or 12 registered cameras. Pyramid's focal changes from
501.568 to 500.428 px and reprojection mean from 0.3611 to 0.3629 px; there is no
ground-truth calibration/geometry to call that an accuracy improvement. Its
search reports a separate bearing refinement with 193 directional observations,
even though the selected final bundle has none. Graham's estimated-focal finite
map is identical between enabled versions (433.529 px focal); the new output
retains 386 unresolved tracks without inventing their depth. Those pending
tracks are not all classified as sky or exact infinity.

The new Graham row also makes within-version effects explicit: in either
version, enabling the existing separation changes 2,422 default finite points
to 2,405 under its stated pinhole assumptions. That pre-existing flag effect
is distinct from the new cross-version tracking retention. No ordinary-sample
accuracy gain or general wall-time speedup is established. Keep the directional
option false by default and persistent depth explicitly requested by the caller.

Reproduction (baseline checkout must be pinned to `dd086b44`):

```sh
node scripts/benchmark-focal-directions-samples.mjs \
  --baseline=/path/to/main-baseline \
  --assert-default-parity=1 \
  --output=docs/experiments/general-video/focal-directions-completion-samples.json
```

## CI browser completion condition

The prior branch's CI1051 passed both Node checks and WebGPU, then timed out
waiting for a renderer-owned camera-pulse dataset. The viewer intentionally
pauses rendering outside the viewport; the browser test had resized the page
and operated drawers without ensuring the viewer was visible. The test now
scrolls that viewer into view before the exact pulse assertion and includes
visibility/dataset diagnostics on failure. Its assertions and timeout remain
unchanged; production offscreen rendering policy is preserved. Final targeted
browser validation and exact-head remote CI are recorded separately below.

## Final local validation

The [validation receipt](experiments/general-video/focal-directions-completion-validation.json)
records source/input/report/log hashes and exact scope. All 363 core tests,
28 affected TypeShade CPU tests (including the four publication regressions),
four new and eight additional SwiftShader WebGPU tests pass. Callback mutation,
failed refinement rollback, borrowed descriptors, live-array cleanup, persistent
IDs and archived revocation are exercised. Type/format/library-GPU build and
fresh TypeShade runtime audit pass; the website builds 109 pages. The formerly
failing dense-activity browser target passes (one test, zero failures/skips).

GPU evidence is software WebGPU: runtime audit identifies Google SwiftShader.
The comparison is CPU f64 over cached feature inputs. There is no physical-GPU,
full RGB-front-end or general accuracy/speed claim. Source is frozen after
`9a0f1da`; the final visibility-test/documentation/evidence commit changes no
reconstruction math. Exact-head remote CI is a separate final merge gate.

The U4 checkbox above records this pre-merge documentation checkpoint. The PR's
live checklist records the subsequent delivered-head CI and merge outcome.
