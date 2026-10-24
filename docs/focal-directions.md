# Directional support during automatic focal calibration

Status: implemented, experimental and opt-in. Baseline: `e4e5046` (PR #221).

The opt-in `separateDirectionalObservations` mode can keep rotation-compatible
observations out of finite two-view structure. The focal search and subsequent
bundle adjustment still use finite landmarks only. At a plausible calibration,
unresolved directions therefore count as dropped observations; at a wrong
calibration, fabricated finite points can instead increase the candidate's
support. Joint focal refinement cannot use the missing directional residuals.

## Reproduction and scope

The committed [direction-depth experiment](experiments/general-video/direction-depth-cpu.json)
contains `estimated-focal-noise0.2`: 60 finite tracks and 60 direction-only tracks,
five views, 0.2 px synthetic noise, supplied correspondences, and an input focal
guess of 512 px rather than the evaluator's 480 px. With the experimental mode,
all five cameras register but all 60 direction-only tracks become finite points;
the maximum focal error is 28.2%. Ground-truth poses, depths and track labels are
only evaluator inputs. This is a geometry experiment, not an RGB benchmark.

A fresh baseline grid probe confirms that 0.694 and 0.798 times image width keep
60 finite points, while 0.918 keeps 120. Simply adding a post-hoc direction error
to the score does not fix this: the poses were optimized without those factors,
so the nearest grid candidate has worse directional residuals than the wrong
finite reconstruction. Direction factors must participate in the optimization;
changing the score alone is insufficient.

This stage will reuse the existing CPU and TypeShade bundle solvers. Finite
structure and unresolved directional observations must remain separate at API
and export boundaries. Direction factors may constrain orientation/calibration;
they must have zero translation derivative and cannot establish a camera center,
depth, metric scale or exact infinity. Observation compatibility remains
conditional on calibration, poses, noise and the static-scene assumption.

## Tasks and completion gates

- [x] T1: inspect the merged focal/geometry paths and existing focal PRs; reproduce
  the mixed-direction calibration failure and record the bounded design.
- [x] T2: add a separate unresolved-direction factor path to existing bundle
  optimization, preserving finite XYZ behavior and translation observability.
  Validate the CPU reference and the TypeShade execution path.
- [x] T3: carry compatible direction observations into focal candidate scoring
  and refinement without rewarding invented finite observations. Preserve
  pending depth and reject unsupported calibration/structure.
- [x] T4: run regression and paired baseline comparisons, including ordinary
  samples, pure rotation, far structure, finite-only structure and noisy mixed
  observations. Record CPU/GPU backend identity, transfer costs and limitations.
  Decide defaults from measured regressions as well as improvements.
- [x] T5: document the implemented conditions and remaining limits, finish
  validation, and update the PR's task checklist with the actual results.

Tasks are checked only after their implementation and stated validation finish.
Each completed task is committed and reflected in the PR. This stage does not
add semantic sky detection, variable zoom calibration, dynamic-object geometry,
shot/map merging, dense MVS or dynamic Radiance rendering.

## T2 result

`BundleAdjustmentOptions.directionalConstraints` accepts separate `BundleDirection`
records; the result returns optimized bearings separately from finite landmarks.
The existing 3-column point blocks and Schur solvers are reused. Bearing vectors
are normalized in their own radial gauge; finite-map similarity normalization
does not move them. The projection is `R d`, and all translation Jacobian entries
are exactly zero. The existing finite-only connectivity and baseline checks run
before directions can contribute to a solve.

The three CPU reference tests, two TypeShade CPU tests (resident and a 100,000-byte
working budget), and the WebGPU test pass. Both fixed and shared-focal solvers
recover the explicit noiseless factor fixture within 0.01 degrees/0.01 fixture
units; finite gauge, ownership, pending representation and input snapshots are
checked. Existing finite BA tests (7) and shared-focal CPU-tier tests (4) pass;
typecheck and library/GPU build pass. Broader regressions remain T4 work.

WebGPU here is Google SwiftShader, fallback=true, not a physical GPU. Fixed and
shared-focal factor runs upload 251,024/275,360 application bytes and read back
3,624/3,656 bytes. Both release all live buffers. These are solver-only tests
with supplied direction factors; they do not establish automatic calibration,
RGB reconstruction quality, peak memory or hardware acceleration.

## T3 result

The object/scene CPU mappers and the GPU mapper now fit unresolved tracks to
world bearings for the final global BA, after finite-only mapping and final
retriangulation. Automatic selection requires at least three already registered
observations and every selected residual within the existing pixel error bound.
It also reuses the rotation estimator's default consensus/spread gates: every
participating view must retain at least 12 bearing tracks with at least 0.5 degrees
of minor-axis calibrated spread. Selection is pruned until those conditions hold.
Two-view tracks stay pending without an automatic BA constraint. These are
conditional bearing hypotheses, not a semantic or exact-infinity classification.
Earlier local/streaming updates do not impose this prior: an initial far-finite
approximation must not absorb the parallax that would establish its later depth.

The focal search includes these residuals and their observations. When its
best-covered subset supports a bearing model, it compares the normalized BIC
`log(SSE/N) + k log(N)/N`, with two scalar residuals per pixel observation,
3 coordinates per finite point and 2 per bearing, plus free pose/focal parameters.
Observations a candidate drops still incur the previous Huber-threshold penalty.
This Gaussian-noise model selection is a heuristic, not an identifiability
certificate. If no bearing model supports that view count, the established
finite-only score is retained.

The search jointly refines the best finite and bearing basins (at most two),
keeps the existing focal brackets, and compares the resulting objective. It does
not interpolate across a discrete change of point model. `focalSearch.scoring`,
candidate parameter/support counts and optional `refinements` explain that
decision. Unknown-calibration results also retain pending 2D tracks. Focal
search still belongs to the object/GPU pipeline; this does not add automatic
focal calibration to the separate experimental scene protocol.

The CPU geometry suite (308 tests), mixed-depth/far-reobservation regressions,
and five focused TypeShade tests (CPU and SwiftShader WebGPU, including automatic
focal calibration) pass. The noiseless and 0.2px mixed-direction focal cases
retain 60 finite points and 60 pending tracks instead of 120 finite points.
The 1px case still fails to identify the directional model and retains the old
wrong finite reconstruction. Broader comparisons and defaults remain T4 work.

The bearing seed/selection is f64 CPU work over keypoints/poses already on the
host; GPU execution reports it explicitly. BA residuals and solves stay on
TypeShade, reusing existing point-block buffers. Directions enlarge those
buffers and final result readback, but add no separate readback stage. PnP still
requires finite points: bearings alone never register a world camera center.

## Relationship to PR #222

[PR #222](https://github.com/typeshade/typecv/pull/222) evaluates finite-depth
support and image boundaries inside MVS with registered cameras held fixed.
This stage estimates cameras/calibration and sparse structure before MVS. Both
use rotation-only projection without a huge finite surrogate. The existing
`SparseReconstruction`/MVS camera and finite-landmark format is preserved;
pending bearings stay out of dense depth seeds and normalization. Each option
keeps its own default and evidence gate. Compatibility tests are T4 work;
neither PR's separate results establish an additive end-to-end quality gain.

## T4 measurements and defaults

The [26-case CPU report](experiments/general-video/focal-directions-cpu.json)
compares `e4e5046` with the experimental flag **on in both versions**. Each
version has one warmup and two alternating measured runs. Recorded production
source hashes match `a472446`; original dirty-checkout metadata is retained.
Inputs are synthetic keypoints and supplied identities, not an RGB extraction
benchmark. Pose/point truth only enters evaluation, with one fixed seed gauge per
whole connected map. Concurrent checks make wall times diagnostic, not speedup
evidence.

| Input | Baseline → this stage |
| --- | --- |
| Unknown focal, no noise, 60 finite + 60 directional tracks | Wrong finite directions 60 → 0; focal error 28.85% → 0.00012%; 60 tracks remain pending |
| Same input, 0.2px noise | Wrong finite directions 60 → 0; focal error 28.20% → 0.67%; maximum rotation error 1.600° → 0.0286°; point RMSE 1.343 → 0.0542 fixture units |
| Known K, 60 directions, 0.2px noise | Maximum rotation error 0.0418° → 0.0145°; point RMSE 0.0245 → 0.0152 |
| Known K, 180 directions, 0.2px noise | Maximum rotation error 0.0550° → 0.00444°; point RMSE 0.0410 → 0.0158 |
| Unknown focal, 1px noise | Still 60 wrong finite directions and 61.0% focal error; no improvement |
| Pure directions, pure rotation, rotation + known zoom, extremely far structure | Still defer finite initialization; no invented map |
| Screen-fixed HUD without a mask | Still produces wrong finite structure; requires separate observation selection |

The [ordinary-sample comparison](experiments/general-video/focal-directions-samples.json)
uses the same cached RGB-derived features/matches for four modes: baseline/current,
each with the option off/on. It excludes decoding, extraction, dense reconstruction
and GPU work. COLMAP reference poses are estimates, not ground truth; evaluation
uses one whole-map similarity alignment.

| Sample | Opt-in baseline → this stage |
| --- | --- |
| South-building | 12 cameras / 5,785 points; geometry unchanged |
| Gerrard-hall | 12 / 7,775; geometry unchanged |
| Person-hall | 12 / 12,217; geometry unchanged |
| Graham-hall, COLMAP matches | 12 / 5,935; geometry unchanged |
| Graham-hall, cached Lab matches | 12 / 2,423; geometry unchanged |
| Pyramid, cached 8 frames | 8 / 689 → 8 / 685; focal 501.568 → 500.428px; no ground-truth focal/poses to establish an accuracy improvement |

All six **default-off** geometry hashes are identical to the baseline. The
Pyramid row is not a replay of the current 47-frame catalogue video. Its mean
reprojection error changes from 0.3611 to 0.3629px; relief remains positive. Neither
the changed point count nor appearance proves better geometry.

**Keep `separateDirectionalObservations` default false.** The mixed-direction
gains do not remove the high-noise failures, the existing noisy-prefix pair
refinement regression, or the lack of demonstrated ordinary-sample gains. The
automatic support gate prevents isolated/poorly distributed pending tracks from
becoming a global prior. It is not a general static/dynamic or finite/infinite
classifier.

The final [SwiftShader run](experiments/general-video/focal-directions-swiftshader.json)
reproduces the 0.2px unknown-focal result: five cameras, 60 finite points, 60 pending
tracks, maximum rotation error 0.0286° and point RMSE 0.0542. Chromium uses Google
SwiftShader, fallback=true. Total browser/compile/run time is 63.19s, including
45,413,256 application upload bytes, 633,644 readback bytes, 1,186 reads and 652
read waves. All live buffers/bytes are released. Decode/extraction and peak GPU
memory are unmeasured; there is no physical GPU or acceleration claim. The
existing focal pair-base verification cache remains f64 CPU work when applicable;
the added bearing seed/selection is explicitly reported as CPU work too.

Validation passes: 309 core tests, 45 affected TypeShade CPU tests, four focused
software WebGPU tests and a final automatic-focal rerun. Typecheck, formatting,
library/GPU build and the 89-page site build pass. Admission tests caught an
unnecessary array slice before derived integer-bound checks; retaining the
original finite iterable preserves those checks and avoids that extra copy.

## Local integration with #222

The [integration record](experiments/general-video/focal-directions-integration.json)
pins #223 production source `a472446` and #222 `484d9e0`. Only the package test
lists conflict; the local preview keeps both lists. The resolved tree is
`3881cce7d1304b63bb8670011f46ba403d43ac8e`. Its library/GPU build, 14 CPU tests,
one MVS boundary/observability WebGPU test, 53 website tests and 91-page site
build pass. An earlier preview also passes the existing full and nine-view
streamed Dense WebGPU regressions. These are API/merge/regression checks, not
combined RGB quality measurements; #222 has not been merged by this work.

The finite scene remains compatible with `planMvsViews`: pass the actual source
image width/height alongside each camera. Pending directions do not enter dense
seeds, point normalization or 3DGS finite-point exports. Preserve pixel-center,
crop/resize, world-to-camera (`q = R X + t`, row-major) and camera-Z depth conventions.

After both revisions are fixed, compare the same RGB input in four combinations:
baseline/new SfM × baseline/new MVS. Predeclare MVS options and keep decoding,
frame selection, masks and their provenance fixed. Record registration failures
and disconnected maps before evaluating dense output; do not fill missing poses
or hide failures by comparing only a successful subset. Use a separate fixed-camera
ablation to isolate MVS changes. Report camera/focal/geometry errors where truth
exists, foreground retention and background false depth, plus full pipeline time,
transfers and memory. ROI counts alone do not establish surface accuracy or
completeness. This four-way real-input experiment remains future work.

## Use and remaining scope

```ts
const result = reconstructIncremental(images, pairs, {
  focalSearch: true, // One shared focal; cx/cy and pixel aspect are supplied.
  separateDirectionalObservations: true, // Experimental, opt-in.
});
// result.reconstruction contains finite XYZ only.
// result.tracking.untriangulatedTrackIds retains pending 2D tracks.
```

With supplied calibration, omit `focalSearch`; known per-view K can include zoom.
The shared-focal search still assumes the existing centered-principal-point width
convention. Unknown zoom/crop, multi-projection scopes, automatic masks, objects,
shots/relocalization/map merging, dynamic Gaussians and observation uncertainty
beyond existing conditional depth sensitivity are not added here. Textureless
sky supplies no bearings. A direction-only frame still lacks a registered world
center; an existing connected finite map remains necessary for this BA path.
The uploaded Shorts is not rerun as a complete RGB reconstruction in this stage.

The [final validation receipt](experiments/general-video/focal-directions-validation.json)
records the tested code commit and log hashes. Final review also keeps the
reported focal-score units consistent when a refined finite-only map registers
more views than a partial bearing model. Format, types, 309 core tests,
library/GPU build, the TypeShade CPU focal check and 89-page site build pass
again after that metadata correction. GitHub CI is reported separately on the PR.
