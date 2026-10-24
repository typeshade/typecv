# Persistent anchored inverse depth

Follow-up to #223. This branch starts at `c1465cb` and does not include #222's Dense changes.
Depth observability and explicit optical-path selection remain independent.

## Tasks

- [x] T1: Stable observation-owned track identity, deduplicated history and independent snapshots.
- [x] T2: Anchored inverse-depth projection/Jacobians and CPU multi-view fitting, including rotation degeneracy.
- [x] T3: Joint pose/point/focal refinement and undamped, gauge-fixed marginal uncertainty.
- [x] T4: Persistent updates, promotion/demotion, reanchoring/scale and CPU/GPU mapper/session integration.
- [x] T5: TypeShade measurement linearization and CPU/WebGPU agreement; no unreported CPU fallback.
- [x] T6: Diagnostics/export contracts, negative/uncertainty tests, builds and validation receipt.

## Contract

Unknown depth is an explicit state, never a huge XYZ or a zero-variance infinity hypothesis.
Unknown translation is never replaced with a zero camera center. Repeated observation identities
are evaluated once; relinearizing a BA result does not add the same measurements as new evidence.
Raw observations, not successive correlated point estimates, are the estimator's evidence.

The point solver uses a fixed reference chart `(R_ref, C_ref)` and parameters `(a,b,rhoZ)`.
The camera projection is evaluated without division by inverse depth:

`q_i = R_i R_ref^T [a,b,1] + rhoZ (R_i C_ref + t_i)`.

The chart is a coordinate choice, not an extra pose measurement. The view originally supplying
the chart remains free to move in joint BA. At rhoZ=0 translation derivatives are exactly zero.
Public states distinguish inverse optical-axis depth from inverse range. Only validated finite
states may be materialized as XYZ or passed to PnP/MVS; bearing constraints remain separate.

Uncertainty must state its conditioning. A point-only covariance with fixed poses/K is not total
SfM uncertainty. Joint estimates use the undamped robust linearization after fixing the world
gauge and eliminating nuisance point/camera/focal parameters. These remain local Gaussian
approximations under the selected inliers and image-noise model, not calibrated probabilities.
Singular information remains unobservable; numerical damping does not count as measured evidence.

Acceptance includes pure rotation with no finite-depth promotion, later-baseline promotion of the
same tracks, duplicate-observation idempotence, calibration/pose relinearization, gauge/scale
consistency, explicit optical exclusion and numerical backend agreement. Full real-video RGB
improvements, automatic reflection detection and physical reflective/refraction models require
separate measurement and are not inferred from synthetic tests.

## API and update boundaries

```ts
const depthTracking = new InverseDepthTracker({
  pixelNoiseStdDevPx: 1, // assumed coordinate noise, not estimated from the residual
  maxRelativeStddev: 0.2,
  minObservations: 3,
  refineSharedFocal: true, // only if a shared focal scale should be estimated
});
const result = reconstructIncremental(images, pairs, { depthTracking });
// The same object is accepted by IncrementalSfmSession / reconstructIncrementalGpu.
const record = result.inverseDepth && inverseDepthRecord(result.inverseDepth);
```

The feature is off unless a tracker is supplied. Existing default solver behavior and output
fields remain unchanged. A supplied tracker enables #223's finite/directional handling unless
the caller explicitly disables it. The object/scene CPU mappers and GPU/float/session paths
apply the estimator at completed updates, suppressing provisional callbacks. Native within-run
triangulation/PnP seeds still follow the existing mapper's admission rules; this is not a rewrite
of every internal PnP operation. Completed results and subsequent session resumes use admitted
finite geometry. Initialization failures carry the bearing-only snapshot without a fake map.

With tracking enabled, result tracks and landmark ids use the persistent observation-owned
ids. Historical tracks remain in snapshots as their frames leave the native matching window.
The estimator keeps raw pixels/calibrations/poses, never source RGB/descriptors. Repeated input
rebuilds the same likelihood rather than recursively fusing fitted means. Retained finite points
are numerical starting values, not extra prior factors. Track splits invalidate old lineages;
merges use each original feature once. Late camera-observation exclusions revoke historical
evidence and persist until replaced with an explicit updated mask.
Updates are transactional: a failed asynchronous refinement restores the previous state and
snapshots never expose a partially rebased chart. A device failure propagates to the caller.

Archived cameras/observations participate in the next joint solve. A common registered baseline
allows a similarity change of coordinates for historical poses/charts; inverse depth and its
covariance transform with scale. Without that baseline, old poses are withheld while raw tracks
survive. This is not scene-cut relocalization or disconnected-map merging. New poses/calibrations
relinearize retained observations; rejected depths may be promoted or demoted after another update.

## Estimation and uncertainty

The solver uses three parameters per finite/estimating point, and two for a depth-unobservable
bearing. Its fixed chart does not fix the camera that originally supplied it. The joint objective
contains raw pixel residuals under the existing radial camera model and Huber weighting. Signed
inverse depth can occur during testing; only positive, consistent, sufficiently certain estimates
are admitted. The default gate requires three registered observations, all retained residuals at
most 3px, positive inverse range beyond three local standard deviations, and relative standard
deviation at most 0.2. These experimental thresholds do not certify surface accuracy.

The first camera pose fixes world SE(3). With only that fixed camera, a two-dimensional tangent
basis for the scale camera's translation removes the remaining scale gauge; candidate similarity
normalization preserves that baseline. The undamped final Schur complement supplies the joint
camera covariance and point marginal `V^-1 + T^T S^-1 T`. Rank-deficient systems publish no depth
covariance. LM damping is used for optimization steps only. Optional focal groups add shared
log-focal variables; principal point and radial coefficients remain fixed. With explicitly fixed
exterior views, the result is labeled `fixed-exterior`, not full pose uncertainty.

The uncertainty is a **local robust Gaussian approximation** conditional on the observation
ownership, image-noise model, camera model and gauge. It is not a nonlinear/multimodal Bayesian
posterior, a rolling-shutter model, or uncertainty for every intrinsic parameter. Broad estimates
near zero stay pending. Bearing-only states retain a separate 2x2 pinhole-(a,b) covariance with
its conditioning; it is not a covariance of world orientation. Records distinguish inverse Z
from inverse Euclidean range and use current monocular-map units, not meters.

## Execution and limits

CPU reference math uses f64. The TypeShade path computes lossless-packed f64 observation
Jacobians on the selected backend, including emulated f64 on WebGPU. Raw-track initialization,
Schur assembly/solve, marginal covariance and state control remain **host f64**; the execution
label explicitly reports this hybrid path. No device error triggers a CPU retry. Software-WebGPU
checks establish numerical behavior, not hardware-GPU performance or real-time throughput.

The reduced solve chooses dense or sparse storage from camera co-visibility. Its explicit bounds
are `maxCameraParameters` (default 1536), `maxSchurEntries` (300,000 upper-triangle structural
entries), and `maxFactorEntries` (600,000; the dense path counts its full square allocation).
These bound logical matrix entries, not JavaScript Map overhead or total process memory. Sparse
fill can exceed the budget even when the input graph fits. Exceeding a bound fails explicitly;
it does not freeze uncertain poses, prune observations, or switch to a fixed-exterior solve.
Explicit `variableViewIds` can bound work but the uncertainty then conditions on the held exterior
poses. This remains a bounded full-history solve, not an unbounded fixed-lag marginalization system.

GPU measurement rows are split to fit active-array, binding and dispatch limits, optionally
further limited by `measurementBatchSize`. A tail batch evaluates only its remaining rows. A
failure in any batch discards that update; published tracker state rolls back without a CPU retry.
Diagnostics record the selected solver, matrix entries, deferred tracks and GPU batches. Host
measurement snapshots, output Jacobians, keypoints and raw history still consume host memory;
GPU batching does not bound them. None of these experiments is enabled in the Lab UI.

## Records and Dense handoff

`result.inverseDepth` and `tracker.snapshot()` preserve estimates, covariance, assumptions,
conditioning, reasons, observation ids and backend. `inverseDepthRecord` converts typed arrays to
portable JSON under `typecv-inverse-depth-v1`. Unresolved depths have no materialized XYZ. This
is a library export contract; automatic website uploads and a Radiance consumer are not wired.
The ordinary reconstruction contains admitted finite points only and retains the existing MVS
camera/landmark schema. #222's explicit optical masks can be applied independently; this branch
does not turn depth observability into a reflection classifier or a reflection-rendering model.


## First-phase validation receipt

The [validation record](experiments/general-video/persistent-inverse-depth-validation.json)
records 327 core tests, four TypeShade CPU tests, two software-WebGPU tests, 42 website unit
checks, type/format checks, library builds and a 91-page site build. Astro reports no errors or
warnings and four existing hints. A strict declaration consumer checks both core-to-GPU and
GPU-to-core tracker options. The completed first-phase code is `5909b0c`; its receipt commit
changes only documentation/data. The scaling refinements below have a separate receipt.

On the 16-observation numerical fixture, WebGPU's maximum relative pixel/Jacobian difference is
1.75e-14 and inverse-depth variance difference is 1.23e-12. The 10-view session with an 8-view
matching window retains 25 finite points and exactly 250 raw observations through `finish()` on
CPU and WebGPU, with zero live arrays under poolBytes=0. An injected shader failure frees arrays;
an asynchronous failure rolls back the published state and propagates without a CPU retry.
The joint-noise experiment's 80 trials place truth within two local standard deviations in
97.5% of trials, with standardized RMS 1.050. This does not calibrate real-video uncertainty.

An isolated composition with #222 `a01fbd7` resolves only test-list conflicts. Nine CPU checks
and type checking pass, including an explicit reflection-mask fixture with the new tracker.
No integration merge was pushed; no combined GPU or real-RGB quality comparison was performed.
Hosted CI was still running at local completion. These local checks do not imply CI success.

## Scaling refinement tasks

- [x] S7: Freeze the three exported sparse inputs and baseline protocol; diagnose dense cross blocks/inversion.
- [x] S8: Sparse Schur/compact cross blocks, exact selected covariance and per-track degeneracy isolation.
- [x] S9: Budgeted GPU measurement batches and large-window CPU/WebGPU/failure checks.
- [x] S10: Full exported-record replay, uncertainty/cost tradeoffs and final regression receipt.

The [frozen inputs](experiments/general-video/inverse-depth-scaling-inputs.json) contain hashes
and schemas only. The canyon/tower/desert exports contain 41/22/21 registered views and
11,897/7,113/5,530 finite input tracks respectively. Their 239/125/119 pose parameters exceed
the first implementation's 96-parameter default. Replay uses all exported sparse observations,
fixed intrinsics, sigma=1px, Huber=3px and 0/1 joint iterations. A dense reference may explicitly
raise its budget for an equal-problem comparison; that alone is not the refinement.
No RGB rematching, semantic ROI changes or ground-truth depth is supplied by this replay.

The refinement stores point/camera cross terms only for cameras observing each point. Sparse
Cholesky uses minimum-degree ordering and Takahashi selected inversion, so point marginals use
the needed camera covariance entries without forming its full inverse. Structural fill is retained;
a missing selected entry is an error, never silently zero. Auto chooses dense storage for small
or highly connected camera graphs and sparse storage for local overlapping tracks. An explicit
`solver` selects either reference. The choice changes arithmetic/storage, not pose conditioning.
Unsupported point blocks are omitted from both the objective and camera information for that
linearization. Their observations remain stored and depth remains unconfirmed. One such block
therefore cannot erase otherwise observable point covariances or supply a fictitious translation
constraint through its arbitrary finite seed.

S9 checks use five TypeShade CPU tests and three software-WebGPU tests. A 24-view, 252-track,
1008-observation problem retains all 137 variable pose parameters, uses sparse Schur and 16
measurement batches of at most 127 rows, and leaves zero live arrays with pooling disabled.
Its WebGPU/CPU inverse-depth variance difference is at most 2.54e-10 relative. A smaller fixture
compares chunked and unchunked Jacobians directly. Admission rejects a budget that cannot fit
one row; injected first/second-batch failures free arrays and return no partial rows. Queued
measurements are snapshotted before caller mutation. A degenerate point regains covariance when
a subsequent baseline observation makes its depth observable.

## Exported-record scaling measurements

The [replay results](experiments/general-video/inverse-depth-scaling-results.json) freeze code
`619f721`, baseline `ce5aad9`, input and executable hashes, and 24 full-record runs (four solvers,
two iteration counts, three records). Each run starts a fresh Node 24 process. The baseline
explicitly raises its old camera budget so the comparison solves the same problem. These inputs
contain already-triangulated finite sparse tracks; this replay cannot measure recovery of missing
tracks, new RGB matching, reflected-surface accuracy, or Dense MVS coverage.

One permitted joint iteration, baseline to auto (the canyon record rejects its attempted step):

| Export | Views | Admitted points, before/after | Peak process MiB | Solve seconds |
| --- | ---: | ---: | ---: | ---: |
| `a4d735ce` | 41 | 4626 / 4626 | 615.0 / 502.4 | 1.633 / 1.895 |
| `6c42e7d3` | 22 | 4386 / 4386 | 435.7 / 411.4 | 1.224 / 1.376 |
| `1e118cc5` | 21 | 4172 / 4172 | 458.0 / 430.8 | 1.591 / 1.810 |

All three graphs are highly connected, so auto chooses dense Schur while retaining compact
point/camera cross blocks. Their scalar cross storage shrinks by 13.39x / 5.58x / 3.39x. Every
point has a covariance in these records and no track is deferred. Baseline/auto marginal variance
relative differences are at most 4.56e-12; forcing sparse instead agrees with dense to 1.28e-11.
Admission counts are unchanged, not evidence of improved geometry. These single runs use less
peak process memory but take 12–16% longer; sparse arithmetic is not universally faster. Peak RSS
includes parsing/module/summary memory, and neither that measure nor logical-entry counts are a
GPU residency measurement.

A separate synthetic local-overlap graph has 128 views, 1500 tracks, 6000 observations and 761
variable pose parameters. Zero-iteration linearization/covariance takes 0.789s baseline and 0.314s
auto in one run; peak process memory is 184.4/129.3 MiB. Auto uses sparse Schur: 16,143 selected
camera covariance entries instead of 579,121 full-square entries, and 107,712 cross scalars instead
of 3,424,500. Variances agree to 1.47e-8 relative. Neither version admits any point under the stated
1px global-uncertainty gate. This measures scaling of the same local information problem, not
convergence, reconstruction quality or guaranteed throughput.

Reproduce with `scripts/measure-inverse-depth-scaling.mjs --records <private-record-root>
--out <result-directory> --baseline <esbuild-bundle-of-ce5aad9>`. The input manifest checks record
hashes before each run. The results JSON includes the separate synthetic command. Private raw
records and per-landmark estimates are not committed.


## Scaling validation receipt

The [scaling validation record](experiments/general-video/inverse-depth-scaling-validation.json)
records code `619f721`: **332 core tests**, **244 full TypeShade CPU-suite tests** (including
five inverse-depth cases), **3 inverse-depth software-WebGPU tests**, and **42 website unit
checks** pass. The existing optional MVS-default tuning experiment is skipped unless
`TYPECV_MVS_DEFAULTS=1` (244 passed, 1 skipped, 0 failed). Type/format checks, core/GPU builds, strict public declaration interoperability,
Astro (zero errors/warnings, four existing hints), and the 91-page site build pass. The scaling
receipt commit contains only documentation, aggregate data and the replay script. Hosted CI for
that final commit is separate from these local results.

## Performance follow-up tasks

- [x] P11: Freeze `b036bd8` and profile the unchanged full-observation replay.
- [ ] P12: Remove allocation/repeated lookup overhead without changing observations, gates or covariance arithmetic.
- [ ] P13: Alternating paired repeated measurements, numerical/CPU/WebGPU regression and final receipt.

The performance baseline is `b036bd8` (CPU bundle SHA256
`9fcd06a54186ddc809bca7f1a5b55ffd9757959e69636421d5f0d221a2c71cd9`).
Inputs and solver/noise settings are the same as the scaling replay above. A canyon worker CPU
profile attributes 22.3% of whole-worker samples to bundle assembly, 11.2% to reduction, 10.0%
to point marginal calculation and 8.0% to GC. This profile includes loading/postprocessing and
is for bottleneck selection, not a timing comparison. Performance acceptance uses five fresh
processes per version/record, alternating execution order, one permitted joint iteration, and
all three frozen real records. Report medians and ranges; keep admission, costs, pose/point
parameters and all covariance entries unchanged within numerical tolerance. Also check the
128-view local graph and the existing degenerate-track, focal and CPU/WebGPU fixtures.
