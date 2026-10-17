# GPU geometric vision and optimization

These operators belong to TypeCV's geometric-vision layer. Pose estimation,
triangulation and bundle adjustment are reusable by registration,
tracking and reconstruction pipelines; SfM is one consumer. All APIs below are
exported from `typecv/gpu` and execute numerical work through compiled TypeShade
source. The root package retains the independent synchronous CPU reference.

## Implemented operators

| API                                                    | Device computation                                                                                                                               | Host boundary                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| `verifyImagePairGpu`                                   | Radial inversion, stable compaction, H/E consensus and relative-pose checks in one plan                                                          | Final source row indices, H support and pose/quality summary   |
| `solveP3pBatchGpu`                                     | Grunert polynomial, bounded real-root isolation, distance refinement and pose validation                                                         | Explicit candidate readback through the resident batch handle  |
| `estimateHomographyGpu` / `estimateEssentialRansacGpu` | Normalization, minimal proposals, QR/direct SVD, cooperative scoring, adaptive budget and refits                                                 | Final model, mask and counters                                 |
| `reconstructTwoViewGpu`                                | Essential consensus, four pose branches, cheirality, joint two-camera LM, final points/quality                                                   | Final pose, points, masks and diagnostics                      |
| `estimatePosePnPGpu`                                   | Radial inversion, distinct-index sampling, every P3P branch, reprojection scores, adaptive trial budget, consensus and pose LM                   | Final pose/metrics, mask and counters; no intermediate reads   |
| `triangulateTracksGpu`                                 | Direct DLT SVD, all-observation/pair hypotheses, robust consensus, point LM, depth and parallax checks                                           | Final points, masks and rejection codes                        |
| `evaluateObservationsGpu`                              | Projection, threshold classification and residual metrics                                                                                        | Masks and a two-number summary                                 |
| `bundleAdjustGpu`                                      | Joint pose/point Jacobians, robust blocks, point elimination, matrix-free Schur PCG, block preconditioning, LM decisions and gauge normalization | Final accepted coordinates and diagnostics; no iteration reads |
| `reconstructIncrementalGpu`                            | Composes resident GPU pair verification, PnP, triangulation, observation filtering and local/global BA                                           | Final result indexing, packing, track graph and mapper control |

The mapper and synchronous reference share a typed operation generator rather
than separate copies of registration/track control flow. The asynchronous driver
selects device operators explicitly. `GeometryEstimationError` identifies an
expected lack of reliable geometry. Device/resource/programming failures escape
per-pair/per-view rejection handlers; they never trigger an automatic CPU retry.

```ts
import {
  ShadeDevice,
  estimatePosePnPGpu,
  triangulateTracksGpu,
  bundleAdjustGpu,
  reconstructIncrementalGpu,
} from 'typecv/gpu';

const device = await ShadeDevice.open();
try {
  const registration = await estimatePosePnPGpu(device, packedWorldXYZ, packedPixels, camera, {
    iterations: 1000,
    thresholdPx: 3,
    minInliers: 12,
    seed: 42,
  });
  const triangulated = await triangulateTracksGpu(device, observationTracks, {
    maxErrorPx: 3,
    minAngleDeg: 0.5,
  });
  const optimized = await bundleAdjustGpu(device, sparseReconstruction, {
    iterations: 20,
    huberPx: 2,
    anchorViewId: 10,
    scaleViewId: 20,
  });
  const mapped = await reconstructIncrementalGpu(device, imageFeatures, pairMatches, {
    seed: 42,
    ransacIterations: 750,
  });
  console.log(registration.pose, triangulated.status, optimized.diagnostics, mapped.execution);
} finally {
  await device.dispose();
}
```

All poses use `Xcamera = R * Xworld + t`; R is row-major. `t` is not a normalized
direction in PnP/BA. Pixels use the supplied pinhole + radial k1/k2 model. Camera
parameters remain fixed unless BA requests `refineSharedFocal: true`, as described
below. Inputs are captured before queued
execution; caller-owned data and observation identities are preserved. Result
arrays are caller-owned snapshots, independent of subsequent buffer reuse.

E/H, radial pair inversion and relative-pose APIs are documented in
[GPU projective estimation and relative pose](gpu-initialization.md). The
mapper uses these operators throughout initialization.

## PnP execution

Inputs are `3*N` world coordinates and `2*N` pixels, with at least four
correspondences. Each trial uses a seeded u32 xorshift stream derived from its
trial index and selects three distinct indices without retry loops. Modulo
indexing has finite-word bias; the confidence budget is the ordinary RANSAC
heuristic, not a statistical guarantee. Reproducibility is within the selected
tier, not an assertion that this sampler reproduces the CPU reference's stream.

All four P3P candidate slots stay resident. A 64-lane workgroup evaluates one
candidate/block and reduces counts and squared inlier costs with shared memory
and barriers. Ties prefer larger consensus, smaller inlier squared cost, then
the earliest candidate. The device updates the adaptive trial budget between
batches. Default minimum attempts are `min(64, iterations)`; attempts respect
the hard cap, including rejected minimal samples.

Three fixed-consensus LM passes refine the winner. Analytic distorted-projection
and left-pose derivatives are authored in TypeShade. A 32-lane workgroup reduces
44 normal-equation/statistic components using 11,264 bytes of shared storage.
Damped 6x6 systems use diagonal equilibration and Cholesky. Trial cost and LM
acceptance are device-side; a refined pose replaces the robust winner only when
its consensus/count-cost ordering improves. This does not implement the CPU
reference's optional normalized-DLT/planar refit.

Solving submits one plan. Only the final pose/summary (128 bytes), N u32 flags
and eight u32 counters are read: `160 + 4*N` application bytes. Device flags
disable trials/LM work after convergence; host scheduling is currently bounded
by the requested caps, so disabled dispatches may still appear in reports.
The explicit CPU test tier uses the same lane operators with serial reductions
because workgroup barriers require a device. It is not an automatic fallback.

## Robust triangulation

Pass an array of tracks, each containing `{ camera, pose, pixel }` observations.
One invocation solves one track; tracks execute in parallel. Direct one-sided
Jacobi SVD operates on normalized DLT rows without forming `A^T A`. It evaluates
an all-observation hypothesis plus up to 32 pair hypotheses by default, ordered
by baseline. Maximum consensus and squared residuals select the seed; a fixed
consensus is refit and refined using point LM. Final classification requires
positive depth and the requested parallax. Angle evaluation is authored in f64
with a bounded arctangent series.

Rejected tracks return `undefined` and a status: 1 insufficient observations,
3 no validated solution/consensus, 4 insufficient parallax. Accepted tracks use
status 0. Output flags/slots are cleared on reuse. Limits are 64 observations
per track, at most 64 pair hypotheses, and at most 100 refinement iterations.
The default is 10 point iterations. Full readback is `44*T + 4*M` bytes for
T tracks/M observations. Numeric kernels have bounded serial work per track;
cooperative SVD/large-track sharding are not implemented.

## Joint bundle adjustment

The shader forms robust camera/point blocks and per-observation cross blocks.
It eliminates each damped 3x3 point block and applies
`S*x = U*x - W*V^-1*W^T*x` without materializing the reduced camera matrix.
Camera Schur-diagonal blocks provide a 6x6 block preconditioner. PCG vector
updates, dot products and convergence checks run on the device. Dot products
currently use one invocation over the bounded camera vector; point gathers and
camera applications run in parallel.

PCG must meet its relative residual tolerance before the LM candidate can be
accepted. Insufficient/failed solves reject the entire step and increase damping;
the accepted state stays intact. `maxCgIterations` defaults to
`min(96, max(24, 12*C))`, and `cgRelativeTolerance` to 1e-6. Diagnostics report
actual CG iterations and linear-solver failures. The fixed Huber observation
objective cannot improve by silently dropping observations.

The anchor pose fixes the world frame. With one fixed camera, a device-side
similarity normalization preserves the initial anchor/scale-camera baseline
without changing projection. Local BA fixes cameras outside `variableViewIds`;
the scale camera must remain fixed when there are multiple fixed poses. An
empty variable window optimizes points only. Fixed host constants are returned
bit-for-bit from captured inputs, including precision beyond the emulated format.
No accepted steps returns independent copies of the original geometry.

Current limits are 2..64 cameras, 0..100 LM iterations, and 1..256 PCG iterations.
A connected observation graph, unique feature ownership, nonzero gauge baseline
and initially projectable observations are required. Broader intrinsic models,
large-scene partitioning and calibration priors remain separate work.

Observation buffers, normals, cross blocks, preconditioners, solver vectors,
current/trial coordinates and LM state remain resident for the entire call.
Each LM iteration is a submission; initialization and output packing add two.
Only the final `12*C + 3*P + 26` f64 values are read. No Hessian, Jacobian,
candidate scene or iteration convergence state crosses the host boundary.
The application buffer budget is `1208*C + 204*P + 380*M + 424` bytes.
Pooling and runtime/staging allocations are separate. f64 initial-zero uploads
follow TypeShade's current `resident(Float64Array)` path; reused outputs avoid
those initialization uploads.

### Optional joint shared-focal refinement

Set `refineSharedFocal: true` on `bundleAdjustGpu` or the incremental mapper to
optimize one global log-focal coordinate together with camera poses and 3D points.
Each accepted focal step multiplies every view's `fx` and `fy` by the same positive
scale. Principal points, radial coefficients, each view's `fx/fy` ratio and the
relative focal ratios between views stay fixed. This is shared scale refinement,
not an unconstrained calibration of every camera.

The GPU path uses TypeShade emulated f64 and a matrix-free Schur/PCG system with
`6*C + 1` rows. The extra global row includes camera–focal and point–focal cross
terms during point elimination, camera application and point back-substitution.
Focal, pose and point changes form one LM candidate; nonpositive/nonfinite focal
values, invalid projections, an unconverged linear solve or a non-improving robust
objective reject the entire candidate and preserve the accepted state. The anchor
pose and scale-camera baseline use the same gauge rules as pose/point BA.
`variableViewIds` constrains poses; the global focal scale remains active even
when that pose window is empty.

The independent CPU `bundleAdjust` reference supports the same flag and assembles
a dense reduced camera/focal Schur system. The default fixed-intrinsics path is
unchanged. Shared-focal refinement replaces the earlier discrete focal-grid
search: `iterations: 0` now returns unchanged geometry and calibration rather
than searching focal candidates.

The GPU variant accepts 2..64 cameras and 0..100 LM iterations. Its
`maxCgIterations` defaults to `min(256, max(32, 12*C + 1))`, with an allowed range
of 1..256; `cgRelativeTolerance` defaults to 1e-6 and accepts 1e-12..1e-2.
Working-byte, storage-binding and dispatch limits are checked before execution;
unsupported scenes fail explicitly. Camera count alone does not guarantee that a
scene fits. These contracts establish the implemented optimization scope, not
COLMAP parity or a measured speed improvement.

## Numeric and resource contracts

GPU f64 is TypeShade's float-flavor emulation: approximately 48 significand bits
over the f32 exponent range, not IEEE binary64. See the
[capability review](typeshade-capabilities.md). Scalar inputs are finite, at
most 1e9 in magnitude; nonzero values are at least 1e-30. Focal lengths are
1e-6..1e6, principal points within ±1e9, radial coefficients within ±1, and
normalized image radii squared below 1e8. Error/Huber thresholds are
1e-12..1e6. Projection rejects nonpositive depth, radial folds and coordinates
or derivatives outside the bounded shader domain. Scale the scene before using
extreme coordinates; these guards do not guarantee conditioning at every input.

PnP limits are 1..100,000 requested trials and 0..100 pose iterations. Batch
capacity also respects storage, dispatch and working-byte limits. Calls queue
per device; disposal drains queued work. Failed operators destroy their owned
buffers instead of recycling them. Limits cause explicit rejection, never
backend substitution. Current APIs repack/read geometry between mapper stages;
a persistent scene with zero-copy handoff across the whole mapper is still a
design target. Initialization E/H solvers and two-view recovery now also execute
in TypeShade. Pair inversion, stable compaction, H/E and pose checks share one
resident plan; see the [initialization guide](gpu-initialization.md).

## Verification and real data

```sh
npm run check
npm run test:webgpu
# Reuse the prepared free-video experiment; raw media is not downloaded by tests.
node scripts/benchmark-geometry.mjs data/pyramid/sequence/sequence.json data/pyramid/gpu-model.json > data/pyramid/gpu-geometry.json
```

Tests cover clean/noisy/outlier/planar/four-point PnP, multiple world scales,
direct-SVD triangulation with corrupted observations, weak baselines, shared
mapper control, input snapshots and failure cleanup. BA checks independent
residuals and the CPU dense-Schur reference, robust objectives, shifted/rotated
anchors, scale invariance, local/point-only updates and forced PCG rollback.
Browser instrumentation counts real dispatches/submissions/maps. PnP reads only
after its plan; BA has one final read, including when an LM step is rejected.

The current [resident-pair record](experiments/gpu/resident-pair-verification-software.json)
uses TypeShade numerical geometry from pair inversion through BA on all 28 pairs.
It registers 8/8 views, 1,723 points and 5,122 observations, with approximately
0.45492px mean Euclidean residual. CPU uses a different seed pair and retains
1,729 points/5,153 observations at 0.46278px. After camera-center similarity
alignment, the center RMS difference is 0.00934 in CPU initial-baseline units,
and the mean rotation difference is 0.237°. The record includes the alignment,
per-camera differences, model hashes, every pair outcome and actual device calls.
Both CPU and GPU model hashes match the previous initialization record, while
whole-reconstruction maps fall from 348 to 152. The initialization guide records
the increased pool/dispatch costs alongside reduced readback traffic.
Different accepted observations and assumed calibration prevent interpreting
the smaller residual as a ground-truth accuracy improvement.

The earlier [pyramid record](experiments/gpu/incremental-geometry-software.json),
which predates the initialization migration, retains
all 28 pair outcomes, registration order, per-stage transfers and actual calls.
Both paths register 8/8 views and 1,729 points. CPU accepts 5,153 observations;
the device path accepts 5,152. Mean Euclidean residuals are approximately
0.46278px and 0.46206px on those different observation sets. All eight local/global
BA calls use the device implementation. This is reference agreement under
assumed calibration, not ground-truth accuracy or COLMAP equivalence.

The recorded adapter is SwiftShader. Device-stage execution/correctness is real
WGSL, but these software timings do not measure physical-GPU acceleration.
Bounded host scheduling still emits disabled iterations and can have substantial
dispatch overhead. Hardware profiling, convergence-aware scheduling and a
persistent scene are the next performance gates.
