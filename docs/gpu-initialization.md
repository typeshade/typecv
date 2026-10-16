# GPU projective estimation and relative pose

Projective model fitting, camera inversion and relative pose are reusable
geometric-vision operators. SfM consumes them through the incremental mapper;
they do not depend on a reconstruction graph. The APIs are exported from
`typecv/gpu`, with independent synchronous references in the root package.

| API | Device work | Returned data |
| --- | --- | --- |
| `undistortCorrespondencesGpu` | Invert both radial models on the branch connected to the optical axis; reject folds | Ideal pixel rows and original row indices |
| `estimateHomographyGpu` | Four-point sampling, normalized DLT, symmetric transfer scoring, adaptive RANSAC and three refits | H, consensus mask/count and trial counter; `undefined` for no reliable model |
| `estimateEssentialRansacGpu` | Radial inversion, eight-point sampling, normalized DLT, essential projection, Sampson scoring, adaptive RANSAC and four refits | E, consensus mask/count, MSAC and trial counter |
| `reconstructTwoViewGpu` | Essential consensus, four pose branches, cheirality, joint two-camera/point LM, point refinement, final consensus and quality checks | Unit-baseline relative pose, points/masks and diagnostics |

```ts
import { ShadeDevice, estimateHomographyGpu, reconstructTwoViewGpu } from 'typecv/gpu';

const device = await ShadeDevice.open();
try {
  // H operates on ideal pinhole pixel rows [x1,y1,x2,y2,...].
  const planar = await estimateHomographyGpu(device, idealPixels, {
    thresholdPx: 2,
    iterations: 500,
  });
  // Relative-pose input can include k1/k2 radial distortion in either camera.
  const geometry = await reconstructTwoViewGpu(device, measuredPixels, firstCamera, secondCamera, {
    threshold: 0.002, // Normalized-camera Sampson distance, approximately radians.
    iterations: 750,
    refinementIterations: 15,
    minParallaxDeg: 0.1,
    maxReprojectionErrorPx: 4,
  });
  console.log(planar?.inlierCount, geometry.R, geometry.t, geometry.execution.report);
} finally {
  await device.dispose();
}
```

## Numerical path

Each hypothesis uses Hartley centering and mean-distance scaling in each image.
Streaming Givens QR compresses the design rows to a 9x9 triangular factor.
Direct one-sided Jacobi SVD of that factor finds the projective null vector.
This keeps local storage bounded for consensus refits without forming `A^T A`
and squaring the condition number. Hypotheses execute in parallel with eight
lanes per workgroup; the QR/SVD within one hypothesis is serial. This is a
correctness-oriented implementation, not a claim of optimal cooperative SVD.

The second-smallest singular value must exceed 1e-7 of the largest for E and
1e-9 for H. H must also be invertible. E is denormalized and projected to equal
leading singular values and zero third singular value using a direct 3x3 SVD.
The two leading directions are orthogonalized, with right-handed basis
completion. Exact planar/collinear eight-point systems and collinear homography
systems are rejected; noisy model competition still needs the mapper's H check.

A stateless trial-index xorshift stream samples distinct indices with bounded
sorted insertion. Modulo indexing has finite-word bias. A 64-lane workgroup
scores one hypothesis/block and reduces counts and squared inlier residuals.
Selection prefers larger consensus, smaller squared inlier cost, then the first
trial. At equal count, this is the same ordering as normalized truncated MSAC.
Only an improving refit replaces the winner. Residuals and fitting use
TypeShade's emulated f64; adaptive probability control uses f32.

The adaptive budget is evaluated on device between batches. For small
probabilities, a bounded series evaluates `log(1-p)`: direct f32 subtraction can
round `1-p` to 1, incorrectly turn the required-trial calculation into negative
infinity, and stop prematurely. E, H and PnP share the stable calculation. A
conservative extra trial handles rounding around integer boundaries. Minimum
attempts default to `min(64, iterations)` for E and `min(32, iterations)` for H;
the hard cap includes rejected samples. Batch size defaults to 32 and respects
the device capacity. The confidence formula remains a RANSAC heuristic.

## Relative pose and refinement

The four E decompositions are evaluated in parallel over correspondences using
direct 4x4 DLT SVD, positive depth in both views and a majority-cheirality check.
Optimization keeps the original positive-depth consensus fixed. Analytic
projection/pose/point Jacobians form robust two-camera blocks. Damped 3x3 point
elimination produces a 6x6 camera Schur system; a 32-lane workgroup reduces
44 components using 11,264 shared bytes. Equilibrated Cholesky solves the
camera step, followed by point back-substitution. This replaces the independent
CPU reference's finite-difference variable-projection optimizer.

The first camera fixes the world frame. Every trial normalizes second-camera
translation and all points by the same scale, preserving image projections and
unit baseline. Failed solves, invalid depth/projection and increasing robust
cost reject the entire trial. Point identity stays fixed during optimization;
cost cannot improve by dropping points. After re-triangulation and point LM,
the original fixed set's mean Euclidean residual must also improve before the
pose replaces the original one. Otherwise the original pose is restored.

The final E is rebuilt from accepted pose, reclassified by Sampson distance,
and all accepted observations are re-triangulated. Final points require positive
depth, bounded per-view distorted-pixel error and sufficient parallax. Median
parallax is selected on device by bounded three-way quickselect. Low baseline,
insufficient reliable points and numerical failure raise `GeometryEstimationError`.
Invalid point rows are returned as NaN triplets with a zero quality flag.

## Execution and resource contracts

E/H RANSAC and optional two-view recovery form one TypeShade plan and one
compute submission. No hypothesis, score, intermediate mask, pose, point or
convergence counter is read during the plan. E/H return three final snapshots:
`200 + 4*N` application bytes. Two-view recovery returns five snapshots:
`608 + 32*N` bytes. Each snapshot adds a staging-copy submission. The legacy
`scoreDispatches`/`scoreSubmissions` counters now cover the complete initialization
plan; `execution.report` provides the authoritative dispatch/submission groups.

For N rows, B hypotheses/batch and Q=ceil(N/64), the consensus application-buffer
budget is `72*N + 200 + B*(148 + 16*Q)` bytes. Recovery adds
`564*N + 960 + 352*ceil(N/32)`. Undistortion uses `68*N` bytes and reads `36*N`.
These counts exclude runtime uniforms/staging, compiler resources and pooled
buffers. Working/storage/dispatch limits are checked before execution. All
output slots and flags used by a call are initialized even when buffers are reused.

The [geometry numeric contract](gpu-geometry.md#numeric-and-resource-contracts)
applies: GPU f64 is approximately 48 significand bits over the f32 exponent
range, not IEEE binary64. Trials are bounded to 1..100,000 and refinement to
0..100 iterations. Severe conditioning can still cause explicit numerical
rejection within the input bounds. Backend failure never selects a CPU retry.
The explicit CPU test tier executes the compiled lane operators with serial
folds in place of device workgroup barriers.

The mapper now executes all implemented numerical geometry through TypeShade:
pair inversion, E/H estimation, relative pose, PnP, triangulation, filtering and
BA. Host work includes validation, row compaction, data packing, track ownership
and mapper control. APIs still repack/read between stages; a shared persistent
scene and multi-pair batching remain performance work. Disabled RANSAC/LM
dispatches are still scheduled up to requested caps. E remains an eight-point
solver; calibrated five-point, MAGSAC/stronger model selection, broader camera
models and intrinsic optimization remain algorithmic work.

Tests use independent scalar image formation, direct CPU DLT, radial folds,
outliers, coordinate offsets/scales, exact degeneracies, weak baseline, pose
constraints and actual browser dispatch/map instrumentation. The software
adapter proves WGSL execution, not physical-GPU speedup. Real-video results
must retain every pair and failure; see the [experiment index](README.md#experiments).

The [recorded pyramid run](experiments/gpu/incremental-initialization-software.json)
registers all eight views through this initialization and the existing GPU mapper:
1,723 points, 5,122 observations and 0.45492px mean Euclidean residual. All 28
pair outcomes are retained, including one cheirality rejection. Its seed pair
is [4,5], versus [1,4] for the independent CPU run; the benchmark therefore also
records a proper-rotation camera-center similarity alignment. The CPU model
hash is unchanged by the shared-driver refactor. Actual calls are 24,806
dispatches, 552 submissions and 348 maps across the entire reconstruction,
including failed geometry and stage boundaries. The uncontrolled software
sample is about 82 seconds; hardware throughput remains unmeasured.
