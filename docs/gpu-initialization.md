# GPU projective estimation and relative pose

Projective model fitting, camera inversion and relative pose are reusable
geometric-vision operators. SfM consumes them through the incremental mapper;
they do not depend on a reconstruction graph. The APIs are exported from
`typecv/gpu`, with independent synchronous references in the root package.

| API                           | Device work                                                                                                                          | Returned data                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `verifyImagePairGpu`          | Radial inversion, stable row compaction, H/E consensus, relative pose and quality checks in one plan                                 | Original inlier row indices, H support, pose/quality summary or an expected rejection reason |
| `undistortCorrespondencesGpu` | Invert both radial models on the branch connected to the optical axis; reject folds                                                  | Ideal pixel rows and original row indices                                                    |
| `estimateHomographyGpu`       | Four-point sampling, normalized DLT, symmetric transfer scoring, adaptive RANSAC and three refits                                    | H, consensus mask/count and trial counter; `undefined` for no reliable model                 |
| `estimateEssentialRansacGpu`  | Radial inversion, eight-point sampling, normalized DLT, essential projection, Sampson scoring, adaptive RANSAC and four refits       | E, consensus mask/count, MSAC and trial counter                                              |
| `reconstructTwoViewGpu`       | Essential consensus, four pose branches, cheirality, joint two-camera/point LM, point refinement, final consensus and quality checks | Unit-baseline relative pose, points/masks and diagnostics                                    |

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

For pair filtering without downloading an initial point cloud, use the same
operator consumed by the mapper:

```ts
import { verifyImagePairGpu } from 'typecv/gpu';

const pair = await verifyImagePairGpu(device, measuredPixels, firstCamera, secondCamera, {
  threshold: 0.002,
  homographyThresholdPx: 2,
  iterations: 750,
});
if (pair.geometry) {
  // Indices refer to the original raw rows, including gaps from lens rejection.
  console.log(pair.inlierRows, pair.homographyInlierCount, pair.geometry.R, pair.geometry.t);
} else {
  console.log(pair.reason);
}
```

The independent root `verifyImagePair` has the same geometric output contract.
Both verification APIs default to zero refinement iterations; the standalone
`reconstructTwoViewGpu` keeps its existing refinement default. Verification H
uses ideal pinhole pixels, a default threshold of `maxReprojectionErrorPx`, a
trial cap of `min(iterations, 500)`, confidence 0.999 and the same seed as E.
E's `threshold` is in normalized camera coordinates. Verification quality
residuals use ideal pixels, including for radial cameras. The standalone
two-view API evaluates quality in the supplied distorted pixels instead.

`inlierRows` is the final epipolar consensus, not the point-quality mask;
`positiveDepthCount` additionally applies triangulation, error and parallax
checks. Fewer than eight invertible rows, degeneracy or failed pose quality
return an empty row list and a reason. A geometric rejection retains H support
when at least eight rows exist. Malformed inputs, unsupported numeric/resource
ranges and backend failures throw. An empty valid input performs no device work.
Mapper-specific homography dominance and seed eligibility stay in the mapper.

## Numerical path

Each hypothesis uses Hartley centering and mean-distance scaling in each image.
Streaming Givens QR compresses the design rows to a 9x9 triangular factor R.
This keeps local storage bounded for consensus refits without forming `A^T A`
and squaring the condition number. One dispatch of fixed cost then solves R
for its null vector (`solvePairModelNull`): every pivot below 1e-12 of the
largest is raised to that value, so that R is invertible (each singular value
moves by at most that much), and four inverse iterations on `R^T R`, two
triangular solves each, refine the null vector from the unit vector of the
smallest pivot. A minimal sample fills eight rows, so its ninth pivot is the one
raised and its null vector is exact to the working precision. The
second-smallest singular value comes from four inverse iterations orthogonal to
the null vector, and the largest from four power iterations. No kernel iterates
to convergence: the error of one inverse iteration is `(sigma_9 / sigma_8)^2`,
far below a double in any well-posed sample, and a sample whose `sigma_8` is
near `sigma_9` is the degenerate one the test below rejects. Hypotheses execute
in parallel with eight lanes per workgroup; the QR and the solve within one
hypothesis are serial.

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

The adaptive budget is evaluated on device, and the search stops where a
sequential RANSAC would stop. `selectPairModel` walks a batch's trials in their
order: a trial counts only while its index is below the budget the trials before
it set, and a trial that improves the best model shrinks the budget at once. A
batch may score more trials than the search uses; those are ignored. The result,
the trial count and the hypothesis count are therefore those of the sequential
search, the same for every batch size and on every device
(`tests/typeshade/pair-batch.test.mjs`). For small probabilities, a bounded
series evaluates `log(1-p)`: direct f32 subtraction can round `1-p` to 1,
incorrectly turn the required-trial calculation into negative infinity, and stop
prematurely. E, H and PnP share the stable calculation. A conservative extra
trial handles rounding around integer boundaries. Minimum attempts default to
`min(64, iterations)` for E and `min(32, iterations)` for H; the hard cap
includes rejected samples. The batch is a performance choice: `batchSize`
(default 512) is its ceiling, the device's `maxModelsPerBatch` and memory bound
it, and on WebGPU the time budget sizes each batch from the scoring rate the
device learned, from the occupancy floor of 256 trials upward; the CPU tier scores 64 at a time. The
confidence formula remains a RANSAC heuristic.

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

Pair verification uploads raw correspondences once. Each 64-lane workgroup
computes an inclusive valid-row prefix scan in shared memory. A device pass
scans block totals, followed by stable scatter into shared ideal/normalized
buffers. H/E sampling therefore sees the same ordered population as separate
undistortion and host compaction, even with an entirely invalid block. H support
is preserved before the E solver reuses consensus state. Pose recovery consumes
the resident ideal rows, then the final E mask is scattered to original row
indices. Point clouds, ideal rows, E/H matrices and intermediate counters
remain on device throughout the plan.

The complete verification call makes one compute submission and three final
snapshots: a 72-f64 summary, N-u32 source mask and 12-u32 state, totalling
`624 + 4*N` application readback bytes. The host enumerates the final source
mask to build the returned index list. These transfers also apply to nonempty
geometric rejections. This is one resident plan with multiple dependent kernels;
`execution.report.groups` records their actual dispatch boundaries.

Composing the three standalone APIs requires three compute submissions and
ten snapshots: `808 + 36*N + 36*K` bytes for N raw rows and K invertible rows.
Verification's warm-pool input upload is `32*N`, versus `32*N + 64*K` bytes for
that composition. Buffer initialization, uniforms and runtime staging have
separate costs; application counters do not represent total driver traffic.

E/H RANSAC and optional two-view recovery form one TypeShade plan and one
compute submission. No hypothesis, score, intermediate mask, pose, point or
convergence counter is read during the plan. E/H return three final snapshots:
`200 + 4*N` application bytes. Two-view recovery returns five snapshots:
`608 + 32*N` bytes. Each snapshot adds a staging-copy submission. The legacy
`scoreDispatches`/`scoreSubmissions` counters now cover the complete initialization
plan; `execution.report` provides the authoritative dispatch/submission groups.

The searches run as chunks under the time budget (`docs/gpu-sfm.md`, "The time
budget"). The GPU writes each batch's workgroup counts (TypeShade change 0075),
and the host reads one u32 scheduling word, the adaptive budget, after every
fourth batch of a search. Each read adds 4 bytes and a staging-copy submission
to the counts above. A search whose maximum fits in four batches reads none.

For N rows, B hypotheses/batch and Q=ceil(N/64), the consensus application-buffer
budget is `72*N + 252 + B*(984 + 16*Q)` bytes: the scheduling word and the twelve
words of workgroup counts are 52 of the fixed bytes, and each model's factor,
null vector and state 836 of its bytes. Recovery adds
`564*N + 964 + 352*ceil(N/32)`. Undistortion uses `68*N` bytes and reads `36*N`.
These counts exclude runtime uniforms/staging, compiler resources and pooled
buffers. Working/storage/dispatch limits are checked before execution. All
output slots and flags used by a call are initialized even when buffers are reused.

Verification adds `108*N + 16 + 8*Q` bytes to consensus plus recovery, for a total
working-buffer budget of
`744*N + 1232 + 8*Q + 352*ceil(N/32) + B*(984 + 16*Q)`.
Capacity remains based on raw N: compaction does not shrink host allocations or
the statically recorded dispatch schedule. No image-pair intermediates cross
the host boundary, but each pair still has its own buffers/plan and final reads.

The [geometry numeric contract](gpu-geometry.md#numeric-and-resource-contracts)
applies: GPU f64 is approximately 48 significand bits over the f32 exponent
range, not IEEE binary64. Trials are bounded to 1..100,000 and refinement to
0..100 iterations. Severe conditioning can still cause explicit numerical
rejection within the input bounds. Backend failure never selects a CPU retry.
The explicit CPU test tier executes the compiled lane operators with serial
folds in place of device workgroup barriers.

The mapper now executes all implemented numerical geometry through TypeShade:
pair inversion, E/H estimation, relative pose, PnP, triangulation, filtering and
BA. The mapper requests one `pairVerification` operation per pair, including
geometric rejections. Host work includes validation, final result indexing,
data packing, track ownership and mapper control. APIs still repack/read between
verification, PnP, triangulation and BA; a shared persistent
scene and multi-pair batching remain performance work. Disabled RANSAC/LM
dispatches are still scheduled up to requested caps. E remains an eight-point
solver; calibrated five-point, MAGSAC/stronger model selection, broader camera
and intrinsic models, and calibration priors remain algorithmic work. Optional
joint shared-focal refinement is implemented in
[bundle adjustment](gpu-geometry.md#optional-joint-shared-focal-refinement).

Tests use independent scalar image formation, direct CPU DLT, radial folds,
outliers, coordinate offsets/scales, exact degeneracies, weak baseline, pose
constraints and actual browser dispatch/map instrumentation. The software
adapter proves WGSL execution, not physical-GPU speedup. Real-video results
must retain every pair and failure; see the [experiment index](README.md#experiments).

The [resident-pair pyramid run](experiments/gpu/resident-pair-verification-software.json)
registers all eight views through this verification and the existing GPU mapper:
1,723 points, 5,122 observations and 0.45492px mean Euclidean residual. All 28
pair outcomes are retained, including one cheirality rejection. Its seed pair
is [4,5], versus [1,4] for the independent CPU run; the benchmark therefore also
records a proper-rotation camera-center similarity alignment. The CPU model
and GPU model hashes are identical to the
[previous initialization run](experiments/gpu/incremental-initialization-software.json).
All pair decisions/counts agree; median parallax differs by less than 1e-9°.

| Whole reconstruction, including failed geometry         | Previous stages | Resident verification |
| ------------------------------------------------------- | --------------: | --------------------: |
| Dispatches                                              |          24,806 |                24,918 |
| All submissions, including final copies                 |             552 |                   300 |
| Maps                                                    |             348 |                   152 |
| Application readback bytes                              |       1,264,148 |               852,628 |
| Application upload bytes, including cold initialization |      22,743,960 |            22,849,688 |
| Retained pool bytes                                     |      20,468,748 |            21,018,044 |

The run reduces application readback by 32.55%. Compaction adds dispatches and
working buffers; cold uploads/pool retention increase. Warm standalone probes
in the same record verify reduced input upload and zero new allocations on
reuse. For 321 invertible rows, reads fall from 23,920 to 1,908 bytes and uploads
from 30,816 to 10,272 bytes. The uncontrolled software sample takes about 96
seconds versus about 82 previously. These results establish numerical
preservation and fewer transfers, not wall-clock or physical-GPU speedup.
