# TypeShade GPU execution for SfM

This guide covers the SfM subset of TypeCV. The shared, project-wide GPU design
and other computer vision domains are described in the [architecture](architecture.md)
and [project roadmap](roadmap.md).

The [TypeShade capability review](typeshade-capabilities.md) corrects the earlier
precision rationale and records direct GPU probes, including features beyond the
current integration. The independent CPU reference is a verification tier, not a
requirement for executing the implemented numerical geometry.

The target architecture keeps observations, features and optimization state on
the GPU across stages. The device layer now provides GPU P3P/PnP RANSAC,
robust track triangulation, observation filtering and joint Schur/PCG bundle
adjustment, connected through `reconstructIncrementalGpu`. Their reusable APIs,
precision/resource contracts and evidence are in [GPU geometric vision](gpu-geometry.md).
GPU E/H and relative-pose initialization are described in the
[initialization guide](gpu-initialization.md). Pair compaction and H/E/pose checks
now share one resident plan. Mapper metadata, final result indexing and packing
stay on the host; geometry is still repacked across later API stages.

## What TypeNN contributes

Inspected TypeNN commit `61259c3733148a8665466163ae1c5c2b920b129c` and
TypeShade commit `212d26d2fb9fb6dd39b72674e537653fb0f7e888` (the pinned build dependency
then; the pin is `d70b5398d6df31cdfbd1ac1de13fdb8b81c4ed93` now, TypeShade changes 0071,
0073, 0074 and 0075).
The relevant patterns are concrete runtime behavior, independent of neural nets:

| Source                                                                                                                            | Pattern applied here                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [TypeNN execution](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/runtime/execute.ts)      | Resident intermediates, synchronous `fuse` recording, no intermediate host reads, failed buffers discarded                         |
| [TypeNN device](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/runtime/device.ts)          | One runtime owner, queued work, bounded buffer reuse, borrowed-device ownership and execution reports                              |
| [TypeNN datasets](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/data/device-dataset.ts)   | Persistent source data and storage-limit awareness; TypeCV currently bounds batches rather than sharding observations              |
| [TypeNN reduction](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/src/kernels/reduce.shade.ts) | Parallel output lanes with bounded inner loops, used for hierarchical partial sums                                                 |
| [TypeNN build](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/scripts/build.mjs)               | Compile `.shade.ts` with the real Vite plugin; emit declarations while generated views exist; keep the compiler out of the browser |
| [TypeNN browser tests](https://github.com/typeshade/typenn/blob/61259c3733148a8665466163ae1c5c2b920b129c/tests/gpu/_browser.mjs)  | Real Chromium WebGPU execution and observable device-call counts                                                                   |

`ShadeDevice.open()` requires WebGPU by default. An explicit CPU tier is available
for differential tests. The CPU geometry entry remains independent of TypeShade.
Importing `typecv/gpu` loads the bundled runtime, without requiring a TypeShade
compiler in the consuming application. Generated declarations expose local
report types, not third-party build-time types.

## Current execution boundary

| Operation                                                                                 | Execution                                                |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Radial pair inversion, ideal-pixel generation and stable compaction                       | GPU emulated f64; 64-lane prefix scan and device scatter |
| E/H sampling, QR/direct SVD proposals and consensus refits                                | GPU emulated f64                                         |
| E/H hypothesis scoring, cooperative partial sums and winner selection                     | GPU emulated f64, 64-lane reductions                     |
| Adaptive trial budget and refit acceptance                                                | GPU; stable small-probability calculation                |
| Relative-pose decomposition, cheirality, triangulation, joint LM and final classification | GPU emulated f64                                         |
| P3P/PnP, robust multi-view triangulation, filtering and joint BA                          | GPU emulated f64                                         |
| Independent `EpipolarSession` scoring utility                                             | GPU f32; caller supplies models                          |
| Validation, data packing, final result indexing, track graph and mapper control           | Host                                                     |

`verifyImagePairGpu`, `reconstructTwoViewGpu` and `estimateEssentialRansacGpu` have empty
`execution.cpuStages`: all numerical initialization executes in TypeShade.
One verification plan includes inversion/compaction, H/E proposals, adaptive
stopping, refits and pose recovery, with only source indices and final summaries
read. The [initialization guide](gpu-initialization.md)
specifies the numeric, transfer and resource contracts. Device failure never
triggers CPU retry. `batchSize` (default 512) is the batch's ceiling, and the
search stops where a sequential RANSAC would, whatever the batch; rejected
samples count against the hard cap. E still uses eight-point fitting,
not calibrated five-point or MAGSAC++.

Joint shared-focal BA also has an opt-in `schurCache` route, documented with its
exact constraints and residual-certificate behavior in
[GPU geometric vision](gpu-geometry.md#opt-in-cached-schur-assembly). It is
available only when the shared-focal GPU route is selected and its camera,
observation-work and submission guards pass. CPU, fixed-focal, explicitly
bounded and out-of-guard problems keep the existing solver route. This option
does not change the emulated-f64 precision or the existing convergence and LM
acceptance criteria. In one RTX 2080 native comparison with `localBundleSubset`
enabled in both settings, `schurCache` improved mean warm whole-pipeline time
from 5,635.7 ms to 5,096.0 ms (1.106×); the final eight-iteration shared-focal
global BA stage changed from 1,011.325 ms to 420.5 ms (2.405×). The small sample
and scene-specific quality evidence are documented in
[GPU geometric vision](gpu-geometry.md#restricted-native-comparison) and the
[`sfm-local-schur-cache-rtx2080.json`](experiments/gpu/sfm-local-schur-cache-rtx2080.json)
report. This does not establish general speedup, COLMAP parity, or high-resolution
support. `schurCache` and `localBundleSubset` both default to `false` and must be
enabled explicitly.

A separate local-subset-only comparison, with `schurCache: false` in both
settings, measured four warm whole-pipeline samples at 5,776.35 ms without
`localBundleSubset` and 5,635.70 ms with it (1.025×, about 2.44% less wall time).
All five paired reconstructions matched exactly across 14,186 numeric leaves;
BA observation work changed from 16,129 to 14,378. This small native scene
sample is separate from the cache A/B above, and the ratios are not combined.
Details are in the [GPU geometry comparison](gpu-geometry.md#restricted-native-comparison)
and the linked
[`sfm-local-schur-cache-rtx2080.json`](experiments/gpu/sfm-local-schur-cache-rtx2080.json)
report.

Set `localBundleSubset: true` on `GpuIncrementalOptions` to restrict eligible
local BA passes to the selected camera-window landmarks and their retained
observations; see the [incremental mapper contract](incremental-sfm.md). Each
bundle entry in `GpuIncrementalResult.execution.stages` reports a `problem`
summary: input view count, optional variable view IDs, landmark count, and
observation count. These describe the actual BA input, including fixed boundary
views and observations before any later mapper filtering. A local subset entry
therefore reports its subset; a final global entry reports the full input.

The independent f32 scoring session below remains available for external model
batches. Its four-dispatch/three-float summary contract is separate from the
resident f64 RANSAC implementation used by current reconstruction.

## Resident P3P API

```ts
import { ShadeDevice, solveP3pBatchGpu, GpuP3pStatus } from 'typecv/gpu';

const device = await ShadeDevice.open();
try {
  // 18 doubles/problem: P0.xyz, P1.xyz, P2.xyz, bearing0.xyz, bearing1.xyz, bearing2.xyz.
  const batch = await solveP3pBatchGpu(device, packedProblems);
  try {
    // Solving makes one dispatch and reads no candidates back. This call explicitly downloads them.
    const { poses, counts, valid, status } = await batch.read();
    if (status[0] === GpuP3pStatus.success) {
      // Four slots/problem. Each slot contains row-major R[9] followed by t[3].
      console.log(counts[0], valid[0], poses.slice(0, 12)); // Xcamera = R * Xworld + t.
    }
  } finally {
    await batch.dispose();
  }
} finally {
  await device.dispose();
}
```

The shader constructs Grunert's quartic, isolates its real roots using a bounded
derivative hierarchy, refines positive ray distances, constructs proper rotations,
checks distance consistency and positive depth, and merges duplicate poses.
Scaled polynomial evaluation preserves signs at large root bounds without
overflowing f64's f32 exponent range. No CPU roots or poses are uploaded.
Each invocation solves one problem; its bounded inner iterations are serial.
The explicit compute entry has 32 lanes/workgroup. This is parallelism across
problems, without a claim of cooperative or optimally tuned root solving.

Inputs must be a nonempty Float64Array with finite scalar magnitudes at most
1e9; nonzero values must be at least 1e-30. Bearings are normalized on device,
using component scaling before their norm. World-edge lengths below 1e-12,
relative triangle altitude below 1e-8, ray-cone diameter below 1e-3, and smallest
world edge below 1e-6 of the largest are rejected. Root/distance ratios beyond
1e9 are excluded. These are numerical domain guards, not universal P3P limits.
Normalize scene units before approaching those bounds. A successful status
means validated candidates were found, not a proof of completeness for every
ill-conditioned problem in that numeric domain.

Status codes identify success (0), degenerate world (1), zero/narrow bearings (2),
degenerate/no-real-root polynomial (3), no validated positive-depth pose (4),
unresolved numerical branch splitting (5), and extreme world-edge ratios (6).
Rejected problems have zero counts/valid flags; unused pose slots are cleared
even when buffers are reused. Solving allocates 552 application bytes/problem,
including the input and four outputs. Default f64 output allocation uploads
its initial zero array once (TypeShade's zero-allocation shortcut has no f64
element type); pooled output reuse avoids that upload. Full explicit readback
is 408 bytes/problem. Runtime/staging memory is separate from these figures.

Candidate buffers stay resident until disposal. The public standalone handle
exposes explicit readback. `estimatePosePnPGpu` reuses this compute entry inside
its own resident sampling/scoring/selection/refinement plan, and the incremental
driver consumes that API. A public arbitrary resident-candidate composition
interface remains separate work.

Both shared-IR CPU and actual WGSL tests cover 111 problems: 100 seeded known-pose
scenes at world scales 0.01/1/100, a symmetric repeated-root case, bearing rescaling,
translated origin, degeneracies, negative depth and the side-ratio guard. Two
fixtures have four branches. All well-conditioned reference branches match with
a pose tolerance of 1e-4 (rotation Frobenius error or translation error divided
by maximum world-edge length); the same fixtures also require positive depth,
proper orthonormal rotations, bearing residual below 1e-7 and bounded truth error.
Input limits, queued disposal, failure cleanup and typed pool reuse are tested.

```sh
npm run build
npm run evaluate:p3p
# Optional real observations from the existing pyramid reconstruction:
node scripts/evaluate-p3p.mjs data/pyramid/typecv/model.json > data/p3p-pyramid.json
```

The [recorded pyramid run](experiments/gpu/p3p-pyramid-software.json) keeps all
256 outcomes (32 seeded triplets/view across eight registered views), model hash,
status codes, branch counts, branch differences and actual device call counts.
All 256 agree with the independent CPU solver. Inputs are reconstructed landmarks
and measured image observations under assumed calibration; this is reference
agreement and geometric consistency, not ground-truth pose accuracy. The run
uses SwiftShader and establishes no physical-GPU speedup or full GPU SfM result.

## Resident scoring API

```ts
import { ShadeDevice, EpipolarSession } from 'typecv/gpu';

const device = await ShadeDevice.open({
  // gpuDevice: existingDevice, // Borrow without destroying it on dispose.
  workingBytes: 64 * 1024 ** 2,
  poolBytes: 32 * 1024 ** 2,
  maxModelsPerBatch: 64,
});
try {
  const session = await EpipolarSession.create(device, pixels, K1, K2);
  try {
    // E is 9 * H row-major values, H <= session.capacity; x2^T E x1 = 0.
    const best = await session.score(E, 0.0015);
    const withMask = await session.score(E, 0.0015, { readMask: true });
    console.log(best.bestIndex, best.inlierCount, best.msacCost, withMask.mask);
  } finally {
    await session.dispose();
  }
} finally {
  await device.dispose();
}
```

Matches are `[x1,y1,x2,y2,...]` pixel coordinates. `EpipolarSession` requires undistorted
pinhole pixels; nonzero radial coefficients are rejected. Finite f32 focal
lengths must be positive normal values, normalized coordinates must remain
within ±10,000 in both host and f32 arithmetic, and the squared threshold must
be a positive normal f32 value. Unit-normalizing each E makes the Sampson
denominator guard independent of E's scale. A zero E is an all-outlier model.
Ties choose larger consensus, then smaller MSAC cost, then the first candidate.
This is thresholded RANSAC/MSAC, not MAGSAC++ or a five-point solver.

Each plan uses a scratch residual array; compilation fuses residual/flag/cost
calls and keeps the residual as a local value. Flags and costs still occupy
two H × N f32 buffers. Blocks of 256 observations are summed in parallel output
lanes, then reduced per model. Inner block loops are serial within an invocation;
this is not yet a specialized shared-memory/subgroup reduction.

Capacity checks account for storage binding sizes, one-dimensional dispatch
limits for the compiled 64-lane kernels, and the per-batch application buffer
budget (including an optional mask). `workingBytes` bounds the active arrays,
the released ones aside. The released arrays are the TypeShade device's pool
(`poolBytes`, TypeShade change 0071): a released array comes back for the next
array of the same element and length, and the oldest released array leaves
first when the pool is full. So the arrays are at most `workingBytes + poolBytes`.
Runtime uniforms/staging also exist. Oversized observations are rejected;
observation sharding is not implemented. `statistics` counts application
arrays, not total driver memory or all `writeBuffer` traffic. Pooled buffers
are included in `liveBuffers`.

Work is serialized per device. Input pixels, intrinsics, models and relevant
options are captured before queued execution. Session disposal rejects new
scores and drains pending work; device disposal drains its queue and releases
all owned arrays. Failed scoring buffers are destroyed instead of pooled.
Each TypeCV device is a TypeShade device owner of its own (change 0071), so two
devices in one JS context, TypeCV's or another consumer's, are independent.

## Verification and measured scope

```sh
npm ci
npm run check
npx playwright install --only-shell chromium
npm run test:webgpu
npm run benchmark:gpu > data/gpu-software.json
# Optional: use the previously prepared pyramid sequence, retaining all pairs/failures.
node scripts/benchmark-gpu.mjs data/pyramid/sequence/sequence.json > data/gpu-pyramid.json
# On a machine with a physical WebGPU adapter and suitable Chromium/driver:
TYPECV_HARDWARE_GPU=1 npm run benchmark:gpu
```

Run `npm run build` after changing GPU source before using the test/benchmark
commands. CI has a dedicated Chromium/SwiftShader job and fails when WebGPU is
unavailable; it does not skip the test or fall back to CPU. Tests compare against
an independent scalar f64 scorer, exercise block tails and invalid inputs,
check borrowed-device ownership, pending disposal, reuse and injected readback
failure, and reconstruct a known noisy scene. Browser instrumentation counts
actual `dispatchWorkgroups`, queue submissions and map calls independently of
TypeShade's reports. The build rejects compiler modules in browser chunks.

The recorded environment has **no physical GPU**. Chromium 141 runs real WGSL
through Google's **SwiftShader software WebGPU**. These results demonstrate
execution/correctness and transfer structure, not hardware acceleration. The
following scoring and pyramid figures describe the earlier recorded hybrid run
linked below; they are not measurements of the new initialization kernels:

- 8,193 matches × 64 hypotheses: four scoring/reduction dispatches, one compute
  submission, 2,304 uploaded application bytes and 12 downloaded bytes per batch;
  no new application buffer allocations after warmup. Each read adds a staging
  copy submission; total steady-state submissions including the summary read
  are two. Pipeline compilation is excluded from steady-state samples.
- In a 321-match scene with noise and 20% injected mismatches, hybrid recovery
  accepts 257 points, has rotation error approximately 0.0106°, translation
  direction cosine 0.99999966, and mean per-view reprojection error 0.02944px.
- On all 28 pairs from the [pyramid experiment](experiments/incremental-pyramid/README.md),
  both CPU and hybrid paths accept 26 pairs and reject pairs 4–6 and 4–7 during
  cheirality. Accepted pairs have identical inlier and valid-point **counts** in
  this run. This does not establish bitwise mask equality, calibrated accuracy,
  multi-view GPU parity, or COLMAP equivalence. The focal length is assumed and
  feature extraction/matching still use OpenCV.

The [recorded run](experiments/gpu/pyramid-software.json) retains every outcome,
adapter identity and software elapsed sample. f32 classification near a threshold
can differ across adapters and from f64; tests deliberately avoid asserting
bitwise parity at that boundary. Hardware testing must measure initialization,
warmup, uploads, readbacks, total stage latency and scaling separately. No speedup
claim is made from software timings.

The current [resident-verification-through-BA run](experiments/gpu/resident-pair-verification-software.json)
registers 8/8 views with 1,723 points and 5,122 observations at 0.45492px mean
Euclidean error. All numerical geometry uses TypeShade. Its different seed pair
is accounted for by a recorded camera-center similarity alignment; see the
[initialization guide](gpu-initialization.md) for scope and transfer counts.
It preserves both previous model hashes and cuts whole-reconstruction maps from
348 to 152; additional scan buffers/dispatches and cold uploads are recorded too.

## Next GPU work, in order

1. **Persistent scene/feature buffers and matching.** Keep feature coordinates,
   descriptors, pair indices and track observations on the device. Implement
   tiled descriptor distance/top-2, ratio/mutual filtering and compact outputs;
   compare match identities with a scalar oracle. Avoid a full descriptor-distance
   matrix. Use TypeShade compute entries for tiling and TypeNN's resident/gather
   patterns for feature access. The inspected TypeNN matmul kernel itself uses
   an output-parallel loop with a serial inner product, not workgroup tiling.
2. **Batched initialization and stronger estimators.** Radial inversion, stable
   compaction, H/E proposals/refits and relative-pose checks now share one
   TypeShade plan per pair. Batch across pairs instead
   of repeatedly uploading each pair, and retain masks for device consumers.
   Add calibrated five-point branch enumeration and stronger model selection,
   with independent solver and degeneracy checks.
3. **Persistent track state and compaction.** Robust GPU DLT, point refinement,
   cheirality/parallax and filtering now exist. Keep their outputs in a shared
   scene across registration/BA, compact accepted observations on device, and
   add large-track/storage sharding. Preserve weak-baseline regression fixtures.
4. **BA scaling and derivatives.** Matrix-free Schur, block preconditioning, PCG,
   joint LM and gauge/rollback run on GPU with analytic shader Jacobians. An
   opt-in cached shared-focal Schur route is available for guarded small problems;
   profile it before drawing performance conclusions and extend beyond its
   current structural limits. The fixed-focal PCG loop runs as chunks of the
   time budget ("The BA's PCG loop" below); the shared-focal route's persistent
   PCG kernels are not under the budget yet. Reduce disabled dispatch overhead
   and profile vector reductions. Evaluate generated derivatives for supported camera models;
   the existing AD probe does not establish automatic differentiation of the
   whole emulated-f64 optimizer. More general intrinsic/distortion optimization,
   calibration priors and physical-adapter precision/flavor validation remain open. f16 needs
   separate numerical evidence for these normal equations.
5. **Kernel specialization after profiling.** The f64 consensus path already
   uses cooperative workgroup-memory partial reductions. The separate f32
   scoring session still has H × N flags/cost traffic. Profile tiled hypothesis/data
   reuse and device compaction before further specialization. TypeShade already exposes workgroup variables,
   barriers and compute entries; use those facilities and compare reports with
   actual device calls. Add storage-limit sharding and adapter-specific benchmark
   gates before increasing scene sizes. Generic tensor expressions alone will
   not produce an efficient sparse SfM solver.
6. **Dense MVS.** Device image pyramids, PatchMatch/depth consistency, fusion and
   mesh extraction follow a stable calibrated sparse solution. Evaluate depth
   accuracy/completeness with independent ground truth.

The CPU reference remains an independent numerical oracle; host graph, packing
and control boundaries stay explicit. Each GPU stage needs its own accuracy, failure, memory and
transfer-inclusive performance evidence before replacing the corresponding
reference path.

## The time budget: pair model generation and the BA's PCG loop run as chunks

A GPU watchdog resets a device whose submission runs too long: Windows loses the
device after about two seconds (`DXGI_ERROR_DEVICE_HUNG`, an RTX 5070 Ti on the
pyramid sample), and iOS Safari loses it with no reason given (the 2026-10-19 lab
records, `docs/lab-run-records.md`). The protection is structural, not a mode:
every long kernel runs as chunks, and no chunk is submitted larger than the time
budget allows (`docs/gpu-resident-scene.md`, step 2).

The budget is the TypeShade device's (TypeShade change 0074, pin `41a140a`):
`ShadeDevice.open({ timeBudgetMs })` passes it to `createDevice()`, 250 ms by
default. A step of `planSteps` that names a chunk key and its units is a plan of
its own, `device.fuse(body, { chunk: { key, units } })`: the TypeShade device
fences it on the queue before and after (`GPUQueue.onSubmittedWorkDone()`;
awaiting `fuse()` alone only submits), times it by the wall clock, and keeps, per
key, the time one unit took (the mean of the last two chunks), the chunks it ran,
the longest one and the last chunk's units (`device.chunkRates`).
`chunkSize(key, max, min)` is the TypeShade device's `chunkSize`: as many units as
fit eight tenths of the budget, at most four times the last chunk, at least `min`
and at most `max`. The first chunk of a key is `min`: the rate is learned from
it. A `ShadePlanReport` carries the rates of the keys the plan ran under
`chunks`. The arithmetic was this repository's `fitChunk` and `learnRate` (pull
request #49) until the compiler took it (`compiler-changes.md`, 0074).

Pair model generation is a state machine of short dispatches. The RANSAC models
of a batch keep their triangular factor, their null vector and their state in
storage buffers, and the machine advances by one dispatch at a time:
`preparePairModels` samples the trials, `appendPairModelRows` appends at most 16
observations per dispatch, `solvePairModelNull` solves each factor for its null
vector at a fixed cost (`docs/gpu-initialization.md`, "Numerical path": no kernel
iterates to convergence), and `finishPairModels` assembles the models. Numerical
models, factorization and degeneracy tests stay on the GPU. The former
one-dispatch `generatePairModels` and the 160-sweep Jacobi SVD it ran per model
are removed: the longest dispatch of the generation is now a row append or a
batch's scoring, each bounded by `maxModelsPerBatch`.

The GPU decides the work of each batch (TypeShade change 0075, pin `d70b539`).
A `Uint32Array` resident, `counts`, holds twelve words: one `[x, y, z]`
workgroup count each for the batch's prepare, for its row appends, null solve
and finish, for `scorePairModels` and for `scorePairModelsOracle`.
`planPairModelBatch` writes the prepare's count after the selection of the batch
before. `summarizePairModelWork` writes the other counts after the prepare. A
count is 0 when the batch's first trial is at or past the adaptive budget, and
the model counts are 0 when no model is active. Those entries dispatch with
`{ indirect: counts, offset }`. No entry that dispatches with the counts binds
them: WebGPU refuses one buffer as the indirect buffer and a writable binding of
one dispatch. A batch whose counts are 0 changes nothing, and `selectPairModel`
returns at once for it. The first batch of a search always runs, as it did when
the host decided. A refit prepares and scores its one model with direct counts:
only its row appends, null solve and finish are indirect.

So the host reads nothing between the kernels of a batch, and between batches
only the scheduling word. `snapshotPairModelBudget` copies the adaptive budget
into it, and the host reads it after every fourth batch of a search
(`PROGRESS_INTERVAL` in `src/gpu/pair-estimation.ts`). The host bounds the loop
by the requested maximum, as before, and by the budget it read last. At most
three batches past the budget are submitted: each runs its three one-workgroup
control kernels (`planPairModelBatch`, `summarizePairModelWork`,
`selectPairModel`) and no other workgroup. A search whose maximum fits in four
batches reads no word. Before change 0075 the host read the word after each
summary and after each batch, up to two reads per batch. The CPU tier reads the
counts from the host copy when each call runs, so it runs what WebGPU
dispatches, and its results are those of the loop that read the word after each
batch, bit for bit (`tests/typeshade/pair-batch.test.mjs`).

The scoring of a batch is a plan of its own (key `scorePairModels`). It is
timed, for the rate of a trial, only when the host knows that it runs: the first
batch of a search, a refit, and the first batch after a read that showed the
search goes on. The scoring of another batch is a chunk step with
`timed: false`: a plan of its own, bounded as a chunk, but not timed. The GPU
may have set its count to 0, and a plan that dispatches nothing would teach the
device a rate far too fast (the open item "A chunk of change 0074" of the
compiler's proposal 0075).

The search stops where a sequential RANSAC would (`selectPairModel` walks the
batch's trials in order and ignores those past the stop), so a batch is a
performance choice alone: on WebGPU the time budget sizes it from the scoring
rate the device learned (key `scorePairModels`), from 32 trials up to
`batchSize` and `maxModelsPerBatch` (1024 by default). What depends on the
device's speed is the batch, the count of submissions, the batches submitted
past the budget and the scheduling-word reads, which the reports carry and no
result does.

The CPU tier runs the whole work of a key as one chunk (`chunkSize` returns
`max` there, TypeShade 0074's D7), so that the CPU oracle's results and its
counts are those of the state machine run once, bit for bit.

### The BA's PCG loop

The fixed-focal bundle adjustment (`bundleAdjustGpu` without
`refineSharedFocal`, `src/gpu/bundle.ts`) runs the preconditioned
conjugate-gradient loop of each LM iteration as chunks (key `bundleCg`). Its
former kernel `bundlePersistentCg` ran the whole PCG loop of an LM iteration in
one workgroup, up to `maxCgIterations` iterations over every point and camera
row, in one dispatch. That kernel is removed. The owner's iPhone lost the
WebGPU device in the final bundle adjustment of lab record
`20261019225221735-85ed0ae6` (site `7eef646`, unbounded submissions,
`refineSharedFocal: true`), after pair verification had passed. That final
bundle ran the shared-focal route, whose own persistent kernel is not chunked
yet (below).

Each PCG iteration is three short dispatches, `bundleSchurPoint`,
`bundleSchurCamera` and `advanceBundleCg`, as before. A chunk is `n` iterations
of one LM iteration in one plan, which the TypeShade device fences and times.
The chunk ends with `snapshotBundleActivity`, and the host reads its two flags
(8 bytes): the PCG active flag and the LM active flag. The host reads them once
after `initializeBundleCg` too. The loop stops when the PCG converged or failed,
and the LM loop stops when LM is done, on every device. Before this change only
`solverCheckpoints` read the flags.

The units of the key are the rows of one PCG iteration, `M + 6C` (one per
observation and one per camera parameter), so that the rate a device learns of
a small local bundle sizes the chunks of the larger final bundle. A chunk holds
`n = floor(chunkSize('bundleCg', remaining × (M + 6C), M + 6C) / (M + 6C))`
iterations, at least one: the minimum wins over what fits the budget. The
dispatches are those of the per-iteration path, in the same order. An iteration
past convergence is a guarded no-op, so a skipped one changes no result. The
CPU oracle's results are bit for bit those of the per-iteration path, with one
chunk of the whole loop and with chunks of one iteration
(`tests/typeshade/bundle-checkpoint.test.mjs`). The BA report carries the key's
rate under `chunks.bundleCg`.

With `boundedSubmissions` the BA keeps one step per dispatch, so that no plan
holds more than `maxDispatchesPerSubmission` dispatches, and reads the flags
every four PCG iterations with `solverCheckpoints`. The shared-focal route
(`src/gpu/joint-focal.ts`) still has its own persistent PCG kernels
(`bundlePersistentCg`, `bundleCachedPersistentCg` of
`src/gpu/joint-focal.shade.ts`) and is not under the budget yet. It is the
route of the lab's final bundle when `refineSharedFocal` is set, and the next
change.

### Bounded submissions

The lab also opens the device with `boundedSubmissions: true` and
`maxModelsPerBatch: 64` on a request (`?bounded=1`). Pair verification, PnP and BA
then run at most `maxDispatchesPerSubmission` TypeShade compute entries per
submission and await `GPUQueue.onSubmittedWorkDone()` before issuing the next.

`onProgress` on the device reports kernel start/completion; the incremental
mapper also reports operation start/completion. The worker includes the last
operation and kernel with device-loss errors. Its two-minute inactivity timer
resets on progress instead of limiting the whole reconstruction to two minutes.
Terminating a worker does not guarantee that already submitted GPU work stops.

Bounded execution trades additional submissions and fences for shorter command
streams of the kernels the time budget does not cover yet (triangulation, PnP,
the shared-focal BA). Reports include those submissions and the control readbacks. Neither path
imposes a portable wall-time bound on a single shader/driver or guarantees
recovery from device loss. CPU differential and SwiftShader WebGPU tests cover
numerical equivalence, chunk isolation, fences and injected native-device loss.
Confirmation on the affected physical RTX 5070 Ti remains necessary.
