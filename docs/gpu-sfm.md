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
TypeShade commit `212d26d2fb9fb6dd39b72674e537653fb0f7e888` (the pinned build dependency).
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
triggers CPU retry. `batchSize` defaults to 32, bounded by device capacity;
rejected samples count against the hard cap. E still uses eight-point fitting,
not calibrated five-point or MAGSAC++.

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
budget (including an optional mask). The normalized resident observations and
bounded free pool are separate from `workingBytes`; runtime uniforms/staging
also exist. Oversized observations are rejected; observation sharding is not
implemented. `statistics` counts application arrays, not total driver memory
or all `writeBuffer` traffic. Pooled buffers are included in `liveBuffers`.

Work is serialized per device. Input pixels, intrinsics, models and relevant
options are captured before queued execution. Session disposal rejects new
scores and drains pending work; device disposal drains its queue and releases
all owned arrays. Failed scoring buffers are destroyed instead of pooled.
One TypeCV device may be active per JS context because TypeShade configuration
is shared. Other TypeShade consumers must coordinate configuration changes;
borrowing the same GPUDevice alone does not isolate global runtime state.

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
   joint LM and gauge/rollback now run on GPU with analytic shader Jacobians.
   Extend beyond bounded scenes, reduce disabled dispatch overhead and profile
   vector reductions. Evaluate generated derivatives for supported camera models;
   the existing AD probe does not establish automatic differentiation of the
   whole emulated-f64 optimizer. Shared focal/distortion optimization, calibration
   priors and physical-adapter precision/flavor validation remain open. f16 needs
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
