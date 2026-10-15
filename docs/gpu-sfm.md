# TypeShade GPU execution for SfM

This guide covers the SfM subset of TypeCV. The shared, project-wide GPU design
and other computer vision domains are described in the [architecture](architecture.md)
and [project roadmap](roadmap.md).

The target architecture keeps observations, features and optimization state on
the GPU across stages. This change establishes an executable device layer and
connects batched Essential scoring to two-view reconstruction. It does **not**
move the incremental mapper or bundle adjustment onto the GPU.

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

| Operation                                                                                    | Execution                                                      |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Packed pixel upload and normalization                                                        | GPU once per session; normalized observations remain resident  |
| Random sampling and f64 eight-point proposal solver                                          | CPU                                                            |
| All hypothesis × match Sampson residuals, flags, truncated MSAC costs                        | GPU; three calls fused into one dispatch                       |
| Partial reduction, per-model reduction, winning-model selection                              | GPU                                                            |
| Per-batch result                                                                             | Three floats: winner index, inlier count, normalized MSAC cost |
| Adaptive stopping and local eight-point refit                                                | CPU, using only selected model/mask reads                      |
| Pose decomposition, cheirality, triangulation, nonlinear refinement and final classification | CPU, shared with the reference core                            |
| Incremental registration, tracks and joint BA                                                | CPU; not yet connected to this asynchronous API                |

`reconstructTwoViewGpu` is an explicitly **hybrid** API. Its `execution.cpuStages`
lists these host stages. `estimateEssentialRansacGpu` exposes the preceding
consensus result, useful for independent solver or mapper work. CPU proposals
use original f64 normalized observations; device scores use f32. The final
refined result's inlier mask is classified by the shared f64 recovery path, so
it may differ from the preceding GPU consensus mask.

`batchSize` defaults to the safe session capacity, at most 64 by default.
Adaptive stopping occurs between batches and can evaluate extra hypotheses
within the last batch. Attempted samples, including degenerate samples, always
respect `iterations`. The confidence rule is the usual eight-point RANSAC
heuristic; it is not a guarantee on scene accuracy or degeneracy.

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

Matches are `[x1,y1,x2,y2,...]` pixel coordinates. These APIs require undistorted
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
execution/correctness and transfer structure, not hardware acceleration:

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

## Next GPU work, in order

1. **Persistent scene/feature buffers and matching.** Keep feature coordinates,
   descriptors, pair indices and track observations on the device. Implement
   tiled descriptor distance/top-2, ratio/mutual filtering and compact outputs;
   compare match identities with a scalar oracle. Avoid a full descriptor-distance
   matrix. Use TypeNN-style tiled compute and device gather for feature access.
2. **Geometry across pairs.** Batch E/H/PnP residuals with explicit degeneracy
   selection, keep masks for device consumers, and add asynchronous mapper
   integration. Five-point hypotheses and robust model selection need independent
   correctness tests before a device minimal solver is trusted. The current API
   still uploads each pair separately and does not batch across pairs.
3. **Triangulation and observation filtering.** Use packed tracks with offsets,
   GPU ray intersection/point refinement, cheirality and parallax checks, then
   compact valid observations. Retain f64 reference fixtures for weak baselines.
4. **GPU bundle adjustment operators.** Start with projection/residual/Jacobian
   parity and per-camera/point block accumulation. Build a matrix-free Schur
   operator, block preconditioner and PCG with device-side vector reductions.
   Keep accepted state resident; synchronize only convergence summaries. Test
   gauge handling, fixed-boundary local BA, LM rejection/rollback, disconnected
   components and calibration Jacobians. WebGPU has no portable f64: scaling,
   summation error, condition estimates and guarded refinement matter. f16 is
   inappropriate for these normal equations without numerical evidence.
5. **Kernel specialization after profiling.** Replace H × N flags/cost traffic
   where useful with workgroup-memory partial reductions, tiled hypothesis/data
   reuse and device compaction. TypeShade already exposes workgroup variables,
   barriers and compute entries; use those facilities and compare reports with
   actual device calls. Add storage-limit sharding and adapter-specific benchmark
   gates before increasing scene sizes. Generic tensor expressions alone will
   not produce an efficient sparse SfM solver.
6. **Dense MVS.** Device image pyramids, PatchMatch/depth consistency, fusion and
   mesh extraction follow a stable calibrated sparse solution. Evaluate depth
   accuracy/completeness with independent ground truth.

The CPU reference is a numerical oracle and a clear hybrid boundary while these
operators land. Each GPU stage needs its own accuracy, failure, memory and
transfer-inclusive performance evidence before replacing the corresponding
reference path.
