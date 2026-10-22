# Reconstruction status and remaining work

Reviewed on 2026-10-22 at 11:38 UTC against
[`main` at `f660b52c`](https://github.com/typeshade/typecv/commit/f660b52c3a073059348d8b7aa65c7c85fc3def9a).
This is a dated implementation and evidence snapshot. Later PR merges and measurements
must update it before its pending items are treated as delivered. The
[project roadmap](roadmap.md) covers other computer vision areas; the
[SfM roadmap](sfm-roadmap.md) keeps the stage-level evidence gates.

## PR snapshot

| PR                                                                                                         | State at review       | Delivered work or remaining boundary                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [#161](https://github.com/typeshade/typecv/pull/161), [#167](https://github.com/typeshade/typecv/pull/167) | Merged                | Windowed/streamed Dense, observation-streamed BA, track-chunked triangulation and worker budgets from adapter limits. Remaining minimum resident arrays and other stages still refuse some limits. Issue [#160](https://github.com/typeshade/typecv/issues/160) remains open.                                                           |
| [#195](https://github.com/typeshade/typecv/pull/195)                                                       | Merged                | Zero-weight carve protection, grouped confidence-weighted visibility and verified local planar completion. Lab presets enable the new behavior; library defaults remain distinct. Synthetic checks do not establish real-video or native-GPU quality.                                                                                   |
| [#197](https://github.com/typeshade/typecv/pull/197)                                                       | Merged                | Independent Dense readbacks overlap, with exact output and failure-lifetime checks; guided-matching workgroup probes compare 64/128/256 lanes. Read waves decrease; transferred bytes and the production 64-lane matching default do not change. Native whole-run speedup is unmeasured.                                                |
| [#198](https://github.com/typeshade/typecv/pull/198)                                                       | Merged                | Corrects duplicate runtime/composite kernel accounting and adds the separate Dense device table to multi-video records. Adds offline source/IR/WGSL inspection and an audit of collected records. Establishes accounting defects and optimization candidates, not a compiler performance defect or a speedup.                           |
| [#199](https://github.com/typeshade/typecv/pull/199)                                                       | Open, head `bb13c166` | Implements 64-lane cooperative BA camera rows and a 448-byte workgroup relative-pose cache for Dense. CPU/software/short-list BA keeps its serial default. CI is running; browser failures are not all classified. Native speedups and the hardware selection threshold are unmeasured. Reference patch/weight caching is not included. |
| [#190](https://github.com/typeshade/typecv/pull/190)                                                       | Open draft            | Earlier BA camera-product parallelization overlaps #199. Compare any unique coverage and retire or narrow it after the replacement is accepted.                                                                                                                                                                                         |
| [#148](https://github.com/typeshade/typecv/pull/148)                                                       | Open draft            | Small-photo all-pairs matching and retrieval candidates above the window remain outside main. Finish integration, lifetime/budget and registration checks before delivery.                                                                                                                                                              |

The #199 validation record reports core, targeted CPU and software WebGPU numerical
regressions passing, including fixed/shared focal, streamed BA and Dense cached/uncached
array comparisons. Its broader browser record still has processed-stat and multi-video
selection/progress failures. A video-set Remove-button timeout was reproduced on unchanged
`6cdd70e`; the other UI failures have not been classified. A passing subset does not
establish that the whole PR is green. Check the current PR head and CI before merging.

## Measurement boundaries

The [#198 audit](gpu-throughput.md#run-record-accounting-and-actual-ir-audit-2026-10-22)
changes how historical evidence can be used:

- Runtime reports and their composite reports were both accumulated, inflating groups,
  dispatches and GPU time. Multiplicities differ across direct and composite plans;
  do not divide every historical total by one factor. Stage elapsed times remain useful.
- Multi-video records omitted the separate Dense device's kernel table. Its aggregate
  application-transfer zeros mean unavailable accounting, not an absence of transfers.
- Timestamp sums cover measured groups. Subtracting them from stage wall time does not
  isolate CPU time. Queue gaps, untimed work, compilation, transfers and overlapping work
  require a timeline and CPU profiling. Application byte counters do not measure physical
  bandwidth or every direct resident write.
- The sparse `timing.totalMs` and Dense `dense.totalMs` describe separate intervals.
  Report both when comparing complete reconstructions; do not present the sparse interval
  as the whole SfM/MVS/TSDF run.
- SwiftShader establishes software WebGPU behavior and comparison timings. It cannot
  select a native-GPU winner. #199's cooperative BA is slower there and is excluded
  automatically; its Dense software timing difference is small.

The ETH3D `pipes` Pixel/TSDF results in [#181](https://github.com/typeshade/typecv/pull/181)
and the depth diagnosis in [#188](https://github.com/typeshade/typecv/pull/188) predate #195.
Keep them as historical evidence. They are neither a quality measurement of current Lab
presets nor a general proof that every fusion error originates in depth. Likewise,
COLMAP pose agreement is a reference comparison, not independent ground-truth accuracy.

## Remaining work, in priority order

The priorities below order verification and implementation. They do not relax the
correctness and reproducibility rules in [design rules](design-rules.md).
GpuScene and BA kernel work can proceed independently; Dense performance and quality
work can use the same inputs with separate acceptance checks.

| ID  | Priority    | Remaining work                                       | Next reviewable result and acceptance evidence                                                                                                                                                                                                                                                                                                                                                                   |
| --- | ----------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R01 | P0          | Complete #199 verification                           | Resolve/classify the unclassified browser failures against main, record CI for the exact head and preserve numerical/streaming regressions. Merge only after the applicable checks pass. Reconcile overlapping #190 coverage.                                                                                                                                                                                    |
| R02 | P0          | Fresh native-GPU baseline                            | Use #198's corrected accounting and identify whether #199 is included. Run fixed inputs/configurations on RTX 2080/5060 Ti; record code/compiler/adapter identities, cold/warm behavior, sparse and Dense stage wall times, timestamp coverage, memory, registration and geometry quality. Measure #199's serial/cooperative selection threshold instead of treating 128 observations per camera as established. |
| R03 | P1          | Persistent common GpuScene                           | Integrate `sceneSteps` with a device scene for registration, continuation, indexed PnP gathering, triangulation, BA and filtering/compaction. Keep results resident; read summaries and requested snapshots. Compare against HostScene, including resumed/local runs, transfer growth and host memory.                                                                                                           |
| R04 | P1          | Complete budget compliance under #160                | Implement block-partitioned BA and split scene/index bindings below the resident floor; handle Dense's remaining minimum-image floor and budget-limited pair estimation, matching, features/ORB/global assignment and other stages. Force split paths with small budgets and compare outputs or explicitly stated final-cost tolerances. See [remaining limits](gpu-budget.md).                                  |
| R05 | P1          | Current Dense quality baseline and depth improvement | Re-run current #195 presets and library modes on fixed ETH3D/TUM inputs. Measure depth and surface accuracy/completeness, F-scores and weak-texture/floor coverage. Investigate slanted planes, far surfaces and low overlap; coarse-to-fine PatchMatch is a candidate, not a delivered improvement.                                                                                                             |
| R06 | P2          | Dense reference sample/weight reuse                  | After #199's relative-pose cache, prototype reference gray/bilateral-weight caching for every supported window size. Measure registers/shared memory and native stage time; compare depth/normal/cost and final geometry.                                                                                                                                                                                        |
| R07 | P2          | Reduced camera solver and Schur cache scaling        | Keep the current cache eligibility/work guards until bounded assembly is available. Compare a budget-based extension and tiled Cholesky on small camera systems with PCG; retain a suitable large-system route. Check residuals, accepted LM costs and final reprojection/registration.                                                                                                                          |
| R08 | P2          | Current TSDF fusion profiling and culling            | Replay fixed depth maps with the current grouped-visibility mode. Separate kernels, uploads/read waves and host work. If warranted, add conservative block/frustum culling with identical retained evidence and geometry, including near-plane and truncation boundaries.                                                                                                                                        |
| R09 | P2          | Descriptor reuse and preview readback                | Profile a bounded decoded/resident descriptor LRU across view-graph candidate batches; reserve its bytes and release before mapping. Reduce preview-only depth on the GPU while preserving edge and positive-depth-mean semantics. Keep final depth/geometry reads and outputs unchanged.                                                                                                                        |
| R10 | P2          | Photo pair selection                                 | Complete #148's all-pairs path for small sets and retrieval path above the window; compare registration, pair counts, order behavior, feature lifetimes, peak bytes and runtime.                                                                                                                                                                                                                                 |
| R11 | P3          | Robust sparse geometry                               | Add calibrated five-point Essential estimation and stronger degeneracy/model selection. Test all real minimal branches, planar/rotation/low-baseline cases, noise and outliers against independent references and held-out datasets.                                                                                                                                                                             |
| R12 | P3          | Remaining calibration                                | Implement camera-group focal BA on GPU, joint distortion optimization and broader camera models where supported. Existing pinhole/radial k1/k2 projection and CPU focal groups are implemented; verify new Jacobians, bounds, observability and calibration recovery.                                                                                                                                            |
| R13 | P3          | Long-term and multi-map reconstruction               | Add relocalization/revisit constraints, loop closure, scalable global correction and the stalled-growth Sim(3) submap fallback. Local/global BA already exists; submap alignment, component connection and drift recovery need their own failure and quality gates.                                                                                                                                              |
| R14 | Maintenance | Keep status and evidence aligned                     | Update these roadmaps when PRs merge, change scope or gain native/real-scene evidence. Record implementation, default/API scope and verification separately; maintain the [physical-GPU run list](physical-gpu-runs.md).                                                                                                                                                                                         |

## What is already implemented

Main contains GPU SIFT/ORB and optional pyramidal KLT; automatic/shared focal;
global descriptors and the Lab's canonical multi-video/view-graph/component pipeline;
GPU numerical SfM, time budgets and streamed BA/triangulation; PatchMatch MVS,
depth confidence, pixel and hashed TSDF fusion, grouped visibility, verified planar
completion, cleanup and meshing. Per-component Dense is also integrated in the Lab.

The public GPU operators are exported by [`src/gpu/index.ts`](../src/gpu/index.ts).
Multi-video orchestration uses [`src/gpu/internal.ts`](../src/gpu/internal.ts).
KLT/ORB and several quality options have opt-in or distinct Lab/library defaults.
Optional Dense graph-cut labeling uses a host min-cut solver; its existence is not
a claim of an all-GPU labeling path. These implementation facts do not establish
broad COLMAP-equivalent accuracy, registration success, throughput or scale.
