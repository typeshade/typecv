# Reconstruction status and remaining work

Reviewed on 2026-10-22 at 11:43 UTC against
[`main` at `3d33c257`](https://github.com/typeshade/typecv/commit/3d33c25789c74250179210f6bcf38d9f2e761693).
This is a dated implementation and evidence snapshot. Later PR merges and measurements
must update it before its pending items are treated as delivered. The
[project roadmap](roadmap.md) covers other computer vision areas; the
[SfM roadmap](sfm-roadmap.md) keeps the stage-level evidence gates.

## PR snapshot

| PR                                                                                                         | State at review      | Delivered work or remaining boundary                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [#161](https://github.com/typeshade/typecv/pull/161), [#167](https://github.com/typeshade/typecv/pull/167) | Merged               | Windowed/streamed Dense, observation-streamed BA, track-chunked triangulation and worker budgets from adapter limits. Remaining minimum resident arrays and other stages still refuse some limits. Issue [#160](https://github.com/typeshade/typecv/issues/160) remains open.                                                                                           |
| [#195](https://github.com/typeshade/typecv/pull/195)                                                       | Merged               | Zero-weight carve protection, grouped confidence-weighted visibility and verified local planar completion. Lab presets enable the new behavior; library defaults remain distinct. Synthetic checks do not establish real-video or native-GPU quality.                                                                                                                   |
| [#197](https://github.com/typeshade/typecv/pull/197)                                                       | Merged               | Independent Dense readbacks overlap, with exact output and failure-lifetime checks; guided-matching workgroup probes compare 64/128/256 lanes. Read waves decrease; transferred bytes and the production 64-lane matching default do not change. Native whole-run speedup is unmeasured.                                                                                |
| [#198](https://github.com/typeshade/typecv/pull/198)                                                       | Merged               | Corrects duplicate runtime/composite kernel accounting and adds the separate Dense device table to multi-video records. Adds offline source/IR/WGSL inspection and an audit of collected records. Establishes accounting defects and optimization candidates, not a compiler performance defect or a speedup.                                                           |
| [#199](https://github.com/typeshade/typecv/pull/199)                                                       | Merged as `3d33c257` | Implements 64-lane cooperative BA camera rows and a 448-byte workgroup relative-pose cache for Dense. CPU/software/short-list BA keeps its serial default. Head `bb13c166` CI is still running at review; browser failures are not all classified. Native speedups and the hardware selection threshold are unmeasured. Reference patch/weight caching is not included. |
| [#190](https://github.com/typeshade/typecv/pull/190)                                                       | Open draft           | Earlier BA camera-product parallelization overlaps #199. Compare any unique coverage and retire or narrow it now that #199 is merged.                                                                                                                                                                                                                                   |
| [#148](https://github.com/typeshade/typecv/pull/148)                                                       | Open draft           | Small-photo all-pairs matching and retrieval candidates above the window remain outside main. Finish integration, lifetime/budget and registration checks before delivery.                                                                                                                                                                                              |

The #199 validation record reports core, targeted CPU and software WebGPU numerical
regressions passing, including fixed/shared focal, streamed BA and Dense cached/uncached
array comparisons. At the initial snapshot its broader browser failures were not all
classified. The follow-up below records their causes and the targeted repair. Full CI
outcomes and native measurements remain separate verification requirements.

## Browser verification follow-up (2026-10-22)

The checks in [#202](https://github.com/typeshade/typecv/pull/202) reproduce three
browser failures using the unchanged `3d33c257` test source and the same runtime
after documentation-only #200 (`1e4b602c`). These failures concern access to folded
panels, not a changed numerical acceptance threshold:

| Original check                        | Reproduced failure                                              | Cause and repair                                                                                                                                                                                                          |
| ------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Public CPU/WebGPU UI, processed count | Rendered text was empty instead of `8` at the original line 418 | Frame statistics are inside a closed `details`. Open it through its summary before reading values, and close it again before layout comparisons.                                                                          |
| Multi-video sample selection/removal  | Remove-button lookup timed out at the original line 1061        | Video rows and their controls now live in Settings. Open the drawer before removal, then close it before returning to the input/run controls.                                                                             |
| Per-component Dense multi-video       | The two-row wait timed out at the original line 1205            | The unopened Settings drawer had never mounted its video rows. This failure occurs before the numerical worker starts. Open Settings to verify the selected rows; also open run-stage details before inspecting progress. |

The repair uses ordinary UI controls. Existing registration, geometry, Dense,
cancellation and layout assertions remain in place; application and GPU source
are unchanged. The sample calibration credit is read from the opened Settings
drawer as well.

Verification environment: Node 24.19.0, Chromium 154.0.8037.57 on Linux with
SwiftShader; TypeShade pin `94203d3`. Runtime and the 73-page website build,
generated-page links/anchors/runtime references, and the independent completed
video/frame/camera synchronization browser regression pass. The repaired public
CPU/WebGPU UI flow and multi-video selection/progress/cancellation flow pass.
All four selected checks pass: generated-page links, public CPU/WebGPU UI,
multi-video selection/progress/cancellation, and per-component Dense completion.
The separate frame/video/camera synchronization check also passes. The Dense fixture
registers all eight views in one component and produces 6,941 points and 31,725 mesh
faces. These are software functional results, not native performance or ground-truth
quality measurements. The complete CI workflow is tracked separately.

Reproduce the affected checks after `npm ci`, `npm --prefix website ci`,
`npm run build` and `npm run build:website`:

```sh
TYPESHADE_CHROMIUM=/path/to/chromium node --test --test-reporter=tap --test-name-pattern='all generated|public UI|the lab shows a sample set card|a multi-video run with' website/site.test.mjs
TYPESHADE_CHROMIUM=/path/to/chromium node --test website/frame-navigation-browser.test.mjs
```

The collector was checked again: its 60 records still end at
`20261022025444698-40ee95c6`, whose site commit is `b2d27fe` and adapter is NVIDIA
Blackwell. There is no newer collected record for #198/#199. This cloud environment
has no native NVIDIA adapter, so RTX 2080/5060 Ti measurements remain R02; these
SwiftShader browser checks do not supply that baseline.

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

| ID  | Priority    | Remaining work                                       | Next reviewable result and acceptance evidence                                                                                                                                                                                                                                                                                                                                                                                                        |
| --- | ----------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R01 | P0          | Complete #199 verification                           | The reported folded-panel browser failures are reproduced and repaired in #202; the affected checks and separate frame/video/camera synchronization regression pass. Record complete CI outcomes for the exact heads and reconcile overlapping #190 coverage, including the additional tests proposed in #201. Native measurements remain R02.                                                                                                        |
| R02 | P0          | Fresh native-GPU baseline                            | Use the main baseline containing #198 and #199; compare #199's scalar/cache-disabled controls where needed. Run fixed inputs/configurations on RTX 2080/5060 Ti; record code/compiler/adapter identities, cold/warm behavior, sparse and Dense stage wall times, timestamp coverage, memory, registration and geometry quality. Measure #199's serial/cooperative selection threshold instead of treating 128 observations per camera as established. |
| R03 | P1          | Persistent common GpuScene                           | Integrate `sceneSteps` with a device scene for registration, continuation, indexed PnP gathering, triangulation, BA and filtering/compaction. Keep results resident; read summaries and requested snapshots. Compare against HostScene, including resumed/local runs, transfer growth and host memory.                                                                                                                                                |
| R04 | P1          | Complete budget compliance under #160                | Implement block-partitioned BA and split scene/index bindings below the resident floor; handle Dense's remaining minimum-image floor and budget-limited pair estimation, matching, features/ORB/global assignment and other stages. Force split paths with small budgets and compare outputs or explicitly stated final-cost tolerances. See [remaining limits](gpu-budget.md).                                                                       |
| R05 | P1          | Current Dense quality baseline and depth improvement | Re-run current #195 presets and library modes on fixed ETH3D/TUM inputs. Measure depth and surface accuracy/completeness, F-scores and weak-texture/floor coverage. Investigate slanted planes, far surfaces and low overlap; coarse-to-fine PatchMatch is a candidate, not a delivered improvement.                                                                                                                                                  |
| R06 | P2          | Dense reference sample/weight reuse                  | After #199's relative-pose cache, prototype reference gray/bilateral-weight caching for every supported window size. Measure registers/shared memory and native stage time; compare depth/normal/cost and final geometry.                                                                                                                                                                                                                             |
| R07 | P2          | Reduced camera solver and Schur cache scaling        | Keep the current cache eligibility/work guards until bounded assembly is available. Compare a budget-based extension and tiled Cholesky on small camera systems with PCG; retain a suitable large-system route. Check residuals, accepted LM costs and final reprojection/registration.                                                                                                                                                               |
| R08 | P2          | Current TSDF fusion profiling and culling            | Replay fixed depth maps with the current grouped-visibility mode. Separate kernels, uploads/read waves and host work. If warranted, add conservative block/frustum culling with identical retained evidence and geometry, including near-plane and truncation boundaries.                                                                                                                                                                             |
| R09 | P2          | Descriptor reuse and preview readback                | Profile a bounded decoded/resident descriptor LRU across view-graph candidate batches; reserve its bytes and release before mapping. Reduce preview-only depth on the GPU while preserving edge and positive-depth-mean semantics. Keep final depth/geometry reads and outputs unchanged.                                                                                                                                                             |
| R10 | P2          | Photo pair selection                                 | Complete #148's all-pairs path for small sets and retrieval path above the window; compare registration, pair counts, order behavior, feature lifetimes, peak bytes and runtime.                                                                                                                                                                                                                                                                      |
| R11 | P3          | Robust sparse geometry                               | Add calibrated five-point Essential estimation and stronger degeneracy/model selection. Test all real minimal branches, planar/rotation/low-baseline cases, noise and outliers against independent references and held-out datasets.                                                                                                                                                                                                                  |
| R12 | P3          | Remaining calibration                                | Implement camera-group focal BA on GPU, joint distortion optimization and broader camera models where supported. Existing pinhole/radial k1/k2 projection and CPU focal groups are implemented; verify new Jacobians, bounds, observability and calibration recovery.                                                                                                                                                                                 |
| R13 | P3          | Long-term and multi-map reconstruction               | Add relocalization/revisit constraints, loop closure, scalable global correction and the stalled-growth Sim(3) submap fallback. Local/global BA already exists; submap alignment, component connection and drift recovery need their own failure and quality gates.                                                                                                                                                                                   |
| R14 | Maintenance | Keep status and evidence aligned                     | Update these roadmaps when PRs merge, change scope or gain native/real-scene evidence. Record implementation, default/API scope and verification separately; maintain the [physical-GPU run list](physical-gpu-runs.md).                                                                                                                                                                                                                              |

## What is already implemented

Main contains GPU SIFT/ORB and optional pyramidal KLT; automatic/shared focal;
global descriptors and the Lab's canonical multi-video/view-graph/component pipeline;
GPU numerical SfM, cooperative BA camera products, Dense relative-pose caching,
time budgets and streamed BA/triangulation; PatchMatch MVS,
depth confidence, pixel and hashed TSDF fusion, grouped visibility, verified planar
completion, cleanup and meshing. Per-component Dense is also integrated in the Lab.

The public GPU operators are exported by [`src/gpu/index.ts`](../src/gpu/index.ts).
Multi-video orchestration uses [`src/gpu/internal.ts`](../src/gpu/internal.ts).
KLT/ORB and several quality options have opt-in or distinct Lab/library defaults.
Optional Dense graph-cut labeling uses a host min-cut solver; its existence is not
a claim of an all-GPU labeling path. These implementation facts do not establish
broad COLMAP-equivalent accuracy, registration success, throughput or scale.
