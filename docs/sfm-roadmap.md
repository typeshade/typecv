# SfM roadmap and evidence gates

Structure from motion is one application area within TypeCV. This document
tracks sparse/dense reconstruction work; the [project roadmap](roadmap.md)
covers the broader computer vision scope and shared infrastructure.

The goal is a controllable, tested reconstruction system. Completing the first
incremental core does not establish COLMAP-equivalent robustness or performance.
Every stage needs fixed datasets/configuration, regression tests and recorded
failure cases. Keep geometry correctness, feature quality and acceleration
measurements separate.

Implementation status reviewed on 2026-10-22 against `main` at `f660b52c`.
The [reconstruction status and remaining work](reconstruction-status.md) separates
merged code, open PRs, outstanding verification and unimplemented work. In particular,
PR #199's BA camera reduction and Dense relative-pose cache are under verification,
not part of this main baseline.

| Stage                                                                 | Implementation status                                                                                                             | Evidence / remaining gate                                                                                                                                         |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Two-view core and numerical refinement                                | Implemented; Essential estimation remains eight-point                                                                             | Known-pose/noise/outlier fixtures; calibrated five-point and stronger model selection remain open                                                                 |
| Camera model, inverse distortion, pose convention, analytic Jacobians | Pinhole + radial k1/k2 implemented                                                                                                | Central-difference checks, round trips, lens-fold and SO(3) rejection; broader camera models remain open                                                          |
| Conflict-safe tracks and multi-view triangulation                     | Implemented, including GPU track chunks                                                                                           | Ownership/order invariants, corrupted observations, positive depth and new visibility; persistent device track updates remain open                                |
| P3P registration and incremental map                                  | CPU/GPU numerical stages implemented                                                                                              | Known trajectories, planar/general PnP, outliers and failed-registration recovery; long-term relocalization remains open                                          |
| Local/global fixed-intrinsics BA                                      | Schur/PCG, growth-based scheduling, local subsets and observation streaming implemented                                           | Fixed objective, gauge and boundary-camera checks; resident-array floors, block-partitioned BA and native-GPU scaling remain open                                 |
| Shared focal and automatic focal                                      | Shared focal refinement and focal search implemented                                                                              | Known-calibration and dataset comparisons in [automatic focal](auto-focal.md); a fresh native-GPU baseline remains open                                           |
| Camera groups and distortion optimization                             | CPU focal groups implemented; multi-group GPU BA and distortion optimization open                                                 | Per-group calibration recovery and Jacobian/observability checks; radial parameters currently stay fixed                                                          |
| Native image features and matching                                    | GPU FAST/BRIEF, SIFT, ORB, Hamming and tiled L2 implemented                                                                       | [Feature records](gpu-features.md); held-out scenes, device agreement, success rates and broader native performance remain open                                   |
| Video tracking                                                        | Pyramidal KLT, SIFT joins and optional Lab gate/matching paths implemented                                                        | [KLT steps 1-6](klt-tracking.md); physical-device agreement, rates and the default-enablement gate remain open                                                    |
| Retrieval and multi-video                                             | Global descriptors, canonical IDs, view graph, component mapping and per-component Dense implemented in the Lab/internal pipeline | [Multi-video records](multi-video.md); GPU focal groups, stalled-growth Sim(3) submap fallback and cross-component integration remain open                        |
| GPU execution                                                         | Resident features/pair verification, time budgets, GPU BA solver state and memory streaming implemented                           | [GPU throughput](gpu-throughput.md) and [budget limits](gpu-budget.md); common GpuScene and all-stage budget compliance remain open                               |
| Dense MVS                                                             | GPU PatchMatch photometric/geometric depth, normals, filtering and confidence implemented                                         | [TUM/ETH3D and synthetic records](dense-mvs.md); depth accuracy/completeness, weak-texture coverage and fresh native-device measurements remain open              |
| Fusion, cleanup and meshing                                           | Pixel fusion, hashed TSDF, carving, grouped visibility, verified planar completion, cleanup and mesh extraction implemented       | [Dense records](dense-mvs.md); real-scene re-evaluation after #195 and current fusion profiling remain open; optional graph-cut labeling still uses a host solver |
| Evaluation                                                            | Controlled COLMAP comparisons and TUM/ETH3D harnesses implemented; broad benchmark coverage incomplete                            | Separate COLMAP agreement from ground-truth accuracy; record registration, reprojection, depth/surface accuracy and completeness, memory and runtime              |

PR #197 delivered independent Dense readback overlap. PR #198 corrected duplicate
GPU kernel accounting and the missing multi-video Dense device table. Historical
kernel sums cannot serve as corrected GPU-time baselines; stage wall times and
software correctness records have narrower uses. See the
[dated PR and measurement snapshot](reconstruction-status.md).

Use a GPU-resident execution model as the target architecture: feature buffers,
batched geometry, triangulation and BA operators should share device state.
The [GPU plan and evidence](gpu-sfm.md) distinguish implemented numerical kernels
from remaining persistent-state and scaling work. Calibrated benchmarks, five-point/model
selection and the remaining calibration work stay correctness priorities alongside this
work. A fixed-intrinsics f64 reference remains useful as an oracle. Preserve the
ability to run every stage independently and inspect its inputs, outputs,
thresholds, rejected hypotheses and solver diagnostics.
