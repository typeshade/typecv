# SfM roadmap and evidence gates

Structure from motion is one application area within TypeCV. This document
tracks sparse/dense reconstruction work; the [project roadmap](roadmap.md)
covers the broader computer vision scope and shared infrastructure.

The goal is a controllable, tested reconstruction system. Completing the first
incremental core does not establish COLMAP-equivalent robustness or performance.
Every stage needs fixed datasets/configuration, regression tests and recorded
failure cases. Keep geometry correctness, feature quality and acceleration
measurements separate.

| Stage                                                                 | Status                                 | Evidence / completion gate                                                                                                                                       |
| --------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Two-view core and numerical refinement                                | Implemented                            | Known-pose synthetic/noisy/outlier fixtures; compatibility tests                                                                                                 |
| Camera model, inverse distortion, pose convention, analytic Jacobians | Implemented for pinhole + radial k1/k2 | Central-difference checks, round trips, lens-fold and SO(3) rejection                                                                                            |
| Conflict-safe tracks and multi-view triangulation                     | Implemented                            | Ownership/order invariants, corrupted observations, positive depth, new visibility                                                                               |
| P3P registration and incremental map                                  | Implemented                            | Known trajectories, planar/general PnP, 25% PnP outliers, disconnected and indirect registration                                                                 |
| Local/global fixed-intrinsics BA                                      | Implemented for small scenes           | Fixed observation objective, anchor/scale invariants, boundary-camera fixation, no input mutation                                                                |
| Independent real-video baseline                                       | One exploratory case recorded          | Same SIFT/matches/fixed camera parameters; all failures retained; no ground-truth claim                                                                          |
| Calibrated public-dataset benchmark suite                             | Pending                                | Pose/structure error after similarity alignment, registration rate, coverage and runtime across multiple scene types                                             |
| Five-point Essential + stronger model selection                       | Pending                                | All real minimal-solver branches; noise/outlier/planar/rotation fixtures and independent solver comparison                                                       |
| Shared focal/distortion optimization                                  | Pending                                | Camera parameter Jacobians, bounds/priors, observability checks and known-calibration recovery                                                                   |
| Large-scale BA and mapper state                                       | Pending                                | Sparse/iterative camera solve, incremental track updates, memory/runtime scaling and recovery from failed registration                                           |
| Native image features and matching                                    | Pending                                | Detector/descriptor fixtures, geometric precision/recall, sequential/retrieval matching and reproducible feature ids                                             |
| TypeShade acceleration                                                | Resident GPU scoring + hybrid two-view | Compiled kernels, f64 differential checks, actual WGSL/fusion/transfer/lifetime tests pass in SwiftShader; hardware timings, features, mapper and BA remain open |
| Dense MVS, fusion, meshing                                            | Pending                                | Calibrated depth accuracy/completeness, occlusions/visibility tests, fusion and geometry comparison                                                              |

Use a GPU-resident execution model as the target architecture: feature buffers,
batched geometry, triangulation and BA operators should share device state.
The [GPU plan and evidence](gpu-sfm.md) distinguish the implemented hybrid boundary
from the remaining device kernels. Calibrated benchmarks, five-point/model
selection and shared calibration remain correctness priorities alongside this
work. A fixed-intrinsics f64 reference remains useful as an oracle. Preserve the
ability to run every stage independently and inspect its inputs, outputs,
thresholds, rejected hypotheses and solver diagnostics.
