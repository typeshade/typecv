# Reconstruction roadmap and evidence gates

The goal is a controllable, tested reconstruction system. Completing the first
incremental core does not establish COLMAP-equivalent robustness or performance.
Every stage needs fixed datasets/configuration, regression tests and recorded
failure cases. Keep geometry correctness, feature quality and acceleration
measurements separate.

| Stage                                                                 | Status                                 | Evidence / completion gate                                                                                                             |
| --------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Two-view core and numerical refinement                                | Implemented                            | Known-pose synthetic/noisy/outlier fixtures; compatibility tests                                                                       |
| Camera model, inverse distortion, pose convention, analytic Jacobians | Implemented for pinhole + radial k1/k2 | Central-difference checks, round trips, lens-fold and SO(3) rejection                                                                  |
| Conflict-safe tracks and multi-view triangulation                     | Implemented                            | Ownership/order invariants, corrupted observations, positive depth, new visibility                                                     |
| P3P registration and incremental map                                  | Implemented                            | Known trajectories, planar/general PnP, 25% PnP outliers, disconnected and indirect registration                                       |
| Local/global fixed-intrinsics BA                                      | Implemented for small scenes           | Fixed observation objective, anchor/scale invariants, boundary-camera fixation, no input mutation                                      |
| Independent real-video baseline                                       | One exploratory case recorded          | Same SIFT/matches/fixed camera parameters; all failures retained; no ground-truth claim                                                |
| Calibrated public-dataset benchmark suite                             | Pending                                | Pose/structure error after similarity alignment, registration rate, coverage and runtime across multiple scene types                   |
| Five-point Essential + stronger model selection                       | Pending                                | All real minimal-solver branches; noise/outlier/planar/rotation fixtures and independent solver comparison                             |
| Shared focal/distortion optimization                                  | Pending                                | Camera parameter Jacobians, bounds/priors, observability checks and known-calibration recovery                                         |
| Large-scale BA and mapper state                                       | Pending                                | Sparse/iterative camera solve, incremental track updates, memory/runtime scaling and recovery from failed registration                 |
| Native image features and matching                                    | Pending                                | Detector/descriptor fixtures, geometric precision/recall, sequential/retrieval matching and reproducible feature ids                   |
| TypeShade acceleration                                                | Experimental sketch only               | Compiler execution, f64 CPU vs f32 CPU/GPU parity including failure boundaries, resource lifetime tests and transfer-inclusive timings |
| Dense MVS, fusion, meshing                                            | Pending                                | Calibrated depth accuracy/completeness, occlusions/visibility tests, fusion and geometry comparison                                    |

Next prioritize a calibrated benchmark suite, five-point/model-selection tests
and shared calibration. A fixed-intrinsics reference core remains useful as an
oracle while native features, optimizers and GPU kernels evolve. Preserve the
ability to run every stage independently and inspect its inputs, outputs,
thresholds, rejected hypotheses and solver diagnostics.
