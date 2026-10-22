# TypeCV roadmap

TypeCV covers computer vision broadly, with TypeShade as the foundation for
GPU computation. SfM is the first substantial application area being developed.
Its current implementation informs reusable geometry, numerical and device
primitives; it does not define the project's full scope.

Implementation status reviewed on 2026-10-22 against `main` at `f660b52c`.
The [reconstruction status and remaining work](reconstruction-status.md) records
the PR snapshot, verification gaps and current reconstruction priorities.

## Capability areas

The following areas describe the intended scope. Planned capabilities are not
available APIs or commitments to a particular algorithm or release order.

| Area                             | Intended capabilities                                                                                   | Current state                                                                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared data and device execution | Images, feature sets, observation buffers, memory ownership, batching, fusion and explicit transfers    | Typed TypeShade pools, resident features and numerical solves, time budgets and streamed BA/triangulation/MVS exist; a persistent cross-operator GPU scene and remaining budget-limit paths are open |
| Image processing                 | Color conversion, filtering, convolution, resampling, image pyramids, gradients and morphology          | Gray conversion, scale-space and tracking pyramids, gradients and CLAHE exist within the current operators; a general image-processing API remains open                                              |
| Image analysis                   | Contours, connected components, region/shape measurements, template matching and classical segmentation | SLIC superpixels and Canny/Hough line operators exist; the broader analysis suite remains planned                                                                                                    |
| Features and correspondence      | Detection, descriptors, nearest-neighbor matching, geometric filtering and feature tracking             | GPU FAST-9/oriented BRIEF, SIFT, ORB, Hamming/tiled L2 matching and pyramidal KLT tracking exist; broader feature-quality and native-device evaluation remain open                                   |
| Geometric vision                 | Camera models, calibration, transforms, robust model fitting, pose estimation and triangulation         | CPU/GPU E/H, relative pose, P3P/PnP, triangulation, radial k1/k2 and automatic/shared focal exist; five-point estimation, distortion optimization and GPU focal groups remain open                   |
| Motion and video                 | Optical flow, temporal tracking, motion estimation, visual odometry and SLAM building blocks            | Video decoding, frame/keyframe selection, optional KLT and a multi-video Lab pipeline with global descriptors/view graphs exist; relocalization and loop closure remain open                         |
| 3D vision and reconstruction     | Stereo/depth, SfM, bundle adjustment, multi-view stereo, fusion and surface reconstruction              | Incremental sparse SfM, Schur/PCG BA, PatchMatch MVS, pixel/TSDF fusion, surface-preservation options, cleanup and meshing exist; depth quality, scale and whole-pipeline GPU residency remain open  |
| Learned vision integration       | Classification, detection, segmentation, learned features, preprocessing and postprocessing             | Planned; TypeNN interoperability is a direction to evaluate, not an existing integration                                                                                                             |
| Evaluation and interoperability  | Reproducible datasets, reference comparisons, data adapters and physical-GPU benchmarks                 | Synthetic/reference and software WebGPU checks, COLMAP comparisons and TUM/ETH3D evaluation harnesses exist; broad dataset coverage and fresh native-GPU baselines remain open                       |

Existing operators have different API scopes: `src/gpu/index.ts` exports the public
GPU surface; the Lab's multi-video orchestration currently uses `src/gpu/internal.ts`.
An implemented path does not establish a public API, a default setting, a measured
speedup or COLMAP-equivalent robustness. See the linked domain guides for those contracts.

## Shared implementation priorities

1. Define reusable data contracts: dimensions, channels, color space, numeric
   type, layout/stride, coordinate conventions and device ownership. Make
   conversions and host/device transfers explicit.
2. Extend the existing TypeShade execution layer into shared image, feature and
   observation storage. Apply tiling, workgroup memory, fusion, gather/scatter,
   compaction and reductions where the algorithm benefits from them.
3. Add foundational image operators and feature/matching primitives that several
   domains can reuse. Keep their APIs independent of reconstruction state.
4. Build motion, geometry, reconstruction and learned-vision integrations from
   these shared primitives. Keep algorithm orchestration separate from device
   operators and preserve inspectable intermediate results.
5. Establish domain benchmarks and hardware profiling alongside each capability,
   covering numerical error, failure behavior, memory and total pipeline latency.

Ongoing SfM work continues within this structure. Its next algorithms and
acceptance criteria live in the [SfM roadmap](sfm-roadmap.md); its GPU operators
are detailed in the [GPU SfM plan](gpu-sfm.md). Work in one domain should expose
reusable operators when their contracts are stable, without requiring other
domains to adopt a reconstruction pipeline.

## Evidence gates

Every new operator or pipeline needs an explicit input/output contract, supported
execution tiers, numerical tolerances and failure cases. CPU references and
independent libraries can serve as oracles where appropriate; they do not imply
that production execution falls back to CPU automatically.

Validation depends on the domain: image operators need border/interpolation and
format checks; matching needs precision/recall and tie behavior; geometry needs
coordinate, degeneracy and observability checks; video needs temporal consistency;
depth/reconstruction needs accuracy and completeness; learned components need
model/preprocessing compatibility and task-level metrics.

For device paths, compile and execute actual kernels, inspect dispatch/fusion
reports, test resource lifetimes and account for transfers. Report software
WebGPU correctness separately from physical-GPU performance. End-to-end latency
must include the relevant uploads, readbacks and synchronization.
