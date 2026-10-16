# TypeCV roadmap

TypeCV covers computer vision broadly, with TypeShade as the foundation for
GPU computation. SfM is the first substantial application area being developed.
Its current implementation informs reusable geometry, numerical and device
primitives; it does not define the project's full scope.

## Capability areas

The following areas describe the intended scope. Planned capabilities are not
available APIs or commitments to a particular algorithm or release order.

| Area                             | Intended capabilities                                                                                   | Current state                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared data and device execution | Images, feature sets, observation buffers, memory ownership, batching, fusion and explicit transfers    | Typed TypeShade pools, resident geometric solves and reports exist; persistent cross-operator scenes and general image/feature contracts are planned |
| Image processing                 | Color conversion, filtering, convolution, resampling, image pyramids, gradients and morphology          | Native operators planned                                                                                                                             |
| Image analysis                   | Contours, connected components, region/shape measurements, template matching and classical segmentation | Planned                                                                                                                                              |
| Features and correspondence      | Detection, descriptors, nearest-neighbor matching, geometric filtering and feature tracking             | Geometric verification exists; extraction/matching currently use an optional OpenCV experiment adapter                                               |
| Geometric vision                 | Camera models, calibration, transforms, robust model fitting, pose estimation and triangulation         | CPU reference and GPU E/H, relative pose, P3P/PnP, triangulation and filtering exist; broader calibration remains open  |
| Motion and video                 | Optical flow, temporal tracking, motion estimation, visual odometry and SLAM building blocks            | Standalone modules planned; current reconstruction consumes prepared features and matches                                                            |
| 3D vision and reconstruction     | Stereo/depth, SfM, bundle adjustment, multi-view stereo, fusion and surface reconstruction              | CPU reference and GPU numerical sparse SfM, including fixed-intrinsics Schur/PCG BA, exist; dense depth/fusion/meshing are planned                            |
| Learned vision integration       | Classification, detection, segmentation, learned features, preprocessing and postprocessing             | Planned; TypeNN interoperability is a direction to evaluate, not an existing integration                                                             |
| Evaluation and interoperability  | Reproducible datasets, reference comparisons, data adapters and physical-GPU benchmarks                 | Geometry fixtures, software WebGPU checks and one exploratory video comparison exist; broader benchmarks and adapters are planned                    |

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
