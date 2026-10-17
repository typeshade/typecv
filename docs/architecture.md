# TypeCV architecture

TypeCV is a computer vision library in TypeScript, with a GPU-oriented design
based on TypeShade. Its intended scope spans image processing and analysis,
correspondence, geometry, motion, 3D vision and learned-vision integration. SfM is one
consumer of the common operators and execution infrastructure.

This document describes the project-wide design direction. Only the components
listed as implemented below are available today; proposed contracts and layers
do not introduce new exports.

## Data and operator contracts

Shared data contracts should describe images, feature sets, descriptors, matches,
camera/pose data and sparse observations independently of any one pipeline.
Image contracts need dimensions, channels, color space, numeric type and
layout/stride. Geometric contracts need coordinate units, camera conventions
and index/ownership rules. Device storage needs an explicit owner and lifetime.
These general contracts are still to be designed; current APIs use packed typed
arrays documented by their individual modules.

Operators should perform reusable tasks such as filtering, interpolation,
descriptor distance, robust residual evaluation, reduction or projection.
Algorithm orchestration composes operators into feature matching, optical flow,
pose estimation, reconstruction or model preprocessing. Pipeline-specific state
should stay in the pipeline rather than becoming a prerequisite for shared
image or geometry operations.

## TypeShade execution

The target device model keeps reusable inputs and intermediates on the GPU.
Uploads, downloads and synchronization are explicit API boundaries. Fuse
compatible operations, batch small jobs and reduce results on the device before
reading summaries. Choose layouts and tile sizes for actual access patterns;
TypeShade compute entries, workgroup memory and barriers support specialized
kernels when portable loop fusion is insufficient.

TypeShade also provides emulated `f64` on the GPU, forward/reverse automatic
differentiation, and CPU execution of the same IR. Precision and derivative
generation should be evaluated through these facilities before introducing
separate CPU-only numerical stages. The [capability review and executable
probes](typeshade-capabilities.md) record their tested scope and primary sources.

The implementation draws on TypeNN's resident execution, resource ownership,
buffer pooling and observable reports. These patterns can support many vision
domains. An image pipeline may reuse a pyramid across feature detection and
tracking; a matcher may keep descriptors resident across candidate pairs.
Image/descriptor pipelines remain design targets. The geometric optimizer now
retains observations and accepted/trial state within a BA call; sharing that
state across all mapper stages remains to be implemented. Pair verification
keeps radial inversion, stable compaction, H/E fitting and relative-pose checks
in one resident plan, returning original row indices and a geometry summary.

CPU implementations remain useful for reference numerics, small control tasks
and explicitly selected hybrid execution. Each API must identify its execution
boundary. Supported backend selection and failure behavior should be explicit,
and GPU correctness must be tested using actual compiled kernels.

Learned-vision integration can cover classification, detection, segmentation
and learned correspondence, with explicit model input/output contracts. TypeNN
can be evaluated for these components and compatible device-data exchange.
TypeCV's shared operators should also support classical vision;
there is no implemented TypeNN integration or zero-copy tensor bridge today.

## Current implementation

| Location / entry                   | Implemented responsibility                                                                                                                  | Current boundary                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `src/geometry`                     | Camera projection/inversion, pose utilities, homography, P3P/PnP, triangulation and supporting numerics                                     | Synchronous CPU geometry                                                                                            |
| `src/sfm`                          | Two-view Essential estimation, relative pose, point recovery and refinement                                                                 | Independent synchronous CPU reference                                                                               |
| `src/reconstruction`               | Tracks, incremental sparse mapping and pose/point bundle adjustment with optional joint shared-focal refinement                             | CPU pipeline consuming feature coordinates and matches; independent dense-Schur reference for shared focal          |
| `src/gpu`, `typecv/gpu`            | TypeShade pair inversion/compaction, E/H/relative pose, P3P/PnP, direct-SVD triangulation, filtering, Schur/PCG BA and asynchronous mapping | Implemented numerical geometry and pair compaction on GPU; final result indexing, packing and graph/control on host |
| `scripts` and experiment documents | Optional input adapters, reference comparisons and reproducible measurements                                                                | OpenCV supplies features for experiments; software WebGPU verifies current kernels                                  |

The root `typecv` entry currently exports CPU geometry and reconstruction.
`typecv/gpu` bundles the TypeShade runtime for its compiled kernels. These are
the current package boundaries, not the final module taxonomy. No image, video
or learned-vision subpath is exported yet.

TypeShade runtime configuration is shared within a JS context. The current
TypeCV device supports one active owner and serialized work; applications using
other TypeShade consumers must coordinate configuration. Borrowing a GPUDevice
preserves device ownership but does not isolate that shared configuration.

## Verification and growth

Add operators with domain-specific contracts and independent reference checks.
Record supported numeric ranges and tolerances, test malformed/degenerate input,
and preserve failures in pipeline experiments. Measure runtime allocations,
transfers, dispatches and total latency separately from task accuracy.

The [project roadmap](roadmap.md) tracks capability areas. The
[SfM roadmap](sfm-roadmap.md) and [GPU SfM implementation](gpu-sfm.md) are
domain-specific applications of this architecture.

The [GPU geometry guide](gpu-geometry.md) documents reusable pose, triangulation,
observation and optimization operators independently of the SfM pipeline.
