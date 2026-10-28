# TypeCV

**Computer vision in TypeScript, with GPU computation built on TypeShade.**

[Public website](https://cv.typeshade.dev/) · [Documentation](https://cv.typeshade.dev/docs/) · [Browser playground](https://cv.typeshade.dev/playground/) · [한국어](https://cv.typeshade.dev/ko/)

TypeCV aims to provide reusable primitives and pipelines across image processing
and analysis, features and correspondence, geometric vision, motion and video, 3D vision, and
learned-vision integration. **Structure from motion (SfM) is one subset of this
scope** and the first substantial pipeline under development.

The design centers on explicit data contracts, composable operators and
GPU-resident execution. TypeShade provides the kernel compilation and execution
foundation. Shared device infrastructure should support multiple vision domains
as they are implemented.

## Scope and current state

TypeCV is at an early stage. The areas below describe the project scope;
implementation is currently concentrated in geometry, sparse reconstruction and
an initial TypeShade execution layer.

| Area                        | Scope                                                                               | Current state                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Image processing            | Color, filtering, resampling, pyramids, gradients and morphology                    | Planned                                                                                                                          |
| Image analysis              | Contours, regions, shape measurements, template matching and classical segmentation | Planned                                                                                                                          |
| Features and correspondence | Detection, description, matching and tracking                                       | Experimental TypeShade GPU FAST-9/oriented BRIEF detector and mutual Hamming matcher; broader feature coverage remains open      |
| Geometric vision            | Cameras, transforms, calibration, robust fitting, pose and triangulation            | Camera/pose primitives, robust estimation and triangulation implemented; broader calibration remains open                        |
| Motion and video            | Optical flow, temporal tracking, visual odometry and SLAM building blocks           | Browser local-video frame sampling and sparse GPU SfM are demonstrated; flow, persistent tracking and SLAM remain planned        |
| 3D vision                   | Stereo/depth, SfM, bundle adjustment, MVS, fusion and surfaces                      | Sparse two-view/incremental SfM and pose/point BA with optional joint shared-focal refinement implemented; dense methods planned |
| Learned vision integration  | Classification, detection, segmentation, learned features and model integration     | Planned; TypeNN interoperability to be evaluated                                                                                 |

The root package currently exposes synchronous CPU geometry and reconstruction
with no runtime dependencies. The optional `typecv/gpu` entry provides compiled
TypeShade kernels for P3P/PnP RANSAC, robust multi-view triangulation, observation
filtering and joint bundle adjustment with a matrix-free Schur/PCG solver.
These operators use GPU emulated f64 and are connected to an asynchronous
incremental mapper. Pair inversion, E/H estimation and two-view pose/point
recovery also execute in TypeShade. FAST-9/oriented BRIEF feature extraction and
bidirectional Hamming matching are experimental GPU operators; see the
[GPU feature guide](docs/gpu-features.md). `verifyImagePairGpu` keeps radial inversion,
stable compaction, H/E consensus and pose checks in one resident plan. Host work
covers validation, final result indexing, packing and graph/control; geometry
still crosses later mapper API boundaries. General filtering, image pyramids,
optical flow and dense video operators are not exported yet.

See the [architecture](docs/architecture.md) for shared design and current module
boundaries, and the [project roadmap](docs/roadmap.md) for capability areas and
validation criteria.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and TypeShade execution](docs/architecture.md)
- [Computer vision roadmap](docs/roadmap.md)
- [GPU geometric vision operators and joint optimization](docs/gpu-geometry.md)
- [Resident GPU pair verification, E/H estimation and relative pose](docs/gpu-initialization.md)
- [SfM quickstart and implemented reconstruction scope](docs/sfm-quickstart.md)
- [Two-view API](docs/sfm.md), [incremental reconstruction](docs/incremental-sfm.md), and [GPU SfM implementation](docs/gpu-sfm.md)
- [SfM-specific roadmap](docs/sfm-roadmap.md)

## Development

Use Node.js 22.18+ or Node.js 24:

```sh
npm ci
npm run check                 # formatting, types, CPU/reference tests and builds
npm run test:surface          # the declaration surface and the kernels' WGSL/GLSL digests, after a build
npm run test:digests          # the sparse pipeline's characterization digests (Node 22), after a build
npm run example               # current two-view SfM example
npm run example:incremental   # current multi-view SfM example
npx playwright install --only-shell chromium
npm run test:webgpu           # actual WGSL execution using software WebGPU
npm run audit:typeshade       # f64, workgroup/atomic and generated-derivative probes
npm run evaluate:p3p          # GPU P3P branch/reference comparison
npm run benchmark:gpu         # current GPU scoring and reconstruction measurements
npm run benchmark:sfm:workgroups # specialize the SfM guided kernel at 64/128/256 lanes
node scripts/profile-kernel-ir.mjs # inspect actual kernel IR and O0/O2 WGSL offline
node scripts/import-graph.mjs # runtime import cycles and layering breaks (docs/library-structure.md 3.1)
```

The public site and browser applications are maintained in [typeshade/web](https://github.com/typeshade/web). This repository contains the TypeCV library package; `npm run build` produces `dist/`.
Examples currently demonstrate the SfM subset; additional domain examples will
accompany their implementations.

## Validation

Current tests cover geometry, sparse reconstruction, compiled TypeShade P3P/scoring,
resource lifetimes and actual WebGPU execution. GPU CI uses SwiftShader;
physical-GPU performance remains unmeasured. The characterization digests of the
sparse pipeline (`npm run test:digests`), the declaration surface and the kernels'
shader text (`npm run test:surface`) hold a structure change to the same outputs
([library structure](docs/library-structure.md), phase 0). The digest fixtures hold
Node 22's bits, since identity across Node majors is not established; CI compares
them on Node 22, and its `digests` job compares a pull request's merge base and head
on one runner for each Node major, SwiftShader included. The
[controlled-input pyramid experiment](docs/experiments/incremental-pyramid/README.md)
is one reconstruction case, not an evaluation of the entire computer vision
scope. Accuracy, failure behavior and transfer-inclusive performance need
separate evidence for each operator and domain.

## License

[Apache-2.0](LICENSE).
