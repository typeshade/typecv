# TypeCV documentation

TypeCV covers computer vision broadly. SfM is a current implementation area
within geometry and 3D vision. Planned capabilities and implemented APIs are
identified separately throughout these documents.

## Project-wide

- [Project introduction and development](../README.md)
- [Architecture and TypeShade execution model](architecture.md)
- [TypeShade capabilities: source review and executable GPU probes](typeshade-capabilities.md)
- [GPU geometric vision operators, resource contracts and optimization](gpu-geometry.md)
- [GPU E/H estimation, relative pose and initialization contracts](gpu-initialization.md)
- [Computer vision roadmap and evidence gates](roadmap.md)

## Geometry and 3D vision: current SfM implementation

- [SfM quickstart and current scope](sfm-quickstart.md)
- [Calibrated two-view API and numerical behavior](sfm.md)
- [Incremental sparse reconstruction and bundle adjustment](incremental-sfm.md)
- [TypeShade GPU execution and resident scoring](gpu-sfm.md)
- [SfM-specific roadmap](sfm-roadmap.md)

## Experiments

- [Exploratory two-view pyramid video experiment](experiments/pyramid-8134952/README.md)
- [Controlled-input pyramid video / COLMAP comparison](experiments/incremental-pyramid/README.md)
- [Recorded software WebGPU run](experiments/gpu/pyramid-software.json)
- [GPU P3P branch comparison on pyramid observations](experiments/gpu/p3p-pyramid-software.json)
- [GPU PnP/triangulation/BA in the incremental pyramid reconstruction](experiments/gpu/incremental-geometry-software.json)
- [GPU numerical initialization through BA, with similarity-aligned camera comparison](experiments/gpu/incremental-initialization-software.json)

Additional domain guides will accompany their implementations; the roadmap
does not imply that image processing, optical flow, SLAM or learned-vision APIs
are already available.
