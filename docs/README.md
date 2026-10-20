# TypeCV documentation

TypeCV covers computer vision broadly. SfM is a current implementation area
within geometry and 3D vision. Planned capabilities and implemented APIs are
identified separately throughout these documents.

## Project-wide

- [Project introduction and development](../README.md)
- [Architecture and TypeShade execution model](architecture.md)
- [TypeShade capabilities: source review and executable GPU probes](typeshade-capabilities.md)
- [GPU geometric vision operators, resource contracts and optimization](gpu-geometry.md)
- [Resident GPU pair verification, E/H estimation and relative pose](gpu-initialization.md)
- [Computer vision roadmap and evidence gates](roadmap.md)

## Geometry and 3D vision: current SfM implementation

- [SfM quickstart and current scope](sfm-quickstart.md)
- [Calibrated two-view API and numerical behavior](sfm.md)
- [Incremental sparse reconstruction and bundle adjustment](incremental-sfm.md)
- [TypeShade GPU execution and resident scoring](gpu-sfm.md)
- [High-resolution video, tiled features and bounded GPU execution](gpu-sfm-scaling.md)
- [Lab run records: what the SfM lab can send, and the collector](lab-run-records.md)
- [Physical-GPU runs owed: the lab runs that only a physical GPU can measure](physical-gpu-runs.md)
- [Hardware observations: the lab's records by device, with each remedy](hardware-observations.md)
- [GPU-resident scene: design (accepted)](gpu-resident-scene.md)
- [Engine v2: direction (accepted)](engine-v2.md)
- [Dense multi-view stereo after the sparse SfM: design (proposed)](dense-mvs.md)
- [Pyramidal Lucas–Kanade (KLT) tracking for video: design (steps 1 to 5 implemented)](klt-tracking.md)
- [Reconstructing one scene from several videos: design (proposed)](multi-video.md)
- [SfM-specific roadmap](sfm-roadmap.md)

## Experiments

- [Exploratory two-view pyramid video experiment](experiments/pyramid-8134952/README.md)
- [Controlled-input pyramid video / COLMAP comparison](experiments/incremental-pyramid/README.md)
- [COLMAP demo datasets (south-building) as controlled benchmarks](experiments/colmap-datasets/README.md)
- [Recorded software WebGPU run](experiments/gpu/pyramid-software.json)
- [GPU P3P branch comparison on pyramid observations](experiments/gpu/p3p-pyramid-software.json)
- [GPU PnP/triangulation/BA in the incremental pyramid reconstruction](experiments/gpu/incremental-geometry-software.json)
- [GPU numerical initialization through BA, with similarity-aligned camera comparison](experiments/gpu/incremental-initialization-software.json)
- [Resident pair verification: identical models, measured maps/bytes and buffer tradeoffs](experiments/gpu/resident-pair-verification-software.json)

Additional domain guides will accompany their implementations; the roadmap
does not imply that image processing, optical flow, SLAM or learned-vision APIs
are already available.
