# TypeCV documentation

TypeCV covers computer vision broadly. SfM is a current implementation area
within geometry and 3D vision. Planned capabilities and implemented APIs are
identified separately throughout these documents.

## Project-wide

- [Project introduction and development](../README.md)
- [Design rules: one answer everywhere, the order of priorities, the front end (rules accepted, front end proposed)](design-rules.md)
- [Architecture and TypeShade execution model](architecture.md)
- [TypeShade capabilities: source review and executable GPU probes](typeshade-capabilities.md)
- [GPU geometric vision operators, resource contracts and optimization](gpu-geometry.md)
- [Resident GPU pair verification, E/H estimation and relative pose](gpu-initialization.md)
- [Computer vision roadmap and evidence gates](roadmap.md)
- [Reconstruction status and remaining work: dated PR and verification snapshot](reconstruction-status.md)
- [Library structure by capability area and the refactoring programme (accepted)](library-structure.md), with its audits of the [sparse pipeline](audits/2026-10-26-sparse-structure.md), the [lab's orchestration](audits/2026-10-26-lab-structure.md) and the [kernel style](audits/2026-10-26-kernel-style.md)

## Geometry and 3D vision: current SfM implementation

- [SfM quickstart and current scope](sfm-quickstart.md)
- [Calibrated two-view API and numerical behavior](sfm.md)
- [Incremental sparse reconstruction and bundle adjustment](incremental-sfm.md)
- [Automatic shared focal for uncalibrated input: implementation and measurements](auto-focal.md)
- [Experimental directional constraints in final BA and automatic focal calibration](focal-directions.md)
- [TypeShade GPU execution and resident scoring](gpu-sfm.md)
- [High-resolution video, tiled features and bounded GPU execution](gpu-sfm-scaling.md)
- [GPU working budget: streamed BA and triangulation, with remaining refusal limits](gpu-budget.md)
- [Lab run records: what the SfM lab can send, and the collector](lab-run-records.md)
- [The lab's live view: what changes, what works, and when the worker last reported (accepted)](lab-live-view.md)
- [Physical-GPU runs owed: the lab runs that only a physical GPU can measure](physical-gpu-runs.md)
- [Hardware observations: the lab's records by device, with each remedy](hardware-observations.md)
- [GPU-resident scene: design (accepted)](gpu-resident-scene.md)
- [Engine v2: direction (accepted)](engine-v2.md)
- [Dense multi-view stereo: implemented depth, fusion, cleanup, meshing and quality records](dense-mvs.md)
- [Pyramidal Lucas–Kanade (KLT) tracking: steps 1 to 6 implemented, physical-device gate open](klt-tracking.md)
- [GPU throughput: measurements, delivered changes and remaining optimization work](gpu-throughput.md)
- [Multi-video reconstruction: implemented Lab pipeline and remaining integration work](multi-video.md)
- [SfM-specific roadmap](sfm-roadmap.md)
- [General gameplay/cinematic audit, observability and staged reconstruction plan](general-video-reconstruction.md)
- [Calibrated relative rotation, retained 2D observations and deferred structure](rotation-tracking.md)
- [Rotation tracking A/B results and calibrated default policy](rotation-tracking-comparison.md)
- [Initial pose/depth quality and foreground observation selection](structure-quality.md)
- [Chimney Rock record 58532c69: focal calibration, relief collapse and the dense stage (diagnosis)](chimney-rock-58532c69.md)

## Experiments

- [Exploratory two-view pyramid video experiment](experiments/pyramid-8134952/README.md)
- [Controlled-input pyramid video / COLMAP comparison](experiments/incremental-pyramid/README.md)
- [COLMAP demo datasets (south-building) as controlled benchmarks](experiments/colmap-datasets/README.md)
- [Recorded software WebGPU run](experiments/gpu/pyramid-software.json)
- [GPU P3P branch comparison on pyramid observations](experiments/gpu/p3p-pyramid-software.json)
- [GPU PnP/triangulation/BA in the incremental pyramid reconstruction](experiments/gpu/incremental-geometry-software.json)
- [GPU numerical initialization through BA, with similarity-aligned camera comparison](experiments/gpu/incremental-initialization-software.json)
- [Resident pair verification: identical models, measured maps/bytes and buffer tradeoffs](experiments/gpu/resident-pair-verification-software.json)

Additional domain guides will accompany their implementations. Image/feature
operators and sparse KLT tracking already exist; their presence does not imply
a general dense optical-flow, SLAM or learned-vision API. The GPU public exports
and Lab/internal orchestration have different scopes, as described in the roadmap.
