# High-resolution video and bounded GPU execution

TypeCV separates **source resolution**, **analysis resolution** and **GPU tile size**.
There is no 18K source-size constant. A 4K, 8K, 16K or 32K source can use the same
bounded GPU path. Codec support and native decoder memory remain separate input
constraints; successful proxy analysis does not prove full-resolution processing.

## Full-resolution tiled features

`extractFeaturesTiledGpu(device, source, options)` accepts a `GpuImageSource` with
`width`, `height` and `readRegion({ x, y, width, height })`. The provider returns only
that region as RGBA bytes. A decoder or an image tile store implements the region
reader; the feature extractor never requests a full-frame GPU texture or buffer.

The design follows TypeNN's `DeviceDataset` sharding: size work from actual device
storage/working limits, retain compact outputs, and execute bounded chunks through
TypeShade. TypeShade's existing workgroup tiles and resident/fuse execution provide
the compute foundation. TypeCV supplies the image-specific ownership and halo rules.

- Core tiles are 480 pixels by default, aligned to 24-pixel detector cells.
- A 24-pixel halo covers FAST and rotated BRIEF sampling. Only the core owns output
  keypoints, so overlap does not duplicate features or clamp descriptors at seams.
- `readRegion` receives at most 528 × 528 RGBA pixels with default settings, including
  at odd final strips. This is 1,115,136 bytes independent of source resolution.
- Coordinates are translated back to the original source grid using JavaScript
  numbers and returned as `Float64Array`; a large source origin never enters local
  f32 feature coordinates.
- A bounded heap retains at most `maxFeatures` compact records across all tiles.
  The default is 2,000. Selection is by score, with deterministic coordinate ties.
- Pixel and feature GPU buffers are reused/released between tiles. Diagnostics keep
  at most 128 detailed groups; `tiling.totalGroups` and `groupsTruncated` distinguish
  sampled group details from complete dispatch/submission totals.

`featureTileRegions(width, height, tileEdge)` exposes the lazy core/read schedule for
region stores. CPU differential tests compare tiled and full-frame scores and BRIEF
bits across ordinary and odd-sized seams. A 32,768 × 16,384 planner test checks every
core, complete coverage and bounded reads without allocating the source image.

The current tiled API returns a compact host feature set. Feature merging and the
bounded top-score heap are host work; FAST and BRIEF run in TypeShade. It does not
yet supply a native region decoder for every video codec or an end-to-end tiled
video import UI. Those adapters can use `GpuImageSource` without changing the GPU
feature API. The lab's video importer below uses analysis proxies.

## Native video preparation for the lab

Browsers can reject a source before WebGPU sees any pixels. Prepare a local frame
bundle with the native decoder when the source codec or dimensions exceed browser
video limits:

```sh
python -m pip install opencv-python-headless
python scripts/prepare-sfm-video.py input.mkv frames.json --frames 8 --max-edge 512
```

Select `frames.json` in the SfM lab and prepare the analysis frames. PNG encoding is
lossless after resizing. The CLI keeps one decoded source frame at a time; native
decoder memory still grows with the source frame size and codec reference frames.
Source dimensions are not limited by the proxy's 4,096-pixel analysis edge bound.
The browser import is limited to 64 MiB of encoded data and 64 megapixels of total
analysis images. Sampling eight frames is not reconstruction of every video frame.

The bundle records source dimensions and timestamps. Camera intrinsics account for
separately rounded x/y scaling. Exported reconstruction JSON contains analysis
`camera`, restored `sourceCamera` using pixel-center resize inversion, and source
metadata. The 512-pixel default is an analysis quality choice, not 18K feature
extraction. Increase it deliberately or use the full-resolution tiled API.

Uploaded videos use Mediabunny demuxing and browser frame decoding in presentation
order, including variable frame timestamps. The keyframe gate (`website/src/lib/keyframe-gate.ts`) chooses the frames the SfM
keeps. The query `?gate=phase|legacy` selects it, and the run record names it
(`options.keyframeGate`, `options.keyframes`).

- `phase` (default): before SIFT, phase correlation of a 128 × 128 grayscale thumbnail
  (Hann window, sub-pixel peak) gives the global translation between the last keyframe
  and the candidate. A frame is a keyframe when the translation reaches 4 % of the
  frame, and is skipped otherwise. A skipped frame costs no features and no matching.
  The first and last frame are always keyframes. A keyframe follows one second after
  the last one at the latest, and no keyframe follows sooner than 0.2 s. A weak
  correlation peak (peak over mean under 20) leaves the decision to the `legacy` rule.
- `legacy`: every sampled frame is analyzed for features and matched against the last
  keyframe. A frame becomes a keyframe when the median match displacement reaches 8
  analysis pixels (at a 640-pixel width) or one second has passed since the last
  keyframe; other frames are skipped and counted.
  There is no whole-video frame ceiling. Backpressure waits for the current GPU result.
  Map snapshots reach the page at most every 500 ms, and the detailed stage log keeps
  the last 400 entries beside per-operation totals. A 640-pixel analysis edge bounds feature extraction.
  The source video follows the decoded timestamp; stage counts, candidate matches,
  real optical-axis depth observations and throughput remain visible. The prepared
  JSON path retains its existing 256-frame / 64 MiB / 64-megapixel import budgets.

IncrementalSfmSession uses a rolling local optimization window (32 by default,
8–64 configurable) and archives all registered cameras and point observations.
When anchors move, their existing world poses fix the local gauge; real shared
observations carry connectivity across eviction and undergo normal verification.
Track indexing and triangulation rebuild within the window. Historical output
and UI snapshot costs grow with the sequence. Final shared-focal refinement is
local; it does not globally correct old poses or close loops. The viewer preserves
orbit controls and links source RGB points, selected photo planes, depth colors
and consecutive-frame camera motion through the full processed-frame timeline.

The active model replaces its historical observations at each snapshot. Rejected
active observations and their feature identities are removed; observations outside
the window stay archived. A local track split can therefore produce separate points
without reconnecting through a rejected feature. The final observation evaluation
also applies its masks when BA is disabled, drops points with fewer than two retained
observations, and measures the mean over the retained active points. A second
evaluation is needed only when dropping a point changes that mean's population.
The final snapshot and JSON export carry the filtered observations. For a sequence
that exceeds the window, the reported mean still describes the active model, not
a new global evaluation or optimization of the archive.

## Submission and feature lifetime

The lab retains three recent frames and up to three reference frames (one initial
reference and two references spaced at least 0.5 seconds apart). Matching only
uses frames still in the active mapper window. The initial reference survives
window eviction until a stable map exists, allowing a longer baseline than
adjacent video frames.
It computes mutual BRIEF matches on the GPU and releases a feature set after its
last candidate pair. Keypoints are read once for the existing mapper. This bounds
live descriptor storage by the temporal window; CPU map/track storage still grows
with the number of views and landmarks.

`maxDispatchesPerSubmission` groups bounded geometry steps. Its default is 1 for
compatibility and its allowed range is 1–64. Readback callbacks force a submission
boundary before a generator can consume their results. Solver state remains
resident between submissions. The bounded SVD kernel checks convergence at the
same four-sweep checkpoints as before.

A successful lab run retains its worker and GPU device so later runs reuse compiled
pipelines. Cancel, timeout, input replacement, fatal failure and component teardown
terminate the worker. First-use shader compilation remains part of cold latency.

The lab uses `solverCheckpoints: true`, `boundedSubmissions: false` and a 32-model
batch cap. It groups work until an actual algorithm readback is required and skips
only kernels whose existing terminal flag makes them no-ops. BA, PnP and two-view
iteration ceilings and tolerances are unchanged. Fully fenced bounded submissions
remain available for callers that need them. Scheduling progress `complete` means
submission in the unbounded mode; final solver results await numerical readback.

## Measuring progress

Run `scripts/benchmark-sfm.mjs` with `TYPECV_HARDWARE_GPU=1` and, where needed,
`TYPESHADE_CHROMIUM` pointing to installed Edge/Chromium. Reports include adapter
identity, a fixture hash, cold/warm samples, output quality and TypeShade execution
counts. Compare like cache conditions and keep proxy dimensions explicit.

`applicationReadbackBytes` counts requested application elements, while
`residentSnapshotBytes` counts full arrays returned by TypeShade `Resident.read()`.
The installed TypeShade API reads an entire resident snapshot, including unused
capacity; neither field is a physical memory-bus counter. A WebGPU backend label
does not establish GPU utilization or a speedup by itself.

The final controlled comparison on RTX 2080 retained identical four-view/40-landmark
geometry and a mean reprojection error of about 1.03e-12 pixels. The previous bounded
worker configuration averaged 6,960.5 ms warm (3 runs); the checkpoint configuration
averaged 388.4 ms warm (5 runs), approximately 17.9× faster. Dispatches fell from
4,080 to 721 and submissions from 4,071 to 48. These numbers describe the prepared
40-point mapping fixture, not a complete 4K/8K/32K video workload. The feature
frontend and native decoding have separate timings.

Cold latency was still about 34.2 seconds (previously about 43.1 seconds in bounded
mode); the 10× target is **not** met for first-use latency. The installed generated
TypeShade host-entry path lazily creates pipelines in serial group order and has
no public preparation hook. That is a separate upstream optimization opportunity,
not proof that all cold latency is compilation. Keep cold, warm and worker-reuse
measurements separate.

The recorded adapter, samples, quality, dispatch counts and tested source extents
are in `docs/experiments/gpu/sfm-scaling-rtx2080.json`. Hardware tests also process
an entire procedural 8,192 × 4,096 image in 162 tiles and a 32,768 × 48 image in
69 tiles without a full-image buffer. This is image feature extraction, not a
full-resolution video reconstruction throughput measurement.

## Subsequent architecture measurements

The newer eight-JPEG workload and persistent GPU PCG implementation are described
in [GPU SfM research and priorities](gpu-sfm-research.md), with machine-readable
evidence in `docs/experiments/gpu/sfm-resident-cg-rtx2080.json`. At identical
8-view/726-point/1,895-observation coverage, warm processing improved from 6.00 s
to 2.41 s (2.49x); its separate fixed-input BA profile improved 9.19x. These are
different workloads from the 40-point mapper fixture above. Neither establishes
100x end-to-end speed or full COLMAP quality parity.
