# Dense multi-view stereo after the sparse SfM: design

Status: proposed, 2026-10-20. The owner approved the writing of this design. No step is
implemented. Implementation waits for the owner's approval of this document. Each step in
"Steps" below is its own pull request with its own evidence.

This document is written for an engineer or an agent who has not read the conversation that
led to it. It names files, symbols, buffer layouts in bytes, kernels, tests and the numbers
that accept each step. A name marked _(new)_ does not exist yet, and so does every name in a
list of proposed changes. Every other path, symbol and option of this repository named here
exists on `main` at `64eafc0`. The compiler's paths are those of its pin, `e0bf150`.

## 1. Problem and goal

The lab (`/lab/`, `website/src/components/SfmLab.tsx`) runs the incremental sparse SfM in a
worker (`website/src/workers/sfm.worker.ts`) and shows its landmarks in a three.js viewer
(`website/src/components/SfmViewer.tsx`). A sparse map holds a few hundred to a few thousand
points. It shows the cameras well and the surfaces poorly.

Measured on 2026-10-20, headless Chromium on SwiftShader, 640 px analysis frames, the lab's
defaults, gerrard-hall (12 photographs), against COLMAP's cameras:

| Front end                                   | Landmarks | Median rotation error |
| ------------------------------------------- | --------- | --------------------- |
| GPU SIFT, first octave 0 (`main`)           | 724       | 0.216°                |
| GPU SIFT, first octave -1 (pull request 84) | 2,296     | 0.019°                |
| The dataset's own SIFT features             | 7,781     | 0.000°                |

The octave-0 row is in the measurement comment on pull request #80 (724 landmarks, 2,107
observations, 0.216°). The octave -1 row and the dataset row are the measurements reported
for pull request #84, which is open. Its option `firstOctave` is not on `main` yet.

Goal: after the final bundle adjustment, compute one depth map and one normal map for each
registered view, fuse them into a colored point cloud with normals, and show that cloud in
the lab beside the sparse one. At 640 px with 12 views the dense cloud is expected to hold
in the order of 10^5 points (inference, section 6). The work runs on the same `ShadeDevice`
as the SfM, under the same time budget, and on the CPU tier with the same kernels.

Out of scope: meshing, texturing, learned depth, and a camera model other than pinhole with
k1, k2 (`docs/engine-v2.md`, "What this does not decide").

## 2. Method choice

Three families were considered.

**Plane sweep** (Collins 1996, Gallup 2007). For each reference pixel, test D depth planes and
keep the plane of least cost. A cost volume of W × H × D values is the usual form. At
640 × 480 with D = 256 planes that volume is 315 MB in f32. A volume-free form keeps only the
best plane per pixel, but then each pixel tests all D planes in each pass, and it gives no
normal: fronto-parallel windows lose accuracy on slanted surfaces such as a facade seen at an
angle.

**Learned MVS** (MVSNet 2018 and its successors). A 3D CNN regularizes a cost volume. The
volume and the feature maps need hundreds of MB at 640 px. The weights must be shipped. The
kernels need workgroup memory and tensor operations that TypeNN has and TypeCV does not use.
`docs/engine-v2.md` keeps the TypeNN path a direction, not a step.

**PatchMatch stereo** (Bleyer, Rhemann and Rother 2011). Each pixel holds one hypothesis, a
depth and a normal, that is a slanted plane. Hypotheses spread to neighbours (propagation) and
are perturbed at random (refinement). Schönberger, Zheng, Pollefeys and Frahm (ECCV 2016,
COLMAP's `patch_match_stereo`) add a photometric pass and a geometric pass. The geometric pass
adds a forward-backward reprojection term against the depth maps of the source views. It then
filters and fuses. Galliani, Lasinger and Schindler (Gipuma, ICCV 2015) replace COLMAP's row
and column sweeps by a red-black checkerboard: all pixels of one color update in parallel,
reading only pixels of the other color.

**Decision: PatchMatch with a red-black checkerboard (Gipuma's propagation), COLMAP's
photometric and geometric passes, and a per-pixel fusion.** The reasons:

1. **Memory.** The state is a few values per pixel: depth, normal and cost. No cost volume
   exists. Section 4 gives 57.4 MiB at 640 × 480 with 12 views and 229.7 MiB at 1280 × 960,
   against the lab's working budgets of 64 MiB and 256 MiB.
2. **Per-pixel independence.** In a red-black pass a pixel reads the other color and writes
   only its own slots. That is one invocation per element, with no workgroup memory, the form
   of the SIFT kernels (`src/gpu/sift.shade.ts`). The CPU tier runs the same kernels.
   COLMAP's sweep is sequential along each row or column, so W or H dependent steps per
   iteration. That form does not fit one invocation per element.
3. **Chunking.** A color pass splits at any pixel boundary with no change of result, as
   `triangulateTracks` splits at track boundaries (`docs/gpu-sfm.md`, "Triangulation"). Each
   pixel does a fixed amount of work: a fixed number of hypotheses, sources and window
   samples, with no loop to convergence. The time budget then sizes the chunks from a rate in
   ms per unit (`ShadeDevice.chunkSize`).

What is not taken from COLMAP: its per-pixel probabilistic source selection. That selection
keeps a probability per pixel and per source (K f32 per pixel) and runs it along the sweep
direction. This design aggregates the best `bestSources` of the `maxSources` source costs per
hypothesis instead, as Gipuma does. Section 9 lists the accuracy question that this raises.

## 3. Pipeline

The stages run in this order, after `session.finish()` in the worker returns the final model.
All of them run on the device except stage 3.1 and the small host packing.

### 3.1 Input and normalization (host)

The input is the final sparse model in the shape of `Scene` (`website/src/lib/sfm-scene.ts`):
views with world-to-camera `pose.R` (row-major) and `pose.t`, intrinsics
`fx, fy, cx, cy, k1, k2`, and landmarks with `xyz` and `observations`. The input also holds
the RGBA pixels of each registered view at the analysis size.

The host moves the world to the map's centroid and scales it by the map's radius, as decision
3 of `docs/engine-v2.md` does for f32 numerics. Every depth and point below is in these
normalized units. The host maps the fused points back to the model's frame before it posts
them.

### 3.2 View selection (host)

For each reference view r and each other registered view s, the score is:

score(r, s) = Σ over landmarks p seen by both r and s of g(α_p),
g(α) = min(1, (α / α0)²) for α ≥ αmin, and 0 below αmin.

α_p is the angle at p between the rays to the two camera centers. α0 = 5° and αmin = 1°. A
pair with fewer than 16 shared landmarks scores 0. The `maxSources` (default 6) best sources
with a positive score form the source list of r. A view with no source gets no depth map and
is reported in the result. This uses only the sparse model, which the host holds.

Inference: the score prefers sources with many shared points and a useful baseline, which is
the intent of COLMAP's automatic source selection. The exact formula is this design's choice.

### 3.3 Depth range and seeds (host)

For each reference view the host takes the optical-axis depths of the landmarks it observes
(`depthPoints` in `website/src/lib/sfm-scene.ts` computes the same depth). The range is
[0.75 × the 1st percentile, 1.25 × the 99th percentile]. A view with fewer than 8 observed
landmarks gets no depth map.

Each observation (pixel, depth) is a seed: the pixel at the rounded observation starts with
the landmark's depth and a normal that faces the camera. Seeds give propagation correct
starting points. Inference from the counts of section 1: with 2,296 landmarks and about 3
observations each, a view holds about 575 seeds, about one in every 23 × 23 pixels at
640 × 480. With 724 landmarks the spacing is about 41 pixels.

### 3.4 Undistortion (device, `mvsUndistort` _(new)_)

The existing `rgbaToGray` (`src/gpu/features.shade.ts`) converts each view's packed RGBA to
f32 gray in [0, 1]. `mvsUndistort` then writes the undistorted gray image with the same
fx, fy, cx, cy and zero distortion. For output pixel (u, v) it computes x = (u − cx) / fx,
y = (v − cy) / fy, r² = x² + y², and the distorted position
(fx · x · (1 + k1 r² + k2 r⁴) + cx, fy · y · (1 + k1 r² + k2 r⁴) + cy). That is the forward
model of `project` in `src/gpu/numerics.shade.ts`, so no iteration is needed. It samples the
input bilinearly. An output pixel whose source position falls outside the image, or whose r²
is at or past `radialLimit`, gets −1, the "no data" mark. The pinhole views of the pyramid and
TUM samples (k1 = k2 = 0) skip the kernel.

### 3.5 Photometric initialization (device, `mvsInitialize` and `mvsSeed` _(new)_)

`mvsInitialize` gives each pixel of the reference a random depth, uniform in inverse depth
over the range, and a random unit normal in the hemisphere that faces the camera. It then
computes the pixel's cost (3.7). `mvsSeed` writes the seeds over the random values and
computes their costs.

Randomness is a counter-based hash of (seed, view, pixel, iteration, draw), as `hash` in
`src/gpu/features.shade.ts`. No random state is stored. The CPU tier and WebGPU therefore draw
the same numbers, and a chunk boundary changes no draw.

### 3.6 Propagation and refinement (device, `mvsPropagate` _(new)_)

One iteration is two passes: all red pixels ((x + y) even), then all black pixels. A pixel
evaluates 10 hypotheses and keeps the one of least cost:

- 8 propagation candidates: the hypotheses of the pixels at (±1, 0), (0, ±1), (±5, 0) and
  (0, ±5). Each offset has an odd |dx| + |dy|, so every candidate has the other color. A
  candidate plane is moved to the pixel: its depth is the depth where the pixel's ray meets
  the candidate's plane.
- 2 refinement candidates: a new random depth and normal, and the current hypothesis
  perturbed. The perturbation is ±(range × 2^−iteration) in inverse depth and up to
  30° × 2^−iteration in normal.

The pass reads the other color and writes only the current color's depth, normal and cost.
So the pass can run in place, with no second buffer, and any split of a pass into chunks
gives the same result bit for bit.

A lane past its chunk does not return before the loops. It takes the chunk's last pixel, runs
the same control flow and writes nothing. This is the clamp-and-mask form that removed the
device hang of `docs/hardware-observations.md`, A1. The other kernels of this design follow
the same rule.

Default iterations: `photometricIterations` 8 and `geometricIterations` 4. Inference: the
red-black pass moves information at most 5 pixels per color pass, so 10 pixels per iteration.
Eight iterations reach 80 pixels, more than the seed spacing of section 3.3. Step 2 measures the
error after each iteration on the synthetic scene and confirms or changes the default.

### 3.7 Cost

For a hypothesis (depth d, normal n) of reference pixel p and a source s, the plane is
nᵀX + δ = 0 in r's camera frame, with X_p = d · K_r⁻¹ p̃ and δ = −nᵀX_p. The homography is
H = K_s (R_rs − t_rs nᵀ / δ) K_r⁻¹, with (R_rs, t_rs) the pose of s relative to r.

The photometric cost is 1 − NCC over a window of radius `windowRadius` (5) sampled at step
`windowStep` (2): offsets −5, −3, −1, 1, 3, 5 in each axis, 36 samples. Each sample reads the
reference at an integer pixel and the source bilinearly at H p̃. The cost lies in [0, 2]. A
window with a reference variance below 1e-6, or with a source sample marked "no data" or
outside the image, costs 2 for that source. A source whose triangulation angle at X_p is
below 1° costs 2 for that source.

The aggregated cost of the hypothesis is the mean of the `bestSources` (3) least source costs
out of the `maxSources` (6).

Bilateral weights, as COLMAP uses (a spatial and a color Gaussian per sample), are an option,
`bilateral` _(new)_, off by default. Step 2 measures their effect on the synthetic scene.

COLMAP's own defaults are a window radius of 5 and a step of 1 (121 samples). The step of 2
here cuts the samples by 3.4 times. Step 2 compares the two steps on the synthetic scene.
COLMAP's default values in this document are quoted from its `PatchMatchOptions` and
`StereoFusionOptions` from memory. Step 1 confirms them against `colmap patch_match_stereo
--help` and `colmap stereo_fusion --help` and corrects this document where they differ.

### 3.8 Geometric consistency (device, `mvsPropagate` with `geometric: 1`)

After every view has its photometric depth map, each reference runs `geometricIterations`
more iterations, starting from its photometric depth and normal. The cost of each source gains
a term (Schönberger 2016):

cost_s = (1 − NCC_s) + λ · min(ψ_s, ψmax), λ = 0.3, ψmax = 3 px.

ψ_s is the forward-backward reprojection error. Project X_p into s at q. Read the source's
photometric depth at the nearest pixel of q. Back-project that point and project it into r at
p′. Then ψ_s = |p − p′| in pixels. The pass reads the photometric depth maps of the sources
and writes the geometric depth map of the reference. So the order of the references changes
no result.

### 3.9 Filtering (device, `mvsFilter` _(new)_)

A pixel of the geometric depth map is kept when all of these hold:

- the photometric cost of its hypothesis, averaged over its best sources, is at most 0.9
  (NCC at least 0.1)
- at least 2 sources have ψ_s ≤ 1 px and a triangulation angle at X_p of at least 3°
- its depth lies inside the view's range

A rejected pixel gets depth 0, which means "no depth". The values are those of COLMAP's
filter options (`filter_min_ncc` 0.1, `filter_min_num_consistent` 2,
`filter_geom_consistency_max_cost` 1, `filter_min_triangulation_angle` 3), subject to the
check of 3.7.

### 3.10 Fusion (device, `mvsFuseFlags` and `mvsScatterPoints` _(new)_)

COLMAP's `stereo_fusion` grows clusters of consistent pixels from a queue. That is sequential.
This design fuses per pixel instead. For each pixel p of reference r with depth d > 0:

1. Back-project X = the point at depth d on p's ray, in world coordinates.
2. For each other view j with a depth map, project X into j at q. Read d_j at the nearest
   pixel of q. View j is consistent when all of these hold:
   - the relative depth error |z_j(X) − d_j(q)| / d_j(q) is at most 0.01
   - the point of j at q, projected back into r, lands within 2 px of p
   - the angle between the two normals is at most 10°
3. Let c be 1 plus the count of consistent views. Emit a point when c ≥ `minConsistentViews`
   (3) and no consistent view has an index lower than r.
4. The point is the mean of X and the consistent views' points. The normal is the normalized
   mean of their normals. The color is the reference pixel's RGBA.

The rule "no consistent view with a lower index" makes each surface element come from one
view in most cases. Inference: duplicates remain where consistency is not symmetric, for
example at depth edges. Step 5 counts them on the synthetic scene. The thresholds 0.01, 2 px
and 10° are COLMAP's `max_depth_error`, `max_reproj_error` and `max_normal_error` defaults,
subject to the check of 3.7.

`mvsFuseFlags` writes one flag per pixel. The shared flag scan (`planFlagScan`,
`scanBlockCount` in `src/gpu/scan.ts`) gives each emitted pixel its slot.
`mvsScatterPoints` writes the points. The host reads one count and the points of that
reference.

### 3.11 Colors

The color of a point is the RGBA of its reference pixel at the analysis size, after
undistortion with the same map as the gray image. The host already holds the RGBA of each
view (5.3). The fusion stage uploads the reference's packed RGBA, and `mvsUndistort` in its
RGBA mode writes it to `mvsRgba`, sampling each channel bilinearly. The array is released
after the reference's pass. Averaging the colors of all consistent views needs the RGBA of every view
resident. That costs 4 B per pixel per view and is not part of this design.

## 4. Data layout and memory

### 4.1 Resident arrays

P = W × H pixels per view, N views with a depth map. Pixel index i = y × W + x. A per-view
array is view-major: element v × P + i. All arrays are TypeShade arrays allocated through
`ShadeDevice` (`allocate`, `allocateU32`), so `reserveWorkingBytes` admits them against
`workingBytes`.

| Array _(new)_    | Element                                                                            | Count      | Bytes per pixel |
| ---------------- | ---------------------------------------------------------------------------------- | ---------- | --------------- |
| `mvsGray`        | f32, undistorted gray in [0, 1], −1 for "no data"                                  | N × P      | 4 per view      |
| `mvsDepth`       | f32, photometric depth (optical-axis z, normalized units), 0 for "no depth"        | N × P      | 4 per view      |
| `mvsGeoDepth`    | f32, geometric depth, then the filtered depth                                      | N × P      | 4 per view      |
| `mvsNormal`      | u32, octahedral unit normal in the view's camera frame, `pack2x16snorm`            | N × P      | 4 per view      |
| `mvsCost`        | f32, the cost of the current hypothesis of the reference                           | P          | 4               |
| `mvsCameras`     | f32 × 24 per view: R (9), t (3), fx, fy, cx, cy, W, H, dmin, dmax, center C (3), 0 | N          | 96 B per view   |
| `mvsSources`     | u32 × 7 per view: source count, then up to 6 view indices                          | N          | 28 B per view   |
| `mvsSeeds`       | f32 × 3 per seed: x, y, depth                                                      | seeds      | 12 B per seed   |
| `mvsRgba`        | u32, packed RGBA of the reference (fusion stage only)                              | P          | 4               |
| `mvsFlags`       | u32, the fusion flag of each pixel                                                 | P          | 4               |
| `mvsFlagOffsets` | u32, the scan's block sums                                                         | 2 × ⌈P/64⌉ | about 0.13      |
| `mvsPoints`      | f32 × 3 per point: position, world                                                 | P          | 12              |
| `mvsPointData`   | u32 × 2 per point: normal (octahedral, world), RGBA                                | P          | 8               |

The normal uses 4 bytes in place of 12. The octahedral map with two 16-bit components has an
angular error below 0.01° (inference from the 2^−15 step). TypeShade's CPU runtime implements
`pack2x16snorm` and `unpack2x16snorm` (`src/core/cpu-runtime.ts` of the compiler), so both
tiers agree. Random state takes 0 bytes (3.5). A stored u32 state per pixel would add 4 B per
pixel.

### 4.2 Totals for N = 12

The depth stage holds `mvsGray`, `mvsDepth`, `mvsGeoDepth`, `mvsNormal` and `mvsCost`:
(16 N + 4) bytes per pixel. Before fusion, `mvsGray`, `mvsDepth` and `mvsCost` are recycled. The fusion
stage holds `mvsGeoDepth`, `mvsNormal`, `mvsRgba`, `mvsFlags`, `mvsFlagOffsets`, `mvsPoints`
and `mvsPointData`: about (8 N + 28) bytes per pixel. The small arrays (`mvsCameras`, `mvsSources`,
`mvsSeeds`) take less than 0.2 MiB.

| Quantity                            | 640 × 480 (P = 307,200) | 1280 × 960 (P = 1,228,800) |
| ----------------------------------- | ----------------------- | -------------------------- |
| One per-view f32 or u32 image       | 1.17 MiB                | 4.69 MiB                   |
| One per-view array × 12 views       | 14.06 MiB               | 56.25 MiB                  |
| Depth stage, 196 B per pixel        | 57.42 MiB               | 229.69 MiB                 |
| Fusion stage, about 124 B per pixel | 36.36 MiB               | 145.46 MiB                 |
| Lab `workingBytes` (worker)         | 64 MiB                  | 256 MiB                    |
| Headroom in the depth stage         | 6.58 MiB                | 26.31 MiB                  |
| Host RGBA of 12 views (5.3)         | 14.7 MB                 | 59.0 MB                    |

gerrard-hall's previews are 640 × 427 (P = 273,280). Its depth stage takes 51.08 MiB.

The worker scales `workingBytes` and `poolBytes` with the square of the analysis edge
(`start` in `website/src/workers/sfm.worker.ts`: 64 and 32 MiB at 640 px, 256 and 128 MiB at
1280 px). The bytes per pixel of the depth stage are the same at both sizes, so the largest N
is the same: N ≤ ⌊(workingBytes / P − 4) / 16⌋ = 13 views at 640 × 480 and at 1280 × 960, and
15 views at 640 × 427.

Each array must also fit `maxStorageBufferBindingSize`, which `ShadeDevice.bindingBytes`
reads (128 MiB when the adapter reports none). The largest array at 1280 × 960 with 12 views
is 56.25 MiB. It passes 128 MiB at N = 28.

`maxKernelElements` is 65,535 × 64 = 4,194,240 elements per dispatch. A color pass at
1280 × 960 has 614,400 pixels, so it fits in one dispatch.

### 4.3 What stays resident, and when

| Stage                 | Resident                                                                       | Released at the end                  |
| --------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| Undistortion          | `mvsGray`, one view's packed RGBA as input                                     | the RGBA input                       |
| Photometric, per view | all of the depth stage                                                         | nothing                              |
| Geometric, per view   | all of the depth stage                                                         | `mvsGray`, `mvsDepth`, `mvsCost`     |
| Fusion, per view      | `mvsGeoDepth`, `mvsNormal`, `mvsRgba`, `mvsFlags`, `mvsPoints`, `mvsPointData` | everything, after the last reference |

Nothing is read back between kernels except the fused point count and points of each
reference, and the optional depth previews (5.3).

The SfM's arrays must be released before the MVS starts. The worker's `releaseFeatures`
releases the descriptors after `session.finish()`. Inference: the mapper's other arrays are
released when `finish()` returns. Step 6 checks `device.statistics.liveBytes` before the
first MVS allocation and records it.

South-building-24 (24 views) does not fit the all-resident layout at either size. Section 9
lists the windowed layout that it needs.

## 5. Scheduling

### 5.1 Kernels as chunks of the time budget

Every kernel that loops over pixels runs as chunks through `device.planSteps`, as
`triangulateTracksGpu` does (`src/gpu/triangulation.ts`). A chunk is a contiguous range of
color indices, `first..end`, in a uniform `range`. For color c of a W × H image, color index
k maps to row y = ⌊k / ⌈W/2⌉⌋, column x = 2 (k mod ⌈W/2⌉) + ((y + c) mod 2). A lane with
x ≥ W, or past `end`, is idle and masked (3.6).

The host sizes each chunk with `device.chunkSize(key, remaining, 64)`, where 64 is one
workgroup of pixels. Each chunk step carries `warm`, its entries over zero workgroups, so the
pipeline is created outside the first timed chunk (`docs/gpu-sfm.md`, "The time budget").

| Chunk key _(new)_       | Kernel                      | One unit                   | Work per unit                     |
| ----------------------- | --------------------------- | -------------------------- | --------------------------------- |
| `mvsInitialize`         | `mvsInitialize`             | one pixel × one source     | 1 hypothesis × 36 samples         |
| `mvsPropagate`          | `mvsPropagate`, photometric | one pixel × one source     | 10 hypotheses × 36 samples        |
| `mvsPropagateGeometric` | `mvsPropagate`, geometric   | one pixel × one source     | 10 × (36 samples + 2 projections) |
| `mvsFilter`             | `mvsFilter`                 | one pixel × one source     | 2 projections                     |
| `mvsFuse`               | `mvsFuseFlags`              | one pixel × one other view | 2 projections                     |

The unit includes the source count, so the rate a view with 6 sources teaches sizes the chunks
of a view with 4. The two `mvsPropagate` variants have their own keys because their cost per
unit differs, as `bundleCg` and `bundleCachedCg` do. `mvsUndistort`, `mvsSeed`, the scan and
`mvsScatterPoints` do a few operations per pixel. Each runs as one plan with no chunk.

The CPU tier runs each key as one chunk (`chunkSize` returns `max` there), so its results are
those of the whole pass.

### 5.2 Dispatch counts and relative cost

With K sources and T iterations, one reference view takes these units:

| Stage       | Units per reference        | 640 × 480, K = 6, T = 8 or 4 |
| ----------- | -------------------------- | ---------------------------- |
| Initialize  | P × K                      | 1.84 M                       |
| Photometric | 2 passes × T × (P / 2) × K | 14.7 M                       |
| Geometric   | 2 passes × T × (P / 2) × K | 7.4 M                        |
| Filter      | P × K                      | 1.84 M                       |
| Fusion      | P × (N − 1)                | 3.38 M                       |

In window samples, the main measure of work, one reference at 640 × 480 takes about
0.07 G samples to initialize, 5.3 G in the photometric pass and 2.7 G in the geometric pass.
For 12 views that is about 97 G samples. Inference from these counts: the photometric pass
takes about 65 % of the time, the geometric pass about 33 %, and initialization, filtering
and fusion together less than 3 %. At 1280 × 960 each count is four times larger.

A color pass is c chunks, with c = ⌈units of the pass × rate / (0.8 × `timeBudgetMs`)⌉ once
the rate has settled. At 640 × 480 with K = 6, one color pass is 921,600 units. One chunk per
pass needs a rate of at most 0.217 µs per unit, so about 1.7 G samples per second. Inference,
not measured: a physical desktop GPU reaches this, and SwiftShader does not. On SwiftShader a
full 640 px run is a long job, so CI runs the small synthetic scenes of section 7 and the full
samples run in the lab. Step 3 measures the SwiftShader rate of each key, and the lab's run
records give the physical rates.

Fences: each key is timed for its first 8 chunks (`CHUNK_TIMING_SAMPLES` in
`src/gpu/device.ts`), with two fences each. The later chunks run untimed. So the five keys add
at most 80 fences to a run, whatever its size.

### 5.3 The worker

Proposed changes to `website/src/workers/sfm.worker.ts`:

1. `Request` of type `start` gains `dense?: boolean`. The page sets it from `?dense=1` first,
   and later from a switch.
2. `LAB_OPTIONS` gains a `dense` record with the defaults of section 3: `maxSources` 6,
   `bestSources` 3, `windowRadius` 5, `windowStep` 2, `photometricIterations` 8,
   `geometricIterations` 4 and `minConsistentViews` 3. The result carries `LAB_OPTIONS`, so a
   run record carries these too.
3. While `dense` is set, `processFrame` and `processPrecomputedFrame` keep each frame's
   `pixels` in a map before they clear `frame.pixels`. A frame that is not a keyframe, or a
   view that does not register, leaves the map. The map costs 4 B per pixel per view on the
   host (section 4.2).
4. `finish` posts the sparse `result` first, as now. It then keeps the job and runs
   `reconstructDenseGpu` _(new)_ on the same device.
5. During the MVS the worker posts, with the arrays transferred, not copied:
   - `{ type: 'denseProgress', stage, viewId, done, total }` after each reference of each
     stage
   - `{ type: 'denseDepth', viewId, width, height, depth }` after a view's filtered depth map,
     with `depth` a Float32Array at a quarter of the size in each axis (19,200 values at
     640 × 480)
   - `{ type: 'densePoints', viewId, positions, colors, normals }` after the fusion of each
     reference: Float32Array × 3, Uint8Array × 3 and Int16Array × 2 per point
   - `{ type: 'denseResult', points, views, timing }` at the end
6. A request `{ type: 'cancelDense', id }` sets a flag that the MVS reads between chunks. A
   chunk in flight completes. The worker then releases the MVS arrays and posts `denseResult`
   with `cancelled: true`.
7. A failure in the MVS posts an error that names the stage and the kernel, as `failJob` does.
   The sparse result has already been posted and stays on the page.

The page thread does no MVS work. It receives at most one message per reference and stage.

The run record (`website/src/lib/run-record.ts`, format `typecv-run-record-v4`) gains an
optional `dense` block: options, views with a depth map, points, elapsed time per stage, and
the chunk keys of 5.1 under `timing.chunks`. Adding a block is a format change, so the step
that adds it moves the format to v5 and keeps v4 in `RUN_RECORD_FORMATS`, as v4 kept v3.

### 5.4 The API

`reconstructDenseGpu(device, scene, images, options)` _(new)_ in `src/gpu/mvs.ts` _(new)_,
with the kernels in `src/gpu/mvs.shade.ts` _(new)_ and the host view selection in
`src/gpu/mvs-views.ts` _(new)_. `src/gpu/index.ts` exports it. `images` maps a view id to its
RGBA, width and height. `options.onView` receives the depth previews and points of 5.3. The
result holds the points, the per-view status (no source, too few landmarks, done) and the
plan report. A new export of `typecv/gpu` changes the public API, so its step waits for the
owner's approval.

## 6. Viewer and page

Proposed changes to `website/src/components/SfmViewer.tsx` and
`website/src/components/SfmLab.tsx`:

- **A second point cloud.** The viewer gets an optional prop `dense` with `positions`
  (Float32Array), `colors` (Uint8Array) and `count`. It draws a second `THREE.Points` with its own
  `BufferGeometry`. Colors use a normalized `Uint8BufferAttribute`, 3 B per point instead of
  the 12 B of the sparse cloud's `GrowingAttribute`. The cloud grows as `densePoints`
  messages arrive.
- **A toggle.** A switch "Dense" in `sfm-map-options` beside "Cameras" and "Photo". It is
  disabled until the first `densePoints` message arrives. When it is on, the sparse cloud is
  hidden and the dense cloud is shown. Picking (`onSelectPoint`) stays on the sparse
  landmarks, because a raycast over 10^6 points on each pointer move is too slow (inference).
- **A point budget.** `densePointBudget`, 1,000,000 by default, 300,000 when
  `navigator.maxTouchPoints > 0`. Over the budget the viewer draws every k-th point, with k
  the smallest integer that fits. The budget limits drawing only, not the export. At
  1,000,000 points the attributes take 15 MB.
- **Export PLY.** The existing `download('ply')` writes ASCII. For the dense cloud it writes
  `binary_little_endian 1.0` with x, y, z (float), nx, ny, nz (float) and red, green, blue
  (uchar): 27 B per point, 27 MB for 10^6 points. ASCII would be about 40 MB for the same
  points.
- **Depth previews.** The photo overlay can show the `denseDepth` map of the selected view
  with `depthColor` (`website/src/lib/sfm-scene.ts`), in place of the sparse depth points.

Expected size, inference: with half the pixels kept by the filter and each surface element
seen consistently in about 4 views, 12 views at 640 × 480 give about
12 × 307,200 × 0.5 / 4 ≈ 460,000 points, and four times that at 1280 × 960. Step 5 and the
lab runs replace this estimate with measured counts.

## 7. Verification plan

### 7.1 Synthetic textured scene on the CPU tier (CI)

`tests/typeshade/mvs.test.mjs` _(new)_, added to `npm run test:shade`:

- The scene: a textured ground plane, a box and a sphere, with a value-noise texture. A host
  ray caster renders 5 views at 64 × 48 px with known K, R and t, and the exact depth and
  normal maps. The size keeps the CPU-tier run under 60 s. The test records its time.
- Photometric pass, central view: median relative depth error ≤ 1 % on the pixels kept by
  the filter, at least 80 % of the pixels kept, median normal error ≤ 5°.
- Geometric pass and filter: the median relative error is not larger than after the
  photometric pass, and the 95th percentile is smaller.
- Fusion: every emitted point lies within 1 % of the scene radius of the true surface for at
  least 95 % of the points. Duplicates (two points within 0.5 px of each other in their
  reference view) are below 10 %.
- Chunk invariance: one chunk and forced chunks of 64 pixels give the same depth, normal and
  cost arrays bit for bit, as `tests/typeshade/cpu.test.mjs` does for `triangulateTracks`.
- Reference order: running the geometric pass in reverse view order gives the same arrays bit
  for bit (3.8).

### 7.2 The same scene on WebGPU (CI)

`tests/typeshade/mvs-webgpu.test.mjs` _(new)_, added to `npm run test:webgpu` (the `webgpu`
job of `.github/workflows/ci.yml`, SwiftShader). Bit equality with the CPU tier is not
expected, because f32 rounding and fused multiply-add can differ. The test asserts that at
least 99 % of the pixels kept on both tiers agree within 1e-3 relative depth, and that the
accuracy bounds of 7.1 hold. It records the rate of each chunk key.

### 7.3 TUM against ground-truth depth (lab, needs a data change)

The TUM sample (`website/public/sample/tum-fr3-structure-texture-far/`) has 16 frames at
640 × 480 and motion-capture poses. Its README states that the depth maps are not included.

The data change: `scripts/prepare-lab-tum-sample.py` gains a `--depth` option. It copies the
depth image nearest in time to each frame (`depth.txt` of the sequence) as
`depth-NN.png`, 16-bit, 640 × 480, depth = value / 5000 m. Raw, a frame is 614,400 B and 16
frames are 9.8 MB. The PNG size is not measured. The TUM file-format page states that the
freiburg3 depth images are registered to the RGB images. This document has not checked that
statement against the sequence.

The comparison: align the reconstruction's cameras to the reference with `alignPoses`
(`website/src/lib/pose-alignment.ts`). Scale the depths by the alignment's `scale`. Compare
on the pixels where both depths are valid. Targets for the first run: median relative depth
error ≤ 3 %, valid on at least 50 % of the pixels with ground truth. These targets are this
design's proposal. The first measured run sets the gate.

### 7.4 COLMAP samples against COLMAP's dense output (offline)

COLMAP's dense stage needs CUDA. CI has no GPU, so this comparison runs offline on a machine
with an NVIDIA GPU and its record is committed. Procedure:

1. Build a COLMAP workspace from the sample's `sample.json` cameras and the preview images.
2. Run `colmap image_undistorter --output_type COLMAP --max_image_size 640`.
3. Run `colmap patch_match_stereo --PatchMatchStereo.geom_consistency true`.
4. Run `colmap stereo_fusion --input_type geometric`.
5. Run `scripts/compare-dense.py` _(new)_ on COLMAP's `stereo/depth_maps/*.geometric.bin`
   and `fused.ply` and on the lab's exported dense PLY.

The record goes to `docs/experiments/dense-mvs/` _(new)_ as JSON: the COLMAP version, the
GPU, the options, and per view the median relative depth difference, the shared valid
fraction, and the median point-to-cloud distance as a fraction of the scene radius. The depth
maps themselves stay out of the repository: 12 views at 640 × 427 in f32 are 13.1 MB per
sample. Targets for the first run: a median relative depth difference ≤ 2 % where both are
valid, and a dense point count at least 25 % of COLMAP's fused count.

### 7.5 Gates

| Gate                                 | Number                                                                                            | Where                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- | --------------------- |
| Synthetic scene, CPU tier            | Median relative depth error ≤ 1 %, ≥ 80 % kept, chunk invariance bit for bit                      | `npm run test:shade`  |
| Synthetic scene, WebGPU              | ≥ 99 % of kept pixels within 1e-3 of the CPU tier                                                 | `npm run test:webgpu` |
| TUM, ground-truth depth              | Median relative error ≤ 3 %, ≥ 50 % valid (first run sets the gate)                               | lab, record           |
| gerrard-hall vs COLMAP dense         | Median relative difference ≤ 2 % (first run sets the gate)                                        | offline record        |
| Physical devices (collector records) | gerrard-hall dense at 640 px completes on devices A and B, no device loss, longest chunk ≤ budget | lab, run records      |
| Memory                               | `liveBytes` at the depth stage's peak equals the formula of 4.2 within 1 %                        | WebGPU test, lab      |

## 8. Steps

Each step is one pull request, in this order. Each names this document on a line of its own
in its commits.

| Step | Pull request content                                                                                                                       | What it proves                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1    | `src/gpu/mvs-views.ts`: view selection, depth ranges, seeds, normalization. `tests/mvs-views.test.ts`. The COLMAP defaults of 3.7 checked. | Source lists of gerrard-hall from the dataset's features are stable. The defaults in this document are right.                 |
| 2    | `mvsUndistort`, `mvsInitialize`, `mvsSeed`, `mvsPropagate` (photometric) on one reference, CPU tier. `tests/typeshade/mvs.test.mjs`.       | Photometric accuracy on the synthetic scene. Chunk invariance. The defaults of iterations, window step and bilateral weights. |
| 3    | The chunk keys and `warm` steps. `tests/typeshade/mvs-webgpu.test.mjs`.                                                                    | WebGPU agrees with the CPU tier. The SwiftShader rate of each key.                                                            |
| 4    | The geometric pass and `mvsFilter`.                                                                                                        | The error tail shrinks. Reference order changes no result.                                                                    |
| 5    | Fusion, `mvsFuseFlags`, the scan, `mvsScatterPoints`, colors. `reconstructDenseGpu` and its export.                                        | Point accuracy and the duplicate count on the synthetic scene. The memory formula of 4.2.                                     |
| 6    | The worker: `dense`, the kept RGBA, the messages of 5.3, cancel, the run record v5.                                                        | A lab run on gerrard-hall posts depth maps and points without a block of the page. `liveBytes` before the MVS.                |
| 7    | The viewer and the page: the dense cloud, the switch, the point budget, binary PLY.                                                        | The page shows 10^5 to 10^6 points. The PLY loads in a standard reader.                                                       |
| 8    | TUM depth maps in the sample and `--depth` in `scripts/prepare-lab-tum-sample.py`. The comparison of 7.3.                                  | Depth accuracy against independent ground truth.                                                                              |
| 9    | The offline COLMAP comparison of 7.4 and its record.                                                                                       | Agreement with COLMAP's dense output on the COLMAP samples.                                                                   |
| 10   | Physical-device runs and the gate rows of 7.5.                                                                                             | No device loss on devices A and B. The physical rates.                                                                        |

Steps 1 to 5 need no change to the lab. Steps 8 and 9 can run in parallel with steps 6 and 7.

## 9. Risks and open questions

**Pose accuracy.** A rotation error θ moves the projection of a point by about f · θ pixels.
For gerrard-hall's 640 px previews f = 437.4 px:

| Front end                 | θ      | f · θ at 640 px | f · θ at 1280 px |
| ------------------------- | ------ | --------------- | ---------------- |
| GPU SIFT, octave 0        | 0.216° | 1.65 px         | 3.30 px          |
| GPU SIFT, octave -1 (#84) | 0.019° | 0.15 px         | 0.29 px          |

The depth error that follows from a pixel error e is Δz ≈ z² e / (f b) for a baseline b. With
e = f · θ this gives Δz / z ≈ θ / (b / z), where b / z is about the triangulation angle α in
radians. For α = 10°: 0.216° gives a relative depth error of about 2.2 %, and 0.019° about
0.19 %. Inference: at 0.216° the misalignment of 1.65 px is larger than the geometric filter's
1 px (3.9), so the filter would reject many correct pixels, and the photometric windows would
blur depth edges. At 0.019° the misalignment is a fraction of a pixel. So the first octave of
-1 (pull request #84) makes MVS viable at 640 px, and MVS at 1280 px needs it. The rotation
error of section 1 is a median of absolute rotations after alignment. The relative rotation
of a pair can be up to about twice that.

**Tangential distortion.** The gerrard-hall sample drops COLMAP's tangential terms
(`droppedTangential` in its `sample.json`, p1 = 0.000116 and p2 = 0.000295). Inference: at the
image corner they move a pixel by about 0.3 px at 640 px. That error enters the undistortion.

**Device loss on mobile.** Each chunk is bounded by the time budget, and each lane's work is
fixed. The kernels are f32 only, so the flavor of the emulated double
(`docs/hardware-observations.md`, B1) does not apply. Open: no MVS kernel has run on device A
or B. The early-return hang of A1 is avoided by the clamp-and-mask rule, but its cause is not
known, so another loop structure could hang.

**Memory at 1280.** The depth stage takes 229.7 MiB of buffers, and the pool can hold up to
128 MiB more. Open: no record gives the memory a phone's browser grants to WebGPU. Two
fallbacks reduce the peak: gray as 16-bit pairs (`pack2x16unorm`) saves 2 B per pixel per view,
and a windowed layout keeps only a reference and its sources resident.

**More than 13 views.** South-building-24 does not fit the all-resident layout. A windowed
layout uploads each reference's sources and keeps the depth maps on the host or in a bounded
device cache. That is a later step with its own design amendment.

**Textureless regions.** Sky, sand, bare walls and water give a reference variance near 0.
Those windows cost 2 (3.7), so their pixels fail the filter and stay empty. Inference: the
pyramid video, with its sand, gives sparse coverage on the ground. A multi-scale scheme
(ACMM, Xu and Tao 2019) fills such regions better and is an open item.

**Source selection.** The best-3-of-6 aggregation replaces COLMAP's per-pixel source
probabilities. Open: whether it loses accuracy at occlusions. The comparison of 7.4 measures
it.

**Order of the work.** `docs/engine-v2.md`, "Order", item 5 says that dense reconstruction
waits on the scene engine. This design reads only the final sparse model and the frames, not
the resident scene. Decision for the owner: whether the MVS steps start before the scene
engine's steps 3 to 5 of `docs/gpu-resident-scene.md`.

**Page responsiveness.** The worker and the page's WebGL viewer share one GPU. Inference:
chunks of up to 200 ms can delay the viewer's frames during the MVS. The lab's `?budget=`
sets a smaller `timeBudgetMs` for a whole run. A budget for the MVS alone would need a
per-call cap, for example a `max` units from `device.chunkRates`. That is open.
