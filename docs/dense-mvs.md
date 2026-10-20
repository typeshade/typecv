# Dense multi-view stereo after the sparse SfM: design

Status: accepted, 2026-10-20. The owner approved this design and the start of its
implementation. Step 1 (host view selection, `src/gpu/mvs-views.ts`) is implemented. Step 2
(the photometric pass on one reference, CPU tier, `src/gpu/mvs.shade.ts` and `src/gpu/mvs.ts`)
is implemented, and its measurements are in "Step 2 record" under section 8. Step 3 (the
chunk keys and warm steps of `mvsInitialize` and `mvsPropagate`, and
`tests/typeshade/mvs-webgpu.test.mjs`) is implemented, and its measurements are in "Step 3
record". Step 4 (the geometric pass and `mvsFilter`, `mvsDepthMapsGpu` in `src/gpu/mvs.ts` and
`tests/typeshade/mvs-geometric.test.mjs`) is implemented, and its measurements are in "Step 4
record". Step 5 (the fusion, `reconstructDenseGpu` and its export from `src/gpu/index.ts`,
`tests/typeshade/mvs-fusion.test.mjs`) is implemented, and its measurements are in "Step 5
record". Each step in "Steps" below is its own pull request with its own evidence.

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
Eight iterations reach 80 pixels, more than the seed spacing of section 3.3. Step 2 measured the
error after each iteration on the synthetic scene and confirms the default of 8 ("Step 2 record").

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
`bilateral`, off by default. Step 2 measured their effect on the synthetic scene: the median
relative depth error goes from 0.63 % to 0.52 % for 1.28 times the time ("Step 2 record").

COLMAP's own defaults are a window radius of 5 and a step of 1 (121 samples). The step of 2
here cuts the samples by 3.4 times. Step 2 compared the two steps on the synthetic scene: step 1
gives 0.58 % against 0.63 % for 2.8 times the time, so the default stays 2.

Step 1 checked every COLMAP default that this document quotes against COLMAP's source, tag
`4.2.1` (commit `bd1fcf654d2dd8fefa1466999c190a246f83f4b9`), on 2026-10-20: the member
initializers of `PatchMatchOptions` in `src/colmap/mvs/patch_match_options.h` and of
`StereoFusionOptions` in `src/colmap/mvs/fusion.h`. Every quoted value agrees:

| COLMAP option                      | COLMAP default | Used in this document     |
| ---------------------------------- | -------------- | ------------------------- |
| `window_radius`                    | 5              | `windowRadius` 5 (3.7)    |
| `window_step`                      | 1              | `windowStep` 2, by choice |
| `min_triangulation_angle`          | 1°             | 1° (3.7)                  |
| `sigma_color`                      | 0.2            | `bilateral`, off (3.7)    |
| `geom_consistency_regularizer`     | 0.3            | λ = 0.3 (3.8)             |
| `geom_consistency_max_cost`        | 3 px           | ψmax = 3 px (3.8)         |
| `filter_min_ncc`                   | 0.1            | 0.1 (3.9)                 |
| `filter_min_triangulation_angle`   | 3°             | 3° (3.9)                  |
| `filter_min_num_consistent`        | 2              | 2 (3.9)                   |
| `filter_geom_consistency_max_cost` | 1 px           | 1 px (3.9)                |
| `max_depth_error` (fusion)         | 0.01           | 0.01 (3.10)               |
| `max_reproj_error` (fusion)        | 2 px           | 2 px (3.10)               |
| `max_normal_error` (fusion)        | 10°            | 10° (3.10)                |
| `num_iterations`                   | 5              | not used, see below       |
| `min_num_pixels` (fusion)          | 5              | not used, see below       |

Two COLMAP defaults have no counterpart here, and this document does not quote them as
COLMAP's. COLMAP's `num_iterations` 5 counts iterations of four sweeps each. This design's
`photometricIterations` 8 counts red-black iterations of two passes, so the counts do not
compare. COLMAP's `min_num_pixels` 5 is the least number of fused pixels of a point. This
design's `minConsistentViews` 3 counts views (3.10). The source is the struct initializers,
not `colmap patch_match_stereo --help`, because no COLMAP binary ran for this check.

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

Amendment, step 4: the pass starts with `mvsGeometricInitialize` _(new in step 4)_, which copies
the reference's photometric depth into `mvsGeoDepth` and computes the cost of the hypothesis
again with the geometric term. The reference's normal stays in `mvsNormal`, which the geometric
pass of another reference never reads. The geometric iterations perturb from 2^−1 again, and
they hash as iterations T + 1 to T + G, so their draws differ from the photometric pass's. ψ_s
back-projects the source's depth along the ray of q itself, as COLMAP does. A q outside the
source, a source pixel without depth or a point behind a camera gives ψmax.

### 3.9 Filtering (device, `mvsFilter` _(new)_)

A pixel of the geometric depth map is kept when all of these hold:

- the photometric cost of its hypothesis, averaged over its best sources, is at most 0.9
  (NCC at least 0.1)
- at least 2 sources have ψ_s ≤ 1 px and a triangulation angle at X_p of at least 3°
- its depth lies inside the view's range

A rejected pixel gets depth 0, which means "no depth". The values are those of COLMAP's
filter options (`filter_min_ncc` 0.1, `filter_min_num_consistent` 2,
`filter_geom_consistency_max_cost` 1, `filter_min_triangulation_angle` 3), checked in 3.7.

Amendment, step 4: the stored cost holds the geometric term, so `mvsFilter` computes the
photometric source costs of the final hypothesis again (36 samples per source). Its ψ_s reads
the sources' photometric depths, as the geometric pass does, so the filter of one reference
does not depend on the geometric pass of another. The filter writes depth 0 in place over a
rejected pixel of `mvsGeoDepth`.

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
checked in 3.7.

`mvsFuseFlags` writes one flag per pixel. The shared flag scan (`planFlagScan`,
`scanBlockCount` in `src/gpu/scan.ts`) gives each emitted pixel its slot.
`mvsScatterPoints` writes the points. The host reads one count and the points of that
reference.

Amendment, step 5: "the point of j at q" is j's filtered depth at the nearest pixel of q,
back-projected along the ray of that pixel's center, with j's normal there. The point and the
normal are in the normalized world frame until the host maps the point back. "A lower index" is
the packed index. The scan rewrites `mvsFlags` in place, and a fused point kept per pixel would
need a second P-sized point array. So `mvsScatterPoints` computes the fused point of its pixel
again, with the same function as `mvsFuseFlags`. It runs as chunks under the key `mvsFuse`, since
its work per unit is the same. The runtime reads a whole array, so the host reads `mvsPoints` and
`mvsPointData` whole and keeps the first `count` points.

### 3.11 Colors

The color of a point is the RGBA of its reference pixel at the analysis size, after
undistortion with the same map as the gray image. The host already holds the RGBA of each
view (5.3). The fusion stage uploads the reference's packed RGBA, and `mvsUndistort` in its
RGBA mode writes it to `mvsRgba`, sampling each channel bilinearly. The array is released
after the reference's pass. Amendment, step 5: `mvsRgba` holds the reference's packed RGBA as
uploaded, and `mvsScatterPoints` samples it at the pixel's position in the distorted image. That
is the map of `mvsUndistort` (the helper `distortedPosition` of both), each channel bilinear and
rounded. So the fusion stage holds one RGBA array and no upload beside it, and no RGBA mode of
`mvsUndistort` exists. Averaging the colors of all consistent views needs the RGBA of every view
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

| Chunk key _(new)_        | Kernel                      | One unit                   | Work per unit                     |
| ------------------------ | --------------------------- | -------------------------- | --------------------------------- |
| `mvsInitialize`          | `mvsInitialize`             | one pixel × one source     | 1 hypothesis × 36 samples         |
| `mvsPropagate`           | `mvsPropagate`, photometric | one pixel × one source     | 10 hypotheses × 36 samples        |
| `mvsGeometricInitialize` | `mvsGeometricInitialize`    | one pixel × one source     | 36 samples + 2 projections        |
| `mvsPropagateGeometric`  | `mvsPropagate`, geometric   | one pixel × one source     | 10 × (36 samples + 2 projections) |
| `mvsFilter`              | `mvsFilter`                 | one pixel × one source     | 36 samples + 2 projections        |
| `mvsFuse`                | `mvsFuseFlags`              | one pixel × one other view | 2 projections                     |

The unit includes the source count, so the rate a view with 6 sources teaches sizes the chunks
of a view with 4. The two `mvsPropagate` variants have their own keys because their cost per
unit differs, as `bundleCg` and `bundleCachedCg` do. `mvsUndistort`, `mvsSeed`, the scan and
`mvsScatterPoints` do a few operations per pixel. Each runs as one plan with no chunk.

The CPU tier runs each key as one chunk (`chunkSize` returns `max` there), so its results are
those of the whole pass.

Step 3 added the keys `mvsInitialize` and `mvsPropagate` with their warm steps
(`src/gpu/mvs.ts`). The other keys come with their kernels: `mvsGeometricInitialize`,
`mvsPropagateGeometric` and `mvsFilter` in step 4, `mvsFuse` in step 5. Amendment, step 5:
`mvsScatterPoints` runs as chunks under `mvsFuse` too (3.10), and the flag scan runs as one plan.
Amendment, step 4: the
key `mvsGeometricInitialize` and the work per unit of `mvsFilter` (the photometric costs again,
3.9) are new in step 4.

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
samples run in the lab. Step 3 measures the SwiftShader rate of each key ("Step 3 record"), and
the lab's run records give the physical rates.

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
  ray caster renders 5 views at 128 × 96 px with known K, R, t and k1, and the exact depth and
  normal maps (`tests/typeshade/mvs-cases.mjs`). The size keeps the CPU-tier run under 60 s.
  The test records its time. Amendment, step 2: the first size, 64 × 48 px, cannot meet the
  depth bound below. Its 11 × 11 px window covers too much of the box and the sphere, and 1 %
  of depth is a shift of less than 0.1 px there. Measured at 64 × 48: a median error of 3.5 %
  after 8 iterations, with 23 % of the pixels within 1 %.
- Photometric pass, central view: median relative depth error ≤ 1 % on the pixels kept by
  the filter, at least 80 % of the pixels kept. Until step 4 adds
  `mvsFilter`, "kept" applies the two photometric rules of 3.9: a cost of at most 0.9 and a
  depth inside the range. The normal bound of 5° applies to the fused points (amended by the
  owner on 2026-10-20, step 5): the depth maps' median normal error was 14.0° after step 2. The
  measured median is 14.0° (7.4° on the interior of the flat surfaces, 12° on the box faces).
  Measured at four interior pixels, a tilt of ±5° about the true normal at the true depth
  changes the aggregated cost by 0.0001 to 0.003. Inference: at `windowRadius` 5 the
  photometric cost does not resolve the normal to 5°. The test holds the normal error to 16°
  as a regression bound. Section 9 lists the question.
- Geometric pass and filter: the median relative error is not larger than after the
  photometric pass, and the 95th percentile is smaller. Step 4 measured both ("Step 4 record").
- Fusion: at least 95 % of the emitted points lie within 5 % of the scene radius of the true
  surface, and the median normal error of the fused points is at most 5°. Duplicates (two points
  within 0.5 px of each other in their reference view) are below 10 %. Amended by the owner on
  2026-10-20 (step 5): the bound was 95 % within 1 %, and the defaults give 74.4 % within 1 % and
  96.8 % within 5 % at 128 × 96 px, where 1 % of the radius is about half a pixel's footprint
  ("Step 5 record").
- Chunk invariance: one chunk and forced chunks of 48 elements give the same depth, normal and
  cost arrays bit for bit, as `tests/typeshade/cpu.test.mjs` does for `triangulateTracks`. A
  chunk of 48 does not divide a row (64 color indices or 128 pixels), so the chunks cross rows.
- Reference order: running the geometric pass in reverse view order gives the same arrays bit
  for bit (3.8). Step 4 runs both passes in reverse order and the geometric pass in a shuffled
  order ("Step 4 record").

### 7.2 The same scene on WebGPU (CI)

`tests/typeshade/mvs-webgpu.test.mjs` _(new)_, added to `npm run test:webgpu` (the `webgpu`
job of `.github/workflows/ci.yml`, SwiftShader). Bit equality with the CPU tier is not
expected, because f32 rounding and fused multiply-add can differ. The test asserts that at
least 99 % of the pixels kept on both tiers agree within 1e-3 relative depth, and that the
accuracy bounds of 7.1 hold. It records the rate of each chunk key. Deviation, step 3, open:
after the full pass, 97.6 % of those pixels agree within 1e-3, not 99 %. The tiers agree within
1e-3 on every kept pixel after the initialization and on 99.98 % after iteration 1. Then a
last-bit difference flips the choice between two candidates of near-equal cost, and the
propagation carries it to the neighbours. The test holds the bounds of "Step 3 record".

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
| Synthetic scene, WebGPU              | ≥ 99 % of kept pixels within 1e-3 of the CPU tier (step 3: the bounds of "Step 3 record")         | `npm run test:webgpu` |
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

### Step 2 record

Configuration: branch `feat/mvs-photometric` from `main` at `fc070a8`, the compiler pin
`e0bf150`, Node 22.22.0, the CPU tier. Measured on 2026-10-20 by
`tests/typeshade/mvs.test.mjs` on the scene of 7.1 (128 × 96 px, 5 views, 30 landmarks, the
reference with 4 sources and 30 seeds). The defaults comparison runs with
`TYPECV_MVS_DEFAULTS=1`.

The modules are internal until step 5. The test reaches them through `src/gpu/internal.ts`,
which `scripts/build-gpu.mjs` builds to `dist/internal/gpu.js`, a bundle of its own. `dist/gpu`
and the exports of `typecv/gpu` do not change.

Error after each iteration, defaults ("kept" as in 7.1):

| Iteration | Median relative depth error | Kept and within 1 % | Kept | Median normal error |
| --------- | --------------------------- | ------------------- | ---- | ------------------- |
| 0 (init)  | 16.6 %                      | 1.5 %               | 45 % | 60.7°               |
| 1         | 3.29 %                      | 17.5 %              | 89 % | 48.7°               |
| 2         | 1.34 %                      | 37.6 %              | 92 % | 31.9°               |
| 4         | 0.80 %                      | 50.6 %              | 93 % | 18.4°               |
| 6         | 0.67 %                      | 52.4 %              | 94 % | 15.5°               |
| 7         | 0.63 %                      | 52.9 %              | 94 % | 14.7°               |
| 8         | 0.63 %                      | 53.2 %              | 94 % | 14.0°               |

The 95th percentile stays near 25 % from iteration 2 on: it is the image border and the
occluded pixels, which the geometric pass and the filter of step 4 address.

The defaults against their alternatives, each after the full pass:

| Options                    | Time on the CPU tier | Median depth error | Kept and within 1 % | Median normal error |
| -------------------------- | -------------------- | ------------------ | ------------------- | ------------------- |
| Defaults                   | 11.1 s               | 0.63 %             | 53.2 %              | 14.0°               |
| `windowStep` 1             | 31.2 s               | 0.58 %             | 54.1 %              | 14.7°               |
| `bilateral` on             | 14.2 s               | 0.52 %             | 55.6 %              | 14.1°               |
| `photometricIterations` 16 | 21.6 s               | 0.62 %             | 53.2 %              | 12.9°               |

Decisions: `photometricIterations` stays 8, because 16 iterations improve the depth by 0.01
points. `windowStep` stays 2. `bilateral` stays off. Its gain of 0.11 points is open for the
owner, since the cost is 28 % more time per pass. Chunks of 48 elements give the arrays of one
chunk bit for bit.

### Step 3 record

Configuration: branch `feat/mvs-webgpu` from `feat/mvs-photometric` at `11db341`, the compiler
pin `e0bf150`, Node 22.22.0, Playwright 1.56.1 with its Chromium 141 headless shell on
SwiftShader, a 4-core Intel Xeon container shared with other jobs. Measured on 2026-10-20 by
`tests/typeshade/mvs-webgpu.test.mjs` on the scene of 7.1 (128 × 96 px, the reference with 4
sources, the defaults of step 2). The full scene runs on SwiftShader in 4.4 s to 5.0 s, so the
test needs no smaller case. The whole test, the CPU tier included, takes about 20 s.

What changed: `mvsPhotometricGpu` (`src/gpu/mvs.ts`) runs `mvsInitialize` and `mvsPropagate` as
chunk steps under the keys of 5.1, one unit a pixel and a source. `device.chunkSize(key,
remaining × K, 64 × K)` sizes each chunk, and a chunk before the last is a whole number of
workgroups. Each key's warm step is its entry over zero workgroups (`warm.id` `mvs`).
`chunkPixels` caps a chunk. The CPU tier runs each pass as one chunk, so
`tests/typeshade/mvs.test.mjs` passes unchanged, the chunk invariance bit for bit included.

Agreement with the CPU tier, on the pixels kept on both tiers ("kept" as in 7.1):

| After               | Pixels kept on both | Within 1e-3 | Within 1e-2 | Median difference | Largest difference |
| ------------------- | ------------------- | ----------- | ----------- | ----------------- | ------------------ |
| Initialization      | 5,530               | 100 %       | 100 %       | 0                 | 0                  |
| Iteration 1         | 10,911              | 99.98 %     | 99.98 %     | 9.2e-8            | 1.7 %              |
| Iteration 8 (final) | 11,523              | 97.61 %     | 99.88 %     | 2.2e-6            | 34.7 %             |

The differences are relative depth differences. Over all pixels, 50 % of the depths and
normals are equal bit for bit after the initialization and 8 % after the full pass. The pixels
that part by more than 1e-3 hold hypotheses of near-equal cost: the median cost difference
between the tiers there is 5.5e-5. Inference: the transcendentals and the fused multiply-adds
round differently on SwiftShader, and the propagation amplifies a flipped choice. The kernels
agree, since the initialization and one iteration agree.

Accuracy against the truth after the full pass:

| Tier   | Median relative depth error | Kept    | Kept and within 1 % | Median normal error |
| ------ | --------------------------- | ------- | ------------------- | ------------------- |
| WebGPU | 0.628 %                     | 93.77 % | 53.14 %             | 14.08°              |
| CPU    | 0.627 %                     | 93.77 % | 53.18 %             | 14.05°              |

The test's bounds: every kept pixel within 1e-3 after the initialization. At least 99.9 %
within 1e-3 after iteration 1. After the full pass, at least 99 % within 1e-2 and at least 95 %
within 1e-3, a median cost difference of the parted pixels of at most 0.01, the bounds of 7.1
on WebGPU, and the two tiers' median error within 0.05 points and their kept share within 1
point. The tolerance of 1e-3 is that of 7.2. It is about 8000 f32 ulps, so a pixel inside it
holds the same hypothesis on both tiers, and a pixel outside it holds another one.

The SwiftShader rate of each key, as the device learned it (`device.chunkRates`, the budget
250 ms), in two runs:

| Key             | One unit                          | ms per unit    | Timed chunks | Longest chunk  | Window samples per second |
| --------------- | --------------------------------- | -------------- | ------------ | -------------- | ------------------------- |
| `mvsInitialize` | a pixel and a source, 36 samples  | 0.0015, 0.0018 | 5 of 5       | 22 ms, 32 ms   | 24 M, 20 M                |
| `mvsPropagate`  | a pixel and a source, 360 samples | 0.0107, 0.0106 | 8 of 35      | 250 ms, 260 ms | 34 M, 34 M                |

The pass of `mvsInitialize` has 49,152 units, so it ends after 5 chunks and its rate is not
settled. Its first chunk of 256 units weighs most in that rate. Each of the 16 color passes of
`mvsPropagate` is 24,576 units, about 260 ms at the learned rate, so a settled chunk holds most
of a pass. Inference from 5.2: one color pass at 640 × 480 with K = 6 (921,600 units) takes
about 9.8 s on this SwiftShader, about 50 times the rate one chunk per pass needs.

Deviations:

- 7.2, the agreement bound of 99 % within 1e-3: not met after the full pass (97.6 %). The test
  holds the bounds above. Disposition: open, for the owner to accept the bounds above as an
  amendment of 7.2 and of the gate row of 7.5.
- 5.1, the five keys: step 3 adds two. The other three come with their kernels in steps 4 and 5. Disposition: closed, by the note in 5.1.

### Step 4 record

Configuration: branch `feat/mvs-geometric` from `main` at `4f24db2`, merged with `main` at
`1a08238` (step 3), the compiler pin `e0bf150`, Node 22.22.0, the CPU tier, a 4-core container
shared with other jobs. Measured on 2026-10-20 by `tests/typeshade/mvs-geometric.test.mjs` on
the scene of 7.1 (128 × 96 px, 5 views, every view with 4 sources). The defaults run with
`TYPECV_MVS_DEFAULTS=1`.

What changed:

- `src/gpu/mvs.shade.ts`: `mvsPropagate` reads and writes the reference's depth in
  `mvsGeoDepth` when `MvsControl.geometric` is 1, and `hypothesisCost` adds
  λ · min(ψ_s, ψmax) to each source cost (`reprojectionError`). New entries
  `mvsGeometricInitialize` and `mvsFilter` (3.8 and 3.9, with their amendments). Each lane past
  its chunk takes the chunk's last element and writes nothing, so no idle lane returns before a
  loop.
- `src/gpu/mvs.ts`: `mvsDepthMapsGpu` runs the photometric pass of every view in `order`, then
  the geometric pass and the filter of every view in `geometricOrder`. It schedules them under
  the keys `mvsInitialize`, `mvsPropagate`, `mvsGeometricInitialize`, `mvsPropagateGeometric`
  and `mvsFilter`, with warm steps, by the rule of step 3. `mvsPhotometricGpu` binds
  `mvsGeoDepth` to `mvsPropagate` and reserves the 16 N + 4 bytes per pixel of 4.2. Its results
  are those of step 2 bit for bit: the defaults give 0.627 % and 14.05°, as in "Step 3 record".
- `src/index.ts` and `src/gpu/index.ts` export nothing new. The test reaches
  `mvsDepthMapsGpu` through `src/gpu/internal.ts`.

The central view, the defaults (8 photometric and 4 geometric iterations). "Kept" is the
photometric rule of 7.1 for the first two rows and a depth above 0 after the filter. The
percentiles are of the relative depth error on the kept pixels. "Within 1 %" counts kept
pixels against all pixels with a true depth:

| Stage                          | Kept   | Median | 90th percentile | 95th percentile | Within 1 % | Median normal error |
| ------------------------------ | ------ | ------ | --------------- | --------------- | ---------- | ------------------- |
| Photometric pass               | 93.8 % | 0.63 % | 10.6 %          | 24.8 %          | 53.2 %     | 14.0°               |
| Geometric pass, before filter  | 78.7 % | 0.43 % | 5.4 %           | 8.1 %           | 51.2 %     | 10.9°               |
| Geometric pass and `mvsFilter` | 76.8 % | 0.39 % | 4.8 %           | 7.8 %           | 51.5 %     | 10.6°               |

On all pixels with a true depth, with no rule, the median error goes from 0.79 % to 0.70 %
through the geometric pass. The CI run uses 4 photometric iterations to keep the test short.
There the photometric pass gives 93.5 %, 0.80 %, 11.0 %, 24.9 %, 50.6 % and 18.4°, and the
filtered map 75.7 %, 0.49 %, 4.6 %, 8.1 %, 49.4 % and 14.6°.

Every view, the defaults, with every view through the geometric pass and the filter:

| View | Median, photometric → filtered | 95th percentile | Kept            | Median normal error |
| ---- | ------------------------------ | --------------- | --------------- | ------------------- |
| 0    | 0.68 % → 0.35 %                | 41.1 % → 9.2 %  | 87.5 % → 70.0 % | 12.0° → 7.6°        |
| 1    | 0.58 % → 0.34 %                | 31.3 % → 7.8 %  | 96.3 % → 76.8 % | 11.9° → 7.4°        |
| 2    | 0.63 % → 0.39 %                | 24.8 % → 7.8 %  | 93.8 % → 76.8 % | 14.0° → 10.6°       |
| 3    | 0.99 % → 0.43 %                | 52.5 % → 8.0 %  | 94.2 % → 68.1 % | 18.8° → 9.6°        |
| 4    | 0.80 % → 0.34 %                | 38.6 % → 6.9 %  | 96.5 % → 71.4 % | 15.3° → 8.5°        |

So the error tail shrinks on every view: the 95th percentile falls by 3 to 7 times, and the
median falls too. View 2's numbers are the same whether only view 2 or every view runs the
geometric pass.

The kept fraction. On the host, 94.6 % of view 2's pixels see a true point that 2 or more
sources see unoccluded at a triangulation angle of at least 3°. That is the most the rule of
2 consistent sources can keep with exact depths. The filter keeps 76.8 %. Inference: the other
pixels fail ψ_s ≤ 1 px against the sources' photometric depths, whose own 95th percentiles are
25 % to 53 %. Two choices were measured on view 2 and are not defaults: perturbations that go on
from 2^−(T + 1) give 0.40 % and 10.65°, against 0.39 % and 10.64° for the restart at 2^−1.

Reference order. The test runs the small configuration (a 3 × 3 window, 1 iteration of each
pass) twice: once in the order 0 to 4, and once with the photometric order 4 to 0, the
geometric order 3, 1, 4, 0, 2 and chunks of 1000 elements. Every view's photometric, geometric
and filtered depth, normal and cost arrays are equal bit for bit. The reasons:

- Every photometric pass ends before the first geometric pass starts. A photometric pass reads
  `mvsGray` and writes only its reference's slots of `mvsDepth` and `mvsNormal`, and `mvsCost`.
- The geometric pass and the filter read `mvsDepth`, which no geometric pass writes, and the
  reference's own slots of `mvsGeoDepth` and `mvsNormal`. They never read another view's slots
  of those two arrays. So the photometric depths act as the read buffer of a double buffer.
- `mvsCost` holds one reference at a time, and `mvsInitialize` and `mvsGeometricInitialize`
  write all of its P values first.
- The draws hash the reference's packed index, not its position in the order (3.5).

The same test holds the photometric pass of `mvsDepthMapsGpu` equal bit for bit to that of
`mvsPhotometricGpu`.

Time on the shared container: the test file takes 85 s to 100 s, of which the accuracy run is
55 s to 71 s and the order test 28 s. `tests/typeshade/mvs.test.mjs` took 22 s for the run that
"Step 2 record" timed at 11.1 s, so the container ran at about half speed. The defaults run
takes 105 s.

A check outside the tests: `mvsDepthMapsGpu` with 4 photometric iterations ran on SwiftShader
through `tests/typeshade/browser.mjs` in 11.6 s. View 2 kept 75.71 % against 75.72 % on the CPU
tier. The device learned a rate for all five keys, and `liveBytes` was 0 after the run.

Deviations and open items:

- Open: the median normal error is 10.6°, not the 5° of 7.1. The geometric term constrains the
  depth and not the normal. The normal is still observed only through the photometric cost, which
  a tilt of ±5° changes by 0.0001 to 0.003 (step 2). The gain from 14.0° comes from the filter,
  which removes wrong hypotheses, and from the geometric pass, which replaces outlier planes.
  Proposal for the owner: either add a normal candidate fitted to the neighbours' filtered depths
  to the geometric pass (a design amendment), or amend 7.1 to the measured regression bound and
  let step 5 measure whether fusion's 10° normal test (3.10) rejects correct points. The test
  holds 12° on the defaults and 16° on the CI configuration.
- Open: the filtered map keeps 76.8 %. 7.1 puts its 80 % on the photometric pass, where it holds
  (93.8 %), but the gate row of 7.5 can be read as a bound on the filtered map. Proposal: 7.5's
  80 % applies to the photometric rule, and the filtered map is held to 70 % as a regression
  bound. The test holds 70 %.
- Closed by amendment: the new kernel `mvsGeometricInitialize` and its key (3.8, 5.1), and the
  filter's photometric costs (3.9, 5.1).
- Open: the WebGPU test of 7.2 covers the photometric pass only. Extending
  `tests/typeshade/mvs-webgpu.test.mjs` to the geometric pass and the filter is left to step 5.
- Open: `mvsPhotometricGpu` and `mvsDepthMapsGpu` each hold the chunk rule of step 3. Step 5
  can build `mvsPhotometricGpu` on `photometricSteps` and keep one copy.

### Step 5 record

Configuration: branch `feat/mvs-fusion` from `feat/mvs-geometric` at `1b25bfe` (step 4), the
compiler pin `e0bf150`, Node 22.22.0, the CPU tier, and Playwright 1.56.1 with its Chromium 141
headless shell on SwiftShader. The container has 4 cores and is shared with other jobs. Measured
on 2026-10-20 by `tests/typeshade/mvs-fusion.test.mjs` and `tests/typeshade/mvs-webgpu.test.mjs` on
the scene of 7.1 (128 × 96 px, 5 views, every view with 4 sources). The defaults run with
`TYPECV_MVS_DEFAULTS=1`.

What changed:

- `src/gpu/mvs.shade.ts`: new entries `mvsFuseFlags` and `mvsScatterPoints` and their function
  `fusePixel` (3.10), the color sample `referenceColor` (3.11), and the helper
  `distortedPosition`, which `mvsUndistort` now uses too. Each lane past its chunk takes the
  chunk's last element and writes nothing, so no idle lane returns before a loop.
- `src/gpu/mvs.ts`: `reconstructDenseGpu(device, scene, images, options)` as 5.4 declares it. It
  plans the views (`planMvsViews`), runs the depth stage of every view, recycles `mvsGray`,
  `mvsDepth`, `mvsCost`, `mvsSeeds` and `mvsSources`, and fuses each reference. The flag scan is
  the shared `planFlagScan` of `src/gpu/scan.ts`. A view without `width` and `height` takes its
  image's size. The depth stage allocates `mvsGray` and undistorts before it allocates the other
  arrays, so its peak is the formula of 4.2. `mvsPhotometricGpu` runs on `photometricSteps`, so
  the chunk rule of step 3 has one copy (the open item of step 4). Its results are those of step
  2 bit for bit: 0.627 % and 14.05°, as in "Step 3 record".
- `src/gpu/index.ts` exports `reconstructDenseGpu` and the types `GpuDenseOptions`,
  `GpuDensePoints`, `GpuDenseResult`, `GpuDenseView`, `GpuDenseViewEvent`,
  `GpuDenseViewStatus`, `MvsImage`, `MvsScene`, `MvsSceneLandmark`, `MvsSceneView` and
  `MvsViewOptions`. The repository has no API-surface snapshot, so no snapshot changed.
- The result holds `points` (`count`, `positions` in the model's frame, `colors` as RGB bytes,
  `normals` as two snorm16 values of the octahedral world normal), `views` (one per input view:
  `id`, `status` `done`, `noSource` or `fewLandmarks`, the source ids, `depthPixels`,
  `firstPoint` and `points`) and `report`, the `ShadePlanReport` of the run. `options.onView`
  receives a `depth` event per view (the preview of 5.3) after the depth stage of every view, and
  a `points` event per reference after its fusion. The events hold copies, so a worker can
  transfer them.

The fused points, all five views, the CPU tier. "Within 1 %" is the share of points within 1 %
of the scene radius (the plan's normalization scale, 1.031) of the nearest true surface. The
distances are in radii. The normal error is against the normal of the nearest true surface. A
duplicate is a point within 0.5 px of an earlier point in that point's reference view, at a depth
within 1 % of the earlier point's:

| Configuration            | Points | Within 1 % | Within 5 % | Median distance | 95th percentile | Median normal error | Duplicates |
| ------------------------ | ------ | ---------- | ---------- | --------------- | --------------- | ------------------- | ---------- |
| Defaults (8 photometric) | 5,660  | 74.4 %     | 96.8 %     | 0.24 %          | 4.0 %           | 2.46°               | 3.9 %      |
| CI run (4 photometric)   | 4,297  | 72.5 %     | 97.3 %     | 0.50 %          | 3.8 %           | 5.03°               | 7.2 %      |

At the defaults the views emit 4,620, 875, 165, 0 and 0 points. Views 3 and 4 emit none, because
each of their surface elements is consistent with a view of a lower index. Of the duplicates at
the defaults, 49 of 218 are pairs of one reference, whose means moved within 0.5 px of each
other. Every color matches the host undistortion of its reference within 1.

The 7.1 point bound. Deviation, open: 74.4 % of the points lie within 1 % of the radius, not
95 %. The tail is on the sphere and the box: by the nearest surface, 60 % of the sphere's points
and 28 % of the box's are past 1 %, against 8 % of the plane's. Inference: at 128 × 96 px, 1 % of
the radius (0.0103 world units) is about half the footprint of a pixel. The depth bound of 7.1
(1 % of the depth, about 3.2 % of the radius here) and the fusion's 1 % depth test are looser
than the point bound, so the point bound does not follow from them. The test holds regression
bounds: at least 70 % within 1 %, at least 95 % within 5 % and a median of at most 0.6 %.
Disposition: closed by amendment. The owner amended 7.1 to 95 % within 5 % of the radius on
2026-10-20.

The duplicate bound of 7.1 holds: 3.9 % at the defaults and 7.2 % in the CI run, below 10 %.

The normal bound (the open item of step 4). After fusion's 10° normal test the median normal
error of the fused points is 2.46° at the defaults, so the 5° of 7.1 is met for the points. The
90th percentile is 19.5°. In the CI run the median is 5.03°. Disposition: met for the fused
points at the defaults. The depth maps keep the regression bounds of step 4 (12° and 16°), since
the normal of a depth map is still observed only through the photometric cost. The test holds 5°
at the defaults and 6° in the CI run. Disposition: closed by amendment. The owner amended 7.1 on
2026-10-20 so that the 5° applies to the fused points.

Memory (4.2). The device's active bytes (`liveBytes` less `pooledBytes`), read in `onView`:

| Stage                         | Formula of 4.2             | Held by the device | The difference                                         |
| ----------------------------- | -------------------------- | ------------------ | ------------------------------------------------------ |
| Depth stage, every view       | (16 N + 4) P = 1,032,192 B | 1,034,480 B        | `mvsCameras` 480, `mvsSources` 140, `mvsSeeds` 1,668   |
| Fusion stage, every reference | (8 N + 28) P = 835,584 B   | 837,608 B          | `mvsFlagOffsets` 1,536, the totals 8, `mvsCameras` 480 |

Both tiers hold these bytes exactly, and both are within 0.3 % of the formula. `liveBytes` is 0
after every run. The undistortion peaks at 4 N P + 8 P and the small arrays, below the depth
stage.

Agreement with the CPU tier on the full path (`tests/typeshade/mvs-webgpu.test.mjs`, 2
photometric and 2 geometric iterations, so that the CPU tier takes about 50 s). The kept pixels
of each view's filtered map are 8,031, 8,947, 8,964, 7,829 and 8,385 on WebGPU, within 3 pixels
of the CPU tier. The points are 1,050 on WebGPU and 1,044 on the CPU tier. 98.9 % of the WebGPU
points have a CPU point of the same reference within 1e-3 of their depth, and 99.5 % within 1e-2.
In the other direction the shares are 99.4 % and 99.8 %. The accuracy of the two tiers agrees:
52.6 % and 52.2 % within 1 %, a median normal error of 12.7° and 12.6°. The test asserts depth
pixels within 2 %, point counts within 2 %, at least 90 % within 1e-3 and 98 % within 1e-2 in each
direction, and the memory formula on WebGPU. The full path ran on SwiftShader in 6.7 s, and the
test file took 84 s, of which the CPU run is 53 s.

The SwiftShader rate of the new keys in that run (`device.chunkRates`, the budget 250 ms), in ms
per unit: `mvsGeometricInitialize` 0.0012, `mvsPropagateGeometric` 0.0072, `mvsFilter` 0.0011 and
`mvsFuse` 0.00027. Each key timed 8 chunks.

Time on the shared container: `tests/typeshade/mvs-fusion.test.mjs` takes 138 s, of which the
accuracy run is 104 s and the chunk test 34 s. The fusion of all five references takes about
0.5 s of the 100 s run. The defaults run takes 145 s.

Chunk invariance. The small configuration (a 5 × 5 window, 1 iteration of each pass), with a
sixth view that observes no landmark, gives the same views and points bit for bit in one chunk
and in chunks of 1000 elements. The sixth view reports `fewLandmarks`, no depth pixels and no
points.

Deviations and open items:

- Closed by amendment: the point bound of 7.1 is now 95 % within 5 % of the radius, and the 5°
  normal bound applies to the fused points (the owner, 2026-10-20).
- Closed: the open items of step 4 on the WebGPU test (the full path now runs there) and on the
  two copies of the chunk rule (one copy now).
- Closed by amendment (3.10, 3.11, 5.1): `mvsScatterPoints` computes the fused point again and
  runs as chunks under `mvsFuse`. The color is sampled in the scatter, not by an RGBA mode of
  `mvsUndistort`, so `mvsRgba` is the reference's RGBA as uploaded.
- The signature of 5.4 is kept. The result's fields, the event shapes of `onView` and the
  exported types are this step's choices, since 5.4 does not name them.
- Open: the depth events come after the depth stage of every view, from one read of
  `mvsGeoDepth`, not after each view's filter. A read after each filter would read N × P values
  each time, because the runtime reads whole arrays. Step 6 can keep this or read a preview on
  the device.
- Open: the host reads `mvsPoints` and `mvsPointData` whole for each reference (20 B per pixel),
  for the same reason. A ranged read in the runtime would read only `count` points.
- Open: the 7.2 bound of 99 % within 1e-3 stays as in "Step 3 record" for the photometric pass.
  The full path agrees within the bounds above.

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

**Normal accuracy.** Step 2 measured a median normal error of 14.0° after the photometric
pass, against the 5° of 7.1. A tilt of ±5° changes the cost by 0.0001 to 0.003 at the pixels
measured, so the normal is weakly observed at `windowRadius` 5 (inference). Step 4 measured
10.6° after the geometric pass and the filter. Open: whether the bound of 7.1 changes, whether
a normal fitted to the filtered depths joins the geometric pass, or whether fusion's normal
test (10°, 3.10) needs a larger window ("Step 4 record"). Step 5 measured the fused points: a median normal
error of 2.46° at the defaults, after fusion's 10° normal test ("Step 5 record").

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
