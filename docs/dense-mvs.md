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
record". Step 6 (the worker's dense MVS behind the lab's `?dense=1`, its messages, cancel and
the run record v5, `website/src/lib/dense-run.ts` and `website/dense-run.test.mjs`) is
implemented, and its measurements are in "Step 6 record". Amendment 3 under section 8 (the dense stage runs within any device budget) replaces the
all-resident layout of 4.2 and 4.3. Section 3.12 (the cleanup of the fused cloud) and its record in "Cleanup record" under section 8
were added after step 8. Section 3.13 (the mesh of the cleaned cloud) and "Mesh record" under section 8 (step 12)
were added after them. Each step in "Steps" below is its own
pull request with its own evidence.

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
   against the lab's working budget of 256 MiB (4.2).
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

For each reference view the host takes the optical-axis depths of all landmarks in the view's
frustum: in front of the view and projecting into its image, whether the view observes them or
not (`mvsFrustumDepths`, Amendment 2 in section 8; `depthPoints` in
`website/src/lib/sfm-scene.ts` computes the same depth). The range is
[0.75 × the 1st percentile, 8 × the 99th percentile]. Hypotheses are uniform in inverse depth
over the range. A view with fewer than 8 observed landmarks gets no depth map. The first
version of this section took only the observed landmarks and a factor of 1.25.

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

### 3.12 Cleaning the fused cloud (CPU reference and GPU kernels)

The fusion of 3.10 keeps a point seen consistently in at least `minConsistentViews` views. Stray
points remain: a few consistent but wrong points at depth edges, and sparse blobs off the
surface. Duplicates remain too, since 3.10 counts them. This section adds three filters after the
fusion, with the semantics of the libraries they are named after. It changes no contract of
3.1 to 3.11: `reconstructDenseGpu` still returns the raw fused cloud, and the filters are new
functions that take any cloud of the layout of `GpuDensePoints`.

All three use a uniform-grid spatial hash (`buildPointGrid` in `src/reconstruction/cloud-clean.ts`).
A cell is a cube. The host sorts the points by the bucket of their cell, with `hashCell` and a
table of the least power of two that holds the points. A bucket holds the points of every cell
that hashes to it, so a lookup of one cell compares the stored cell coordinates too. A cell
coordinate is at least 1 and below 2^20, and the cell size grows to keep it there. The cell of
a point is raised to 1 when rounding of `min − h` puts it at 0.99999: the lookups skip cells
below 1, so a point in cell 0 was lost as a neighbour.

1. **Statistical outlier removal** (`removeStatisticalOutliers`, PCL `StatisticalOutlierRemoval`).
   For each point, d is the mean distance to its k nearest neighbours, the point itself left
   out. The threshold is mean(d) + `stdRatio` · σ(d), with σ the sample standard deviation. A
   point stays when d is at most the threshold. Defaults k = 16 and `stdRatio` = 2. k is at most
   32 (the GPU kernel's list). Open3D's `remove_statistical_outlier(nb_neighbors, std_ratio)`
   counts the point itself among its neighbours and drops a point at exactly the threshold, so
   `nb_neighbors = k + 1` there is the same filter up to a tie.
   The cell size is √k times the median nearest-neighbour distance (at least twice), which
   `medianPointSpacing` takes from 512 points. A search visits the rings of cells around the
   point's cell until the k-th distance is within the ring's inner radius, `r` cells. Past
   `KNN_MAX_RING` = 6 rings it scans every point, so a far outlier costs one pass over the cloud
   and the answer stays exact.
2. **Voxel downsampling** (`voxelDownsample`, Open3D `voxel_down_sample`). One point per occupied
   voxel: the mean position, the mean color (per channel, rounded) and the normalized mean of the
   normals (the first member's normal when they cancel). The voxel size is given, or
   `spacingFactor` (2) times the median nearest-neighbour distance. The voxel of a point is
   `floor((p − min) · inverse)` in f32 on each axis, with `inverse` the f32 of 1 / size, so the CPU
   reference and the GPU kernel agree to the bit. The voxels come in the order of their first
   member.
3. **DBSCAN** (`dbscan`, `keepLargestClusters`, scikit-learn `DBSCAN(eps, min_samples)`). A point
   is core when at least `minPoints` points (itself included) are within `eps` (distance ≤ eps).
   Clusters grow from the core points in input order, so the numbering and the border points
   agree with scikit-learn. The grid cell is `eps`. The filter keeps the `keep` largest clusters
   (1 by default) and any cluster of at least `minFraction` of the largest. A cloud with no
   cluster at all comes back unchanged. The pipeline's default `eps` is `epsFactor` (4) times
   the median nearest-neighbour distance of the cloud it receives, and `minPoints` is 5.

`cleanDenseCloud` (CPU) and `cleanDenseCloudGpu` (`src/gpu/cloud-clean.ts`) run the three in the
order outliers, voxels, clusters, each on the output of the one before, and return the cloud and
a report: the count after each stage, the threshold, voxel size and `eps` used, the clusters found
and kept, and the milliseconds of each stage. Each stage turns off by itself.

The GPU tier runs the two steps that pay on a GPU:

- `knnMeanDistance` (`src/gpu/cloud-knn.shade.ts`): one invocation per sorted point, the ring
  search above with a sorted list of the k best squared distances in private memory. It reads the
  sorted points, their cell coordinates and the bucket offsets, and writes one f32 per point.
  The host gives the invocations a visit order (`cellVisitOrder` in `src/reconstruction/cloud-clean.ts`): the points by cell, z then y
  then x, so that the invocations of a group leave their loops together. The kernel inserts into
  the list in place and skips a row or a cell whose nearest corner is farther than the k-th
  distance (with a margin of 0.1 % for rounding). It goes to ring `KNN_GPU_MAX_RING` = 16 before
  it scans every point. Every variant gives the exact k nearest points, so the ring limit and
  the skips change the time and not the result.
- `voxelClaim` (`src/gpu/cloud-voxel.shade.ts`): one invocation per point. It finds the voxel and
  claims a slot of an open-addressing table (at least 2n slots) with
  `atomicCompareExchangeWeak` on the index of its first member. The result is the owner of its
  voxel. Which member owns it depends on the order of the invocations, so the host orders the
  voxels by first member and sums in double (`reduceVoxels`). The output does not depend on it.

Both run as chunks of a point range under the keys `cloudKnn` and `cloudVoxel`. The host builds
the grid and applies the threshold. DBSCAN stays on the CPU: its label propagation is sequential
(scikit-learn's own loop), and it runs on the cloud after the two reductions. A cloud whose
arrays exceed the device's storage binding limit runs the CPU reference for that step.

The lab (5.3) runs `cleanDenseCloudGpu` after the fusion, on by default. `?denseClean=0` turns it
off. The cleaned cloud replaces the raw points on the page and in the dense PLY. The run record
carries `dense.options.denseClean`, and `dense.clean` with the counts and times (`docs/lab-run-records.md`).

### 3.13 Meshing the cleaned cloud (CPU reference and GPU kernels)

The cleaned cloud of 3.12 carries a normal per point. This section turns it into a triangle mesh
with a color per vertex: a signed distance field on a lattice, then marching cubes. It changes no
contract of 3.1 to 3.12. `reconstructDenseGpu` and `cleanDenseCloudGpu` return what they returned,
and the mesh is a new function of any cloud of the layout of `GpuDensePoints` that has normals.

**The field** (`src/reconstruction/dense-mesh.ts`). A lattice node x takes the points p within the
truncation radius R of it. A point with unit normal n gives the signed distance d = n · (x − p) to
its tangent plane, positive on the side the normal points to, and the weight w = (1 − |x − p|² /
R²)². The field is Σ w d / Σ w and the node color is Σ w c / Σ w, rounded. A node with no point
within R, or with Σ w below `minWeight`, has no value. This is the weighted average of the
nearest oriented points' plane distances, truncated at R, as in a TSDF. The weight has no
transcendental function, so the CPU reference (a scatter in double) and the kernel (a gather in
float32) differ only by rounding.

**The lattice.** The lattice spacing h is `voxelSize`, or `spacingFactor` (1.5) times the median
point spacing of 3.12. R is `truncation` (2.5) spacings. The lattice is sparse: bricks of
8 × 8 × 8 nodes, and a brick is stored only when a node of it lies within R of a point. A node's
address is its brick's slot times 512 plus its place in the brick, the slots following the brick
index. A dense lattice would need 58 million nodes for the lab cloud at the spacing of its points.
The sparse one holds 3,722 bricks, 1.9 million nodes. The spacing grows when the brick grid would
exceed 2^22 cells or the stored bricks 2^24 nodes. A key (below) must stay below 2^32, which the
second limit holds.

**The cubes.** A cube is inside where the field is below 0. A cube whose eight corners all have a
value makes up to five triangles by the 256-case table of Lorensen and Cline in the numbering of
Bourke (`src/reconstruction/marching-cubes-tables.ts`, the table of three.js's `MarchingCubes`,
which `tests/dense-mesh.test.ts` checks for the crossed edges of every case). The vertex of a
crossed edge is at the linear zero of the field, and its color is interpolated the same way. The
faces wind counter-clockwise seen from the side the normals point to. A vertex is named by its
edge, the address of the lower node times 3 plus the axis, and the host welds equal keys, so the
mesh is indexed and two cubes share the vertices of a shared edge. Inference: the table of 1987
(Bourke's form) has ambiguous faces whose neighbours can disagree and leave a hole. A sphere of
points and the lab cloud show none that makes an edge used more than twice ("Mesh record").
Lewiner's tables (scikit-image's default) remove the ambiguity, and are an open item.

**The GPU tier** (`src/gpu/dense-mesh.ts`, `src/gpu/dense-mesh.shade.ts`) runs four kernels:

- `meshNodeField`: one invocation per active node (a node within R of a point). It gathers the
  points from the cell grid of `buildPointGrid` with the cell size R, the 27 cells around the
  node, and writes the field and the color. The grid is the spatial hash of 3.12.
- `meshClassify`: one invocation per active node as the first node of a cube. It writes the flag 1
  when the cube makes triangles. The shared flag scan (`planFlagScan`) turns the flags into slots.
- `meshEmit`: a cube with a slot writes a row of 16 words: the triangle count and three vertex
  keys per triangle. A row has room for the five triangles. The rows follow the order of the
  nodes, so the mesh does not depend on the order of the invocations.
- `meshVertices`: one invocation per welded key, after the host welds the keys of the rows in
  order of first appearance. It writes the position and the packed color.

The kernels run as chunks under the keys `meshField`, `meshCubes` and `meshVertices`, with the
clamp-and-mask rule of the other kernels (a lane past its chunk takes the last element and writes nothing). A
cloud whose arrays exceed the device's storage binding limit runs the CPU reference.
`meshDenseCloudGpu` (exported from `typecv/gpu`) and `meshDenseCloud` (exported from `typecv`)
return `{ mesh, report }`. The mesh holds `positions` (f32 × 3), `colors` (u8 × 3) and `indices`
(u32 × 3 per face). The report holds the lattice spacing, R, the lattice size, the stored bricks,
active nodes, cubes that make triangles, vertices and faces, and the milliseconds of the lattice,
field, cube and vertex stages.

**The lab.** With `?denseMesh=1`, after the cleanup, the worker meshes the cloud that the page shows
(`meshDenseCloudGpu` with the defaults) and posts `denseMesh` (positions, colors and indices,
transferred) before `denseResult`. The mesh is off by default, since it costs 0.7 s on the CPU tier and 2.5 s on SwiftShader here, and more on a phone (not measured). `?denseMesh=1` turns it on, and a failed mesh leaves the
points and records why. The Display popover has a "Mesh" switch, off by default and disabled until
a mesh exists. On, the viewer draws the mesh (a `BufferGeometry` with vertex colors, lit by a
hemisphere light) in place of the points. "Export mesh PLY" writes a binary little-endian PLY with
`x y z` (float), `red green blue` (uchar) and a `vertex_indices` face list. The run record carries
`dense.options.denseMesh` and `dense.mesh` (`docs/lab-run-records.md`), both optional, with no
format bump.

### 3.14 Superpixels as a planar prior (proposed, not implemented)

Status: proposed. This section changes no code. The MVS kernels are as sections 3.4 to 3.10 describe
them. The input is `slicGpu` (`docs/gpu-features.md`, "SLIC superpixels"), which is built and
measured on its own.

**Why.** Sections 3.6 and 3.9 name the gaps: a textureless region has a reference variance near
0, so its windows cost 2 and its pixels fail the filter (section 9, "Textureless regions"). A
slanted surface is also weakly observed (section 9, "Normal accuracy"). A superpixel follows
color edges, and an object edge often is a color edge. Inference: most surfaces of a building
or a wall are near-planar inside one superpixel. A plane per superpixel is then a prior that
the pixel-wise search cannot find by itself.

**Where it plugs in.** There are three places. They are listed from the cheapest. Each one is a
separate step, and each one is off by default.

1. Holes per segment (after 3.9, before 3.10). For each superpixel of a reference view, fit a
   plane to the pixels that survived the filter, by least squares in inverse depth. Fill a
   rejected pixel with the plane's depth when the superpixel has at least `m` surviving pixels
   (start with 30 % of its area) and the fit residual is small (start with 2 % of the depth).
   Mark a filled pixel, so that fusion (3.10) can require one more consistent view for it.
2. Propagation candidates (inside 3.6). Add one candidate per pixel: the plane of its
   superpixel from the previous iteration, moved to the pixel as the 8 candidates are. It is
   hypothesis 11 of the 10 evaluated today, so the cost grows by 10 %. A superpixel plane is
   fitted once per iteration by a reduction over its pixels, the same shape as `slicUpdate`
   (one invocation per cluster, no atomics).
3. A smoothness term (inside 3.7). Add a cost term for the distance between a pixel's
   hypothesis and its superpixel's plane, weighted by `lambda`. This is the planar prior of
   ACMP (Xu and Tao 2020). It changes the cost, so it also changes the gates of 7.1.

**What it costs.** Facts, from the measured rates of `docs/gpu-features.md`: one `slicGpu` call at
640 × 480 takes about 0.45 s on SwiftShader and 0.33 s on the CPU reference. It runs once per
reference view, because the views do not share labels. For 12 views that is 12 calls. The
unit counts of section 5.2 are in window samples: 97 G for 12 views. SLIC does about 3 M
distance evaluations per iteration, so its cost is under 0.1 % of that. Inference: the cost is
the time of the calls and the readbacks, not the samples. The resident arrays grow by one
`u32` label per pixel (1.2 MB at 640 × 480) per view, plus 3 `f32` Lab values per pixel while
`slicGpu` runs (3.7 MB, freed after). Option 2 adds about 10 % to the photometric and geometric
passes (65 % and 33 % of the time), so about 10 % in all. Option 1 adds under 1 %. Option 3
adds about as much as option 2.

**What stays open.** The superpixels come from one view's colors only. An edge that has no color
change is not a superpixel edge, and a plane may then cross a depth discontinuity. Option 1
fills across such an edge when the fit is poor but under the residual limit. Option 2 and
option 3 can pull a pixel that the cost knows is right toward a wrong plane.
Whether a segment is planar is not known in advance. A curved surface needs more, smaller
segments (`nSegments` 400 or more).

**How to measure the gain.** Use the comparisons that exist. Do not add a new gate before the
first run.

1. Baseline: run `reconstructDenseGpu` as it is on the TUM sample of 7.3 and on the synthetic
   scene of 7.1. Record the median relative depth error and the share of valid pixels.
2. Run the same with one option on. Record the same two numbers, the time per reference view and
   the fused point count.
3. Split the valid pixels into the ones the option filled and the others. Report the median
   depth error of the filled ones alone. The gain is real only when the filled pixels are
   valid at a useful rate and their error is near the target of 7.3 (3 %).
4. Repeat the three steps with `nSegments` 100, 400 and 1600 and with `compactness` 10 and 25.
   The result is one table: segments, valid share, median error, time.
5. Compare the dense output with COLMAP's (7.4) once, with the best setting.

Accept an option when the valid share rises by at least 5 points and the median error of the
filled pixels stays at or below 5 %. These thresholds are this proposal's, and the first
measured run sets the gate. Option 1 comes first. It has no effect on 3.6 and 3.7, so the
existing bounds of 7.1 still hold with it off.

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

Amendment 3 (section 8): the all-resident layout below is replaced by a window of the reference
and its sources, streamed when the window does not fit the device's budget. The depth stage then
takes (8 K + 28) bytes per pixel in its geometric pass, K ≤ 6, whatever N is, and no view limit
applies. `planMvsResidency` (`src/gpu/mvs-residency.ts`) sizes every array. Amendment 3 has the
byte math for 640 × 360, 1280 × 720 and 1280 × 960 at N = 12, 25, 50 and 100. The rest of this
section is the layout before Amendment 3.

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
| Lab `workingBytes` (worker)         | 256 MiB                 | 256 MiB                    |
| Headroom in the depth stage         | 198.58 MiB              | 26.31 MiB                  |
| Host RGBA of 12 views (5.3)         | 14.7 MB                 | 59.0 MB                    |

gerrard-hall's previews are 640 × 427 (P = 273,280). Its depth stage takes 51.08 MiB at N = 12.

The worker scales `workingBytes` and `poolBytes` with the square of the analysis edge, and by 4
when the SIFT first octave is -1 (`start` in `website/src/workers/sfm.worker.ts`). Octave -1 is
the lab default up to 640 px (`UPSAMPLED_EDGE_LIMIT` in `website/src/lib/sift-quality.ts`).
So the lab uses 256 and 128 MiB at 640 px with octave -1, and 256 and 128 MiB at 1280 px with
octave 0. A run with `?octave=0` at 640 px uses 64 and 32 MiB.

The bytes per pixel of the depth stage are the same at all sizes. The largest N is
N ≤ ⌊(workingBytes / P − 4) / 16⌋:

| Analysis size and budget  | Largest N |
| ------------------------- | --------- |
| 640 × 480, 256 MiB (lab)  | 54        |
| 640 × 427, 256 MiB (lab)  | 61        |
| 1280 × 960, 256 MiB (lab) | 13        |
| 640 × 480, 64 MiB         | 13        |
| 640 × 427, 64 MiB         | 15        |

Amendment, 2026-10-21: before this amendment the text gave 64 MiB and 13 views at 640 × 480.
That was true until octave -1 became the lab default. The limit at 640 × 480 in the lab is
now 54. Amendment 3 removes the limit and `denseViewLimit`.

Each array must also fit `maxStorageBufferBindingSize`, which `ShadeDevice.bindingBytes`
reads (128 MiB when the adapter reports none). The largest array at 1280 × 960 with 12 views
is 56.25 MiB. It passes 128 MiB at N = 28.

`maxKernelElements` is 65,535 × 64 = 4,194,240 elements per dispatch. A color pass at
1280 × 960 has 614,400 pixels, so it fits in one dispatch.

### 4.3 What stays resident, and when

Amendment 3 (section 8) replaces this table: a depth pass keeps the reference's state and the
slots of its window, and each finished per-view output goes back to the host until a later step
reads it. The table is the layout before Amendment 3.

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

South-building-24 (24 views) does not fit the all-resident layout at either size. Amendment 3
in section 8 is the windowed layout that it needs.

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

Amendment 3 (section 8): streamed mode adds the keys `mvsStreamCosts` (photometric) and
`mvsStreamCostsGeometric` (the geometric pass and the filter), one unit an element × a hypothesis
× a resident source, and `mvsStreamSelect`, one unit an element. The streamed fusion runs
`mvsFuseAccumulate` under `mvsFuse` (one unit a pixel × a resident fusion view) and
`mvsFuseScatterSums` under `mvsFuseScatter` (one unit a pixel). The upload of a view to a slot
(`mvsUndistort`, `mvsCopyF32`, `mvsCopyU32`) runs as one plan per view.

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

Amendments, step 6 ("Step 6 record"):

- Item 3: the kept RGBA is bounded by the all-resident layout of 4.2. When the run keeps more
  keyframes than ⌊(`workingBytes` / P − 4) / 16⌋, the worker clears the map, and the MVS reports
  `skipped` with the reason. Amendment 3 (section 8) removes this bound: the worker keeps the
  RGBA of every keyframe. A view that does not register is left out when the MVS starts, and
  the map is released when `denseResult` is posted.
- Item 5: `denseProgress` has three stages, `photometric`, `geometric` (the geometric pass and
  the filter) and `fusion`. `denseDepth` follows the depth stage of every view, from one read of
  `mvsGeoDepth` (the open item of "Step 5 record", kept). `denseResult.points` is the count of
  the fused points, since the points came in `densePoints`. During the MVS the worker posts no
  kernel `progress` messages, so the page receives only the messages of this list.
- Items 6 and 7: `denseResult` is the one message that ends the MVS. It carries `cancelled`,
  `skipped` or `failure` (the message, the stage, the kernel and the device-loss reason). An
  `error` message would end the whole run on the page, and the sparse result must stay.
- The record's `dense` block also holds the device's bytes of 4.3 and 7.5: `liveBytes` and
  `pooledBytes` before the first MVS allocation, the active bytes at the depth stage and at the
  fusion stage, and the depth stage's formula (16 N + 4) P. Amendment 3: the formula is the
  plan's bytes of the depth stage, and `residency` holds the plan of `planMvsResidency`.

### 5.4 The API

`reconstructDenseGpu(device, scene, images, options)` _(new)_ in `src/gpu/mvs.ts` _(new)_,
with the kernels in `src/gpu/mvs.shade.ts` _(new)_ and the host view selection in
`src/gpu/mvs-views.ts` _(new)_. `src/gpu/index.ts` exports it. `images` maps a view id to its
RGBA, width and height. `options.onView` receives the depth previews and points of 5.3. The
result holds the points, the per-view status (no source, too few landmarks, done) and the
plan report. A new export of `typecv/gpu` changes the public API, so its step waits for the
owner's approval. Amendment, step 6: `options.onProgress` receives `{ stage, viewId, done,
total }` after each reference of each stage, and `options.signal` (an `AbortSignal`) stops the
run before its next plan. A cancelled run recycles its arrays and rejects with the signal's
reason. `src/gpu/index.ts` exports the types `GpuDenseProgress` and `GpuDenseStage` beside
those of step 5.

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

Step 8 did this with two differences, both recorded in "Step 8 record": the poses are metric,
so the comparison reports the error with no alignment and the median scale as a check; and it
ran at 320 × 240 px.

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

| Gate                                 | Number                                                                                             | Where                 |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- | --------------------- |
| Synthetic scene, CPU tier            | Median relative depth error ≤ 1 %, ≥ 80 % kept, chunk invariance bit for bit                       | `npm run test:shade`  |
| Synthetic scene, WebGPU              | ≥ 99 % of kept pixels within 1e-3 of the CPU tier (step 3: the bounds of "Step 3 record")          | `npm run test:webgpu` |
| TUM, ground-truth depth              | Median relative error ≤ 3 %, ≥ 50 % valid (step 8: 0.71 %, 81.9 %; the gate is set by the record)  | lab, record           |
| gerrard-hall vs COLMAP dense         | Median relative difference ≤ 2 % (first run sets the gate)                                         | offline record        |
| Physical devices (collector records) | gerrard-hall dense at 640 px completes on devices A and B, no device loss, longest chunk ≤ budget  | lab, run records      |
| Memory                               | `liveBytes` at the depth stage's peak equals the formula of 4.2 within 1 % (Amendment 3: the plan) | WebGPU test, lab      |

## 8. Steps

Each step is one pull request, in this order. Each names this document on a line of its own
in its commits.

| Step | Pull request content                                                                                                                       | What it proves                                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `src/gpu/mvs-views.ts`: view selection, depth ranges, seeds, normalization. `tests/mvs-views.test.ts`. The COLMAP defaults of 3.7 checked. | Source lists of gerrard-hall from the dataset's features are stable. The defaults in this document are right.                         |
| 2    | `mvsUndistort`, `mvsInitialize`, `mvsSeed`, `mvsPropagate` (photometric) on one reference, CPU tier. `tests/typeshade/mvs.test.mjs`.       | Photometric accuracy on the synthetic scene. Chunk invariance. The defaults of iterations, window step and bilateral weights.         |
| 3    | The chunk keys and `warm` steps. `tests/typeshade/mvs-webgpu.test.mjs`.                                                                    | WebGPU agrees with the CPU tier. The SwiftShader rate of each key.                                                                    |
| 4    | The geometric pass and `mvsFilter`.                                                                                                        | The error tail shrinks. Reference order changes no result.                                                                            |
| 5    | Fusion, `mvsFuseFlags`, the scan, `mvsScatterPoints`, colors. `reconstructDenseGpu` and its export.                                        | Point accuracy and the duplicate count on the synthetic scene. The memory formula of 4.2.                                             |
| 6    | The worker: `dense`, the kept RGBA, the messages of 5.3, cancel, the run record v5.                                                        | A lab run on gerrard-hall posts depth maps and points without a block of the page. `liveBytes` before the MVS.                        |
| 7    | The viewer and the page: the dense cloud, the switch, the point budget, binary PLY.                                                        | The page shows 10^5 to 10^6 points. The PLY loads in a standard reader.                                                               |
| 8    | TUM depth maps in the sample and `--depth` in `scripts/prepare-lab-tum-sample.py`. The comparison of 7.3.                                  | Depth accuracy against independent ground truth.                                                                                      |
| 9    | The offline COLMAP comparison of 7.4 and its record.                                                                                       | Agreement with COLMAP's dense output on the COLMAP samples.                                                                           |
| 10   | Physical-device runs and the gate rows of 7.5.                                                                                             | No device loss on devices A and B. The physical rates.                                                                                |
| 11   | The cleanup of 3.12: `cleanDenseCloud`, `cleanDenseCloudGpu`, the lab default and `?denseClean=0`.                                         | Outliers, voxel and DBSCAN agree with the CPU reference and scikit-learn. A lab cloud cleans in seconds.                              |
| 12   | The mesh of 3.13: `meshDenseCloud`, `meshDenseCloudGpu`, the lab switch, the mesh PLY and `dense.mesh` in the run record.                  | A sphere and a plane mesh within half a voxel, the sphere closed. The device equals the CPU reference. A lab cloud meshes in seconds. |

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

### Step 6 record

Configuration: branch `feat/mvs-worker` from `feat/mvs-fusion` at `00ee2f2` (step 5), merged
with `main` at `9c2a08f` (#97) and at `b430f41` (#96 and #98), the compiler pin `e0bf150`, Node
22.22.0, Playwright 1.56.1 with its Chromium 141 headless shell on SwiftShader. Checked on
2026-10-20.

What changed:

- `src/gpu/mvs.ts`: `GpuDenseOptions` gains `onProgress` and `signal` (5.4, amended).
  `onProgress` runs after the last plan of each reference's photometric pass, of its geometric
  pass and filter, and of its fusion. The run reads `signal` before each plan, so a chunk in
  flight completes. A cancelled run recycles its arrays and rejects with the signal's reason.
  `src/gpu/index.ts` exports the types `GpuDenseProgress` and `GpuDenseStage`.
- `website/src/lib/dense-run.ts`: the worker's dense MVS. `keepFrame` keeps a keyframe's RGBA,
  bounded by `denseViewLimit` (4.2). `runDense` runs the MVS over the registered views with a
  kept image and posts the messages of 5.3. `endDense` posts `denseResult` once.
- `website/src/workers/sfm.worker.ts`: `start` takes `dense`. `LAB_OPTIONS.dense` holds the
  defaults of section 3. `finish` posts the sparse `result`, then runs `runDense` on the same
  device. `cancelDense` bypasses the request queue and aborts the job's controller. A device
  loss or an uncaught error during the MVS ends the MVS only, with `denseResult.failure`.
- `website/src/components/SfmLab.tsx`: `?dense=1` sets `dense`. The status shows the stage, the
  reference count, the depth maps and the points. During the MVS the Cancel button reads
  "Cancel dense MVS" and posts `cancelDense`. The page keeps the depth and point messages for
  step 7 and builds the record after `denseResult`.
- `website/src/lib/run-record.ts`: format `typecv-run-record-v5` with the optional `dense` block
  (status, skip reason, options, views with and without a depth map, points, total and per-stage
  time, the device's bytes of 4.3, failure). `RUN_RECORD_FORMATS` keeps v3 and v4. `denseRecord`
  builds the block from `denseResult`, and `denseLine` is its line on the page.
- `website/worker.mjs` adds `denseStatus` and `densePoints` to the listing metadata.
  `scripts/export-run-records.mjs` shows them in the summary line and prints a run whose dense
  MVS failed under `--print failures`. `docs/lab-run-records.md` describes v5.

The message shapes (5.3, as amended):

| Message         | Fields                                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `denseProgress` | `stage` (`photometric`, `geometric`, `fusion`), `viewId`, `done`, `total`                                        |
| `denseDepth`    | `viewId`, `width`, `height`, `depth` (Float32Array, transferred)                                                 |
| `densePoints`   | `viewId`, `positions`, `colors`, `normals` (transferred)                                                         |
| `denseResult`   | `cancelled`, `failure`, `skipped`, `points` (count), `views`, `memory`, `timing`, `options`, `kernels`, `chunks` |

Verification that ran on 2026-10-20 at the merged configuration:

- `npm run typecheck`, `npm run format:check`, `npm run build`, `npm run build:website` and
  `npx tsc --noEmit` in `website`: pass.
- `website/dense-run.test.mjs` (new, a fake MVS in place of `reconstructDenseGpu`): the view
  limit, the message order and shapes with their transfer lists, the bytes before the MVS and at
  each stage, cancel, a failure with its stage and kernel, a device loss from outside the run, and
  a skipped run. Pass.
- `website/worker.test.mjs` (the v5 block, v4 and v3 without it, the collector) and
  `tests/export-run-records.test.ts`: pass.
- `tests/typeshade/mvs-fusion.test.mjs` on the CPU tier, before the merge with `b430f41`, which
  does not change `src/gpu`: the progress sequence of every reference and stage, and a cancel
  after the first event (no further event, `liveBytes` 0, a pre-aborted signal allocates
  nothing, the next run succeeds). Pass, 154 s.
- `website/site.test.mjs`: a warm pyramid run with `?dense=1` is cancelled during the MVS. The
  status says "Dense MVS cancelled", the worker and the sparse map stay, and the record is a valid
  v5 record with `dense.status` `cancelled` and `liveBytesBefore`. Pass, 634 s for the whole file.

Not run locally: `npm run test:shade` and `npm run test:webgpu` in full, including
`tests/typeshade/mvs-webgpu.test.mjs`, which step 6 does not change. CI runs them.

Deviations and open items:

- Open, deferred: the gerrard-hall measurement at 640 px with the dense MVS (the dense point
  count, the time of each stage, `mainThread` long tasks, and `liveBytes` before the MVS). A run
  with the lab's defaults (octave −1, guided matching) had not finished the sparse stage after
  12 minutes on this SwiftShader, and 5.2 infers about 10 s per color pass, so about 40 minutes
  for the MVS. The run was stopped for time. Next action: run the lab with `?dense=1` on a
  physical GPU, or the headless harness with a 2-hour timeout, and record the numbers here.
- Open: the site test cancels the MVS and does not run it whole. A whole pyramid run takes about
  20 minutes on SwiftShader (inference from 5.2). So CI does not prove `denseDepth` and
  `densePoints` in a browser. `website/dense-run.test.mjs` proves their shapes, and
  `tests/typeshade/mvs-webgpu.test.mjs` the full path on WebGPU.
- Closed by amendment (5.3, 5.4): the stages of `denseProgress`, `denseResult` as the one
  terminal message with `failure`, `points` as a count, the bound on the kept RGBA, the memory
  fields, and the options `onProgress` and `signal`.
- Kept from step 5: the depth previews follow the depth stage of every view, not each view's
  filter.
- The page's watchdog waits 10 minutes between messages during the MVS, against 2 minutes for
  the SfM. A reference's photometric pass can take minutes on a software GPU (5.2).
- Added beyond 5.3: the collector's `denseStatus` and `densePoints` and the exporter's line,
  so that the device runs of step 10 (7.5) show a dense failure in the export.

### Step 7 record

Configuration: branch `feat/mvs-viewer` from `feat/mvs-worker` at `e95531a` (step 6), the compiler
pin `e0bf150`, Node 22.22.0, three.js 0.184.0. Checked on 2026-10-20.

What changed:

- `website/src/lib/dense-cloud.ts` _(new)_: `DenseCloud`, the chunks of the `densePoints`
  messages of one run. `densePointBudget` gives the budget of section 6 (1,000,000, or 300,000
  when `navigator.maxTouchPoints > 0`). `denseStride` gives k, the smallest integer that keeps
  ⌈count / k⌉ in the budget. `densePly` writes the binary PLY, and `decodeOctahedral` decodes
  the normals as `decodeMvsNormal` does.
- `website/src/components/SfmViewer.tsx`: the props `dense`, `showDense` and
  `densePointBudget`. A second `THREE.Points` draws the points whose index in the run is a
  multiple of k: float32 positions and normalized `Uint8BufferAttribute` colors, 15 B per drawn
  point. The vertex shader turns the sRGB bytes linear, as the sparse cloud's table does. The
  chunks of each new message append, and only the appended range uploads. A new run, or a new k
  as the cloud grows, rebuilds the subset from the first chunk. Growth makes a new geometry and
  disposes the old one. A cleared reconstruction disposes the geometry too.
- `website/src/components/SfmLab.tsx`: the run option "Dense MVS", off by default. `?dense=1`
  turns it on at load, and a run still reads `?dense=1` at its start. The first points turn the
  display switch "Dense" on, and the dense cloud then replaces the sparse cloud. The switch is
  disabled until the run has points, and turning it off shows the sparse landmarks. "Export
  dense PLY" writes every point of the run, not the drawn subset. "Export PLY" stays the ASCII
  PLY of the sparse map. A small block in the run column shows the status, the points, the depth
  maps, the time of each stage and, over the budget, the drawn fraction.
- `docs/lab-run-records.md` names the run option beside `?dense=1`.

The PLY (section 6): `format binary_little_endian 1.0`, one `vertex` element with x, y, z
(float), nx, ny, nz (float) and red, green, blue (uchar), 27 B per point. A cloud without normals
leaves out nx, ny and nz (15 B per point). The positions are in the model's frame. The
normalization of 3.1 only scales and moves, so the world normals need no change.

Verification that ran on 2026-10-20:

- `npm run typecheck`, `npm run format:check`, `npm run build`, `npm run build:website` and
  `npx tsc --noEmit` in `website`: pass.
- `website/dense-cloud.test.mjs` _(new, in `npm run test:website`)_: the budgets, k as the
  smallest integer that fits, the octahedral decode, and a PLY whose header gives the properties
  and the vertex count, with a byte length of the header plus 27 B per point and the values of
  each vertex. Pass.
- A PLY of 1,010 points from `densePly`, read with `plyfile` 1.x (Python): the format, the count,
  the nine properties and the values agree. Pass. This check is not in CI.

Not run: `website/site.test.mjs` and the other suites (CI runs them), and a screenshot of the dense
cloud. A whole dense run of the pyramid sample takes about 20 minutes on SwiftShader ("Step 6
record"), so no headless browser drew the dense layer.

Deviations and open items:

- Open, deferred: the depth previews of section 6. The page keeps the `denseDepth` messages, and
  the photo overlay still shows the sparse depth points. Next action: draw the selected view's
  map with `depthColor` in the "Analyzed frame" card, after the layout of that card settles.
- Open: the proof that the page shows 10^5 to 10^6 points. The code path is in place, and no run
  of this step reached the fusion stage in a browser. Next action: a lab run with "Dense MVS" on
  a physical GPU, with the point count and the drawn count (`data-dense-points`,
  `data-dense-drawn` on the viewer) recorded here.
- Closed: picking. The viewer picks cameras only, not points, so the dense cloud adds no raycast.
- Added beyond section 6: the dense summary block and the viewer's `data-dense-*` and
  `data-cloud` attributes for the tests.

### Step 8 record

Configuration: branch `feat/mvs-tum-depth` from `main` at `b430f41`, runner commit `23b20cc`, the
compiler pin `e0bf150`, Node 22.22.0, Playwright 1.56.1 with its Chromium 141 headless shell on
SwiftShader (headless WebGPU), 4 cores shared with other jobs. Measured on 2026-10-20. The
result is `docs/experiments/dense-mvs/tum-fr3-structure-texture-far.json`.

What changed:

- `scripts/prepare-lab-tum-sample.py --depth` writes `depth-NN.png` for each frame: the depth
  image of the sequence nearest in time, 16-bit grayscale, at the analysis size (nearest
  neighbour if resized), depth = value / 5000 m. `sample.json` gains `depthFile` per frame and
  `depthUnitsPerMeter`. The 16 files are 976,381 B in all (about 61 KB each). The frames did not
  change. `tests/tum-sample-depth.test.ts` checks the size and the format of each file.
- `scripts/compare-mvs-tum-landmarks.py` triangulates SIFT matches with the sample's poses, so
  the sparse landmarks come from the images and not from the depth maps: 2,568 landmarks, 3.4
  observations on average.
- `scripts/compare-mvs-tum.mjs` runs `planMvsViews` and `mvsDepthMapsGpu` (the depth stage of
  `reconstructDenseGpu`: photometric, geometric, filter, default options) on headless WebGPU
  and compares each view's filtered depth with the ground truth. `--scale k` divides the size by
  k. `tests/typeshade/browser.mjs` allows deadlines up to 7,200 s (it was 300 s).

Result, 320 × 240 px (`--scale 2`), 15 of 16 views, the pixels with ground truth depth. The
poses are the motion-capture poses, in metres, with no alignment:

| Measure                                  | Value                  |
| ---------------------------------------- | ---------------------- |
| Median relative depth error              | 0.707 %                |
| Within 1 %, 2 % and 5 %                  | 65.7 %, 90.7 %, 97.5 % |
| Valid fraction (a filtered depth exists) | 81.9 %                 |
| Median scale (ground truth / estimate)   | 0.9996                 |
| Median error after that scale            | 0.705 %                |

The per-view medians are 0.6 % to 0.9 %, and the valid fractions 65 % to 91 %. The depth stage
took 681 s on SwiftShader. Both targets of 7.3 are met (at most 3 %, at least 50 % valid).
Disposition: the first run sets the gate at its targets, a median of at most 3 % and at least
50 % valid.

Deviations and open items:

- Closed by the amendment below: the analysis size. The depth stage now ran at 640 × 480
  (see "Amendment 1"). The 320 × 240 gate is unchanged.
- The runner calls `mvsDepthMapsGpu`, not `reconstructDenseGpu`: the public result holds the
  points and a quarter-size depth preview, not the full-size depth maps. The depth stage is the
  same code.
- View 15 has fewer than 8 landmarks (`fewLandmarks`) and gets no depth map. The valid fraction
  counts the other 15 views.
- Closed by the amendment below: the fused points against the ground truth, at 320 × 240.
  Open: the same at 640 × 480 (the run did not finish, see "Amendment 1").
- Not measured: the CPU tier, and the depth registration to the RGB image beyond the
  sub-percent medians above.

#### Amendment 1: full size and the fused points

Configuration: branch `feat/mvs-tum-full-size` from `main` at `e9f91ea`, the compiler pin
`e0bf150`, Node 22.22.0, Playwright 1.56.1 with its Chromium headless shell on SwiftShader,
4 cores shared with other jobs. Measured on 2026-10-20. The results are
`docs/experiments/dense-mvs/tum-fr3-structure-texture-far-640x480.json` and
`docs/experiments/dense-mvs/tum-fr3-structure-texture-far-fused-320x240.json`. The runner is
`scripts/compare-mvs-tum.mjs`, run with the same 2,568 landmarks as above.

What changed in the runner:

- `--working-mib N` sets the device's `workingBytes` (default 64). At 640 × 480 the depth stage
  of 15 views needs (16 · 15 + 4) · 307,200 B = 75 MiB, so the default 64 MiB refuses it with
  "Allocation exceeds TypeShade working budget" (3.8 gives 13 views as the limit at 64 MiB).
  The run used 256 MiB, the lab's budget at 1280 px.
- The full size needs no new flag: the default `--scale 1` is the sample's 640 × 480.
- `--fuse` runs `reconstructDenseGpu` (depth stage and fusion) in place of `mvsDepthMapsGpu`.
  The public result holds no full-size depth maps, so one run gives one of the two comparisons.
  The runner projects each fused point into all 16 views (all have a ground-truth depth image)
  and compares the z of the camera frame with the ground truth at the nearest pixel.
  `--completeness-step s` samples every s-th pixel of the grid for the completeness.

Result 1, depth at 640 × 480 (`--working-mib 256`), 15 of 16 views, 3,951,790 pixels with
ground truth, no alignment. The depth stage took 3,040 s (50.7 min), 4.5 times the 681 s at
320 × 240:

| Measure                                  | 640 × 480              | 320 × 240 (above)      |
| ---------------------------------------- | ---------------------- | ---------------------- |
| Median relative depth error              | 0.693 %                | 0.707 %                |
| Within 1 %, 2 % and 5 %                  | 66.4 %, 91.2 %, 97.8 % | 65.7 %, 90.7 %, 97.5 % |
| Valid fraction (a filtered depth exists) | 67.7 %                 | 81.9 %                 |
| Median scale (ground truth / estimate)   | 0.99997                | 0.9996                 |

The per-view medians are 0.59 % to 0.88 %, and the valid fractions 47 % to 78 %. Both targets
of 7.3 are met (at most 3 %, at least 50 % valid). Observation: the accuracy of the pixels
that survive is the same at both sizes, but the valid fraction is 14 points lower at 640 × 480
(views 6 and 14 are the lowest, 52 % and 47 %). The same default options (window radius 5,
the same iteration counts) cover half the scene extent at the larger size. Cause not tested.

Result 2, the fused points at 320 × 240 (`--scale 2 --fuse`), 15 views fused, 97,153 points.
The run took 992 s (the depth stage alone took 681 s above). Each point is projected into all
16 views. A pair is a point that lands on a pixel with ground truth (1,382,414 pairs). The
own-view row keeps the pairs of a point in the view that emitted it (91,746 pairs).

| Measure (relative depth error) | All views              | Own reference view     |
| ------------------------------ | ---------------------- | ---------------------- |
| Median                         | 1.13 %                 | 0.62 %                 |
| Within 1 %, 2 % and 5 %        | 45.9 %, 67.3 %, 79.1 % | 72.7 %, 96.0 %, 99.2 % |

Completeness: of the 1,052,566 ground-truth pixels in all 16 views, 49.9 % have a fused point
that lands on the pixel with a depth within 2 % of the ground truth, and 77.1 % have one within
one pixel of it (a point at the pixel or in its 3 × 3 neighbourhood).

Reading of Result 2:

- Fact: a fused point is correct in the view that emitted it (median 0.62 %, 96 % within 2 %).
- Fact: in the other views the same points err more (median 1.13 %, 20.9 % beyond 5 %). Those
  pairs include points that an occluder hides in that view, so the tail is not the depth error
  of the point. This run cannot separate the two (no visibility test).
- Fact: the fusion emits each surface once, in the first view that claims it. View 0 emits
  39,940 points, views 13 and 14 emit none, though they have 55,982 and 43,699 filtered depth
  pixels. So the radius-0 completeness depends on the splat density of the other views.
- Inference: the 77.1 % at radius 1 is the fairer completeness for a point cloud, and the 49.9 %
  at radius 0 a lower bound.

Dispositions:

- Closed: the comparison at 640 × 480 (Result 1). The gate stays at the 320 × 240 run.
- Closed: the fused points against the ground truth at 320 × 240 (Result 2). No gate is set on
  it: the first number sets the baseline.
- Open: the fused points at 640 × 480. The run (`--working-mib 256 --fuse
--completeness-step 2`) started at 18:12 UTC and ended at 19:07 UTC (55 min) with
  "Target page, context or browser has been closed", before it printed a result. The cause is
  not known (inference: the headless shell ran out of memory or the GPU process died during
  the fusion of 4.6 M candidate pixels). A second run did not fit the 90 minutes. Next action:
  run it on a physical GPU, or read the points back in chunks per view.
- Deviation: `node_modules` held a stale `typeshade` (no `recommendFp64Flavor`), so
  `npm run build:gpu` failed there. The build used the `node_modules` of an up-to-date sibling
  worktree at the same pin. The measurements are from that build.

#### Amendment 2: the depth range keeps the far field

Configuration: branch `fix/dense-depth-range` from `main` at `b0867f6`, Node 22.22.0, headless
Chromium on SwiftShader (no physical GPU), 4 cores shared with other jobs. Measured on
2026-10-21. The runner is `scripts/measure-mvs-range.mjs`.

Problem. An owner run on 2026-10-21 (physical GPU, water-tower sample) gave view 0 the range
3.19 to 8.49, while fused points reached a depth of about 34. The top quarter of the image was
1 % covered. The old range clipped everything beyond 1.25 × the 99th percentile of the depths
of the landmarks that the view observes. A far surface has few landmarks, so it fell outside.

Rule before: [0.75 × the 1st percentile, 1.25 × the 99th percentile] of the depths of the
landmarks that the view observes.

Rule after (3.3): [0.75 × the 1st percentile, 8 × the 99th percentile] of the depths of all
landmarks in the view's frustum (in front of the view, projecting into its image, observed by
the view or not). The 99th percentile and the factor 8 are `mvsFrustumDepths` and
`MvsViewOptions.farStretch`. Hypotheses stay uniform in inverse depth, so a far bound 8 times
larger costs no more samples and few of them lie in the far field. The CPU reference and
WebGPU share `planMvsViews`, so both use the same range and the same seeds. The minimum of 8
observed landmarks for a depth map, the gates of 7.1 and 7.2 and the seeds do not change.

Measured, 6 views, default options. "Valid" is the share of pixels with a
filtered depth, mean over the views. "Quarters" is the share of pixels that a fused point of the
whole cloud lands on within one pixel, by image row quarter, top to bottom, mean over the views.

| Sample (size, views)       | Rule   | View 0 range | Valid  | Quarters (top to bottom) | Fused  | Dense time |
| -------------------------- | ------ | ------------ | ------ | ------------------------ | ------ | ---------- |
| water tower (320 x 180, 6) | before | 3.19 - 8.49  | 80.1 % | 38 / 71 / 71 / 50 %      | 18,899 | 436 s      |
| water tower (320 x 180, 6) | after  | 3.19 - 54.41 | 79.5 % | 38 / 68 / 72 / 47 %      | 18,365 | 207 s      |
| graham-hall (640 x 427, 6) | before | 0.73 - 7.37  | 52.6 % | 17 / 36 / 33 / 24 %      | 29,389 | 1,598 s    |
| graham-hall (640 x 427, 6) | after  | 0.74 - 46.03 | 52.2 % | 18 / 35 / 31 / 23 %      | 27,793 | 963 s      |

Per-view valid share, water tower, before and after: 72.8 / 79.1 / 85.2 / 86.1 / 82.4 / 75.0 %
and 72.4 / 79.1 / 84.4 / 85.3 / 81.8 / 73.9 %. Graham-hall: 40.2 / 49.3 / 62.6 / 66.6 / 53.4 /
43.4 % and 40.0 / 49.1 / 62.2 / 66.1 / 52.9 / 43.1 %.

Reading:

- Fact: on these two samples, with SwiftShader and the first views, the far field does not
  show. The fused cloud of the water tower lies between depth 5.8 and 6.8 (50th to 99th
  percentile) before and after. The owner's coverage of 1 % in the top quarter did not appear
  at 320 x 180 (38 %), so this run does not reproduce the owner's image.
- Fact: the new rule costs a little on these samples: the valid share falls by 0.2 to 0.6
  points, and the fused points by 2.8 % (water tower) and 5.4 % (graham-hall). A wider
  range spends some hypotheses where nothing is.
- Fact: the dense times differ by a factor of 2 in the other direction. The runs shared
  4 cores with other jobs, so the times are not a comparison. No claim is made on time.
- Inference: the rule is a guard for scenes with surfaces beyond the last landmarks. The
  factor 8 reaches the depth 34 of the owner's run from the 99th percentile of about 6.8, with
  room. A factor of 4 (depth 27) does not.
- Not tested: the owner's run itself (needs the owner's GPU and full image size).

Dispositions:

- Closed: the range rule on the CPU reference and WebGPU, with a test of the frustum depths
  (`tests/mvs-views.test.ts`).
- Open: confirm on the owner's GPU that the top quarter of the water tower now has points.
  Next action: the owner runs the lab sample again.
- Open: a smaller factor (4) may keep the 0.2 point of valid share. Not measured on a sample
  with a far field.

#### Amendment 3: the dense stage runs within any device budget

Status: accepted, 2026-10-21, with its implementation in pull request #161. The owner's
rule for this amendment: the dense stage runs within any device budget. Size changes speed, not
success.

Problem. An owner run on 2026-10-21 (record `20261021…-de6c6079`, the pyramid sample at
1280 × 720, 25 registered views, an adapter with `maxBufferSize` and
`maxStorageBufferBindingSize` of 2 GiB) skipped the dense MVS: "More keyframes than the 17 views
the dense layout holds (section 4.2)". The cause has three parts:

- The depth stage of 4.2 keeps every view resident: (16 N + 4) bytes per pixel.
- The worker fixes `workingBytes` at 64 MiB × (edge / 640)², whatever the adapter reports.
- `ShadeDevice.open` requests the WebGPU default limits (`maxBufferSize` 256 MiB,
  `maxStorageBufferBindingSize` 128 MiB), not the adapter's.

So 25 views at 1280 × 720 needed 355 MiB against a budget of 256 MiB, on an adapter of 2 GiB.

Rule. These decisions replace the all-resident layout of 4.2 and 4.3:

1. **Residency is streamed.** The device holds the current reference's state, the views the
   current step reads, and the outputs it writes. A finished per-view output goes back to the
   host: the photometric depth and normal after the photometric pass, the filtered depth and the
   geometric normal after the filter. It comes back up only when a later step reads it: the
   geometric pass of a reference reads its sources' photometric depths, and the fusion of a
   reference reads its fusion views' filtered depths and normals.
2. **Window mode.** A depth pass keeps 1 + K view slots (K the most sources of one reference,
   at most 6): the reference and its sources. A slot holds a view's gray image, and in the
   geometric pass its photometric depth too. The slots are a cache: a view already in a slot
   stays there, and only a new view is uploaded and undistorted. The kernels of steps 2 to 4 run
   unchanged on the slots: `mvsControl.reference` is the reference's slot, `mvsSources` lists the
   sources' slots, and `mvsControl.state` (new, 0) indexes the reference's own arrays.
   `mvsControl.view` (new) is the packed index that the random draws hash (3.5), so a slot changes
   no draw.
3. **Streamed mode.** When the window does not fit, a pass keeps the reference and G < K
   sources at a time, and streams the sources through in groups. `mvsStreamCosts` (new) computes
   each hypothesis's source costs for the resident group and keeps the `bestSources` least costs
   of each hypothesis in `mvsPartial` (new, 9 hypotheses × `bestSources` f32 per element). `mvsStreamSelect` (new)
   then aggregates them and keeps the best hypothesis, in the order of 3.6. The candidates are
   regenerated from the other color, which no pass of this color writes, and from the
   counter-based draws, so each group sees the same candidates. The least costs of a set do not
   depend on the order of the groups, and the mean adds them in ascending order, as
   `hypothesisCost` does. So streamed mode gives the window mode's arrays bit for bit on the CPU
   tier. A propagation pass runs as candidate costs (all groups), selection, perturbed-candidate
   costs (all groups) and selection, since the perturbation reads the best hypothesis so far.
   When `mvsPartial` cannot hold a whole pass, the pass runs in chunks of elements (rows of the
   reference), and each chunk loads every group again.
4. **Fusion.** Each reference fuses against its fusion views: the `fusionViews` (default 50)
   other views that share the most landmarks with it, ties to the lower packed index. This is
   COLMAP's `check_num_images` (50, `StereoFusionOptions` in `src/colmap/mvs/fusion.h` at tag
   `4.2.1`, checked on 2026-10-21). With N ≤ 51 the set is every other view, as before. The fusion
   iterates the set in ascending packed index, through an order list after the slots' camera records
   in `mvsCameras` (`fuseControl.order`), and "a lower index" compares the packed index in element
   23 of each slot's camera record (it was 0). So the
   sums and the decisions are those of 3.10. In window mode the reference and its fusion views
   are slots of `mvsGeoDepth` and `mvsNormal`, a cache as in item 2. In streamed mode the
   reference stays in slot 0, the fusion views pass through in groups, and `mvsFuseAccumulate`
   (new) adds each group to per-pixel sums in `mvsFuseSums` (new, 8 f32 per pixel) for a chunk of
   pixels. `mvsFuseEmit` (new) writes the flags of the chunk, the shared scan gives the rows, and
   `mvsFuseScatterSums` (new) writes the points from the sums.
5. **The budget comes from the device.** `planMvsResidency` (`src/gpu/mvs-residency.ts`, new)
   sizes every array from two numbers: the budget, which is `device.workingBytes` less the
   active bytes when the MVS starts, and `device.bindingBytes`, the largest array. Each stage
   takes window mode when it fits, else streamed mode with the fewest chunks and then the most
   sources per group. No view count and no fixed MiB enter the plan. The lab worker opens its
   device with the adapter's `maxBufferSize` and `maxStorageBufferBindingSize`
   (`ShadeDeviceOptions.adapterLimits`, new), and sets `workingBytes` to the larger of the
   SfM's scale (64 MiB × (edge / 640)², × 4 with octave -1) and half of the device's
   `maxBufferSize` (`denseWorkingBytes` in `website/src/lib/dense-run.ts`). The half is the
   conservative fraction: WebGPU reports no memory size, and `maxBufferSize` is the largest
   buffer the device says it can make. The worker keeps every keyframe's RGBA, with no view
   limit, and `denseResult` carries the plan (`residency`), and the run record's `depthStageFormulaBytes` is the plan's bytes of the depth stage.
6. **The floor.** Streamed mode needs one source slot and one chunk of 64 elements: 32 bytes per
   pixel in the photometric pass, 36 in the geometric pass and 28 in the fusion, plus about
   7 KB. A budget below that is the one refusal, and its message names the bytes needed and the
   budget. Inference: the budget is at least half of 256 MiB (the WebGPU minimum of
   `maxBufferSize`) less the SfM's leftovers, so the floor is reached only above about
   3.7 megapixels per view. The lab's largest analysis size is 1280 × 960, 1.2 megapixels.
   A smaller floor needs the reference in row tiles with a halo, which this amendment leaves
   out.
7. **Host memory.** The host holds per view its RGBA (4 B per pixel, the worker's) and at most
   its photometric depth and normal and its filtered depth and normal (16 B per pixel). A
   photometric output is released after the last geometric pass that reads it, and a filtered
   output after the last fusion that reads it.

Byte math. "All-resident" is the layout of 4.2, (16 N + 4) bytes per pixel. The other columns
are `planMvsResidency` with K = 6, `fusionViews` 50, 600 seeds: the depth stage (window mode,
the geometric pass, the larger of the two) and the fusion, for a device that grants 2 GiB
(budget 1 GiB) and one with the WebGPU defaults (budget 128 MiB, arrays of 128 MiB). "Host" is
the most the host holds, 20 B per pixel per view. MiB throughout.

| Size, N         | All-resident | Depth stage | Fusion, 2 GiB | Fusion, defaults        | Host    |
| --------------- | ------------ | ----------- | ------------- | ----------------------- | ------- |
| 640 × 360, 12   | 43.1         | 16.7        | 29.0          | 29.0                    | 52.7    |
| 640 × 360, 25   | 88.8         | 16.7        | 51.9          | 51.9                    | 109.9   |
| 640 × 360, 50   | 176.7        | 16.7        | 95.8          | 95.8                    | 219.7   |
| 640 × 360, 100  | 352.4        | 16.7        | 97.6          | 97.6                    | 439.5   |
| 1280 × 720, 12  | 172.3        | 66.8        | 116.1         | 116.1                   | 210.9   |
| 1280 × 720, 25  | 355.1        | 66.8        | 207.5         | 123.2 (streamed, 8 + 1) | 439.5   |
| 1280 × 720, 50  | 706.6        | 66.8        | 383.3         | 123.2 (streamed, 8 + 1) | 878.9   |
| 1280 × 720, 100 | 1,409.8      | 66.8        | 390.3         | 123.2 (streamed, 8 + 1) | 1,757.8 |
| 1280 × 960, 12  | 229.7        | 89.1        | 154.8         | 126.7 (streamed, 4 + 1) | 281.3   |
| 1280 × 960, 25  | 473.4        | 89.1        | 276.7         | 126.7 (streamed, 4 + 1) | 585.9   |
| 1280 × 960, 50  | 942.2        | 89.1        | 511.1         | 126.7 (streamed, 4 + 1) | 1,171.9 |
| 1280 × 960, 100 | 1,879.7      | 89.1        | 520.5         | 126.7 (streamed, 4 + 1) | 2,343.8 |

"Streamed, 8 + 1" is 8 fusion views and the reference at a time, all pixels in one chunk. The
depth stage takes window mode in every row: (8 K + 28) bytes per pixel, independent of N. The
fusion stops growing at 51 views. The owner's run (1280 × 720, 25 views, 2 GiB) takes window
mode in every stage, 207.5 MiB at its peak, against a budget of 1 GiB.

What does not change: the kernels' arithmetic, the options and defaults of 3.2 to 3.11 (with
`fusionViews` new), the messages of 5.3, the gates of 7.1 and 7.2. The tests of 7.1 run window
mode and streamed mode on the same scene and hold them equal bit for bit.

Record. Configuration: branch `feat/dense-window` from `main` at `bd3c5a9`, the compiler pin
`e0bf150`, Node 22.22.0, Playwright 1.56.1 with its Chromium 141 headless shell on SwiftShader (no
physical GPU), a shared 4-core container. Measured on 2026-10-21. The times are wall times on that
shared machine, so a difference below about 20 % is noise.

Functional results:

- Fact: window mode gives `main`'s outputs bit for bit on the CPU tier. A script hashed the depth
  maps of `mvsDepthMapsGpu` (photometric, geometric and filtered, with costs and normals) and the
  points of `reconstructDense` on the 5-view scene of 7.1 and on a 9-view scene
  (`mvsWidePlacements(9)`), with the `main` build at `bd3c5a9` and with this branch. Every hash
  matched. A budget of 460,000 B (every stage streamed) matched too.
- Fact: `tests/typeshade/mvs-window.test.mjs` holds window mode, streamed mode with 2 sources or
  fusion views per group, and streamed mode at 470,000 B (1 at a time, a short last chunk) equal
  bit for bit on the CPU tier. It also runs the 9-view scene on a device of 1.5 MiB, below the
  1.73 MiB of the all-resident layout (7 views at most before this amendment).
- Fact: on WebGPU (SwiftShader) the 9-view scene, every stage streamed in 470,000 B, gives 3,883
  points against 3,873 on the CPU tier in window mode. 98.3 % and 98.6 % of the points of each
  tier have a point of the other within 1e-3 of their depth, by the bounds of 7.2
  (`tests/typeshade/mvs-webgpu.test.mjs`).
- Fact, a defect found and fixed on this branch: the shared flag scan reads every block of its
  arrays, so a short last chunk of the streamed fusion counted the flags of the chunk before it.
  The emit now writes the whole flag array. A chunk that divides the image evenly hid the defect
  on the first CPU test, so the test now takes a budget whose last chunk is short.
- Fact, a second defect found and fixed: with the order list in `mvsSources`,
  `mvsScatterPoints` bound nine storage arrays. WebGPU's default is eight per stage, and the
  pipeline was invalid. The order list now follows the camera records in `mvsCameras`.

Measured, south-building-24 (24 views, COLMAP's poses, the dataset's SIFT matches triangulated,
11,761 landmarks) at 160 × 120 (`--scale 4`), default options, `scripts/measure-mvs-range.mjs`
on SwiftShader. The all-resident layout needs (16 · 24 + 4) · 19,200 = 7,449,600 B (7.1 MiB). The
records are `docs/experiments/dense-mvs/south-building-24-160x120-*.json`.

| Device budget             | Photometric     | Geometric       | Fusion                        | Peak active bytes | Fused points | Dense time |
| ------------------------- | --------------- | --------------- | ----------------------------- | ----------------- | ------------ | ---------- |
| 512 MiB                   | window, 7 slots | window, 7 slots | window, 24 slots              | 4,382,408         | 9,565        | 538 s      |
| 4 MiB (13 views before)   | window, 7 slots | window, 7 slots | streamed, 17 + 1, whole image | 4,074,608         | 9,565        | 623 s      |
| 0.8 MiB (11 % of 7.1 MiB) | streamed, 1 + 1 | streamed, 1 + 1 | streamed, 1 + 1, 5,312 pixels | 838,816           | 9,565        | 1,309 s    |

- Fact: the three runs give the same fused positions, byte for byte, and every view has a depth
  map (mean valid share 69.5 %).
- Fact: the peak active bytes stay within each budget. The 4 MiB budget held at most 13 views in
  the all-resident layout.
- Fact: at 0.8 MiB the dense stage took 2.4 times as long as at 512 MiB: photometric 804 s
  against 353 s, geometric 477 s against 175 s, fusion 28 s against 9 s. The 4 MiB run shared
  the machine with a test run, so its 16 % over 538 s is not a measure of the streamed fusion.
- Fact, CPU tier, the 5-view scene with a 5 × 5 window and 1 iteration of each pass
  (`tests/typeshade/mvs-window.test.mjs`): window mode 26 s, 2 sources per group 40 s, and
  460,000 B 94 s. A later run under load took 50 s, 69 s and 137 s (at 470,000 B).
- Inference: the time of streamed mode comes from the regenerated hypotheses and the reference
  window statistics that each group computes again, and from the uploads of each group's views.
  It does not change a result.

Dispositions:

- Closed: the view limit of 4.2 and the skip of the worker. The owner's run (1280 × 720, 25 views,
  2 GiB) needs 207.5 MiB in window mode against a budget of 1 GiB (inference from the plan, not
  run on the owner's GPU).
- Open: the floor of item 6, below which the run is refused. It breaks the owner's rule (no
  failure path at a limit). Next action: row tiles of the reference with a halo, tracked in
  issue #160 with the other stages.
- Open: the lab run of the owner's record on a physical GPU. Next action: the owner runs the
  pyramid sample at 1280 × 720 again.
- Not run: the lab page itself (`?dense=1`) on this branch. The worker's dense path is covered by
  `website/dense-run.test.mjs` with a fake MVS.

### Cleanup record

Configuration: branch `feat/dense-cleanup` from `main` at `43ffd73`, the compiler pin `e0bf150`,
Node 22.22.0. Measured on 2026-10-20 on a shared 4-core container with no physical GPU. The
WebGPU tier is SwiftShader in headless Chromium 1194. The input is a dense PLY that the owner
exported from the lab, 209,283 points (`scripts/measure-cloud-clean.mjs`).

| Stage                     | Points after | CPU reference (ms) | WebGPU on SwiftShader, warm (ms) |
| ------------------------- | ------------ | ------------------ | -------------------------------- |
| Input                     | 209,283      | -                  | -                                |
| Outliers (k 16, ratio 2)  | 204,541      | 2,947              | 42,278                           |
| Voxel (size 0.0442)       | 113,921      | 180                | 196                              |
| DBSCAN, largest cluster   | 67,579       | 987                | 433                              |
| DBSCAN, `minFraction` 0.1 | 82,968       | 409                | not measured                     |

- The CPU and WebGPU tiers give the same count after each stage. DBSCAN finds 237 clusters with
  eps 0.131.
- The lab keeps every cluster of at least 10 % of the largest (`minFraction: 0.1`). Here that
  keeps 2 clusters. The library default stays 0, the largest cluster only.
- Against scikit-learn 1.9.1 on a 3,072-point cloud (`scripts/compare-cloud-clean.mjs`): the
  worst relative difference of the kNN mean distance is 4e-16, the DBSCAN labels do not differ,
  and the voxel counts are 172 and 172.

Update of 2026-10-20 (branch `perf/cloud-knn`, PR #129, base `5eee5cd`): the GPU kNN kernel
was changed (see 3.12) and the grid fix below was made. The same 209,283-point PLY, the same
machine (shared 4 cores, load average 8 to 12 from other jobs, so the times vary by 30 % or more
between runs), warm WebGPU run:

| Stage                    | Before (ms) | After (ms)     | CPU reference after (ms) |
| ------------------------ | ----------- | -------------- | ------------------------ |
| Outliers, 209,283 points | 42,278      | 2,700 to 5,400 | 2,300 to 2,550           |
| Outliers, 20,000 points  | 1,225       | 430 to 760     | 170 to 190               |

- The GPU is 8 to 15 times faster than before and about 1.5 to 2 times slower than the CPU
  reference. The two tiers give the same count after each stage (204,541, 113,921, then 67,580
  points and 235 clusters) and the same threshold. The worst relative difference of the kNN
  mean distance is 2.2e-7 on all 209,283 points (limit 1e-6).
- The DBSCAN result moved from 67,579 points and 237 clusters to 67,580 and 235 on the CPU and
  the GPU tier. Cause: the grid put a point of the minimum on an axis in cell 0 by rounding, and
  the lookups skip cells below 1. The cell is now raised to 1. The outlier and voxel counts did
  not change.
- Cause of the 42 s (inference, the changes were not measured one by one): the kernel copied a
  32-entry list by value on each insertion, walked the whole cube of cells of every ring and
  hashed each of its cells, and the invocations of a group worked on points of unrelated cells
  (the sort is by hash bucket), so a group waited for its slowest lane. Far points took a pass
  over all points, 1,105 of them here.
- Why the GPU still loses (inference): SwiftShader runs the kernel on the same 4 cores as the
  CPU reference, which is one thread with early exits and no SIMD padding. The kernel keeps a
  32-entry private list per invocation and reads the cells through the buckets. The rate on a
  physical GPU is not measured, so the lab keeps its default path for the cleanup. This change
  has no measured reason to switch it.

Deviations and open items:

- Closed: the voxel probe loop ran over the whole table after it found the voxel owner. It took
  27 s for 20,000 points on SwiftShader. It now stops at the owner (16 ms warm).
- Updated: the GPU kNN took 42 s on SwiftShader against 2.9 s on the CPU. It takes 2.7 to 5.4 s
  now (see the update above). It is still slower than the CPU on SwiftShader. Next action: measure
  on a physical GPU.
- Not tested: Open3D's own functions.

### Mesh record

Configuration: branch `feat/dense-mesh` from `main` at `281dbb5`, kernels at `4b784ef`, the
compiler pin `e0bf150`, Node 22.22.0. Measured on 2026-10-20 on a shared 4-core container with no
physical GPU. The WebGPU tier is SwiftShader in headless Chromium (Playwright). The input is the
dense PLY that the owner exported from the lab, 209,283 points with normals, cleaned on the CPU
tier with the lab's options (`minFraction` 0.1) to 82,968 points (`scripts/measure-dense-mesh.mjs`).
The defaults give a lattice spacing of 0.0483 (1.5 times the median spacing 0.0322) and R = 0.121.

| Stage                          | CPU reference (ms)  | WebGPU on SwiftShader, warm (ms) |
| ------------------------------ | ------------------- | -------------------------------- |
| Lattice (bricks, active nodes) | 274                 | 279                              |
| Field                          | 266                 | 1,414                            |
| Cubes (classify, scan, emit)   | 109                 | 636                              |
| Weld and vertices              | 65                  | 89                               |
| Total                          | 680 to 730 (3 runs) | 2,467                            |

- Both tiers give 3,722 bricks, 544,713 active nodes, 115,265 cubes, 133,605 vertices and 234,829
  faces, and the same index array.
- Edge use on the lab mesh: 333,557 edges are shared by two faces, 37,373 by one (the open
  surface and the holes of the cloud) and none by more than two.
- Spacing factors 1 and 2 give 327,080 and 153,318 faces in 810 ms and 580 ms on the CPU tier.
- Sphere test (4,000 points, spacing 0.0818): the worst distance of a vertex to the sphere is
  0.0053, 0.064 of a voxel. Every edge is shared by two faces, V − E + F = 2, and the volume is
  1.011 times 4π/3.
- The device equals the CPU reference on the CPU tier and on SwiftShader: the same faces, the
  worst position difference 5e-6 (the bound is 1e-5), the colors within 1.

Deviations and open items:

- Open: on SwiftShader the kernels are slower than the CPU reference (2.5 s against 0.7 s). The
  field and the classification are small kernels over 0.5 million nodes, and SwiftShader runs
  them on the same 4 cores. The rate on a physical GPU is not measured.
- Open: Bourke's table is not manifold by construction in its ambiguous cases. Lewiner's tables
  would remove that.
- Open: the weld of the keys is on the host (0.1 s here). A GPU hash would remove the read of
  the key rows.
- Not done: a visual check of the mesh in the lab viewer on a real dense run (20 minutes on
  SwiftShader). The viewer test covers the switch and the data attributes.

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

**More than 13 views.** South-building-24 does not fit the all-resident layout. Amendment 3 in
section 8 keeps a window of each reference's sources on the device and the depth maps on the
host, so the number of views no longer limits the run.

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

## 10. Dense quality overhaul (proposed)

Status: proposed, 2026-10-21. The owner approved the plan and its order on 2026-10-21. This
section is a design record. Step 0 (the evaluation harness) is implemented. Steps 2 to 4 below
are not implemented, and a name marked _(new)_ does not exist yet. Nothing here changes a
contract of 3.1 to 3.13 until a step is accepted. The old fusion (3.10) and the old cleanup
(3.12) stay selectable for comparison.

**Out of scope (owner, 2026-10-21): learned models.** No step uses a trained network for depth,
confidence, segmentation or filtering. typenn is not used in typecv at this stage. A step that
needs a learned component is a new proposal.

### 10.1 Problem

Run `dfc97665` (Desert canyon video sample, 1280 px, 41 views, site `8d06a38`) fused 1,082,026
dense points. The cleanup of 3.12 kept 565,272:

| Stage               | Removed | Cause (observed)                                           |
| ------------------- | ------- | ---------------------------------------------------------- |
| Statistical outlier | 28k     | one k-NN distance threshold for the whole cloud            |
| Voxel downsampling  | 421k    | one voxel size, 2 × the median spacing                     |
| DBSCAN, 338 found   | 67k     | 2 clusters kept: the rule keeps those of ≥ 10 % of largest |

Observed on the exported PLYs: 28.7 % of the sparse landmarks have no dense point within 0.2,
and along the length of the canyon one band has 100 % missing. The same rule dropped the water
tower in run `4a8aa2ff` (1 of 101 clusters kept) and trees in run `0de918f8`.
Inference: a global threshold treats a naturally sparser far region as noise, and a long scene
falls into clusters that the 10 % rule then discards. A change of the rule's constants moves the
loss and does not remove its cause. Hence this overhaul.

### 10.2 Step 0: evaluation (implemented)

`src/reconstruction/dense-score.ts` scores a cloud against a ground-truth cloud with the
definitions of ETH3D and Tanks and Temples, at stated thresholds τ (meters):

- accuracy(τ): the fraction of reconstructed points with a ground-truth point within τ.
- completeness(τ): the fraction of ground-truth points with a reconstructed point within τ.
- F-score(τ) = 2 · accuracy · completeness / (accuracy + completeness).

Two approximations of the benchmarks' masks, both stated in each record:

1. The ground truth is the laser scan limited to the points that at least two cameras see, with
   a depth buffer for occlusion (`visibleGroundTruth`).
2. A reconstructed point farther than `maskRadius` (0.5 m) from every ground-truth point is not
   judged (a background that the scan does not cover). The record gives their count.

The regression metric of `dfc97665` is `landmarkCoverage`: the fraction of sparse landmarks with
no dense point within k × the expected spacing at the landmark, expected spacing being depth /
focal at the analysis size, in the nearest view that observes it. Default k = 4. A far landmark is
held to a wider radius than a near one, so a sparser far region is not a miss by construction.

Data: ETH3D high-resolution multi-view training scene `pipes` (14 undistorted DSLR images of
6220 × 4141 px, COLMAP calibration in the frame of the scan, a laser scan of 11.5 M points). It
downloads from `https://www.eth3d.net/data/` as `pipes_dslr_undistorted.7z` (145 MB) and
`pipes_dslr_scan_eval.7z` (54 MB). The sparse model is the dataset's: its cameras, and its
triangulated points as landmarks. Tanks and Temples and DTU were not fetched. Command:
`node scripts/eth3d-dense.mjs --scene <dir> --scan <dir> --out <dir> --scale 8`. Tiers: headless
Chromium on SwiftShader. Tests: `tests/dense-score.test.ts`.

### 10.3 Step 2: depth confidence before fusion

Data. Each depth map gets a confidence map: u8 per pixel, 0 for no depth, 1 to 255 for c in
(0, 1]. It lives on the host with the filtered depth (the streamed layout of Amendment 3 already
moves that depth), so it costs P bytes per view on the host and 4 P bytes on the device for the
reference being filtered (`mvsConfidence` _(new)_, f32).

Computation, in `mvsFilter` _(extended)_, from values the passes already hold:

1. Photometric term c_p = clamp(1 − cost / `maxCost`, 0, 1), with the aggregated cost of the
   chosen hypothesis and `maxCost` the filter's cost bound of 3.9.
2. Geometric term c_g = n_ok / n_src: the sources whose forward-backward check of 3.8 passes,
   over the sources that see the pixel (ACMM and ACMMP use the same count).
3. View-normal term c_a = clamp((cos θ − 0.15) / 0.55, 0, 1), θ the angle between the pixel's
   normal and the direction to the camera. A grazing surface carries a depth that moves by
   many pixels per unit of normal error.
4. c = c_p · c_g · c_a. A pixel with c below `minConfidence` _(new option)_ loses its depth.
   The default is set by the measurement of this step and recorded. The fusion (3.10) then
   counts a consistent view only when its own confidence is at least `minConfidence`, and the
   confidence of the fused point is the mean over the views it merges.

Budget. The new device array is 4 P bytes of the reference only, so the formulas of 4.2 and the
streamed mode of Amendment 3 change by that term and nothing else. No limit refuses a run
(issue #160): the existing window and streamed modes apply.

Proof. On `pipes` at `--scale 8`, with the raw cloud: the F-score at τ = 2 cm does not fall, and
the accuracy at τ = 2 cm rises or stays within 0.5 point; the completeness at 5 cm falls by at
most 2 points; `landmarkCoverage` does not worsen. If a number is missed, the record says so.
CPU and WebGPU tiers agree on the confidence maps within 1 level in 99 % of pixels.

### 10.4 Step 3: GPU TSDF fusion with voxel hashing

Replaces the per-pixel fusion (3.10) as an option `fusion: 'tsdf'` _(new)_. The default stays
`'pixel'` until the step's record shows the TSDF is better.

Data. A hash table of blocks of 8 × 8 × 8 voxels (Nießner et al. 2013). The key is the block
coordinate (ix, iy, iz) and the level. The table is open addressing with
`atomicCompareExchangeWeak` on a key word (as `voxelClaim` does in 3.12), at least 2 × the
blocks. A voxel holds six u32 words (24 B), all updated with `atomicAdd` since WGSL has no float
atomics: `sumWSdf` (fixed point), `sumW`, three weighted color sums and the free-space count of
`tsdfCarve`. A block of 512 voxels is 12 KiB. The layout is final in the first commit of the step and is recorded in 4.1.

Voxel size adapted to distance. Level L has voxel size v_L = v_0 · 2^L. A depth sample at depth d
with focal f has the footprint d / f. It goes to the level with the least v_L that is at least that
footprint, so a far region is integrated at a coarser level and is not lost, and a near region
keeps its detail. v_0 is the footprint of the nearest valid depth of the scene (a percentile).
The truncation is μ_L = 4 v_L. Extraction skips a voxel of level L when the finer level has
weight there.

Kernels _(new)_, all chunked through `device.chunkSize` like the fusion:

1. `tsdfAllocate`: one invocation per pixel with depth. It claims the blocks of the band
   [d − μ, d + μ] along the pixel's ray.
2. `tsdfIntegrate`: one invocation per pixel. It walks the band and adds weight w = confidence
   (10.3) times the cosine term, and the signed distance clamped to ±μ.
3. `tsdfCarve`: one invocation per pixel. It walks the ray from the near plane to d − μ and adds
   one free-space count to each existing voxel. It allocates nothing. A voxel with a free count
   over `carveRatio` times its weight is deleted. This removes floaters.
4. `tsdfExtract`: one invocation per occupied voxel with weight ≥ `minWeight`. It emits a point at
   the zero crossing toward each +x, +y, +z neighbour, with the interpolated position, the
   weighted color and the normal from the signed-distance gradient. The meshing of 3.13 reads
   these points through the same path as it reads the cleaned cloud.

Memory and the budget rule (#160). The device never refuses at a limit. The host counts the
blocks with an allocation-only dry run. If the table and voxel arrays fit `workingBytes`, one pass
runs. Otherwise the volume is cut into slabs along its longest axis, sized to fit. Each slab runs
the four kernels over all depth maps (read again from the host, as the fusion does) and its
points go to the host before the next slab. Points of one slab are extracted only from blocks
inside it, so no point repeats. Speed changes with the budget. The result does not.

Proof. On `pipes`: F-score at τ = 2 cm of the TSDF cloud is not below the old fusion's, at
equal or fewer points; `landmarkCoverage` is lower than the old fusion's with the old cleanup; a
test that forces slabs of 1/4 of the volume returns the same points as one pass (as a set, to
1e-6). The CPU tier is the reference: a CPU function with the same arithmetic and a test of the
kernels against it on the synthetic scene of 7.1.

### 10.5 Step 4: visibility graph-cut labeling

Replaces the cluster rule of 3.12 (DBSCAN and "10 % of the largest"). The statistical outlier
filter and the voxel filter of 3.12 are replaced by this labeling when it is on. The old chain
stays selectable.

Idea (Labatut et al. 2009, Vu et al. 2012, adapted to points). Label each extracted point
`surface` or `noise`. The energy is a sum of unary terms and a pairwise smoothness term, minimized
exactly by a min-cut.

Graph. A node for each point of 10.4. Edges join the k = 8 nearest neighbours (the grid of 3.12),
with weight w_ij = exp(−d²/2σ²) · max(0, n_i · n_j), σ the local spacing, so a surface patch
holds together and a point that is far from or turned away from the others does not.

Unary terms.

1. Confidence: the mean confidence of 10.3 of the merged samples.
2. Visibility: the weight of free-space counts (10.4, `tsdfCarve`) against occupied weight in
   the point's voxel. A point that many rays pass through is noise.
3. Sparse support: a sparse landmark within its radius (10.2) gives the point a surface bias. The
   landmark is an independent measurement.

A connected component of surface points under the pairwise edges is kept when it has either at
least `minComponent` points or a sparse landmark. The relative rule "10 % of the largest" does
not exist in this step. Solver: Boykov-Kolmogorov max-flow on the CPU first. A GPU solver is a
later item and is not promised here.

Proof. The record of the step gives, on `pipes` and on the stored `dfc97665` clouds: the F-score
at τ = 2 cm, `landmarkCoverage`, the kept component sizes and the points removed by each term.
The step is accepted when a whole band of the scene is no longer lost (the regression metric
at most 10 % on `dfc97665`'s sparse landmarks, from 28.7 %) at an accuracy no lower than 3.12's.

### 10.6 What replaces 3.12

When step 4 is accepted: the chain "outliers, voxels, clusters" is replaced by "TSDF extraction
(10.4), labeling (10.5)". `cleanDenseCloudGpu` stays for the pixel fusion and for comparison.
The lab's option names are fixed in the step's pull request.

### 10.7 Measurements per step

Each step records, on the same machine and the same scene: the scores of 10.2 before and after
with cleanup on and off, the regression metric, the time of the stage, the peak device bytes
(`MvsResidency.peakBytes` or the step's own count) and the tests. A step that does not improve
the scores is reported with its numbers and is not hidden.

### 10.8 Step records

#### Step 0 record: harness and baseline

Configuration: branch `feat/dense-quality`, PR #177, base `8d1c4de`, Node 22, headless Chromium
1194 on SwiftShader (no physical GPU), a shared container with a load average of 6 to 17 from other
jobs (times vary by 30 % or more). Date: 2026-10-21. Scene: ETH3D `pipes`, 14 views, scale 8
(778 × 518 px), 2,473 landmarks from the dataset's COLMAP model, defaults of
`reconstructDenseGpu` (8 photometric and 4 geometric iterations, `minConsistentViews` 3). The
ground truth is the scan (11,482,717 points) moved by `scan_alignment.mlp`, reduced to one point
per 4 mm cube (3,839,672) and limited to the 1,619,417 points that at least 2 cameras see.
Mask radius 0.5 m, landmark radius 4 pixel footprints. Command:
`node scripts/eth3d-dense.mjs --scene … --scan … --out … --scale 8`.

| Cloud                           | Points  | Acc 1 cm | Compl 1 cm | F 1 cm | F 2 cm | F 5 cm | F 10 cm | Landmarks uncovered |
| ------------------------------- | ------- | -------- | ---------- | ------ | ------ | ------ | ------- | ------------------- |
| Raw fused (cleanup off)         | 156,751 | 73.8     | 21.9       | 33.7   | 45.9   | 59.6   | 67.4    | 42.1 %              |
| Lab cleanup (`minFraction` 0.1) | 70,302  | 77.5     | 16.5       | 27.2   | 34.5   | 41.7   | 47.7    | 61.3 %              |

Accuracy at 2, 5 and 10 cm: raw 83.3, 92.3 and 95.5; cleaned 87.6, 95.9 and 98.1.
Completeness at 2, 5 and 10 cm: raw 31.7, 44.0 and 52.1; cleaned 21.5, 26.7 and 31.5.
Time: 2,946 s in all (photometric 2,096 s, geometric 677 s, fusion 173 s). Peak device bytes
(`MvsResidency.peakBytes`): 59,696,376. Cleanup (CPU tier): 2.1 s.

- Observed: the lab cleanup removes 55 % of the points (outliers 3,386, voxel 60,641, clusters
  22,422 with 11 of 362 clusters kept). It raises accuracy by 1 to 4 points and loses 5 to 21
  points of completeness. Its F-score is lower than the raw cloud's at every τ. The share of
  sparse landmarks with no dense point near them rises from 42.1 % to 61.3 %.
- Observed: the same direction as in run `dfc97665`, on a scene with a scan for ground truth.
  So the cleanup's loss is a loss of surface, not only of noise.
- Observed: the raw cloud itself leaves 42.1 % of landmarks uncovered. Fusion's rule of 3
  consistent views and the filter are part of that. Step 3 addresses it.
- Regression metric on the stored `dfc97665` clouds (the cleaned dense PLY of 565,272 points and
  the sparse PLY), with a fixed radius of 0.2: 4,194 of 14,732 landmarks uncovered, 28.5 %. The
  owner's figure is 28.7 %. The difference is below the rounding of what each count includes
  (inference: the cameras in the sparse PLY).
- Observed, second owner record `20261021145305440-66f7f868` (Desert arch video sample, 48
  frames, 1280 px, NVIDIA Turing, site `4f6b49a`, record v6, 13 views): fusion 1,350,966 points,
  and the 3.12 cleanup gives 1,299,164 (outliers), 785,401 (voxel), 359 DBSCAN clusters of which 1
  is kept, 763,727 points. The same failure as `dfc97665`. Dense total 192.6 s: photometric
  16.3 s, geometric 8.3 s, fusion 162.1 s. The per-kernel GPU rows give `mvsPropagate` 25.7 s,
  `mvsFuseFlags` 0.89 s and `mvsScatterPoints` 0.94 s, with 36,482 and 36,480 dispatches. So the
  fusion's 162 s hold about 1.8 s of GPU time. Inference: the current fusion is bound by the
  host (the chunked dispatches, the flag scan and the readback of each reference's arrays, 905 MB
  of uploads in the run), not by the GPU. Step 3's TSDF fusion must be measured against these
  162 s and 1.8 s, and it should cut the dispatch count by integrating a whole depth map in few
  dispatches. The adapter reports `shader-f16`, `subgroups`, `subgroup-size-control`, 32 KiB of
  workgroup storage, 1024 invocations and 16 storage buffers per stage. The kernels of step 3 may
  use them where TypeShade allows (not verified here).
- Not run: Tanks and Temples and DTU (ETH3D was reachable and sufficient). Not run: the scoring
  on the WebGPU cleanup tier (the CPU tier gives the same clouds, 3.12).
- Open: the ground truth is the laser scan and its masks are approximated (10.2). Scores are
  comparable between steps and are not the benchmark's official numbers.
