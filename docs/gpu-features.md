# GPU image features

TypeCV includes an experimental single-scale feature frontend implemented with
TypeShade compute kernels. `extractFeaturesGpu` converts RGBA pixels to
grayscale, scores FAST-9 corners, keeps up to four high-scoring candidates in
each 24 × 24 tile, estimates orientation from the intensity centroid, and
computes a deterministic 256-bit rotated BRIEF descriptor. `matchFeaturesGpu`
calculates Hamming distances with `countOneBits`, applies a nearest/second
nearest ratio in both directions, and returns mutual matches.
Extraction supports frames from 48 × 48 through the active device's storage and
dispatch limits, with a maximum of 2,000 selected features per frame. Matching
has no cap on pairwise comparisons: it rejects only inputs above the device's working-memory
and dispatch limits.

```ts
import { ShadeDevice, extractFeaturesGpu, matchFeaturesGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const first = await extractFeaturesGpu(device, firstRgba, width, height);
  const second = await extractFeaturesGpu(device, secondRgba, width, height);
  const correspondences = await matchFeaturesGpu(device, first, second);
  // correspondences.matches is [firstFeature, secondFeature, ...].
} finally {
  await device.dispose();
}
```

The matcher returns final feature indices and Hamming distances to the host;
frame decoding, RGBA packing, camera graph control, and current mapper inputs
are host work. The pixel tests and descriptor math run on the selected
TypeShade tier. The browser video lab explicitly requests WebGPU and does not
retry on CPU. The existing incremental mapper then runs GPU geometric
verification, PnP, triangulation, filtering and bundle adjustment.

This is an initial CV operator, not a COLMAP-equivalent feature frontend. It is
single-scale and does not model lens distortion. RTX 2080 measurements and the
new resident/tiled APIs are described in [GPU scaling](gpu-sfm-scaling.md).
Textureless scenes, repeated patterns, rolling shutter, motion
blur, weak parallax, inaccurate focal assumptions, or very long video can
produce too few matches, unregistered frames, or misleading scale. Inspect the
reported reprojection error, camera path, unregistered views, and source video;
the sparse viewer is not a dense surface reconstruction.

## Scale-space SIFT

`extractSiftGpuResident` is the scale-space front end of the engine v2 direction
([engine-v2.md](engine-v2.md)): a Gaussian scale space with three intervals per
octave (sigma 1.6, k = 2^(1/3)), differences of Gaussians, refined 3 × 3 × 3 extrema
with a peak threshold (0.02 / 3 on intensities in [0, 1]) and an edge ratio (10),
up to two dominant orientations from a smoothed 36-bin gradient histogram, and
4 × 4 × 8 descriptors with trilinear binning, normalization and 0.2 clipping. By
default the clipped descriptor is then L1-normalized and square-rooted (RootSIFT,
Arandjelović and Zisserman 2012), so that the matcher's Euclidean distance is the
Hellinger distance of the reference descriptors; `rootSift: false` keeps the reference
descriptor. The descriptors stay resident as 128-d f32 and match with
`matchFloatDescriptorsGpu`.
The option `clahe: true | { tiles, clipLimit }` runs contrast-limited adaptive histogram
equalization (after scikit-image's `equalize_adapthist`) on the gray image before the scale
space, for bright or low-contrast frames. The defaults are 8 × 8 tiles, 256 bins and a clip
limit of 0.01 (the largest bin count as a fraction of the pixels of a tile). It is off by
default, and then the result does not change. `equalizeGrayGpu` and its CPU reference
`equalizeGrayCpu` expose the pass alone. The tile kernels run one invocation per tile.
It differs from scikit-image in two ways: tiles are `ceil(size / tiles)` pixels with a smaller
last tile instead of a padded image, and the input and output are not rescaled to their
own minimum and maximum.
Every kernel is one invocation per element without workgroup memory, so the CPU
tier runs the same code as WebGPU; the shared flag scan compacts the extrema of
each interval and the orientations of each octave on the device, and the whole
image is one plan with one count readback.

```ts
import { ShadeDevice, extractSiftGpuResident, matchFloatDescriptorsGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const first = await extractSiftGpuResident(device, firstRgba, width, height);
  const second = await extractSiftGpuResident(device, secondRgba, width, height);
  const keypoints = await first.readKeypoints(); // [x0, y0, x1, y1, …] in image pixels
  const matched = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
    ratio: 0.8,
  });
  await first.dispose();
  await second.dispose();
} finally {
  await device.dispose();
}
```

Options: `octaves` (default: until the smallest side reaches 16 pixels, at most 4),
`sigma`, `peakThreshold`, `edgeThreshold`, `maxFeatures` (capacity of oriented
keypoints, default 4096, at most 16384; extrema beyond it are cut in octave,
interval, pixel order and counted in `requested`) and `maxOrientations` (1 or 2).
Frames run from 32 × 32 up to the device's binding and working budgets; the scale
space needs about 13 images of the frame's size in f32.

Measured on the COLMAP south-building previews (640 × 480, SwiftShader in this
container, 2026-10-19): 779 to 998 keypoints per image, 1.1 s per image on the
software device, and 238, 124 and 238 ratio-test matches for the pairs (0, 1),
(0, 2) and (1, 2). The corner front end gave 12 to 43 matches on the same pairs
with the video controls and verified zero inliers two frames apart. COLMAP's SIFT
on the 3072 × 2304 originals with about 8,000 keypoints per image gives 1679, 923
and 1785 raw matches; the comparison at equal resolution is in the lab's run
records.

## SLIC superpixels

`slicGpu(device, rgba, width, height, options?)` segments an 8-bit sRGB image into superpixels
(Achanta et al. 2012). `slic(rgba, width, height, options?)` is its CPU reference, in double
precision. Both return `{ width, height, labels, count }`: a `Uint32Array` of row-major labels in
`0 .. count - 1`. The defaults are those of scikit-image's `segmentation.slic`: `nSegments` 100,
`compactness` 10, `maxIterations` 10 and `enforceConnectivity` true (with `minSizeFactor` 0.5).

Method, in the order the kernels run (`src/gpu/slic.shade.ts`):

1. `slicToLab`: sRGB to CIELAB (D65, L in [0, 100]), one invocation per pixel.
2. The grid: `S = sqrt(pixels / nSegments)`, then `round(width / S) × round(height / S)` cells. So the
   cluster count is near `nSegments`, not equal to it. Cluster `k` starts at the center of cell
   `k`.
3. `slicSeeds`: each center moves to the pixel of lowest Lab gradient in the 3 × 3 pixels around it.
   The first pixel in scan order wins a tie. The image border is never chosen.
4. `slicAssign`: each pixel takes the nearest of the clusters in its own cell and the eight cells
   around it. The distance is `dLab² + (compactness / S)² · dXY²`, as scikit-image. This is the
   form of gSLICr: a cluster only owns pixels of the 3 × 3 cells around its cell.
5. `slicUpdate`: one invocation per cluster gathers the pixels of those cells that carry its label
   and writes their mean Lab color and position. The sums run in a fixed order, so there are no
   atomics, the result repeats, and the CPU tier runs the same code as WebGPU. A cluster with no
   pixel keeps its center. Steps 4 and 5 run `maxIterations` times.
6. On the CPU: `enforceConnectivity`, as scikit-image. Every label is split into its 4-connected
   components. A component under `minSizeFactor` of the mean segment size joins the component
   left of or above its first pixel. Then the labels are numbered in scan order, so each label
   is one 4-connected component. This pass is cheap (13 to 33 ms at 640 × 480, below).

Every long pass runs as chunks of the device's time budget (`chunkedPass`, keys `slicLab`,
`slicSeeds`, `slicAssign` and `slicUpdate`), and the CPU tier runs each pass as one chunk.

Differences from scikit-image: the seeds move to the lowest gradient (scikit-image does not
move them), the grid is `round(size / S)` cells per side instead of a fixed `S` spacing, and the
search window is the 3 × 3 cells instead of 2S × 2S around each center. The distance and the
defaults are the same. scikit-image is not installed in this container, so no boundary recall
was compared against it. The tests below hold the recall against synthetic ground truth.

Tests (`tests/typeshade/slic.test.mjs`, `tests/typeshade/slic-webgpu.test.mjs`): on an image of
six colored rectangles with noise, the boundary recall at 2 px is at least 0.95, the count is
near `nSegments`, every label is one 4-connected component, and the device labels equal the CPU
reference labels on at least 99 % of the pixels. The remaining pixels are, by inference, f32 ties against double
ties at a segment edge. Two flat halves give exactly two labels on both.

Measured on frame 0 of the COLMAP south-building sample, resized to 640 × 480
(`node scripts/benchmark-slic.mjs`), 2026-10-20, commit `627653c` plus the script, a shared
4-core Xeon container with no physical GPU (WebGPU is SwiftShader in headless Chromium).
Median of 3 runs after one warm run, milliseconds per call:

| Stage                                                | CPU reference | CPU tier of the device | WebGPU (SwiftShader) |
| ---------------------------------------------------- | ------------- | ---------------------- | -------------------- |
| Setup (upload, Lab, seeds, one assignment, readback) | 56            | 237                    | 39                   |
| One iteration (assign + update)                      | 25            | 120                    | 39                   |
| Connectivity (CPU, after the readback)               | 19            | 33                     | 13                   |
| Total, 10 iterations and connectivity                | 326           | 1469                   | 446                  |

The stages are differences of whole calls: the call with no iteration, the slope of ten
iterations, and the call with connectivity against the call without. The call gave 80 segments
for `nSegments` 100 (108 grid cells before connectivity). The WebGPU labels equal the CPU
reference labels on 99.996 % of the pixels. On this textured photograph about 14,000 small
fragments of the 108 labels exist before the merge, so the merge removes 28 labels. Open: the
count is lower than `nSegments`. A higher `compactness` or a lower `minSizeFactor` keeps more.

```ts
import { ShadeDevice, slicGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const { labels, count } = await slicGpu(device, rgba, width, height, { nSegments: 200 });
} finally {
  await device.dispose();
}
```

The proposed use for the dense MVS is in `docs/dense-mvs.md`, section 3.14. It is not built.

## Straight lines: Canny and Hough

`cannyGpu(device, rgba, width, height, options?)` finds edges. `houghLinesGpu(device, rgba, width,
height, options?)` finds straight lines and segments. `canny` and `houghLines` are their CPU
references, in double precision. The input is 8-bit RGBA. Both Hough functions return the edge map
(`edges`, `count`), the accumulator, `lines` (`rho` in pixels, `theta` in radians, `votes`) and
`segments` (`x1, y1, x2, y2` and the index of the `line`). A line is `x cos(theta) + y sin(theta) =
rho` with `theta` in [0, pi), so `theta` is the angle of the line's normal (the OpenCV convention).

Defaults: `sigma` 1; `thresholdMode` `'quantile'` with `lowThreshold` 0.85 and `highThreshold`
0.95; `thetaBins` 180 and rho step 1 px (`transform.hough_line`); `minDistance` 9 and `minAngle` 10
(`hough_line_peaks`); `threshold` `ceil(min(width, height) / 6)` votes, `maxLines` 256;
`minLineLength` 30 and `maxLineGap` 3 (`HoughLinesP`). The result carries the absolute `low` and
`high` that were applied.

The threshold rule, and why. A fixed magnitude threshold depends on the contrast of the image:
scikit-image `feature.canny` takes 0.1 and 0.2 of the dtype range, OpenCV `cv2.Canny` takes
absolute Sobel magnitudes, and both leave the caller to tune them. A common automatic rule is
`0.66 median` and `1.33 median` of the gradient magnitude. This library takes the quantile form
instead. In `'quantile'` mode the two thresholds are fractions of the gradient magnitudes of the
pixels above `GRADIENT_FLOOR` (0.001, a quarter of a gray level), by nearest rank. The median rule
is not used, because the median of a photograph is dominated by flat regions and texture, and the
quantile of the pixels with a gradient keeps the share of edge pixels about the same (11,700 to
13,000 of 307,200 at 640 x 480 on the four photographs). The strong pixels are the top 5 % and
the weak pixels the top 15 % of the pixels with a gradient. `thresholdMode: 'absolute'` keeps the
scikit-image meaning (0.1 and 0.2 unless set). The vote threshold is a fraction of the shorter side
of the image: a line must have a run of about one sixth of it, which scales with the image and
not with the edge count, so a cluttered image does not raise the bar for its real lines. The
quantiles are taken on the f32 magnitudes on both tiers. The device reads the gradient back
once (`3 * width * height` floats) to resolve them.

Method, in the order the kernels run (`src/gpu/slic.shade.ts`):

1. `slicToLab`: sRGB to CIELAB (D65, L in [0, 100]), one invocation per pixel.
2. The grid: `S = sqrt(pixels / nSegments)`, then `round(width / S) × round(height / S)` cells. So the
   cluster count is near `nSegments`, not equal to it. Cluster `k` starts at the center of cell
   `k`.
3. `slicSeeds`: each center moves to the pixel of lowest Lab gradient in the 3 × 3 pixels around it.
   The first pixel in scan order wins a tie. The image border is never chosen.
4. `slicAssign`: each pixel takes the nearest of the clusters in its own cell and the eight cells
   around it. The distance is `dLab² + (compactness / S)² · dXY²`, as scikit-image. This is the
   form of gSLICr: a cluster only owns pixels of the 3 × 3 cells around its cell.
5. `slicUpdate`: one invocation per cluster gathers the pixels of those cells that carry its label
   and writes their mean Lab color and position. The sums run in a fixed order, so there are no
   atomics, the result repeats, and the CPU tier runs the same code as WebGPU. A cluster with no
   pixel keeps its center. Steps 4 and 5 run `maxIterations` times.
6. On the CPU: `enforceConnectivity`, as scikit-image. Every label is split into its 4-connected
   components. A component under `minSizeFactor` of the mean segment size joins the component
   left of or above its first pixel. Then the labels are numbered in scan order, so each label
   is one 4-connected component. This pass is cheap (13 to 33 ms at 640 × 480, below).

Every long pass runs as chunks of the device's time budget (`chunkedPass`, keys `slicLab`,
`slicSeeds`, `slicAssign` and `slicUpdate`), and the CPU tier runs each pass as one chunk.

Differences from scikit-image: the seeds move to the lowest gradient (scikit-image does not
move them), the grid is `round(size / S)` cells per side instead of a fixed `S` spacing, and the
search window is the 3 × 3 cells instead of 2S × 2S around each center. The distance and the
defaults are the same. scikit-image is not installed in this container, so no boundary recall
was compared against it. The tests below hold the recall against synthetic ground truth.

Tests (`tests/typeshade/slic.test.mjs`, `tests/typeshade/slic-webgpu.test.mjs`): on an image of
six colored rectangles with noise, the boundary recall at 2 px is at least 0.95, the count is
near `nSegments`, every label is one 4-connected component, and the device labels equal the CPU
reference labels on at least 99 % of the pixels. The remaining pixels are, by inference, f32 ties against double
ties at a segment edge. Two flat halves give exactly two labels on both.

Measured on frame 0 of the COLMAP south-building sample, resized to 640 × 480
(`node scripts/benchmark-slic.mjs`), 2026-10-20, commit `627653c` plus the script, a shared
4-core Xeon container with no physical GPU (WebGPU is SwiftShader in headless Chromium).
Median of 3 runs after one warm run, milliseconds per call:

| Stage                                                | CPU reference | CPU tier of the device | WebGPU (SwiftShader) |
| ---------------------------------------------------- | ------------- | ---------------------- | -------------------- |
| Setup (upload, Lab, seeds, one assignment, readback) | 56            | 237                    | 39                   |
| One iteration (assign + update)                      | 25            | 120                    | 39                   |
| Connectivity (CPU, after the readback)               | 19            | 33                     | 13                   |
| Total, 10 iterations and connectivity                | 326           | 1469                   | 446                  |

The stages are differences of whole calls: the call with no iteration, the slope of ten
iterations, and the call with connectivity against the call without. The call gave 80 segments
for `nSegments` 100 (108 grid cells before connectivity). The WebGPU labels equal the CPU
reference labels on 99.996 % of the pixels. On this textured photograph about 14,000 small
fragments of the 108 labels exist before the merge, so the merge removes 28 labels. Open: the
count is lower than `nSegments`. A higher `compactness` or a lower `minSizeFactor` keeps more.

```ts
import { ShadeDevice, slicGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const { labels, count } = await slicGpu(device, rgba, width, height, { nSegments: 200 });
} finally {
  await device.dispose();
}
```

The proposed use for the dense MVS is in `docs/dense-mvs.md`, section 3.14. It is not built.

## Straight lines: Canny and Hough

`cannyGpu(device, rgba, width, height, options?)` finds edges. `houghLinesGpu(device, rgba, width,
height, options?)` finds straight lines and segments. `canny` and `houghLines` are their CPU
references, in double precision. The input is 8-bit RGBA. Both Hough functions return the edge map
(`edges`, `count`), the accumulator, `lines` (`rho` in pixels, `theta` in radians, `votes`) and
`segments` (`x1, y1, x2, y2` and the index of the `line`). A line is `x cos(theta) + y sin(theta) =
rho` with `theta` in [0, pi), so `theta` is the angle of the line's normal (the OpenCV convention).

Defaults: `sigma` 1, `lowThreshold` 0.1, `highThreshold` 0.2 (scikit-image `feature.canny`);
`thetaBins` 180 and rho step 1 px (`transform.hough_line`); `minDistance` 9 and `minAngle` 10
(`hough_line_peaks`); `threshold` `ceil(min(width, height) / 4)` votes, `maxLines` 256;
`minLineLength` 30 and `maxLineGap` 3 (`HoughLinesP`).

Method, in the order the kernels run (`src/gpu/hough.shade.ts`, host code in `src/gpu/hough.ts`):

1. `houghToGray`, `houghBlur` (twice), `houghGradient`: gray with the `rgb2gray` weights, a separable
   Gaussian of radius `4 sigma` that renormalizes at the border (scikit-image's masked smoothing),
   and the Sobel gradient with scikit-image's normalization.
2. `houghSuppress`: non-maximum suppression on the gradient direction quantized to four. A pixel
   stays when it is above its neighbour behind it and not below the one ahead, so a two-pixel
   plateau keeps one pixel. Ties within 1e-5 break the same way in f32 and f64. The border is never
   an edge. The state is strong (above `highThreshold`), weak (above `lowThreshold`) or none.
3. Hysteresis, in one of two modes (`hysteresis` option). `'cpu'` (default) reads the state back
   and flood-fills from the strong pixels. `'gpu'` runs `houghGrow` in rounds of 8 passes until a
   round changes nothing. Measured below: `'gpu'` is slower on both tiers, because a chain of
   edge pixels grows one pixel per pass (40 to 96 passes here), so `'cpu'` is the default.
4. `houghCollect` appends the edge pixels to a list. `houghVote` runs one invocation per (edge
   pixel, theta) pair with an `atomicAdd` into the accumulator, chunked by `chunkedPass`
   (`houghVote`, capped at 65535 workgroups per dispatch). The cos and sin tables are integers
   scaled by 65536, and the rho is `(x cos + y sin + 32768) >> 16`. The vote is therefore
   integer arithmetic, and the device accumulator equals the CPU accumulator of the same edge map
   exactly.
5. `houghPeaks`: a cell is a peak when it has at least `threshold` votes and no cell within
   `minDistance` rho bins and `minAngle` angle bins has more (the lower cell wins a tie). Theta
   wraps at pi with the sign of rho flipped. This is a window maximum, not the greedy suppression
   of `hough_line_peaks`. The host sorts the peaks by votes.
6. `houghSegment`: one invocation per accepted line walks the line along its major axis. A step is
   supported when an edge pixel lies within one pixel across the line. A run ends after more than
   `maxLineGap` unsupported steps, and it is kept when its end points are `minLineLength` apart. A
   line keeps at most 16 segments. This is deterministic, not the randomized pixel picking of
   OpenCV `HoughLinesP`: a segment is a part of an accepted Hough line, so a segment on a line
   below `threshold` is never found.

Tests (`tests/typeshade/hough.test.mjs`, `tests/typeshade/hough-webgpu.test.mjs`). The synthetic
tests below use `thresholdMode: 'absolute'` and 64 votes (`ABSOLUTE` in `hough-cases.mjs`), the
settings they were written with, except the first one:

- A textured, noisy photo-like image with four strokes gives, with the defaults, 2,331 edge
  pixels and exactly the eight border lines on both tiers. The absolute 0.1 and 0.2 give no line.

- Four 12 px strokes at normal angles 0, 30, 90 and 135 degrees give eight border lines. Each is
  recovered within 1 degree and 1 px.
- Uniform noise gives 1458 edge pixels and no line above the default threshold. The same
  call finds a line when the threshold is lowered to the noise maximum, so the test can fail.
- A vertical step edge gives one edge pixel per row, at column 31 or 32, on both tiers. A flat
  image gives none.
- A weak edge is kept when it joins a strong one (both hysteresis modes) and dropped alone.
- The device equals the CPU reference: the edges on at least 99.9 % of the pixels (equal on the
  test images), the accumulator exactly equal to the accumulator of the device's own edge map
  (and to the CPU accumulator when the edge maps are equal), and the same lines and segments.
- A bar with a 30 px gap gives two segments per border, and `maxLineGap` 40 joins them.

scikit-image is not installed in this container, so nothing was compared with it. OpenCV 5.0.0
(Python) is installed, and one informal comparison ran outside the test suite on the same
256 px images, 2026-10-20. `cv2.GaussianBlur` (sigma 1) and `cv2.Canny` (L2 gradient, thresholds
scaled to match 0.1 and 0.2) give 2138 edge pixels on the strokes against 2169 here, and every
pixel of either lies within 1 px of the other. On the bar image both give 400. On the noise
image 96 % of the pixels agree within 1 px. `cv2.HoughLines` (threshold 64) on our edge map finds
the same eight lines (its rho differs by up to 1 px) plus neighbours, as it has no suppression.

Measured on frame 0 of the COLMAP south-building sample, resized to 640 x 480
(`node scripts/benchmark-hough.mjs`), 2026-10-20, commit `535bdaf` plus the script, a shared
4-core container with no physical GPU (WebGPU is SwiftShader in headless Chromium). Best of 7 runs
after one warm run, milliseconds per call. The shared machine is noisy, so the stage split (a
difference of whole calls) is rough and only the totals are reliable.

With the absolute `lowThreshold` 0.04 and `highThreshold` 0.08 (the settings of this run, not the defaults any more): 22,071 edge pixels, 13 lines, 23 segments.

| Path                                     | Canny | Vote, peaks, segments | Total | Hysteresis passes |
| ---------------------------------------- | ----- | --------------------- | ----- | ----------------- |
| CPU reference                            | 53    | 14                    | 67    | 0                 |
| CPU tier of the device, `hysteresis` cpu | 297   | 489                   | 785   | 0                 |
| CPU tier of the device, `hysteresis` gpu | 2335  | 667                   | 3002  | 40                |
| WebGPU (SwiftShader), `hysteresis` cpu   | 88    | 174                   | 262   | 0                 |
| WebGPU (SwiftShader), `hysteresis` gpu   | 1376  | 19                    | 1395  | 56                |

With the absolute 0.1 and 0.2 (the earlier defaults): 3,989 edge pixels, no line (the highest cell is below 120
votes), so no segment. The totals are 55 ms (CPU reference), 478 ms and 5564 ms (CPU tier, `cpu`
and `gpu`), 161 ms and 1528 ms (WebGPU, `cpu` and `gpu`). On the device, WebGPU labels equal the
CPU reference: the edge maps agree on 100 % of the pixels, and the lines and the accumulator are
equal. Findings (inferences from this run): the 4 million atomic votes cost about 170 ms on
SwiftShader, which a physical GPU would not show; the GPU hysteresis loses because each pass is a
full-image dispatch; the CPU reference is fastest here because the device has no physical GPU.
The timings were not repeated for the quantile defaults. They add one readback of the gradient
and a sort of the nonzero magnitudes on the host, and they raise the edge count from 3,989 to
11,782, so the vote cost is about three times larger than in the 0.1 and 0.2 run.

### Choosing the defaults (2026-10-20)

The earlier open item: with 0.1 and 0.2 the south-building frame gave 3,989 edge pixels and no
line. The decision was taken by measurement on the CPU reference (`houghLines`) with
`node scripts/measure-hough-defaults.mjs`, commit `4e274e0` plus the change, on the first frame
of each sample resized to 640 px on the long side (south-building 640 x 480, gerrard-hall,
person-hall and graham-hall 640 x 427, pyramid 640 x 360).
The columns: edge pixels, lines, segments, the share of segments longer than 60 px, the lines
that have another line within 2 degrees ("parallel": a wall has many parallel lines, so this is
not an error by itself) and the lines that have another line within 2 degrees and 18 px of rho
("duplicate": the same edge found twice, which is the redundancy to avoid). There was no
ground truth and no visual check: the criteria were the line count (tens on a building), the
duplicate count and the share of long segments.

| sample         | settings                                           | edges | lines | segments | >60 px | parallel | duplicate |
| -------------- | -------------------------------------------------- | ----: | ----: | -------: | -----: | -------: | --------: |
| south-building | absolute 0.1/0.2, min/4 (votes 120)                |  3989 |     0 |        0 |    0 % |        0 |         0 |
| south-building | absolute 0.04/0.08, min/4 (votes 120)              | 22071 |    13 |       23 |   35 % |        4 |         2 |
| south-building | quantile 0.85/0.95, min/6 (new default) (votes 80) | 11782 |    16 |       21 |   33 % |        4 |         0 |
| south-building | quantile 0.9/0.95, min/6 (votes 80)                |  9689 |     5 |        6 |   67 % |        0 |         0 |
| south-building | quantile 0.85/0.93, min/8 (votes 60)               | 13056 |    72 |       89 |   22 % |       50 |        15 |
| south-building | quantile 0.8/0.9, min/6 (votes 80)                 | 16977 |    52 |       76 |   21 % |       40 |         6 |
| gerrard-hall   | absolute 0.1/0.2, min/4 (votes 107)                |  6458 |     1 |        1 |  100 % |        0 |         0 |
| gerrard-hall   | absolute 0.04/0.08, min/4 (votes 107)              | 19893 |    15 |       44 |   27 % |        2 |         0 |
| gerrard-hall   | quantile 0.85/0.95, min/6 (new default) (votes 72) | 12869 |    27 |       46 |   24 % |       13 |         4 |
| gerrard-hall   | quantile 0.9/0.95, min/6 (votes 72)                | 10898 |    17 |       27 |   30 % |        8 |         0 |
| gerrard-hall   | quantile 0.85/0.93, min/8 (votes 54)               | 14414 |    82 |      116 |   20 % |       65 |        18 |
| gerrard-hall   | quantile 0.8/0.9, min/6 (votes 72)                 | 17894 |    66 |      125 |   17 % |       47 |        26 |
| person-hall    | absolute 0.1/0.2, min/4 (votes 107)                |  5874 |     3 |        5 |   80 % |        2 |         0 |
| person-hall    | absolute 0.04/0.08, min/4 (votes 107)              | 23587 |    23 |       51 |   27 % |       16 |         4 |
| person-hall    | quantile 0.85/0.95, min/6 (new default) (votes 72) | 11678 |    25 |       34 |   41 % |       16 |         5 |
| person-hall    | quantile 0.9/0.95, min/6 (votes 72)                |  9881 |    20 |       25 |   44 % |       11 |         3 |
| person-hall    | quantile 0.85/0.93, min/8 (votes 54)               | 13340 |    82 |       91 |   29 % |       65 |        22 |
| person-hall    | quantile 0.8/0.9, min/6 (votes 72)                 | 18561 |    66 |      101 |   25 % |       45 |        14 |
| graham-hall    | absolute 0.1/0.2, min/4 (votes 107)                |  6231 |     7 |       10 |   80 % |        4 |         0 |
| graham-hall    | absolute 0.04/0.08, min/4 (votes 107)              | 24313 |    41 |       62 |   42 % |       37 |        13 |
| graham-hall    | quantile 0.85/0.95, min/6 (new default) (votes 72) | 12952 |    35 |       38 |   50 % |       26 |         9 |
| graham-hall    | quantile 0.9/0.95, min/6 (votes 72)                | 10584 |    22 |       24 |   71 % |       19 |         6 |
| graham-hall    | quantile 0.85/0.93, min/8 (votes 54)               | 14469 |    98 |       69 |   36 % |       87 |        32 |
| graham-hall    | quantile 0.8/0.9, min/6 (votes 72)                 | 18164 |    74 |       81 |   31 % |       63 |        24 |
| pyramid        | absolute 0.1/0.2, min/4 (votes 90)                 |  1188 |     1 |        1 |  100 % |        0 |         0 |
| pyramid        | absolute 0.04/0.08, min/4 (votes 90)               | 14029 |    23 |       31 |   52 % |        8 |         5 |
| pyramid        | quantile 0.85/0.95, min/6 (new default) (votes 60) | 12168 |    52 |       51 |   35 % |       34 |         7 |
| pyramid        | quantile 0.9/0.95, min/6 (votes 60)                | 10356 |    36 |       26 |   54 % |       17 |         4 |
| pyramid        | quantile 0.85/0.93, min/8 (votes 45)               | 13590 |    91 |       68 |   25 % |       74 |        20 |
| pyramid        | quantile 0.8/0.9, min/6 (votes 60)                 | 17102 |    81 |       84 |   25 % |       66 |        23 |

Findings, from the table:

- The absolute 0.1 and 0.2 find 0 to 7 lines on the four buildings and 1 on the pyramid. The
  edge count differs by a factor of 5 between 0.1/0.2 and 0.04/0.08, so no single absolute pair
  suits these photographs.
- The quantile rules give 9,700 to 17,900 edge pixels on every image (11,700 to 13,000 for the default), because the share of
  edge pixels is set by the rule.
- At the same quantiles, the vote threshold sets the line count: for 0.85/0.93, min/6 gives
  25 to 60 lines and min/8 gives 72 to 98 with 15 to 32 duplicates. Lowering the vote threshold
  adds lines from texture and from the sides of one thick edge.
- 0.85/0.95 with min/6 gives 16 to 52 lines (tens), 0 to 9 duplicates, and 24 to 50 % of the
  segments longer than 60 px. 0.9/0.95 gives fewer duplicates but only 5 lines on south-building.
  0.8/0.9 doubles the duplicates (up to 26).

Decision: `thresholdMode` `'quantile'`, `lowThreshold` 0.85, `highThreshold` 0.95, `threshold`
`ceil(min(width, height) / 6)`. Limits: four buildings and one video frame, no ground truth. On an
image with no edges at all (a flat or a very smooth image), the rule still takes the top 5 %
of the weak gradients, and the vote threshold then decides whether a line appears. A caller
that needs a fixed contrast uses `thresholdMode: 'absolute'`.

```ts
import { ShadeDevice, houghLinesGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const { lines, segments } = await houghLinesGpu(device, rgba, width, height);
} finally {
  await device.dispose();
}
```

The vanishing-point hint that could use these lines is described in
`docs/research/vanishing-point-focal.md`. It is not built.

## Opt-in resident float descriptors

`matchFloatDescriptorsGpu` accepts external row-major f32 descriptors, such as
learned descriptors. `reconstructIncrementalFloatGpu` matches an explicit list
of image pairs and passes their correspondences through the existing geometric
verification, track construction, PnP, triangulation, filtering and bundle
adjustment. This adapter does not run a feature extractor. ALIKED extraction
remains experimental; the browser worker's FAST/BRIEF defaults are unchanged.

Each image supplies the usual `id`, `camera` and packed `Float64Array`
keypoints plus a descriptor record:

```ts
import {
  reconstructIncrementalFloatGpu,
  type GpuFloatImageFeatures,
  type ShadeDevice,
} from 'typecv/gpu';
import type { Resident } from 'typeshade/runtime';

async function reconstructLearnedPair(
  device: ShadeDevice,
  first: Omit<GpuFloatImageFeatures, 'descriptors'>,
  second: Omit<GpuFloatImageFeatures, 'descriptors'>,
  firstBuffer: Resident<Float32Array>,
  secondBuffer: Resident<Float32Array>,
  dimensions: number,
) {
  const images = [first, second].map((image, index) => {
    const count = image.keypoints.length / 2;
    return {
      ...image,
      descriptors: {
        descriptors: index === 0 ? firstBuffer : secondBuffer,
        count,
        dimensions,
        elements: count * dimensions,
      },
    };
  });
  return reconstructIncrementalFloatGpu(device, images, [{ first: first.id, second: second.id }], {
    matching: { ratio: 0.8, maxDistance: 2, mutual: true },
  });
}
```

Coordinates must use the camera's pixel coordinate system and match descriptor
row order. Buffers must belong to the same TypeShade runtime/device as the
`ShadeDevice`. Source and target require distinct resident handles, including
when their data are identical; self-aliasing is unsupported. The caller keeps
the descriptors alive and unchanged until the returned promise settles, then
retains responsibility for releasing them. The adapter snapshots host camera,
keypoint, graph and option data before awaiting work. It does not inspect
opaque storage to verify the caller's declared capacity or runtime identity.

Descriptors stay resident. Host-visible outputs include keypoints supplied by
the caller, compact matched indices and L2 distances; mapper graph/control and
result indexing also remain on the host. To match without reconstruction, call
`matchFloatDescriptorsGpu(device, images[0].descriptors, images[1].descriptors,
options)` instead. The mapper returns its usual result plus `matching.elapsedMs`
and per-pair match counts, plan reports and transfer accounting. Matching time
excludes extraction and subsequent geometry.

The matcher exhaustively computes Euclidean L2 nearest and second-nearest
neighbors in both directions using f32 arithmetic. The ratio test is strict;
the distance bound is inclusive. `mutual` defaults to `true` and requires both
directional ratio tests and mutual nearest neighbors. Equal-distance ties
choose the lower target index; duplicate nearest descriptors fail the ratio
test. Floating-point rounding can affect near ties and threshold boundaries
relative to a double-precision reference. The default ratio is `0.8`; the
default maximum distance is `1e19`, so choose a bound suited to the descriptor.

For N and M descriptor rows of dimension D, matching uses O(N + M) auxiliary
output/scratch memory and O(N × M × D) arithmetic. The descriptor inputs require
O((N + M) × D) storage; no N × M distance matrix is allocated. Unlike the BRIEF
matcher above, this path has no fixed four-million-comparison cutoff, but
working-memory, storage-binding and dispatch limits still apply. A physical
RTX 2080 test passed 2,001 × 2,001 descriptors; this is not a native high
resolution video or unlimited-scene throughput claim. Explicit sparse image
pairs bound graph growth, while retrieval and learned matchers remain future
work. `ShadeDevice.open({ backend: 'cpu' })` selects an explicit scalar testing
tier; WebGPU remains the default and device failures do not trigger CPU retries.

## ORB

`extractOrbGpuResident` is an opt-in alternative to the SIFT front end. It follows ORB (Rublee,
Rabaud, Konolige and Bradski, ICCV 2011) as OpenCV's `ORB_create` and scikit-image's
`feature.ORB` implement it. It gives binary 256-bit descriptors, so the matching is Hamming
matching, and the result is a `GpuResidentFeatures`: the keypoints and the matches reach the
mapper in the same form as the corner front end's. The default front end stays SIFT. The lab
runs ORB with `?features=orb` and records `options.features: "orb"` (`docs/lab-run-records.md`).

### Method

The defaults are OpenCV's, except the feature count.

1. The RGBA frame becomes gray in integers (weights 4899, 9617 and 1868 at 14 bits).
2. The pyramid has 8 levels with scale factor 1.2. Each level is resized from the one before it
   with OpenCV's `INTER_LINEAR_EXACT` (8-bit weights). A level under 63 pixels on a side holds no
   keypoint.
3. FAST-9 scores each pixel with threshold 20, as OpenCV's score: the largest threshold at which
   the pixel is still a corner. A corner needs a score above all eight neighbours (3 × 3
   suppression) and a distance of 31 pixels or more to the border of its level.
4. Each level keeps twice its share of the features by FAST score, then its share by Harris score
   (7 × 7 block of Sobel derivatives, k = 0.04, OpenCV's scaling). The share of a level is
   geometric, as in OpenCV, so the finest level gets the most.
5. The orientation is the angle of the intensity centroid of the disc of radius 15, with OpenCV's
   `fastAtan2` polynomial.
6. The descriptor rotates the learned pattern of 256 point pairs by that angle, samples the image
   smoothed by a 7 × 7 Gaussian of sigma 2, and sets a bit where the first point is darker than the
   second.
7. `matchResidentFeaturesGpu` matches two results: Hamming distance, a strict ratio test of the
   nearest and second-nearest in both directions, and the mutual check.

The pattern is OpenCV's `bit_pattern_31_` (`modules/features2d/src/orb.cpp`), the pattern learned
by Rublee et al. OpenCV is licensed Apache-2.0 from version 4.5 and under the 3-clause BSD licence
before. The table in `src/gpu/orb-pattern.ts` was read from OpenCV 5.0.0 and is checked by the
descriptors that OpenCV computes (below).

```ts
import { ShadeDevice, extractOrbGpuResident, matchResidentFeaturesGpu } from 'typecv/gpu';

const device = await ShadeDevice.open({ backend: 'webgpu' });
try {
  const first = await extractOrbGpuResident(device, firstRgba, width, height);
  const second = await extractOrbGpuResident(device, secondRgba, width, height);
  const keypoints = await first.readKeypoints(); // [x0, y0, x1, y1, …] in image pixels
  const matched = await matchResidentFeaturesGpu(device, first, second, {
    ratio: 0.8,
    maxDistance: 256,
  });
  await first.dispose();
  await second.dispose();
} finally {
  await device.dispose();
}
```

Options: `maxFeatures` (default 1000, at most 8000), `levels` (8, at most 12), `scaleFactor` (1.2)
and `fastThreshold` (20). OpenCV's own default for `maxFeatures` is 500. The Hamming matcher has
no pair cap: on a WebGPU device it is tiled (`matchBriefTiles`, 64 query lanes share a tile of 64
target descriptors), and the CPU tier runs the untiled `matchBrief`, its oracle. `readDetails()` returns eight
f32 per keypoint: x, y, size (31 times the level scale), angle in radians, Harris response, level,
0, 0. `extractOrbCpu` is the CPU reference with the same output. Frames run from 64 × 64 to the
device's limits.

### Checks

- `extractOrbCpu` against OpenCV 5.0.0 (`cv2.ORB_create`, Python wheel 5.0.0.93). On a 640 × 427
  photograph with 1000 features, 996 keypoints coincide and all 996 descriptors are equal bit for
  bit, with the same count on every level. On the 320 × 240 synthetic frame in
  `tests/fixtures/orb-opencv.json` (committed with OpenCV's output), 97 % or more of the keypoints
  coincide and their descriptors are equal (`tests/orb-reference.test.ts`). The other keypoints
  differ by a tie at a ranking boundary.
- The Hamming matcher (`tests/typeshade/hamming-match.test.mjs` on the CPU tier,
  `tests/typeshade/hamming-match-webgpu.test.mjs` on SwiftShader): the matches and distances
  equal a plain JavaScript oracle, and the tiled kernel equals the untiled one, on sets with
  ties, duplicates and sizes that are not multiples of 64, up to 4500 × 4500 (20 million pairs).
- The kernels against `extractOrbCpu` on the CPU tier: counts, keypoints, details and descriptors
  are equal (`tests/typeshade/orb.test.mjs`).
- WebGPU on SwiftShader (`tests/typeshade/orb-webgpu.test.mjs`): the same frame gives the same
  count within 3 and 263 of 271 identical descriptors; the others differ by a few bits, because
  the GPU rounds `cos`, `sin` and `log2` in its own way.
- FAST on a square finds its four vertices only. The intensity centroid of a square rotated by
  eight angles points from each vertex to the inside within 12°. Under a 30° rotation the
  median Hamming distance of 271 true correspondences is 54 bits (unrelated pairs: about 128).
  On a projective motion the mutual matches have an inlier ratio of 0.95 at 3 px.

### Differences from OpenCV

- A keypoint is the pixel centre in the frame, `(x + 0.5) · s − 0.5` for the level ratio `s`.
  OpenCV multiplies the level pixel by the nominal scale, which shifts the keypoints of coarse
  levels by up to a pixel.
- The ranking is a histogram: 256 exact bins for the FAST score and bins of 1/64 octave for the
  Harris score. A tie keeps the earlier pixel in raster order. OpenCV's `nth_element` keeps
  an arbitrary one.
- A level lists at most 16,384 corners. Corners beyond that are cut in raster order and counted
  in `requested`.
- There is no mask and no padded pyramid, because no sample reaches within 31 pixels of the
  border.
- `maxFeatures` defaults to 1000 and is at most 8000.

### Measurements

Lab runs of `scripts/measure-orb.mjs` on 2026-10-20 at commit `f3c5ed7` (the tiled Hamming
matcher), headless Chromium 141 on SwiftShader, no physical GPU, 4 shared cores with a load
average of 10 (other jobs ran at the same time), so times vary by a factor of two between runs.
Read the times as orders of magnitude and the counts as exact. All runs use `octave=0` and
`guided=0` (the lab's test configuration), analysis frames of 640 px, the Hamming bound off
(256) and one run per cell. "Matches" are the matches of the pairs the mapper verified, mean.
Features and matching times are per frame and per pair. Pyramid benchmark: 8 frames, 22 pairs.
South-building: 12 frames, 38 pairs.

Feature count, ratio test 0.8:

| Sample         | `maxFeatures` | Features per frame | Matches per pair | Registered views | Landmarks | Error   | Features, ms | Matching, ms |
| -------------- | ------------- | ------------------ | ---------------- | ---------------- | --------- | ------- | ------------ | ------------ |
| Pyramid        | 1000          | 958                | 226              | 8 of 8           | 1508      | 0.45 px | 1760         | 64           |
| Pyramid        | 2000          | 1777               | 406              | 8 of 8           | 2721      | 0.45 px | 2009         | 159          |
| Pyramid        | 4000          | 3082               | 678              | 8 of 8           | 4568      | 0.43 px | 1841         | 406          |
| Pyramid        | 8000          | 4804               | 1012             | 8 of 8           | 7192      | 0.41 px | 2226         | 924          |
| South-building | 1000          | 1000               | 29               | 5 of 12          | 285       | 0.49 px | 1597         | 70           |
| South-building | 2000          | 2000               | 45               | 6 of 12          | 615       | 0.46 px | 2402         | 278          |
| South-building | 4000          | 3917               | 72               | 6 of 12          | 1114      | 0.44 px | 1802         | 417          |
| South-building | 8000          | 7253               | 120              | 7 of 12          | 1899      | 0.39 px | 2429         | 1697         |

Ratio test, 4000 features:

| Sample         | Ratio | Matches per pair | Registered views | Landmarks | Error   | Matching, ms |
| -------------- | ----- | ---------------- | ---------------- | --------- | ------- | ------------ |
| Pyramid        | 0.7   | 484              | 8 of 8           | 3886      | 0.53 px | 362          |
| Pyramid        | 0.8   | 678              | 8 of 8           | 4568      | 0.43 px | 406          |
| Pyramid        | 0.9   | 938              | 8 of 8           | 5015      | 0.47 px | 326          |
| Pyramid        | 0.95  | 1128             | 8 of 8           | 5144      | 0.49 px | 490          |
| South-building | 0.7   | 56               | 5 of 12          | 664       | 0.38 px | 723          |
| South-building | 0.8   | 72               | 6 of 12          | 1114      | 0.44 px | 417          |
| South-building | 0.9   | 226              | 7 of 12          | 1732      | 0.49 px | 391          |
| South-building | 0.95  | 484              | 12 of 12         | 2328      | 0.56 px | 497          |

South-building at ratio 0.95 and other feature counts: 2000 features give 6 of 12 views (1061
landmarks, 0.55 px), 8000 features give 12 of 12 (3947 landmarks, 0.52 px, 1774 ms per pair).

Hamming bound, ratio 0.95, 4000 features: bounds of 80, 96 and 256 give the same result on both
samples. A bound of 64 removes 22 to 26 matches per pair and changes nothing else (12 of 12 and
8 of 8 views, landmarks 2476 and 5167). The bound is off.

Decisions (the observations above, the choice is the project's):

- A ratio of 0.95 is the only value that registers all 12 south-building views. The views a lower
  ratio loses are the wide-baseline ones, whose true matches have a poor ratio. The cost is a
  higher reprojection error (0.56 px against 0.44 px at 0.8 on south-building, 0.49 px against
  0.43 px on the pyramid) and a low share of epipolar inliers among the matches (127 of 484 on
  south-building). The mapper's verification and bundle adjustment remove the outliers.
- 4000 features is the lab default. It registers all views at ratio 0.95 with about half the
  matching time of 8000 features, whose time grows as the square of the count (1.8 s per pair on
  SwiftShader). 8000 gives 70 % more landmarks. It stays available through `?orbFeatures=8000`.
- The library default of `extractOrbGpuResident` stays 1000, and the matcher default ratio stays
  0.82. The lab sets 4000 and 0.95 for ORB (`ORB_LAB_FEATURES`, `ORB_LAB_RATIO` in
  `website/src/workers/sfm.worker.ts`). SIFT stays the lab default front end.

The lab at its new ORB defaults (commit `f3c5ed7`, no overrides) reproduces the cells above:
south-building 12 of 12 views, 2328 landmarks, 0.56 px; pyramid 8 of 8, 5144 landmarks, 0.49 px.
The earlier measurement at 2000 features (commit `b95435e`, the one-thread-per-query matcher) gave
6 of 12 views and 615 landmarks on south-building and 8 of 8 and 2721 landmarks on the pyramid. SIFT
(same settings, earlier measurement) gave 12 of 12 views, 875 landmarks, 0.32 px on south-building
and 8 of 8, 673 landmarks, 0.54 px on the pyramid. ORB with the new defaults now registers the same
views as SIFT, with more landmarks and a higher error on south-building.

### Open items

- The error of ORB at the lab defaults is higher than SIFT's on south-building (0.56 px against
  0.32 px), and the share of epipolar inliers among its matches is low. Next action: guided
  matching for binary descriptors, or a cross-check before the mapper.
- No measurement on a physical GPU. The times above are SwiftShader's.
- Guided (epipolar) matching is implemented for float descriptors only, so `?features=orb` runs
  without it.
- Keypoints have no sub-pixel refinement. The mapper's reprojection error is a little higher
  than SIFT's.
- The lab switch has no control in the page, by design (`?features=orb` only).
