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
rejects inputs above four million pairwise descriptor comparisons to bound GPU
work.

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

Options: `maxFeatures` (default 1000, at most 2000), `levels` (8, at most 12), `scaleFactor` (1.2)
and `fastThreshold` (20). OpenCV's own default for `maxFeatures` is 500. The bound of 2000 comes
from the Hamming matcher, which allows 4,000,000 descriptor pairs. `readDetails()` returns eight
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
- `maxFeatures` defaults to 1000 and is at most 2000.

### Measurements

Lab runs of `scripts/measure-orb.mjs` on 2026-10-20 at commit `b95435e` (the front end of this
change), headless Chromium 141 on SwiftShader, no physical GPU, 4 shared cores with a load average
of 4 to 6, so times vary by tens of percent. ORB and SIFT use the same settings: `octave=0`,
`guided=0` (the lab's test configuration), analysis frames of 640 px, the lab's other defaults.
"Matches" are the matches of the pairs the mapper verified, mean (minimum to maximum).

| Sample (frames)       | Front end         | Features per frame | Matches per pair | Registered views | Landmarks | Mean reprojection error | Features, ms per frame | Matching, ms per pair  |
| --------------------- | ----------------- | ------------------ | ---------------- | ---------------- | --------- | ----------------------- | ---------------------- | ---------------------- |
| Pyramid benchmark (8) | ORB (2000 cap)    | 1777               | 406 (54 to 919)  | 8 of 8           | 2721      | 0.45 px                 | 1204                   | 203                    |
| Pyramid benchmark (8) | SIFT              | 403                | 145 (55 to 229)  | 8 of 8           | 673       | 0.54 px                 | 936                    | 280                    |
| Pyramid benchmark (8) | SIFT, lab default | 1507               | 399 (122 to 633) | 8 of 8           | 2253      | 0.31 px                 | 3793                   | 4612 (plus 816 guided) |
| South-building (12)   | ORB (2000 cap)    | 2000               | 45 (8 to 319)    | 6 of 12          | 615       | 0.46 px                 | 1251                   | 232                    |
| South-building (12)   | SIFT              | 975                | 58 (9 to 245)    | 12 of 12         | 875       | 0.32 px                 | 1733                   | 1972                   |

"SIFT, lab default" is the lab without the two switches (first octave -1 and guided matching).
The pyramid sample has 22 matched pairs, and south-building 34 (ORB) and 37 (SIFT). The SIFT
matching times include the pairs of 975 to 1507 float descriptors of 128 values.

ORB finds four times the features of SIFT at the same settings on the pyramid sample and gives the
mapper more matches and landmarks. It is not more accurate: its reprojection error is higher, and
on south-building, whose photographs differ in scale and viewpoint, ORB registers 6 of the 12
views against 12 of 12 for SIFT. Its stage times are of the order of SIFT's on a software device.

### Open items

- Registration of wide-baseline photographs (south-building: 6 of 12). Next action: try a
  larger feature count with a faster matcher, and a ratio and distance bound chosen by
  measurement.
- The Hamming matcher runs one invocation per query descriptor over all targets and allows 4,000,000
  pairs, which bounds `maxFeatures` at 2000. Next action: a tiled matcher as for float
  descriptors.
- No measurement on a physical GPU. The times above are SwiftShader's.
- Guided (epipolar) matching is implemented for float descriptors only, so `?features=orb` runs
  without it.
- Keypoints have no sub-pixel refinement. The mapper's reprojection error is a little higher
  than SIFT's.
- The lab switch has no control in the page, by design (`?features=orb` only).
