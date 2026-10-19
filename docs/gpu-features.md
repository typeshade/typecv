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
4 × 4 × 8 descriptors with trilinear binning, normalization and 0.2 clipping. The
descriptors stay resident as 128-d f32 and match with `matchFloatDescriptorsGpu`.
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
