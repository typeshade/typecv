// ORB (Rublee et al. 2011) on the CPU: the reference that the TypeShade kernels in `orb.shade.ts`
// are held to, and the constants both share. The structure and the defaults follow OpenCV's
// `ORB_create` (modules/features2d/src/orb.cpp): an image pyramid with scale factor 1.2 and eight
// levels, FAST-9 corners with a threshold of 20 and a 3 x 3 non-maximum suppression, a Harris
// score on a 7 x 7 block (k = 0.04) to keep the best corners of each level, the intensity centroid
// of a circular patch of radius 15 for the orientation, and the learned rBRIEF pattern on the
// image smoothed by a 7 x 7 Gaussian of sigma 2.
//
// The resize, the FAST score and the moments are integer arithmetic, as in OpenCV, so that the
// kernels agree with this file bit for bit on the CPU tier. The Gaussian, the Harris response, the
// angle and the rotation of the pattern use floats, in the order of OpenCV's operations. This file
// is held to OpenCV 5.0.0 (`cv2.ORB_create`) in `tests/orb-reference.test.ts`. Where the output
// differs from OpenCV by design, `docs/gpu-features.md` (ORB) lists the difference.
import { ORB_PATTERN } from './orb-pattern.ts';

export { ORB_PATTERN };

/** The 31 x 31 patch of the pattern, the orientation and the border of the keypoints. */
export const ORB_PATCH = 31;
export const ORB_HALF_PATCH = 15;
/** Keypoints closer than this to the border of their level are dropped, as `edgeThreshold`. */
export const ORB_BORDER = 31;
/** Harris constant and block of OpenCV's `HarrisResponses`. */
export const ORB_HARRIS_K = 0.04;
export const ORB_HARRIS_BLOCK = 7;
/** Bins of the score keys that `orbSelectTop` ranks; see `orbHarrisKey`. */
export const ORB_FAST_BINS = 256;
export const ORB_HARRIS_BINS = 4096;

export interface OrbParameters {
  /** Features kept over all levels, split over the levels as OpenCV does. */
  readonly maxFeatures: number;
  /** Pyramid levels, 1 to 12. */
  readonly levels: number;
  /** Ratio of the sizes of two levels, above 1 and at most 2. */
  readonly scaleFactor: number;
  /** FAST threshold on intensities in 0..255: a neighbour must differ by more than this. */
  readonly fastThreshold: number;
}
export const ORB_DEFAULTS: OrbParameters = {
  maxFeatures: 1000,
  levels: 8,
  scaleFactor: 1.2,
  fastThreshold: 20,
};

export interface OrbLevel {
  readonly index: number;
  readonly width: number;
  readonly height: number;
  /** Element offset of the level in the concatenated pyramid. */
  readonly offset: number;
  /** Image pixels per level pixel along x and y: level (x, y) is image ((x + 0.5) sx - 0.5, ...). */
  readonly scaleX: number;
  readonly scaleY: number;
  /** The nominal scale of the level, `scaleFactor ** index` as a float: the keypoint size is 31 times it. */
  readonly scale: number;
  /** Corners ranked and kept on this level. */
  readonly quota: number;
}

/** Round half to even, as `cvRound`. */
export function roundEven(value: number): number {
  const floor = Math.floor(value),
    diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** The sizes of the pyramid levels as OpenCV computes them, and each level's share of the features. */
export function orbLevelLayout(
  width: number,
  height: number,
  parameters: OrbParameters = ORB_DEFAULTS,
): OrbLevel[] {
  const { levels, maxFeatures } = parameters,
    // OpenCV takes the factor as a float and widens it to a double.
    factor = Math.fround(parameters.scaleFactor),
    inverse = Math.fround(1 / factor),
    perLevel: number[] = [];
  let desired = Math.fround(
      (maxFeatures * Math.fround(1 - inverse)) /
        Math.fround(1 - Math.fround(Math.pow(inverse, levels))),
    ),
    sum = 0;
  for (let level = 0; level < levels - 1; level++) {
    perLevel.push(roundEven(desired));
    sum += perLevel[level]!;
    desired = Math.fround(desired * inverse);
  }
  perLevel.push(Math.max(maxFeatures - sum, 0));
  const layout: OrbLevel[] = [];
  let offset = 0;
  for (let level = 0; level < levels; level++) {
    const scale = Math.fround(Math.pow(factor, level)),
      w = level === 0 ? width : roundEven(Math.fround(width / scale)),
      h = level === 0 ? height : roundEven(Math.fround(height / scale));
    layout.push({
      index: level,
      width: w,
      height: h,
      offset,
      scaleX: Math.fround(width / w),
      scaleY: Math.fround(height / h),
      scale,
      quota: perLevel[level]!,
    });
    offset += w * h;
  }
  return layout;
}

/**
 * The per-pixel weights of OpenCV's bilinear `resize` with `INTER_LINEAR_EXACT` along one axis,
 * three integers for each output pixel: the first source index, and the 8-bit weights of it and
 * of the next index (they add up to 256). ORB resizes each level with this mode. The kernels
 * read the table from a buffer because OpenCV computes the positions in double precision.
 */
export function orbResizeTable(sourceSize: number, destinationSize: number): Int32Array {
  const table = new Int32Array(destinationSize * 3),
    scale = sourceSize / destinationSize;
  for (let d = 0; d < destinationSize; d++) {
    const position = (d + 0.5) * scale - 0.5;
    let s = Math.floor(position),
      fraction = position - s;
    if (s < 0) {
      fraction = 0;
      s = 0;
    }
    if (s >= sourceSize - 1) {
      fraction = 0;
      s = sourceSize - 1;
    }
    const weight = Math.floor(fraction * 256 + 0.5);
    table[d * 3] = s;
    table[d * 3 + 1] = 256 - weight;
    table[d * 3 + 2] = weight;
  }
  return table;
}

/** OpenCV's RGB to gray in integers (weights 0.299, 0.587, 0.114 at 14 bits). */
export function orbGray(rgba: ArrayLike<number>, width: number, height: number): Uint8Array {
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < gray.length; i++)
    gray[i] =
      (rgba[i * 4]! * 4899 + rgba[i * 4 + 1]! * 9617 + rgba[i * 4 + 2]! * 1868 + 8192) >> 14;
  return gray;
}

/** One level from the one before it: `cv::resize` with `INTER_LINEAR_EXACT` on 8-bit data. */
export function orbResize(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Uint8Array {
  const xs = orbResizeTable(sourceWidth, width),
    ys = orbResizeTable(sourceHeight, height),
    out = new Uint8Array(width * height);
  const horizontal = (row: number, x: number) => {
    const s = xs[x * 3]!;
    return (
      source[row * sourceWidth + s]! * xs[x * 3 + 1]! +
      source[row * sourceWidth + Math.min(s + 1, sourceWidth - 1)]! * xs[x * 3 + 2]!
    );
  };
  for (let y = 0; y < height; y++) {
    const r0 = ys[y * 3]!,
      r1 = Math.min(r0 + 1, sourceHeight - 1),
      b0 = ys[y * 3 + 1]!,
      b1 = ys[y * 3 + 2]!;
    for (let x = 0; x < width; x++)
      out[y * width + x] = (b0 * horizontal(r0, x) + b1 * horizontal(r1, x) + 32768) >> 16;
  }
  return out;
}

/** The pyramid, one array per level, each resized from the level before it. */
export function orbPyramid(gray: Uint8Array, layout: readonly OrbLevel[]): Uint8Array[] {
  const levels: Uint8Array[] = [gray];
  for (let i = 1; i < layout.length; i++) {
    const previous = layout[i - 1]!,
      level = layout[i]!;
    levels.push(
      orbResize(levels[i - 1]!, previous.width, previous.height, level.width, level.height),
    );
  }
  return levels;
}

/**
 * The 7-tap Gaussian kernel of sigma 2 as floats: `cv::getGaussianKernel(7, 2, CV_32F)`. Taps 0 to
 * 3 are the centre and the first three neighbours on each side.
 */
export const ORB_BLUR_KERNEL: readonly number[] = (() => {
  const weights = [0, 1, 2, 3].map((i) => Math.exp(-(i * i) / 8)),
    total = weights[0]! + 2 * (weights[1]! + weights[2]! + weights[3]!);
  return weights.map((w) => Math.fround(w / total));
})();

/**
 * The 7 x 7 Gaussian (sigma 2) of one level, as OpenCV's ORB applies it before it samples the
 * pattern: a separable float filter with reflected borders, the sum in the order of the symmetric
 * filter (centre, then the pairs by distance), rounded half to even to 8 bits.
 */
export function orbBlur(level: Uint8Array, width: number, height: number): Uint8Array {
  const reflect = (i: number, n: number) => {
      if (n === 1) return 0;
      for (;;) {
        if (i < 0) i = -i;
        else if (i >= n) i = 2 * (n - 1) - i;
        else return i;
      }
    },
    [k0, k1, k2, k3] = ORB_BLUR_KERNEL as [number, number, number, number],
    filter = (at: (offset: number) => number) =>
      Math.fround(
        Math.fround(
          Math.fround(
            Math.fround(Math.fround(k0 * at(0)) + Math.fround(Math.fround(at(-1) + at(1)) * k1)) +
              Math.fround(Math.fround(at(-2) + at(2)) * k2),
          ) + Math.fround(Math.fround(at(-3) + at(3)) * k3),
        ),
      ),
    mid = new Float32Array(width * height),
    out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      mid[y * width + x] = filter((k) => level[y * width + reflect(x + k, width)]!);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      out[y * width + x] = roundEven(filter((k) => mid[reflect(y + k, height) * width + x]!));
  return out;
}

/** Offsets (dx, dy) of the 16 pixels of the FAST circle of radius 3, clockwise from the top. */
export const FAST_CIRCLE: readonly (readonly [number, number])[] = [
  [0, -3],
  [1, -3],
  [2, -2],
  [3, -1],
  [3, 0],
  [3, 1],
  [2, 2],
  [1, 3],
  [0, 3],
  [-1, 3],
  [-2, 2],
  [-3, 1],
  [-3, 0],
  [-3, -1],
  [-2, -2],
  [-1, -3],
];

/**
 * OpenCV's FAST-9 score of a pixel: zero when the pixel is no corner at `threshold`, else the
 * largest threshold at which it still is one. A corner has nine contiguous circle pixels that
 * are all brighter than the centre plus `t`, or all darker than the centre minus `t`; the largest
 * such `t` is the best arc's smallest difference, less one.
 */
export function fastScore(
  level: Uint8Array,
  width: number,
  x: number,
  y: number,
  threshold: number,
): number {
  const centre = level[y * width + x]!,
    d: number[] = [];
  for (const [dx, dy] of FAST_CIRCLE) d.push(centre - level[(y + dy) * width + x + dx]!);
  let best = 0;
  for (let start = 0; start < 16; start++) {
    let darkest = 1 << 20,
      brightest = 1 << 20;
    for (let k = 0; k < 9; k++) {
      const v = d[(start + k) % 16]!;
      darkest = Math.min(darkest, v);
      brightest = Math.min(brightest, -v);
    }
    best = Math.max(best, darkest, brightest);
  }
  return best > threshold ? best - 1 : 0;
}

/**
 * FAST scores of a level, with OpenCV's non-maximum suppression: a corner stays when its score
 * is strictly above all eight neighbours (a non-corner scores zero). The result lists the
 * suppressed corners that are at least `ORB_BORDER` from the border, as `x`, `y` and score, in
 * raster order.
 */
export function orbCorners(
  level: Uint8Array,
  width: number,
  height: number,
  threshold: number,
): { x: number; y: number; score: number }[] {
  const scores = new Int32Array(width * height);
  for (let y = 3; y < height - 3; y++)
    for (let x = 3; x < width - 3; x++)
      scores[y * width + x] = fastScore(level, width, x, y, threshold);
  const corners: { x: number; y: number; score: number }[] = [];
  for (let y = ORB_BORDER; y < height - ORB_BORDER; y++)
    for (let x = ORB_BORDER; x < width - ORB_BORDER; x++) {
      const s = scores[y * width + x]!;
      if (s === 0) continue;
      let maximum = true;
      for (let dy = -1; dy <= 1 && maximum; dy++)
        for (let dx = -1; dx <= 1; dx++)
          if ((dx !== 0 || dy !== 0) && scores[(y + dy) * width + x + dx]! >= s) {
            maximum = false;
            break;
          }
      if (maximum) corners.push({ x, y, score: s });
    }
  return corners;
}

const f = Math.fround;

/** Harris response on a 7 x 7 block of Sobel derivatives, scaled as OpenCV's `HarrisResponses`. */
export function orbHarris(level: Uint8Array, width: number, x: number, y: number): number {
  const r = 3;
  let a = 0,
    b = 0,
    c = 0;
  for (let j = -r; j <= r; j++)
    for (let i = -r; i <= r; i++) {
      const p = (y + j) * width + x + i,
        ix =
          (level[p + 1]! - level[p - 1]!) * 2 +
          (level[p - width + 1]! - level[p - width - 1]!) +
          (level[p + width + 1]! - level[p + width - 1]!),
        iy =
          (level[p + width]! - level[p - width]!) * 2 +
          (level[p + width - 1]! - level[p - width - 1]!) +
          (level[p + width + 1]! - level[p - width + 1]!);
      a += ix * ix;
      b += iy * iy;
      c += ix * iy;
    }
  const scale = f(1 / (4 * ORB_HARRIS_BLOCK * 255)),
    scaleFourth = f(f(f(scale * scale) * scale) * scale),
    k = f(ORB_HARRIS_K),
    fa = f(a),
    fb = f(b),
    fc = f(c),
    sum = f(fa + fb);
  return f(f(f(f(fa * fb) - f(fc * fc)) - f(f(k * sum) * sum)) * scaleFourth);
}

/** A rank key for a Harris response: 1/64 octave bins from 2^-40, zero for a response of zero or less. */
export function orbHarrisKey(response: number): number {
  if (!(response > 0)) return 0;
  return Math.min(
    ORB_HARRIS_BINS - 1,
    Math.max(1, Math.floor(f(f(Math.log2(response)) * 64)) + 2560),
  );
}

/**
 * Which of `keys` the top `quota` are: the keys above the boundary bin, then the boundary bin's
 * keys in index order until the quota is full. This is the ranking of the kernel `orbSelect`.
 */
export function orbSelectTop(keys: ArrayLike<number>, quota: number, bins: number): Uint8Array {
  const n = keys.length,
    flags = new Uint8Array(n);
  if (n <= quota) return flags.fill(1);
  const histogram = new Uint32Array(bins);
  for (let i = 0; i < n; i++) histogram[keys[i]!]!++;
  let above = 0,
    boundary = 0;
  for (let b = bins - 1; b >= 0; b--) {
    if (above + histogram[b]! >= quota) {
      boundary = b;
      break;
    }
    above += histogram[b]!;
  }
  let room = quota - above;
  for (let i = 0; i < n; i++) {
    const key = keys[i]!;
    if (key > boundary) flags[i] = 1;
    else if (key === boundary && room > 0) {
      flags[i] = 1;
      room--;
    }
  }
  return flags;
}

/** The rows of the circle of the orientation patch: `u_max[v]` of OpenCV's `ICAngles`. */
export const ORB_UMAX: Int32Array = (() => {
  const half = ORB_HALF_PATCH,
    umax = new Int32Array(half + 2),
    vmax = Math.floor((half * Math.sqrt(2)) / 2 + 1),
    vmin = Math.ceil((half * Math.sqrt(2)) / 2);
  for (let v = 0; v <= vmax; v++) umax[v] = roundEven(Math.sqrt(half * half - v * v));
  for (let v = half, v0 = 0; v >= vmin; --v) {
    while (umax[v0] === umax[v0 + 1]) ++v0;
    umax[v] = v0;
    ++v0;
  }
  return umax;
})();

/** `cv::fastAtan2` in degrees: the minimax polynomial of OpenCV, in floats. */
export function fastAtan2(y: number, x: number): number {
  const p1 = f(f(0.9997878412794807) * f(180 / Math.PI)),
    p3 = f(f(-0.3258083974640975) * f(180 / Math.PI)),
    p5 = f(f(0.1555786518463281) * f(180 / Math.PI)),
    p7 = f(f(-0.04432655554792128) * f(180 / Math.PI)),
    ax = Math.abs(x),
    ay = Math.abs(y),
    eps = f(1e-5);
  const swap = ax < ay,
    c = f((swap ? ax : ay) / f((swap ? ay : ax) + eps)),
    c2 = f(c * c),
    poly = f(f(f(f(f(f(f(p7 * c2) + p5) * c2) + p3) * c2) + p1) * c);
  let a = swap ? f(90 - poly) : poly;
  if (x < 0) a = f(180 - a);
  if (y < 0) a = f(360 - a);
  return a;
}

/** The orientation of a keypoint in degrees, 0 to 360: the angle of the intensity centroid of a radius-15 disc. */
export function orbAngle(level: Uint8Array, width: number, x: number, y: number): number {
  const at = (dx: number, dy: number) => level[(y + dy) * width + x + dx]!;
  let m01 = 0,
    m10 = 0;
  for (let u = -ORB_HALF_PATCH; u <= ORB_HALF_PATCH; u++) m10 += u * at(u, 0);
  for (let v = 1; v <= ORB_HALF_PATCH; v++) {
    let vSum = 0;
    const d = ORB_UMAX[v]!;
    for (let u = -d; u <= d; u++) {
      const plus = at(u, v),
        minus = at(u, -v);
      vSum += plus - minus;
      m10 += u * (plus + minus);
    }
    m01 += v * vSum;
  }
  return fastAtan2(f(m01), f(m10));
}

/** The 256-bit rBRIEF descriptor of a keypoint as eight words, from the smoothed level. */
export function orbDescriptor(
  blurred: Uint8Array,
  width: number,
  x: number,
  y: number,
  angleDegrees: number,
): Uint32Array {
  const angle = f(angleDegrees * f(Math.PI / 180)),
    a = f(Math.cos(angle)),
    b = f(Math.sin(angle)),
    words = new Uint32Array(8),
    centre = y * width + x;
  const sample = (point: number) => {
    const px = ORB_PATTERN[point * 2]!,
      py = ORB_PATTERN[point * 2 + 1]!,
      rx = Math.floor(f(f(px * a) - f(py * b)) + 0.5),
      ry = Math.floor(f(f(px * b) + f(py * a)) + 0.5);
    return blurred[centre + ry * width + rx]!;
  };
  for (let bit = 0; bit < 256; bit++)
    if (sample(bit * 2) < sample(bit * 2 + 1))
      words[bit >> 5] = (words[bit >> 5]! | (1 << (bit & 31))) >>> 0;
  return words;
}

export interface OrbFeaturesCpu {
  readonly count: number;
  /** Image-pixel [x0, y0, x1, y1, …], integer pixel centres. */
  readonly keypoints: Float64Array;
  /** Eight f32 per keypoint: x, y, size, angle (radians), response, level, 0, 0. */
  readonly details: Float32Array;
  /** Eight u32 words per keypoint. */
  readonly descriptors: Uint32Array;
  /** Corners that passed the FAST test and the suppression, over all levels, before ranking. */
  readonly requested: number;
}

/** ORB features of an RGBA frame on the CPU: the reference of `extractOrbGpuResident`. */
export function extractOrbCpu(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  options: Partial<OrbParameters> = {},
): OrbFeaturesCpu {
  const parameters = { ...ORB_DEFAULTS, ...options },
    layout = orbLevelLayout(width, height, parameters),
    pyramid = orbPyramid(orbGray(rgba, width, height), layout),
    rows: {
      x: number;
      y: number;
      scaleX: number;
      scaleY: number;
      size: number;
      angle: number;
      response: number;
      level: number;
      words: Uint32Array;
    }[] = [];
  let requested = 0;
  for (const level of layout) {
    const image = pyramid[level.index]!,
      corners = orbCorners(image, level.width, level.height, parameters.fastThreshold);
    requested += corners.length;
    // Keep twice the quota by FAST score, rank those by Harris, keep the quota.
    const fastKeep = orbSelectTop(
        corners.map((c) => c.score),
        2 * level.quota,
        ORB_FAST_BINS,
      ),
      candidates = corners.filter((_, i) => fastKeep[i]),
      responses = candidates.map((c) => orbHarris(image, level.width, c.x, c.y)),
      harrisKeep = orbSelectTop(responses.map(orbHarrisKey), level.quota, ORB_HARRIS_BINS),
      blurred = orbBlur(image, level.width, level.height);
    candidates.forEach((c, i) => {
      if (!harrisKeep[i]) return;
      const degrees = orbAngle(image, level.width, c.x, c.y);
      rows.push({
        x: c.x,
        y: c.y,
        scaleX: level.scaleX,
        scaleY: level.scaleY,
        size: f(ORB_PATCH * level.scale),
        angle: degrees,
        response: responses[i]!,
        level: level.index,
        words: orbDescriptor(blurred, level.width, c.x, c.y, degrees),
      });
    });
  }
  const keypoints = new Float64Array(rows.length * 2),
    details = new Float32Array(rows.length * 8),
    descriptors = new Uint32Array(rows.length * 8);
  rows.forEach((row, i) => {
    const x = f(f(f(row.x + 0.5) * row.scaleX) - 0.5),
      y = f(f(f(row.y + 0.5) * row.scaleY) - 0.5);
    keypoints[i * 2] = x;
    keypoints[i * 2 + 1] = y;
    details.set(
      [x, y, row.size, f(row.angle * f(Math.PI / 180)), row.response, row.level, 0, 0],
      i * 8,
    );
    descriptors.set(row.words, i * 8);
  });
  return { count: rows.length, keypoints, details, descriptors, requested };
}
