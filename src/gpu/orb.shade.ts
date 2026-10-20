'use typeshade';

// ORB on resident images (Rublee et al. 2011, after OpenCV's `ORB_create`). Every kernel is one
// invocation per element without workgroup memory, so the CPU tier runs the same code as WebGPU.
// `orb-reference.ts` is the CPU reference these kernels are held to; the arithmetic follows it
// operation by operation.
//
// The pyramid is one u32 array of gray levels (0..255), level after level. Per level the kernels
// score FAST-9 corners, suppress non-maxima, keep twice the quota by FAST score and the quota by
// Harris score (`orbPick`, a histogram ranking), and then describe the kept corners.
//
// Picked record, four f32: level x, level y, Harris response, level.
// Detail record, eight f32: image x, image y, size, angle (radians), response, level, 0, 0.

class OrbLevelControl {
  width: u32;
  height: u32;
  // Element offset of the level in the pyramid, and of its source level for the resize.
  offset: u32;
  srcWidth: u32;
  srcHeight: u32;
  srcOffset: u32;
  // Element offsets of the level's x and y resize tables in `resizeTable`.
  xTable: u32;
  yTable: u32;
  threshold: u32;
  // Capacity of the corner lists.
  capacity: u32;
  level: u32;
  // Image pixels per level pixel, and the keypoint size (31 times the nominal scale).
  scaleX: f32;
  scaleY: f32;
  size: f32;
  // Capacity of the final keypoint arrays.
  outCapacity: u32;
}
class OrbMath {
  // fastAtan2 polynomial (degrees), its epsilon, and degrees to radians.
  p1: f32;
  p3: f32;
  p5: f32;
  p7: f32;
  eps: f32;
  radians: f32;
  // Harris: k and scale^4.
  harrisK: f32;
  harrisScale: f32;
  // The Gaussian taps: centre, then distance 1 to 3.
  k0: f32;
  k1: f32;
  k2: f32;
  k3: f32;
}
class OrbPick {
  // 0: rank the FAST scores of `cornerList`; 1: rank the Harris keys of `harrisKey`.
  stage: u32;
  quota: u32;
  bins: u32;
  capacity: u32;
}
class OrbMark {
  slot: u32;
}
class OrbAdvance {
  capacity: u32;
}

declare const lv: uniform<OrbLevelControl>;
declare const orbMath: uniform<OrbMath>;
declare const pick: uniform<OrbPick>;
declare const mark: uniform<OrbMark>;
declare const advance: uniform<OrbAdvance>;

declare const rgba: storage<array<u32>>;
declare const pyr: storage<array<u32>, 'read_write'>;
declare const resizeTable: storage<array<u32>>;
declare const scores: storage<array<u32>, 'read_write'>;
declare const cornerFlags: storage<array<u32>, 'read_write'>;
declare const cornerOffsets: storage<array<u32>>;
declare const cornerList: storage<array<u32>, 'read_write'>;
declare const fastList: storage<array<u32>, 'read_write'>;
declare const harrisValue: storage<array<f32>, 'read_write'>;
declare const harrisKey: storage<array<u32>, 'read_write'>;
declare const pickFlags: storage<array<u32>, 'read_write'>;
declare const pickCount: storage<array<u32>>;
declare const pickHist: storage<array<u32>, 'read_write'>;
declare const pickOffsets: storage<array<u32>>;
declare const pickTotals: storage<array<u32>>;
declare const markBounds: storage<array<u32>, 'read_write'>;
declare const markCounter: storage<array<u32>>;
declare const advanceCounter: storage<array<u32>, 'read_write'>;
declare const advanceTotals: storage<array<u32>>;
declare const advanceRequested: storage<array<u32>>;
declare const outCount: storage<array<u32>>;
declare const picked: storage<array<f32>, 'read_write'>;
declare const bounds: storage<array<u32>>;
declare const patternTable: storage<array<f32>>;
declare const umaxTable: storage<array<u32>>;
declare const points: storage<array<f32>, 'read_write'>;
declare const details: storage<array<f32>, 'read_write'>;
declare const descriptors: storage<array<u32>, 'read_write'>;

/** OpenCV's RGB to gray: weights 4899, 9617 and 1868 at 14 bits. */
@compute([64])
export function orbGray(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.width * lv.height) return;
  const pixel: u32 = rgba[i];
  const r: u32 = pixel & 255;
  const g: u32 = (pixel >> 8) & 255;
  const b: u32 = (pixel >> 16) & 255;
  pyr[lv.offset + i] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
}

/** One level from the one before it: OpenCV's INTER_LINEAR_EXACT, 8-bit weights from the tables. */
@compute([64])
export function orbResize(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.width * lv.height) return;
  const x: u32 = i % lv.width;
  const y: u32 = i / lv.width;
  const sx: u32 = resizeTable[lv.xTable + x * 3];
  const xa: u32 = resizeTable[lv.xTable + x * 3 + 1];
  const xb: u32 = resizeTable[lv.xTable + x * 3 + 2];
  const sy: u32 = resizeTable[lv.yTable + y * 3];
  const ya: u32 = resizeTable[lv.yTable + y * 3 + 1];
  const yb: u32 = resizeTable[lv.yTable + y * 3 + 2];
  const sx1: u32 = min(sx + 1, lv.srcWidth - 1);
  const sy1: u32 = min(sy + 1, lv.srcHeight - 1);
  const row0: u32 = lv.srcOffset + sy * lv.srcWidth;
  const row1: u32 = lv.srcOffset + sy1 * lv.srcWidth;
  const h0: u32 = pyr[row0 + sx] * xa + pyr[row0 + sx1] * xb;
  const h1: u32 = pyr[row1 + sx] * xa + pyr[row1 + sx1] * xb;
  pyr[lv.offset + i] = (ya * h0 + yb * h1 + 32768) >> 16;
}

/** Offset of circle pixel j of the FAST test, as dx + 3 and dy + 3. */
function circleX(j: u32): i32 {
  if (j === 0) return 0;
  if (j === 1) return 1;
  if (j === 2) return 2;
  if (j === 3) return 3;
  if (j === 4) return 3;
  if (j === 5) return 3;
  if (j === 6) return 2;
  if (j === 7) return 1;
  if (j === 8) return 0;
  if (j === 9) return -1;
  if (j === 10) return -2;
  if (j === 11) return -3;
  if (j === 12) return -3;
  if (j === 13) return -3;
  if (j === 14) return -2;
  return -1;
}
function circleY(j: u32): i32 {
  if (j === 0) return -3;
  if (j === 1) return -3;
  if (j === 2) return -2;
  if (j === 3) return -1;
  if (j === 4) return 0;
  if (j === 5) return 1;
  if (j === 6) return 2;
  if (j === 7) return 3;
  if (j === 8) return 3;
  if (j === 9) return 3;
  if (j === 10) return 2;
  if (j === 11) return 1;
  if (j === 12) return 0;
  if (j === 13) return -1;
  if (j === 14) return -2;
  return -3;
}

/**
 * OpenCV's FAST-9 score at every pixel of the level that has a full circle: zero for no corner,
 * else the largest threshold at which the pixel is still one (the best arc's smallest difference,
 * less one).
 */
@compute([64])
export function orbFast(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.width * lv.height) return;
  const x: u32 = i % lv.width;
  const y: u32 = i / lv.width;
  scores[lv.offset + i] = 0;
  if (x < 3 || y < 3 || x + 3 >= lv.width || y + 3 >= lv.height) return;
  const centre: i32 = i32(pyr[lv.offset + i]);
  let d: array<i32, 16> = fill<i32, 16>(0);
  for (let j: u32 = 0; j < 16; j++) {
    const px: u32 = u32(i32(x) + circleX(j));
    const py: u32 = u32(i32(y) + circleY(j));
    d[j] = centre - i32(pyr[lv.offset + py * lv.width + px]);
  }
  let best: i32 = 0;
  for (let start: u32 = 0; start < 16; start++) {
    let darkest: i32 = 1048576;
    let brightest: i32 = 1048576;
    for (let k: u32 = 0; k < 9; k++) {
      const v: i32 = d[(start + k) % 16];
      darkest = min(darkest, v);
      brightest = min(brightest, -v);
    }
    best = max(best, max(darkest, brightest));
  }
  if (best > i32(lv.threshold)) scores[lv.offset + i] = u32(best - 1);
}

/** A corner: at least 31 from the border, scoring above zero and strictly above all eight neighbours. */
function isCorner(i: u32): bool {
  const x: u32 = i % lv.width;
  const y: u32 = i / lv.width;
  if (x < 31 || y < 31 || x + 31 >= lv.width || y + 31 >= lv.height) return false;
  const s: u32 = scores[lv.offset + i];
  if (s === 0) return false;
  for (let dy: i32 = -1; dy <= 1; dy++)
    for (let dx: i32 = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      const n: u32 = scores[lv.offset + u32(i32(i) + dy * i32(lv.width) + dx)];
      if (n >= s) return false;
    }
  return true;
}
@compute([64])
export function orbFlagCorners(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.width * lv.height) return;
  let flag: u32 = 0;
  if (isCorner(i)) flag = 1;
  cornerFlags[i] = flag;
}
/** After the scan: the corners of the level in raster order, as x, y and score, up to the capacity. */
@compute([64])
export function orbScatterCorners(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.width * lv.height) return;
  if (!isCorner(i)) return;
  const row: u32 = cornerOffsets[(i / 64) * 2] + cornerFlags[i] - 1;
  if (row >= lv.capacity) return;
  cornerList[row * 3] = i % lv.width;
  cornerList[row * 3 + 1] = i / lv.width;
  cornerList[row * 3 + 2] = scores[lv.offset + i];
}

function rankKey(i: u32): u32 {
  if (pick.stage === 0) return cornerList[i * 3 + 2];
  return harrisKey[i];
}
/**
 * Flag the best `quota` entries of the list by key: all entries above the boundary bin, then the
 * boundary bin's entries in index order until the quota is full. One invocation: the lists hold a
 * few thousand entries, and the ranking must not depend on the order of invocations. The entry
 * count is `pickCount[0]` (the total of the scan before), at most the capacity.
 */
@compute([1])
export function orbPickTop(): void {
  const n: u32 = min(pickCount[0], pick.capacity);
  for (let b: u32 = 0; b < pick.bins; b++) pickHist[b] = 0;
  for (let i: u32 = 0; i < pick.capacity; i++) pickFlags[i] = 0;
  if (n <= pick.quota) {
    for (let i: u32 = 0; i < n; i++) pickFlags[i] = 1;
    return;
  }
  for (let i: u32 = 0; i < n; i++) {
    const key: u32 = rankKey(i);
    pickHist[key] = pickHist[key] + 1;
  }
  let above: u32 = 0;
  let boundary: u32 = 0;
  for (let b: u32 = pick.bins; b > 0; b--) {
    if (above + pickHist[b - 1] >= pick.quota) {
      boundary = b - 1;
      break;
    }
    above += pickHist[b - 1];
  }
  let room: u32 = pick.quota - above;
  for (let i: u32 = 0; i < n; i++) {
    const key: u32 = rankKey(i);
    if (key > boundary) pickFlags[i] = 1;
    else if (key === boundary && room > 0) {
      pickFlags[i] = 1;
      room--;
    }
  }
}

/** Was entry i selected? The scan left inclusive block prefixes, so a step of one marks a selection. */
function wasPicked(i: u32): bool {
  let before: u32 = 0;
  if (i % 64 !== 0) before = pickFlags[i - 1];
  return pickFlags[i] - before === 1;
}
/** Compact the corners kept by FAST score, in list order. */
@compute([64])
export function orbScatterFast(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.capacity) return;
  if (!wasPicked(i)) return;
  const row: u32 = pickOffsets[(i / 64) * 2] + pickFlags[i] - 1;
  for (let k: u32 = 0; k < 3; k++) fastList[row * 3 + k] = cornerList[i * 3 + k];
}

/** Harris response of a 7 x 7 block of Sobel derivatives (OpenCV's `HarrisResponses`), and its rank key. */
@compute([64])
export function orbHarris(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.capacity) return;
  harrisKey[i] = 0;
  harrisValue[i] = 0;
  if (i >= min(pickTotals[0], lv.capacity)) return;
  const x: i32 = i32(fastList[i * 3]);
  const y: i32 = i32(fastList[i * 3 + 1]);
  const w: i32 = i32(lv.width);
  let a: i32 = 0;
  let b: i32 = 0;
  let c: i32 = 0;
  for (let j: i32 = -3; j <= 3; j++)
    for (let k: i32 = -3; k <= 3; k++) {
      const p: u32 = lv.offset + u32((y + j) * w + x + k);
      const left: i32 = i32(pyr[p - 1]);
      const right: i32 = i32(pyr[p + 1]);
      const upLeft: i32 = i32(pyr[p - u32(w) - 1]);
      const upRight: i32 = i32(pyr[p - u32(w) + 1]);
      const downLeft: i32 = i32(pyr[p + u32(w) - 1]);
      const downRight: i32 = i32(pyr[p + u32(w) + 1]);
      const up: i32 = i32(pyr[p - u32(w)]);
      const down: i32 = i32(pyr[p + u32(w)]);
      const ix: i32 = (right - left) * 2 + (upRight - upLeft) + (downRight - downLeft);
      const iy: i32 = (down - up) * 2 + (downLeft - upLeft) + (downRight - upRight);
      a += ix * ix;
      b += iy * iy;
      c += ix * iy;
    }
  const fa: f32 = f32(a);
  const fb: f32 = f32(b);
  const fc: f32 = f32(c);
  const sum: f32 = fa + fb;
  const response: f32 = (fa * fb - fc * fc - orbMath.harrisK * sum * sum) * orbMath.harrisScale;
  harrisValue[i] = response;
  // 1/64 octave bins from 2^-40; zero for a response of zero or less.
  if (response > 0) {
    const bin: i32 = i32(floor(log2(response) * 64)) + 2560;
    harrisKey[i] = u32(clamp(bin, 1, 4095));
  }
}

/** Write the Harris-ranked corners after the scan: level x, level y, response, level. */
@compute([64])
export function orbEmit(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= lv.capacity) return;
  if (!wasPicked(i)) return;
  const row: u32 = outCount[0] + pickOffsets[(i / 64) * 2] + pickFlags[i] - 1;
  if (row >= lv.outCapacity) return;
  picked[row * 4] = f32(fastList[i * 3]);
  picked[row * 4 + 1] = f32(fastList[i * 3 + 1]);
  picked[row * 4 + 2] = harrisValue[i];
  picked[row * 4 + 3] = f32(lv.level);
}
/** counter[0] = min(counter[0] + totals[0], capacity); counter[1] += requested[0] (corners before ranking). */
@compute([1])
export function orbAdvance(): void {
  advanceCounter[0] = min(advanceCounter[0] + advanceTotals[0], advance.capacity);
  advanceCounter[1] = advanceCounter[1] + advanceRequested[0];
}
/** bounds[slot] = counter[0]. */
@compute([1])
export function orbMark(): void {
  markBounds[mark.slot] = markCounter[0];
}

/** `cv::fastAtan2` in degrees: the minimax polynomial of OpenCV, in floats. */
function atan2Degrees(y: f32, x: f32): f32 {
  const ax: f32 = abs(x);
  const ay: f32 = abs(y);
  const swap: bool = ax < ay;
  const c: f32 = select(ay / (ax + orbMath.eps), ax / (ay + orbMath.eps), swap);
  const c2: f32 = c * c;
  const poly: f32 = (((orbMath.p7 * c2 + orbMath.p5) * c2 + orbMath.p3) * c2 + orbMath.p1) * c;
  let a: f32 = select(poly, 90 - poly, swap);
  if (x < 0) a = 180 - a;
  if (y < 0) a = 360 - a;
  return a;
}

/** Gray value at level pixel (x, y). */
function grayAt(x: i32, y: i32): f32 {
  return f32(pyr[lv.offset + u32(y) * lv.width + u32(x)]);
}
/** One 7-tap Gaussian pass in the order of OpenCV's symmetric filter. */
function taps(c: f32, l1: f32, r1: f32, l2: f32, r2: f32, l3: f32, r3: f32): f32 {
  return (
    orbMath.k0 * c + (l1 + r1) * orbMath.k1 + (l2 + r2) * orbMath.k2 + (l3 + r3) * orbMath.k3
  );
}
/**
 * The 7 x 7 Gaussian of the level at (x, y), rounded to 8 bits: a horizontal pass over seven
 * rows, then a vertical pass. The keypoints are 31 from the border, so no tap leaves the level.
 */
function blurredAt(x: i32, y: i32): f32 {
  let column: array<f32, 7> = fill<f32, 7>(0);
  for (let r: i32 = 0; r < 7; r++) {
    const yy: i32 = y + r - 3;
    column[u32(r)] = taps(
      grayAt(x, yy),
      grayAt(x - 1, yy),
      grayAt(x + 1, yy),
      grayAt(x - 2, yy),
      grayAt(x + 2, yy),
      grayAt(x - 3, yy),
      grayAt(x + 3, yy),
    );
  }
  return round(taps(column[3], column[2], column[4], column[1], column[5], column[0], column[6]));
}

/**
 * Orientation and descriptor of each picked corner of the level (`bounds[0] <= k < bounds[1]`).
 * The angle is the intensity centroid of the radius-15 disc, `fastAtan2` of (m01, m10). The
 * descriptor rotates the 256 learned point pairs by that angle, samples the smoothed level at the
 * rounded points and sets a bit where the first is darker than the second.
 */
@compute([64])
export function orbDescribe(@builtin('global_invocation_id') gid: vec3u): void {
  const k: u32 = gid.x;
  if (k >= lv.outCapacity || k < bounds[0] || k >= bounds[1]) return;
  const x: i32 = i32(picked[k * 4]);
  const y: i32 = i32(picked[k * 4 + 1]);
  let m10: i32 = 0;
  let m01: i32 = 0;
  for (let u: i32 = -15; u <= 15; u++) m10 += u * i32(grayAt(x + u, y));
  for (let v: i32 = 1; v <= 15; v++) {
    let vSum: i32 = 0;
    const reach: i32 = i32(umaxTable[u32(v)]);
    for (let u: i32 = -reach; u <= reach; u++) {
      const plus: i32 = i32(grayAt(x + u, y + v));
      const minus: i32 = i32(grayAt(x + u, y - v));
      vSum += plus - minus;
      m10 += u * (plus + minus);
    }
    m01 += v * vSum;
  }
  const degrees: f32 = atan2Degrees(f32(m01), f32(m10));
  const angle: f32 = degrees * orbMath.radians;
  const ca: f32 = cos(angle);
  const sa: f32 = sin(angle);
  for (let word: u32 = 0; word < 8; word++) {
    let bits: u32 = 0;
    for (let j: u32 = 0; j < 32; j++) {
      const pair: u32 = (word * 32 + j) * 4;
      const ax: f32 = patternTable[pair];
      const ay: f32 = patternTable[pair + 1];
      const bx: f32 = patternTable[pair + 2];
      const by: f32 = patternTable[pair + 3];
      const first: f32 = blurredAt(
        x + i32(floor(ax * ca - ay * sa + 0.5)),
        y + i32(floor(ax * sa + ay * ca + 0.5)),
      );
      const second: f32 = blurredAt(
        x + i32(floor(bx * ca - by * sa + 0.5)),
        y + i32(floor(bx * sa + by * ca + 0.5)),
      );
      if (first < second) bits |= u32(1) << j;
    }
    descriptors[k * 8 + word] = bits;
  }
  const ix: f32 = (f32(x) + 0.5) * lv.scaleX - 0.5;
  const iy: f32 = (f32(y) + 0.5) * lv.scaleY - 0.5;
  points[k * 4] = ix;
  points[k * 4 + 1] = iy;
  points[k * 4 + 2] = picked[k * 4 + 2];
  points[k * 4 + 3] = 1;
  details[k * 8] = ix;
  details[k * 8 + 1] = iy;
  details[k * 8 + 2] = lv.size;
  details[k * 8 + 3] = angle;
  details[k * 8 + 4] = picked[k * 4 + 2];
  details[k * 8 + 5] = f32(lv.level);
  details[k * 8 + 6] = 0;
  details[k * 8 + 7] = 0;
}
