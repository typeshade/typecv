'use typeshade';

// Scale-space SIFT on resident f32 images. Every kernel is one invocation per element
// without workgroup memory, so the CPU tier runs the same code as the GPU tier.
//
// Octave layout (S = 3 intervals): Gaussian levels g0..g5 at sigma0 * k^i, k = 2^(1/S);
// difference-of-Gaussian levels d_i = g_{i+1} - g_i for i = 0..4; extrema are searched in
// d_1..d_3 against their 26 neighbours. The next octave's base is g3 sampled at even pixels.
//
// Keypoint record, eight f32: image x, image y, image sigma, orientation (radians),
// response (interpolated DoG), octave, continuous interval, unused.

class SiftResample {
  inWidth: u32;
  inHeight: u32;
  outWidth: u32;
  outHeight: u32;
  inOffset: u32;
}
class SiftLevel {
  width: u32;
  height: u32;
  sigma: f32;
  radius: u32;
  // Element offsets of the input and output images inside their buffers.
  inOffset: u32;
  outOffset: u32;
}
class SiftImage {
  width: u32;
  height: u32;
  offsetA: u32;
  offsetB: u32;
}
class SiftExtrema {
  width: u32;
  height: u32;
  interval: u32;
  octave: u32;
  // 2^octave: octave pixels to image pixels.
  scale: f32;
  sigma0: f32;
  k: f32;
  peak: f32;
  // (r + 1)^2 / r for the edge ratio r.
  edge: f32;
  capacity: u32;
}
class SiftKeys {
  width: u32;
  height: u32;
  scale: f32;
  sigma0: f32;
  k: f32;
  intervals: u32;
  capacity: u32;
  maxOrientations: u32;
  // Element stride between Gaussian levels in `gaussians`.
  stride: u32;
  // 1 for RootSIFT: the clipped descriptor is L1-normalized and square-rooted, so that the
  // Euclidean distance between two descriptors is the Hellinger distance of the originals.
  rootSift: u32;
}
class SiftAdvance {
  capacity: u32;
}
class SiftMark {
  slot: u32;
}

declare const resample: uniform<SiftResample>;
declare const level: uniform<SiftLevel>;
declare const image: uniform<SiftImage>;
declare const extrema: uniform<SiftExtrema>;
declare const keys: uniform<SiftKeys>;
declare const advance: uniform<SiftAdvance>;
declare const mark: uniform<SiftMark>;

declare const levelIn: storage<array<f32>>;
declare const baseOut: storage<array<f32>, 'read_write'>;
declare const blurIn: storage<array<f32>>;
declare const blurOut: storage<array<f32>, 'read_write'>;
declare const gaussianLevels: storage<array<f32>>;
declare const dogOut: storage<array<f32>, 'read_write'>;
declare const dogBelow: storage<array<f32>>;
declare const dogMid: storage<array<f32>>;
declare const dogAbove: storage<array<f32>>;
declare const candFlags: storage<array<u32>, 'read_write'>;
declare const candOffsets: storage<array<u32>>;
declare const keyCount: storage<array<u32>>;
declare const candidates: storage<array<f32>, 'read_write'>;
declare const candidateList: storage<array<f32>>;
declare const bounds: storage<array<u32>>;
declare const markBounds: storage<array<u32>, 'read_write'>;
declare const markCounter: storage<array<u32>>;
declare const advanceCounter: storage<array<u32>, 'read_write'>;
declare const advanceTotals: storage<array<u32>>;
declare const gaussians: storage<array<f32>>;
declare const orientFlags: storage<array<u32>, 'read_write'>;
declare const orientValues: storage<array<f32>, 'read_write'>;
declare const orientOffsets: storage<array<u32>>;
declare const orientedCount: storage<array<u32>>;
declare const oriented: storage<array<f32>, 'read_write'>;
declare const orientedList: storage<array<f32>>;
declare const descriptors: storage<array<f32>, 'read_write'>;

/** Sample pixel (x, y) of the Gaussian level `index` (0..5), clamped to the image. */
function levelAt(index: u32, x: i32, y: i32): f32 {
  const xx: u32 = u32(clamp(x, 0, i32(keys.width) - 1));
  const yy: u32 = u32(clamp(y, 0, i32(keys.height) - 1));
  return gaussians[index * keys.stride + yy * keys.width + xx];
}
function blurAt(x: i32, y: i32): f32 {
  const xx: u32 = u32(clamp(x, 0, i32(level.width) - 1));
  const yy: u32 = u32(clamp(y, 0, i32(level.height) - 1));
  return blurIn[level.inOffset + yy * level.width + xx];
}
function dogAt(which: u32, x: i32, y: i32): f32 {
  const xx: u32 = u32(clamp(x, 0, i32(extrema.width) - 1));
  const yy: u32 = u32(clamp(y, 0, i32(extrema.height) - 1));
  const at: u32 = yy * extrema.width + xx;
  if (which === 0) return dogBelow[at];
  if (which === 1) return dogMid[at];
  return dogAbove[at];
}

/** The next octave's base: even pixels of a level with twice the target sigma. */
@compute([64])
export function siftDownsample(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= resample.outWidth * resample.outHeight) return;
  const x: u32 = i % resample.outWidth;
  const y: u32 = i / resample.outWidth;
  const sx: u32 = min(x * 2, resample.inWidth - 1);
  const sy: u32 = min(y * 2, resample.inHeight - 1);
  baseOut[i] = levelIn[resample.inOffset + sy * resample.inWidth + sx];
}

/** Horizontal pass of a separable Gaussian with weights exp(-k² / 2σ²) over ±radius. */
@compute([64])
export function siftBlurH(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= level.width * level.height) return;
  const x: i32 = i32(i % level.width);
  const y: i32 = i32(i / level.width);
  const denominator: f32 = 2 * level.sigma * level.sigma;
  let sum: f32 = 0;
  let weight: f32 = 0;
  for (let k: i32 = -i32(level.radius); k <= i32(level.radius); k++) {
    const w: f32 = exp(-f32(k * k) / denominator);
    sum += w * blurAt(x + k, y);
    weight += w;
  }
  blurOut[level.outOffset + i] = sum / weight;
}
/** Vertical pass of the separable Gaussian. */
@compute([64])
export function siftBlurV(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= level.width * level.height) return;
  const x: i32 = i32(i % level.width);
  const y: i32 = i32(i / level.width);
  const denominator: f32 = 2 * level.sigma * level.sigma;
  let sum: f32 = 0;
  let weight: f32 = 0;
  for (let k: i32 = -i32(level.radius); k <= i32(level.radius); k++) {
    const w: f32 = exp(-f32(k * k) / denominator);
    sum += w * blurAt(x, y + k);
    weight += w;
  }
  blurOut[level.outOffset + i] = sum / weight;
}
/** d = level B - level A, two images of `gaussianLevels`. */
@compute([64])
export function siftDog(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= image.width * image.height) return;
  dogOut[i] = gaussianLevels[image.offsetB + i] - gaussianLevels[image.offsetA + i];
}

/**
 * Refine a 3 × 3 × 3 extremum at integer (x, y) of dogMid. Returns (dx, dy, ds, contrast)
 * with contrast = 0 when the point is rejected: off-centre by more than half a sample in any
 * direction, below the peak threshold, or on an edge by the Hessian ratio.
 */
function refineExtremum(x: i32, y: i32): vec4f {
  const v: f32 = dogAt(1, x, y);
  // Gradient and Hessian by central differences.
  const gx: f32 = 0.5 * (dogAt(1, x + 1, y) - dogAt(1, x - 1, y));
  const gy: f32 = 0.5 * (dogAt(1, x, y + 1) - dogAt(1, x, y - 1));
  const gs: f32 = 0.5 * (dogAt(2, x, y) - dogAt(0, x, y));
  const hxx: f32 = dogAt(1, x + 1, y) + dogAt(1, x - 1, y) - 2 * v;
  const hyy: f32 = dogAt(1, x, y + 1) + dogAt(1, x, y - 1) - 2 * v;
  const hss: f32 = dogAt(2, x, y) + dogAt(0, x, y) - 2 * v;
  const hxy: f32 =
    0.25 *
    (dogAt(1, x + 1, y + 1) - dogAt(1, x - 1, y + 1) - dogAt(1, x + 1, y - 1) + dogAt(1, x - 1, y - 1));
  const hxs: f32 =
    0.25 * (dogAt(2, x + 1, y) - dogAt(2, x - 1, y) - dogAt(0, x + 1, y) + dogAt(0, x - 1, y));
  const hys: f32 =
    0.25 * (dogAt(2, x, y + 1) - dogAt(2, x, y - 1) - dogAt(0, x, y + 1) + dogAt(0, x, y - 1));
  // Edge response on the spatial Hessian.
  const trace: f32 = hxx + hyy;
  const det: f32 = hxx * hyy - hxy * hxy;
  if (det <= 0 || trace * trace >= extrema.edge * det) return vec4f(0, 0, 0, 0);
  // Solve H d = -g by the adjugate.
  const d: f32 =
    hxx * (hyy * hss - hys * hys) - hxy * (hxy * hss - hys * hxs) + hxs * (hxy * hys - hyy * hxs);
  if (abs(d) < 1e-20) return vec4f(0, 0, 0, 0);
  const a00: f32 = hyy * hss - hys * hys;
  const a01: f32 = hxs * hys - hxy * hss;
  const a02: f32 = hxy * hys - hxs * hyy;
  const a11: f32 = hxx * hss - hxs * hxs;
  const a12: f32 = hxy * hxs - hxx * hys;
  const a22: f32 = hxx * hyy - hxy * hxy;
  const dx: f32 = -(a00 * gx + a01 * gy + a02 * gs) / d;
  const dy: f32 = -(a01 * gx + a11 * gy + a12 * gs) / d;
  const ds: f32 = -(a02 * gx + a12 * gy + a22 * gs) / d;
  if (abs(dx) > 0.6 || abs(dy) > 0.6 || abs(ds) > 0.6) return vec4f(0, 0, 0, 0);
  const contrast: f32 = v + 0.5 * (gx * dx + gy * dy + gs * ds);
  if (abs(contrast) < extrema.peak) return vec4f(0, 0, 0, 0);
  return vec4f(dx, dy, ds, contrast);
}
/** Is (x, y) of dogMid a strict extremum of its 26 neighbours above the pre-filter? */
function isExtremum(x: i32, y: i32): bool {
  const v: f32 = dogAt(1, x, y);
  if (abs(v) < 0.8 * extrema.peak) return false;
  let maximum: bool = v > 0;
  let minimum: bool = v < 0;
  for (let s: u32 = 0; s < 3; s++) {
    if (!(maximum || minimum)) {
      break;
    }
    for (let dy: i32 = -1; dy <= 1; dy++)
      for (let dx: i32 = -1; dx <= 1; dx++) {
        if (s === 1 && dx === 0 && dy === 0) continue;
        const n: f32 = dogAt(s, x + dx, y + dy);
        if (n >= v) maximum = false;
        if (n <= v) minimum = false;
      }
  }
  return maximum || minimum;
}
/** One flag per pixel of dogMid: an accepted, refined extremum. */
@compute([64])
export function siftFlagExtrema(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= extrema.width * extrema.height) return;
  const x: i32 = i32(i % extrema.width);
  const y: i32 = i32(i / extrema.width);
  let flag: u32 = 0;
  // A border of five pixels keeps the refinement stencil and the smallest window inside.
  if (x >= 5 && y >= 5 && x + 5 < i32(extrema.width) && y + 5 < i32(extrema.height))
    if (isExtremum(x, y)) {
      const r = refineExtremum(x, y);
      if (r.w !== 0) flag = 1;
    }
  candFlags[i] = flag;
}
/** Write the accepted extrema of this interval after the scan, in pixel order, after the octave's earlier keypoints. */
@compute([64])
export function siftScatterExtrema(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= extrema.width * extrema.height) return;
  const x: i32 = i32(i % extrema.width);
  const y: i32 = i32(i / extrema.width);
  // The scan replaced the flags by inclusive block prefixes, so selection is derived again.
  if (x < 5 || y < 5 || x + 5 >= i32(extrema.width) || y + 5 >= i32(extrema.height)) return;
  if (!isExtremum(x, y)) return;
  const r = refineExtremum(x, y);
  if (r.w === 0) return;
  const row: u32 = keyCount[0] + candOffsets[(i / 64) * 2] + candFlags[i] - 1;
  if (row >= extrema.capacity) return;
  const s: f32 = f32(extrema.interval) + r.z;
  const sigmaOctave: f32 = extrema.sigma0 * pow(extrema.k, s);
  candidates[row * 8] = (f32(x) + r.x) * extrema.scale;
  candidates[row * 8 + 1] = (f32(y) + r.y) * extrema.scale;
  candidates[row * 8 + 2] = sigmaOctave * extrema.scale;
  candidates[row * 8 + 3] = 0;
  candidates[row * 8 + 4] = r.w;
  candidates[row * 8 + 5] = f32(extrema.octave);
  candidates[row * 8 + 6] = s;
  candidates[row * 8 + 7] = 0;
}
/** counter[0] = min(counter[0] + totals[0], capacity); counter[1] += totals[0] (requested). */
@compute([1])
export function siftAdvance(): void {
  const total: u32 = advanceTotals[0];
  advanceCounter[0] = min(advanceCounter[0] + total, advance.capacity);
  advanceCounter[1] = advanceCounter[1] + total;
}
/** bounds[slot] = counter[0]. */
@compute([1])
export function siftMark(): void {
  markBounds[mark.slot] = markCounter[0];
}

/** Gradient (magnitude, angle) of a Gaussian level at (x, y). */
function gradientAt(index: u32, x: i32, y: i32): vec2f {
  const dx: f32 = levelAt(index, x + 1, y) - levelAt(index, x - 1, y);
  const dy: f32 = levelAt(index, x, y + 1) - levelAt(index, x, y - 1);
  return vec2f(sqrt(dx * dx + dy * dy), atan2(dy, dx));
}
/** Nearest Gaussian level of a continuous interval. */
function levelOf(s: f32): u32 {
  return u32(clamp(i32(round(s)), 0, i32(keys.intervals) + 2));
}
const TWO_PI: f32 = 6.283185307179586;
/**
 * Dominant orientations of each keypoint of the octave (`bounds[0] <= k < bounds[1]`): a
 * 36-bin gradient histogram over a window of 3 × 1.5σ, smoothed, with up to `maxOrientations`
 * peaks at or above 0.8 of the maximum, each refined by a parabola. Keypoints outside the
 * octave get zero flags.
 */
@compute([64])
export function siftOrient(@builtin('global_invocation_id') gid: vec3u): void {
  const k: u32 = gid.x;
  if (k >= keys.capacity) return;
  for (let j: u32 = 0; j < 2; j++) {
    orientFlags[k * 2 + j] = 0;
    orientValues[k * 2 + j] = -1;
  }
  if (k < bounds[0] || k >= bounds[1]) return;
  const x: f32 = candidateList[k * 8] / keys.scale;
  const y: f32 = candidateList[k * 8 + 1] / keys.scale;
  const sigma: f32 = candidateList[k * 8 + 2] / keys.scale;
  const index: u32 = levelOf(candidateList[k * 8 + 6]);
  const windowSigma: f32 = 1.5 * sigma;
  const radius: i32 = i32(round(3 * windowSigma));
  let hist: array<f32, 36> = fill<f32, 36>(0);
  const cx: i32 = i32(round(x));
  const cy: i32 = i32(round(y));
  for (let dy: i32 = -radius; dy <= radius; dy++)
    for (let dx: i32 = -radius; dx <= radius; dx++) {
      const px: i32 = cx + dx;
      const py: i32 = cy + dy;
      if (px < 1 || py < 1 || px + 1 >= i32(keys.width) || py + 1 >= i32(keys.height)) continue;
      const fx: f32 = f32(px) - x;
      const fy: f32 = f32(py) - y;
      const weight: f32 = exp(-(fx * fx + fy * fy) / (2 * windowSigma * windowSigma));
      const g = gradientAt(index, px, py);
      let bin: i32 = i32(floor((g.y / TWO_PI) * 36 + 36.5)) % 36;
      if (bin < 0) bin += 36;
      hist[u32(bin)] += weight * g.x;
    }
  // Smooth with [1, 1, 1] / 3 six times, as the reference implementation does.
  for (let pass: u32 = 0; pass < 6; pass++) {
    const first: f32 = hist[0];
    let previous: f32 = hist[35];
    for (let b: u32 = 0; b < 36; b++) {
      const current: f32 = hist[b];
      const next: f32 = select(hist[(b + 1) % 36], first, b === 35);
      hist[b] = (previous + current + next) / 3;
      previous = current;
    }
  }
  let maximum: f32 = 0;
  for (let b: u32 = 0; b < 36; b++) maximum = max(maximum, hist[b]);
  if (maximum <= 0) return;
  let written: u32 = 0;
  for (let b: u32 = 0; b < 36; b++) {
    if (written >= keys.maxOrientations) {
      break;
    }
    const h0: f32 = hist[b];
    const hm: f32 = hist[(b + 35) % 36];
    const hp: f32 = hist[(b + 1) % 36];
    if (h0 < 0.8 * maximum || h0 <= hm || h0 <= hp) continue;
    const denominator: f32 = hm - 2 * h0 + hp;
    const offset: f32 = select(0.5 * (hm - hp) / denominator, 0, abs(denominator) < 1e-12);
    let angle: f32 = (TWO_PI * (f32(b) + offset)) / 36;
    if (angle < 0) angle += TWO_PI;
    if (angle >= TWO_PI) angle -= TWO_PI;
    orientFlags[k * 2 + written] = 1;
    orientValues[k * 2 + written] = angle;
    written++;
  }
}
/** Write the oriented keypoints of the octave after the scan over the 2 × capacity slots. */
@compute([64])
export function siftScatterOriented(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= keys.capacity * 2) return;
  // A slot without an orientation holds -1; the scan replaced its flag by a prefix.
  const angle: f32 = orientValues[i];
  if (angle < 0) return;
  const row: u32 = orientedCount[0] + orientOffsets[(i / 64) * 2] + orientFlags[i] - 1;
  if (row >= keys.capacity) return;
  const k: u32 = i / 2;
  for (let j: u32 = 0; j < 8; j++) oriented[row * 8 + j] = candidateList[k * 8 + j];
  oriented[row * 8 + 3] = angle;
}
/**
 * The 4 × 4 × 8 descriptor of each oriented keypoint of the octave (`bounds[0] <= k <
 * bounds[1]`): gradients within 3σ per bin, rotated into the keypoint frame, weighted by a
 * Gaussian of half the window, trilinearly binned, normalized, clipped at 0.2 and renormalized;
 * with `keys.rootSift`, then L1-normalized and square-rooted (RootSIFT).
 */
@compute([64])
export function siftDescribe(@builtin('global_invocation_id') gid: vec3u): void {
  const k: u32 = gid.x;
  if (k >= keys.capacity || k < bounds[0] || k >= bounds[1]) return;
  const x: f32 = orientedList[k * 8] / keys.scale;
  const y: f32 = orientedList[k * 8 + 1] / keys.scale;
  const sigma: f32 = orientedList[k * 8 + 2] / keys.scale;
  const angle: f32 = orientedList[k * 8 + 3];
  const index: u32 = levelOf(orientedList[k * 8 + 6]);
  const cosA: f32 = cos(angle);
  const sinA: f32 = sin(angle);
  const binWidth: f32 = 3 * sigma;
  const radius: i32 = i32(round(binWidth * 2.5 * 1.4142135));
  let d: array<f32, 128> = fill<f32, 128>(0);
  const cx: i32 = i32(round(x));
  const cy: i32 = i32(round(y));
  for (let dy: i32 = -radius; dy <= radius; dy++)
    for (let dx: i32 = -radius; dx <= radius; dx++) {
      const px: i32 = cx + dx;
      const py: i32 = cy + dy;
      if (px < 1 || py < 1 || px + 1 >= i32(keys.width) || py + 1 >= i32(keys.height)) continue;
      const fx: f32 = f32(px) - x;
      const fy: f32 = f32(py) - y;
      // Rotate into the keypoint frame and scale to bin units; the window centre is at (1.5, 1.5).
      const rx: f32 = (cosA * fx + sinA * fy) / binWidth + 1.5;
      const ry: f32 = (-sinA * fx + cosA * fy) / binWidth + 1.5;
      if (rx <= -1 || ry <= -1 || rx >= 4 || ry >= 4) continue;
      const g = gradientAt(index, px, py);
      let theta: f32 = g.y - angle;
      theta -= TWO_PI * floor(theta / TWO_PI);
      const ro: f32 = (theta / TWO_PI) * 8;
      const weight: f32 = exp(-((rx - 1.5) * (rx - 1.5) + (ry - 1.5) * (ry - 1.5)) / 8) * g.x;
      const x0: i32 = i32(floor(rx));
      const y0: i32 = i32(floor(ry));
      const o0: i32 = i32(floor(ro));
      const wx1: f32 = rx - f32(x0);
      const wy1: f32 = ry - f32(y0);
      const wo1: f32 = ro - f32(o0);
      for (let ix: i32 = 0; ix < 2; ix++) {
        const bx: i32 = x0 + ix;
        if (bx < 0 || bx > 3) continue;
        const wxv: f32 = select(1 - wx1, wx1, ix === 1);
        for (let iy: i32 = 0; iy < 2; iy++) {
          const by: i32 = y0 + iy;
          if (by < 0 || by > 3) continue;
          const wyv: f32 = select(1 - wy1, wy1, iy === 1);
          for (let io: i32 = 0; io < 2; io++) {
            const bo: u32 = u32((o0 + io) % 8);
            const wov: f32 = select(1 - wo1, wo1, io === 1);
            d[u32(by * 4 + bx) * 8 + bo] += weight * wxv * wyv * wov;
          }
        }
      }
    }
  let norm: f32 = 0;
  for (let j: u32 = 0; j < 128; j++) norm += d[j] * d[j];
  norm = max(sqrt(norm), 1e-12);
  let clipped: f32 = 0;
  for (let j: u32 = 0; j < 128; j++) {
    d[j] = min(d[j] / norm, 0.2);
    clipped += d[j] * d[j];
  }
  clipped = max(sqrt(clipped), 1e-12);
  if (keys.rootSift === 1) {
    // RootSIFT (Arandjelović and Zisserman, 2012): the square root of the L1-normalized vector
    // is a unit vector under L2, so the matcher's ratio test needs no change.
    let l1: f32 = 0;
    for (let j: u32 = 0; j < 128; j++) l1 += d[j];
    l1 = max(l1, 1e-12);
    for (let j: u32 = 0; j < 128; j++) descriptors[k * 128 + j] = sqrt(d[j] / l1);
    return;
  }
  for (let j: u32 = 0; j < 128; j++) descriptors[k * 128 + j] = d[j] / clipped;
}
