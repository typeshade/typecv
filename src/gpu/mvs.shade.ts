'use typeshade';

// Dense MVS, the photometric depth stage (`docs/dense-mvs.md`, sections 3.4 to 3.7): the
// undistortion of the gray images, the random initialization and the seeds of a reference view,
// and the red-black PatchMatch propagation with its refinement. Every kernel is one invocation
// per element without workgroup memory, so the CPU tier runs the same code as the GPU tier.
//
// Layouts (section 4.1). P = W × H. Pixel index i = y × W + x. A per-view array is view-major.
// - `mvsCameras`: f32 × 24 per view: R (9, row-major, world to camera), t (3), fx, fy, cx, cy,
//   W, H, dmin, dmax, the camera center (3), 0.
// - `mvsSources`: u32 × 7 per view: the source count, then up to 6 view indices.
// - `mvsSeeds`: f32 × 3 per seed: x, y (integer pixels of the undistorted image), depth.
// - `mvsGray`: f32, N × P, the undistorted gray in [0, 1], −1 for "no data".
// - `mvsDepth`: f32, N × P, the optical-axis depth in normalized units.
// - `mvsNormal`: u32, N × P, the octahedral unit normal in the view's camera frame.
// - `mvsCost`: f32, P, the cost of the reference's current hypothesis.
//
// A lane past its chunk takes the chunk's last element, runs the same control flow and writes
// nothing (the clamp-and-mask rule of section 3.6).

class MvsControl {
  width: u32;
  height: u32;
  reference: u32;
  // The color of a propagation pass: 0 for the red pixels ((x + y) even), 1 for the black ones.
  color: u32;
  // 0 for the initialization and the seeds, 1..T for the propagation iterations.
  iteration: u32;
  seed: u32;
  bestSources: u32;
  // Window offsets are −radius + j × step for j < axisSamples, in each axis.
  windowRadius: u32;
  windowStep: u32;
  axisSamples: u32;
  // The least number of reference samples of a window that are inside the image with data.
  minSamples: u32;
  bilateral: u32;
  // 1 / (2 σ²) of the spatial and the color Gaussian of the bilateral weights.
  spatialFalloff: f32;
  colorFalloff: f32;
  // cos of the least triangulation angle at the point (1°).
  minAngleCos: f32;
  // The refinement's perturbation: a fraction of the inverse-depth range, and an angle in radians.
  perturbDepth: f32;
  perturbAngle: f32;
}
// The elements of one chunk: pixels, color indices or seeds `first..end`.
class MvsRange {
  first: u32;
  end: u32;
}
class MvsUndistortControl {
  width: u32;
  height: u32;
  view: u32;
  // 0: the view has no distortion, and the kernel copies its gray image.
  distorted: u32;
  fx: f32;
  fy: f32;
  cx: f32;
  cy: f32;
  k1: f32;
  k2: f32;
  // The squared radius past which the radial model folds (1e30 when it does not).
  radialLimit: f32;
}
class MvsWindow {
  mean: f32;
  variance: f32;
  weight: f32;
  center: f32;
  ok: u32;
}

declare const mvsControl: uniform<MvsControl>;
declare const mvsRange: uniform<MvsRange>;
declare const undistortControl: uniform<MvsUndistortControl>;
declare const undistortInput: storage<array<f32>>;
declare const undistortOutput: storage<array<f32>, 'read_write'>;
declare const mvsCameras: storage<array<f32>>;
declare const mvsSources: storage<array<u32>>;
declare const mvsSeeds: storage<array<f32>>;
declare const mvsGray: storage<array<f32>>;
declare const mvsDepth: storage<array<f32>, 'read_write'>;
declare const mvsNormal: storage<array<u32>, 'read_write'>;
declare const mvsCost: storage<array<f32>, 'read_write'>;

const NO_COST: f32 = 2.;
const CAMERA_FLOATS: u32 = 24;
const SOURCE_SLOTS: u32 = 7;
const TWO_PI: f32 = 6.283185307179586;

/** Element `j` of view `v`'s camera record. */
function cameraAt(v: u32, j: u32): f32 {
  return mvsCameras[v * CAMERA_FLOATS + j];
}
function rotationRow(v: u32, row: u32): vec3 {
  return vec3(cameraAt(v, row * 3), cameraAt(v, row * 3 + 1), cameraAt(v, row * 3 + 2));
}
function translation(v: u32): vec3 {
  return vec3(cameraAt(v, 9), cameraAt(v, 10), cameraAt(v, 11));
}
/** K_r⁻¹ (x, y, 1) of the reference: the ray of a pixel at depth 1. */
function referenceRay(x: f32, y: f32): vec3 {
  const r: u32 = mvsControl.reference;
  return vec3((x - cameraAt(r, 14)) / cameraAt(r, 12), (y - cameraAt(r, 15)) / cameraAt(r, 13), 1.);
}

/** The integer hash of `hash` in `src/gpu/features.shade.ts`. */
function hash(value: u32): u32 {
  let x: u32 = value + 0x9e3779b9;
  x = (x ^ (x >> 16)) * 0x85ebca6b;
  x = (x ^ (x >> 13)) * 0xc2b2ae35;
  return x ^ (x >> 16);
}
/** A uniform number in [0, 1) from (seed, view, pixel, iteration, draw), with no stored state. */
function random(pixel: u32, draw: u32): f32 {
  const h: u32 = hash(
    hash(hash(hash(mvsControl.seed) ^ mvsControl.reference) ^ pixel) ^
      (mvsControl.iteration * 16 + draw),
  );
  return f32(h >> 8) * 0.000000059604644775390625;
}
/** A uniform unit vector from two draws. */
function randomUnit(pixel: u32, draw: u32): vec3 {
  const z: f32 = 2. * random(pixel, draw) - 1.;
  const phi: f32 = TWO_PI * random(pixel, draw + 1);
  const s: f32 = sqrt(max(1. - z * z, 0.));
  return vec3(s * cos(phi), s * sin(phi), z);
}

/** The octahedral map of a unit vector, two snorm16 components (section 4.1). */
function encodeNormal(n: vec3): u32 {
  const s: f32 = abs(n.x) + abs(n.y) + abs(n.z);
  let px: f32 = n.x / s;
  let py: f32 = n.y / s;
  if (n.z < 0.) {
    const ox: f32 = (1. - abs(py)) * (px >= 0. ? 1. : -1.);
    const oy: f32 = (1. - abs(px)) * (py >= 0. ? 1. : -1.);
    px = ox;
    py = oy;
  }
  return pack2x16snorm(vec2(px, py));
}
function decodeNormal(word: u32): vec3 {
  const p: vec2 = unpack2x16snorm(word);
  const z: f32 = 1. - abs(p.x) - abs(p.y);
  const t: f32 = max(-z, 0.);
  const x: f32 = p.x + (p.x >= 0. ? -t : t);
  const y: f32 = p.y + (p.y >= 0. ? -t : t);
  return normalize(vec3(x, y, z));
}

/** The reference's gray at an integer pixel, −1 outside the image or without data. */
function referenceAt(x: i32, y: i32): f32 {
  if (x < 0 || y < 0 || x >= i32(mvsControl.width) || y >= i32(mvsControl.height)) return -1.;
  const P: u32 = mvsControl.width * mvsControl.height;
  return mvsGray[mvsControl.reference * P + u32(y) * mvsControl.width + u32(x)];
}
/** View `v`'s gray at (x, y), bilinear, −1 outside the image or next to a pixel without data. */
function sourceAt(v: u32, x: f32, y: f32): f32 {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  if (!(x >= 0. && y >= 0. && x <= f32(W - 1) && y <= f32(H - 1))) return -1.;
  const fx: f32 = floor(x);
  const fy: f32 = floor(y);
  const x0: u32 = u32(fx);
  const y0: u32 = u32(fy);
  const x1: u32 = min(x0 + 1, W - 1);
  const y1: u32 = min(y0 + 1, H - 1);
  const base: u32 = v * W * H;
  const g00: f32 = mvsGray[base + y0 * W + x0];
  const g10: f32 = mvsGray[base + y0 * W + x1];
  const g01: f32 = mvsGray[base + y1 * W + x0];
  const g11: f32 = mvsGray[base + y1 * W + x1];
  if (min(min(g00, g10), min(g01, g11)) < 0.) return -1.;
  const ax: f32 = x - fx;
  const ay: f32 = y - fy;
  return mix(mix(g00, g10, ax), mix(g01, g11, ax), ay);
}
/** The weight of a window sample: 1, or the bilateral weight (section 3.7). */
function sampleWeight(dx: i32, dy: i32, value: f32, center: f32): f32 {
  if (mvsControl.bilateral === 0) return 1.;
  const d2: f32 = f32(dx * dx + dy * dy);
  const c: f32 = value - center;
  return exp(-d2 * mvsControl.spatialFalloff - c * c * mvsControl.colorFalloff);
}
function windowOffset(j: u32): i32 {
  return i32(j) * i32(mvsControl.windowStep) - i32(mvsControl.windowRadius);
}

/** The weighted mean and variance of the reference window at (x, y). */
function referenceWindow(x: i32, y: i32): MvsWindow {
  const center: f32 = referenceAt(x, y);
  let weight: f32 = 0.;
  let sum: f32 = 0.;
  let count: u32 = 0;
  for (let j: u32 = 0; j < mvsControl.axisSamples; j++) {
    for (let k: u32 = 0; k < mvsControl.axisSamples; k++) {
      const dx: i32 = windowOffset(k);
      const dy: i32 = windowOffset(j);
      const a: f32 = referenceAt(x + dx, y + dy);
      if (a < 0.) continue;
      const w: f32 = sampleWeight(dx, dy, a, center);
      weight += w;
      sum += w * a;
      count++;
    }
  }
  let mean: f32 = 0.;
  if (weight > 0.) mean = sum / weight;
  let spread: f32 = 0.;
  for (let j: u32 = 0; j < mvsControl.axisSamples; j++) {
    for (let k: u32 = 0; k < mvsControl.axisSamples; k++) {
      const dx: i32 = windowOffset(k);
      const dy: i32 = windowOffset(j);
      const a: f32 = referenceAt(x + dx, y + dy);
      if (a < 0.) continue;
      const w: f32 = sampleWeight(dx, dy, a, center);
      spread += w * (a - mean) * (a - mean);
    }
  }
  let variance: f32 = 0.;
  if (weight > 0.) variance = spread / weight;
  let ok: u32 = 0;
  if (center >= 0. && count >= mvsControl.minSamples && variance >= 0.000001) ok = 1;
  return { mean: mean, variance: variance, weight: weight, center: center, ok: ok };
}

/**
 * 1 − NCC of the plane (point X on the pixel's ray, normal n) between the reference window and
 * source view `s`, through the homography H = K_s (R_rs − t_rs nᵀ / δ) K_r⁻¹ (section 3.7).
 */
function sourceCost(s: u32, x: i32, y: i32, X: vec3, n: vec3, stats: MvsWindow): f32 {
  const r: u32 = mvsControl.reference;
  const a0: vec3 = rotationRow(r, 0);
  const a1: vec3 = rotationRow(r, 1);
  const a2: vec3 = rotationRow(r, 2);
  const b0: vec3 = rotationRow(s, 0);
  const b1: vec3 = rotationRow(s, 1);
  const b2: vec3 = rotationRow(s, 2);
  // R_rs = R_s R_rᵀ, by rows; t_rs = t_s − R_rs t_r.
  const m0: vec3 = vec3(dot(b0, a0), dot(b0, a1), dot(b0, a2));
  const m1: vec3 = vec3(dot(b1, a0), dot(b1, a1), dot(b1, a2));
  const m2: vec3 = vec3(dot(b2, a0), dot(b2, a1), dot(b2, a2));
  const tr: vec3 = translation(r);
  const t: vec3 = translation(s) - vec3(dot(m0, tr), dot(m1, tr), dot(m2, tr));
  // The triangulation angle at X between the rays to the two centers.
  const C: vec3 = -(m0 * t.x + m1 * t.y + m2 * t.z);
  const toR: vec3 = -X;
  const toS: vec3 = C - X;
  const cosine: f32 = dot(toR, toS) / (length(toR) * length(toS));
  if (!(cosine <= mvsControl.minAngleCos)) return NO_COST;
  const delta: f32 = -dot(n, X);
  if (!(delta > 0.000001 * length(X))) return NO_COST;
  // M = R_rs − t_rs nᵀ / δ, then A = M K_r⁻¹ and H = K_s A, by rows.
  const q0: vec3 = m0 - n * (t.x / delta);
  const q1: vec3 = m1 - n * (t.y / delta);
  const q2: vec3 = m2 - n * (t.z / delta);
  const fxr: f32 = cameraAt(r, 12);
  const fyr: f32 = cameraAt(r, 13);
  const cxr: f32 = cameraAt(r, 14);
  const cyr: f32 = cameraAt(r, 15);
  const e0: vec3 = vec3(q0.x / fxr, q0.y / fyr, q0.z - (q0.x * cxr) / fxr - (q0.y * cyr) / fyr);
  const e1: vec3 = vec3(q1.x / fxr, q1.y / fyr, q1.z - (q1.x * cxr) / fxr - (q1.y * cyr) / fyr);
  const e2: vec3 = vec3(q2.x / fxr, q2.y / fyr, q2.z - (q2.x * cxr) / fxr - (q2.y * cyr) / fyr);
  const h0: vec3 = e0 * cameraAt(s, 12) + e2 * cameraAt(s, 14);
  const h1: vec3 = e1 * cameraAt(s, 13) + e2 * cameraAt(s, 15);
  let sumB: f32 = 0.;
  let sumBB: f32 = 0.;
  let sumAB: f32 = 0.;
  let bad: bool = false;
  for (let j: u32 = 0; j < mvsControl.axisSamples; j++) {
    for (let k: u32 = 0; k < mvsControl.axisSamples; k++) {
      const dx: i32 = windowOffset(k);
      const dy: i32 = windowOffset(j);
      const a: f32 = referenceAt(x + dx, y + dy);
      if (a < 0.) continue;
      const w: f32 = sampleWeight(dx, dy, a, stats.center);
      const p: vec3 = vec3(f32(x + dx), f32(y + dy), 1.);
      const hz: f32 = dot(e2, p);
      if (!(hz > 0.)) {
        bad = true;
        break;
      }
      const b: f32 = sourceAt(s, dot(h0, p) / hz, dot(h1, p) / hz);
      if (b < 0.) {
        bad = true;
        break;
      }
      sumB += w * b;
      sumBB += w * b * b;
      sumAB += w * (a - stats.mean) * b;
    }
    if (bad) break;
  }
  if (bad) return NO_COST;
  const meanB: f32 = sumB / stats.weight;
  const varianceB: f32 = sumBB / stats.weight - meanB * meanB;
  if (!(varianceB >= 0.000001)) return NO_COST;
  const ncc: f32 = sumAB / stats.weight / sqrt(stats.variance * varianceB);
  return clamp(1. - ncc, 0., NO_COST);
}

/** The aggregated cost: the mean of the `bestSources` least source costs (section 3.7). */
function hypothesisCost(x: u32, y: u32, depth: f32, n: vec3, stats: MvsWindow): f32 {
  if (stats.ok === 0) return NO_COST;
  const X: vec3 = referenceRay(f32(x), f32(y)) * depth;
  const base: u32 = mvsControl.reference * SOURCE_SLOTS;
  const count: u32 = min(mvsSources[base], SOURCE_SLOTS - 1);
  let costs: array<f32, 6> = fill<f32, 6>(NO_COST);
  for (let k: u32 = 0; k < count; k++)
    costs[k] = sourceCost(mvsSources[base + 1 + k], i32(x), i32(y), X, n, stats);
  const best: u32 = min(mvsControl.bestSources, count);
  let sum: f32 = 0.;
  for (let j: u32 = 0; j < best; j++) {
    let least: u32 = 0;
    for (let k: u32 = 1; k < count; k++) if (costs[k] < costs[least]) least = k;
    sum += costs[least];
    costs[least] = 3.;
  }
  if (best === 0) return NO_COST;
  return sum / f32(best);
}

/** A random normal of the hemisphere that faces the camera along `ray`. */
function randomNormal(pixel: u32, draw: u32, ray: vec3): vec3 {
  const n: vec3 = randomUnit(pixel, draw);
  if (dot(n, ray) > 0.) return -n;
  return n;
}
/** A random depth, uniform in inverse depth over [dmin, dmax]. */
function randomDepth(pixel: u32, draw: u32): f32 {
  const r: u32 = mvsControl.reference;
  const near: f32 = 1. / cameraAt(r, 18);
  const far: f32 = 1. / cameraAt(r, 19);
  return 1. / (far + random(pixel, draw) * (near - far));
}

/** Undistortion of one view's gray image into `mvsGray` (section 3.4). One lane per pixel. */
@compute([64])
export function mvsUndistort(@builtin('global_invocation_id') gid: vec3u): void {
  const W: u32 = undistortControl.width;
  const H: u32 = undistortControl.height;
  const P: u32 = W * H;
  const active: bool = gid.x < P;
  const i: u32 = min(gid.x, P - 1);
  const u: f32 = f32(i % W);
  const v: f32 = f32(i / W);
  let value: f32 = undistortInput[i];
  if (undistortControl.distorted !== 0) {
    const x: f32 = (u - undistortControl.cx) / undistortControl.fx;
    const y: f32 = (v - undistortControl.cy) / undistortControl.fy;
    const r2: f32 = x * x + y * y;
    value = -1.;
    if (r2 < undistortControl.radialLimit) {
      const d: f32 = 1. + undistortControl.k1 * r2 + undistortControl.k2 * r2 * r2;
      const sx: f32 = undistortControl.fx * x * d + undistortControl.cx;
      const sy: f32 = undistortControl.fy * y * d + undistortControl.cy;
      if (sx >= 0. && sy >= 0. && sx <= f32(W - 1) && sy <= f32(H - 1)) {
        const fx: f32 = floor(sx);
        const fy: f32 = floor(sy);
        const x0: u32 = u32(fx);
        const y0: u32 = u32(fy);
        const x1: u32 = min(x0 + 1, W - 1);
        const y1: u32 = min(y0 + 1, H - 1);
        const ax: f32 = sx - fx;
        const ay: f32 = sy - fy;
        value = mix(
          mix(undistortInput[y0 * W + x0], undistortInput[y0 * W + x1], ax),
          mix(undistortInput[y1 * W + x0], undistortInput[y1 * W + x1], ax),
          ay,
        );
      }
    }
  }
  if (active) undistortOutput[undistortControl.view * P + i] = value;
}

/**
 * A random hypothesis and its cost for each pixel `first..end` of the reference (section 3.5).
 * Draws 0 to 2 of iteration 0.
 */
@compute([64])
export function mvsInitialize(@builtin('global_invocation_id') gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const x: u32 = i % W;
  const y: u32 = i / W;
  const ray: vec3 = referenceRay(f32(x), f32(y));
  const depth: f32 = randomDepth(i, 0);
  const word: u32 = encodeNormal(randomNormal(i, 1, ray));
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  const cost: f32 = hypothesisCost(x, y, depth, decodeNormal(word), stats);
  if (active) {
    mvsDepth[r * P + i] = depth;
    mvsNormal[r * P + i] = word;
    mvsCost[i] = cost;
  }
}

/**
 * The seeds `first..end` of `mvsSeeds` over the random values: the seed's depth, a normal that
 * faces the camera along the pixel's ray, and its cost (section 3.5).
 */
@compute([64])
export function mvsSeed(@builtin('global_invocation_id') gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  let j: u32 = mvsRange.first + gid.x;
  const active: bool = j < mvsRange.end;
  if (!active) j = max(mvsRange.end, 1) - 1;
  const x: u32 = min(u32(mvsSeeds[j * 3]), W - 1);
  const y: u32 = min(u32(mvsSeeds[j * 3 + 1]), mvsControl.height - 1);
  const depth: f32 = mvsSeeds[j * 3 + 2];
  const i: u32 = y * W + x;
  const word: u32 = encodeNormal(-normalize(referenceRay(f32(x), f32(y))));
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  const cost: f32 = hypothesisCost(x, y, depth, decodeNormal(word), stats);
  if (active) {
    mvsDepth[r * P + i] = depth;
    mvsNormal[r * P + i] = word;
    mvsCost[i] = cost;
  }
}

/**
 * One color pass of a propagation iteration over the color indices `first..end` (sections 3.6
 * and 5.1). Color index k is row y = ⌊k / ⌈W/2⌉⌋ and column
 * x = 2 (k mod ⌈W/2⌉) + ((y + c) mod 2).
 * A pixel evaluates 8 propagation candidates of the other color, a random hypothesis and the
 * best one so far perturbed, and keeps the least cost. It reads the other color and writes only
 * its own slots, so the pass runs in place and any split into chunks gives the same result.
 * Draws 0 to 6 of the iteration.
 */
@compute([64])
export function mvsPropagate(@builtin('global_invocation_id') gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const P: u32 = W * H;
  const r: u32 = mvsControl.reference;
  const base: u32 = r * P;
  const half: u32 = (W + 1) / 2;
  let k: u32 = mvsRange.first + gid.x;
  let active: bool = k < mvsRange.end;
  if (!active) k = max(mvsRange.end, 1) - 1;
  const y: u32 = min(k / half, H - 1);
  let x: u32 = 2 * (k % half) + ((y + mvsControl.color) % 2);
  if (x >= W) {
    active = false;
    x = W - 1;
  }
  const i: u32 = y * W + x;
  const ray: vec3 = referenceRay(f32(x), f32(y));
  const near: f32 = cameraAt(r, 18);
  const far: f32 = cameraAt(r, 19);
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  let bestDepth: f32 = mvsDepth[base + i];
  let bestWord: u32 = mvsNormal[base + i];
  let bestCost: f32 = mvsCost[i];
  // Propagation: (±1, 0), (0, ±1), (±5, 0), (0, ±5), all of the other color.
  for (let c: u32 = 0; c < 8; c++) {
    let dx: i32 = 0;
    let dy: i32 = 0;
    const step: i32 = c < 4 ? 1 : 5;
    const side: u32 = c % 4;
    if (side === 0) dx = -step;
    else if (side === 1) dx = step;
    else if (side === 2) dy = -step;
    else dy = step;
    const qx: i32 = i32(x) + dx;
    const qy: i32 = i32(y) + dy;
    if (qx < 0 || qy < 0 || qx >= i32(W) || qy >= i32(H)) continue;
    const q: u32 = u32(qy) * W + u32(qx);
    const dq: f32 = mvsDepth[base + q];
    if (!(dq > 0.)) continue;
    const word: u32 = mvsNormal[base + q];
    const n: vec3 = decodeNormal(word);
    // The candidate's plane moved to this pixel: where this pixel's ray meets it.
    const facing: f32 = dot(n, ray);
    if (!(facing < 0.)) continue;
    const depth: f32 = dot(n, referenceRay(f32(qx), f32(qy)) * dq) / facing;
    if (!(depth >= near && depth <= far)) continue;
    const cost: f32 = hypothesisCost(x, y, depth, n, stats);
    if (cost < bestCost) {
      bestDepth = depth;
      bestWord = word;
      bestCost = cost;
    }
  }
  // Refinement 1: a new random depth and normal.
  const randomDepthValue: f32 = randomDepth(i, 0);
  const randomWord: u32 = encodeNormal(randomNormal(i, 1, ray));
  const randomCost: f32 = hypothesisCost(x, y, randomDepthValue, decodeNormal(randomWord), stats);
  if (randomCost < bestCost) {
    bestDepth = randomDepthValue;
    bestWord = randomWord;
    bestCost = randomCost;
  }
  // Refinement 2: the best hypothesis so far, perturbed by ±(range × 2^−iteration) in inverse
  // depth and up to 30° × 2^−iteration in normal.
  const inverseNear: f32 = 1. / near;
  const inverseFar: f32 = 1. / far;
  const span: f32 = mvsControl.perturbDepth * (inverseNear - inverseFar);
  const shift: f32 = (2. * random(i, 3) - 1.) * span;
  const inverse: f32 = clamp(1. / bestDepth + shift, inverseFar, inverseNear);
  const perturbedDepth: f32 = 1. / inverse;
  const bestNormal: vec3 = decodeNormal(bestWord);
  const e: vec3 = randomUnit(i, 4);
  const tangent: vec3 = e - bestNormal * dot(e, bestNormal);
  const tangentLength: f32 = length(tangent);
  let turned: vec3 = bestNormal;
  if (tangentLength > 0.000001) {
    const angle: f32 = random(i, 6) * mvsControl.perturbAngle;
    turned = bestNormal * cos(angle) + tangent * (sin(angle) / tangentLength);
  }
  if (!(dot(turned, ray) < 0.)) turned = bestNormal;
  const perturbedWord: u32 = encodeNormal(turned);
  const perturbedNormal: vec3 = decodeNormal(perturbedWord);
  const perturbedCost: f32 = hypothesisCost(x, y, perturbedDepth, perturbedNormal, stats);
  if (perturbedCost < bestCost) {
    bestDepth = perturbedDepth;
    bestWord = perturbedWord;
    bestCost = perturbedCost;
  }
  if (active) {
    mvsDepth[base + i] = bestDepth;
    mvsNormal[base + i] = bestWord;
    mvsCost[i] = bestCost;
  }
}
