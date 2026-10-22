'use typeshade';

// Dense MVS (`docs/dense-mvs.md`, sections 3.4 to 3.11): the undistortion of the gray images,
// the random initialization and the seeds of a reference view, the red-black PatchMatch
// propagation with its refinement in its photometric and its geometric form, the filter, and the
// fusion of the filtered depth maps into points. Depth kernels cache the source poses once
// per workgroup on WebGPU; the CPU tier retains the scalar calculation. The fusion's flag
// scan is the shared one of `scan.shade.ts`.
//
// Layouts (section 4.1). P = W × H. Pixel index i = y × W + x. A per-view array is view-major.
// - `mvsCameras`: f32 × 24 per view: R (9, row-major, world to camera), t (3), fx, fy, cx, cy,
//   W, H, dmin, dmax, the camera center (3), 0.
// - `mvsSources`: u32 × 7 per view: the source count, then up to 6 view indices.
// - `mvsSeeds`: f32 × 3 per seed: x, y (integer pixels of the undistorted image), depth.
// - `mvsGray`: f32, N × P, the undistorted gray in [0, 1], −1 for "no data".
// - `mvsDepth`: f32, N × P, the photometric optical-axis depth in normalized units.
// - `mvsGeoDepth`: f32, N × P, the geometric depth, then the filtered depth (0: no depth).
// - `mvsNormal`: u32, N × P, the octahedral unit normal in the view's camera frame.
// - `mvsCost`: f32, P, the cost of the reference's current hypothesis.
// - `mvsRgba`: u32, P, the packed RGBA of the reference as uploaded (fusion).
// - `mvsFlags`: u32, P, the fusion flag of each pixel, then the scan's counts (`scan.shade.ts`).
// - `mvsFlagOffsets`: u32 × 2 per block of 64 pixels, the scan's block offsets.
// - `mvsPoints`: f32 × 3 per fused point, the position in the normalized world frame.
// - `mvsPointData`: u32 × 2 per fused point, the octahedral world normal and the packed RGBA.
//
// Amendment 3: a depth pass binds a window. `mvsGray` and, in the geometric pass, `mvsDepth` hold
// view slots, `mvsCameras` the slots' records (element 23 the packed index), `mvsSources` one
// record of source slots, and `mvsGeoDepth`, `mvsNormal`, `mvsCost` the reference alone
// (`mvsControl.state` 0). The fusion binds slots of `mvsGeoDepth` and `mvsNormal`, with the
// slots' order by packed index after the camera records of `mvsCameras`.
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
  // 1 for the geometric pass (section 3.8): the reference's hypotheses are in `mvsGeoDepth`, and
  // each source cost gains geometricWeight × min(ψ, geometricMaxCost).
  geometric: u32;
  geometricWeight: f32;
  geometricMaxCost: f32;
  // The packed index of the reference, which the random draws hash (3.5). `reference` is its slot
  // (Amendment 3), so a slot changes no draw.
  view: u32;
  // The view index of the reference's own arrays (`mvsGeoDepth`, `mvsNormal` and, in the
  // photometric pass, `mvsDepth`) and of its record in `mvsSources`: 0 when they hold the
  // reference alone (Amendment 3).
  state: u32;
  // WebGPU only: cache each resident source pose once per workgroup (at most 7 view slots).
  relativePoseCache: u32;
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
// The thresholds of the filter (section 3.9).
class MvsFilterControl {
  // The largest photometric cost averaged over the best sources (1 − the least NCC).
  maxCost: f32;
  // The least number of consistent sources.
  minConsistent: u32;
  // The largest forward-backward reprojection error of a consistent source, pixels.
  maxError: f32;
  // cos of the least triangulation angle of a consistent source.
  minAngleCos: f32;
  // The least confidence of a kept depth (section 10.3); 0 keeps every depth that passes the rules.
  minConfidence: f32;
  // 1: the filter writes the confidence of each pixel to `mvsCost`, 0 for a pixel without depth,
  // for the weights of the TSDF fusion (section 10.4). 0: `mvsCost` keeps the hypothesis's cost.
  writeConfidence: u32;
}
// The pose of source s relative to the reference: X_s = R_rs X_r + t_rs, R_rs by rows.
class MvsRelativePose {
  m0: vec3;
  m1: vec3;
  m2: vec3;
  t: vec3;
}
let relativePoses: workgroup<array<MvsRelativePose, 7>>;
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
declare const filterControl: uniform<MvsFilterControl>;
declare const undistortInput: storage<array<f32>>;
declare const undistortOutput: storage<array<f32>, 'read_write'>;
declare const mvsCameras: storage<array<f32>>;
declare const mvsSources: storage<array<u32>>;
declare const mvsSeeds: storage<array<f32>>;
declare const mvsGray: storage<array<f32>>;
declare const mvsDepth: storage<array<f32>, 'read_write'>;
declare const mvsGeoDepth: storage<array<f32>, 'read_write'>;
declare const mvsNormal: storage<array<u32>, 'read_write'>;
declare const mvsCost: storage<array<f32>, 'read_write'>;
// The fusion of one reference (sections 3.10 and 3.11).
class MvsFuseControl {
  // N, the packed views with a depth map.
  views: u32;
  // The least number of consistent views of a point, the reference included.
  minViews: u32;
  // The largest relative depth error, reprojection error in pixels and cos of the largest normal
  // angle of a consistent view.
  maxDepthError: f32;
  maxReprojection: f32;
  minNormalCos: f32;
  // The reference's distortion, for the color sample through the map of `mvsUndistort` (3.11).
  distorted: u32;
  k1: f32;
  k2: f32;
  radialLimit: f32;
  // Where the slots of the fusion views, in ascending packed index, start in `mvsCameras`.
  order: u32;
  // Streamed fusion (Amendment 3): the chunk's pixels `first..end`, and 1 for the first group of
  // fusion views, whose sums start from the reference pixel.
  first: u32;
  end: u32;
  start: u32;
}
// The fused point of a reference pixel: whether it is emitted, the point and the unit normal, both
// in the normalized world frame.
class MvsFused {
  emit: u32;
  point: vec3;
  normal: vec3;
}
declare const fuseControl: uniform<MvsFuseControl>;
declare const mvsRgba: storage<array<u32>>;
declare const mvsFlags: storage<array<u32>, 'read_write'>;
declare const mvsFlagOffsets: storage<array<u32>>;
declare const mvsPoints: storage<array<f32>, 'read_write'>;
declare const mvsPointData: storage<array<u32>, 'read_write'>;
// The sums of the streamed fusion (Amendment 3): 8 f32 per pixel of the chunk, the point sum (3),
// the normal sum (3), the count and the lower flag.
declare const mvsFuseSums: storage<array<f32>, 'read_write'>;
// The partial costs of the streamed depth passes (Amendment 3): per element `stride` f32, for
// each hypothesis the `costs` least source costs so far in ascending order, and in the filter the
// count of consistent sources after them.
declare const mvsPartial: storage<array<f32>, 'read_write'>;
class MvsStreamControl {
  // 0 the initialization, 1 the seeds, 2 the propagation candidates (8 and the random one), 3 the
  // perturbed candidate, 4 the geometric initialization, 5 the filter.
  phase: u32;
  // 1 for the first group of sources: the lists start empty.
  start: u32;
  // The first element of the chunk that `mvsPartial` holds.
  first: u32;
  // f32 per element of `mvsPartial`, and the least costs kept per hypothesis (`bestSources`).
  stride: u32;
  costs: u32;
  // K: the sources of the reference over all groups.
  sources: u32;
}
declare const streamControl: uniform<MvsStreamControl>;
// The upload of one view to a slot (Amendment 3): `count` values to `offset`.
class MvsCopyControl {
  count: u32;
  offset: u32;
}
declare const copyControl: uniform<MvsCopyControl>;
declare const copyInputF32: storage<array<f32>>;
declare const copyOutputF32: storage<array<f32>, 'read_write'>;
declare const copyInputU32: storage<array<u32>>;
declare const copyOutputU32: storage<array<u32>, 'read_write'>;

const NO_COST: f32 = 2.;
// Marks a source cost already summed: larger than any source cost, photometric or geometric.
const TAKEN: f32 = 1000000.;
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
    hash(hash(hash(mvsControl.seed) ^ mvsControl.view) ^ pixel) ^
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
  const pose: MvsRelativePose = relativePose(s);
  const m0: vec3 = pose.m0;
  const m1: vec3 = pose.m1;
  const m2: vec3 = pose.m2;
  const t: vec3 = pose.t;
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

/** The pose of view `s` relative to the reference (R_rs = R_s R_rᵀ, t_rs = t_s − R_rs t_r). */
function computeRelativePose(s: u32): MvsRelativePose {
  const r: u32 = mvsControl.reference;
  const a0: vec3 = rotationRow(r, 0);
  const a1: vec3 = rotationRow(r, 1);
  const a2: vec3 = rotationRow(r, 2);
  const b0: vec3 = rotationRow(s, 0);
  const b1: vec3 = rotationRow(s, 1);
  const b2: vec3 = rotationRow(s, 2);
  const m0: vec3 = vec3(dot(b0, a0), dot(b0, a1), dot(b0, a2));
  const m1: vec3 = vec3(dot(b1, a0), dot(b1, a1), dot(b1, a2));
  const m2: vec3 = vec3(dot(b2, a0), dot(b2, a1), dot(b2, a2));
  const tr: vec3 = translation(r);
  const t: vec3 = translation(s) - vec3(dot(m0, tr), dot(m1, tr), dot(m2, tr));
  return { m0: m0, m1: m1, m2: m2, t: t };
}

// All lanes call this before any pixel-dependent control flow. Only the source lanes
// calculate a transform; the barrier publishes their 448-byte table to the whole group.
// The CPU path never executes the barrier and keeps its original operation order.
function prepareRelativePoses(lane: u32): void {
  if (mvsControl.relativePoseCache !== 0) {
    const base: u32 = mvsControl.state * SOURCE_SLOTS;
    const count: u32 = min(mvsSources[base], SOURCE_SLOTS - 1);
    if (lane < count) {
      const s: u32 = mvsSources[base + 1 + lane];
      relativePoses[s] = computeRelativePose(s);
    }
    workgroupBarrier();
  }
}

function relativePose(s: u32): MvsRelativePose {
  if (mvsControl.relativePoseCache !== 0) return relativePoses[s];
  return computeRelativePose(s);
}

/**
 * The forward-backward reprojection error ψ_s of section 3.8, in pixels, at most `limit`: X (the
 * point of reference pixel (x, y), reference frame) projects into source `s` at q. The source's
 * photometric depth at the nearest pixel of q, on the ray of q, projects back into the reference
 * at p′, and ψ_s = |(x, y) − p′|. A q outside the source, a source pixel without depth or a
 * point behind a camera gives `limit`.
 */
function reprojectionError(s: u32, x: f32, y: f32, X: vec3, limit: f32): f32 {
  const r: u32 = mvsControl.reference;
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const pose: MvsRelativePose = relativePose(s);
  const Xs: vec3 = vec3(dot(pose.m0, X), dot(pose.m1, X), dot(pose.m2, X)) + pose.t;
  if (!(Xs.z > 0.)) return limit;
  const fxs: f32 = cameraAt(s, 12);
  const fys: f32 = cameraAt(s, 13);
  const cxs: f32 = cameraAt(s, 14);
  const cys: f32 = cameraAt(s, 15);
  const qx: f32 = (fxs * Xs.x) / Xs.z + cxs;
  const qy: f32 = (fys * Xs.y) / Xs.z + cys;
  if (!(qx >= -0.5 && qy >= -0.5 && qx < f32(W) - 0.5 && qy < f32(H) - 0.5)) return limit;
  const ix: u32 = min(u32(floor(qx + 0.5)), W - 1);
  const iy: u32 = min(u32(floor(qy + 0.5)), H - 1);
  const ds: f32 = mvsDepth[s * W * H + iy * W + ix];
  if (!(ds > 0.)) return limit;
  const d: vec3 = vec3((qx - cxs) / fxs, (qy - cys) / fys, 1.) * ds - pose.t;
  const back: vec3 = pose.m0 * d.x + pose.m1 * d.y + pose.m2 * d.z;
  if (!(back.z > 0.)) return limit;
  const px: f32 = (cameraAt(r, 12) * back.x) / back.z + cameraAt(r, 14);
  const py: f32 = (cameraAt(r, 13) * back.y) / back.z + cameraAt(r, 15);
  const error: f32 = length(vec2(px - x, py - y));
  if (!(error < limit)) return limit;
  return error;
}

/** cos of the triangulation angle at X (reference frame) between the rays to r and to `s`. */
function triangulationCosine(s: u32, X: vec3): f32 {
  const pose: MvsRelativePose = relativePose(s);
  const C: vec3 = -(pose.m0 * pose.t.x + pose.m1 * pose.t.y + pose.m2 * pose.t.z);
  const toR: vec3 = -X;
  const toS: vec3 = C - X;
  return dot(toR, toS) / (length(toR) * length(toS));
}

/** The reference's current depth at pixel index `index` of the view-major arrays. */
function stateDepth(index: u32): f32 {
  if (mvsControl.geometric !== 0) return mvsGeoDepth[index];
  return mvsDepth[index];
}

/** The cost of one source, with the geometric term of section 3.8 in the geometric pass. */
function termCost(s: u32, x: u32, y: u32, X: vec3, n: vec3, stats: MvsWindow): f32 {
  let cost: f32 = sourceCost(s, i32(x), i32(y), X, n, stats);
  // Section 3.8: cost_s = (1 − NCC_s) + λ · min(ψ_s, ψmax).
  if (mvsControl.geometric !== 0)
    cost +=
      mvsControl.geometricWeight *
      reprojectionError(s, f32(x), f32(y), X, mvsControl.geometricMaxCost);
  return cost;
}

/**
 * The aggregated cost: the mean of the `bestSources` least source costs (section 3.7), with the
 * geometric term of section 3.8 in the geometric pass.
 */
function hypothesisCost(x: u32, y: u32, depth: f32, n: vec3, stats: MvsWindow): f32 {
  if (stats.ok === 0) return NO_COST;
  const X: vec3 = referenceRay(f32(x), f32(y)) * depth;
  const base: u32 = mvsControl.state * SOURCE_SLOTS;
  const count: u32 = min(mvsSources[base], SOURCE_SLOTS - 1);
  let costs: array<f32, 6> = fill<f32, 6>(NO_COST);
  for (let k: u32 = 0; k < count; k++) costs[k] = termCost(mvsSources[base + 1 + k], x, y, X, n, stats);
  const best: u32 = min(mvsControl.bestSources, count);
  let sum: f32 = 0.;
  for (let j: u32 = 0; j < best; j++) {
    let least: u32 = 0;
    for (let k: u32 = 1; k < count; k++) if (costs[k] < costs[least]) least = k;
    sum += costs[least];
    costs[least] = TAKEN;
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

/**
 * The position in the distorted image of undistorted pixel (u, v), in x and y, and 1 in z when it
 * lies inside the W × H image and r² is below `radialLimit` (section 3.4). Otherwise z is 0.
 */
function distortedPosition(
  u: f32,
  v: f32,
  fx: f32,
  fy: f32,
  cx: f32,
  cy: f32,
  k1: f32,
  k2: f32,
  radialLimit: f32,
  W: u32,
  H: u32,
): vec3 {
  const x: f32 = (u - cx) / fx;
  const y: f32 = (v - cy) / fy;
  const r2: f32 = x * x + y * y;
  let inside: f32 = 0.;
  let sx: f32 = 0.;
  let sy: f32 = 0.;
  if (r2 < radialLimit) {
    const d: f32 = 1. + k1 * r2 + k2 * r2 * r2;
    sx = fx * x * d + cx;
    sy = fy * y * d + cy;
    if (sx >= 0. && sy >= 0. && sx <= f32(W - 1) && sy <= f32(H - 1)) inside = 1.;
  }
  return vec3(sx, sy, inside);
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
    const s: vec3 = distortedPosition(
      u,
      v,
      undistortControl.fx,
      undistortControl.fy,
      undistortControl.cx,
      undistortControl.cy,
      undistortControl.k1,
      undistortControl.k2,
      undistortControl.radialLimit,
      W,
      H,
    );
    value = -1.;
    if (s.z > 0.) {
      const fx: f32 = floor(s.x);
      const fy: f32 = floor(s.y);
      const x0: u32 = u32(fx);
      const y0: u32 = u32(fy);
      const x1: u32 = min(x0 + 1, W - 1);
      const y1: u32 = min(y0 + 1, H - 1);
      const ax: f32 = s.x - fx;
      const ay: f32 = s.y - fy;
      value = mix(
        mix(undistortInput[y0 * W + x0], undistortInput[y0 * W + x1], ax),
        mix(undistortInput[y1 * W + x0], undistortInput[y1 * W + x1], ax),
        ay,
      );
    }
  }
  if (active) undistortOutput[undistortControl.view * P + i] = value;
}

/**
 * A random hypothesis and its cost for each pixel `first..end` of the reference (section 3.5).
 * Draws 0 to 2 of iteration 0.
 */
@compute([64])
export function mvsInitialize(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsInitializeAt(gid);
}
@compute([64])
export function mvsInitializeOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsInitializeAt(gid);
}
function mvsInitializeAt(gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.state;
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
export function mvsSeed(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsSeedAt(gid);
}
@compute([64])
export function mvsSeedOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsSeedAt(gid);
}
function mvsSeedAt(gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.state;
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
export function mvsPropagate(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsPropagateAt(gid);
}
@compute([64])
export function mvsPropagateOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsPropagateAt(gid);
}
function mvsPropagateAt(gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const P: u32 = W * H;
  const r: u32 = mvsControl.reference;
  const base: u32 = mvsControl.state * P;
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
  let bestDepth: f32 = stateDepth(base + i);
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
    const dq: f32 = stateDepth(base + q);
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
    if (mvsControl.geometric !== 0) mvsGeoDepth[base + i] = bestDepth;
    else mvsDepth[base + i] = bestDepth;
    mvsNormal[base + i] = bestWord;
    mvsCost[i] = bestCost;
  }
}

/**
 * The start of the geometric pass for the pixels `first..end` of the reference (section 3.8): the
 * photometric depth is copied to `mvsGeoDepth`, the normal stays, and the cost of the hypothesis
 * is computed again with the geometric term (`mvsControl.geometric` is 1).
 */
@compute([64])
export function mvsGeometricInitialize(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsGeometricInitializeAt(gid);
}
@compute([64])
export function mvsGeometricInitializeOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsGeometricInitializeAt(gid);
}
function mvsGeometricInitializeAt(gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const x: u32 = i % W;
  const y: u32 = i / W;
  const depth: f32 = mvsDepth[r * P + i];
  const n: vec3 = decodeNormal(mvsNormal[mvsControl.state * P + i]);
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  const cost: f32 = hypothesisCost(x, y, depth, n, stats);
  if (active) {
    mvsGeoDepth[mvsControl.state * P + i] = depth;
    mvsCost[i] = cost;
  }
}

/**
 * The confidence of a depth (section 10.3): the product of a photometric term (1 − cost / the
 * cost bound), the fraction of the sources that are consistent, and a view-normal term that is 0
 * at a grazing angle (cos 0.15) and 1 from cos 0.7. `ray` is the pixel's ray in the reference's
 * frame and `n` its normal there.
 */
function depthConfidence(photometric: f32, consistent: f32, sources: u32, n: vec3, ray: vec3): f32 {
  const c1: f32 = clamp(1. - photometric / filterControl.maxCost, 0., 1.);
  const c2: f32 = consistent / f32(max(sources, 1));
  const c3: f32 = clamp((-dot(n, normalize(ray)) - 0.15) / 0.55, 0., 1.);
  return c1 * c2 * c3;
}

/**
 * The filter of section 3.9 over the pixels `first..end` of the reference's geometric depth map,
 * in place. A pixel keeps its depth when the depth lies in [dmin, dmax], the mean of the
 * `bestSources` least photometric source costs is at most `maxCost`, and at least
 * `minConsistent` sources have ψ_s ≤ `maxError` and a triangulation angle of at least the
 * filter's least angle. Any other pixel gets depth 0, "no depth". ψ_s reads the sources'
 * photometric depths, as the geometric pass does. With `writeConfidence`, the pixel's confidence
 * (section 10.3) goes to `mvsCost`, 0 for a pixel without depth.
 */
@compute([64])
export function mvsFilter(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsFilterAt(gid);
}
@compute([64])
export function mvsFilterOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsFilterAt(gid);
}
function mvsFilterAt(gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const x: u32 = i % W;
  const y: u32 = i / W;
  const depth: f32 = mvsGeoDepth[mvsControl.state * P + i];
  const n: vec3 = decodeNormal(mvsNormal[mvsControl.state * P + i]);
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  const X: vec3 = referenceRay(f32(x), f32(y)) * depth;
  const base: u32 = mvsControl.state * SOURCE_SLOTS;
  const count: u32 = min(mvsSources[base], SOURCE_SLOTS - 1);
  let costs: array<f32, 6> = fill<f32, 6>(NO_COST);
  let consistent: u32 = 0;
  for (let k: u32 = 0; k < count; k++) {
    const s: u32 = mvsSources[base + 1 + k];
    costs[k] = sourceCost(s, i32(x), i32(y), X, n, stats);
    const error: f32 = reprojectionError(s, f32(x), f32(y), X, mvsControl.geometricMaxCost);
    if (error <= filterControl.maxError && triangulationCosine(s, X) <= filterControl.minAngleCos)
      consistent++;
  }
  const best: u32 = min(mvsControl.bestSources, count);
  let sum: f32 = 0.;
  for (let j: u32 = 0; j < best; j++) {
    let least: u32 = 0;
    for (let k: u32 = 1; k < count; k++) if (costs[k] < costs[least]) least = k;
    sum += costs[least];
    costs[least] = TAKEN;
  }
  let photometric: f32 = NO_COST;
  if (stats.ok !== 0 && best > 0) photometric = sum / f32(best);
  const confidence: f32 = depthConfidence(
    photometric,
    f32(consistent),
    count,
    n,
    referenceRay(f32(x), f32(y)),
  );
  const keep: bool =
    depth >= cameraAt(r, 18) &&
    depth <= cameraAt(r, 19) &&
    photometric <= filterControl.maxCost &&
    consistent >= filterControl.minConsistent &&
    !(confidence < filterControl.minConfidence);
  if (active && !keep) mvsGeoDepth[mvsControl.state * P + i] = 0.;
  if (active && filterControl.writeConfidence !== 0) mvsCost[i] = select(0., confidence, keep);
}

/** The camera center of view `v` in the world frame. */
function cameraCenter(v: u32): vec3 {
  return vec3(cameraAt(v, 20), cameraAt(v, 21), cameraAt(v, 22));
}
/** R_vᵀ c: a vector of view `v`'s camera frame in the world frame. */
function toWorld(v: u32, c: vec3): vec3 {
  return rotationRow(v, 0) * c.x + rotationRow(v, 1) * c.y + rotationRow(v, 2) * c.z;
}
/** R_v X + t_v: a world point in view `v`'s camera frame. */
function toCamera(v: u32, X: vec3): vec3 {
  return (
    vec3(dot(rotationRow(v, 0), X), dot(rotationRow(v, 1), X), dot(rotationRow(v, 2), X)) +
    translation(v)
  );
}

/** The running sums of a fused point (section 3.10). */
class MvsFuseState {
  pointSum: vec3;
  normalSum: vec3;
  count: u32;
  lower: u32;
}

/** The point of reference pixel `i` at its filtered depth, in the normalized world frame. */
function fusePoint(i: u32): vec3 {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  const depth: f32 = mvsGeoDepth[r * P + i];
  return toWorld(r, referenceRay(f32(i % W), f32(i / W)) * depth) + cameraCenter(r);
}
function fuseNormal(i: u32): vec3 {
  const P: u32 = mvsControl.width * mvsControl.height;
  const r: u32 = mvsControl.reference;
  return toWorld(r, decodeNormal(mvsNormal[r * P + i]));
}

/**
 * View j's test of reference pixel `i` (section 3.10), with X the pixel's point and `normal` its
 * world normal. X projects to q, and j's filtered depth d_j at the nearest pixel of q gives j's
 * point there. View j is consistent when |z_j(X) − d_j| ≤ `maxDepthError` · d_j, j's point
 * projects into the reference within `maxReprojection` pixels of the pixel, and the world normals
 * part by at most the largest normal angle. A consistent view adds its point and its normal to
 * the sums, and sets `lower` when its packed index (element 23 of its camera record) is lower than
 * the reference's.
 */
function fuseView(j: u32, i: u32, X: vec3, normal: vec3, state: MvsFuseState): MvsFuseState {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const P: u32 = W * H;
  const r: u32 = mvsControl.reference;
  const x: u32 = i % W;
  const y: u32 = i / W;
  const Xj: vec3 = toCamera(j, X);
  if (!(Xj.z > 0.)) return state;
  const fx: f32 = cameraAt(j, 12);
  const fy: f32 = cameraAt(j, 13);
  const cx: f32 = cameraAt(j, 14);
  const cy: f32 = cameraAt(j, 15);
  const qx: f32 = (fx * Xj.x) / Xj.z + cx;
  const qy: f32 = (fy * Xj.y) / Xj.z + cy;
  if (!(qx >= -0.5 && qy >= -0.5 && qx < f32(W) - 0.5 && qy < f32(H) - 0.5)) return state;
  const ix: u32 = min(u32(floor(qx + 0.5)), W - 1);
  const iy: u32 = min(u32(floor(qy + 0.5)), H - 1);
  const k: u32 = j * P + iy * W + ix;
  const dj: f32 = mvsGeoDepth[k];
  if (!(dj > 0.)) return state;
  if (!(abs(Xj.z - dj) <= fuseControl.maxDepthError * dj)) return state;
  // j's point at its pixel, back into the reference.
  const Pj: vec3 =
    toWorld(j, vec3((f32(ix) - cx) / fx, (f32(iy) - cy) / fy, 1.) * dj) + cameraCenter(j);
  const Pr: vec3 = toCamera(r, Pj);
  if (!(Pr.z > 0.)) return state;
  const px: f32 = (cameraAt(r, 12) * Pr.x) / Pr.z + cameraAt(r, 14);
  const py: f32 = (cameraAt(r, 13) * Pr.y) / Pr.z + cameraAt(r, 15);
  if (!(length(vec2(px - f32(x), py - f32(y))) <= fuseControl.maxReprojection)) return state;
  const nj: vec3 = toWorld(j, decodeNormal(mvsNormal[k]));
  if (!(dot(normal, nj) >= fuseControl.minNormalCos)) return state;
  let lower: u32 = state.lower;
  if (cameraAt(j, 23) < cameraAt(r, 23)) lower = 1;
  return {
    pointSum: state.pointSum + Pj,
    normalSum: state.normalSum + nj,
    count: state.count + 1,
    lower: lower,
  };
}

/**
 * The fusion views of the window, in ascending packed index, into `state`: their slots are the
 * `views` values of `mvsCameras` from `fuseControl.order` on (a storage array of its own would be
 * a ninth of `mvsScatterPoints`, past WebGPU's default of 8 per stage).
 */
function fuseViews(i: u32, X: vec3, normal: vec3, state: MvsFuseState): MvsFuseState {
  const r: u32 = mvsControl.reference;
  let s: MvsFuseState = state;
  for (let k: u32 = 0; k < fuseControl.views; k++) {
    const j: u32 = u32(mvsCameras[fuseControl.order + k]);
    if (j === r) continue;
    s = fuseView(j, i, X, normal, s);
  }
  return s;
}

/** The fused point of a reference pixel from its sums: emitted when the rule of 3.10 holds. */
function fuseResult(i: u32, state: MvsFuseState): MvsFused {
  const P: u32 = mvsControl.width * mvsControl.height;
  const depth: f32 = mvsGeoDepth[mvsControl.reference * P + i];
  let emit: u32 = 0;
  if (depth > 0. && state.count >= fuseControl.minViews && state.lower === 0) emit = 1;
  return {
    emit: emit,
    point: state.pointSum * (1. / f32(state.count)),
    normal: normalize(state.normalSum),
  };
}

/**
 * The fusion of reference pixel `i` (section 3.10) against the fusion views of the window. The
 * point is emitted when the reference has a depth, 1 + the consistent views is at least
 * `minViews`, and no consistent view has a lower packed index than the reference. The point and
 * the normal are the means over the reference and the consistent views.
 */
function fusePixel(i: u32): MvsFused {
  const X: vec3 = fusePoint(i);
  const normal: vec3 = fuseNormal(i);
  const state: MvsFuseState = fuseViews(i, X, normal, {
    pointSum: X,
    normalSum: normal,
    count: 1,
    lower: 0,
  });
  return fuseResult(i, state);
}

/** Byte `k` of a packed RGBA word, as f32. */
function channel(word: u32, k: u32): f32 {
  return f32((word >> (8 * k)) & 255);
}
/**
 * The color of reference pixel `i` (section 3.11): the packed RGBA at the pixel's position in the
 * distorted image, the map of `mvsUndistort`, each channel bilinear and rounded. 0 outside the
 * image. A view without distortion takes its pixel unchanged.
 */
function referenceColor(i: u32): u32 {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const r: u32 = mvsControl.reference;
  let color: u32 = mvsRgba[i];
  if (fuseControl.distorted !== 0) {
    const s: vec3 = distortedPosition(
      f32(i % W),
      f32(i / W),
      cameraAt(r, 12),
      cameraAt(r, 13),
      cameraAt(r, 14),
      cameraAt(r, 15),
      fuseControl.k1,
      fuseControl.k2,
      fuseControl.radialLimit,
      W,
      H,
    );
    color = 0;
    if (s.z > 0.) {
      const fx: f32 = floor(s.x);
      const fy: f32 = floor(s.y);
      const x0: u32 = u32(fx);
      const y0: u32 = u32(fy);
      const x1: u32 = min(x0 + 1, W - 1);
      const y1: u32 = min(y0 + 1, H - 1);
      const ax: f32 = s.x - fx;
      const ay: f32 = s.y - fy;
      const c00: u32 = mvsRgba[y0 * W + x0];
      const c10: u32 = mvsRgba[y0 * W + x1];
      const c01: u32 = mvsRgba[y1 * W + x0];
      const c11: u32 = mvsRgba[y1 * W + x1];
      for (let k: u32 = 0; k < 4; k++) {
        const value: f32 = mix(
          mix(channel(c00, k), channel(c10, k), ax),
          mix(channel(c01, k), channel(c11, k), ax),
          ay,
        );
        color = color | (min(u32(floor(value + 0.5)), 255) << (8 * k));
      }
    }
  }
  return color;
}

/**
 * The fusion flags of the reference's pixels `first..end` (section 3.10): 1 for a pixel that
 * emits a point, 0 otherwise, for the shared flag scan (`planFlagScan` in `scan.ts`).
 */
@compute([64])
export function mvsFuseFlags(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const fused: MvsFused = fusePixel(i);
  if (active) mvsFlags[i] = fused.emit;
}

/**
 * The fused points of the reference's pixels `first..end`, after the scan: an emitting pixel
 * computes its point again and writes it, its normal and its color to its slot, the block's
 * offset plus its count within the block, minus 1. The order of the points is the pixel order.
 */
@compute([64])
export function mvsScatterPoints(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const fused: MvsFused = fusePixel(i);
  const color: u32 = referenceColor(i);
  let row: u32 = 0;
  if (fused.emit !== 0) row = mvsFlagOffsets[(i / 64) * 2] + mvsFlags[i] - 1;
  if (active && fused.emit !== 0) {
    mvsPoints[row * 3] = fused.point.x;
    mvsPoints[row * 3 + 1] = fused.point.y;
    mvsPoints[row * 3 + 2] = fused.point.z;
    mvsPointData[row * 2] = encodeNormal(fused.normal);
    mvsPointData[row * 2 + 1] = color;
  }
}

/** The upload of one view to its slot (Amendment 3): `count` f32 values to `offset`. */
@compute([64])
export function mvsCopyF32(@builtin('global_invocation_id') gid: vec3u): void {
  const active: bool = gid.x < copyControl.count;
  const i: u32 = min(gid.x, max(copyControl.count, 1) - 1);
  const value: f32 = copyInputF32[i];
  if (active) copyOutputF32[copyControl.offset + i] = value;
}
/** The upload of one view to its slot (Amendment 3): `count` u32 values to `offset`. */
@compute([64])
export function mvsCopyU32(@builtin('global_invocation_id') gid: vec3u): void {
  const active: bool = gid.x < copyControl.count;
  const i: u32 = min(gid.x, max(copyControl.count, 1) - 1);
  const value: u32 = copyInputU32[i];
  if (active) copyOutputU32[copyControl.offset + i] = value;
}

// ═══ Streamed depth passes (Amendment 3) ═══
// The reference and a group of its sources are resident. `mvsStreamCosts` adds the costs of the
// group's sources to the least costs of each hypothesis in `mvsPartial`, and `mvsStreamSelect`
// aggregates them after the last group, by the rules of `hypothesisCost`, `mvsPropagate`,
// `mvsGeometricInitialize` and `mvsFilter`. A hypothesis is regenerated from the state and the
// counter-based draws in each kernel, so every group sees the same hypothesis.

// A hypothesis of one element of a streamed pass: its pixel, its depth and normal word, whether
// it is evaluated, and whether the element's lane writes.
class MvsHypothesis {
  pixel: u32;
  x: u32;
  y: u32;
  depth: f32;
  word: u32;
  valid: u32;
  active: u32;
}

/** The pixel of element `e` of a streamed pass, by the phase's mapping. */
function streamPixel(e: u32): vec3u {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const phase: u32 = streamControl.phase;
  if (phase === 1) {
    const sx: u32 = min(u32(mvsSeeds[e * 3]), W - 1);
    const sy: u32 = min(u32(mvsSeeds[e * 3 + 1]), H - 1);
    return vec3u(sx, sy, 1);
  }
  if (phase === 2 || phase === 3) {
    const half: u32 = (W + 1) / 2;
    const y: u32 = min(e / half, H - 1);
    const x: u32 = 2 * (e % half) + ((y + mvsControl.color) % 2);
    if (x >= W) return vec3u(W - 1, y, 0);
    return vec3u(x, y, 1);
  }
  return vec3u(e % W, e / W, 1);
}

/** Hypothesis `h` of element `e` (see `MvsStreamControl.phase`). */
function streamHypothesis(e: u32, h: u32): MvsHypothesis {
  const W: u32 = mvsControl.width;
  const H: u32 = mvsControl.height;
  const P: u32 = W * H;
  const r: u32 = mvsControl.reference;
  const base: u32 = mvsControl.state * P;
  const phase: u32 = streamControl.phase;
  const at: vec3u = streamPixel(e);
  const x: u32 = at.x;
  const y: u32 = at.y;
  const i: u32 = y * W + x;
  const ray: vec3 = referenceRay(f32(x), f32(y));
  let depth: f32 = 0.;
  let word: u32 = 0;
  let valid: u32 = 1;
  if (phase === 0) {
    depth = randomDepth(i, 0);
    word = encodeNormal(randomNormal(i, 1, ray));
  } else if (phase === 1) {
    depth = mvsSeeds[e * 3 + 2];
    word = encodeNormal(-normalize(ray));
  } else if (phase === 2 && h < 8) {
    // Propagation: (±1, 0), (0, ±1), (±5, 0), (0, ±5), all of the other color (3.6).
    let dx: i32 = 0;
    let dy: i32 = 0;
    const step: i32 = h < 4 ? 1 : 5;
    const side: u32 = h % 4;
    if (side === 0) dx = -step;
    else if (side === 1) dx = step;
    else if (side === 2) dy = -step;
    else dy = step;
    const qx: i32 = i32(x) + dx;
    const qy: i32 = i32(y) + dy;
    valid = 0;
    if (qx >= 0 && qy >= 0 && qx < i32(W) && qy < i32(H)) {
      const q: u32 = u32(qy) * W + u32(qx);
      const dq: f32 = stateDepth(base + q);
      if (dq > 0.) {
        const candidate: u32 = mvsNormal[base + q];
        const n: vec3 = decodeNormal(candidate);
        const facing: f32 = dot(n, ray);
        if (facing < 0.) {
          const moved: f32 = dot(n, referenceRay(f32(qx), f32(qy)) * dq) / facing;
          if (moved >= cameraAt(r, 18) && moved <= cameraAt(r, 19)) {
            depth = moved;
            word = candidate;
            valid = 1;
          }
        }
      }
    }
  } else if (phase === 2) {
    // Refinement 1: a new random depth and normal.
    depth = randomDepth(i, 0);
    word = encodeNormal(randomNormal(i, 1, ray));
  } else if (phase === 3) {
    // Refinement 2: the best hypothesis so far, perturbed (3.6), as `mvsPropagate` does.
    const near: f32 = cameraAt(r, 18);
    const far: f32 = cameraAt(r, 19);
    const bestDepth: f32 = stateDepth(base + i);
    const bestWord: u32 = mvsNormal[base + i];
    const inverseNear: f32 = 1. / near;
    const inverseFar: f32 = 1. / far;
    const span: f32 = mvsControl.perturbDepth * (inverseNear - inverseFar);
    const shift: f32 = (2. * random(i, 3) - 1.) * span;
    const inverse: f32 = clamp(1. / bestDepth + shift, inverseFar, inverseNear);
    depth = 1. / inverse;
    const bestNormal: vec3 = decodeNormal(bestWord);
    const v: vec3 = randomUnit(i, 4);
    const tangent: vec3 = v - bestNormal * dot(v, bestNormal);
    const tangentLength: f32 = length(tangent);
    let turned: vec3 = bestNormal;
    if (tangentLength > 0.000001) {
      const angle: f32 = random(i, 6) * mvsControl.perturbAngle;
      turned = bestNormal * cos(angle) + tangent * (sin(angle) / tangentLength);
    }
    if (!(dot(turned, ray) < 0.)) turned = bestNormal;
    word = encodeNormal(turned);
  } else if (phase === 4) {
    depth = mvsDepth[r * P + i];
    word = mvsNormal[base + i];
  } else {
    depth = mvsGeoDepth[base + i];
    word = mvsNormal[base + i];
  }
  return { pixel: i, x: x, y: y, depth: depth, word: word, valid: valid, active: at.z };
}

/** The hypotheses of one element in the current phase. */
function streamHypotheses(): u32 {
  if (streamControl.phase === 2) return 9;
  return 1;
}

/** The mean of the `bestSources` least costs of hypothesis `h` of element `e` (3.7). */
function streamAggregate(e: u32, h: u32): f32 {
  const at: u32 = (e - streamControl.first) * streamControl.stride + h * streamControl.costs;
  const best: u32 = min(mvsControl.bestSources, streamControl.sources);
  let sum: f32 = 0.;
  for (let j: u32 = 0; j < best; j++) sum += mvsPartial[at + j];
  if (best === 0) return NO_COST;
  return sum / f32(best);
}

/**
 * The costs of the resident sources for each hypothesis of the elements `first..end`, added to
 * the least costs in `mvsPartial` (the first group starts them empty). In the filter (phase 5)
 * the costs are photometric, and the count of consistent sources follows the least costs.
 */
@compute([64])
export function mvsStreamCosts(
  @builtin('global_invocation_id') gid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  prepareRelativePoses(lid.x);
  mvsStreamCostsAt(gid);
}
@compute([64])
export function mvsStreamCostsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  mvsStreamCostsAt(gid);
}
function mvsStreamCostsAt(gid: vec3u): void {
  let e: u32 = mvsRange.first + gid.x;
  const active: bool = e < mvsRange.end;
  if (!active) e = max(mvsRange.end, 1) - 1;
  const phase: u32 = streamControl.phase;
  const L: u32 = streamControl.costs;
  const last: u32 = L - 1;
  const at: vec3u = streamPixel(e);
  const stats: MvsWindow = referenceWindow(i32(at.x), i32(at.y));
  const count: u32 = min(mvsSources[mvsControl.state * SOURCE_SLOTS], SOURCE_SLOTS - 1);
  const base: u32 = (e - streamControl.first) * streamControl.stride;
  const hypotheses: u32 = streamHypotheses();
  for (let h: u32 = 0; h < hypotheses; h++) {
    const hyp: MvsHypothesis = streamHypothesis(e, h);
    let list: array<f32, 6> = fill<f32, 6>(TAKEN);
    let consistent: f32 = 0.;
    if (streamControl.start === 0) {
      for (let j: u32 = 0; j < L; j++) list[j] = mvsPartial[base + h * L + j];
      if (phase === 5) consistent = mvsPartial[base + L];
    }
    if (hyp.valid !== 0 && (stats.ok !== 0 || phase === 5)) {
      const X: vec3 = referenceRay(f32(hyp.x), f32(hyp.y)) * hyp.depth;
      const n: vec3 = decodeNormal(hyp.word);
      for (let k: u32 = 0; k < count; k++) {
        const s: u32 = mvsSources[mvsControl.state * SOURCE_SLOTS + 1 + k];
        let cost: f32 = 0.;
        if (phase === 5) {
          cost = sourceCost(s, i32(hyp.x), i32(hyp.y), X, n, stats);
          const error: f32 = reprojectionError(
            s,
            f32(hyp.x),
            f32(hyp.y),
            X,
            mvsControl.geometricMaxCost,
          );
          if (
            error <= filterControl.maxError &&
            triangulationCosine(s, X) <= filterControl.minAngleCos
          )
            consistent += 1.;
        } else cost = termCost(s, hyp.x, hyp.y, X, n, stats);
        // Insert into the ascending list of the L least costs.
        if (cost < list[last]) {
          let m: u32 = last;
          for (let q: u32 = 0; q < last; q++) {
            if (!(cost < list[m - 1])) break;
            list[m] = list[m - 1];
            m--;
          }
          list[m] = cost;
        }
      }
    }
    if (active) {
      for (let j: u32 = 0; j < L; j++) mvsPartial[base + h * L + j] = list[j];
      if (phase === 5) mvsPartial[base + L] = consistent;
    }
  }
}

/**
 * The end of a streamed phase for the elements `first..end`: each element's hypotheses with their
 * aggregated costs replace the state by the rules of the window-mode kernel of the phase.
 */
@compute([64])
export function mvsStreamSelect(@builtin('global_invocation_id') gid: vec3u): void {
  const W: u32 = mvsControl.width;
  const P: u32 = W * mvsControl.height;
  const r: u32 = mvsControl.reference;
  const base: u32 = mvsControl.state * P;
  let e: u32 = mvsRange.first + gid.x;
  let active: bool = e < mvsRange.end;
  if (!active) e = max(mvsRange.end, 1) - 1;
  const phase: u32 = streamControl.phase;
  const first: MvsHypothesis = streamHypothesis(e, 0);
  if (first.active === 0) active = false;
  const i: u32 = first.pixel;
  const stats: MvsWindow = referenceWindow(i32(first.x), i32(first.y));
  let depth: f32 = stateDepth(base + i);
  let word: u32 = mvsNormal[base + i];
  let cost: f32 = mvsCost[i];
  if (phase === 2) {
    for (let h: u32 = 0; h < 9; h++) {
      const hyp: MvsHypothesis = streamHypothesis(e, h);
      if (hyp.valid === 0) continue;
      let c: f32 = NO_COST;
      if (stats.ok !== 0) c = streamAggregate(e, h);
      if (c < cost) {
        depth = hyp.depth;
        word = hyp.word;
        cost = c;
      }
    }
  } else if (phase === 3) {
    let c: f32 = NO_COST;
    if (stats.ok !== 0) c = streamAggregate(e, 0);
    if (c < cost) {
      depth = first.depth;
      word = first.word;
      cost = c;
    }
  } else if (phase !== 5) {
    depth = first.depth;
    word = first.word;
    cost = NO_COST;
    if (stats.ok !== 0) cost = streamAggregate(e, 0);
  }
  if (phase === 5) {
    // The filter (3.9): the photometric cost and the consistent sources of the final hypothesis.
    let photometric: f32 = NO_COST;
    const best: u32 = min(mvsControl.bestSources, streamControl.sources);
    if (stats.ok !== 0 && best > 0) photometric = streamAggregate(e, 0);
    const consistent: f32 =
      mvsPartial[(e - streamControl.first) * streamControl.stride + streamControl.costs];
    const confidence: f32 = depthConfidence(
      photometric,
      consistent,
      streamControl.sources,
      decodeNormal(word),
      referenceRay(f32(first.x), f32(first.y)),
    );
    const keep: bool =
      first.depth >= cameraAt(r, 18) &&
      first.depth <= cameraAt(r, 19) &&
      photometric <= filterControl.maxCost &&
      consistent >= f32(filterControl.minConsistent) &&
      !(confidence < filterControl.minConfidence);
    if (active && !keep) mvsGeoDepth[base + i] = 0.;
    if (active && filterControl.writeConfidence !== 0) mvsCost[i] = select(0., confidence, keep);
    return;
  }
  if (active) {
    if (phase === 4 || mvsControl.geometric !== 0) mvsGeoDepth[base + i] = depth;
    else mvsDepth[base + i] = depth;
    if (phase !== 4) mvsNormal[base + i] = word;
    mvsCost[i] = cost;
  }
}

// ═══ Streamed fusion (Amendment 3) ═══
// The reference stays in its slot and the fusion views pass through in groups. Each group adds
// its consistent views to the sums of the chunk's pixels `fuseControl.first..end`, in ascending
// packed index, as `fusePixel` does in one pass.

function fuseSumsAt(i: u32): u32 {
  return (i - fuseControl.first) * 8;
}

/** The resident fusion views' tests of the pixels `first..end`, added to `mvsFuseSums`. */
@compute([64])
export function mvsFuseAccumulate(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const X: vec3 = fusePoint(i);
  const normal: vec3 = fuseNormal(i);
  const at: u32 = fuseSumsAt(i);
  let state: MvsFuseState = { pointSum: X, normalSum: normal, count: 1, lower: 0 };
  if (fuseControl.start === 0)
    state = {
      pointSum: vec3(mvsFuseSums[at], mvsFuseSums[at + 1], mvsFuseSums[at + 2]),
      normalSum: vec3(mvsFuseSums[at + 3], mvsFuseSums[at + 4], mvsFuseSums[at + 5]),
      count: u32(mvsFuseSums[at + 6]),
      lower: u32(mvsFuseSums[at + 7]),
    };
  state = fuseViews(i, X, normal, state);
  if (active) {
    mvsFuseSums[at] = state.pointSum.x;
    mvsFuseSums[at + 1] = state.pointSum.y;
    mvsFuseSums[at + 2] = state.pointSum.z;
    mvsFuseSums[at + 3] = state.normalSum.x;
    mvsFuseSums[at + 4] = state.normalSum.y;
    mvsFuseSums[at + 5] = state.normalSum.z;
    mvsFuseSums[at + 6] = f32(state.count);
    mvsFuseSums[at + 7] = f32(state.lower);
  }
}

function fuseSumsState(i: u32): MvsFuseState {
  const at: u32 = fuseSumsAt(i);
  return {
    pointSum: vec3(mvsFuseSums[at], mvsFuseSums[at + 1], mvsFuseSums[at + 2]),
    normalSum: vec3(mvsFuseSums[at + 3], mvsFuseSums[at + 4], mvsFuseSums[at + 5]),
    count: u32(mvsFuseSums[at + 6]),
    lower: u32(mvsFuseSums[at + 7]),
  };
}

/**
 * The fusion flags of the chunk's elements `first..end` (local indices): 1 for a pixel that emits
 * a point. An element past the chunk's last pixel gets 0, so the scan of the chunk counts no
 * flag of an earlier chunk.
 */
@compute([64])
export function mvsFuseEmit(@builtin('global_invocation_id') gid: vec3u): void {
  let e: u32 = mvsRange.first + gid.x;
  const active: bool = e < mvsRange.end;
  if (!active) e = max(mvsRange.end, 1) - 1;
  const inside: bool = fuseControl.first + e < fuseControl.end;
  const i: u32 = min(fuseControl.first + e, fuseControl.end - 1);
  const fused: MvsFused = fuseResult(i, fuseSumsState(i));
  let flag: u32 = 0;
  if (inside) flag = fused.emit;
  if (active) mvsFlags[e] = flag;
}

/** The points of the chunk's pixels `first..end` from their sums, after the scan of the chunk. */
@compute([64])
export function mvsFuseScatterSums(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = mvsRange.first + gid.x;
  const active: bool = i < mvsRange.end;
  if (!active) i = max(mvsRange.end, 1) - 1;
  const e: u32 = i - fuseControl.first;
  const fused: MvsFused = fuseResult(i, fuseSumsState(i));
  const color: u32 = referenceColor(i);
  let row: u32 = 0;
  if (fused.emit !== 0) row = mvsFlagOffsets[(e / 64) * 2] + mvsFlags[e] - 1;
  if (active && fused.emit !== 0) {
    mvsPoints[row * 3] = fused.point.x;
    mvsPoints[row * 3 + 1] = fused.point.y;
    mvsPoints[row * 3 + 2] = fused.point.z;
    mvsPointData[row * 2] = encodeNormal(fused.normal);
    mvsPointData[row * 2 + 1] = color;
  }
}
