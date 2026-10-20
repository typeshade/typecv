'use typeshade';
// Model generation as a state machine in resident TypeShade storage: the rows of a model are
// appended to its triangular factor sixteen at a time, and one dispatch then solves the factor
// for its null vector. No kernel iterates to convergence: a minimal sample's null vector is a
// back-substitution, and a refit's is refined by a fixed number of inverse iterations.
import { CameraModel } from './numerics.shade.ts';
import { appendRow, Null9, basis3, inverse3 } from './projective.shade.ts';
class PairControl {
  first: CameraModel;
  second: CameraModel;
  // 0: essential in normalized coordinates, 1: homography in input pixels.
  kind: u32;
  threshold: f64;
  seed: u32;
  start: u32;
  trials: u32;
  maximum: u32;
  minTrials: u32;
  blocks: u32;
  refit: u32;
  logFailure: f32;
  // Packed valid rows shared by H/E pair verification; zero for standalone APIs.
  verification: u32;
  rowStart: u32;
  rowCount: u32;
}
declare const control: uniform<PairControl>;
declare const coords: storage<array<f64>>;
declare const invertible: storage<array<u32>>;
declare const models: storage<array<f64>, 'read_write'>;
declare const valid: storage<array<u32>, 'read_write'>;
declare const state: storage<array<u32>>;
declare const mask: storage<array<u32>>;
// Per model: the triangular factor R[81], the null vector x[9], mean[4], the two normalization
// scales, and the second-smallest and the largest singular value of R.
declare const modelWork: storage<array<f64>, 'read_write'>;
// Per model: active, row limit, eight sampled rows, solved, reserved.
declare const modelState: storage<array<u32>, 'read_write'>;
// The scheduling word the host reads every few batches: the adaptive trial budget.
declare const modelProgress: storage<array<u32>, 'read_write'>;
// The workgroup counts of the entries that dispatch indirectly (TypeShade change 0075), one
// [x, y, z] triple each: the batch's prepare (words 0 to 2), its row appends, null solve and
// finish (3 to 5), `scorePairModels` (6 to 8) and `scorePairModelsOracle` (9 to 11). The GPU
// writes them, so the host submits a batch without reading what the batch before it decided.
// No entry that dispatches with these counts binds them.
declare const counts: storage<array<u32>, 'read_write'>;
function randomWord(word: u32): u32 {
  let s = word;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
function getPoint(i: u32): vec4f64 {
  return vec4f64(coords[i * 4], coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]);
}
function pointIndex(j: u32, indices: array<u32, 8>): u32 {
  if (control.refit === 0) return indices[j];
  return j;
}
function included(i: u32): bool {
  return invertible[i] !== 0 && (control.refit === 0 || mask[i] !== 0);
}

@compute([8])
export function preparePairModels(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length) return;
  valid[h] = 0;
  for (let j: u32 = 0; j < 12; j++) modelState[h * 12 + j] = 0;
  for (let j: u32 = 0; j < 18; j++) models[h * 18 + j] = 0;
  let minimum: u32 = 8;
  if (control.kind === 1) minimum = 4;
  if (state[7] < minimum || h >= control.trials) return;
  if (control.refit !== 0 && (h !== 0 || state[6] === 0 || state[3] < minimum)) return;
  if (
    control.refit === 0 &&
    (control.start + h >= state[0] || control.start + h >= control.maximum)
  )
    return;
  let n: u32 = invertible.length;
  if (control.verification !== 0) n = state[7];
  let indices: array<u32, 8> = array<u32, 8>();
  let s: u32 = control.seed ^ ((control.start + h + 1) * 0x9e3779b9);
  if (s === 0) s = 0x85ebca77;
  // Bounded sampling without replacement. Sorted insertion avoids rejection loops.
  for (let j: u32 = 0; j < minimum; j++) {
    s = randomWord(s);
    let index: u32 = s % (n - j);
    for (let k: u32 = 0; k < j; k++) if (index >= indices[k]) index++;
    let place: u32 = j;
    for (let k: u32 = 0; k < j; k++)
      if (index < indices[k]) {
        place = k;
        break;
      }
    for (let k: u32 = j; k > place; k--) indices[k] = indices[k - 1];
    indices[place] = index;
  }
  let limit: u32 = minimum;
  if (control.refit !== 0) limit = n;
  let count: u32 = 0;
  let mean: vec4f64 = vec4f64(f64(0));
  for (let j: u32 = 0; j < limit; j++) {
    const i: u32 = pointIndex(j, indices);
    if (!included(i)) {
      if (control.refit === 0) return;
      else continue;
    }
    mean += getPoint(i);
    count++;
  }
  if (count < minimum) return;
  mean /= f64(f32(count));
  let da: f64 = 0,
    db: f64 = 0;
  for (let j: u32 = 0; j < limit; j++) {
    const i: u32 = pointIndex(j, indices);
    if (!included(i)) continue;
    const d = getPoint(i) - mean;
    da += sqrt(d.x * d.x + d.y * d.y);
    db += sqrt(d.z * d.z + d.w * d.w);
  }
  if (!(da > 1e-12 && db > 1e-12)) return;
  const sa: f64 = (1.4142135623730951 * f64(f32(count))) / da;
  const sb: f64 = (1.4142135623730951 * f64(f32(count))) / db;
  const base: u32 = h * 98;
  for (let j: u32 = 0; j < 98; j++) modelWork[base + j] = 0;
  modelWork[base + 90] = mean.x; modelWork[base + 91] = mean.y;
  modelWork[base + 92] = mean.z; modelWork[base + 93] = mean.w;
  modelWork[base + 94] = sa; modelWork[base + 95] = sb;
  modelState[h * 12] = 1;
  modelState[h * 12 + 1] = limit;
  for (let j: u32 = 0; j < 8; j++) modelState[h * 12 + 2 + j] = indices[j];
}
@compute([8])
export function appendPairModelRows(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0) return;
  const base: u32 = h * 98;
  const limit: u32 = modelState[h * 12 + 1];
  let indices: array<u32, 8> = array<u32, 8>();
  for (let j: u32 = 0; j < 8; j++) indices[j] = modelState[h * 12 + 2 + j];
  const mean: vec4f64 = vec4f64(modelWork[base + 90], modelWork[base + 91], modelWork[base + 92], modelWork[base + 93]);
  const sa: f64 = modelWork[base + 94];
  const sb: f64 = modelWork[base + 95];
  let qr: array<f64, 81> = fill<f64, 81>(f64(0));
  for (let j: u32 = 0; j < 81; j++) qr[j] = modelWork[base + j];
  for (let j: u32 = control.rowStart; j < min(limit, control.rowStart + control.rowCount); j++) {
    const i: u32 = pointIndex(j, indices);
    if (!included(i)) continue;
    const d = getPoint(i) - mean;
    const x: f64 = d.x * sa,
      y: f64 = d.y * sa,
      u: f64 = d.z * sb,
      v: f64 = d.w * sb;
    if (control.kind === 0) {
      let row: array<f64, 9> = [u * x, u * y, u, v * x, v * y, v, x, y, 1];
      qr = appendRow(qr, row);
    } else {
      let row0: array<f64, 9> = [-x, -y, -1, 0, 0, 0, u * x, u * y, u];
      let row1: array<f64, 9> = [0, 0, 0, -x, -y, -1, v * x, v * y, v];
      qr = appendRow(qr, row0);
      qr = appendRow(qr, row1);
    }
  }
  for (let j: u32 = 0; j < 81; j++) modelWork[base + j] = qr[j];
}
// Solve the upper triangular factor R of a model for its null vector and the two singular values
// the degeneracy test needs, in one dispatch of fixed cost. Nothing iterates to convergence.
//
// The null vector starts as a back-substitution at the smallest pivot and is refined by four
// inverse iterations on R^T R (two triangular solves each); for those solves every pivot below
// 1e-12 of the largest is raised to that value, which moves each singular value by at most that
// much (Weyl). A minimal sample fills eight rows of R, so its smallest pivot is at the working
// precision and the start is already the null vector. One inverse iteration leaves an error
// factor of (sigma_9 / sigma_8)^2, far below a double in any well-posed sample; a sample with
// sigma_8 near sigma_9 is the degenerate one the test rejects.
//
// The second-smallest singular value: for a factor whose smallest singular value |R x| is at
// most 1e-6 of the largest pivot (every minimal sample, and a refit of exact data), it is found
// through the null vector by `secondSingular` below; for a refit with noise, by six inverse
// iterations on R^T R orthogonal to the null vector, whose amplification is then bounded. The
// largest singular value comes from eight power iterations.
function forwardSolve(r: array<f64, 81>, v: array<f64, 9>, n: u32): array<f64, 9> {
  // R^T y = v on the leading n x n block: R^T is lower triangular.
  let y: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let i: u32 = 0; i < n; i++) {
    let sum: f64 = v[i];
    for (let j: u32 = 0; j < i; j++) sum -= r[j * 9 + i] * y[j];
    y[i] = sum / r[i * 9 + i];
  }
  return y;
}
function backSolve(r: array<f64, 81>, v: array<f64, 9>, n: u32): array<f64, 9> {
  // R z = v on the leading n x n block.
  let z: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let ii: u32 = 0; ii < n; ii++) {
    const i: u32 = n - 1 - ii;
    let sum: f64 = v[i];
    for (let j: u32 = i + 1; j < n; j++) sum -= r[i * 9 + j] * z[j];
    z[i] = sum / r[i * 9 + i];
  }
  return z;
}
function dot9(a: array<f64, 9>, b: array<f64, 9>): f64 {
  let sum: f64 = 0;
  for (let j: u32 = 0; j < 9; j++) sum += a[j] * b[j];
  return sum;
}
function normalized(v: array<f64, 9>): array<f64, 9> {
  // Scaled by the largest entry first, so that a vector a solve amplified does not overflow.
  let out = v;
  let largest: f64 = 0;
  for (let j: u32 = 0; j < 9; j++) largest = max(largest, abs(v[j]));
  if (!(largest > 0)) return out;
  for (let j: u32 = 0; j < 9; j++) out[j] = v[j] / largest;
  const norm: f64 = sqrt(dot9(out, out));
  for (let j: u32 = 0; j < 9; j++) out[j] /= norm;
  return out;
}
function residualNorm(r: array<f64, 81>, v: array<f64, 9>): f64 {
  // |R v| for a unit v: the singular value v approximates.
  let norm: f64 = 0;
  for (let i: u32 = 0; i < 9; i++) {
    let sum: f64 = 0;
    for (let j: u32 = i; j < 9; j++) sum += r[i * 9 + j] * v[j];
    norm += sum * sum;
  }
  return sqrt(norm);
}
// The second-smallest singular value of R, given its unit null vector x. The Householder
// reflection H that maps x onto the ninth axis turns R into R H whose ninth column is R x, zero to
// the working precision, so the second-smallest singular value of R is the smallest of the 9x8
// block B of R H. B is reduced by a Householder QR to an 8x8 triangular T (a zero pivot is a rank
// loss: the sample is degenerate and the value is zero), and six inverse iterations on T^T T, two
// triangular solves each, give the singular vector; the value is |T w| for the unit w. R itself
// is never solved with, so a factor whose first eight columns are dependent, as every exact
// sample of a pure translation is, is handled like any other.
function secondSingular(r: array<f64, 81>, x: array<f64, 9>): f64 {
  let u = x;
  let sign: f64 = 1;
  if (x[8] < 0) sign = -1;
  u[8] = x[8] + sign;
  const uu: f64 = dot9(u, u);
  let ru: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let i: u32 = 0; i < 9; i++) for (let k: u32 = i; k < 9; k++) ru[i] += r[i * 9 + k] * u[k];
  // B, 9 rows by 8 columns.
  let b: array<f64, 72> = fill<f64, 72>(f64(0));
  let scale: f64 = 0;
  for (let i: u32 = 0; i < 9; i++)
    for (let j: u32 = 0; j < 8; j++) {
      let value: f64 = -2 * u[j] * ru[i] / uu;
      if (j >= i) value += r[i * 9 + j];
      b[i * 8 + j] = value;
      scale = max(scale, abs(value));
    }
  if (!(scale > 0)) return 0;
  // Householder QR of B in place: the triangle T is what stays on and above the diagonal.
  for (let k: u32 = 0; k < 8; k++) {
    let norm: f64 = 0;
    for (let i: u32 = k; i < 9; i++) norm += b[i * 8 + k] * b[i * 8 + k];
    norm = sqrt(norm);
    if (norm <= 1e-13 * scale) return 0;
    let alpha: f64 = -norm;
    if (b[k * 8 + k] < 0) alpha = norm;
    let v: array<f64, 9> = fill<f64, 9>(f64(0));
    for (let i: u32 = k; i < 9; i++) v[i] = b[i * 8 + k];
    v[k] -= alpha;
    const vv: f64 = dot9(v, v);
    if (vv > 0)
      for (let j: u32 = k; j < 8; j++) {
        let along: f64 = 0;
        for (let i: u32 = k; i < 9; i++) along += v[i] * b[i * 8 + j];
        const factor: f64 = 2 * along / vv;
        for (let i: u32 = k; i < 9; i++) b[i * 8 + j] -= factor * v[i];
      }
  }
  for (let k: u32 = 0; k < 8; k++) if (abs(b[k * 8 + k]) <= 1e-13 * scale) return 0;
  let w: array<f64, 8> = fill<f64, 8>(f64(1));
  for (let step: u32 = 0; step < 6; step++) {
    // T^T t = w, then T z = t.
    let t: array<f64, 8> = fill<f64, 8>(f64(0));
    for (let i: u32 = 0; i < 8; i++) {
      let sum: f64 = w[i];
      for (let j: u32 = 0; j < i; j++) sum -= b[j * 8 + i] * t[j];
      t[i] = sum / b[i * 8 + i];
    }
    let largest: f64 = 0;
    for (let i: u32 = 0; i < 8; i++) largest = max(largest, abs(t[i]));
    if (!(largest > 0)) return 0;
    for (let i: u32 = 0; i < 8; i++) t[i] /= largest;
    let z: array<f64, 8> = fill<f64, 8>(f64(0));
    for (let ii: u32 = 0; ii < 8; ii++) {
      const i: u32 = 7 - ii;
      let sum: f64 = t[i];
      for (let j: u32 = i + 1; j < 8; j++) sum -= b[i * 8 + j] * z[j];
      z[i] = sum / b[i * 8 + i];
    }
    largest = 0;
    for (let i: u32 = 0; i < 8; i++) largest = max(largest, abs(z[i]));
    if (!(largest > 0)) return 0;
    let norm: f64 = 0;
    for (let i: u32 = 0; i < 8; i++) {
      z[i] /= largest;
      norm += z[i] * z[i];
    }
    norm = sqrt(norm);
    for (let i: u32 = 0; i < 8; i++) w[i] = z[i] / norm;
  }
  let norm: f64 = 0;
  for (let i: u32 = 0; i < 8; i++) {
    let sum: f64 = 0;
    for (let j: u32 = i; j < 8; j++) sum += b[i * 8 + j] * w[j];
    norm += sum * sum;
  }
  return sqrt(norm);
}
@compute([8])
export function solvePairModelNull(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0 || modelState[h * 12 + 10] !== 0) return;
  const base: u32 = h * 98;
  let r: array<f64, 81> = fill<f64, 81>(f64(0));
  let scale: f64 = 0;
  for (let j: u32 = 0; j < 81; j++) scale = max(scale, abs(modelWork[base + j]));
  if (!(scale > 1e-20 && scale < 1e20)) {
    modelState[h * 12] = 0;
    return;
  }
  for (let j: u32 = 0; j < 81; j++) r[j] = modelWork[base + j] / scale;
  let largestPivot: f64 = 0;
  for (let i: u32 = 0; i < 9; i++) largestPivot = max(largestPivot, abs(r[i * 9 + i]));
  if (!(largestPivot > 0)) {
    modelState[h * 12] = 0;
    return;
  }
  let smallest: u32 = 0;
  for (let i: u32 = 1; i < 9; i++) if (abs(r[i * 9 + i]) < abs(r[smallest * 9 + smallest])) smallest = i;
  let smallestOfEight: u32 = 0;
  for (let i: u32 = 1; i < 8; i++) if (abs(r[i * 9 + i]) < abs(r[smallestOfEight * 9 + smallestOfEight])) smallestOfEight = i;
  let original = r;
  const floor: f64 = 1e-14 * largestPivot;
  for (let i: u32 = 0; i < 9; i++) if (abs(r[i * 9 + i]) < floor) r[i * 9 + i] = floor;
  // The null vector: a back-substitution at the smallest pivot, refined by inverse iteration.
  let x: array<f64, 9> = fill<f64, 9>(f64(0));
  x[smallest] = 1;
  for (let ii: u32 = 0; ii < smallest; ii++) {
    const i: u32 = smallest - 1 - ii;
    let sum: f64 = 0;
    for (let j: u32 = i + 1; j <= smallest; j++) sum += r[i * 9 + j] * x[j];
    x[i] = -sum / r[i * 9 + i];
  }
  x = normalized(x);
  for (let step: u32 = 0; step < 4; step++) x = normalized(backSolve(r, normalized(forwardSolve(r, x, 9)), 9));
  // The smallest singular value, |R x|, decides the method for the second-smallest: the pivots
  // of a triangular factor can all be moderate while the factor is singular.
  const exact: bool = residualNorm(original, x) <= 1e-6 * largestPivot;
  let second: f64 = 0;
  if (exact) second = secondSingular(original, x);
  else {
    // A refit: the second-smallest singular value by inverse iteration orthogonal to x.
    let w: array<f64, 9> = fill<f64, 9>(f64(0));
    w[smallestOfEight] = 1;
    if (smallest === smallestOfEight) w[8] = 1;
    for (let step: u32 = 0; step < 7; step++) {
      const along: f64 = dot9(w, x);
      for (let j: u32 = 0; j < 9; j++) w[j] -= along * x[j];
      w = normalized(w);
      if (step < 6) {
        w = backSolve(r, normalized(forwardSolve(r, w, 9)), 9);
        const again: f64 = dot9(w, x);
        for (let j: u32 = 0; j < 9; j++) w[j] -= again * x[j];
      }
    }
    second = residualNorm(r, w);
  }
  // The largest singular value: power iteration on R^T R.
  let v: array<f64, 9> = fill<f64, 9>(f64(1));
  v = normalized(v);
  for (let step: u32 = 0; step < 8; step++) {
    let rv: array<f64, 9> = fill<f64, 9>(f64(0));
    for (let i: u32 = 0; i < 9; i++) for (let j: u32 = i; j < 9; j++) rv[i] += r[i * 9 + j] * v[j];
    let next: array<f64, 9> = fill<f64, 9>(f64(0));
    for (let i: u32 = 0; i < 9; i++) for (let j: u32 = i; j < 9; j++) next[j] += r[i * 9 + j] * rv[i];
    v = normalized(next);
  }
  for (let j: u32 = 0; j < 9; j++) modelWork[base + 81 + j] = x[j];
  modelWork[base + 96] = second * scale;
  modelWork[base + 97] = residualNorm(r, v) * scale;
  modelState[h * 12 + 10] = 1;
}
@compute([8])
export function finishPairModels(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0 || modelState[h * 12 + 10] === 0) return;
  const base: u32 = h * 98;
  const mean: vec4f64 = vec4f64(modelWork[base + 90], modelWork[base + 91], modelWork[base + 92], modelWork[base + 93]);
  const sa: f64 = modelWork[base + 94];
  const sb: f64 = modelWork[base + 95];
  let zero: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let j: u32 = 0; j < 9; j++) zero[j] = modelWork[base + 81 + j];
  const fit: Null9 = { vector: zero, second: modelWork[base + 96], largest: modelWork[base + 97], ok: 1 };
  let relative: f64 = 1e-7;
  if (control.kind === 1) relative = 1e-9;
  if (fit.ok === 0 || fit.second <= relative * fit.largest) return;
  let left: array<f64, 9> = [sb, 0, 0, 0, sb, 0, -sb * mean.z, -sb * mean.w, 1];
  if (control.kind === 1) {
    const inv = inverse3(fit.vector);
    if (inv[9] === 0) return;
    left = [1 / sb, 0, mean.z, 0, 1 / sb, mean.w, 0, 0, 1];
  }
  let right: array<f64, 9> = [sa, 0, -sa * mean.x, 0, sa, -sa * mean.y, 0, 0, 1];
  let temp: array<f64, 9> = fill<f64, 9>(f64(0));
  let model: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let r: u32 = 0; r < 3; r++)
    for (let c: u32 = 0; c < 3; c++)
      for (let k: u32 = 0; k < 3; k++) temp[r * 3 + c] += left[r * 3 + k] * fit.vector[k * 3 + c];
  let norm: f64 = 0;
  for (let r: u32 = 0; r < 3; r++)
    for (let c: u32 = 0; c < 3; c++) {
      for (let k: u32 = 0; k < 3; k++) model[r * 3 + c] += temp[r * 3 + k] * right[k * 3 + c];
      norm += model[r * 3 + c] * model[r * 3 + c];
    }
  if (!(norm > 1e-30 && norm < 1e30)) return;
  for (let j: u32 = 0; j < 9; j++) model[j] /= sqrt(norm);
  if (control.kind === 0) {
    const basis = basis3(model);
    if (basis.ok === 0) return;
    for (let r: u32 = 0; r < 3; r++)
      for (let c: u32 = 0; c < 3; c++)
        models[h * 18 + r * 3 + c] =
          (basis.U[r * 3] * basis.V[c * 3] + basis.U[r * 3 + 1] * basis.V[c * 3 + 1]) *
          0.7071067811865475;
  } else {
    const inv = inverse3(model);
    if (inv[9] === 0) return;
    for (let j: u32 = 0; j < 9; j++) {
      models[h * 18 + j] = model[j];
      models[h * 18 + 9 + j] = inv[j];
    }
  }
  valid[h] = 1;
}

// Whether the batch of `control` runs: a refit always does; a search batch does when its first
// trial is below the adaptive budget (`state[0]`) and the requested maximum. The first batch of a
// search always runs, as it did when the host decided: its prepare clears the models of the
// search before it, and its scoring is the chunk that teaches the device the rate of a trial.
// A batch that does not run changes nothing: its prepare, row appends, null solve, finish and
// scoring dispatch zero workgroups, and `selectPairModel` returns at once.
function batchRuns(): bool {
  if (control.refit !== 0 || control.start === 0) return true;
  return control.start < state[0] && control.start < control.maximum;
}
function writeCount(offset: u32, x: u32): void {
  counts[offset] = x;
  counts[offset + 1] = 1;
  counts[offset + 2] = 1;
}
// The workgroups of the batch's prepare: one per eight models, or none when the batch does not
// run. It runs after the selection of the batch before, which is what shrinks the budget.
@compute([1])
export function planPairModelBatch(): void {
  let groups: u32 = 0;
  if (batchRuns()) groups = (modelState.length / 12 + 7) / 8;
  writeCount(0, groups);
}
// The workgroups of the rest of the batch, after its prepare: the row appends, the null solve and
// the finish run while some model is active and unsolved, and the scoring runs with the batch.
@compute([1])
export function summarizePairModelWork(): void {
  let remaining: u32 = 0;
  for (let h: u32 = 0; h < modelState.length / 12; h++) {
    if (modelState[h * 12] !== 0 && modelState[h * 12 + 10] === 0) remaining++;
  }
  const runs: bool = batchRuns();
  let groups: u32 = 0;
  if (runs && remaining !== 0) groups = (modelState.length / 12 + 7) / 8;
  writeCount(3, groups);
  let lanes: u32 = 0;
  if (runs) lanes = control.trials * control.blocks;
  writeCount(6, lanes);
  writeCount(9, (lanes + 7) / 8);
}

// The scheduling word: the adaptive trial budget, read by the host every few batches to leave
// the search early.
@compute([1])
export function snapshotPairModelBudget(): void {
  modelProgress[0] = state[0];
}
