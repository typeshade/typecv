'use typeshade';
// Watchdog-compatible model generation. All QR/SVD state stays in resident
// TypeShade storage; row appends are chunked and SVD convergence is bounded.
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
// Per model: A[81], V[81], mean[4], two normalization scales.
declare const modelWork: storage<array<f64>, 'read_write'>;
// Per model: active, row limit, eight sampled rows, converged, reserved.
declare const modelState: storage<array<u32>, 'read_write'>;
declare const modelProgress: storage<array<u32>, 'read_write'>;
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
  const base: u32 = h * 168;
  for (let j: u32 = 0; j < 168; j++) modelWork[base + j] = 0;
  modelWork[base + 162] = mean.x; modelWork[base + 163] = mean.y;
  modelWork[base + 164] = mean.z; modelWork[base + 165] = mean.w;
  modelWork[base + 166] = sa; modelWork[base + 167] = sb;
  modelState[h * 12] = 1;
  modelState[h * 12 + 1] = limit;
  for (let j: u32 = 0; j < 8; j++) modelState[h * 12 + 2 + j] = indices[j];
}
@compute([8])
export function appendPairModelRows(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0) return;
  const base: u32 = h * 168;
  const limit: u32 = modelState[h * 12 + 1];
  let indices: array<u32, 8> = array<u32, 8>();
  for (let j: u32 = 0; j < 8; j++) indices[j] = modelState[h * 12 + 2 + j];
  const mean: vec4f64 = vec4f64(modelWork[base + 162], modelWork[base + 163], modelWork[base + 164], modelWork[base + 165]);
  const sa: f64 = modelWork[base + 166];
  const sb: f64 = modelWork[base + 167];
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
@compute([8])
export function initializePairModelSvd(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0) return;
  const base: u32 = h * 168;
  let scale: f64 = 0;
  for (let j: u32 = 0; j < 81; j++) scale = max(scale, abs(modelWork[base + j]));
  if (!(scale > 1e-20 && scale < 1e20)) { modelState[h * 12] = 0; return; }
  for (let j: u32 = 0; j < 81; j++) modelWork[base + j] /= scale;
  for (let j: u32 = 0; j < 9; j++) modelWork[base + 81 + j * 9 + j] = 1;
}
@compute([8])
export function sweepPairModelSvd(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0 || modelState[h * 12 + 10] !== 0) return;
  const base: u32 = h * 168;
  for (let batch: u32 = 0; batch < 40; batch++) {
    for (let sweep: u32 = 0; sweep < 4; sweep++) {
      let changed: bool = false;
      for (let p: u32 = 0; p < 9; p++)
        for (let q: u32 = p + 1; q < 9; q++) {
          let aa: f64 = 0,
            bb: f64 = 0,
            ab: f64 = 0;
          for (let r: u32 = 0; r < 9; r++) {
            const x = modelWork[base + r * 9 + p],
              y = modelWork[base + r * 9 + q];
            aa += x * x;
            bb += y * y;
            ab += x * y;
          }
          if (aa < 1e-28 || bb < 1e-28 || abs(ab) <= 1e-12 * sqrt(aa * bb)) continue;
          const tau: f64 = (bb - aa) / (2 * ab),
            at: f64 = abs(tau);
          let hyp: f64 = sqrt(1 + tau * tau),
            sign: f64 = 1;
          if (at > 1) hyp = at * sqrt(1 + (1 / at) * (1 / at));
          if (tau < 0) sign = -1;
          const t: f64 = sign / (at + hyp),
            c: f64 = 1 / sqrt(1 + t * t),
            s: f64 = c * t;
          for (let r: u32 = 0; r < 9; r++) {
            const x = modelWork[base + r * 9 + p],
              y = modelWork[base + r * 9 + q];
            modelWork[base + r * 9 + p] = c * x - s * y;
            modelWork[base + r * 9 + q] = s * x + c * y;
            const vx = modelWork[base + 81 + r * 9 + p],
              vy = modelWork[base + 81 + r * 9 + q];
            modelWork[base + 81 + r * 9 + p] = c * vx - s * vy;
            modelWork[base + 81 + r * 9 + q] = s * vx + c * vy;
          }
          changed = true;
        }
      if (!changed) {
        modelState[h * 12 + 10] = 1;
        return;
      }
    }
  }
}
@compute([8])
export function finishPairModels(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length || modelState[h * 12] === 0 || modelState[h * 12 + 10] === 0) return;
  const base: u32 = h * 168;
  const mean: vec4f64 = vec4f64(modelWork[base + 162], modelWork[base + 163], modelWork[base + 164], modelWork[base + 165]);
  const sa: f64 = modelWork[base + 166];
  const sb: f64 = modelWork[base + 167];
  let zero: array<f64, 9> = fill<f64, 9>(f64(0));
  let norms: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let j: u32 = 0; j < 9; j++)
    for (let r: u32 = 0; r < 9; r++) norms[j] += modelWork[base + r * 9 + j] * modelWork[base + r * 9 + j];
  let index: u32 = 0;
  for (let j: u32 = 1; j < 9; j++) if (norms[j] < norms[index]) index = j;
  let second: f64 = 1e30,
    largest: f64 = 0;
  for (let j: u32 = 0; j < 9; j++) {
    if (j !== index) second = min(second, norms[j]);
    largest = max(largest, norms[j]);
    zero[j] = modelWork[base + 81 + j * 9 + index];
  }
  const fit: Null9 = { vector: zero, second: sqrt(second), largest: sqrt(largest), ok: 1 };
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

@compute([1])
export function summarizePairModelWork(): void {
  let remaining: u32 = 0;
  for (let h: u32 = 0; h < modelState.length / 12; h++) {
    if (modelState[h * 12] !== 0 && modelState[h * 12 + 10] === 0) remaining++;
  }
  // One scheduling word: adaptive trial budget and whether any model is active.
  modelProgress[0] = (state[0] << 1) | u32(remaining !== 0);
}
