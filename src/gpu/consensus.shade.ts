'use typeshade';
import { CameraModel, unproject, adaptiveTrials } from './numerics.shade.ts';
import { appendRow, null9, basis3, inverse3 } from './projective.shade.ts';

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
}
declare const control: uniform<PairControl>;
declare const pixels: storage<array<f64>>;
declare const coords: storage<array<f64>, 'read_write'>;
declare const invertible: storage<array<u32>, 'read_write'>;
declare const models: storage<array<f64>, 'read_write'>;
declare const valid: storage<array<u32>, 'read_write'>;
declare const partials: storage<array<f64>, 'read_write'>;
// budget, trials, valid models, best count, winning trial, batches, refit active, invertible count.
declare const state: storage<array<u32>, 'read_write'>;
// E/H9, H inverse9, count, sum squared inlier residuals, normalized MSAC cost.
declare const best: storage<array<f64>, 'read_write'>;
declare const mask: storage<array<u32>, 'read_write'>;
let countsTile: workgroup<array<u32, 64>>;
let costsTile: workgroup<array<f64, 64>>;

@compute([64])
export function preparePair(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= pixels.length / 4) return;
  invertible[i] = 1;
  if (control.kind === 1) {
    for (let j: u32 = 0; j < 4; j++) coords[i * 4 + j] = pixels[i * 4 + j];
    return;
  }
  const a = unproject(control.first, vec2f64(pixels[4 * i], pixels[4 * i + 1]));
  const b = unproject(control.second, vec2f64(pixels[4 * i + 2], pixels[4 * i + 3]));
  coords[4 * i] = a.x;
  coords[4 * i + 1] = a.y;
  coords[4 * i + 2] = b.x;
  coords[4 * i + 3] = b.y;
  if (a.z === 0 || b.z === 0 || max(max(abs(a.x), abs(a.y)), max(abs(b.x), abs(b.y))) > 1e4)
    invertible[i] = 0;
}

// Standalone undistortion writes ideal pixels; the host only compacts valid rows.
@compute([64])
export function undistortPair(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= pixels.length / 4) return;
  const a = unproject(control.first, vec2f64(pixels[4 * i], pixels[4 * i + 1]));
  const b = unproject(control.second, vec2f64(pixels[4 * i + 2], pixels[4 * i + 3]));
  invertible[i] = 0;
  for (let j: u32 = 0; j < 4; j++) coords[4 * i + j] = 0;
  if (a.z === 0 || b.z === 0 || max(max(abs(a.x), abs(a.y)), max(abs(b.x), abs(b.y))) > 1e4) return;
  coords[4 * i] = a.x * control.first.fx + control.first.cx;
  coords[4 * i + 1] = a.y * control.first.fy + control.first.cy;
  coords[4 * i + 2] = b.x * control.second.fx + control.second.cx;
  coords[4 * i + 3] = b.y * control.second.fy + control.second.cy;
  invertible[i] = 1;
}
@compute([1])
export function initializePair(): void {
  for (let j: u32 = 0; j < 8; j++) state[j] = 0;
  for (let j: u32 = 0; j < best.length; j++) best[j] = 0;
  for (let i: u32 = 0; i < invertible.length; i++) if (invertible[i] !== 0) state[7]++;
  state[0] = control.maximum;
  if (control.verification !== 0 && state[7] < 8) state[0] = 0;
  state[6] = 1;
  best[19] = 1e30;
  best[20] = f64(f32(invertible.length));
  if (control.verification !== 0) best[20] = f64(f32(state[7]));
}
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
export function generatePairModels(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= valid.length) return;
  valid[h] = 0;
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
  let qr: array<f64, 81> = fill<f64, 81>(f64(0));
  for (let j: u32 = 0; j < limit; j++) {
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
  const fit = null9(qr);
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

function error(model: array<f64, 18>, xy: vec4f64): f64 {
  if (control.kind === 0) {
    const a: f64 = model[0] * xy.x + model[1] * xy.y + model[2];
    const b: f64 = model[3] * xy.x + model[4] * xy.y + model[5];
    const c: f64 = model[6] * xy.x + model[7] * xy.y + model[8];
    const d: f64 = model[0] * xy.z + model[3] * xy.w + model[6];
    const e: f64 = model[1] * xy.z + model[4] * xy.w + model[7];
    const denom: f64 = a * a + b * b + d * d + e * e;
    const residual: f64 = xy.z * a + xy.w * b + c;
    if (!(denom > 1e-12)) return 1e30;
    return (residual * residual) / denom;
  }
  const z: f64 = model[6] * xy.x + model[7] * xy.y + model[8];
  const iz: f64 = model[15] * xy.z + model[16] * xy.w + model[17];
  if (abs(z) < 1e-12 || abs(iz) < 1e-12) return 1e30;
  const a: f64 = (model[0] * xy.x + model[1] * xy.y + model[2]) / z - xy.z;
  const b: f64 = (model[3] * xy.x + model[4] * xy.y + model[5]) / z - xy.w;
  const c: f64 = (model[9] * xy.z + model[10] * xy.w + model[11]) / iz - xy.x;
  const d: f64 = (model[12] * xy.z + model[13] * xy.w + model[14]) / iz - xy.y;
  return (a * a + b * b + c * c + d * d) * 0.5;
}
function scoreLane(h: u32, i: u32): vec2f64 {
  if (i >= invertible.length || valid[h] === 0 || invertible[i] === 0) return vec2f64(f64(0));
  let model: array<f64, 18> = fill<f64, 18>(f64(0));
  for (let j: u32 = 0; j < 18; j++) model[j] = models[h * 18 + j];
  const e: f64 = error(model, getPoint(i));
  if (e <= control.threshold * control.threshold) return vec2f64(f64(1), e);
  return vec2f64(f64(0));
}
@compute([64])
export function scorePairModels(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const h: u32 = wid.x / control.blocks,
    block: u32 = wid.x % control.blocks;
  const v = scoreLane(h, block * 64 + lid.x);
  countsTile[lid.x] = u32(f32(v.x));
  costsTile[lid.x] = v.y;
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) {
      countsTile[lid.x] += countsTile[lid.x + stride];
      costsTile[lid.x] += costsTile[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x === 0) {
    partials[wid.x * 2] = f64(f32(countsTile[0]));
    partials[wid.x * 2 + 1] = costsTile[0];
  }
}
@compute([8])
export function scorePairModelsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= control.trials * control.blocks) return;
  let total: vec2f64 = vec2f64(f64(0));
  for (let lane: u32 = 0; lane < 64; lane++)
    total += scoreLane(i / control.blocks, (i % control.blocks) * 64 + lane);
  partials[i * 2] = total.x;
  partials[i * 2 + 1] = total.y;
}
@compute([1])
export function selectPairModel(): void {
  if (control.refit === 0) {
    if (control.start >= state[0]) return;
    state[1] += min(control.trials, min(state[0] - control.start, control.maximum - control.start));
    state[5]++;
  }
  let count: f64 = best[18],
    cost: f64 = best[19];
  let winner: u32 = 0;
  let changed: bool = false;
  for (let h: u32 = 0; h < control.trials; h++) {
    if (valid[h] === 0) continue;
    if (control.refit === 0) state[2]++;
    let c: f64 = 0,
      e: f64 = 0;
    for (let b: u32 = 0; b < control.blocks; b++) {
      c += partials[(h * control.blocks + b) * 2];
      e += partials[(h * control.blocks + b) * 2 + 1];
    }
    if (c > count || (c === count && e < cost)) {
      winner = h;
      count = c;
      cost = e;
      changed = true;
    }
  }
  if (!changed) {
    if (control.refit !== 0) state[6] = 0;
    return;
  }
  for (let j: u32 = 0; j < 18; j++) best[j] = models[winner * 18 + j];
  best[18] = count;
  best[19] = cost;
  let population: u32 = invertible.length;
  if (control.verification !== 0) population = state[7];
  best[20] = f64(f32(population)) - count + cost / (control.threshold * control.threshold);
  state[3] = u32(f32(count));
  if (control.refit !== 0) return;
  state[4] = control.start + winner;
  let sampleSize: u32 = 8;
  if (control.kind === 1) sampleSize = 4;
  state[0] = min(
    state[0],
    adaptiveTrials(
      u32(f32(count)),
      population,
      sampleSize,
      control.logFailure,
      control.maximum,
      control.minTrials,
    ),
  );
}
@compute([64])
export function classifyPair(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= invertible.length) return;
  mask[i] = 0;
  if (state[3] === 0 || invertible[i] === 0) return;
  let model: array<f64, 18> = fill<f64, 18>(f64(0));
  for (let j: u32 = 0; j < 18; j++) model[j] = best[j];
  if (error(model, getPoint(i)) <= control.threshold * control.threshold) mask[i] = 1;
}
