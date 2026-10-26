'use typeshade';
import { CameraModel, unproject, adaptiveTrials } from './numerics.shade.ts';

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
function getPoint(i: u32): vec4f64 {
  return vec4f64(coords[i * 4], coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]);
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
// The selection walks the batch's trials in order and stops where a sequential RANSAC would
// have stopped: a trial counts only while its index is below the budget the trials before it
// set, and a trial that improves the best model shrinks the budget at once. A batch may score
// more trials than the search uses; those are ignored. The result is therefore the result of
// the sequential search, the same for every batch size and on every device, and the batch is a
// performance choice alone.
@compute([1])
export function selectPairModel(): void {
  if (control.refit === 0) {
    if (control.start >= state[0]) return;
    state[5]++;
  }
  let count: f64 = best[18],
    cost: f64 = best[19];
  let winner: u32 = 0;
  let changed: bool = false;
  let population: u32 = invertible.length;
  if (control.verification !== 0) population = state[7];
  let sampleSize: u32 = 8;
  if (control.kind === 1) sampleSize = 4;
  for (let h: u32 = 0; h < control.trials; h++) {
    if (control.refit === 0) {
      if (control.start + h >= state[0] || control.start + h >= control.maximum) break;
      state[1]++;
    }
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
      if (control.refit === 0)
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
  }
  if (!changed) {
    if (control.refit !== 0) state[6] = 0;
    return;
  }
  for (let j: u32 = 0; j < 18; j++) best[j] = models[winner * 18 + j];
  best[18] = count;
  best[19] = cost;
  best[20] = f64(f32(population)) - count + cost / (control.threshold * control.threshold);
  state[3] = u32(f32(count));
  if (control.refit !== 0) return;
  state[4] = control.start + winner;
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
