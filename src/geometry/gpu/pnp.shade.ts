'use typeshade';
import {
  CameraModel,
  unproject,
  project,
  transform,
  poseRow,
  robustCost,
  solve6,
  movePose,
  adaptiveTrials,
} from './numerics.shade.ts';

class PnpControl {
  camera: CameraModel;
  threshold: f64;
  seed: u32;
  start: u32;
  trials: u32;
  maximum: u32;
  minimum: u32;
  minTrials: u32;
  blocks: u32;
  logFailure: f32;
}
declare const control: uniform<PnpControl>;
declare const xyz: storage<array<f64>>;
declare const pixels: storage<array<f64>>;
// [X,Y,Z,u,v,x_normalized,y_normalized,invertible].
declare const observations: storage<array<f64>, 'read_write'>;
declare const problems: storage<array<f64>, 'read_write'>;
declare const poses: storage<array<f64>>;
declare const valid: storage<array<u32>>;
declare const partials: storage<array<f64>, 'read_write'>;
// [budget,attempts,validCandidates,bestCount,winnerTrial,LM accepts,LM rejects,invalidPixels,
//  batches selected].
declare const state: storage<array<u32>, 'read_write'>;
// [R9,t3,count,squared inlier cost,sum pixel norms,reserved].
declare const best: storage<array<f64>, 'read_write'>;
declare const working: storage<array<f64>, 'read_write'>;
declare const candidate: storage<array<f64>, 'read_write'>;
declare const mask: storage<array<u32>, 'read_write'>;
// [lambda,active,stepValid,oldCost].
declare const lm: storage<array<f64>, 'read_write'>;
// Bounded host-side checkpoints read the active bit without copying the LM state.
declare const activity: storage<array<u32>, 'read_write'>;
declare const normalBlocks: storage<array<f64>, 'read_write'>;
declare const trialCosts: storage<array<f64>, 'read_write'>;
// The scheduling word the host reads every few batches: the adaptive trial budget.
declare const progress: storage<array<u32>, 'read_write'>;
// The workgroup counts of the entries of a search batch that dispatch indirectly (TypeShade
// change 0075), one [x, y, z] triple each: `samplePnP` and `solveP3p` (words 0 to 2), `scorePnP`
// (3 to 5), `scorePnPOracle` (6 to 8) and `selectPnP` (9 to 11). The GPU writes them, so the
// host submits a batch without reading what the batch before it decided. No entry that
// dispatches with these counts binds them.
declare const batchCounts: storage<array<u32>, 'read_write'>;
let countTile: workgroup<array<u32, 64>>;
let costTile: workgroup<array<f64, 64>>;
let normalTile: workgroup<array<f64, 1408>>;

function getPose(source: u32): array<f64, 12> {
  let p: array<f64, 12> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let j: u32 = 0; j < 12; j++) {
    if (source === 0) p[j] = best[j];
    else if (source === 1) p[j] = working[j];
    else p[j] = candidate[j];
  }
  return p;
}
function observationError(p: array<f64, 12>, i: u32): f64 {
  const at: u32 = i * 8;
  const q = project(
    control.camera,
    transform(p, vec3f64(observations[at], observations[at + 1], observations[at + 2])),
  );
  if (q.ok === 0 || observations[at + 7] === 0) return 1e30;
  const du: f64 = q.pixel.x - observations[at + 3],
    dv: f64 = q.pixel.y - observations[at + 4];
  return du * du + dv * dv;
}

@compute([64])
export function normalizePnP(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= xyz.length / 3) return;
  const ray: vec3f64 = unproject(control.camera, vec2f64(pixels[2 * i], pixels[2 * i + 1]));
  for (let j: u32 = 0; j < 3; j++) observations[8 * i + j] = xyz[3 * i + j];
  observations[8 * i + 3] = pixels[2 * i];
  observations[8 * i + 4] = pixels[2 * i + 1];
  observations[8 * i + 5] = ray.x;
  observations[8 * i + 6] = ray.y;
  observations[8 * i + 7] = ray.z;
  if (max(abs(ray.x), abs(ray.y)) > 1e4) observations[8 * i + 7] = 0;
}
@compute([1])
export function initializePnP(): void {
  for (let j: u32 = 0; j < state.length; j++) state[j] = 0;
  for (let j: u32 = 0; j < best.length; j++) best[j] = 0;
  best[13] = 1e30;
  for (let i: u32 = 0; i < observations.length / 8; i++)
    if (observations[i * 8 + 7] === 0) state[7]++;
  if (state[7] === 0) state[0] = control.maximum;
}
function randomWord(word: u32): u32 {
  let s: u32 = word;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
@compute([32])
export function samplePnP(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= problems.length / 18) return;
  for (let j: u32 = 0; j < 18; j++) problems[h * 18 + j] = 0;
  if (h >= control.trials) return;
  const trial: u32 = control.start + h;
  if (trial >= state[0] || trial >= control.maximum) return;
  const n: u32 = observations.length / 8;
  let s: u32 = control.seed ^ ((trial + 1) * 0x9e3779b9);
  if (s === 0) s = 0x85ebca77;
  s = randomWord(s);
  const a: u32 = s % n;
  s = randomWord(s);
  let b: u32 = s % (n - 1);
  if (b >= a) b++;
  s = randomWord(s);
  let c: u32 = s % (n - 2);
  const lo: u32 = min(a, b),
    hi: u32 = max(a, b);
  if (c >= lo) c++;
  if (c >= hi) c++;
  let indices: array<u32, 3> = [a, b, c];
  for (let j: u32 = 0; j < 3; j++) {
    const input: u32 = indices[j] * 8;
    for (let k: u32 = 0; k < 3; k++) problems[h * 18 + j * 3 + k] = observations[input + k];
    problems[h * 18 + 9 + j * 3] = observations[input + 5];
    problems[h * 18 + 10 + j * 3] = observations[input + 6];
    problems[h * 18 + 11 + j * 3] = 1;
  }
}
function scoreLane(model: u32, i: u32): vec2f64 {
  if (i >= observations.length / 8 || valid[model] === 0) return vec2f64(f64(0));
  let p: array<f64, 12> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let j: u32 = 0; j < 12; j++) p[j] = poses[model * 12 + j];
  const error: f64 = observationError(p, i);
  if (error <= control.threshold * control.threshold) return vec2f64(f64(1), error);
  return vec2f64(f64(0));
}
@compute([64])
export function scorePnP(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const model: u32 = wid.x / control.blocks,
    block: u32 = wid.x % control.blocks;
  const value = scoreLane(model, block * 64 + lid.x);
  countTile[lid.x] = u32(f32(value.x));
  costTile[lid.x] = value.y;
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) {
      countTile[lid.x] += countTile[lid.x + stride];
      costTile[lid.x] += costTile[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x === 0) {
    partials[wid.x * 2] = f64(f32(countTile[0]));
    partials[wid.x * 2 + 1] = costTile[0];
  }
}
// Explicit CPU oracle tier: identical lane operator, serial fold instead of GPU barriers.
@compute([32])
export function scorePnPOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const index: u32 = gid.x;
  if (index >= partials.length / 2) return;
  const model: u32 = index / control.blocks,
    block: u32 = index % control.blocks;
  let count: f64 = 0,
    cost: f64 = 0;
  for (let lane: u32 = 0; lane < 64; lane++) {
    const v = scoreLane(model, block * 64 + lane);
    count += v.x;
    cost += v.y;
  }
  partials[index * 2] = count;
  partials[index * 2 + 1] = cost;
}
// The budget a best consensus of `count` inliers sets: the trials a sequential search runs.
function trialBudget(count: f64): u32 {
  return adaptiveTrials(u32(f32(count)), observations.length / 8, 3, control.logFailure,
    control.maximum, control.minTrials);
}
// The selection walks the batch's trials in order and stops where a sequential RANSAC would
// have stopped: a trial counts only while its index is below the budget the trials before it
// set, and a trial whose candidates improve the best pose shrinks the budget as soon as its four
// candidates are compared. A batch may score more trials than the search uses; those are
// ignored. The result is therefore the result of the sequential search (a batch of one trial),
// the same for every batch size and on every device, and the batch is a performance choice
// alone (as `selectPairModel` of `consensus.shade.ts`).
@compute([1])
export function selectPnP(): void {
  if (control.start >= state[0]) return;
  state[8]++;
  let count: f64 = best[12],
    cost: f64 = best[13];
  let winner: u32 = 0;
  let changed: bool = false;
  let improved: bool = false;
  for (let model: u32 = 0; model < control.trials * 4; model++) {
    if (model % 4 === 0) {
      if (improved) state[0] = min(state[0], trialBudget(count));
      improved = false;
      const trial: u32 = control.start + model / 4;
      if (trial >= state[0] || trial >= control.maximum) break;
      state[1]++;
    }
    if (valid[model] === 0) continue;
    state[2]++;
    let c: f64 = 0,
      v: f64 = 0;
    for (let block: u32 = 0; block < control.blocks; block++) {
      const at = (model * control.blocks + block) * 2;
      c += partials[at];
      v += partials[at + 1];
    }
    if (c > count || (c === count && v < cost)) {
      count = c;
      cost = v;
      winner = model;
      changed = true;
      improved = true;
    }
  }
  if (improved) state[0] = min(state[0], trialBudget(count));
  if (changed) {
    for (let j: u32 = 0; j < 12; j++) best[j] = poses[winner * 12 + j];
    best[12] = count;
    best[13] = cost;
    state[3] = u32(f32(count));
    state[4] = control.start + winner / 4;
  }
}

// Whether the batch of `control` runs: when its first trial is below the adaptive budget
// (`state[0]`) and the requested maximum. The first batch of a search always runs, as it did when
// the host decided: its scoring is the chunk that teaches the device the rate of a trial. A batch
// that does not run changes nothing: its sampling, solve, scoring and selection dispatch zero
// workgroups.
function batchRuns(): bool {
  if (control.start === 0) return true;
  return control.start < state[0] && control.start < control.maximum;
}
function writeCount(offset: u32, x: u32): void {
  batchCounts[offset] = x;
  batchCounts[offset + 1] = 1;
  batchCounts[offset + 2] = 1;
}
// The workgroups of a search batch, written after the selection of the batch before, which is
// what shrinks the budget: one per 32 trials for the sampling and the solve, one per score block
// of each of the four candidates of a trial for `scorePnP`, one per 32 of those for its oracle,
// and one for the selection; all zero when the batch does not run.
@compute([1])
export function planPnPBatch(): void {
  let trials: u32 = 0;
  if (batchRuns()) trials = control.trials;
  const lanes: u32 = trials * 4 * control.blocks;
  writeCount(0, (trials + 31) / 32);
  writeCount(3, lanes);
  writeCount(6, (lanes + 31) / 32);
  writeCount(9, min(trials, 1));
}
// The scheduling word: the adaptive trial budget, read by the host every few batches to leave
// the search early.
@compute([1])
export function snapshotPnPBudget(): void {
  progress[0] = state[0];
}

@compute([64])
export function consensusMask(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length) return;
  mask[i] = 0;
  if (
    state[3] >= control.minimum &&
    observationError(getPose(0), i) <= control.threshold * control.threshold
  )
    mask[i] = 1;
}
@compute([1])
export function beginPoseRefinement(): void {
  for (let j: u32 = 0; j < 12; j++) working[j] = best[j];
  lm[0] = 1e-3;
  lm[1] = 0;
  lm[2] = 0;
  lm[3] = 0;
  if (state[3] >= control.minimum) lm[1] = 1;
}
function normalLane(i: u32, p: array<f64, 12>): array<f64, 44> {
  let out: array<f64, 44> = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ];
  if (i >= mask.length || mask[i] === 0 || lm[1] === 0) return out;
  const at = i * 8;
  const q = transform(p, vec3f64(observations[at], observations[at + 1], observations[at + 2]));
  const projection = project(control.camera, q);
  if (projection.ok === 0) {
    out[43] = 1;
    return out;
  }
  const ru: f64 = projection.pixel.x - observations[at + 3],
    rv: f64 = projection.pixel.y - observations[at + 4];
  const error: f64 = sqrt(ru * ru + rv * rv),
    huber: f64 = control.threshold * 0.5;
  let weight: f64 = 1;
  if (error > huber) weight = huber / error;
  const ju = poseRow(q, projection.du),
    jv = poseRow(q, projection.dv);
  for (let k: u32 = 0; k < 6; k++) {
    out[36 + k] = -weight * (ju[k] * ru + jv[k] * rv);
    for (let l: u32 = 0; l < 6; l++) out[k * 6 + l] = weight * (ju[k] * ju[l] + jv[k] * jv[l]);
  }
  out[42] = robustCost(error, huber);
  return out;
}
@compute([32])
export function poseNormals(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const values = normalLane(wid.x * 32 + lid.x, getPose(1));
  for (let j: u32 = 0; j < 44; j++) normalTile[lid.x * 44 + j] = values[j];
  workgroupBarrier();
  for (let stride: u32 = 16; stride > 0; stride /= 2) {
    if (lid.x < stride)
      for (let j: u32 = 0; j < 44; j++)
        normalTile[lid.x * 44 + j] += normalTile[(lid.x + stride) * 44 + j];
    workgroupBarrier();
  }
  if (lid.x === 0) for (let j: u32 = 0; j < 44; j++) normalBlocks[wid.x * 44 + j] = normalTile[j];
}
@compute([32])
export function poseNormalsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= normalBlocks.length / 44) return;
  for (let j: u32 = 0; j < 44; j++) normalBlocks[block * 44 + j] = 0;
  const p = getPose(1);
  for (let lane: u32 = 0; lane < 32; lane++) {
    const v = normalLane(block * 32 + lane, p);
    for (let j: u32 = 0; j < 44; j++) normalBlocks[block * 44 + j] += v[j];
  }
}
@compute([1])
export function poseStep(): void {
  lm[2] = 0;
  if (lm[1] === 0) return;
  let A: array<f64, 36> = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
  ];
  let b: array<f64, 6> = [0, 0, 0, 0, 0, 0];
  let cost: f64 = 0,
    invalid: f64 = 0;
  for (let block: u32 = 0; block < normalBlocks.length / 44; block++) {
    for (let j: u32 = 0; j < 36; j++) A[j] += normalBlocks[block * 44 + j];
    for (let j: u32 = 0; j < 6; j++) b[j] += normalBlocks[block * 44 + 36 + j];
    cost += normalBlocks[block * 44 + 42];
    invalid += normalBlocks[block * 44 + 43];
  }
  if (invalid > 0) {
    lm[1] = 0;
    return;
  }
  lm[3] = cost;
  for (let j: u32 = 0; j < 6; j++) A[j * 6 + j] += lm[0] * max(A[j * 6 + j], f64(1e-10));
  const step = solve6(A, b);
  if (step.ok === 0) return;
  const moved = movePose(getPose(1), step.values);
  for (let j: u32 = 0; j < 12; j++) {
    if (!(abs(moved[j]) < 1e12)) return;
    candidate[j] = moved[j];
  }
  lm[2] = 1;
}
@compute([32])
export function poseTrialCosts(@builtin('global_invocation_id') gid: vec3u): void {
  const block = gid.x;
  if (block >= trialCosts.length) return;
  let cost: f64 = 0;
  if (lm[1] !== 0 && lm[2] !== 0) {
    const p = getPose(2);
    for (let k: u32 = 0; k < 64; k++) {
      const i = block * 64 + k;
      if (i < mask.length && mask[i] !== 0) {
        const error = observationError(p, i);
        if (error >= 1e30) {
          cost = 1e30;
          break;
        }
        cost += robustCost(sqrt(error), control.threshold * 0.5);
      }
    }
  }
  trialCosts[block] = cost;
}
@compute([1])
export function acceptPoseStep(): void {
  if (lm[1] === 0) return;
  let cost: f64 = 0;
  for (let j: u32 = 0; j < trialCosts.length; j++) cost += trialCosts[j];
  if (lm[2] !== 0 && cost < lm[3]) {
    for (let j: u32 = 0; j < 12; j++) working[j] = candidate[j];
    lm[0] = max(f64(1e-10), lm[0] / 3);
    state[5]++;
    if (lm[3] - cost < 1e-10 * max(f64(1), cost)) lm[1] = 0;
  } else {
    lm[0] *= 10;
    state[6]++;
    if (lm[0] > 1e12) lm[1] = 0;
  }
}
@compute([1])
export function snapshotPoseActivity(): void {
  activity[0] = u32(f32(lm[1]));
}
@compute([1])
export function finishPoseRefinement(): void {
  if (state[3] < control.minimum) return;
  const p = getPose(1);
  let count: u32 = 0;
  let cost: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) {
    const e = observationError(p, i);
    if (e <= control.threshold * control.threshold) {
      count++;
      cost += e;
    }
  }
  if (f64(f32(count)) > best[12] || (f64(f32(count)) === best[12] && cost < best[13])) {
    for (let j: u32 = 0; j < 12; j++) best[j] = working[j];
    best[12] = f64(f32(count));
    best[13] = cost;
    state[3] = count;
  }
}
@compute([1])
export function summarizePnP(): void {
  const p = getPose(0);
  let sum: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) if (mask[i] !== 0) sum += sqrt(observationError(p, i));
  best[14] = sum;
}
