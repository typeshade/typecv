'use typeshade';
import {
  CameraModel,
  project,
  transform,
  poseRow,
  robustCost,
  solve3,
  solve6,
  movePose,
} from './numerics.shade.ts';

class BundleControl {
  cameras: u32;
  points: u32;
  observations: u32;
  anchor: u32;
  scaleView: u32;
  normalizeScale: u32;
  huber: f64;
  relativeTolerance: f64;
  // The window of a streamed run (`docs/gpu-budget.md`, section 3.4). A resident run has one window
  // that holds every observation and point: obsBase = pointBase = 0, obsEnd = observations,
  // pointEnd = points, obsCapacity = observations, pointCapacity = points.
  // First and last observation (end exclusive) and point of the window in the global numbering.
  obsBase: u32;
  obsEnd: u32;
  pointBase: u32;
  pointEnd: u32;
  // The observations and points the `blocks` window holds: sets the offsets of its sections.
  obsCapacity: u32;
  pointCapacity: u32;
  // 1 in the first window of a sweep, 1 in its last window.
  first: u32;
  last: u32;
  // 0 assembly sweep, 1 PCG sweep (skipped once PCG ended), 2 update sweep (skipped on failure).
  sweep: u32;
  // 1 when the problem streams: the point check reads `state[26]`, not the resident blocks.
  streamed: u32;
  // The first element of a dispatch that the host splits at the dispatch limit.
  base: u32;
}
declare const control: uniform<BundleControl>;
// Current poses C*12, XYZ P*3, camera parameters C*6, observed pixels M*2.
declare const scene: storage<array<f64>, 'read_write'>;
declare const trial: storage<array<f64>, 'read_write'>;
// Observation camera M, observation point M, point offsets P+1, camera offsets C+1,
// camera-major observation permutation M, variable camera flags C.
declare const index: storage<array<u32>>;
// The window of observations and points (`obsCapacity`, `pointCapacity`):
// J/residual/weight/metrics 24 per observation, cross blocks 18 per observation, inverse point
// block/RHS/valid 13 per point. Then, for all cameras:
// damped camera blocks C*36, inverse Schur-diagonal preconditioner C*36, reduced RHS C*6.
declare const blocks: storage<array<f64>, 'read_write'>;
// One damped six-by-six Schur camera block per camera; rows are assembled independently.
declare const cameraScratch: storage<array<f64>, 'read_write'>;
// PCG x/r/p/Sp/z, each C*6; point gather scratch P*3.
declare const vectors: storage<array<f64>, 'read_write'>;
// LM [lambda, active, baseline, origin.xyz, current cost/SSE, initial cost/SSE,
//     accepted, rejected, commit, gauge scale, input error, trial valid, CG steps, failed solves].
// PCG [rho, active, failed, initial residual^2, residual^2, steps, converged, reserved].
declare const state: storage<array<f64>, 'read_write'>;
// Bounded host-side checkpoints read only these activity flags, not the full BA state.
declare const activity: storage<array<u32>, 'read_write'>;
// Per block of 64 observations: cost, squared error and invalid count (metrics), or cost and squares (candidate).
declare const partials: storage<array<f64>, 'read_write'>;
let costTile: workgroup<array<f64, 64>>;
let squareTile: workgroup<array<f64, 64>>;
let invalidTile: workgroup<array<u32, 64>>;
let schurTile: workgroup<array<f64, 64>>;
declare const output: storage<array<f64>, 'read_write'>;

function poseAt(camera: u32, candidate: bool): array<f64, 12> {
  let p: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) {
    if (candidate) p[j] = trial[camera * 12 + j];
    else p[j] = scene[camera * 12 + j];
  }
  return p;
}
function pointAt(point: u32, candidate: bool): vec3f64 {
  const at = control.cameras * 12 + point * 3;
  if (candidate) return vec3f64(trial[at], trial[at + 1], trial[at + 2]);
  return vec3f64(scene[at], scene[at + 1], scene[at + 2]);
}
function cameraAt(camera: u32): CameraModel {
  const at = control.cameras * 12 + control.points * 3 + camera * 6;
  return {
    fx: scene[at],
    fy: scene[at + 1],
    cx: scene[at + 2],
    cy: scene[at + 3],
    k1: scene[at + 4],
    k2: scene[at + 5],
  };
}
function cameraCenter(camera: u32, candidate: bool): vec3f64 {
  const p = poseAt(camera, candidate);
  return vec3f64(
    -p[0] * p[9] - p[3] * p[10] - p[6] * p[11],
    -p[1] * p[9] - p[4] * p[10] - p[7] * p[11],
    -p[2] * p[9] - p[5] * p[10] - p[8] * p[11],
  );
}
function variable(camera: u32): bool {
  return index[3 * control.observations + control.points + control.cameras + 2 + camera] !== 0;
}
function pointStart(p: u32): u32 {
  return index[2 * control.observations + p];
}
function cameraStart(c: u32): u32 {
  return index[2 * control.observations + control.points + 1 + c];
}
function cameraObservation(i: u32): u32 {
  return index[2 * control.observations + control.points + control.cameras + 2 + i];
}
// Offsets in `blocks`: `i` and `p` are global, the window holds [obsBase, obsEnd) and
// [pointBase, pointEnd).
function obsRow(i: u32): u32 {
  return (i - control.obsBase) * 24;
}
function crossAt(i: u32): u32 {
  return control.obsCapacity * 24 + (i - control.obsBase) * 18;
}
function pointBlock(p: u32): u32 {
  return control.obsCapacity * 42 + (p - control.pointBase) * 13;
}
function cameraBlock(c: u32): u32 {
  return control.obsCapacity * 42 + control.pointCapacity * 13 + c * 36;
}
function inverseBlock(c: u32): u32 {
  return control.obsCapacity * 42 + control.pointCapacity * 13 + control.cameras * 36 + c * 36;
}
function cameraRhs(c: u32): u32 {
  return control.obsCapacity * 42 + control.pointCapacity * 13 + control.cameras * 72 + c * 6;
}
// The first entry of camera c's observation list at or after the window's first observation. The
// list is ascending, so a window reads one contiguous slice of it, in the order of a resident run.
function cameraFirst(c: u32): u32 {
  let lo: u32 = cameraStart(c),
    hi: u32 = cameraStart(c + 1);
  if (control.obsBase === 0) return lo;
  for (let step: u32 = 0; step < 40; step++) {
    if (lo >= hi) break;
    const mid: u32 = lo + (hi - lo) / 2;
    if (cameraObservation(mid) < control.obsBase) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
// True when the sweep has nothing to do: PCG ended during a PCG sweep, or the solve failed during an
// update sweep.
function sweepIdle(): bool {
  if (state[1] === 0) return true;
  if (control.sweep === 1 && state[19] === 0) return true;
  if (control.sweep === 2 && state[20] !== 0) return true;
  return false;
}
function bundleSchurPointAt(p: u32): void {
  if (p >= control.pointEnd || state[19] === 0) return;
  const m = control.cameras * 6,
    at = pointBlock(p),
    out = m * 5 + p * 3;
  let rhs: array<f64, 3> = [0, 0, 0];
  for (let i: u32 = pointStart(p); i < pointStart(p + 1); i++) {
    const c = index[i],
      cross = crossAt(i);
    for (let j: u32 = 0; j < 3; j++)
      for (let k: u32 = 0; k < 6; k++)
        rhs[j] += blocks[cross + k * 3 + j] * vectors[m * 2 + c * 6 + k];
  }
  for (let j: u32 = 0; j < 3; j++) {
    let v: f64 = 0;
    for (let k: u32 = 0; k < 3; k++) v += blocks[at + j * 3 + k] * rhs[k];
    vectors[out + j] = v;
  }
}
function bundleSchurCameraRow(rowId: u32): void {
  const c = rowId / 6,
    j = rowId % 6;
  if (c >= control.cameras || state[19] === 0) return;
  const m = control.cameras * 6,
    A = cameraBlock(c);
  // The first window starts the row; a later window continues the sum where the last one stopped.
  let v: f64 = 0;
  if (control.first !== 0) {
    for (let k: u32 = 0; k < 6; k++) v += blocks[A + j * 6 + k] * vectors[m * 2 + c * 6 + k];
  } else {
    v = vectors[m * 3 + rowId];
  }
  for (let n: u32 = cameraFirst(c); n < cameraStart(c + 1); n++) {
    const i = cameraObservation(n);
    if (i >= control.obsEnd) break;
    const p = index[control.observations + i],
      cross = crossAt(i);
    for (let k: u32 = 0; k < 3; k++) v -= blocks[cross + j * 3 + k] * vectors[m * 5 + p * 3 + k];
  }
  vectors[m * 3 + rowId] = v;
}
function bundlePreconditionedResidualAt(i: u32, m: u32): f64 {
  const c = i / 6,
    j = i % 6;
  let z: f64 = 0;
  for (let k: u32 = 0; k < 6; k++)
    z += blocks[inverseBlock(c) + j * 6 + k] * vectors[m + c * 6 + k];
  return z;
}

@compute([1])
export function initializeBundle(): void {
  for (let j: u32 = 0; j < 26; j++) state[j] = 0;
  state[0] = 1e-3;
  state[1] = 1;
  const origin = cameraCenter(control.anchor, false),
    baseline = length(origin - cameraCenter(control.scaleView, false));
  state[2] = baseline;
  state[3] = origin.x;
  state[4] = origin.y;
  state[5] = origin.z;
  if (!(baseline > 1e-10)) {
    state[14] = 1;
    state[1] = 0;
  }
}
@compute([64])
export function linearizeBundle(@builtin('global_invocation_id') gid: vec3u): void {
  const i = control.obsBase + gid.x;
  if (i >= control.obsEnd || sweepIdle()) return;
  const c = index[i],
    p = index[control.observations + i],
    pose = poseAt(c, false),
    X = pointAt(p, false),
    q = transform(pose, X);
  const projection = project(cameraAt(c), q),
    base = obsRow(i);
  for (let j: u32 = 0; j < 24; j++) blocks[base + j] = 0;
  if (projection.ok === 0) {
    blocks[base + 23] = 1;
    return;
  }
  const pixelAt = control.cameras * 18 + control.points * 3 + i * 2;
  const ru: f64 = projection.pixel.x - scene[pixelAt],
    rv: f64 = projection.pixel.y - scene[pixelAt + 1],
    squared: f64 = ru * ru + rv * rv,
    error: f64 = sqrt(squared);
  let weight: f64 = 1;
  if (error > control.huber) weight = control.huber / error;
  const ju = poseRow(q, projection.du),
    jv = poseRow(q, projection.dv);
  for (let k: u32 = 0; k < 6; k++) {
    blocks[base + k] = ju[k];
    blocks[base + 6 + k] = jv[k];
  }
  for (let k: u32 = 0; k < 3; k++) {
    blocks[base + 12 + k] =
      projection.du.x * pose[k] + projection.du.y * pose[3 + k] + projection.du.z * pose[6 + k];
    blocks[base + 15 + k] =
      projection.dv.x * pose[k] + projection.dv.y * pose[3 + k] + projection.dv.z * pose[6 + k];
  }
  blocks[base + 18] = ru;
  blocks[base + 19] = rv;
  blocks[base + 20] = weight;
  blocks[base + 21] = robustCost(error, control.huber);
  blocks[base + 22] = squared;
}
// The linearized cost, squared error and invalid flag of observation `i`, from the scene: the
// values `linearizeBundle` writes (the same expressions), so that the cost sweep needs no window
// of `blocks`.
function metricLane(i: u32): vec3f64 {
  if (i >= control.obsEnd) return vec3f64(f64(0));
  const c = index[i],
    p = index[control.observations + i],
    q = transform(poseAt(c, false), pointAt(p, false));
  const projection = project(cameraAt(c), q);
  if (projection.ok === 0) return vec3f64(f64(0), f64(0), f64(1));
  const pixelAt = control.cameras * 18 + control.points * 3 + i * 2;
  const ru: f64 = projection.pixel.x - scene[pixelAt],
    rv: f64 = projection.pixel.y - scene[pixelAt + 1],
    squared: f64 = ru * ru + rv * rv,
    error: f64 = sqrt(squared);
  return vec3f64(robustCost(error, control.huber), squared, f64(0));
}
function observationBlocks(): u32 {
  return (control.observations + 63) / 64;
}
/** Cooperative partial sums of the linearized cost, squared error and invalid projections. */
@compute([64])
export function bundleMetricPartials(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const v = metricLane(control.obsBase + wid.x * 64 + lid.x);
  costTile[lid.x] = v.x;
  squareTile[lid.x] = v.y;
  invalidTile[lid.x] = u32(f32(v.z));
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) {
      costTile[lid.x] += costTile[lid.x + stride];
      squareTile[lid.x] += squareTile[lid.x + stride];
      invalidTile[lid.x] += invalidTile[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x === 0) {
    // A window starts at a multiple of 64 observations, so its blocks are those of a resident run.
    const block = control.obsBase / 64 + wid.x;
    partials[block * 3] = costTile[0];
    partials[block * 3 + 1] = squareTile[0];
    partials[block * 3 + 2] = f64(f32(invalidTile[0]));
  }
}
/** CPU differential tier: one invocation sums one block in observation order; no barriers. */
@compute([8])
export function bundleMetricPartialsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  if (gid.x >= (control.obsEnd - control.obsBase + 63) / 64) return;
  const block: u32 = control.obsBase / 64 + gid.x;
  let total: vec3f64 = vec3f64(f64(0));
  for (let lane: u32 = 0; lane < 64; lane++) total += metricLane(block * 64 + lane);
  partials[block * 3] = total.x;
  partials[block * 3 + 1] = total.y;
  partials[block * 3 + 2] = total.z;
}
@compute([1])
export function finishBundleMetrics(): void {
  if (state[14] !== 0) return;
  let cost: f64 = 0,
    squares: f64 = 0,
    invalid: f64 = 0;
  for (let b: u32 = 0; b < observationBlocks(); b++) {
    cost += partials[b * 3];
    squares += partials[b * 3 + 1];
    invalid += partials[b * 3 + 2];
  }
  if (invalid > 0) {
    state[14] = 2;
    state[1] = 0;
    return;
  }
  state[6] = cost;
  state[7] = squares;
  state[8] = cost;
  state[9] = squares;
  if (cost < 1e-20) state[1] = 0;
}
@compute([64])
export function bundlePointBlocks(@builtin('global_invocation_id') gid: vec3u): void {
  const p = control.pointBase + gid.x;
  if (p >= control.pointEnd || sweepIdle()) return;
  let V: array<f64, 9> = fill<f64, 9>(f64(0));
  let rhs: array<f64, 3> = fill<f64, 3>(f64(0));
  const at = pointBlock(p);
  for (let j: u32 = 0; j < 13; j++) blocks[at + j] = 0;
  for (let i: u32 = pointStart(p); i < pointStart(p + 1); i++) {
    const base = obsRow(i),
      w = blocks[base + 20],
      cross = crossAt(i),
      active = variable(index[i]);
    if (blocks[base + 23] !== 0) return;
    for (let j: u32 = 0; j < 18; j++) blocks[cross + j] = 0;
    for (let row: u32 = 0; row < 2; row++)
      for (let j: u32 = 0; j < 3; j++) {
        const g = blocks[base + 12 + row * 3 + j];
        rhs[j] -= w * g * blocks[base + 18 + row];
        for (let k: u32 = 0; k < 3; k++) V[j * 3 + k] += w * g * blocks[base + 12 + row * 3 + k];
        if (active)
          for (let k: u32 = 0; k < 6; k++)
            blocks[cross + k * 3 + j] += w * blocks[base + row * 6 + k] * g;
      }
  }
  for (let j: u32 = 0; j < 3; j++) V[j * 3 + j] += state[0] * max(V[j * 3 + j], f64(1e-10));
  for (let column: u32 = 0; column < 3; column++) {
    let b: array<f64, 3> = [0, 0, 0];
    b[column] = 1;
    const solution = solve3(V, vec3f64(b[0], b[1], b[2]));
    if (solution.ok === 0) return;
    blocks[at + column] = solution.values.x;
    blocks[at + 3 + column] = solution.values.y;
    blocks[at + 6 + column] = solution.values.z;
    blocks[at + 9 + column] = rhs[column];
  }
  // Symmetrize the inverse before it enters the matrix-free SPD operator.
  for (let j: u32 = 0; j < 3; j++)
    for (let k: u32 = j + 1; k < 3; k++) {
      const v: f64 = (blocks[at + j * 3 + k] + blocks[at + k * 3 + j]) * 0.5;
      blocks[at + j * 3 + k] = v;
      blocks[at + k * 3 + j] = v;
    }
  blocks[at + 12] = 1;
}
@compute([64])
export function bundleCameraBlocks(@builtin('global_invocation_id') gid: vec3u): void {
  const c = gid.x;
  if (c >= control.cameras || state[1] === 0) return;
  const out = cameraBlock(c),
    inv = inverseBlock(c),
    bAt = cameraRhs(c);
  let A: array<f64, 36> = fill<f64, 36>(f64(0));
  let S: array<f64, 36> = fill<f64, 36>(f64(0));
  let rhs: array<f64, 6> = fill<f64, 6>(f64(0));
  for (let j: u32 = 0; j < 36; j++) {
    blocks[out + j] = 0;
    blocks[inv + j] = 0;
  }
  for (let j: u32 = 0; j < 6; j++) blocks[bAt + j] = 0;
  if (!variable(c)) {
    for (let j: u32 = 0; j < 6; j++) {
      blocks[out + j * 6 + j] = 1;
      blocks[inv + j * 6 + j] = 1;
    }
    return;
  }
  for (let k: u32 = cameraStart(c); k < cameraStart(c + 1); k++) {
    const i = cameraObservation(k),
      p = index[control.observations + i],
      point = pointBlock(p),
      base = obsRow(i),
      cross = crossAt(i),
      w = blocks[base + 20];
    let T: array<f64, 18> = fill<f64, 18>(f64(0));
    for (let j: u32 = 0; j < 6; j++) {
      for (let row: u32 = 0; row < 2; row++) {
        rhs[j] -= w * blocks[base + row * 6 + j] * blocks[base + 18 + row];
        for (let l: u32 = 0; l < 6; l++)
          A[j * 6 + l] += w * blocks[base + row * 6 + j] * blocks[base + row * 6 + l];
      }
      for (let l: u32 = 0; l < 3; l++)
        for (let r: u32 = 0; r < 3; r++)
          T[j * 3 + l] += blocks[cross + j * 3 + r] * blocks[point + r * 3 + l];
      for (let l: u32 = 0; l < 3; l++) rhs[j] -= T[j * 3 + l] * blocks[point + 9 + l];
      for (let l: u32 = 0; l < 6; l++)
        for (let r: u32 = 0; r < 3; r++) S[j * 6 + l] -= T[j * 3 + r] * blocks[cross + l * 3 + r];
    }
  }
  for (let j: u32 = 0; j < 6; j++) A[j * 6 + j] += state[0] * max(A[j * 6 + j], f64(1e-10));
  for (let j: u32 = 0; j < 36; j++) {
    blocks[out + j] = A[j];
    S[j] += A[j];
  }
  for (let j: u32 = 0; j < 6; j++) blocks[bAt + j] = rhs[j];
  for (let column: u32 = 0; column < 6; column++) {
    let b: array<f64, 6> = [0, 0, 0, 0, 0, 0];
    b[column] = 1;
    const solution = solve6(S, b);
    if (solution.ok === 0) {
      blocks[inv] = -1;
      return;
    }
    for (let j: u32 = 0; j < 6; j++) blocks[inv + j * 6 + column] = solution.values[j];
  }
  for (let j: u32 = 0; j < 6; j++)
    for (let k: u32 = j + 1; k < 6; k++) {
      const v: f64 = (blocks[inv + j * 6 + k] + blocks[inv + k * 6 + j]) * 0.5;
      blocks[inv + j * 6 + k] = v;
      blocks[inv + k * 6 + j] = v;
    }
}
// GPU path: one invocation assembles a single six-wide camera row. Keeping each row's
// observation traversal and accumulation order identical to the scalar oracle makes
// the parallelization independent across rows without changing its arithmetic.
// A streamed run assembles the row in windows (`docs/gpu-budget.md`, section 3.4): the first window
// starts the sums, each later window continues them from the accumulators in `blocks` (A row),
// `cameraScratch` (S row) and the reduced RHS, and the last window adds the damping. A resident run
// is one window that is first and last.
@compute([64])
export function bundleCameraRows(
  @builtin('global_invocation_id') gid: vec3u,
): void {
  const rowId = gid.x,
    c = rowId / 6,
    j = rowId % 6;
  if (c >= control.cameras || state[1] === 0) return;
  const out = cameraBlock(c),
    schur = c * 36,
    bAt = cameraRhs(c);
  let aRow: array<f64, 6> = fill<f64, 6>(f64(0));
  let sRow: array<f64, 6> = fill<f64, 6>(f64(0));
  let rhs: f64 = 0;
  if (!variable(c)) {
    aRow[j] = 1;
    sRow[j] = 1;
  } else {
    if (control.first === 0) {
      for (let l: u32 = 0; l < 6; l++) {
        aRow[l] = blocks[out + j * 6 + l];
        sRow[l] = cameraScratch[schur + j * 6 + l];
      }
      rhs = blocks[bAt + j];
    }
    for (let k: u32 = cameraFirst(c); k < cameraStart(c + 1); k++) {
      const i = cameraObservation(k);
      if (i >= control.obsEnd) break;
      const p = index[control.observations + i],
        point = pointBlock(p),
        base = obsRow(i),
        cross = crossAt(i),
        w = blocks[base + 20];
      let T: array<f64, 3> = fill<f64, 3>(f64(0));
      for (let observationRow: u32 = 0; observationRow < 2; observationRow++) {
        const value = blocks[base + observationRow * 6 + j];
        rhs -= w * value * blocks[base + 18 + observationRow];
        for (let l: u32 = 0; l < 6; l++)
          aRow[l] += w * value * blocks[base + observationRow * 6 + l];
      }
      for (let l: u32 = 0; l < 3; l++)
        for (let r: u32 = 0; r < 3; r++)
          T[l] += blocks[cross + j * 3 + r] * blocks[point + r * 3 + l];
      for (let l: u32 = 0; l < 3; l++) rhs -= T[l] * blocks[point + 9 + l];
      for (let l: u32 = 0; l < 6; l++)
        for (let r: u32 = 0; r < 3; r++)
          sRow[l] -= T[r] * blocks[cross + l * 3 + r];
    }
    if (control.last !== 0) {
      aRow[j] += state[0] * max(aRow[j], f64(1e-10));
      for (let l: u32 = 0; l < 6; l++) sRow[l] += aRow[l];
    }
  }
  for (let l: u32 = 0; l < 6; l++) {
    blocks[out + j * 6 + l] = aRow[l];
    cameraScratch[schur + j * 6 + l] = sRow[l];
  }
  blocks[bAt + j] = rhs;
}
// Marks a point whose block failed in this window as a failed solve (`state[20]`): a streamed run
// cannot read the blocks of earlier windows. The first window of the sweep clears the flag, and
// `initializeBundleCg` keeps it.
@compute([1])
export function checkBundlePoints(): void {
  if (control.first !== 0) state[20] = 0;
  if (state[1] === 0) return;
  for (let p: u32 = control.pointBase; p < control.pointEnd; p++)
    if (blocks[pointBlock(p) + 12] === 0) state[20] = 1;
}
@compute([64])
export function invertBundleCameraColumns(
  @builtin('global_invocation_id') gid: vec3u,
): void {
  const columnId = gid.x,
    c = columnId / 6,
    column = columnId % 6;
  if (c >= control.cameras || state[1] === 0) return;
  const inv = inverseBlock(c),
    at = c * 36;
  let basis: array<f64, 6> = [0, 0, 0, 0, 0, 0];
  basis[column] = 1;
  let S: array<f64, 36> = fill<f64, 36>(f64(0));
  for (let i: u32 = 0; i < 36; i++) S[i] = cameraScratch[at + i];
  const solution = solve6(S, basis);
  if (solution.ok === 0) {
    // Each invocation owns its diagonal marker; finalization folds these markers.
    blocks[inv + column * 6 + column] = -1;
    return;
  }
  for (let row: u32 = 0; row < 6; row++)
    blocks[inv + row * 6 + column] = solution.values[row];
}
@compute([64])
export function finalizeBundleCameraInverse(@builtin('global_invocation_id') gid: vec3u): void {
  const c = gid.x;
  if (c >= control.cameras || state[1] === 0) return;
  const inv = inverseBlock(c);
  for (let j: u32 = 0; j < 6; j++)
    if (blocks[inv + j * 6 + j] < 0) {
      blocks[inv] = -1;
      return;
    }
  for (let j: u32 = 0; j < 6; j++)
    for (let k: u32 = j + 1; k < 6; k++) {
      const v: f64 = (blocks[inv + j * 6 + k] + blocks[inv + k * 6 + j]) * 0.5;
      blocks[inv + j * 6 + k] = v;
      blocks[inv + k * 6 + j] = v;
    }
}
@compute([1])
export function initializeBundleCg(): void {
  const failed: f64 = state[20];
  for (let j: u32 = 18; j < 26; j++) state[j] = 0;
  // A streamed run set the flag in `checkBundlePoints`, during the assembly sweep.
  if (control.streamed !== 0) state[20] = failed;
  if (state[1] === 0) return;
  if (control.streamed !== 0) {
    if (state[20] !== 0) return;
  } else {
    for (let p: u32 = 0; p < control.points; p++)
      if (blocks[pointBlock(p) + 12] === 0) {
        state[20] = 1;
        return;
      }
  }
  const m = control.cameras * 6;
  let rho: f64 = 0,
    residual: f64 = 0;
  for (let c: u32 = 0; c < control.cameras; c++) {
    const inv = inverseBlock(c),
      rhs = cameraRhs(c);
    if (!(blocks[inv] > 0)) {
      state[20] = 1;
      return;
    }
    for (let j: u32 = 0; j < 6; j++) {
      const i = c * 6 + j,
        r = blocks[rhs + j];
      let z: f64 = 0;
      // Initialize z from the camera RHS. The residual vector is written by this
      // loop, so reading it here makes the result depend on pooled-buffer contents.
      for (let k: u32 = 0; k < 6; k++) z += blocks[inv + j * 6 + k] * blocks[rhs + k];
      vectors[i] = 0;
      vectors[m + i] = r;
      vectors[2 * m + i] = z;
      vectors[4 * m + i] = z;
      rho += r * z;
      residual += r * r;
    }
  }
  state[18] = rho;
  state[21] = residual;
  state[22] = residual;
  if (residual > 0 && rho > 0) state[19] = 1;
  else if (residual > 1e-24) state[20] = 1;
}
@compute([64])
export function bundleSchurPoint(@builtin('global_invocation_id') gid: vec3u): void {
  bundleSchurPointAt(control.pointBase + gid.x);
}
@compute([64])
export function bundleSchurCamera(@builtin('global_invocation_id') gid: vec3u): void {
  bundleSchurCameraRow(gid.x);
}
// One workgroup per row: long observation lists no longer run on one lane. The serial
// entry remains the CPU oracle and the small-problem path. All lanes reach every barrier,
// including an inactive PCG iteration.
@compute([64])
export function bundleSchurCameraCooperative(
  @builtin('workgroup_id') wid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  const rowId: u32 = wid.x;
  const c: u32 = rowId / 6;
  const j: u32 = rowId % 6;
  const m: u32 = control.cameras * 6;
  const active: bool = c < control.cameras && state[19] !== 0;
  let v: f64 = 0;
  if (active) {
    for (let at: u32 = cameraFirst(c) + lid.x; at < cameraStart(c + 1); at += 64) {
      const i: u32 = cameraObservation(at);
      if (i >= control.obsEnd) break;
      const p: u32 = index[control.observations + i];
      const cross: u32 = crossAt(i);
      for (let k: u32 = 0; k < 3; k++)
        v -= blocks[cross + j * 3 + k] * vectors[m * 5 + p * 3 + k];
    }
  }
  schurTile[lid.x] = v;
  workgroupBarrier();
  for (let step: u32 = 32; step > 0; step /= 2) {
    if (lid.x < step) schurTile[lid.x] += schurTile[lid.x + step];
    workgroupBarrier();
  }
  if (active && lid.x === 0) {
    let base: f64 = 0;
    if (control.first !== 0) {
      const A: u32 = cameraBlock(c);
      for (let k: u32 = 0; k < 6; k++)
        base += blocks[A + j * 6 + k] * vectors[m * 2 + c * 6 + k];
    } else base = vectors[m * 3 + rowId];
    vectors[m * 3 + rowId] = base + schurTile[0];
  }
}
@compute([1])
export function advanceBundleCg(): void {
  if (state[19] === 0) return;
  const m = control.cameras * 6;
  let denominator: f64 = 0;
  for (let i: u32 = 0; i < m; i++) denominator += vectors[m * 2 + i] * vectors[m * 3 + i];
  if (!(denominator > 0)) {
    state[19] = 0;
    state[20] = 1;
    return;
  }
  const alpha: f64 = state[18] / denominator;
  if (!(abs(alpha) < 1e20)) {
    state[19] = 0;
    state[20] = 1;
    return;
  }
  let residual: f64 = 0,
    rho: f64 = 0;
  for (let i: u32 = 0; i < m; i++) {
    vectors[i] += alpha * vectors[m * 2 + i];
    vectors[m + i] -= alpha * vectors[m * 3 + i];
    residual += vectors[m + i] * vectors[m + i];
  }
  for (let i: u32 = 0; i < m; i++) {
    const z = bundlePreconditionedResidualAt(i, m);
    vectors[m * 4 + i] = z;
    rho += z * vectors[m + i];
  }
  state[16]++;
  state[23]++;
  state[22] = residual;
  if (residual <= control.relativeTolerance * control.relativeTolerance * state[21]) {
    state[19] = 0;
    state[24] = 1;
    return;
  }
  if (!(rho > 0)) {
    state[19] = 0;
    state[20] = 1;
    return;
  }
  const beta: f64 = rho / state[18];
  state[18] = rho;
  for (let i: u32 = 0; i < m; i++)
    vectors[m * 2 + i] = vectors[m * 4 + i] + beta * vectors[m * 2 + i];
}
@compute([64])
export function updateBundleCandidate(@builtin('global_invocation_id') gid: vec3u): void {
  const h = gid.x;
  if (state[1] === 0 || state[20] !== 0) return;
  const m = control.cameras * 6;
  // The first window of the update sweep also moves the cameras.
  if (h < control.cameras && control.first !== 0) {
    let delta: array<f64, 6> = fill<f64, 6>(f64(0));
    for (let j: u32 = 0; j < 6; j++) delta[j] = vectors[h * 6 + j];
    const p = poseAt(h, false);
    let moved = p;
    if (variable(h)) moved = movePose(p, delta);
    for (let j: u32 = 0; j < 12; j++) trial[h * 12 + j] = moved[j];
  }
  const point = control.pointBase + h;
  if (point < control.pointEnd) {
    const at = pointBlock(point),
      out = control.cameras * 12 + point * 3;
    let rhs: array<f64, 3> = [blocks[at + 9], blocks[at + 10], blocks[at + 11]];
    for (let i: u32 = pointStart(point); i < pointStart(point + 1); i++) {
      const c = index[i],
        cross = crossAt(i);
      for (let j: u32 = 0; j < 3; j++)
        for (let k: u32 = 0; k < 6; k++) rhs[j] -= blocks[cross + k * 3 + j] * vectors[c * 6 + k];
    }
    for (let j: u32 = 0; j < 3; j++) {
      let v: f64 = scene[out + j];
      for (let k: u32 = 0; k < 3; k++) v += blocks[at + j * 3 + k] * rhs[k];
      trial[out + j] = v;
    }
  }
}
@compute([1])
export function bundleCandidateGauge(): void {
  state[15] = 0;
  if (state[1] === 0 || state[20] !== 0) return;
  // An unconverged PCG solve is rejected rather than silently treated as exact.
  if (
    state[21] > 0 &&
    state[22] > control.relativeTolerance * control.relativeTolerance * state[21]
  ) {
    state[20] = 1;
    return;
  }
  const origin = vec3f64(state[3], state[4], state[5]),
    baseline = length(origin - cameraCenter(control.scaleView, true));
  if (!(baseline > 1e-12)) return;
  let scale: f64 = 1;
  if (control.normalizeScale !== 0) scale = state[2] / baseline;
  if (!(scale > 1e-9 && scale < 1e9)) return;
  for (let c: u32 = 0; c < control.cameras; c++) {
    const p = poseAt(c, true);
    for (let j: u32 = 0; j < 12; j++) if (!(abs(p[j]) <= 1e9)) return;
    for (let j: u32 = 0; j < 3; j++)
      for (let k: u32 = 0; k < 3; k++) {
        let d: f64 = 0;
        for (let r: u32 = 0; r < 3; r++) d += p[j * 3 + r] * p[k * 3 + r];
        if (j === k) d -= 1;
        if (!(abs(d) < 1e-6)) return;
      }
  }
  state[13] = scale;
  state[15] = 1;
}
@compute([64])
export function normalizeBundleCandidate(@builtin('global_invocation_id') gid: vec3u): void {
  const h = control.base + gid.x;
  if (state[1] === 0 || state[15] === 0 || control.normalizeScale === 0) return;
  const s: f64 = state[13];
  if (h < control.points) {
    const at = control.cameras * 12 + h * 3;
    for (let j: u32 = 0; j < 3; j++)
      trial[at + j] = state[3 + j] + s * (trial[at + j] - state[3 + j]);
  }
  if (h < control.cameras && h !== control.anchor) {
    const p = poseAt(h, true),
      o = vec3f64(state[3], state[4], state[5]);
    const rotated = transform(p, o) - vec3f64(p[9], p[10], p[11]);
    trial[h * 12 + 9] = s * p[9] + (s - 1) * rotated.x;
    trial[h * 12 + 10] = s * p[10] + (s - 1) * rotated.y;
    trial[h * 12 + 11] = s * p[11] + (s - 1) * rotated.z;
  }
}
function candidateLane(i: u32): vec2f64 {
  if (i >= control.obsEnd) return vec2f64(f64(0));
  const c = index[i],
    X = pointAt(index[control.observations + i], true);
  if (!(max(abs(X.x), max(abs(X.y), abs(X.z))) <= 1e9)) return vec2f64(f64(1e30), f64(1e30));
  const projection = project(cameraAt(c), transform(poseAt(c, true), X));
  if (projection.ok === 0) return vec2f64(f64(1e30), f64(1e30));
  const pixels = control.cameras * 18 + control.points * 3 + i * 2;
  const x: f64 = projection.pixel.x - scene[pixels],
    y: f64 = projection.pixel.y - scene[pixels + 1],
    squares: f64 = x * x + y * y;
  return vec2f64(robustCost(sqrt(squares), control.huber), squares);
}
// The candidate cost is evaluated for every block; acceptance reads the sums only while the
// candidate is valid, so no lane returns before a barrier.
@compute([64])
export function bundleCandidatePartials(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const v = candidateLane(control.obsBase + wid.x * 64 + lid.x);
  costTile[lid.x] = v.x;
  squareTile[lid.x] = v.y;
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) {
      costTile[lid.x] += costTile[lid.x + stride];
      squareTile[lid.x] += squareTile[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x === 0) {
    const block = control.obsBase / 64 + wid.x;
    partials[block * 2] = costTile[0];
    partials[block * 2 + 1] = squareTile[0];
  }
}
@compute([8])
export function bundleCandidatePartialsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  if (gid.x >= (control.obsEnd - control.obsBase + 63) / 64) return;
  const block: u32 = control.obsBase / 64 + gid.x;
  let total: vec2f64 = vec2f64(f64(0));
  for (let lane: u32 = 0; lane < 64; lane++) total += candidateLane(block * 64 + lane);
  partials[block * 2] = total.x;
  partials[block * 2 + 1] = total.y;
}
@compute([1])
export function acceptBundleCandidate(): void {
  state[12] = 0;
  if (state[1] === 0) return;
  let cost: f64 = 0,
    squares: f64 = 0;
  if (state[15] !== 0)
    for (let b: u32 = 0; b < observationBlocks(); b++) {
      cost += partials[b * 2];
      squares += partials[b * 2 + 1];
    }
  if (state[15] !== 0 && cost < state[6]) {
    const change: f64 = state[6] - cost;
    state[6] = cost;
    state[7] = squares;
    state[10]++;
    state[12] = 1;
    state[0] = max(f64(1e-10), state[0] / 3);
    if (change < 1e-9 * max(f64(1), cost)) state[1] = 0;
  } else {
    state[11]++;
    state[0] *= 10;
    if (state[20] !== 0) state[17]++;
    if (state[0] > 1e12) state[1] = 0;
  }
}
@compute([1])
export function snapshotBundleActivity(): void {
  activity[0] = u32(f32(state[19]));
  activity[1] = u32(f32(state[1]));
}
@compute([64])
export function commitBundleCandidate(@builtin('global_invocation_id') gid: vec3u): void {
  const i = control.base + gid.x;
  if (state[12] === 0 || i >= trial.length) return;
  scene[i] = trial[i];
}
@compute([64])
export function packBundleResult(@builtin('global_invocation_id') gid: vec3u): void {
  const i = control.base + gid.x,
    total = control.cameras * 12 + control.points * 3;
  if (i < total) output[i] = scene[i];
  if (i < 26) {
    output[total + i] = state[i];
    if (i === 24) output[total + i] = sqrt(state[9] / f64(f32(control.observations)));
    if (i === 25) output[total + i] = sqrt(state[7] / f64(f32(control.observations)));
  }
}
