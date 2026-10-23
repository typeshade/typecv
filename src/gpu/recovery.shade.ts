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
  angleBetween,
} from './numerics.shade.ts';
import { basis3, triangulatePair } from './projective.shade.ts';

class RecoveryControl {
  first: CameraModel;
  second: CameraModel;
  threshold: f64;
  huber: f64;
  minAngle: f64;
  maxError: f64;
  iterations: u32;
  separateDirectional: u32;
}
declare const control: uniform<RecoveryControl>;
declare const pixels: storage<array<f64>>;
declare const coords: storage<array<f64>>;
declare const invertible: storage<array<u32>>;
declare const essential: storage<array<f64>>;
declare const mask: storage<array<u32>, 'read_write'>;
declare const poses: storage<array<f64>, 'read_write'>;
// Four candidate XYZ/state records per match: 1 finite positive, -1 direction-compatible, 0 rejected.
declare const cloud: storage<array<f64>, 'read_write'>;
// Current/original/trial pose12, step6, cost, damping, mean, initialMean,
// positive count, accepted, rejected, active, step valid, scale, applied,
// median, final valid count, final error, status, final E9, directional count, winner, commit.
declare const info: storage<array<f64>, 'read_write'>;
// Bounded host-side checkpoints read the active bit without copying the recovery state.
declare const activity: storage<array<u32>, 'read_write'>;
declare const points: storage<array<f64>, 'read_write'>;
declare const trialPoints: storage<array<f64>, 'read_write'>;
// 0 excluded, 1 finite XYZ, 2 unresolved [bearing.x, bearing.y, inverseDepth].
declare const fixed: storage<array<u32>, 'read_write'>;
declare const flags: storage<array<u32>, 'read_write'>;
// W18, damped V inverse9, point RHS3.
declare const pointBlocks: storage<array<f64>, 'read_write'>;
declare const normalBlocks: storage<array<f64>, 'read_write'>;
// Robust cost, mean Euclidean error, invalid flag, final angle.
declare const metrics: storage<array<f64>, 'read_write'>;
declare const angles: storage<array<f64>, 'read_write'>;
let normalTile: workgroup<array<f64, 1408>>;

function getPose(at: u32): array<f64, 12> {
  let p: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) p[j] = info[at + j];
  return p;
}
function xy(i: u32): vec4f64 {
  return vec4f64(coords[i * 4], coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]);
}
// Symmetric rotation-only transfer within the finite-point pixel budget.
// Compatibility leaves depth unresolved; it does not establish exact infinity.
function directionCompatible(i: u32, p: array<f64, 12>): bool {
  if (control.separateDirectional === 0) return false;
  const c = xy(i);
  const a = vec3f64(p[0] * c.x + p[1] * c.y + p[2], p[3] * c.x + p[4] * c.y + p[5], p[6] * c.x + p[7] * c.y + p[8]);
  const b = vec3f64(p[0] * c.z + p[3] * c.w + p[6], p[1] * c.z + p[4] * c.w + p[7], p[2] * c.z + p[5] * c.w + p[8]);
  const forward = project(control.second, a),
    backward = project(control.first, b);
  return forward.ok !== 0 && backward.ok !== 0 &&
    length(forward.pixel - vec2f64(pixels[4 * i + 2], pixels[4 * i + 3])) <= control.maxError &&
    length(backward.pixel - vec2f64(pixels[4 * i], pixels[4 * i + 1])) <= control.maxError;
}
function observedPoint(X: vec3f64, p: array<f64, 12>, second: bool, inverse: bool): vec3f64 {
  let point = X;
  if (inverse) point = vec3f64(X.x, X.y, f64(1));
  if (!second) return point;
  if (!inverse) return transform(p, point);
  return vec3f64(
    p[0] * X.x + p[1] * X.y + p[2] + X.z * p[9],
    p[3] * X.x + p[4] * X.y + p[5] + X.z * p[10],
    p[6] * X.x + p[7] * X.y + p[8] + X.z * p[11],
  );
}
function pointRow(g: vec3f64, p: array<f64, 12>, second: bool, inverse: bool): array<f64, 3> {
  if (!second) {
    if (inverse) return [g.x, g.y, f64(0)];
    return [g.x, g.y, g.z];
  }
  const x: f64 = g.x * p[0] + g.y * p[3] + g.z * p[6],
    y: f64 = g.x * p[1] + g.y * p[4] + g.z * p[7];
  if (inverse) return [x, y, g.x * p[9] + g.y * p[10] + g.z * p[11]];
  return [x, y, g.x * p[2] + g.y * p[5] + g.z * p[8]];
}
function evaluate(i: u32, X: vec3f64, p: array<f64, 12>, inverse: bool): vec3f64 {
  const a = project(control.first, observedPoint(X, p, false, inverse)),
    b = project(control.second, observedPoint(X, p, true, inverse));
  if (a.ok === 0 || b.ok === 0) return vec3f64(f64(1e30), f64(1e30), f64(1));
  const da = a.pixel - vec2f64(pixels[4 * i], pixels[4 * i + 1]);
  const db = b.pixel - vec2f64(pixels[4 * i + 2], pixels[4 * i + 3]);
  const ea: f64 = length(da),
    eb: f64 = length(db);
  let mean: f64 = (ea + eb) * 0.5;
  // The public mean before/after remains conditional on the fixed finite set.
  if (inverse) mean = 0;
  return vec3f64(
    robustCost(ea, control.huber) + robustCost(eb, control.huber),
    mean,
    f64(0),
  );
}

@compute([1])
export function decomposeEssential(): void {
  for (let j: u32 = 0; j < info.length; j++) info[j] = 0;
  for (let j: u32 = 0; j < poses.length; j++) poses[j] = 0;
  info[56] = 1;
  if (essential[18] < 8) return;
  let E: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let j: u32 = 0; j < 9; j++) {
    E[j] = essential[j];
    info[57 + j] = E[j];
  }
  const b = basis3(E);
  if (b.ok === 0) return;
  for (let h: u32 = 0; h < 4; h++) {
    let W: array<f64, 9> = [0, -1, 0, 1, 0, 0, 0, 0, 1];
    if (h >= 2) W = [0, 1, 0, -1, 0, 0, 0, 0, 1];
    for (let r: u32 = 0; r < 3; r++) {
      for (let c: u32 = 0; c < 3; c++) {
        let value: f64 = 0;
        for (let j: u32 = 0; j < 3; j++)
          for (let k: u32 = 0; k < 3; k++) value += b.U[r * 3 + j] * W[j * 3 + k] * b.V[c * 3 + k];
        poses[h * 12 + r * 3 + c] = value;
      }
      let sign: f64 = 1;
      if (h % 2 !== 0) sign = -1;
      poses[h * 12 + 9 + r] = sign * b.U[r * 3 + 2];
    }
  }
  info[56] = 0;
}
@compute([16])
export function triangulatePoseCandidates(@builtin('global_invocation_id') gid: vec3u): void {
  const at: u32 = gid.x,
    i: u32 = at / 4,
    h: u32 = at % 4;
  if (i >= mask.length) return;
  if (at === 0) info[70] = f64(f32(control.separateDirectional));
  for (let j: u32 = 0; j < 4; j++) cloud[at * 4 + j] = 0;
  if (info[56] !== 0 || mask[i] === 0) return;
  let p: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) p[j] = poses[h * 12 + j];
  if (directionCompatible(i, p)) {
    cloud[at * 4 + 3] = -1;
    return;
  }
  const X = triangulatePair(xy(i), p);
  if (X.w === 0 || X.z <= 1e-12 || transform(p, X.xyz).z <= 1e-12) return;
  cloud[at * 4] = X.x;
  cloud[at * 4 + 1] = X.y;
  cloud[at * 4 + 2] = X.z;
  cloud[at * 4 + 3] = 1;
}
@compute([1])
export function chooseTwoViewPose(): void {
  if (info[56] !== 0) return;
  let best: u32 = 0,
    winner: u32 = 0,
    bestDirectional: u32 = 0;
  for (let h: u32 = 0; h < 4; h++) {
    let positive: u32 = 0,
      directional: u32 = 0;
    for (let i: u32 = 0; i < mask.length; i++) {
      const state: f64 = cloud[(i * 4 + h) * 4 + 3];
      if (state === 1) positive++;
      if (state === -1) directional++;
    }
    if (positive > best) {
      best = positive;
      winner = h;
      bestDirectional = directional;
    }
  }
  if (f64(f32(best)) < max(f64(6), floor((essential[18] - f64(f32(bestDirectional))) * 0.6))) {
    info[56] = 2;
    return;
  }
  info[46] = f64(f32(best));
  info[67] = f64(f32(winner));
  for (let j: u32 = 0; j < 12; j++) {
    info[j] = poses[winner * 12 + j];
    info[12 + j] = info[j];
  }
}
@compute([64])
export function seedTwoViewPoints(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length) return;
  fixed[i] = 0;
  for (let j: u32 = 0; j < 3; j++) points[i * 3 + j] = 0;
  for (let j: u32 = 0; j < 4; j++) metrics[i * 4 + j] = 0;
  if (info[56] !== 0) return;
  const at: u32 = (i * 4 + u32(f32(info[67]))) * 4;
  if (cloud[at + 3] === 0) return;
  let X = vec3f64(cloud[at], cloud[at + 1], cloud[at + 2]);
  let inverse: bool = false;
  fixed[i] = 1;
  if (cloud[at + 3] === -1) {
    const c = xy(i);
    X = vec3f64(c.x, c.y, f64(0));
    inverse = true;
    fixed[i] = 2;
  }
  const e = evaluate(i, X, getPose(0), inverse);
  points[i * 3] = X.x;
  points[i * 3 + 1] = X.y;
  points[i * 3 + 2] = X.z;
  metrics[i * 4] = e.x;
  metrics[i * 4 + 1] = e.y;
  metrics[i * 4 + 2] = e.z;
}
@compute([1])
export function beginTwoViewRefinement(): void {
  if (info[56] !== 0) return;
  let value: f64 = 0,
    mean: f64 = 0,
    invalid: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) {
    value += metrics[i * 4];
    mean += metrics[i * 4 + 1];
    invalid += metrics[i * 4 + 2];
  }
  if (invalid !== 0) {
    info[56] = 5;
    return;
  }
  info[42] = value;
  info[43] = 1e-3;
  info[44] = mean / info[46];
  info[45] = info[44];
  if (control.iterations > 0) info[49] = 1;
}

function normalLane(i: u32): array<f64, 44> {
  let out: array<f64, 44> = fill<f64, 44>(f64(0));
  if (i >= mask.length || info[49] === 0 || fixed[i] === 0) return out;
  const inverse: bool = fixed[i] === 2;
  const p = getPose(0),
    X = vec3f64(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
  let U: array<f64, 36> = fill<f64, 36>(f64(0));
  let V: array<f64, 9> = fill<f64, 9>(f64(0));
  let W: array<f64, 18> = fill<f64, 18>(f64(0));
  let bc: array<f64, 6> = fill<f64, 6>(f64(0));
  let bp: array<f64, 3> = fill<f64, 3>(f64(0));
  for (let view: u32 = 0; view < 2; view++) {
    const q = observedPoint(X, p, view === 1, inverse);
    let camera = control.first;
    if (view === 1) {
      camera = control.second;
    }
    const projection = project(camera, q);
    if (projection.ok === 0) {
      out[43] = 1;
      return out;
    }
    const residual =
      projection.pixel - vec2f64(pixels[i * 4 + view * 2], pixels[i * 4 + view * 2 + 1]);
    const norm: f64 = length(residual);
    let weight: f64 = 1;
    if (norm > control.huber) weight = control.huber / norm;
    for (let row: u32 = 0; row < 2; row++) {
      let g = projection.du;
      let e: f64 = residual.x;
      if (row === 1) {
        g = projection.dv;
        e = residual.y;
      }
      const jp = pointRow(g, p, view === 1, inverse);
      let jc: array<f64, 6> = fill<f64, 6>(f64(0));
      if (view === 1) {
        jc = poseRow(q, g);
        if (inverse) for (let j: u32 = 3; j < 6; j++) jc[j] *= X.z;
      }
      for (let j: u32 = 0; j < 3; j++) {
        bp[j] -= weight * jp[j] * e;
        for (let k: u32 = 0; k < 3; k++) V[j * 3 + k] += weight * jp[j] * jp[k];
      }
      for (let j: u32 = 0; j < 6; j++) {
        bc[j] -= weight * jc[j] * e;
        for (let k: u32 = 0; k < 6; k++) U[j * 6 + k] += weight * jc[j] * jc[k];
        for (let k: u32 = 0; k < 3; k++) W[j * 3 + k] += weight * jc[j] * jp[k];
      }
    }
  }
  for (let j: u32 = 0; j < 3; j++) V[j * 3 + j] += info[43] * max(V[j * 3 + j], f64(1e-12));
  let inv: array<f64, 9> = fill<f64, 9>(f64(0));
  for (let j: u32 = 0; j < 3; j++) {
    let basis: array<f64, 3> = [0, 0, 0];
    basis[j] = 1;
    const solution = solve3(V, vec3f64(basis[0], basis[1], basis[2]));
    if (solution.ok === 0) {
      out[43] = 1;
      return out;
    }
    inv[j] = solution.values.x;
    inv[3 + j] = solution.values.y;
    inv[6 + j] = solution.values.z;
  }
  for (let j: u32 = 0; j < 18; j++) pointBlocks[i * 30 + j] = W[j];
  for (let j: u32 = 0; j < 9; j++) pointBlocks[i * 30 + 18 + j] = inv[j];
  for (let j: u32 = 0; j < 3; j++) pointBlocks[i * 30 + 27 + j] = bp[j];
  for (let j: u32 = 0; j < 6; j++) {
    out[36 + j] = bc[j];
    for (let k: u32 = 0; k < 6; k++) out[j * 6 + k] = U[j * 6 + k];
    out[j * 6 + j] += info[43] * max(U[j * 6 + j], f64(1e-12));
    for (let a: u32 = 0; a < 3; a++)
      for (let b: u32 = 0; b < 3; b++) {
        const wv: f64 = W[j * 3 + a] * inv[a * 3 + b];
        out[36 + j] -= wv * bp[b];
        for (let k: u32 = 0; k < 6; k++) out[j * 6 + k] -= wv * W[k * 3 + b];
      }
  }
  return out;
}
@compute([32])
export function twoViewNormals(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const values = normalLane(wid.x * 32 + lid.x);
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
@compute([8])
export function twoViewNormalsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= normalBlocks.length / 44) return;
  let total: array<f64, 44> = fill<f64, 44>(f64(0));
  for (let lane: u32 = 0; lane < 32; lane++) {
    const values = normalLane(block * 32 + lane);
    for (let j: u32 = 0; j < 44; j++) total[j] += values[j];
  }
  for (let j: u32 = 0; j < 44; j++) normalBlocks[block * 44 + j] = total[j];
}
@compute([1])
export function twoViewStep(): void {
  info[50] = 0;
  if (info[49] === 0) return;
  let A: array<f64, 36> = fill<f64, 36>(f64(0));
  let b: array<f64, 6> = fill<f64, 6>(f64(0));
  let invalid: f64 = 0;
  for (let block: u32 = 0; block < normalBlocks.length / 44; block++) {
    for (let j: u32 = 0; j < 36; j++) A[j] += normalBlocks[block * 44 + j];
    for (let j: u32 = 0; j < 6; j++) b[j] += normalBlocks[block * 44 + 36 + j];
    invalid += normalBlocks[block * 44 + 43];
  }
  if (invalid !== 0) return;
  const step = solve6(A, b);
  if (step.ok === 0) return;
  let moved = movePose(getPose(0), step.values);
  const scale: f64 = sqrt(moved[9] * moved[9] + moved[10] * moved[10] + moved[11] * moved[11]);
  if (!(scale > 1e-8 && scale < 1e8)) return;
  info[51] = scale;
  for (let j: u32 = 0; j < 3; j++) moved[9 + j] /= scale;
  for (let j: u32 = 0; j < 12; j++) info[24 + j] = moved[j];
  for (let j: u32 = 0; j < 6; j++) info[36 + j] = step.values[j];
  info[50] = 1;
}
@compute([64])
export function twoViewTrial(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length) return;
  for (let j: u32 = 0; j < 3; j++) metrics[i * 4 + j] = 0;
  if (info[49] === 0 || info[50] === 0 || fixed[i] === 0) return;
  let rhs: array<f64, 3> = [0, 0, 0];
  for (let j: u32 = 0; j < 3; j++) {
    rhs[j] = pointBlocks[i * 30 + 27 + j];
    for (let k: u32 = 0; k < 6; k++) rhs[j] -= pointBlocks[i * 30 + k * 3 + j] * info[36 + k];
  }
  let X: array<f64, 3> = [0, 0, 0];
  for (let j: u32 = 0; j < 3; j++) {
    X[j] = points[i * 3 + j];
    for (let k: u32 = 0; k < 3; k++) X[j] += pointBlocks[i * 30 + 18 + j * 3 + k] * rhs[k];
    if (fixed[i] === 1) X[j] /= info[51];
    else if (j === 2) X[j] *= info[51];
    trialPoints[i * 3 + j] = X[j];
  }
  const e = evaluate(i, vec3f64(X[0], X[1], X[2]), getPose(24), fixed[i] === 2);
  metrics[i * 4] = e.x;
  metrics[i * 4 + 1] = e.y;
  metrics[i * 4 + 2] = e.z;
}
@compute([1])
export function acceptTwoViewStep(): void {
  info[68] = 0;
  if (info[49] === 0) return;
  let value: f64 = 0,
    mean: f64 = 0,
    invalid: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) {
    value += metrics[i * 4];
    mean += metrics[i * 4 + 1];
    invalid += metrics[i * 4 + 2];
  }
  if (info[50] !== 0 && invalid === 0 && value < info[42]) {
    const change: f64 = info[42] - value;
    for (let j: u32 = 0; j < 12; j++) info[j] = info[24 + j];
    info[42] = value;
    info[44] = mean / info[46];
    info[43] = max(f64(1e-10), info[43] / 3);
    info[47]++;
    info[68] = 1;
    if (change < 1e-12 * max(f64(1), value)) info[49] = 0;
  } else {
    info[43] *= 10;
    info[48]++;
    if (info[43] > 1e12) info[49] = 0;
  }
}
@compute([1])
export function snapshotTwoViewActivity(): void {
  activity[0] = u32(f32(info[49]));
  // One skipped inactive iteration would clear these transient flags before exit.
  if (info[49] === 0) {
    info[50] = 0;
    info[68] = 0;
  }
}
@compute([64])
export function commitTwoViewPoints(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length || info[68] === 0 || fixed[i] === 0) return;
  for (let j: u32 = 0; j < 3; j++) points[i * 3 + j] = trialPoints[i * 3 + j];
}
@compute([1])
export function finishTwoViewRefinement(): void {
  if (info[56] !== 0) return;
  let mean: f64 = 0,
    invalid: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) {
    mean += metrics[i * 4 + 1];
    invalid += metrics[i * 4 + 2];
  }
  mean /= max(f64(1), info[46]);
  if (info[47] > 0 && invalid === 0 && mean <= info[45]) {
    info[44] = mean;
    info[52] = 1;
    const p = getPose(0);
    let skew: array<f64, 9> = [0, -p[11], p[10], p[11], 0, -p[9], -p[10], p[9], 0];
    let E: array<f64, 9> = fill<f64, 9>(f64(0));
    let norm: f64 = 0;
    for (let r: u32 = 0; r < 3; r++)
      for (let c: u32 = 0; c < 3; c++) {
        for (let j: u32 = 0; j < 3; j++) E[r * 3 + c] += skew[r * 3 + j] * p[j * 3 + c];
        norm += E[r * 3 + c] * E[r * 3 + c];
      }
    for (let j: u32 = 0; j < 9; j++) info[57 + j] = E[j] / sqrt(norm);
  } else {
    for (let j: u32 = 0; j < 12; j++) info[j] = info[12 + j];
    info[44] = info[45];
  }
}

function pointCost(i: u32, X: vec3f64, p: array<f64, 12>, inverse: bool): f64 {
  const a = project(control.first, observedPoint(X, p, false, inverse)),
    b = project(control.second, observedPoint(X, p, true, inverse));
  if (a.ok === 0 || b.ok === 0) return 1e30;
  const da = a.pixel - vec2f64(pixels[4 * i], pixels[4 * i + 1]);
  const db = b.pixel - vec2f64(pixels[4 * i + 2], pixels[4 * i + 3]);
  return dot(da, da) + dot(db, db);
}
function refinePoint(i: u32, initial: vec3f64, p: array<f64, 12>, inverse: bool): vec3f64 {
  let X = initial;
  let cost: f64 = pointCost(i, X, p, inverse),
    lambda: f64 = 1e-3;
  for (let it: u32 = 0; it < 8; it++) {
    let A: array<f64, 9> = fill<f64, 9>(f64(0));
    let rhs: array<f64, 3> = [0, 0, 0];
    for (let view: u32 = 0; view < 2; view++) {
      const q = observedPoint(X, p, view === 1, inverse);
      let camera = control.first;
      if (view === 1) {
        camera = control.second;
      }
      const projection = project(camera, q);
      if (projection.ok === 0) return X;
      const residual =
        projection.pixel - vec2f64(pixels[i * 4 + view * 2], pixels[i * 4 + view * 2 + 1]);
      for (let row: u32 = 0; row < 2; row++) {
        let g = projection.du;
        let e: f64 = residual.x;
        if (row === 1) {
          g = projection.dv;
          e = residual.y;
        }
        const jp = pointRow(g, p, view === 1, inverse);
        for (let j: u32 = 0; j < 3; j++) {
          rhs[j] -= jp[j] * e;
          for (let k: u32 = 0; k < 3; k++) A[j * 3 + k] += jp[j] * jp[k];
        }
      }
    }
    for (let j: u32 = 0; j < 3; j++) A[j * 3 + j] += lambda * max(A[j * 3 + j], f64(1e-12));
    const step = solve3(A, vec3f64(rhs[0], rhs[1], rhs[2]));
    if (step.ok === 0) break;
    const moved = X + step.values;
    const next: f64 = pointCost(i, moved, p, inverse);
    if (next < cost) {
      const change: f64 = cost - next;
      X = moved;
      cost = next;
      lambda = max(f64(1e-10), lambda / 3);
      if (change < 1e-12 * max(f64(1), cost)) break;
    } else lambda *= 10;
  }
  return X;
}
// Check the public Euclidean metric after re-solving points at the proposed
// pose, on the same original positive-depth set. No point may disappear.
@compute([16])
export function validateTwoViewRefinement(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= fixed.length) return;
  for (let j: u32 = 0; j < 3; j++) metrics[i * 4 + j] = 0;
  if (fixed[i] === 0 || info[56] !== 0 || info[47] === 0) return;
  const p = getPose(0),
    c = xy(i);
  const inverse: bool = fixed[i] === 2;
  let initial = vec4f64(c.x, c.y, f64(0), f64(1));
  if (!inverse) initial = triangulatePair(c, p);
  if (initial.w === 0) {
    metrics[i * 4 + 2] = 1;
    return;
  }
  const X = refinePoint(i, initial.xyz, p, inverse),
    e = evaluate(i, X, p, inverse);
  metrics[i * 4] = e.x;
  metrics[i * 4 + 1] = e.y;
  metrics[i * 4 + 2] = e.z;
}
@compute([16])
export function finishTwoViewPoints(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length) return;
  flags[i] = 0;
  for (let j: u32 = 0; j < 3; j++) points[i * 3 + j] = 0;
  metrics[i * 4 + 1] = 0;
  metrics[i * 4 + 3] = -1;
  if (info[56] !== 0 || invertible[i] === 0) {
    mask[i] = 0;
    return;
  }
  if (info[52] !== 0) {
    const c = xy(i);
    const a: f64 = info[57] * c.x + info[58] * c.y + info[59];
    const b: f64 = info[60] * c.x + info[61] * c.y + info[62];
    const z: f64 = info[63] * c.x + info[64] * c.y + info[65];
    const d: f64 = info[57] * c.z + info[60] * c.w + info[63];
    const e: f64 = info[58] * c.z + info[61] * c.w + info[64];
    const denom: f64 = a * a + b * b + d * d + e * e,
      residual: f64 = c.z * a + c.w * b + z;
    mask[i] = 0;
    if (denom > 1e-12 && residual * residual <= control.threshold * control.threshold * denom)
      mask[i] = 1;
  }
  if (mask[i] === 0) return;
  const p = getPose(0);
  if (directionCompatible(i, p)) {
    metrics[i * 4 + 3] = -2;
    return;
  }
  const linear = triangulatePair(xy(i), p);
  if (linear.w === 0) return;
  let X = linear.xyz;
  if (info[52] !== 0) X = refinePoint(i, X, p, false);
  const a = project(control.first, X),
    b = project(control.second, transform(p, X));
  if (a.ok === 0 || b.ok === 0) return;
  const rotated = transform(p, X) - vec3f64(p[9], p[10], p[11]);
  const angle: f64 = angleBetween(rotated, transform(p, X));
  if (angle < 0) return;
  metrics[i * 4 + 3] = angle;
  const ea: f64 = length(a.pixel - vec2f64(pixels[4 * i], pixels[4 * i + 1]));
  const eb: f64 = length(b.pixel - vec2f64(pixels[4 * i + 2], pixels[4 * i + 3]));
  if (angle < control.minAngle || ea > control.maxError || eb > control.maxError) return;
  flags[i] = 1;
  points[i * 3] = X.x;
  points[i * 3 + 1] = X.y;
  points[i * 3 + 2] = X.z;
  metrics[i * 4 + 1] = (ea + eb) * 0.5;
}
function selectAngle(target: u32, count: u32): f64 {
  let left: u32 = 0,
    right: u32 = count;
  for (let iteration: u32 = 0; iteration < 128; iteration++) {
    if (right - left === 1) return angles[left];
    const a: f64 = angles[left],
      b: f64 = angles[left + (right - left) / 2],
      c: f64 = angles[right - 1];
    const pivot: f64 = max(min(a, b), min(max(a, b), c));
    let lo: u32 = left,
      i: u32 = left,
      hi: u32 = right;
    while (i < hi) {
      if (angles[i] < pivot) {
        const value: f64 = angles[lo];
        angles[lo] = angles[i];
        angles[i] = value;
        lo++;
        i++;
      } else if (angles[i] > pivot) {
        hi--;
        const value: f64 = angles[hi];
        angles[hi] = angles[i];
        angles[i] = value;
      } else i++;
    }
    if (target < lo) right = lo;
    else if (target >= hi) left = hi;
    else return pivot;
  }
  return -1;
}
@compute([1])
export function summarizeTwoView(): void {
  if (info[56] !== 0) return;
  let count: u32 = 0,
    valid: u32 = 0,
    inliers: u32 = 0,
    directional: u32 = 0;
  let error: f64 = 0;
  for (let i: u32 = 0; i < mask.length; i++) {
    if (mask[i] !== 0) inliers++;
    if (metrics[i * 4 + 3] === -2) directional++;
    if (metrics[i * 4 + 3] >= 0) {
      angles[count] = metrics[i * 4 + 3];
      count++;
    }
    if (flags[i] !== 0) {
      valid++;
      error += metrics[i * 4 + 1];
    }
  }
  info[54] = f64(f32(valid));
  info[69] = f64(f32(inliers));
  info[66] = f64(f32(directional));
  if (count === 0) {
    info[56] = 4;
    return;
  }
  let median: f64 = selectAngle(count / 2, count);
  if (count % 2 === 0) median = (median + selectAngle(count / 2 - 1, count)) * 0.5;
  info[53] = median;
  if (median < 0) {
    info[56] = 5;
    return;
  }
  if (median < control.minAngle) {
    info[56] = 3;
    return;
  }
  if (valid < 6) {
    info[56] = 4;
    return;
  }
  info[55] = error / f64(f32(valid));
}
