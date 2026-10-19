'use typeshade';
import {
  CameraModel,
  unproject,
  project,
  transform,
  solve3,
  angleBetween,
} from './numerics.shade.ts';

class TrackControl {
  maxError: f64;
  minAngle: f64;
  maxPairs: u32;
  iterations: u32;
}
declare const control: uniform<TrackControl>;
declare const raw: storage<array<f64>>;
declare const offsets: storage<array<u32>>;
// Normalized image xy, camera center XYZ, inversion flag.
declare const geometry: storage<array<f64>, 'read_write'>;
declare const result: storage<array<f64>, 'read_write'>;
declare const flags: storage<array<u32>, 'read_write'>;
declare const statuses: storage<array<u32>, 'read_write'>;
declare const points: storage<array<f64>>;
declare const errors: storage<array<f64>, 'read_write'>;
declare const summary: storage<array<f64>, 'read_write'>;
// Per block of 64 observations: accepted error sum, accepted count.
declare const partials: storage<array<f64>, 'read_write'>;
let errorTile: workgroup<array<f64, 64>>;
let countTile: workgroup<array<u32, 64>>;

function getCamera(i: u32): CameraModel {
  const at: u32 = i * 20 + 12;
  return {
    fx: raw[at],
    fy: raw[at + 1],
    cx: raw[at + 2],
    cy: raw[at + 3],
    k1: raw[at + 4],
    k2: raw[at + 5],
  };
}
function getPose(i: u32): array<f64, 12> {
  let p: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) p[j] = raw[i * 20 + j];
  return p;
}
function center(i: u32): vec3f64 {
  return vec3f64(geometry[i * 6 + 2], geometry[i * 6 + 3], geometry[i * 6 + 4]);
}
function errorAt(i: u32, X: vec3f64): f64 {
  if (geometry[i * 6 + 5] === 0) return 1e30;
  const p = project(getCamera(i), transform(getPose(i), X));
  if (p.ok === 0) return 1e30;
  const x: f64 = p.pixel.x - raw[i * 20 + 18],
    y: f64 = p.pixel.y - raw[i * 20 + 19];
  return x * x + y * y;
}
@compute([64])
export function prepareTriangulation(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= raw.length / 20) return;
  const ray = unproject(getCamera(i), vec2f64(raw[i * 20 + 18], raw[i * 20 + 19]));
  const p = getPose(i);
  geometry[i * 6] = ray.x;
  geometry[i * 6 + 1] = ray.y;
  for (let j: u32 = 0; j < 3; j++)
    geometry[i * 6 + 2 + j] = -p[j] * p[9] - p[3 + j] * p[10] - p[6 + j] * p[11];
  geometry[i * 6 + 5] = 0;
  if (ray.z !== 0 && max(abs(ray.x), abs(ray.y)) <= 1e4) geometry[i * 6 + 5] = 1;
}

@compute([64])
export function projectObservations(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= points.length / 3) return;
  flags[i] = 0;
  errors[i] = 0;
  const X = vec3f64(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
  const p = project(getCamera(i), transform(getPose(i), X));
  if (p.ok === 0) return;
  const x: f64 = p.pixel.x - raw[i * 20 + 18],
    y: f64 = p.pixel.y - raw[i * 20 + 19];
  const e: f64 = sqrt(x * x + y * y);
  if (e <= control.maxError) {
    flags[i] = 1;
    errors[i] = e;
  }
}
function observationLane(i: u32): vec2f64 {
  if (i >= points.length / 3 || flags[i] === 0) return vec2f64(f64(0));
  return vec2f64(errors[i], f64(1));
}
@compute([64])
export function observationPartials(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const v = observationLane(wid.x * 64 + lid.x);
  errorTile[lid.x] = v.x;
  countTile[lid.x] = u32(f32(v.y));
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) {
      errorTile[lid.x] += errorTile[lid.x + stride];
      countTile[lid.x] += countTile[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x === 0) {
    partials[wid.x * 2] = errorTile[0];
    partials[wid.x * 2 + 1] = f64(f32(countTile[0]));
  }
}
@compute([8])
export function observationPartialsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= partials.length / 2) return;
  let total: vec2f64 = vec2f64(f64(0));
  for (let lane: u32 = 0; lane < 64; lane++) total += observationLane(block * 64 + lane);
  partials[block * 2] = total.x;
  partials[block * 2 + 1] = total.y;
}
@compute([1])
export function observationSummary(): void {
  let count: f64 = 0;
  let error: f64 = 0;
  for (let b: u32 = 0; b < partials.length / 2; b++) {
    error += partials[b * 2];
    count += partials[b * 2 + 1];
  }
  summary[0] = error;
  summary[1] = count;
}

// Direct one-sided Jacobi SVD of normalized DLT rows; never forms A^T A.
function linear(
  start: u32,
  end: u32,
  mode: u32,
  first: u32,
  second: u32,
  mask: array<u32, 64>,
): vec4f64 {
  let a: array<f64, 512> = fill<f64, 512>(f64(0));
  let v: array<f64, 16> = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  let rows: u32 = 0;
  for (let i: u32 = start; i < end; i++) {
    if ((mode === 1 && i !== first && i !== second) || (mode === 2 && mask[i - start] === 0))
      continue;
    if (geometry[i * 6 + 5] === 0) return vec4f64(f64(0));
    const p = getPose(i);
    for (let k: u32 = 0; k < 2; k++) {
      const xy: f64 = geometry[i * 6 + k];
      let row: array<f64, 4> = [
        xy * p[6] - p[k * 3],
        xy * p[7] - p[k * 3 + 1],
        xy * p[8] - p[k * 3 + 2],
        xy * p[11] - p[9 + k],
      ];
      const norm: f64 = sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2] + row[3] * row[3]);
      if (!(norm > 0)) return vec4f64(f64(0));
      for (let j: u32 = 0; j < 4; j++) a[rows * 4 + j] = row[j] / norm;
      rows++;
    }
  }
  if (rows < 4) return vec4f64(f64(0));
  let converged: bool = false;
  for (let sweep: u32 = 0; sweep < 160; sweep++) {
    let changed: bool = false;
    for (let p: u32 = 0; p < 4; p++)
      for (let q: u32 = p + 1; q < 4; q++) {
        let aa: f64 = 0,
          bb: f64 = 0,
          ab: f64 = 0;
        for (let r: u32 = 0; r < rows; r++) {
          const x = a[r * 4 + p],
            y = a[r * 4 + q];
          aa += x * x;
          bb += y * y;
          ab += x * y;
        }
        if (abs(ab) <= 1e-13 * sqrt(aa * bb) || aa < 1e-30 || bb < 1e-30) continue;
        const tau: f64 = (bb - aa) / (2 * ab),
          at: f64 = abs(tau);
        let hypot: f64 = sqrt(1 + tau * tau);
        if (at > 1) hypot = at * sqrt(1 + (1 / at) * (1 / at));
        let sign: f64 = 1;
        if (tau < 0) sign = -1;
        const t: f64 = sign / (at + hypot),
          c: f64 = 1 / sqrt(1 + t * t),
          s: f64 = c * t;
        for (let r: u32 = 0; r < rows; r++) {
          const x = a[r * 4 + p],
            y = a[r * 4 + q];
          a[r * 4 + p] = c * x - s * y;
          a[r * 4 + q] = s * x + c * y;
        }
        for (let r: u32 = 0; r < 4; r++) {
          const x = v[r * 4 + p],
            y = v[r * 4 + q];
          v[r * 4 + p] = c * x - s * y;
          v[r * 4 + q] = s * x + c * y;
        }
        changed = true;
      }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) return vec4f64(f64(0));
  let singular: array<f64, 4> = [0, 0, 0, 0];
  for (let j: u32 = 0; j < 4; j++)
    for (let r: u32 = 0; r < rows; r++) singular[j] += a[r * 4 + j] * a[r * 4 + j];
  let index: u32 = 0;
  for (let j: u32 = 1; j < 4; j++) if (singular[j] < singular[index]) index = j;
  let secondValue: f64 = 1e30,
    maxValue: f64 = 0;
  for (let j: u32 = 0; j < 4; j++) {
    maxValue = max(maxValue, singular[j]);
    if (j !== index) secondValue = min(secondValue, singular[j]);
  }
  if (secondValue < 1e-20 * maxValue || abs(v[12 + index]) < 1e-12) return vec4f64(f64(0));
  const X = vec3f64(v[index], v[4 + index], v[8 + index]) / v[12 + index];
  if (!(max(abs(X.x), max(abs(X.y), abs(X.z))) <= 1e9)) return vec4f64(f64(0));
  return vec4f64(X, f64(1));
}
function selectedCost(start: u32, end: u32, X: vec3f64, mask: array<u32, 64>): f64 {
  let cost: f64 = 0;
  for (let i: u32 = start; i < end; i++) if (mask[i - start] !== 0) cost += errorAt(i, X);
  return cost;
}
function refine(start: u32, end: u32, initial: vec3f64, mask: array<u32, 64>): vec3f64 {
  let X = initial;
  let value: f64 = selectedCost(start, end, X, mask),
    lambda: f64 = 1e-3;
  for (let it: u32 = 0; it < control.iterations; it++) {
    let A: array<f64, 9> = fill<f64, 9>(f64(0));
    let rhs: array<f64, 3> = fill<f64, 3>(f64(0));
    for (let i: u32 = start; i < end; i++) {
      if (mask[i - start] === 0) continue;
      const p = getPose(i),
        projection = project(getCamera(i), transform(p, X));
      if (projection.ok === 0) return X;
      for (let row: u32 = 0; row < 2; row++) {
        let g = projection.du;
        let e: f64 = projection.pixel.x - raw[i * 20 + 18];
        if (row === 1) {
          g = projection.dv;
          e = projection.pixel.y - raw[i * 20 + 19];
        }
        let J: array<f64, 3> = [
          g.x * p[0] + g.y * p[3] + g.z * p[6],
          g.x * p[1] + g.y * p[4] + g.z * p[7],
          g.x * p[2] + g.y * p[5] + g.z * p[8],
        ];
        for (let j: u32 = 0; j < 3; j++) {
          rhs[j] -= J[j] * e;
          for (let k: u32 = 0; k < 3; k++) A[j * 3 + k] += J[j] * J[k];
        }
      }
    }
    for (let j: u32 = 0; j < 3; j++) A[j * 3 + j] += lambda * max(A[j * 3 + j], f64(1e-10));
    const step = solve3(A, vec3f64(rhs[0], rhs[1], rhs[2]));
    if (step.ok === 0) break;
    const moved = X + step.values;
    const next: f64 = selectedCost(start, end, moved, mask);
    if (next < value) {
      const change: f64 = value - next;
      X = moved;
      value = next;
      lambda = max(f64(1e-10), lambda / 3);
      if (change < 1e-12 * max(f64(1), value)) break;
    } else lambda *= 10;
  }
  return X;
}
@compute([16])
export function triangulateTracks(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= statuses.length) return;
  const start: u32 = offsets[h],
    end: u32 = offsets[h + 1],
    n: u32 = end - start;
  statuses[h] = 1;
  for (let j: u32 = 0; j < 5; j++) result[h * 5 + j] = 0;
  for (let i: u32 = start; i < end; i++) flags[i] = 0;
  if (n < 2) return;
  statuses[h] = 3;
  let pairs: array<u32, 128> = array<u32, 128>();
  let baselines: array<f64, 64> = fill<f64, 64>(f64(0));
  let pairCount: u32 = 0;
  for (let i: u32 = start; i < end; i++)
    for (let j: u32 = i + 1; j < end; j++) {
      const d: f64 = length(center(i) - center(j));
      let place: u32 = pairCount;
      for (let k: u32 = 0; k < pairCount; k++)
        if (d > baselines[k]) {
          place = k;
          break;
        }
      if (place >= control.maxPairs) continue;
      if (pairCount < control.maxPairs) pairCount++;
      for (let k: u32 = pairCount - 1; k > place; k--) {
        baselines[k] = baselines[k - 1];
        pairs[k * 2] = pairs[(k - 1) * 2];
        pairs[k * 2 + 1] = pairs[(k - 1) * 2 + 1];
      }
      baselines[place] = d;
      pairs[place * 2] = i;
      pairs[place * 2 + 1] = j;
    }
  let bestPoint: vec3f64 = vec3f64(f64(0));
  let bestCount: u32 = 0;
  let bestCost: f64 = 1e30;
  let bestMask: array<u32, 64> = array<u32, 64>();
  for (let hypothesis: u32 = 0; hypothesis <= pairCount; hypothesis++) {
    let mode: u32 = 0,
      first: u32 = start,
      second: u32 = start;
    if (hypothesis > 0) {
      mode = 1;
      first = pairs[(hypothesis - 1) * 2];
      second = pairs[(hypothesis - 1) * 2 + 1];
    }
    const point = linear(start, end, mode, first, second, bestMask);
    if (point.w === 0) continue;
    let count: u32 = 0;
    let cost: f64 = 0;
    let mask: array<u32, 64> = array<u32, 64>();
    for (let i: u32 = start; i < end; i++) {
      const e: f64 = errorAt(i, point.xyz);
      if (e <= control.maxError * control.maxError) {
        count++;
        cost += e;
        mask[i - start] = 1;
      }
    }
    if (count > bestCount || (count === bestCount && cost < bestCost)) {
      bestPoint = point.xyz;
      bestCount = count;
      bestCost = cost;
      bestMask = mask;
    }
  }
  if (bestCount < 2) return;
  const fitted = linear(start, end, 2, start, start, bestMask);
  if (
    fitted.w !== 0 &&
    selectedCost(start, end, fitted.xyz, bestMask) < selectedCost(start, end, bestPoint, bestMask)
  )
    bestPoint = fitted.xyz;
  bestPoint = refine(start, end, bestPoint, bestMask);
  if (!(max(abs(bestPoint.x), max(abs(bestPoint.y), abs(bestPoint.z))) <= 1e9)) return;
  let finalMask: array<u32, 64> = array<u32, 64>();
  let accepted: u32 = 0;
  let totalError: f64 = 0,
    maxAngle: f64 = 0;
  for (let i: u32 = start; i < end; i++) {
    const e = errorAt(i, bestPoint);
    if (e <= control.maxError * control.maxError) {
      finalMask[i - start] = 1;
      accepted++;
      totalError += sqrt(e);
    }
  }
  if (accepted < 2) return;
  for (let i: u32 = start; i < end; i++)
    for (let j: u32 = i + 1; j < end; j++) {
      if (finalMask[i - start] !== 0 && finalMask[j - start] !== 0)
        maxAngle = max(maxAngle, angleBetween(bestPoint - center(i), bestPoint - center(j)));
    }
  if (maxAngle < control.minAngle) {
    statuses[h] = 4;
    return;
  }
  statuses[h] = 0;
  result[h * 5] = bestPoint.x;
  result[h * 5 + 1] = bestPoint.y;
  result[h * 5 + 2] = bestPoint.z;
  result[h * 5 + 3] = maxAngle;
  result[h * 5 + 4] = totalError / f64(f32(accepted));
  for (let i: u32 = start; i < end; i++) flags[i] = finalMask[i - start];
}
