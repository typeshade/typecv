'use typeshade';
import { CameraModel, project, transform, solve3 } from './numerics.shade.ts';

class ObservationControl {
  maxError: f64;
  maxRelativeDepthStdDev: f64;
  pixelNoiseStdDevPx: f64;
  observations: u32;
}
declare const control: uniform<ObservationControl>;
// Camera records contain R, t and intrinsics once per referenced view. Point coordinates are
// stored once per landmark; observation order remains the original landmark/measurement order.
declare const cameras: storage<array<f64>>;
declare const coordinates: storage<array<f64>>;
declare const pixels: storage<array<f64>>;
// Camera and point indices for each observation.
declare const indices: storage<array<u32>>;
declare const offsets: storage<array<u32>>;
declare const flags: storage<array<u32>, 'read_write'>;
declare const errors: storage<array<f64>, 'read_write'>;
declare const partials: storage<array<f64>, 'read_write'>;
let errorTile: workgroup<array<f64, 64>>;
let countTile: workgroup<array<u32, 64>>;

function cameraAt(i: u32): CameraModel {
  const at: u32 = indices[2 * i] * 18 + 12;
  return {
    fx: cameras[at], fy: cameras[at + 1], cx: cameras[at + 2], cy: cameras[at + 3],
    k1: cameras[at + 4], k2: cameras[at + 5],
  };
}
function poseAt(i: u32): array<f64, 12> {
  const at: u32 = indices[2 * i] * 18;
  let p: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) p[j] = cameras[at + j];
  return p;
}
function pointAt(i: u32): vec3f64 {
  const at: u32 = 3 * indices[2 * i + 1];
  return vec3f64(coordinates[at], coordinates[at + 1], coordinates[at + 2]);
}

@compute([64])
export function projectIndexedObservations(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= control.observations) return;
  flags[i] = 0;
  errors[i] = 0;
  const p = project(cameraAt(i), transform(poseAt(i), pointAt(i)));
  if (p.ok === 0) return;
  const x: f64 = p.pixel.x - pixels[2 * i], y: f64 = p.pixel.y - pixels[2 * i + 1];
  const e: f64 = sqrt(x * x + y * y);
  if (e <= control.maxError) { flags[i] = 1; errors[i] = e; }
}

// Keep the original fixed-pose/K depth-information accumulation and solve order exactly.
function stableDepth(start: u32, end: u32, X: vec3f64): bool {
  let A: array<f64, 9> = fill<f64, 9>(f64(0));
  let count: u32 = 0;
  for (let i: u32 = start; i < end; i++) {
    if (flags[i] === 0) continue;
    count++;
    const p = poseAt(i), projection = project(cameraAt(i), transform(p, X));
    if (projection.ok === 0) return false;
    for (let row: u32 = 0; row < 2; row++) {
      let g = projection.du;
      if (row === 1) g = projection.dv;
      const J: array<f64, 3> = [
        g.x*p[0] + g.y*p[3] + g.z*p[6],
        g.x*p[1] + g.y*p[4] + g.z*p[7],
        g.x*p[2] + g.y*p[5] + g.z*p[8],
      ];
      for (let r: u32 = 0; r < 3; r++) for (let c: u32 = 0; c < 3; c++) A[3*r+c] += J[r]*J[c];
    }
  }
  if (count < 2) return false;
  for (let i: u32 = start; i < end; i++) {
    if (flags[i] === 0) continue;
    const p = poseAt(i), axis = vec3f64(p[6], p[7], p[8]), q = transform(p, X), solved = solve3(A, axis);
    if (solved.ok === 0 || !(q.z > 0)) return false;
    const variance: f64 = dot(axis, solved.values);
    if (!(variance >= 0)) return false;
    const relative: f64 = control.pixelNoiseStdDevPx * sqrt(variance) / q.z;
    if (!(relative <= control.maxRelativeDepthStdDev)) return false;
  }
  return true;
}

@compute([64])
export function filterIndexedObservationDepth(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h + 1 >= offsets.length) return;
  const start: u32 = offsets[h], end: u32 = offsets[h+1];
  flags[control.observations+h] = 0;
  if (end <= start) return;
  let accepted: u32 = 0;
  for (let i: u32 = start; i < end; i++) if (flags[i] !== 0) accepted++;
  if (accepted < 2) return;
  if (!stableDepth(start, end, pointAt(start))) {
    flags[control.observations+h] = 1;
    for (let i: u32 = start; i < end; i++) { flags[i] = 0; errors[i] = 0; }
  }
}

function observationLane(i: u32): vec2f64 {
  if (i >= control.observations || flags[i] === 0) return vec2f64(f64(0));
  return vec2f64(errors[i], f64(1));
}
@compute([64])
export function indexedObservationPartials(
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
export function indexedObservationPartialsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= partials.length / 2) return;
  let total: vec2f64 = vec2f64(f64(0));
  for (let lane: u32 = 0; lane < 64; lane++) total += observationLane(block * 64 + lane);
  partials[block * 2] = total.x;
  partials[block * 2 + 1] = total.y;
}
