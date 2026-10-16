'use typeshade';
import { CameraModel, project, transform, robustCost } from './numerics.shade.ts';

class FocalControl {
  observations: u32;
  huber: f64;
}
declare const control: uniform<FocalControl>;
// Each observation: pose[12], XYZ[3], intrinsics[6], measured pixel[2].
declare const raw: storage<array<f64>>;
declare const scales: storage<array<f64>>;
declare const costs: storage<array<f64>, 'read_write'>;
declare const totals: storage<array<f64>, 'read_write'>;

@compute([64])
export function scoreFocalCandidates(@builtin('global_invocation_id') gid: vec3u): void {
  const i = gid.x;
  if (i >= control.observations * 9) return;
  const candidate = i / control.observations,
    observation = i % control.observations,
    at = observation * 23;
  if (scales[candidate] <= 0) {
    costs[i] = -1;
    return;
  }
  let pose: array<f64, 12> = fill<f64, 12>(f64(0));
  for (let j: u32 = 0; j < 12; j++) pose[j] = raw[at + j];
  const camera: CameraModel = {
    fx: raw[at + 15] * scales[candidate],
    fy: raw[at + 16] * scales[candidate],
    cx: raw[at + 17],
    cy: raw[at + 18],
    k1: raw[at + 19],
    k2: raw[at + 20],
  };
  const pixel = project(camera, transform(pose, vec3f64(raw[at + 12], raw[at + 13], raw[at + 14])));
  if (pixel.ok === 0) {
    costs[i] = -1;
    return;
  }
  const du = pixel.pixel.x - raw[at + 21],
    dv = pixel.pixel.y - raw[at + 22];
  costs[i] = robustCost(sqrt(du * du + dv * dv), control.huber);
}

@compute([1])
export function sumFocalCandidates(@builtin('global_invocation_id') gid: vec3u): void {
  const candidate = gid.x;
  if (candidate >= 9) return;
  let total: f64 = 0;
  for (let i: u32 = 0; i < control.observations; i++) {
    const cost = costs[candidate * control.observations + i];
    if (cost < 0) {
      totals[candidate] = -1;
      return;
    }
    total += cost;
  }
  totals[candidate] = total;
}
