'use typeshade';
/** Fixed charts and poses are packed as 12 floats (R,t); seven calibrated camera floats. */
declare const idMeasurements: storage<array<f64>>;
declare const idLinearizations: storage<array<f64>, 'read_write'>;
class IdControl { count: u32; }
declare const idControl: uniform<IdControl>;

/** One observation/lane. Host f64 Schur assembly/covariance consumes these derivatives. */
@compute([64])
export function linearizeInverseDepth(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= idControl.count) return;
  const base: u32 = i * 34;
  const out: u32 = i * 23;
  for (let k: u32 = 0; k < 23; k++) idLinearizations[out + k] = 0.;
  const a: f64 = idMeasurements[base];
  const b: f64 = idMeasurements[base + 1];
  const rho: f64 = idMeasurements[base + 2];
  // R_ref begins at 3, t_ref at 12; observed R at 15, t at 24; K at 27.
  let cx: f64 = 0.;
  let cy: f64 = 0.;
  let cz: f64 = 0.;
  for (let k: u32 = 0; k < 3; k++) {
    cx -= idMeasurements[base + 3 + 3 * k] * idMeasurements[base + 12 + k];
    cy -= idMeasurements[base + 4 + 3 * k] * idMeasurements[base + 12 + k];
    cz -= idMeasurements[base + 5 + 3 * k] * idMeasurements[base + 12 + k];
  }
  const C: vec3f64 = vec3f64(cx, cy, cz);
  let c0: array<f64, 3> = fill<f64, 3>(f64(0));
  let c1: array<f64, 3> = fill<f64, 3>(f64(0));
  let c2: array<f64, 3> = fill<f64, 3>(f64(0));
  let tr: array<f64, 3> = fill<f64, 3>(f64(0));
  for (let row: u32 = 0; row < 3; row++) {
    const R: vec3f64 = vec3f64(idMeasurements[base + 15 + 3 * row], idMeasurements[base + 16 + 3 * row], idMeasurements[base + 17 + 3 * row]);
    c0[row] = dot(R, vec3f64(idMeasurements[base + 3], idMeasurements[base + 4], idMeasurements[base + 5]));
    c1[row] = dot(R, vec3f64(idMeasurements[base + 6], idMeasurements[base + 7], idMeasurements[base + 8]));
    c2[row] = dot(R, vec3f64(idMeasurements[base + 9], idMeasurements[base + 10], idMeasurements[base + 11]));
    tr[row] = dot(R, C) + idMeasurements[base + 24 + row];
  }
  const A0: vec3f64 = vec3f64(c0[0], c0[1], c0[2]);
  const A1: vec3f64 = vec3f64(c1[0], c1[1], c1[2]);
  const A2: vec3f64 = vec3f64(c2[0], c2[1], c2[2]);
  const T: vec3f64 = vec3f64(tr[0], tr[1], tr[2]);
  const q: vec3f64 = a * A0 + b * A1 + A2 + rho * T;
  if (!(q.z > 0.000000000001)) return;
  const x: f64 = q.x / q.z;
  const y: f64 = q.y / q.z;
  const r2: f64 = x * x + y * y;
  const fx: f64 = idMeasurements[base + 27];
  const fy: f64 = idMeasurements[base + 28];
  const k1: f64 = idMeasurements[base + 31];
  const k2: f64 = idMeasurements[base + 32];
  if (!(r2 < idMeasurements[base + 33])) return;
  const d: f64 = 1. + k1 * r2 + k2 * r2 * r2;
  if (!(d > 0. && 1. + 3. * k1 * r2 + 5. * k2 * r2 * r2 > 0.)) return;
  const g: f64 = 2. * (k1 + 2. * k2 * r2);
  const xx: f64 = d + g * x * x;
  const xy: f64 = g * x * y;
  const yy: f64 = d + g * y * y;
  const J0: vec3f64 = vec3f64(fx * xx / q.z, fx * xy / q.z, -fx * (xx * x + xy * y) / q.z);
  const J1: vec3f64 = vec3f64(fy * xy / q.z, fy * yy / q.z, -fy * (xy * x + yy * y) / q.z);
  idLinearizations[out] = 1.;
  idLinearizations[out + 1] = fx * x * d + idMeasurements[base + 29];
  idLinearizations[out + 2] = fy * y * d + idMeasurements[base + 30];
  for (let row: u32 = 0; row < 2; row++) {
    let J: vec3f64 = J0;
    if (row === 1) J = J1;
    idLinearizations[out + 3 + row * 3] = dot(J, A0);
    idLinearizations[out + 4 + row * 3] = dot(J, A1);
    idLinearizations[out + 5 + row * 3] = dot(J, T);
    const at: u32 = out + 9 + row * 6;
    idLinearizations[at] = -J.y * q.z + J.z * q.y;
    idLinearizations[at + 1] = J.x * q.z - J.z * q.x;
    idLinearizations[at + 2] = -J.x * q.y + J.y * q.x;
    idLinearizations[at + 3] = rho * J.x;
    idLinearizations[at + 4] = rho * J.y;
    idLinearizations[at + 5] = rho * J.z;
  }
  idLinearizations[out + 21] = fx * x * d;
  idLinearizations[out + 22] = fy * y * d;
}
