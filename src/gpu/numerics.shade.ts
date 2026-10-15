'use typeshade';

export class CameraModel {
  fx: f64;
  fy: f64;
  cx: f64;
  cy: f64;
  k1: f64;
  k2: f64;
}
export class Projection {
  pixel: vec2f64;
  du: vec3f64;
  dv: vec3f64;
  ok: u32;
}
export class Linear6 {
  values: array<f64, 6>;
  ok: u32;
}
export class Linear3 {
  values: vec3f64;
  ok: u32;
}
export function solve3(a: array<f64, 9>, b: vec3f64): Linear3 {
  const zero: Linear3 = { values: vec3f64(f64(0)), ok: 0 };
  if (!(min(a[0], min(a[4], a[8])) > 0)) return zero;
  const d: vec3f64 = vec3f64(1 / sqrt(a[0]), 1 / sqrt(a[4]), 1 / sqrt(a[8]));
  const l10: f64 = a[3] * d.x * d.y,
    l20: f64 = a[6] * d.x * d.z;
  const pivot1: f64 = 1 - l10 * l10;
  if (!(pivot1 > 1e-14)) return zero;
  const l11: f64 = sqrt(pivot1),
    l21: f64 = (a[7] * d.y * d.z - l20 * l10) / l11;
  const pivot2: f64 = 1 - l20 * l20 - l21 * l21;
  if (!(pivot2 > 1e-14)) return zero;
  const l22: f64 = sqrt(pivot2),
    y0: f64 = b.x * d.x;
  const y1: f64 = (b.y * d.y - l10 * y0) / l11,
    y2: f64 = (b.z * d.z - l20 * y0 - l21 * y1) / l22;
  const x2: f64 = y2 / l22,
    x1: f64 = (y1 - l21 * x2) / l11,
    x0: f64 = y0 - l10 * x1 - l20 * x2;
  return { values: vec3f64(x0 * d.x, x1 * d.y, x2 * d.z), ok: 1 };
}
function smallAtan(x: f64): f64 {
  let y: f64 = x,
    offset: f64 = 0;
  if (x > 0.4142135623730951) {
    y = (x - 1) / (x + 1);
    offset = 0.7853981633974483;
  }
  let value: f64 = 0,
    term: f64 = y;
  for (let j: u32 = 0; j < 24; j++) {
    value += term / f64(f32(2 * j + 1));
    term *= -y * y;
  }
  return offset + value;
}
export function angleBetween(u: vec3f64, v: vec3f64): f64 {
  const nu: f64 = length(u),
    nv: f64 = length(v);
  if (!(nu > 1e-12 && nv > 1e-12)) return -1;
  const a = u / nu,
    b = v / nv;
  const crossLength: f64 = length(
    vec3f64(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x),
  );
  const cosine: f64 = dot(a, b),
    c: f64 = abs(cosine);
  let angle: f64 = 0;
  if (c > crossLength) angle = smallAtan(crossLength / c);
  else if (crossLength > 0) angle = 1.5707963267948966 - smallAtan(c / crossLength);
  if (cosine < 0) angle = 3.141592653589793 - angle;
  return angle * 57.29577951308232;
}

export function radialLimit(k: CameraModel): f64 {
  if (k.k2 === 0) {
    if (k.k1 < 0) return -1 / (3 * k.k1);
    return 1e30;
  }
  const a: f64 = 5 * k.k2,
    b: f64 = 3 * k.k1;
  const d: f64 = b * b - 4 * a;
  if (d < 0) return 1e30;
  let sign: f64 = 1;
  if (b < 0) sign = -1;
  const q: f64 = -0.5 * (b + sign * sqrt(d));
  let limit: f64 = 1e30;
  const first: f64 = q / a;
  if (first > 0) limit = min(limit, first);
  if (q !== 0) {
    const second: f64 = 1 / q;
    if (second > 0) limit = min(limit, second);
  }
  return limit;
}

// xy and validity. Invert only the radial branch connected to the optical axis.
export function unproject(k: CameraModel, pixel: vec2f64): vec3f64 {
  const xd: f64 = (pixel.x - k.cx) / k.fx,
    yd: f64 = (pixel.y - k.cy) / k.fy;
  const rd: f64 = sqrt(xd * xd + yd * yd),
    limit: f64 = radialLimit(k);
  if (rd === 0) return vec3f64(f64(0), f64(0), f64(1));
  if (limit < 1e30 && rd >= sqrt(limit) * (1 + k.k1 * limit + k.k2 * limit * limit))
    return vec3f64(f64(0));
  let r: f64 = rd;
  for (let it: u32 = 0; it < 50; it++) {
    const r2: f64 = r * r,
      d: f64 = 1 + k.k1 * r2 + k.k2 * r2 * r2,
      slope: f64 = 1 + 3 * k.k1 * r2 + 5 * k.k2 * r2 * r2;
    if (!(slope > 1e-10 && d > 0)) return vec3f64(f64(0));
    const residual: f64 = r * d - rd;
    if (abs(residual) <= 1e-12 * max(f64(1), rd))
      return vec3f64((xd * r) / rd, (yd * r) / rd, f64(1));
    let step: f64 = residual / slope;
    let accepted: bool = false;
    for (let j: u32 = 0; j < 25; j++) {
      const next: f64 = r - step,
        n2: f64 = next * next;
      if (
        next >= 0 &&
        n2 < limit &&
        1 + 3 * k.k1 * n2 + 5 * k.k2 * n2 * n2 > 0 &&
        abs(next * (1 + k.k1 * n2 + k.k2 * n2 * n2) - rd) < abs(residual)
      ) {
        r = next;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) return vec3f64(f64(0));
  }
  return vec3f64(f64(0));
}

export function project(k: CameraModel, q: vec3f64): Projection {
  const zero: vec3f64 = vec3f64(f64(0));
  const bad: Projection = { pixel: vec2f64(f64(0)), du: zero, dv: zero, ok: 0 };
  if (!(q.z > 1e-12)) return bad;
  const x: f64 = q.x / q.z,
    y: f64 = q.y / q.z,
    r2: f64 = x * x + y * y;
  if (!(r2 < radialLimit(k) && r2 < 1e8)) return bad;
  const d: f64 = 1 + k.k1 * r2 + k.k2 * r2 * r2,
    g: f64 = 2 * (k.k1 + 2 * k.k2 * r2);
  if (!(d > 0 && 1 + 3 * k.k1 * r2 + 5 * k.k2 * r2 * r2 > 0)) return bad;
  const xx: f64 = d + g * x * x,
    xy: f64 = g * x * y,
    yy: f64 = d + g * y * y;
  const pixel: vec2f64 = vec2f64(k.fx * x * d + k.cx, k.fy * y * d + k.cy);
  const du: vec3f64 = vec3f64(
    (k.fx * xx) / q.z,
    (k.fx * xy) / q.z,
    (-k.fx * (xx * x + xy * y)) / q.z,
  );
  const dv: vec3f64 = vec3f64(
    (k.fy * xy) / q.z,
    (k.fy * yy) / q.z,
    (-k.fy * (xy * x + yy * y)) / q.z,
  );
  if (!(
    max(abs(pixel.x), abs(pixel.y)) < 1e12 &&
    max(max(abs(du.x), abs(du.y)), abs(du.z)) < 1e12 &&
    max(max(abs(dv.x), abs(dv.y)), abs(dv.z)) < 1e12
  ))
    return bad;
  return { pixel, du, dv, ok: 1 };
}
export function transform(p: array<f64, 12>, X: vec3f64): vec3f64 {
  return vec3f64(
    p[0] * X.x + p[1] * X.y + p[2] * X.z + p[9],
    p[3] * X.x + p[4] * X.y + p[5] * X.z + p[10],
    p[6] * X.x + p[7] * X.y + p[8] * X.z + p[11],
  );
}
export function poseRow(q: vec3f64, j: vec3f64): array<f64, 6> {
  return [j.z * q.y - j.y * q.z, j.x * q.z - j.z * q.x, j.y * q.x - j.x * q.y, j.x, j.y, j.z];
}
export function robustCost(error: f64, huber: f64): f64 {
  if (error <= huber) return error * error;
  return 2 * huber * error - huber * huber;
}

// Diagonal equilibration followed by Cholesky; damping makes the system SPD.
export function solve6(matrix: array<f64, 36>, rhs: array<f64, 6>): Linear6 {
  let L: array<f64, 36> = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
  ];
  let scale: array<f64, 6> = [0, 0, 0, 0, 0, 0];
  let values: array<f64, 6> = [0, 0, 0, 0, 0, 0];
  for (let i: u32 = 0; i < 6; i++) {
    if (!(matrix[i * 6 + i] > 0)) return { values, ok: 0 };
    scale[i] = 1 / sqrt(matrix[i * 6 + i]);
  }
  for (let i: u32 = 0; i < 6; i++) {
    for (let j: u32 = 0; j <= i; j++) {
      let value: f64 = matrix[i * 6 + j] * scale[i] * scale[j];
      for (let k: u32 = 0; k < j; k++) value -= L[i * 6 + k] * L[j * 6 + k];
      if (i === j) {
        if (!(value > 1e-14)) return { values, ok: 0 };
        L[i * 6 + j] = sqrt(value);
      } else L[i * 6 + j] = value / L[j * 6 + j];
    }
    let v: f64 = rhs[i] * scale[i];
    for (let j: u32 = 0; j < i; j++) v -= L[i * 6 + j] * values[j];
    values[i] = v / L[i * 6 + i];
  }
  for (let index: u32 = 6; index > 0; index--) {
    const i: u32 = index - 1;
    let v: f64 = values[i];
    for (let j: u32 = i + 1; j < 6; j++) v -= L[j * 6 + i] * values[j];
    values[i] = v / L[i * 6 + i];
  }
  for (let i: u32 = 0; i < 6; i++) values[i] *= scale[i];
  return { values, ok: 1 };
}
export function movePose(p: array<f64, 12>, delta: array<f64, 6>): array<f64, 12> {
  const x: f64 = delta[0],
    y: f64 = delta[1],
    z: f64 = delta[2];
  const theta: f64 = sqrt(x * x + y * y + z * z);
  let a: f64 = 1 - (theta * theta) / 6,
    b: f64 = 0.5 - (theta * theta) / 24;
  if (theta >= 1e-7) {
    a = sin(theta) / theta;
    b = (1 - cos(theta)) / (theta * theta);
  }
  let W: array<f64, 9> = [0, -z, y, z, 0, -x, -y, x, 0];
  let dR: array<f64, 9> = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  let out: array<f64, 12> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i: u32 = 0; i < 3; i++)
    for (let j: u32 = 0; j < 3; j++) {
      dR[i * 3 + j] += a * W[i * 3 + j];
      for (let k: u32 = 0; k < 3; k++) dR[i * 3 + j] += b * W[i * 3 + k] * W[k * 3 + j];
    }
  for (let i: u32 = 0; i < 3; i++) {
    out[9 + i] = delta[3 + i];
    for (let k: u32 = 0; k < 3; k++) {
      out[9 + i] += dR[i * 3 + k] * p[9 + k];
      for (let j: u32 = 0; j < 3; j++) out[i * 3 + j] += dR[i * 3 + k] * p[k * 3 + j];
    }
  }
  return out;
}
