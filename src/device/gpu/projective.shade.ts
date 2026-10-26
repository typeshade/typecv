'use typeshade';

// Streaming Givens QR of the design rows. QR bounds local storage for arbitrary consensus
// sizes without forming the normal matrix A^T A; `pair-models.shade.ts` solves the factor.
export class Null9 {
  vector: array<f64, 9>;
  second: f64;
  largest: f64;
  ok: u32;
}
export function appendRow(qr: array<f64, 81>, input: array<f64, 9>): array<f64, 81> {
  let out = qr;
  let row = input;
  for (let j: u32 = 0; j < 9; j++) {
    const a: f64 = out[j * 9 + j],
      b: f64 = row[j];
    const scale: f64 = max(abs(a), abs(b));
    if (scale === 0) continue;
    const length: f64 = scale * sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale));
    const c: f64 = a / length,
      s: f64 = b / length;
    out[j * 9 + j] = length;
    for (let k: u32 = j + 1; k < 9; k++) {
      const x: f64 = out[j * 9 + k],
        y: f64 = row[k];
      out[j * 9 + k] = c * x + s * y;
      row[k] = c * y - s * x;
    }
  }
  return out;
}

export function cross3(a: vec3f64, b: vec3f64): vec3f64 {
  return vec3f64(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
export class Basis3 {
  U: array<f64, 9>;
  V: array<f64, 9>;
  ok: u32;
}
// Two leading singular directions, completed to right-handed U and V. The
// essential projection is (u0 v0^T + u1 v1^T)/sqrt(2); no small singular division.
export function basis3(input: array<f64, 9>): Basis3 {
  let a = input;
  let v: array<f64, 9> = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  let zero: array<f64, 9> = fill<f64, 9>(f64(0));
  const bad: Basis3 = { U: zero, V: zero, ok: 0 };
  for (let sweep: u32 = 0; sweep < 64; sweep++) {
    let changed: bool = false;
    for (let p: u32 = 0; p < 3; p++)
      for (let q: u32 = p + 1; q < 3; q++) {
        let aa: f64 = 0,
          bb: f64 = 0,
          ab: f64 = 0;
        for (let r: u32 = 0; r < 3; r++) {
          const x = a[r * 3 + p],
            y = a[r * 3 + q];
          aa += x * x;
          bb += y * y;
          ab += x * y;
        }
        if (aa < 1e-28 || bb < 1e-28 || abs(ab) <= 1e-12 * sqrt(aa * bb)) continue;
        const tau: f64 = (bb - aa) / (2 * ab),
          at: f64 = abs(tau);
        let hyp: f64 = sqrt(1 + tau * tau),
          sign: f64 = 1;
        if (at > 1) hyp = at * sqrt(1 + (1 / at) * (1 / at));
        if (tau < 0) sign = -1;
        const t: f64 = sign / (at + hyp),
          c: f64 = 1 / sqrt(1 + t * t),
          s: f64 = c * t;
        for (let r: u32 = 0; r < 3; r++) {
          const x = a[r * 3 + p],
            y = a[r * 3 + q];
          a[r * 3 + p] = c * x - s * y;
          a[r * 3 + q] = s * x + c * y;
          const vx = v[r * 3 + p],
            vy = v[r * 3 + q];
          v[r * 3 + p] = c * vx - s * vy;
          v[r * 3 + q] = s * vx + c * vy;
        }
        changed = true;
      }
    if (!changed) break;
  }
  let n: array<f64, 3> = [0, 0, 0];
  let order: array<u32, 3> = [0, 1, 2];
  for (let j: u32 = 0; j < 3; j++)
    for (let r: u32 = 0; r < 3; r++) n[j] += a[r * 3 + j] * a[r * 3 + j];
  for (let j: u32 = 0; j < 2; j++)
    for (let k: u32 = j + 1; k < 3; k++)
      if (n[order[k]] > n[order[j]]) {
        const tmp = order[j];
        order[j] = order[k];
        order[k] = tmp;
      }
  const j: u32 = order[0],
    k: u32 = order[1];
  if (!(n[j] > 1e-28 && n[k] > 1e-24 * n[j])) return bad;
  const v0 = normalize(vec3f64(v[j], v[3 + j], v[6 + j]));
  const vraw = vec3f64(v[k], v[3 + k], v[6 + k]);
  const v1 = normalize(vraw - dot(vraw, v0) * v0),
    v2 = cross3(v0, v1);
  const u0 = normalize(vec3f64(a[j], a[3 + j], a[6 + j]));
  const uraw = vec3f64(a[k], a[3 + k], a[6 + k]);
  const u1 = normalize(uraw - dot(uraw, u0) * u0),
    u2 = cross3(u0, u1);
  let U: array<f64, 9> = [u0.x, u1.x, u2.x, u0.y, u1.y, u2.y, u0.z, u1.z, u2.z];
  let V: array<f64, 9> = [v0.x, v1.x, v2.x, v0.y, v1.y, v2.y, v0.z, v1.z, v2.z];
  return { U, V, ok: 1 };
}

export function inverse3(a: array<f64, 9>): array<f64, 10> {
  let out: array<f64, 10> = fill<f64, 10>(f64(0));
  out[0] = a[4] * a[8] - a[5] * a[7];
  out[1] = a[2] * a[7] - a[1] * a[8];
  out[2] = a[1] * a[5] - a[2] * a[4];
  out[3] = a[5] * a[6] - a[3] * a[8];
  out[4] = a[0] * a[8] - a[2] * a[6];
  out[5] = a[2] * a[3] - a[0] * a[5];
  out[6] = a[3] * a[7] - a[4] * a[6];
  out[7] = a[1] * a[6] - a[0] * a[7];
  out[8] = a[0] * a[4] - a[1] * a[3];
  const d: f64 = a[0] * out[0] + a[1] * out[3] + a[2] * out[6];
  // Hartley-normalized H is tested before denormalization by the caller.
  if (!(abs(d) > 1e-30 && abs(d) < 1e30)) return out;
  for (let j: u32 = 0; j < 9; j++) out[j] /= d;
  out[9] = 1;
  return out;
}

// Direct 4x4 Jacobi SVD for two-camera DLT, with row normalization.
export function triangulatePair(xy: vec4f64, p: array<f64, 12>): vec4f64 {
  let a: array<f64, 16> = [
    -1,
    0,
    xy.x,
    0,
    0,
    -1,
    xy.y,
    0,
    xy.z * p[6] - p[0],
    xy.z * p[7] - p[1],
    xy.z * p[8] - p[2],
    xy.z * p[11] - p[9],
    xy.w * p[6] - p[3],
    xy.w * p[7] - p[4],
    xy.w * p[8] - p[5],
    xy.w * p[11] - p[10],
  ];
  let v: array<f64, 16> = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let r: u32 = 0; r < 4; r++) {
    let norm: f64 = 0;
    for (let j: u32 = 0; j < 4; j++) norm += a[r * 4 + j] * a[r * 4 + j];
    if (!(norm > 1e-30)) return vec4f64(f64(0));
    for (let j: u32 = 0; j < 4; j++) a[r * 4 + j] /= sqrt(norm);
  }
  let converged: bool = false;
  for (let sweep: u32 = 0; sweep < 160; sweep++) {
    let changed: bool = false;
    for (let j: u32 = 0; j < 4; j++)
      for (let k: u32 = j + 1; k < 4; k++) {
        let aa: f64 = 0,
          bb: f64 = 0,
          ab: f64 = 0;
        for (let r: u32 = 0; r < 4; r++) {
          const x = a[r * 4 + j],
            y = a[r * 4 + k];
          aa += x * x;
          bb += y * y;
          ab += x * y;
        }
        if (aa < 1e-28 || bb < 1e-28 || abs(ab) <= 1e-12 * sqrt(aa * bb)) continue;
        const tau: f64 = (bb - aa) / (2 * ab),
          at: f64 = abs(tau);
        let hyp: f64 = sqrt(1 + tau * tau),
          sign: f64 = 1;
        if (at > 1) hyp = at * sqrt(1 + (1 / at) * (1 / at));
        if (tau < 0) sign = -1;
        const t: f64 = sign / (at + hyp),
          c: f64 = 1 / sqrt(1 + t * t),
          s: f64 = t * c;
        for (let r: u32 = 0; r < 4; r++) {
          const x = a[r * 4 + j],
            y = a[r * 4 + k];
          a[r * 4 + j] = c * x - s * y;
          a[r * 4 + k] = s * x + c * y;
          const vx = v[r * 4 + j],
            vy = v[r * 4 + k];
          v[r * 4 + j] = c * vx - s * vy;
          v[r * 4 + k] = s * vx + c * vy;
        }
        changed = true;
      }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) return vec4f64(f64(0));
  let norms: array<f64, 4> = [0, 0, 0, 0];
  for (let j: u32 = 0; j < 4; j++)
    for (let r: u32 = 0; r < 4; r++) norms[j] += a[r * 4 + j] * a[r * 4 + j];
  let index: u32 = 0;
  for (let j: u32 = 1; j < 4; j++) if (norms[j] < norms[index]) index = j;
  let second: f64 = 1e30,
    largest: f64 = 0;
  for (let j: u32 = 0; j < 4; j++) {
    if (j !== index) second = min(second, norms[j]);
    largest = max(largest, norms[j]);
  }
  if (second < 1e-20 * largest || abs(v[12 + index]) < 1e-12) return vec4f64(f64(0));
  const X = vec3f64(v[index], v[4 + index], v[8 + index]) / v[12 + index];
  if (!(max(abs(X.x), max(abs(X.y), abs(X.z))) <= 1e9)) return vec4f64(f64(0));
  return vec4f64(X, f64(1));
}
