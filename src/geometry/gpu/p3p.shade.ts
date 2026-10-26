'use typeshade';

// Each invocation solves one complete Grunert problem, including every real branch.
declare const problems: storage<array<f64>>;
declare const poses: storage<array<f64>, 'read_write'>;
declare const valid: storage<array<u32>, 'read_write'>;
declare const counts: storage<array<u32>, 'read_write'>;
declare const status: storage<array<u32>, 'read_write'>;

class RootSet {
  values: array<f64, 8>;
  count: u32;
  overflow: u32;
}
class Basis {
  u: vec3f64;
  v: vec3f64;
  w: vec3f64;
  ok: u32;
}

// p(x) / max(1, abs(x)^degree). The sign is preserved without overflowing
// TypeShade f64's f32 exponent range at the Cauchy bound.
function scaledValue(a: array<f64, 5>, degree: u32, x: f64): f64 {
  let result: f64 = 0;
  if (abs(x) > 1) {
    result = a[0];
    for (let j: u32 = 1; j <= degree; j++) result = result / x + a[j];
    if (x < 0 && degree % 2 === 1) result = -result;
  } else {
    result = a[degree];
    for (let j: u32 = degree; j > 0; j--) result = result * x + a[j - 1];
  }
  return result;
}

function isolate(a: array<f64, 5>, degree: u32, critical: RootSet): RootSet {
  let values: array<f64, 8> = [0, 0, 0, 0, 0, 0, 0, 0];
  let cuts: array<f64, 10> = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let count: u32 = 0;
  let overflow: u32 = critical.overflow;
  let bound: f64 = 0;
  for (let j: u32 = 0; j < degree; j++) bound = max(bound, abs(a[j] / a[degree]));
  bound += 1;
  cuts[0] = -bound;
  let cutCount: u32 = 1;
  for (let j: u32 = 0; j < critical.count; j++) {
    const x: f64 = critical.values[j];
    if (x > -bound && x < bound) {
      cuts[cutCount] = x;
      cutCount++;
      if (abs(scaledValue(a, degree, x)) < 1e-12) {
        if (count < 8) {
          values[count] = x;
          count++;
        } else overflow = 1;
      }
    }
  }
  cuts[cutCount] = bound;
  cutCount++;
  for (let j: u32 = 1; j < cutCount; j++) {
    let lo: f64 = cuts[j - 1];
    let hi: f64 = cuts[j];
    let left: f64 = scaledValue(a, degree, lo);
    const right: f64 = scaledValue(a, degree, hi);
    if ((left >= 0 && right >= 0) || (left <= 0 && right <= 0)) continue;
    for (let iteration: u32 = 0; iteration < 100; iteration++) {
      const mid: f64 = (lo + hi) * 0.5;
      const v: f64 = scaledValue(a, degree, mid);
      if ((left < 0 && v >= 0) || (left > 0 && v <= 0)) hi = mid;
      else {
        lo = mid;
        left = v;
      }
      if (hi - lo < 1e-13 * max(f64(1), abs(mid))) break;
    }
    if (count < 8) {
      values[count] = (lo + hi) * 0.5;
      count++;
    } else overflow = 1;
  }
  // Sort and merge before these roots become the next derivative's cut points.
  for (let j: u32 = 1; j < count; j++) {
    for (let k: u32 = j; k > 0; k--) {
      if (values[k] < values[k - 1]) {
        const swap: f64 = values[k];
        values[k] = values[k - 1];
        values[k - 1] = swap;
      }
    }
  }
  let unique: u32 = 0;
  let previous: f64 = 0;
  for (let j: u32 = 0; j < count; j++) {
    const x: f64 = values[j];
    if (j === 0 || abs(x - previous) > 1e-8 * max(f64(1), abs(x))) {
      values[unique] = x;
      unique++;
    }
    previous = x;
  }
  return { values, count: unique, overflow };
}

function realRoots(polynomial: array<f64, 5>): RootSet {
  let coefficients: array<f64, 25> = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ];
  let values: array<f64, 8> = [0, 0, 0, 0, 0, 0, 0, 0];
  let result: RootSet = { values, count: 0, overflow: 0 };
  let scale: f64 = 0;
  for (let j: u32 = 0; j < 5; j++) scale = max(scale, abs(polynomial[j]));
  if (!(scale > 0)) return result;
  let degree: u32 = 4;
  for (let j: u32 = 4; j > 0; j--) {
    if (abs(polynomial[j] / scale) < 1e-14) degree--;
    else break;
  }
  if (degree === 0) return result;
  for (let j: u32 = 0; j <= degree; j++) coefficients[degree * 5 + j] = polynomial[j] / scale;
  for (let d: u32 = degree; d > 1; d--) {
    for (let j: u32 = 1; j <= d; j++)
      coefficients[(d - 1) * 5 + j - 1] = f64(f32(j)) * coefficients[d * 5 + j];
  }
  values[0] = -coefficients[5] / coefficients[6];
  result = { values, count: 1, overflow: 0 };
  for (let d: u32 = 2; d <= degree; d++) {
    let row: array<f64, 5> = [0, 0, 0, 0, 0];
    let rowScale: f64 = 0;
    for (let j: u32 = 0; j <= d; j++) rowScale = max(rowScale, abs(coefficients[d * 5 + j]));
    for (let j: u32 = 0; j <= d; j++) row[j] = coefficients[d * 5 + j] / rowScale;
    result = isolate(row, d, result);
  }
  return result;
}

function crossWide(a: vec3f64, b: vec3f64): vec3f64 {
  return vec3f64(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
function makeBasis(p0: vec3f64, p1: vec3f64, p2: vec3f64): Basis {
  let u: vec3f64 = p1 - p0;
  let v: vec3f64 = p2 - p0;
  const un: f64 = length(u);
  const vn: f64 = length(v);
  if (!(un > 1e-12 && vn > 1e-12)) return { u, v, w: vec3f64(f64(0)), ok: 0 };
  u = u / un;
  v = v - u * dot(u, v);
  const n: f64 = length(v);
  if (!(n >= 1e-8 * vn)) return { u, v, w: vec3f64(f64(0)), ok: 0 };
  v = v / n;
  return { u, v, w: crossWide(u, v), ok: 1 };
}
function rotatePoint(camera: Basis, world: Basis, p: vec3f64): vec3f64 {
  return camera.u * dot(world.u, p) + camera.v * dot(world.v, p) + camera.w * dot(world.w, p);
}
function refineLengths(
  initial: vec3f64,
  alpha: f64,
  beta: f64,
  gamma: f64,
  A: f64,
  B: f64,
): vec3f64 {
  let lengths: vec3f64 = initial;
  for (let iteration: u32 = 0; iteration < 12; iteration++) {
    const u: f64 = lengths.x;
    const v: f64 = lengths.y;
    const w: f64 = lengths.z;
    const r0: f64 = u * u + v * v - 2 * gamma * u * v - 1;
    const r1: f64 = u * u + w * w - 2 * beta * u * w - B;
    const r2: f64 = v * v + w * w - 2 * alpha * v * w - A;
    if (max(abs(r0), max(abs(r1), abs(r2))) < 1e-12) break;
    const a: f64 = 2 * (u - gamma * v);
    const b: f64 = 2 * (v - gamma * u);
    const c: f64 = 2 * (u - beta * w);
    const d: f64 = 2 * (w - beta * u);
    const e: f64 = 2 * (v - alpha * w);
    const f: f64 = 2 * (w - alpha * v);
    const det: f64 = -a * d * e - b * c * f;
    const s: f64 = max(max(abs(a), abs(b)), max(max(abs(c), abs(d)), max(abs(e), abs(f))));
    if (!(abs(det) > 1e-14 * s * s * s)) break;
    const step: vec3f64 = vec3f64(
      (d * e * r0 + b * f * r1 - b * d * r2) / det,
      (c * f * r0 - a * f * r1 + a * d * r2) / det,
      (-c * e * r0 + a * e * r1 + b * c * r2) / det,
    );
    const next: vec3f64 = lengths + step;
    if (!(min(next.x, min(next.y, next.z)) > 0 && max(next.x, max(next.y, next.z)) < 1e9)) break;
    lengths = next;
  }
  return lengths;
}

@compute([32])
export function solveP3p(@builtin('global_invocation_id') gid: vec3u): void {
  const h: u32 = gid.x;
  if (h >= counts.length) return;
  const input: u32 = h * 18;
  const output: u32 = h * 48;
  counts[h] = 0;
  status[h] = 4;
  for (let j: u32 = 0; j < 48; j++) poses[output + j] = 0;
  for (let j: u32 = 0; j < 4; j++) valid[h * 4 + j] = 0;
  const p0: vec3f64 = vec3f64(problems[input], problems[input + 1], problems[input + 2]);
  const p1: vec3f64 = vec3f64(problems[input + 3], problems[input + 4], problems[input + 5]);
  const p2: vec3f64 = vec3f64(problems[input + 6], problems[input + 7], problems[input + 8]);
  const world: Basis = makeBasis(p0, p1, p2);
  if (world.ok === 0) {
    status[h] = 1;
    return;
  }
  let ray0: vec3f64 = vec3f64(problems[input + 9], problems[input + 10], problems[input + 11]);
  let ray1: vec3f64 = vec3f64(problems[input + 12], problems[input + 13], problems[input + 14]);
  let ray2: vec3f64 = vec3f64(problems[input + 15], problems[input + 16], problems[input + 17]);
  const s0: f64 = max(abs(ray0.x), max(abs(ray0.y), abs(ray0.z)));
  const s1: f64 = max(abs(ray1.x), max(abs(ray1.y), abs(ray1.z)));
  const s2: f64 = max(abs(ray2.x), max(abs(ray2.y), abs(ray2.z)));
  if (!(min(s0, min(s1, s2)) > 0)) {
    status[h] = 2;
    return;
  }
  ray0 = ray0 / s0;
  ray1 = ray1 / s1;
  ray2 = ray2 / s2;
  ray0 = ray0 / length(ray0);
  ray1 = ray1 / length(ray1);
  ray2 = ray2 / length(ray2);
  if (max(length(ray0 - ray1), max(length(ray0 - ray2), length(ray1 - ray2))) < 1e-3) {
    status[h] = 2;
    return;
  }
  const c: f64 = length(p1 - p0);
  const b: f64 = length(p2 - p0);
  const a: f64 = length(p2 - p1);
  if (min(a, min(b, c)) < 1e-6 * max(a, max(b, c))) {
    status[h] = 6;
    return;
  }
  const alpha: f64 = dot(ray1, ray2);
  const beta: f64 = dot(ray0, ray2);
  const gamma: f64 = dot(ray0, ray1);
  const A: f64 = (a / c) * (a / c);
  const B: f64 = (b / c) * (b / c);
  let D: array<f64, 3> = [1, -2 * gamma, 1];
  let N: array<f64, 3> = [A - B + 1, -2 * gamma * (A - B), A - B - 1];
  let L: array<f64, 2> = [2 * beta, -2 * alpha];
  let polynomial: array<f64, 5> = [0, 0, 0, 0, 0];
  for (let j: u32 = 0; j < 3; j++) {
    for (let k: u32 = 0; k < 3; k++) polynomial[j + k] += N[j] * N[k];
    for (let k: u32 = 0; k < 2; k++) polynomial[j + k] -= 2 * beta * N[j] * L[k];
    let q: f64 = -B * D[j];
    if (j === 0) q += 1;
    for (let k: u32 = 0; k < 2; k++) {
      for (let m: u32 = 0; m < 2; m++) polynomial[j + k + m] += q * L[k] * L[m];
    }
  }
  const roots: RootSet = realRoots(polynomial);
  if (roots.overflow !== 0) {
    status[h] = 5;
    return;
  }
  if (roots.count === 0) {
    status[h] = 3;
    return;
  }
  let accepted: u32 = 0;
  for (let root: u32 = 0; root < roots.count; root++) {
    const x: f64 = roots.values[root];
    if (!(x > 0 && x < 1e9)) continue;
    const denominator: f64 = (x - 2 * gamma) * x + 1;
    const l: f64 = L[0] + L[1] * x;
    const n: f64 = (N[2] * x + N[1]) * x + N[0];
    if (!(denominator > 1e-14)) continue;
    let ys: array<f64, 2> = [0, 0];
    let yCount: u32 = 1;
    if (abs(l) > 1e-8) ys[0] = n / l;
    else {
      if (abs(n) > 1e-6) continue;
      const radical: f64 = B * denominator - 1 + beta * beta;
      if (radical < 0) continue;
      ys[0] = beta + sqrt(radical);
      ys[1] = beta - sqrt(radical);
      yCount = 2;
    }
    for (let branch: u32 = 0; branch < yCount; branch++) {
      const y: f64 = ys[branch];
      if (!(y > 0 && y < 1e9)) continue;
      const lengths: vec3f64 = refineLengths(
        vec3f64(f64(1), x, y) / sqrt(denominator),
        alpha,
        beta,
        gamma,
        A,
        B,
      );
      const q0: vec3f64 = ray0 * (lengths.x * c);
      const q1: vec3f64 = ray1 * (lengths.y * c);
      const q2: vec3f64 = ray2 * (lengths.z * c);
      const error: f64 = max(
        abs(length(q0 - q1) / c - 1),
        max(abs(length(q0 - q2) / b - 1), abs(length(q1 - q2) / a - 1)),
      );
      if (!(error <= 1e-8)) continue;
      const camera: Basis = makeBasis(q0, q1, q2);
      if (camera.ok === 0) continue;
      const t: vec3f64 = q0 - rotatePoint(camera, world, p0);
      const r0: vec3f64 = rotatePoint(camera, world, p0) + t;
      const r1: vec3f64 = rotatePoint(camera, world, p1) + t;
      const r2: vec3f64 = rotatePoint(camera, world, p2) + t;
      const sceneScale: f64 = max(a, max(b, c));
      if (!(min(r0.z, min(r1.z, r2.z)) > 0)) continue;
      if (!(max(length(r0 - q0), max(length(r1 - q1), length(r2 - q2))) <= 1e-7 * sceneScale))
        continue;
      const col0: vec3f64 = camera.u * world.u.x + camera.v * world.v.x + camera.w * world.w.x;
      const col1: vec3f64 = camera.u * world.u.y + camera.v * world.v.y + camera.w * world.w.y;
      const col2: vec3f64 = camera.u * world.u.z + camera.v * world.v.z + camera.w * world.w.z;
      let candidate: array<f64, 12> = [
        col0.x,
        col1.x,
        col2.x,
        col0.y,
        col1.y,
        col2.y,
        col0.z,
        col1.z,
        col2.z,
        t.x,
        t.y,
        t.z,
      ];
      let duplicate: bool = false;
      for (let old: u32 = 0; old < accepted; old++) {
        let dr: f64 = 0;
        let dt: f64 = 0;
        for (let j: u32 = 0; j < 9; j++) {
          const v: f64 = poses[output + old * 12 + j] - candidate[j];
          dr += v * v;
        }
        for (let j: u32 = 9; j < 12; j++) {
          const v: f64 = poses[output + old * 12 + j] - candidate[j];
          dt += v * v;
        }
        if (sqrt(dr) < 1e-5 && sqrt(dt) < 1e-5 * sceneScale) duplicate = true;
      }
      if (duplicate) continue;
      if (accepted === 4) {
        for (let j: u32 = 0; j < 48; j++) poses[output + j] = 0;
        for (let j: u32 = 0; j < 4; j++) valid[h * 4 + j] = 0;
        counts[h] = 0;
        status[h] = 5;
        return;
      }
      for (let j: u32 = 0; j < 12; j++) poses[output + accepted * 12 + j] = candidate[j];
      valid[h * 4 + accepted] = 1;
      accepted++;
    }
  }
  counts[h] = accepted;
  if (accepted > 0) status[h] = 0;
}
