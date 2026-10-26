import { multiply, transpose } from './linear.ts';
import { transformPoint } from './pose.ts';
import type { Pose } from './pose.ts';
import { solveSystem } from './numerics.ts';

function add(a: number[], b: number[], scale = 1): number[] {
  const out = Array(Math.max(a.length, b.length)).fill(0);
  a.forEach((v, i) => (out[i] += v));
  b.forEach((v, i) => (out[i] += scale * v));
  return out;
}
function mul(a: number[], b: number[]): number[] {
  const out = Array(a.length + b.length - 1).fill(0);
  a.forEach((v, i) => b.forEach((w, j) => (out[i + j] += v * w)));
  return out;
}
function value(a: number[], x: number): number {
  let v = 0;
  for (let i = a.length - 1; i >= 0; i--) v = v * x + a[i]!;
  return v;
}
/** Isolate all real roots using derivative critical points and safeguarded bisection. */
function roots(coefficients: number[]): number[] {
  const scale = Math.max(...coefficients.map(Math.abs));
  if (!(scale > 0)) return [];
  const a = coefficients.map((v) => v / scale);
  while (a.length > 1 && Math.abs(a[a.length - 1]!) < 1e-14) a.pop();
  const degree = a.length - 1;
  if (degree === 0) return [];
  if (degree === 1) return [-a[0]! / a[1]!];
  const derivative = a.slice(1).map((v, i) => v * (i + 1));
  const bound = 1 + Math.max(...a.slice(0, -1).map((v) => Math.abs(v / a[degree]!)));
  const critical = roots(derivative)
      .filter((x) => x > -bound && x < bound)
      .sort((a, b) => a - b),
    cuts = [-bound, ...critical, bound],
    out: number[] = [];
  for (const x of critical)
    if (Math.abs(value(a, x)) < 1e-12 * Math.max(1, Math.abs(x) ** degree)) out.push(x);
  for (let i = 0; i < cuts.length - 1; i++) {
    let lo = cuts[i]!,
      hi = cuts[i + 1]!,
      fl = value(a, lo),
      fh = value(a, hi);
    if (fl * fh >= 0) continue;
    for (let it = 0; it < 100; it++) {
      const mid = (lo + hi) / 2,
        fm = value(a, mid);
      if (fl * fm <= 0) {
        hi = mid;
        fh = fm;
      } else {
        lo = mid;
        fl = fm;
      }
      if (hi - lo < 1e-13 * Math.max(1, Math.abs(mid))) break;
    }
    out.push((lo + hi) / 2);
  }
  return out
    .sort((a, b) => a - b)
    .filter((x, i, a) => i === 0 || Math.abs(x - a[i - 1]!) > 1e-8 * Math.max(1, Math.abs(x)));
}
function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}
function basis(points: readonly Float64Array[]): Float64Array | undefined {
  const u = points[1]!.map((v, i) => v - points[0]![i]!),
    v = points[2]!.map((v, i) => v - points[0]![i]!);
  const un = Math.hypot(...u),
    vn = Math.hypot(...v);
  if (!(un > 1e-12 && vn > 1e-12)) return undefined;
  for (let i = 0; i < 3; i++) u[i] /= un;
  const p = dot(u, v);
  for (let i = 0; i < 3; i++) v[i] -= p * u[i]!;
  const n = Math.hypot(...v);
  if (n < 1e-8 * vn) return undefined;
  for (let i = 0; i < 3; i++) v[i] /= n;
  const w = [
    u[1]! * v[2]! - u[2]! * v[1]!,
    u[2]! * v[0]! - u[0]! * v[2]!,
    u[0]! * v[1]! - u[1]! * v[0]!,
  ];
  return Float64Array.of(u[0]!, v[0]!, w[0]!, u[1]!, v[1]!, w[1]!, u[2]!, v[2]!, w[2]!);
}

/** Grunert distance-ratio P3P. Returns every validated positive-depth real branch. */
export function p3p(world: readonly Float64Array[], bearings: readonly Float64Array[]): Pose[] {
  if (world.length !== 3 || bearings.length !== 3)
    throw new RangeError('P3P requires three points and bearings');
  const Bworld = basis(world);
  if (!Bworld) return [];
  const rays = bearings.map((b) => {
    const n = Math.hypot(...b);
    return b.map((x) => x / n);
  });
  // Extremely narrow ray cones make the quartic nearly multiple in f64.
  const separation = (i: number, j: number) =>
    Math.hypot(...rays[i]!.map((x, k) => x - rays[j]![k]!));
  if (Math.max(separation(0, 1), separation(0, 2), separation(1, 2)) < 1e-3) return [];
  const distance = (i: number, j: number) =>
    Math.hypot(...world[i]!.map((v, k) => v - world[j]![k]!));
  const c = distance(0, 1),
    b = distance(0, 2),
    a = distance(1, 2),
    alpha = dot(rays[1]!, rays[2]!),
    beta = dot(rays[0]!, rays[2]!),
    gamma = dot(rays[0]!, rays[1]!);
  const A = (a / c) ** 2,
    B = (b / c) ** 2,
    D = [1, -2 * gamma, 1],
    N = add(
      D.map((v) => (A - B) * v),
      [1, 0, -1],
    ),
    L = [2 * beta, -2 * alpha];
  const polynomial = add(add(mul(N, N), mul(N, L), -2 * beta), mul(add([1], D, -B), mul(L, L)));
  const poses: Pose[] = [];
  for (const x of roots(polynomial)) {
    if (!(x > 0)) continue;
    const denominator = value(D, x),
      l = value(L, x),
      n = value(N, x);
    if (!(denominator > 1e-14)) continue;
    let ys: number[];
    if (Math.abs(l) > 1e-8) ys = [n / l];
    else {
      if (Math.abs(n) > 1e-6) continue;
      const radical = B * denominator - 1 + beta * beta;
      if (radical < 0) continue;
      ys = [beta + Math.sqrt(radical), beta - Math.sqrt(radical)];
    }
    for (const y of ys) {
      if (!(y > 0)) continue;
      // Distances in units of c improve Newton conditioning across arbitrary world scales.
      let lengths = Float64Array.of(
        1 / Math.sqrt(denominator),
        x / Math.sqrt(denominator),
        y / Math.sqrt(denominator),
      );
      for (let it = 0; it < 12; it++) {
        const [u, v, w] = lengths;
        const r = Float64Array.of(
          u! * u! + v! * v! - 2 * gamma * u! * v! - 1,
          u! * u! + w! * w! - 2 * beta * u! * w! - B,
          v! * v! + w! * w! - 2 * alpha * v! * w! - A,
        );
        if (Math.max(...r.map(Math.abs)) < 1e-12) break;
        const J = Float64Array.of(
          2 * (u! - gamma * v!),
          2 * (v! - gamma * u!),
          0,
          2 * (u! - beta * w!),
          0,
          2 * (w! - beta * u!),
          0,
          2 * (v! - alpha * w!),
          2 * (w! - alpha * v!),
        );
        const step = solveSystem(
          J,
          r.map((x) => -x),
        );
        if (!step) break;
        const next = lengths.map((x, i) => x + step[i]!);
        if (next.some((x) => x <= 0 || !Number.isFinite(x))) break;
        lengths = next;
      }
      const q = rays.map((ray, i) => ray.map((v) => v * lengths[i]! * c));
      const error = Math.max(
        Math.abs(Math.hypot(...q[0]!.map((v, i) => v - q[1]![i]!)) / c - 1),
        Math.abs(Math.hypot(...q[0]!.map((v, i) => v - q[2]![i]!)) / b - 1),
        Math.abs(Math.hypot(...q[1]!.map((v, i) => v - q[2]![i]!)) / a - 1),
      );
      if (error > 1e-8) continue;
      const Bcamera = basis(q);
      if (!Bcamera) continue;
      const R = multiply(Bcamera, transpose(Bworld, 3), 3),
        r0 = transformPoint({ R, t: new Float64Array(3) }, world[0]!);
      const pose = { R, t: q[0]!.map((v, i) => v - r0[i]!) };
      if (
        world.some((X, i) => {
          const transformed = transformPoint(pose, X);
          return (
            transformed[2]! <= 0 ||
            Math.hypot(...transformed.map((v, j) => v - q[i]![j]!)) > 1e-7 * Math.max(c, b, a)
          );
        })
      )
        continue;
      if (
        !poses.some(
          (p) =>
            Math.hypot(...p.t.map((v, i) => v - pose.t[i]!)) < 1e-5 * Math.max(c, b, a) &&
            Math.hypot(...p.R.map((v, i) => v - pose.R[i]!)) < 1e-5,
        )
      )
        poses.push(pose);
    }
  }
  // A quartic P3P problem has at most four distinct branches. Reject unresolved numerical splitting.
  return poses.length <= 4 ? poses : [];
}
