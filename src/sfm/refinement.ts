import { multiply } from './linear.ts';
import { triangulate } from './triangulation.ts';
import type { CameraIntrinsics } from './two-view.ts';

type Pose = { R: Float64Array; t: Float64Array };

/** Pivoted dense solve for the small damped normal equations. */
function solve(a: number[][], b: number[]): number[] | undefined {
  const n = b.length;
  const rows = a.map((row, i) => [...row, b[i]!]);
  for (let j = 0; j < n; j++) {
    let p = j;
    for (let i = j + 1; i < n; i++) if (Math.abs(rows[i]![j]!) > Math.abs(rows[p]![j]!)) p = i;
    if (Math.abs(rows[p]![j]!) < 1e-20) return undefined;
    [rows[p], rows[j]] = [rows[j]!, rows[p]!];
    const d = rows[j]![j]!;
    for (let k = j; k <= n; k++) rows[j]![k] /= d;
    for (let i = 0; i < n; i++)
      if (i !== j) {
        const f = rows[i]![j]!;
        for (let k = j; k <= n; k++) rows[i]![k] -= f * rows[j]![k]!;
      }
  }
  const result = rows.map((row) => row[n]!);
  return result.every(Number.isFinite) ? result : undefined;
}

function residual(
  X: Float64Array,
  pose: Pose,
  xy: Float64Array,
  j: number,
  a: CameraIntrinsics,
  b: CameraIntrinsics,
): number[] | undefined {
  const q = [0, 1, 2].map(
    (i) =>
      pose.R[i * 3]! * X[0]! + pose.R[i * 3 + 1]! * X[1]! + pose.R[i * 3 + 2]! * X[2]! + pose.t[i]!,
  );
  if (!(X[2]! > 1e-12 && q[2]! > 1e-12)) return undefined;
  const r = [
    (X[0]! / X[2]! - xy[j]!) * a.fx,
    (X[1]! / X[2]! - xy[j + 1]!) * a.fy,
    (q[0]! / q[2]! - xy[j + 2]!) * b.fx,
    (q[1]! / q[2]! - xy[j + 3]!) * b.fy,
  ];
  return r.every(Number.isFinite) ? r : undefined;
}

/** LM with numerical Jacobians; accepts only objective-decreasing steps. */
function minimize(
  initial: Float64Array,
  evaluate: (p: Float64Array) => number[] | undefined,
  iterations: number,
  huber: number,
): Float64Array {
  let p = initial,
    r = evaluate(p);
  if (!r) return p;
  const cost = (v: number[]) =>
    v.reduce(
      (s, x) => s + (Math.abs(x) <= huber ? x * x : 2 * huber * Math.abs(x) - huber * huber),
      0,
    );
  let value = cost(r),
    damping = 1e-3;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const derivatives: number[][] = [];
    for (let k = 0; k < p.length; k++) {
      const h = 1e-6 * Math.max(1, Math.abs(p[k]!));
      const next = p.slice();
      next[k] += h;
      const v = evaluate(next);
      if (!v || v.length !== r.length) return p;
      derivatives.push(v.map((x, i) => (x - r![i]!) / h));
    }
    const A = Array.from({ length: p.length }, () => Array(p.length).fill(0));
    const b = Array(p.length).fill(0);
    for (let i = 0; i < r.length; i++) {
      const w = Math.abs(r[i]!) <= huber ? 1 : huber / Math.abs(r[i]!);
      for (let k = 0; k < p.length; k++) {
        b[k] -= w * derivatives[k]![i]! * r[i]!;
        for (let l = 0; l < p.length; l++)
          A[k]![l] += w * derivatives[k]![i]! * derivatives[l]![i]!;
      }
    }
    for (let k = 0; k < p.length; k++) A[k]![k] += damping * Math.max(1e-12, A[k]![k]!);
    const step = solve(A, b);
    if (!step) break;
    const next = p.map((x, k) => x + step[k]!);
    const candidate = evaluate(next);
    const nextValue = candidate ? cost(candidate) : Infinity;
    if (nextValue < value) {
      const improvement = value - nextValue;
      p = next;
      r = candidate!;
      value = nextValue;
      damping = Math.max(1e-10, damping / 3);
      if (improvement < 1e-10 * Math.max(1, value)) break;
    } else damping *= 10;
  }
  return p;
}

export function refinePoint(
  X: Float64Array,
  pose: Pose,
  xy: Float64Array,
  j: number,
  a: CameraIntrinsics,
  b: CameraIntrinsics,
): Float64Array {
  return minimize(X, (p) => residual(p, pose, xy, j, a, b), 8, Infinity);
}

function perturb(pose: Pose, p: Float64Array): Pose {
  const angle = Math.hypot(p[0]!, p[1]!, p[2]!);
  const A = angle < 1e-8 ? 1 - (angle * angle) / 6 : Math.sin(angle) / angle;
  const B = angle < 1e-8 ? 0.5 - (angle * angle) / 24 : (1 - Math.cos(angle)) / (angle * angle);
  const [x, y, z] = p;
  const W = Float64Array.of(0, -z!, y!, z!, 0, -x!, -y!, x!, 0);
  const W2 = multiply(W, W, 3);
  const dR = W.map((v, i) => (i % 4 === 0 ? 1 : 0) + A * v + B * W2[i]!);
  // Two coordinates in the tangent plane of the unit translation sphere.
  const t = pose.t;
  const axis = Math.abs(t[0]!) < 0.8 ? [1, 0, 0] : [0, 1, 0];
  let u = [
    t[1]! * axis[2]! - t[2]! * axis[1]!,
    t[2]! * axis[0]! - t[0]! * axis[2]!,
    t[0]! * axis[1]! - t[1]! * axis[0]!,
  ];
  const length = Math.hypot(...u);
  u = u.map((v) => v / length);
  const v = [
    t[1]! * u[2]! - t[2]! * u[1]!,
    t[2]! * u[0]! - t[0]! * u[2]!,
    t[0]! * u[1]! - t[1]! * u[0]!,
  ];
  const translated = t.map((value, i) => value + p[3]! * u[i]! + p[4]! * v[i]!);
  const norm = Math.hypot(...translated);
  return { R: multiply(dR, pose.R, 3), t: translated.map((value) => value / norm) };
}

/** Variable projection: optimize the five pose DOFs while re-solving each 3D point. */
export function refinePose(
  pose: Pose,
  xy: Float64Array,
  indices: number[],
  a: CameraIntrinsics,
  b: CameraIntrinsics,
  iterations: number,
  huber: number,
): Pose {
  const evaluate = (p: Float64Array): number[] | undefined => {
    const candidate = perturb(pose, p),
      out: number[] = [];
    for (const i of indices) {
      const j = i * 4;
      const X = triangulate(xy[j]!, xy[j + 1]!, xy[j + 2]!, xy[j + 3]!, candidate.R, candidate.t);
      if (!X) return undefined;
      const r = residual(refinePoint(X, candidate, xy, j, a, b), candidate, xy, j, a, b);
      if (!r) return undefined;
      out.push(...r);
    }
    return out;
  };
  return perturb(pose, minimize(new Float64Array(5), evaluate, iterations, huber));
}
