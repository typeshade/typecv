import { unprojectPixel, validateCamera } from './camera.ts';
import type { Camera, Pixel } from './camera.ts';
import { cameraCenter, observationJacobian, projectPoint, validatePose } from './pose.ts';
import type { Pose } from './pose.ts';
import { nullspace, solveSystem } from './numerics.ts';

export interface TriangulationObservation {
  readonly camera: Camera;
  readonly pose: Pose;
  readonly pixel: Pixel;
}
export interface TrackTriangulation {
  readonly xyz: Float64Array;
  readonly inliers: Uint8Array;
  readonly maxAngleDeg: number;
  readonly meanErrorPx: number;
}
export function triangulateLinear(
  observations: readonly TriangulationObservation[],
): Float64Array | undefined {
  if (observations.length < 2) return undefined;
  const rows: number[][] = [];
  for (const o of observations) {
    const xy = unprojectPixel(o.camera, o.pixel);
    if (!xy) return undefined;
    const { R, t } = o.pose;
    for (let k = 0; k < 2; k++) {
      const coord = xy[k]!,
        row = [0, 1, 2].map((j) => coord * R[6 + j]! - R[3 * k + j]!);
      row.push(coord * t[2]! - t[k]!);
      const norm = Math.hypot(...row);
      if (!(norm > 0)) return undefined;
      rows.push(row.map((v) => v / norm));
    }
  }
  try {
    const { vector: v, singularValues: s } = nullspace(rows, 4);
    if (s[1]! < 1e-10 * s[3]! || Math.abs(v[3]!) < 1e-12) return undefined;
    const X = v.slice(0, 3).map((x) => x / v[3]!);
    return X.every(Number.isFinite) ? X : undefined;
  } catch {
    return undefined;
  }
}
function error(o: TriangulationObservation, X: Float64Array): number {
  const p = projectPoint(o.camera, o.pose, X);
  return p ? Math.hypot(p[0] - o.pixel[0], p[1] - o.pixel[1]) : Infinity;
}
export function refineTrackPoint(
  initial: Float64Array,
  observations: readonly TriangulationObservation[],
  iterations = 10,
): Float64Array {
  let X = initial.slice(),
    lambda = 1e-3;
  const cost = (p: Float64Array) => observations.reduce((s, o) => s + error(o, p) ** 2, 0);
  let value = cost(X);
  if (!Number.isFinite(value)) return X;
  for (let it = 0; it < iterations; it++) {
    const A = new Float64Array(9),
      b = new Float64Array(3);
    for (const o of observations) {
      const j = observationJacobian(o.camera, o.pose, X);
      if (!j) return X;
      for (let r = 0; r < 2; r++) {
        const e = j.pixel[r]! - o.pixel[r]!;
        for (let k = 0; k < 3; k++) {
          b[k] -= j.point[r * 3 + k]! * e;
          for (let l = 0; l < 3; l++) A[k * 3 + l] += j.point[r * 3 + k]! * j.point[r * 3 + l]!;
        }
      }
    }
    for (let k = 0; k < 3; k++) A[k * 3 + k] += lambda * Math.max(A[k * 3 + k]!, 1e-10);
    const step = solveSystem(A, b);
    if (!step) break;
    const candidate = X.map((v, k) => v + step[k]!),
      next = cost(candidate);
    if (next < value) {
      const change = value - next;
      X = candidate;
      value = next;
      lambda = Math.max(1e-10, lambda / 3);
      if (change < 1e-12 * Math.max(1, value)) break;
    } else lambda *= 10;
  }
  return X;
}
function angleAt(X: Float64Array, a: Pose, b: Pose): number {
  const ca = cameraCenter(a),
    cb = cameraCenter(b),
    u = X.map((x, i) => x - ca[i]!),
    v = X.map((x, i) => x - cb[i]!);
  const cross = [
    u[1]! * v[2]! - u[2]! * v[1]!,
    u[2]! * v[0]! - u[0]! * v[2]!,
    u[0]! * v[1]! - u[1]! * v[0]!,
  ];
  return (
    (Math.atan2(
      Math.hypot(...cross),
      u.reduce((s, x, i) => s + x * v[i]!, 0),
    ) *
      180) /
    Math.PI
  );
}

/** Robust small-track triangulation; all-observation fit plus pair hypotheses, with held-out observation filtering. */
export function triangulateTrack(
  observations: readonly TriangulationObservation[],
  options: { maxErrorPx?: number; minAngleDeg?: number; maxPairHypotheses?: number } = {},
): TrackTriangulation | undefined {
  const maxError = options.maxErrorPx ?? 3,
    minAngle = options.minAngleDeg ?? 0.5,
    maxPairs = options.maxPairHypotheses ?? 32;
  if (
    !Number.isFinite(maxError) ||
    maxError <= 0 ||
    !Number.isFinite(minAngle) ||
    minAngle < 0 ||
    minAngle >= 180 ||
    !Number.isSafeInteger(maxPairs) ||
    maxPairs < 1
  )
    throw new RangeError('Invalid triangulation options');
  for (const o of observations) {
    validateCamera(o.camera);
    validatePose(o.pose);
    if (!o.pixel.every(Number.isFinite)) throw new RangeError('Invalid triangulation pixel');
  }
  if (observations.length < 2) return undefined;
  const pairs: { i: number; j: number; baseline: number }[] = [];
  for (let i = 0; i < observations.length; i++)
    for (let j = i + 1; j < observations.length; j++) {
      const a = cameraCenter(observations[i]!.pose),
        b = cameraCenter(observations[j]!.pose);
      pairs.push({ i, j, baseline: Math.hypot(...a.map((v, k) => v - b[k]!)) });
    }
  pairs.sort((a, b) => b.baseline - a.baseline || a.i - b.i || a.j - b.j);
  let best: Float64Array | undefined,
    bestMask = new Uint8Array(observations.length),
    bestCount = 0,
    bestCost = Infinity;
  const candidates = [
    triangulateLinear(observations),
    ...pairs
      .slice(0, maxPairs)
      .map((p) => triangulateLinear([observations[p.i]!, observations[p.j]!])),
  ];
  for (const X of candidates) {
    if (!X) continue;
    const mask = new Uint8Array(observations.length);
    let count = 0,
      cost = 0;
    observations.forEach((o, i) => {
      const e = error(o, X);
      if (e <= maxError) {
        mask[i] = 1;
        count++;
        cost += e * e;
      }
    });
    if (count > bestCount || (count === bestCount && cost < bestCost)) {
      best = X;
      bestCount = count;
      bestCost = cost;
      bestMask = mask;
    }
  }
  if (!best || bestCount < 2) return undefined;
  // Fixed consensus refinement cannot lower the objective by dropping observations.
  const selected = observations.filter((_, i) => bestMask[i]);
  const linear = triangulateLinear(selected);
  const cost = (X: Float64Array) => selected.reduce((s, o) => s + error(o, X) ** 2, 0);
  if (linear && cost(linear) < cost(best)) best = linear;
  const refined = refineTrackPoint(best, selected);
  if (cost(refined) < cost(best)) best = refined;
  const inliers = Uint8Array.from(observations, (o) => (error(o, best!) <= maxError ? 1 : 0)),
    accepted = observations.filter((_, i) => inliers[i]);
  if (accepted.length < 2) return undefined;
  let maxAngle = 0;
  for (let i = 0; i < accepted.length; i++)
    for (let j = i + 1; j < accepted.length; j++)
      maxAngle = Math.max(maxAngle, angleAt(best, accepted[i]!.pose, accepted[j]!.pose));
  if (maxAngle < minAngle || !Number.isFinite(maxAngle)) return undefined;
  return {
    xyz: best,
    inliers,
    maxAngleDeg: maxAngle,
    meanErrorPx: accepted.reduce((s, o) => s + error(o, best!), 0) / accepted.length,
  };
}
