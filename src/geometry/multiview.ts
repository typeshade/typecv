import { unprojectPixel, validateCamera } from './camera.ts';
import type { Camera, Pixel, Pose } from '../contracts/camera.ts';
import {
  cameraCenter,
  observationJacobian,
  projectPoint,
  transformPoint,
  validatePose,
} from './pose.ts';
import { inverse3, nullspace, solveSystem } from './numerics.ts';

export interface DepthSensitivityOptions {
  /** Optional bound on linearized depth stddev / positive camera Z, conditional on fixed poses/K. */
  readonly maxRelativeDepthStdDev?: number;
  /** Assumed independent pixel-coordinate noise stddev; default 1px, not estimated from residuals. */
  readonly pixelNoiseStdDevPx?: number;
}
export interface TrackTriangulationOptions extends DepthSensitivityOptions {
  readonly maxErrorPx?: number;
  readonly minAngleDeg?: number;
  readonly maxPairHypotheses?: number;
}
/** @internal Validate even when there are too few observations. A zero internal limit disables it. */
export function depthSensitivityControls(options: DepthSensitivityOptions) {
  const limit = options.maxRelativeDepthStdDev,
    noise = options.pixelNoiseStdDevPx ?? 1;
  if (
    (limit !== undefined && (!Number.isFinite(limit) || limit <= 0)) ||
    !Number.isFinite(noise) ||
    noise <= 0
  )
    throw new RangeError('Invalid conditional depth sensitivity options');
  return { maxRelativeDepthStdDev: limit ?? 0, pixelNoiseStdDevPx: noise };
}
/**
 * Linearized depth sensitivity to independent pixel noise. Poses, intrinsics and inlier
 * membership are held fixed: this is NOT total SfM uncertainty or a calibrated probability.
 * Undefined means singular/invalid information; it never supplies a point or a new observation.
 */
export function assessTrackDepth(
  point: Float64Array,
  observations: readonly TriangulationObservation[],
  pixelNoiseStdDevPx = 1,
):
  | {
      readonly maxRelativeDepthStdDev: number;
      readonly conditionalOn: 'fixed-poses-and-intrinsics';
    }
  | undefined {
  depthSensitivityControls({ pixelNoiseStdDevPx });
  if (point.length !== 3 || !point.every(Number.isFinite))
    throw new RangeError('Invalid depth assessment point');
  const information = new Float64Array(9);
  for (const o of observations) {
    validateCamera(o.camera);
    validatePose(o.pose);
    if (o.pixel.length !== 2 || !o.pixel.every(Number.isFinite))
      throw new RangeError('Invalid depth assessment pixel');
    const j = observationJacobian(o.camera, o.pose, point);
    if (!j) return undefined;
    for (let row = 0; row < 2; row++)
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
          information[3 * r + c] += j.point[3 * row + r]! * j.point[3 * row + c]!;
  }
  // Normalize before inversion: changing the map's arbitrary scale must not change admission.
  const magnitude = Math.max(...information.map(Math.abs));
  if (!(magnitude > 0)) return undefined;
  const covariance = inverse3(information.map((x) => x / magnitude));
  if (!covariance || observations.length < 2) return undefined;
  let maximum = 0;
  for (const o of observations) {
    const z = transformPoint(o.pose, point)[2]!,
      axis = o.pose.R.subarray(6, 9);
    let variance = 0;
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) variance += axis[r]! * covariance[3 * r + c]! * axis[c]!;
    if (!(z > 0 && variance >= 0 && Number.isFinite(variance))) return undefined;
    maximum = Math.max(maximum, (pixelNoiseStdDevPx * Math.sqrt(variance / magnitude)) / z);
  }
  return Number.isFinite(maximum)
    ? { maxRelativeDepthStdDev: maximum, conditionalOn: 'fixed-poses-and-intrinsics' }
    : undefined;
}

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
/** Status codes follow GPU triangulation; rejected results carry no geometry or mask. */
export type DetailedTrackTriangulation =
  | { readonly status: 0; readonly track: TrackTriangulation }
  | { readonly status: 1 | 3 | 4 | 5; readonly track?: undefined };
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
  options: TrackTriangulationOptions = {},
): TrackTriangulation | undefined {
  const result = triangulateTrackDetailed(observations, options);
  return result.status === 0 ? result.track : undefined;
}

/**
 * 0 accepted; 1 insufficient observations; 3 no consensus; 4 low parallax; 5 depth sensitivity.
 * Status4 follows a valid residual/positive-depth consensus but exposes no XYZ or mask.
 * Only a two-observation status4 implies both inputs passed those checks; with more
 * observations it does not authorize every input row. Invalid inputs still throw.
 */
export function triangulateTrackDetailed(
  observations: readonly TriangulationObservation[],
  options: TrackTriangulationOptions = {},
): DetailedTrackTriangulation {
  const depth = depthSensitivityControls(options);
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
  if (observations.length < 2) return { status: 1 };
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
  if (!best || bestCount < 2) return { status: 3 };
  // Fixed consensus refinement cannot lower the objective by dropping observations.
  const selected = observations.filter((_, i) => bestMask[i]);
  const linear = triangulateLinear(selected);
  const cost = (X: Float64Array) => selected.reduce((s, o) => s + error(o, X) ** 2, 0);
  if (linear && cost(linear) < cost(best)) best = linear;
  const refined = refineTrackPoint(best, selected);
  if (cost(refined) < cost(best)) best = refined;
  const inliers = Uint8Array.from(observations, (o) => (error(o, best!) <= maxError ? 1 : 0)),
    accepted = observations.filter((_, i) => inliers[i]);
  if (accepted.length < 2) return { status: 3 };
  let maxAngle = 0;
  for (let i = 0; i < accepted.length; i++)
    for (let j = i + 1; j < accepted.length; j++)
      maxAngle = Math.max(maxAngle, angleAt(best, accepted[i]!.pose, accepted[j]!.pose));
  if (!Number.isFinite(maxAngle)) return { status: 3 };
  if (maxAngle < minAngle) return { status: 4 };
  if (depth.maxRelativeDepthStdDev) {
    const sensitivity = assessTrackDepth(best, accepted, depth.pixelNoiseStdDevPx);
    if (!sensitivity || sensitivity.maxRelativeDepthStdDev > depth.maxRelativeDepthStdDev)
      return { status: 5 };
  }
  return {
    status: 0,
    track: {
      xyz: best,
      inliers,
      maxAngleDeg: maxAngle,
      meanErrorPx: accepted.reduce((s, o) => s + error(o, best!), 0) / accepted.length,
    },
  };
}
