import { cameraProjection, unprojectPixel, validateCamera } from './camera.ts';
import type { Camera, Pose } from '../contracts/camera.ts';
import { cameraCenter, clonePose, transformPoint, validatePose } from './pose.ts';
import { inverse3, solveSystem } from './numerics.ts';
import type { TriangulationObservation } from './multiview.ts';

/** A fixed coordinate chart, not an extra pose measurement. rho is inverse optical-axis Z. */
export interface AnchoredInverseDepth {
  readonly reference: Pose;
  readonly parameters: Float64Array;
}
export interface InverseDepthLinearization {
  readonly pixel: readonly [number, number];
  readonly point: Float64Array;
  readonly pose: Float64Array;
  /** Derivative with respect to log shared focal scale (principal point/distortion held fixed). */
  readonly focal: Float64Array;
}
export function initializeInverseDepth(
  o: TriangulationObservation,
  xyz?: ArrayLike<number>,
): AnchoredInverseDepth {
  validateCamera(o.camera);
  validatePose(o.pose);
  const ray = unprojectPixel(o.camera, o.pixel);
  if (!ray) throw new RangeError('Inverse-depth anchor cannot be unprojected');
  let parameters = Float64Array.of(ray[0], ray[1], 0);
  if (xyz) {
    const q = transformPoint(o.pose, xyz);
    if (!(q[2]! > 0) || !q.every(Number.isFinite))
      throw new RangeError('Invalid finite inverse-depth seed');
    parameters = Float64Array.of(q[0]! / q[2]!, q[1]! / q[2]!, 1 / q[2]!);
  }
  return { reference: clonePose(o.pose), parameters };
}

/** Stable at rho=0; no large XYZ surrogate and exactly zero translation sensitivity there. */
export function inverseDepthJacobian(
  camera: Camera,
  pose: Pose,
  p: AnchoredInverseDepth,
): InverseDepthLinearization | undefined {
  const r = p.parameters,
    A = new Float64Array(9),
    C = cameraCenter(p.reference),
    T = transformPoint(pose, C);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) A[3 * i + j] += pose.R[3 * i + k]! * p.reference.R[3 * j + k]!;
  const q = Float64Array.from(
    { length: 3 },
    (_, i) => A[3 * i]! * r[0]! + A[3 * i + 1]! * r[1]! + A[3 * i + 2]! + r[2]! * T[i]!,
  );
  const projection = cameraProjection(camera, q);
  if (!projection) return undefined;
  const J = projection.jacobian,
    point = new Float64Array(6),
    cam = new Float64Array(12),
    B = [0, q[2]!, -q[1]!, -q[2]!, 0, q[0]!, q[1]!, -q[0]!, 0];
  for (let row = 0; row < 2; row++)
    for (let col = 0; col < 3; col++) {
      for (let k = 0; k < 3; k++) {
        point[3 * row + col] += J[3 * row + k]! * (col < 2 ? A[3 * k + col]! : T[k]!);
        cam[6 * row + col] += J[3 * row + k]! * B[3 * k + col]!;
      }
      cam[6 * row + col + 3] = r[2]! * J[3 * row + col]!;
    }
  return {
    pixel: projection.pixel,
    point,
    pose: cam,
    focal: Float64Array.of(projection.pixel[0] - camera.cx, projection.pixel[1] - camera.cy),
  };
}

/** Materialization is permitted only after admission; signed/zero rho remains non-finite. */
export function inverseDepthPoint(p: AnchoredInverseDepth): Float64Array | undefined {
  const [a, b, rho] = p.parameters;
  if (!(rho! > 0)) return undefined;
  const C = cameraCenter(p.reference);
  const X = Float64Array.from(
    { length: 3 },
    (_, i) =>
      C[i]! + (p.reference.R[i]! * a! + p.reference.R[3 + i]! * b! + p.reference.R[6 + i]!) / rho!,
  );
  return X.every(Number.isFinite) ? X : undefined;
}

export function transformCovariance(J: Float64Array, covariance: Float64Array): Float64Array {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        for (let l = 0; l < 3; l++)
          out[3 * i + j] += J[3 * i + k]! * covariance[3 * k + l]! * J[3 * j + l]!;
  return out;
}

/** Deterministic coordinate change. Does not introduce a measurement of the new camera pose. */
export function reanchorInverseDepth(
  p: AnchoredInverseDepth,
  reference: Pose,
  covariance?: Float64Array,
) {
  validatePose(reference);
  const C = cameraCenter(p.reference),
    T = transformPoint(reference, C),
    A = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        A[3 * i + j] += reference.R[3 * i + k]! * p.reference.R[3 * j + k]!;
  const [a, b, rho] = p.parameters,
    q = Float64Array.from(
      { length: 3 },
      (_, i) => A[3 * i]! * a! + A[3 * i + 1]! * b! + A[3 * i + 2]! + rho! * T[i]!,
    );
  if (!(q[2]! > 1e-12)) throw new RangeError('New inverse-depth chart is behind the bearing');
  const parameters = Float64Array.of(q[0]! / q[2]!, q[1]! / q[2]!, rho! / q[2]!),
    J = new Float64Array(9);
  for (let k = 0; k < 3; k++) {
    const dz = k < 2 ? A[6 + k]! : T[2]!;
    for (let row = 0; row < 2; row++)
      J[3 * row + k] = ((k < 2 ? A[3 * row + k]! : T[row]!) - parameters[row]! * dz) / q[2]!;
    J[6 + k] = ((k === 2 ? 1 : 0) - parameters[2]! * dz) / q[2]!;
  }
  return {
    point: { reference: clonePose(reference), parameters },
    ...(covariance ? { covariance: transformCovariance(J, covariance) } : {}),
  };
}

export function rescaleInverseDepth(
  p: AnchoredInverseDepth,
  scale: number,
  covariance?: Float64Array,
  origin: Float64Array = new Float64Array(3),
) {
  if (!(scale > 0 && Number.isFinite(scale)))
    throw new RangeError('Invalid inverse-depth map scale');
  const C = cameraCenter(p.reference).map((c, i) => origin[i]! + scale * (c - origin[i]!)),
    R = p.reference.R.slice(),
    t = Float64Array.from(
      { length: 3 },
      (_, i) => -R[3 * i]! * C[0]! - R[3 * i + 1]! * C[1]! - R[3 * i + 2]! * C[2]!,
    ),
    parameters = p.parameters.slice(),
    J = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1 / scale);
  parameters[2] /= scale;
  return {
    point: { reference: { R, t }, parameters },
    ...(covariance ? { covariance: transformCovariance(J, covariance) } : {}),
  };
}

/** Local covariance only; singular or nearly rank-deficient information stays unobservable. */
export function inverseDepthCovariance(information: Float64Array): Float64Array | undefined {
  const diagonal = [0, 4, 8].map((i) => Math.sqrt(information[i]!));
  if (diagonal.some((x) => !(x > 0 && Number.isFinite(x)))) return undefined;
  const scaled = information.map((x, i) => x / (diagonal[Math.floor(i / 3)]! * diagonal[i % 3]!)),
    inv = inverse3(scaled);
  if (!inv || [0, 4, 8].some((i) => !(inv[i]! > 0))) return undefined;
  return inv.map((x, i) => x / (diagonal[Math.floor(i / 3)]! * diagonal[i % 3]!));
}

export function inverseRangeUncertainty(p: AnchoredInverseDepth, covariance?: Float64Array) {
  const [a, b, rho] = p.parameters,
    norm = Math.hypot(a!, b!, 1),
    mean = rho! / norm,
    J = [(-rho! * a!) / norm ** 3, (-rho! * b!) / norm ** 3, 1 / norm];
  let variance = 0;
  if (covariance)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) variance += J[i]! * covariance[3 * i + j]! * J[j]!;
  return {
    mean,
    ...(covariance && variance >= 0 && Number.isFinite(variance)
      ? { stddev: Math.sqrt(variance) }
      : {}),
  };
}

/** Refit unique raw observations; never recursively fuse correlated triangulated estimates. */
export function fitInverseDepth(
  observations: readonly TriangulationObservation[],
  options: { iterations?: number; pixelNoiseStdDevPx?: number; huberPx?: number } = {},
) {
  if (!observations.length) throw new RangeError('An inverse-depth fit needs an anchor');
  const sigma = options.pixelNoiseStdDevPx ?? 1,
    huber = options.huberPx ?? 3,
    iterations = options.iterations ?? 12;
  if (
    !(sigma > 0 && Number.isFinite(sigma) && huber > 0 && Number.isFinite(huber)) ||
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    iterations > 100
  )
    throw new RangeError('Invalid inverse-depth fit options');
  let point = initializeInverseDepth(observations[0]!),
    lambda = 1e-3;
  const linearize = (p: AnchoredInverseDepth) => {
    const H = new Float64Array(9),
      b = new Float64Array(3);
    let cost = 0,
      squared = 0,
      maxErrorPx = 0;
    for (const o of observations) {
      const j = inverseDepthJacobian(o.camera, o.pose, p);
      if (!j) return undefined;
      const r = [j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]],
        e = Math.hypot(...r),
        w = (e <= huber ? 1 : huber / e) / sigma ** 2;
      cost += e <= huber ? e * e : 2 * huber * e - huber * huber;
      squared += e * e;
      maxErrorPx = Math.max(maxErrorPx, e);
      for (let row = 0; row < 2; row++)
        for (let k = 0; k < 3; k++) {
          b[k] -= w * j.point[3 * row + k]! * r[row]!;
          for (let l = 0; l < 3; l++)
            H[3 * k + l] += w * j.point[3 * row + k]! * j.point[3 * row + l]!;
        }
    }
    return { H, b, cost, rmsePx: Math.sqrt(squared / observations.length), maxErrorPx };
  };
  let current = linearize(point)!;
  for (let i = 0; i < iterations && current; i++) {
    const H = current.H.slice();
    for (let k = 0; k < 3; k++) H[4 * k] += lambda * Math.max(H[4 * k]!, 1e-10);
    const d = solveSystem(H, current.b);
    if (!d) break;
    const candidate = { ...point, parameters: point.parameters.map((x, k) => x + d[k]!) },
      next = linearize(candidate);
    if (next && next.cost < current.cost) {
      point = candidate;
      current = next;
      lambda /= 3;
    } else lambda *= 10;
  }
  if (!current) throw new RangeError('Invalid inverse-depth observations');
  const covariance = inverseDepthCovariance(current.H);
  return {
    point,
    covariance,
    conditionalOn: 'fixed-poses-and-intrinsics' as const,
    rmsePx: current.rmsePx,
    maxErrorPx: current.maxErrorPx,
    inverseRange: inverseRangeUncertainty(point, covariance),
  };
}
