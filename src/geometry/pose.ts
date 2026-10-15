import { determinant3, identity, multiply, transpose } from '../sfm/linear.ts';
import { cameraProjection } from './camera.ts';
import type { Camera, Pixel } from './camera.ts';

/** World-to-camera convention: q = R X + t. Translation is not normalized. */
export interface Pose {
  readonly R: Float64Array;
  readonly t: Float64Array;
}
export function identityPose(): Pose {
  return { R: identity(3), t: new Float64Array(3) };
}
export function clonePose(p: Pose): Pose {
  return { R: p.R.slice(), t: p.t.slice() };
}
export function validatePose(p: Pose): void {
  if (
    p.R.length !== 9 ||
    p.t.length !== 3 ||
    !p.R.every(Number.isFinite) ||
    !p.t.every(Number.isFinite)
  )
    throw new RangeError('Invalid pose dimensions or values');
  const q = multiply(p.R, transpose(p.R, 3), 3);
  if (
    q.some((v, i) => Math.abs(v - (i % 4 === 0 ? 1 : 0)) > 1e-6) ||
    Math.abs(determinant3(p.R) - 1) > 1e-6
  )
    throw new RangeError('Pose rotation must belong to SO(3)');
}
export function transformPoint(p: Pose, X: ArrayLike<number>): Float64Array {
  return Float64Array.from(
    { length: 3 },
    (_, i) => p.R[i * 3]! * X[0]! + p.R[i * 3 + 1]! * X[1]! + p.R[i * 3 + 2]! * X[2]! + p.t[i]!,
  );
}
export function cameraCenter(p: Pose): Float64Array {
  return Float64Array.from(
    { length: 3 },
    (_, i) => -p.R[i]! * p.t[0]! - p.R[i + 3]! * p.t[1]! - p.R[i + 6]! * p.t[2]!,
  );
}
export function rotationExp(w: ArrayLike<number>): Float64Array {
  const x = w[0]!,
    y = w[1]!,
    z = w[2]!;
  if (![x, y, z].every(Number.isFinite)) throw new RangeError('Invalid rotation vector');
  const theta = Math.hypot(x, y, z);
  const a = theta < 1e-7 ? 1 - (theta * theta) / 6 : Math.sin(theta) / theta;
  const b = theta < 1e-7 ? 0.5 - (theta * theta) / 24 : (1 - Math.cos(theta)) / (theta * theta);
  const W = Float64Array.of(0, -z, y, z, 0, -x, -y, x, 0),
    W2 = multiply(W, W, 3);
  return W.map((v, i) => (i % 4 === 0 ? 1 : 0) + a * v + b * W2[i]!);
}
/** Left rotation, additive camera translation: q' = exp(w) q + v. */
export function updatePose(p: Pose, delta: ArrayLike<number>): Pose {
  const dR = rotationExp(delta),
    rotated = transformPoint({ R: dR, t: new Float64Array(3) }, p.t);
  return { R: multiply(dR, p.R, 3), t: rotated.map((v, i) => v + delta[i + 3]!) };
}
export function projectPoint(camera: Camera, pose: Pose, X: ArrayLike<number>): Pixel | undefined {
  return cameraProjection(camera, transformPoint(pose, X))?.pixel;
}
export function observationJacobian(
  camera: Camera,
  pose: Pose,
  X: ArrayLike<number>,
):
  | {
      pixel: Pixel;
      point: Float64Array;
      pose: Float64Array;
    }
  | undefined {
  const q = transformPoint(pose, X),
    projection = cameraProjection(camera, q);
  if (!projection) return undefined;
  const J = projection.jacobian,
    point = new Float64Array(6),
    cam = new Float64Array(12);
  const B = [0, q[2]!, -q[1]!, -q[2]!, 0, q[0]!, q[1]!, -q[0]!, 0]; // -[q]x, row-major
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        point[i * 3 + j] += J[i * 3 + k]! * pose.R[k * 3 + j]!;
        cam[i * 6 + j] += J[i * 3 + k]! * B[k * 3 + j]!;
      }
      cam[i * 6 + j + 3] = J[i * 3 + j]!;
    }
  return { pixel: projection.pixel, point, pose: cam };
}
