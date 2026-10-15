import { determinant3, multiply, svd3, symmetricEigen, transpose } from '../sfm/linear.ts';
import { unprojectPixel, validateCamera } from './camera.ts';
import type { Camera, Pixel } from './camera.ts';
import { observationJacobian, projectPoint, transformPoint, updatePose } from './pose.ts';
import type { Pose } from './pose.ts';
import { p3p } from './p3p.ts';
import { fitHomography } from './homography.ts';
import { nullspace, ransacBudget, sample, seededRandom, solveSystem } from './numerics.ts';

export interface PnPOptions {
  readonly iterations?: number;
  readonly thresholdPx?: number;
  readonly confidence?: number;
  readonly seed?: number;
  readonly minInliers?: number;
  readonly refinementIterations?: number;
}
export interface PnPResult {
  readonly pose: Pose;
  readonly inliers: Uint8Array;
  readonly inlierCount: number;
  readonly meanErrorPx: number;
  readonly iterations: number;
}
interface Correspondence {
  X: Float64Array;
  pixel: Pixel;
  normalized: Pixel;
}
function closestRotation(M: Float64Array): Float64Array {
  const { U, Vt } = svd3(M);
  return multiply(U, Vt, 3);
}
function fitPose(data: readonly Correspondence[], camera: Camera): Pose {
  const mean = Float64Array.from(
    { length: 3 },
    (_, j) => data.reduce((s, o) => s + o.X[j]!, 0) / data.length,
  );
  const distance =
    data.reduce((s, o) => s + Math.hypot(...o.X.map((v, j) => v - mean[j]!)), 0) / data.length;
  if (!(distance > 1e-12)) throw new Error('Coincident PnP world points');
  const covariance = new Float64Array(9);
  for (const o of data)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        covariance[i * 3 + j] += (o.X[i]! - mean[i]!) * (o.X[j]! - mean[j]!);
  const eig = symmetricEigen(covariance, 3);
  if (eig.values[1]! < 1e-10 * eig.values[2]!) throw new Error('Collinear PnP points');
  if (eig.values[0]! < 1e-10 * eig.values[2]!) {
    const u = Float64Array.of(eig.vectors[2]!, eig.vectors[5]!, eig.vectors[8]!),
      v = Float64Array.of(eig.vectors[1]!, eig.vectors[4]!, eig.vectors[7]!);
    const n = [
      u[1]! * v[2]! - u[2]! * v[1]!,
      u[2]! * v[0]! - u[0]! * v[2]!,
      u[0]! * v[1]! - u[1]! * v[0]!,
    ];
    const B = Float64Array.of(u[0]!, v[0]!, n[0]!, u[1]!, v[1]!, n[1]!, u[2]!, v[2]!, n[2]!);
    const H = fitHomography(
      data.map((o) => {
        const q = o.X.map((x, j) => x - mean[j]!);
        return [
          q.reduce((s, x, j) => s + x * u[j]!, 0),
          q.reduce((s, x, j) => s + x * v[j]!, 0),
          ...o.normalized,
        ];
      }),
    );
    const length = (Math.hypot(H[0]!, H[3]!, H[6]!) + Math.hypot(H[1]!, H[4]!, H[7]!)) / 2;
    const poses = [1, -1].map((sign) => {
      const a = [H[0]!, H[3]!, H[6]!].map((x) => (sign * x) / length),
        b = [H[1]!, H[4]!, H[7]!].map((x) => (sign * x) / length);
      const c = [
        a[1]! * b[2]! - a[2]! * b[1]!,
        a[2]! * b[0]! - a[0]! * b[2]!,
        a[0]! * b[1]! - a[1]! * b[0]!,
      ];
      const R = multiply(
        closestRotation(
          Float64Array.of(a[0]!, b[0]!, c[0]!, a[1]!, b[1]!, c[1]!, a[2]!, b[2]!, c[2]!),
        ),
        transpose(B, 3),
        3,
      );
      const rmean = transformPoint({ R, t: new Float64Array(3) }, mean);
      return {
        R,
        t: Float64Array.of(
          (sign * H[2]!) / length - rmean[0]!,
          (sign * H[5]!) / length - rmean[1]!,
          (sign * H[8]!) / length - rmean[2]!,
        ),
      };
    });
    return poses.sort((a, b) => poseCost(a, data, camera) - poseCost(b, data, camera))[0]!;
  }
  const scale = Math.sqrt(3) / distance,
    mx = data.reduce((s, o) => s + o.normalized[0], 0) / data.length,
    my = data.reduce((s, o) => s + o.normalized[1], 0) / data.length;
  const dist2 =
    data.reduce((s, o) => s + Math.hypot(o.normalized[0] - mx, o.normalized[1] - my), 0) /
    data.length;
  if (dist2 < 1e-12) throw new Error('Coincident PnP rays');
  const s2 = Math.SQRT2 / dist2,
    rows: number[][] = [];
  for (const o of data) {
    const X = [...o.X.map((x, j) => (x - mean[j]!) * scale), 1],
      x = (o.normalized[0] - mx) * s2,
      y = (o.normalized[1] - my) * s2;
    rows.push(
      [...X, 0, 0, 0, 0, ...X.map((v) => -x * v)],
      [0, 0, 0, 0, ...X, ...X.map((v) => -y * v)],
    );
  }
  const fit = nullspace(rows, 12);
  if (fit.singularValues[1]! < 1e-9 * fit.singularValues[11]!)
    throw new Error('Ambiguous PnP design');
  const P = fit.vector,
    tmp = new Float64Array(12);
  for (let j = 0; j < 4; j++) {
    tmp[j] = P[j]! / s2 + mx * P[8 + j]!;
    tmp[4 + j] = P[4 + j]! / s2 + my * P[8 + j]!;
    tmp[8 + j] = P[8 + j]!;
  }
  let M = Float64Array.from({ length: 9 }, (_, i) => tmp[Math.floor(i / 3) * 4 + (i % 3)]! * scale);
  let t = Float64Array.from(
    { length: 3 },
    (_, i) =>
      tmp[i * 4 + 3]! - M[i * 3]! * mean[0]! - M[i * 3 + 1]! * mean[1]! - M[i * 3 + 2]! * mean[2]!,
  );
  if (determinant3(M) < 0) {
    M = M.map((x) => -x);
    t = t.map((x) => -x);
  }
  const R = closestRotation(M),
    s = M.reduce((sum, x, i) => sum + x * R[i]!, 0) / 3;
  if (!(s > 1e-12)) throw new Error('Invalid PnP scale');
  return { R, t: t.map((x) => x / s) };
}
function poseCost(
  pose: Pose,
  data: readonly Correspondence[],
  camera: Camera,
  huber = Infinity,
): number {
  let cost = 0;
  for (const o of data) {
    const p = projectPoint(camera, pose, o.X);
    if (!p) return Infinity;
    const e = Math.hypot(p[0] - o.pixel[0], p[1] - o.pixel[1]);
    cost += e <= huber ? e * e : 2 * huber * e - huber * huber;
  }
  return cost;
}
function refine(
  pose: Pose,
  data: readonly Correspondence[],
  camera: Camera,
  iterations: number,
  huber: number,
): Pose {
  let best = pose,
    value = poseCost(best, data, camera, huber),
    lambda = 1e-3;
  for (let it = 0; it < iterations; it++) {
    const A = new Float64Array(36),
      b = new Float64Array(6);
    for (const o of data) {
      const j = observationJacobian(camera, best, o.X);
      if (!j) return best;
      const residual = [j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]],
        norm = Math.hypot(...residual),
        w = norm <= huber ? 1 : huber / norm;
      for (let r = 0; r < 2; r++)
        for (let k = 0; k < 6; k++) {
          b[k] -= w * j.pose[r * 6 + k]! * residual[r]!;
          for (let l = 0; l < 6; l++) A[k * 6 + l] += w * j.pose[r * 6 + k]! * j.pose[r * 6 + l]!;
        }
    }
    for (let k = 0; k < 6; k++) A[k * 6 + k] += lambda * Math.max(A[k * 6 + k]!, 1e-10);
    const delta = solveSystem(A, b);
    if (!delta) break;
    const candidate = updatePose(best, delta),
      cost = poseCost(candidate, data, camera, huber);
    if (cost < value) {
      const improvement = value - cost;
      best = candidate;
      value = cost;
      lambda = Math.max(1e-10, lambda / 3);
      if (improvement < 1e-10 * Math.max(1, cost)) break;
    } else lambda *= 10;
  }
  return best;
}

/** P3P RANSAC with all real branches, normalized DLT/planar refit and analytic pose LM. */
export function estimatePosePnP(
  worldPoints: Float64Array,
  pixels: Float64Array,
  camera: Camera,
  options: PnPOptions = {},
): PnPResult {
  validateCamera(camera);
  const n = worldPoints.length / 3;
  const iterations = options.iterations ?? 1000,
    threshold = options.thresholdPx ?? 3,
    confidence = options.confidence ?? 0.999,
    seed = options.seed ?? 17,
    minInliers = options.minInliers ?? Math.min(6, n),
    refinement = options.refinementIterations ?? 20;
  if (
    !Number.isInteger(n) ||
    n < 4 ||
    pixels.length !== 2 * n ||
    !worldPoints.every(Number.isFinite) ||
    !pixels.every(Number.isFinite)
  )
    throw new RangeError('PnP needs at least four finite XYZ/pixel correspondences');
  if (
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    !Number.isFinite(threshold) ||
    threshold <= 0 ||
    !(confidence > 0 && confidence < 1) ||
    !Number.isSafeInteger(seed) ||
    !Number.isSafeInteger(minInliers) ||
    minInliers < 4 ||
    !Number.isSafeInteger(refinement) ||
    refinement < 0
  )
    throw new RangeError('Invalid PnP options');
  const data: Correspondence[] = Array.from({ length: n }, (_, i) => {
    const pixel: Pixel = [pixels[2 * i]!, pixels[2 * i + 1]!],
      normalized = unprojectPixel(camera, pixel);
    if (!normalized) throw new RangeError('Cannot invert camera distortion for PnP pixel');
    return { X: worldPoints.slice(3 * i, 3 * i + 3), pixel, normalized };
  });
  const classify = (pose: Pose) => {
    const mask = new Uint8Array(n);
    let count = 0,
      cost = 0;
    data.forEach((o, i) => {
      const p = projectPoint(camera, pose, o.X);
      if (!p) return;
      const e = Math.hypot(p[0] - o.pixel[0], p[1] - o.pixel[1]);
      if (e <= threshold) {
        mask[i] = 1;
        count++;
        cost += e * e;
      }
    });
    return { mask, count, cost };
  };
  const rng = seededRandom(seed);
  let best: Pose | undefined,
    mask = new Uint8Array(n),
    count = 0,
    cost = Infinity,
    budget = iterations,
    trials = 0;
  for (let i = 0; i < budget; i++) {
    trials++;
    try {
      const subset = sample(n, 3, rng).map((i) => data[i]!);
      const hypotheses = p3p(
        subset.map((o) => o.X),
        subset.map((o) => Float64Array.of(o.normalized[0], o.normalized[1], 1)),
      );
      for (const pose of hypotheses) {
        const s = classify(pose);
        if (s.count > count || (s.count === count && s.cost < cost)) {
          best = pose;
          mask = s.mask;
          count = s.count;
          cost = s.cost;
          budget = Math.min(
            budget,
            Math.max(Math.min(64, iterations), ransacBudget(count, n, 3, confidence)),
          );
        }
      }
    } catch {
      /* Rank-deficient minimal samples are expected. */
    }
  }
  if (!best || count < minInliers)
    throw new Error('PnP could not register camera: insufficient nondegenerate consensus');
  for (let pass = 0; pass < 3; pass++) {
    const selected = data.filter((_, i) => mask[i]);
    let pose: Pose = best!;
    try {
      if (selected.length < 6) throw new Error('Refit requires six correspondences');
      const refit = fitPose(selected, camera);
      if (poseCost(refit, selected, camera) < poseCost(pose, selected, camera)) pose = refit;
    } catch {
      /* Keep the robust model. */
    }
    pose = refine(pose, selected, camera, refinement, threshold / 2);
    const s = classify(pose);
    if (s.count < count || (s.count === count && s.cost >= cost)) break;
    best = pose;
    mask = s.mask;
    count = s.count;
    cost = s.cost;
  }
  return {
    pose: best!,
    inliers: mask,
    inlierCount: count,
    meanErrorPx:
      data.reduce((sum, o, i) => {
        if (!mask[i]) return sum;
        const p = projectPoint(camera, best!, o.X)!;
        return sum + Math.hypot(p[0] - o.pixel[0], p[1] - o.pixel[1]);
      }, 0) / count,
    iterations: trials,
  };
}
