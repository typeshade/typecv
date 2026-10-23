import { nullspace } from '../geometry/numerics.ts';
import { GeometryEstimationError } from '../geometry/errors.ts';
import { triangulate } from './triangulation.ts';
import { refinePoint, refinePose } from './refinement.ts';
import { determinant3, multiply, svd3, transpose } from './linear.ts';

/** Pixel-coordinate matches, packed [x1,y1,x2,y2, ...]. */
export interface CameraIntrinsics {
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
}
export interface TwoViewOptions {
  /** Experimental finite/directional separation and inverse-depth two-view refinement. Default false. */
  readonly separateDirectionalObservations?: boolean;
  /** Hard cap on RANSAC trials. Default: 500. */
  readonly iterations?: number;
  /** Sampson distance in normalized camera coordinates, i.e. approximately angle in radians. */
  readonly threshold?: number;
  readonly seed?: number;
  /** Adaptive stopping confidence; iterations remains a hard cap. */
  readonly confidence?: number;
  readonly minIterations?: number;
  /** 0 disables nonlinear pose and point refinement. Default: 15. */
  readonly refinementIterations?: number;
  /** Huber transition for pose refinement, in pixels. Default: 2. */
  readonly huberPx?: number;
  /** Minimum triangulation angle in degrees. Default: 0.1. */
  readonly minParallaxDeg?: number;
  /** Maximum error in either view for an accepted point. Default: 4 pixels. */
  readonly maxReprojectionErrorPx?: number;
}
/**
 * @internal The essential matrix of trial `trial` of the consensus search, or undefined when its
 * sample is degenerate. The focal search supplies the trials of one pair from a cache
 * (`src/reconstruction/focal-pairs.ts`) in place of an eight-point fit for each focal.
 */
export type TrialHypotheses = (trial: number) => Float64Array | undefined;
export interface TwoViewResult {
  readonly diagnostics: {
    readonly ransacIterations: number;
    readonly refinementApplied: boolean;
    /** E inliers explained by rotation within the per-view pixel error budget; depth stays unresolved. */
    readonly directionCompatibleCount?: number;
    /** Median over positive finite candidates after excluding direction-compatible observations. */
    readonly medianParallaxDeg: number;
    readonly meanReprojectionErrorBeforePx: number;
    readonly meanReprojectionErrorAfterPx: number;
  };
  /** Final point quality mask; inliers separately records epipolar consensus. */
  readonly validPoints: Uint8Array;
  readonly E: Float64Array; // 3×3 row-major, x2^T E x1 = 0
  readonly R: Float64Array; // X2 = R X1 + t
  readonly t: Float64Array; // direction only, ||t||=1
  readonly inliers: Uint8Array;
  readonly points: Float64Array; // 3D in camera-1 coordinates; NaN triplets for rejected matches
  readonly inlierCount: number;
  readonly positiveDepthCount: number; // final accepted points, after all quality filters
  readonly meanReprojectionErrorPx: number; // mean per-view Euclidean error over valid inliers
}

const EPS = 1e-12;
function validIntrinsics(k: CameraIntrinsics): boolean {
  return [k.fx, k.fy, k.cx, k.cy].every(Number.isFinite) && k.fx > 0 && k.fy > 0;
}
/** @internal Shared with the asynchronous GPU consensus path. */
export function normalizePixels(
  pixels: Float64Array,
  a: CameraIntrinsics,
  b: CameraIntrinsics,
): Float64Array {
  const out = new Float64Array(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    out[i] = (pixels[i]! - a.cx) / a.fx;
    out[i + 1] = (pixels[i + 1]! - a.cy) / a.fy;
    out[i + 2] = (pixels[i + 2]! - b.cx) / b.fx;
    out[i + 3] = (pixels[i + 3]! - b.cy) / b.fy;
  }
  return out;
}

function hartley(
  points: Float64Array,
  strideOffset: number,
  indices: number[],
): {
  T: Float64Array;
  coords: Float64Array;
} {
  let cx = 0,
    cy = 0;
  for (const index of indices) {
    cx += points[index * 4 + strideOffset]!;
    cy += points[index * 4 + strideOffset + 1]!;
  }
  cx /= indices.length;
  cy /= indices.length;
  let dist = 0;
  for (const index of indices)
    dist += Math.hypot(
      points[index * 4 + strideOffset]! - cx,
      points[index * 4 + strideOffset + 1]! - cy,
    );
  dist /= indices.length;
  if (!(dist > EPS)) throw new Error('Degenerate feature distribution');
  const scale = Math.SQRT2 / dist;
  const coords = new Float64Array(indices.length * 2);
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k]!;
    coords[2 * k] = (points[i * 4 + strideOffset]! - cx) * scale;
    coords[2 * k + 1] = (points[i * 4 + strideOffset + 1]! - cy) * scale;
  }
  const T = Float64Array.from([scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1]);
  return { T, coords };
}

/**
 * @internal The normalized eight-point solution on the chosen rows, in the coordinates of
 * `points`: the least-squares null vector with Hartley's normalization, before any constraint on
 * its singular values. It is the part of an essential-matrix hypothesis that does not depend on
 * the focal length (docs/auto-focal.md, section 8).
 */
export function eightPoint(points: Float64Array, indices: number[]): Float64Array {
  if (indices.length < 8) throw new RangeError('Eight or more matches are required');
  const left = hartley(points, 0, indices),
    right = hartley(points, 2, indices);
  const rows: number[][] = [];
  for (let j = 0; j < indices.length; j++) {
    const x = left.coords[2 * j]!,
      y = left.coords[2 * j + 1]!;
    const u = right.coords[2 * j]!,
      v = right.coords[2 * j + 1]!;
    rows.push([u * x, u * y, u, v * x, v * y, v, x, y, 1]);
  }
  // Direct one-sided Jacobi SVD avoids squaring the design matrix condition number.
  const { vector: Fhat, singularValues } = nullspace(rows, 9);
  if (!(singularValues[1]! > 1e-7 * singularValues[8]!))
    throw new Error('Degenerate eight-point system (planar or collinear matches)');
  return multiply(multiply(transpose(right.T, 3), Fhat, 3), left.T, 3);
}

/** @internal Small f64 solver; dense hypothesis scoring can run on the device. */
export function estimateEssential(points: Float64Array, indices: number[]): Float64Array {
  return projectEssential(eightPoint(points, indices));
}

/** @internal The nearest essential matrix of a 3 x 3 matrix, at unit norm. */
export function projectEssential(E: Float64Array): Float64Array {
  const { U, S, Vt } = svd3(E);
  const avg = (S[0]! + S[1]!) * 0.5;
  const UD = Float64Array.from([
    U[0]! * avg,
    U[1]! * avg,
    0,
    U[3]! * avg,
    U[4]! * avg,
    0,
    U[6]! * avg,
    U[7]! * avg,
    0,
  ]);
  const projected = multiply(UD, Vt, 3);
  const magnitude = Math.hypot(...projected);
  return projected.map((x) => x / magnitude);
}

/** Squared Sampson residual (normalized-camera-coordinate units). */
export function sampsonSquared(
  E: ArrayLike<number>,
  x: number,
  y: number,
  u: number,
  v: number,
): number {
  const ex0 = E[0]! * x + E[1]! * y + E[2]!;
  const ex1 = E[3]! * x + E[4]! * y + E[5]!;
  const ex2 = E[6]! * x + E[7]! * y + E[8]!;
  const et0 = E[0]! * u + E[3]! * v + E[6]!;
  const et1 = E[1]! * u + E[4]! * v + E[7]!;
  const error = u * ex0 + v * ex1 + ex2;
  const denominator = ex0 * ex0 + ex1 * ex1 + et0 * et0 + et1 * et1;
  return denominator > EPS ? (error * error) / denominator : Infinity;
}

/** @internal */
export function sampleIndices(n: number, count: number, rng: () => number): number[] {
  const chosen = new Set<number>();
  while (chosen.size < count) chosen.add(Math.floor(rng() * n));
  return [...chosen];
}
/** @internal */
export function random(seed: number): () => number {
  let s = seed >>> 0;
  if (s === 0) s = 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}
function classify(
  E: Float64Array,
  coords: Float64Array,
  thresholdSquared: number,
): { mask: Uint8Array; count: number; cost: number } {
  const n = coords.length / 4;
  const mask = new Uint8Array(n);
  let count = 0,
    cost = 0;
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    const d = sampsonSquared(E, coords[j]!, coords[j + 1]!, coords[j + 2]!, coords[j + 3]!);
    if (d <= thresholdSquared) {
      mask[i] = 1;
      count++;
      cost += d;
    }
  }
  return { mask, count, cost };
}

function depth2(X: ArrayLike<number>, R: ArrayLike<number>, t: ArrayLike<number>): number {
  return R[6]! * X[0]! + R[7]! * X[1]! + R[8]! * X[2]! + t[2]!;
}
function candidates(E: Float64Array): { R: Float64Array; t: Float64Array }[] {
  const { U, Vt } = svd3(E);
  const W = Float64Array.from([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const Wt = transpose(W, 3);
  const t = Float64Array.of(U[2]!, U[5]!, U[8]!);
  return [W, Wt].flatMap((w) => {
    const r = multiply(multiply(U, w, 3), Vt, 3);
    const R = determinant3(r) < 0 ? r.map((x) => -x) : r;
    return [
      { R, t },
      { R, t: t.map((x) => -x) },
    ];
  });
}

/** @internal Validate once for both synchronous and asynchronous execution. */
export function resolveTwoViewInput(
  matchesPixels: Float64Array,
  K1: CameraIntrinsics,
  K2: CameraIntrinsics,
  options: TwoViewOptions = {},
) {
  if (!validIntrinsics(K1) || !validIntrinsics(K2))
    throw new RangeError('Camera intrinsics must be finite, with positive focal lengths');
  if (
    matchesPixels.length % 4 !== 0 ||
    matchesPixels.length < 32 ||
    !Array.from(matchesPixels).every(Number.isFinite)
  )
    throw new RangeError('Pass at least 8 finite correspondences packed [x1,y1,x2,y2]');
  return resolveTwoViewOptions(options);
}

/** @internal Option validation shared with image-pair verification, including short input. */
export function resolveTwoViewOptions(options: TwoViewOptions = {}) {
  const iterations = options.iterations ?? 500,
    threshold = options.threshold ?? 0.002;
  if (
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    !(threshold > 0) ||
    !Number.isFinite(threshold)
  )
    throw new RangeError('Invalid RANSAC options');
  const confidence = options.confidence ?? 0.999;
  const minIterations = options.minIterations ?? Math.min(64, iterations);
  const refinementIterations = options.refinementIterations ?? 15;
  const huberPx = options.huberPx ?? 2;
  const minParallax = options.minParallaxDeg ?? 0.1;
  const maxError = options.maxReprojectionErrorPx ?? 4;
  const separateDirectionalObservations = options.separateDirectionalObservations ?? false;
  if (
    typeof separateDirectionalObservations !== 'boolean' ||
    !(confidence > 0 && confidence < 1) ||
    !Number.isSafeInteger(minIterations) ||
    minIterations < 1 ||
    minIterations > iterations ||
    !Number.isSafeInteger(refinementIterations) ||
    refinementIterations < 0 ||
    !Number.isFinite(huberPx) ||
    huberPx <= 0 ||
    !Number.isFinite(minParallax) ||
    minParallax < 0 ||
    minParallax >= 180 ||
    !Number.isFinite(maxError) ||
    maxError <= 0 ||
    (options.seed !== undefined && !Number.isSafeInteger(options.seed))
  )
    throw new RangeError('Invalid reconstruction options');
  return {
    iterations,
    threshold,
    confidence,
    minIterations,
    refinementIterations,
    huberPx,
    minParallax,
    maxError,
    separateDirectionalObservations,
    seed: options.seed ?? 17,
  };
}

export function reconstructTwoView(
  matchesPixels: Float64Array,
  K1: CameraIntrinsics,
  K2: CameraIntrinsics,
  options: TwoViewOptions = {},
  hypotheses?: TrialHypotheses,
): TwoViewResult {
  const controls = resolveTwoViewInput(matchesPixels, K1, K2, options);
  const { iterations, threshold, confidence, minIterations, seed } = controls;
  const points = normalizePixels(matchesPixels, K1, K2);
  const n = points.length / 4,
    rng = random(seed);
  const thresholdSquared = threshold * threshold;
  let bestE: Float64Array | undefined,
    bestMask: Uint8Array | undefined,
    bestCount = 0,
    bestCost = Infinity;
  let budget = iterations,
    trials = 0;
  for (let trial = 0; trial < budget; trial++) {
    trials++;
    try {
      const E = hypotheses
        ? hypotheses(trial)
        : estimateEssential(points, sampleIndices(n, 8, rng));
      if (!E) continue;
      const score = classify(E, points, thresholdSquared);
      if (score.count > bestCount || (score.count === bestCount && score.cost < bestCost)) {
        bestE = E;
        bestMask = score.mask;
        bestCount = score.count;
        bestCost = score.cost;
        const allInliers = Math.pow(bestCount / n, 8);
        const required =
          allInliers >= 1 ? 1 : Math.ceil(Math.log1p(-confidence) / Math.log1p(-allInliers));
        budget = Math.min(budget, Math.max(minIterations, required));
      }
    } catch {
      /* degenerate minimal samples are normal in RANSAC */
    }
  }
  if (!bestE || !bestMask || bestCount < 8)
    throw new GeometryEstimationError('Could not find a valid two-view geometry');
  // Local refit. Never replace the best hypothesis by a weaker model.
  for (let pass = 0; pass < 4; pass++) {
    const indices = Array.from({ length: n }, (_, i) => i).filter((i) => bestMask![i] === 1);
    try {
      const E = estimateEssential(points, indices);
      const score = classify(E, points, thresholdSquared);
      if (score.count < bestCount || (score.count === bestCount && score.cost >= bestCost)) break;
      bestE = E;
      bestMask = score.mask;
      bestCount = score.count;
      bestCost = score.cost;
    } catch {
      break;
    }
  }
  return recoverTwoViewGeometry(points, K1, K2, controls, {
    E: bestE,
    mask: bestMask,
    count: bestCount,
    trials,
  });
}

/** @internal f64 pose recovery and refinement shared by both consensus backends. */
export function recoverTwoViewGeometry(
  points: Float64Array,
  K1: CameraIntrinsics,
  K2: CameraIntrinsics,
  controls: ReturnType<typeof resolveTwoViewInput>,
  consensus: { E: Float64Array; mask: Uint8Array; count: number; trials: number },
): TwoViewResult {
  const { refinementIterations, huberPx, minParallax, maxError, threshold } = controls;
  const n = points.length / 4,
    thresholdSquared = threshold * threshold;
  let bestE = consensus.E,
    bestMask = consensus.mask,
    bestCount = consensus.count;
  const trials = consensus.trials;
  // Cheirality disambiguates the four decompositions of E.
  const hypotheses = candidates(bestE);
  // A rotation-compatible bearing need not have observable finite depth. This is a
  // conditional measurement test, not proof of exact infinity.
  // Check both projections and front-facing rays; translation never enters this test.
  const directionCompatible = (i: number, pose: (typeof hypotheses)[number]): boolean => {
    if (!controls.separateDirectionalObservations) return false;
    const j = i * 4,
      x = points[j]!,
      y = points[j + 1]!,
      u = points[j + 2]!,
      v = points[j + 3]!,
      { R } = pose,
      a = [
        R[0]! * x + R[1]! * y + R[2]!,
        R[3]! * x + R[4]! * y + R[5]!,
        R[6]! * x + R[7]! * y + R[8]!,
      ],
      b = [
        R[0]! * u + R[3]! * v + R[6]!,
        R[1]! * u + R[4]! * v + R[7]!,
        R[2]! * u + R[5]! * v + R[8]!,
      ];
    return (
      a[2]! > EPS &&
      b[2]! > EPS &&
      Math.hypot((a[0]! / a[2]! - u) * K2.fx, (a[1]! / a[2]! - v) * K2.fy) <= maxError &&
      Math.hypot((b[0]! / b[2]! - x) * K1.fx, (b[1]! / b[2]! - y) * K1.fy) <= maxError
    );
  };
  let bestPose: (typeof hypotheses)[number] | undefined,
    bestPositive = -1,
    bestDirectional = 0;
  const selected = Array.from({ length: n }, (_, i) => i).filter((i) => bestMask![i] === 1);
  for (const pose of hypotheses) {
    let positive = 0,
      directional = 0;
    for (const i of selected) {
      if (directionCompatible(i, pose)) {
        directional++;
        continue;
      }
      const j = i * 4;
      const X = triangulate(
        points[j]!,
        points[j + 1]!,
        points[j + 2]!,
        points[j + 3]!,
        pose.R,
        pose.t,
      );
      if (X && X[2]! > EPS && depth2(X, pose.R, pose.t) > EPS) positive++;
    }
    if (positive > bestPositive) {
      bestPositive = positive;
      bestDirectional = directional;
      bestPose = pose;
    }
  }
  if (!bestPose || bestPositive < Math.max(6, Math.floor((bestCount - bestDirectional) * 0.6)))
    throw new GeometryEstimationError(
      'Cheirality failed: low parallax, planar scene, or invalid matching',
    );
  // Keep a fixed positive-depth set for optimization so cost cannot improve by dropping points.
  const positive = selected.filter((i) => {
    if (directionCompatible(i, bestPose!)) return false;
    const j = i * 4,
      X = triangulate(
        points[j]!,
        points[j + 1]!,
        points[j + 2]!,
        points[j + 3]!,
        bestPose!.R,
        bestPose!.t,
      );
    return X && X[2]! > EPS && depth2(X, bestPose!.R, bestPose!.t) > EPS;
  });
  const directional = selected.filter((i) => directionCompatible(i, bestPose!));
  const errors = (X: Float64Array, pose: typeof bestPose, j: number): number[] => {
    const z = depth2(X, pose!.R, pose!.t);
    const u = (pose!.R[0]! * X[0]! + pose!.R[1]! * X[1]! + pose!.R[2]! * X[2]! + pose!.t[0]!) / z;
    const v = (pose!.R[3]! * X[0]! + pose!.R[4]! * X[1]! + pose!.R[5]! * X[2]! + pose!.t[1]!) / z;
    return [
      Math.hypot((X[0]! / X[2]! - points[j]!) * K1.fx, (X[1]! / X[2]! - points[j + 1]!) * K1.fy),
      Math.hypot((u - points[j + 2]!) * K2.fx, (v - points[j + 3]!) * K2.fy),
    ];
  };
  const meanError = (pose: typeof bestPose, refine: boolean): number => {
    let total = 0;
    for (const i of positive) {
      const j = i * 4;
      let X = triangulate(
        points[j]!,
        points[j + 1]!,
        points[j + 2]!,
        points[j + 3]!,
        pose!.R,
        pose!.t,
      );
      if (!X || X[2]! <= EPS || depth2(X, pose!.R, pose!.t) <= EPS) return Infinity;
      if (refine) X = refinePoint(X, pose!, points, j, K1, K2);
      total += errors(X, pose, j).reduce((a, b) => a + b, 0) / 2;
    }
    return total / positive.length;
  };
  const before = meanError(bestPose, false);
  let after = before,
    refinementApplied = false;
  if (refinementIterations > 0) {
    const refined = refinePose(
      bestPose,
      points,
      positive,
      K1,
      K2,
      refinementIterations,
      huberPx,
      directional,
    );
    const error = meanError(refined, true);
    // Also guard the public mean Euclidean metric, beyond the robust optimization objective.
    if (Number.isFinite(error) && error <= before) {
      bestPose = refined;
      after = error;
      refinementApplied = true;
      const cross = Float64Array.of(
        0,
        -bestPose.t[2]!,
        bestPose.t[1]!,
        bestPose.t[2]!,
        0,
        -bestPose.t[0]!,
        -bestPose.t[1]!,
        bestPose.t[0]!,
        0,
      );
      bestE = multiply(cross, bestPose.R, 3);
      const magnitude = Math.hypot(...bestE);
      bestE = bestE.map((x) => x / magnitude);
      const finalScore = classify(bestE, points, thresholdSquared);
      bestMask = finalScore.mask;
      bestCount = finalScore.count;
    }
  }
  const reconstructed = new Float64Array(n * 3).fill(NaN),
    validPoints = new Uint8Array(n);
  const angles: number[] = [];
  let valid = 0,
    directionCompatibleCount = 0,
    reprojection = 0;
  for (let i = 0; i < n; i++) {
    if (!bestMask[i]) continue;
    if (directionCompatible(i, bestPose)) {
      directionCompatibleCount++;
      continue;
    }
    const j = i * 4;
    let X = triangulate(
      points[j]!,
      points[j + 1]!,
      points[j + 2]!,
      points[j + 3]!,
      bestPose.R,
      bestPose.t,
    );
    if (!X || X[2]! <= EPS || depth2(X, bestPose.R, bestPose.t) <= EPS) continue;
    if (refinementApplied) X = refinePoint(X, bestPose, points, j, K1, K2);
    const second = [0, 1, 2].map(
      (k) =>
        bestPose!.R[k * 3]! * X![0]! +
        bestPose!.R[k * 3 + 1]! * X![1]! +
        bestPose!.R[k * 3 + 2]! * X![2]!,
    );
    const ray2 = second.map((v, k) => v + bestPose!.t[k]!);
    const cosine =
      second.reduce((sum, v, k) => sum + v * ray2[k]!, 0) /
      (Math.hypot(...second) * Math.hypot(...ray2));
    const angle = (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
    if (!Number.isFinite(angle)) continue;
    angles.push(angle);
    const error = errors(X, bestPose, j);
    if (angle < minParallax || error.some((e) => !Number.isFinite(e) || e > maxError)) continue;
    reconstructed.set(X, i * 3);
    validPoints[i] = 1;
    reprojection += (error[0]! + error[1]!) / 2;
    valid++;
  }
  angles.sort((a, b) => a - b);
  const mid = Math.floor(angles.length / 2);
  const median = angles.length
    ? angles.length % 2
      ? angles[mid]!
      : (angles[mid - 1]! + angles[mid]!) / 2
    : 0;
  if (median < minParallax)
    throw new GeometryEstimationError(
      'Low parallax: insufficient baseline for stable reconstruction',
    );
  if (valid < 6)
    throw new GeometryEstimationError(
      'Insufficient reliable points after parallax and reprojection filtering',
    );
  return {
    E: bestE,
    R: bestPose.R,
    t: bestPose.t,
    inliers: bestMask,
    points: reconstructed,
    validPoints,
    inlierCount: bestCount,
    positiveDepthCount: valid,
    meanReprojectionErrorPx: reprojection / valid,
    diagnostics: {
      ransacIterations: trials,
      refinementApplied,
      ...(controls.separateDirectionalObservations ? { directionCompatibleCount } : {}),
      medianParallaxDeg: median,
      meanReprojectionErrorBeforePx: before,
      meanReprojectionErrorAfterPx: after,
    },
  };
}
