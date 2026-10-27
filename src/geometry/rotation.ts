import { multiply, svd3 } from './linear.ts';
import { unprojectPixel, validateCamera } from './camera.ts';
import type { Camera } from '../contracts/camera.ts';
import { ransacBudget, sample, seededRandom } from './numerics.ts';
import { DEFAULT_SEED } from './operator-defaults.ts';

export interface RelativeRotationOptions {
  readonly iterations?: number;
  /** Maximum error in each direction, in ideal pinhole pixels. Default: 3. */
  readonly maxErrorPx?: number;
  readonly confidence?: number;
  readonly minInliers?: number;
  readonly minInlierRatio?: number;
  /** Minor-axis spread of calibrated image bearings, atan(stddev) in degrees. Default: 0.5. */
  readonly minSpreadDeg?: number;
  readonly seed?: number;
}

/** A calibrated rotation-compatible hypothesis, not proof of a stationary camera center. */
export interface RelativeRotationResult {
  /** Row-major rotation from the first camera's bearings to the second camera's bearings. */
  readonly R: Float64Array;
  /** Original raw row indices of [x1,y1,x2,y2], not feature ids. */
  readonly inlierRows: Uint32Array;
  readonly invertibleCount: number;
  readonly meanReprojectionErrorPx: number;
  readonly spreadDeg: readonly [number, number];
  readonly translation: 'unobservable';
  readonly depth: 'not-estimated';
}

/** @internal Validate even when there are no observations; return an owned options snapshot. */
export function relativeRotationControls(
  options: RelativeRotationOptions = {},
): Required<RelativeRotationOptions> {
  const controls = {
    iterations: options.iterations ?? 128,
    maxErrorPx: options.maxErrorPx ?? 3,
    confidence: options.confidence ?? 0.999,
    minInliers: options.minInliers ?? 12,
    minInlierRatio: options.minInlierRatio ?? 0.6,
    minSpreadDeg: options.minSpreadDeg ?? 0.5,
    seed: options.seed ?? DEFAULT_SEED,
  };
  if (
    !Number.isSafeInteger(controls.iterations) ||
    controls.iterations < 1 ||
    controls.iterations > 100000 ||
    !Number.isFinite(controls.maxErrorPx) ||
    controls.maxErrorPx <= 0 ||
    !Number.isFinite(controls.confidence) ||
    controls.confidence <= 0 ||
    controls.confidence >= 1 ||
    !Number.isSafeInteger(controls.minInliers) ||
    controls.minInliers < 3 ||
    !Number.isFinite(controls.minInlierRatio) ||
    controls.minInlierRatio <= 0 ||
    controls.minInlierRatio > 1 ||
    !Number.isFinite(controls.minSpreadDeg) ||
    controls.minSpreadDeg <= 0 ||
    controls.minSpreadDeg >= 90 ||
    !Number.isSafeInteger(controls.seed)
  )
    throw new RangeError('Invalid relative-rotation options');
  return controls;
}

type BearingPair = { row: number; a: number[]; b: number[] };

function fit(pairs: readonly BearingPair[]): Float64Array {
  const covariance = new Float64Array(9);
  for (const { a, b } of pairs)
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) covariance[3 * r + c] += b[r]! * a[c]!;
  // Existing svd3 constructs right-handed bases, including the determinant correction.
  const { U, Vt } = svd3(covariance);
  return multiply(U, Vt, 3);
}

function error(R: Float64Array, { a, b }: BearingPair, first: Camera, second: Camera): number {
  const ab = [0, 1, 2].map(
      (r) => R[3 * r]! * a[0]! + R[3 * r + 1]! * a[1]! + R[3 * r + 2]! * a[2]!,
    ),
    ba = [0, 1, 2].map((r) => R[r]! * b[0]! + R[3 + r]! * b[1]! + R[6 + r]! * b[2]!);
  if (ab[2]! <= 0 || ba[2]! <= 0) return Infinity;
  const forward =
      ((ab[0]! / ab[2]! - b[0]! / b[2]!) * second.fx) ** 2 +
      ((ab[1]! / ab[2]! - b[1]! / b[2]!) * second.fy) ** 2,
    backward =
      ((ba[0]! / ba[2]! - a[0]! / a[2]!) * first.fx) ** 2 +
      ((ba[1]! / ba[2]! - a[1]! / a[2]!) * first.fy) ** 2;
  return Math.sqrt(Math.max(forward, backward));
}

/** @internal Minor-axis spread of calibrated forward bearings; shared by automatic bearing support. */
export function bearingSpreadDeg(bearings: readonly ArrayLike<number>[]): number {
  if (!bearings.length) return 0;
  let x = 0,
    y = 0;
  for (const p of bearings) {
    x += p[0]! / p[2]!;
    y += p[1]! / p[2]!;
  }
  x /= bearings.length;
  y /= bearings.length;
  let xx = 0,
    yy = 0,
    xy = 0;
  for (const p of bearings) {
    const u = p[0]! / p[2]! - x,
      v = p[1]! / p[2]! - y;
    xx += u * u;
    yy += v * v;
    xy += u * v;
  }
  const minor = Math.max(0, (xx + yy - Math.hypot(xx - yy, 2 * xy)) / (2 * bearings.length));
  return (Math.atan(Math.sqrt(minor)) * 180) / Math.PI;
}

/**
 * Robust SO(3) fit to calibrated RGB feature correspondences. Does not estimate translation,
 * initialize structure, or distinguish pure rotation from distant structure/small translation.
 */
export function estimateRelativeRotation(
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: RelativeRotationOptions = {},
): RelativeRotationResult | undefined {
  validateCamera(first);
  validateCamera(second);
  const controls = relativeRotationControls(options);
  if (!(matches instanceof Float64Array) || matches.length % 4 || !matches.every(Number.isFinite))
    throw new RangeError('Relative rotation requires finite Float64Array rows [x1,y1,x2,y2]');
  const pairs: BearingPair[] = [];
  for (let row = 0; row < matches.length / 4; row++) {
    const a = unprojectPixel(first, [matches[4 * row]!, matches[4 * row + 1]!]),
      b = unprojectPixel(second, [matches[4 * row + 2]!, matches[4 * row + 3]!]);
    if (!a || !b) continue;
    const na = Math.hypot(...a, 1),
      nb = Math.hypot(...b, 1);
    pairs.push({ row, a: [a[0] / na, a[1] / na, 1 / na], b: [b[0] / nb, b[1] / nb, 1 / nb] });
  }
  const needed = Math.max(controls.minInliers, Math.ceil(pairs.length * controls.minInlierRatio));
  if (pairs.length < needed) return undefined;
  const score = (R: Float64Array) => {
    const inliers: BearingPair[] = [];
    let total = 0;
    for (const p of pairs) {
      const e = error(R, p, first, second);
      if (e <= controls.maxErrorPx) {
        inliers.push(p);
        total += e;
      }
    }
    return { R, inliers, total };
  };
  let best: ReturnType<typeof score> | undefined,
    budget = controls.iterations;
  const rng = seededRandom(controls.seed);
  for (let trial = 0; trial < budget; trial++) {
    let R: Float64Array;
    try {
      R = fit(sample(pairs.length, 3, rng).map((i) => pairs[i]!));
    } catch {
      continue;
    }
    const current = score(R);
    if (
      !best ||
      current.inliers.length > best.inliers.length ||
      (current.inliers.length === best.inliers.length && current.total < best.total)
    ) {
      best = current;
      budget = Math.min(
        budget,
        Math.max(
          trial + 1,
          ransacBudget(current.inliers.length, pairs.length, 3, controls.confidence),
        ),
      );
    }
  }
  if (!best || best.inliers.length < needed) return undefined;
  for (let step = 0; step < 3; step++) {
    let refined: ReturnType<typeof score>;
    try {
      refined = score(fit(best.inliers));
    } catch {
      break;
    }
    if (
      refined.inliers.length < best.inliers.length ||
      (refined.inliers.length === best.inliers.length && refined.total >= best.total)
    )
      break;
    best = refined;
  }
  const spans = [
    bearingSpreadDeg(best.inliers.map((p) => p.a)),
    bearingSpreadDeg(best.inliers.map((p) => p.b)),
  ] as const;
  if (!spans.every(Number.isFinite) || Math.min(...spans) < controls.minSpreadDeg) return undefined;
  return {
    R: best.R,
    inlierRows: Uint32Array.from(best.inliers, (p) => p.row),
    invertibleCount: pairs.length,
    meanReprojectionErrorPx: best.total / best.inliers.length,
    spreadDeg: spans,
    translation: 'unobservable',
    depth: 'not-estimated',
  };
}
