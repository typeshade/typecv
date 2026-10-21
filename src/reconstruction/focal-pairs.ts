import { estimateHomography } from '../geometry/homography.ts';
import { verifyImagePair, type ImagePairVerification } from '../geometry/image-pair.ts';
import type { Camera } from '../geometry/camera.ts';
import {
  classify,
  estimateFundamental,
  normalizePixels,
  random,
  sampleIndices,
} from '../sfm/two-view.ts';
import { pairSeed } from './identity.ts';
import type { ImageFeatures, PairMatches } from './types.ts';

/**
 * The pair work of the focal search that does not depend on the focal (docs/auto-focal.md,
 * section 8). The matches, their homography and their epipolar inliers are the same at every
 * focal of the grid. Only the essential matrix `E(f) = K(f)ᵀ F K(f)` and what follows from it
 * depend on `f`. The search computes the base of a pair once and derives each grid point's
 * verification from it on the host.
 */
export interface PairBase {
  readonly pair: PairMatches;
  /** The raw correspondences [x1, y1, x2, y2] of the pair's matches. */
  readonly raw: Float64Array;
  readonly homographyInliers: number;
  /** The rows (indices of `raw`) that the fundamental matrix keeps. */
  readonly rows: readonly number[];
  readonly trials: number;
}

/** The controls of the base and of its derivation: the mapper's own numerical controls. */
export interface PairBaseControls {
  readonly seed: number;
  readonly iterations: number;
  readonly maxError: number;
  readonly pointAngle: number;
}

const MIN_ROWS = 8;
const CONFIDENCE = 0.999;

/** The correspondences of the pair's matches in pixels. */
export function pairRows(a: ImageFeatures, b: ImageFeatures, pair: PairMatches): Float64Array {
  const raw = new Float64Array(pair.matches.length * 2);
  for (let k = 0; k < pair.matches.length; k += 2) {
    const ai = pair.matches[k]!,
      bi = pair.matches[k + 1]!;
    raw.set(
      [
        a.keypoints[ai * 2]!,
        a.keypoints[ai * 2 + 1]!,
        b.keypoints[bi * 2]!,
        b.keypoints[bi * 2 + 1]!,
      ],
      k * 2,
    );
  }
  return raw;
}

/**
 * The base of one pair. `first` and `second` are the pair's cameras at one reference focal. The
 * fundamental matrix is fitted in the normalized coordinates of that focal, which differ from
 * the pixels by one scale: the Sampson distance against the threshold `maxError / f` of the
 * reference is then the Sampson distance in pixels against `maxError`, at every focal.
 */
export function pairBase(
  a: ImageFeatures,
  b: ImageFeatures,
  pair: PairMatches,
  first: Camera,
  second: Camera,
  controls: PairBaseControls,
): PairBase {
  const raw = pairRows(a, b, pair);
  const n = raw.length / 4;
  const seed = pairSeed(controls.seed, pair.first, pair.second);
  const homography =
    n >= MIN_ROWS
      ? estimateHomography(raw, {
          thresholdPx: controls.maxError,
          iterations: Math.min(controls.iterations, 500),
          seed,
        })
      : undefined;
  const base = { pair, raw, homographyInliers: homography?.inlierCount ?? 0 };
  if (n < MIN_ROWS) return { ...base, rows: [], trials: 0 };
  const points = normalizePixels(raw, first, second);
  const thresholdSquared =
    (controls.maxError / Math.min(first.fx, first.fy, second.fx, second.fy)) ** 2;
  const minIterations = Math.min(64, controls.iterations);
  const rng = random(seed);
  let best: Uint8Array | undefined,
    bestCount = 0,
    bestCost = Infinity,
    budget = controls.iterations,
    trials = 0;
  for (let trial = 0; trial < budget; trial++) {
    trials++;
    try {
      const F = estimateFundamental(points, sampleIndices(n, MIN_ROWS, rng));
      const score = classify(F, points, thresholdSquared);
      if (score.count > bestCount || (score.count === bestCount && score.cost < bestCost)) {
        best = score.mask;
        bestCount = score.count;
        bestCost = score.cost;
        const allInliers = Math.pow(bestCount / n, MIN_ROWS);
        const required =
          allInliers >= 1 ? 1 : Math.ceil(Math.log1p(-CONFIDENCE) / Math.log1p(-allInliers));
        budget = Math.min(budget, Math.max(minIterations, required));
      }
    } catch {
      /* degenerate minimal samples are normal in RANSAC */
    }
  }
  if (!best || bestCount < MIN_ROWS) return { ...base, rows: [], trials };
  // Local refit on the inliers, never to a weaker model.
  for (let pass = 0; pass < 4; pass++) {
    const indices = indicesOf(best);
    try {
      const F = estimateFundamental(points, indices);
      const score = classify(F, points, thresholdSquared);
      if (score.count < bestCount || (score.count === bestCount && score.cost >= bestCost)) break;
      best = score.mask;
      bestCount = score.count;
      bestCost = score.cost;
    } catch {
      break;
    }
  }
  return { ...base, rows: indicesOf(best), trials };
}

function indicesOf(mask: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
}

/**
 * The verification of the pair at the cameras `first` and `second`, in the shape of
 * `verifyImagePair`: its essential-matrix search runs on the base's rows only, which the
 * fundamental matrix already cleared of outliers, so that it stops after its minimum trials. The
 * counts of the full pair and its homography are the base's.
 */
export function verificationFromBase(
  base: PairBase,
  first: Camera,
  second: Camera,
  controls: PairBaseControls,
): ImagePairVerification {
  const n = base.raw.length / 4;
  const common = { invertibleCount: n, homographyInlierCount: base.homographyInliers };
  if (n < MIN_ROWS)
    return {
      ...common,
      inlierRows: new Uint32Array(0),
      reason: 'Insufficient invertible correspondences',
    };
  const rows = base.rows;
  if (rows.length < MIN_ROWS)
    return {
      ...common,
      inlierRows: new Uint32Array(0),
      reason: 'Could not find a valid two-view geometry',
    };
  const kept = new Float64Array(rows.length * 4);
  rows.forEach((row, k) => kept.set(base.raw.subarray(row * 4, row * 4 + 4), k * 4));
  const result = verifyImagePair(kept, first, second, {
    iterations: controls.iterations,
    seed: pairSeed(controls.seed, base.pair.first, base.pair.second),
    threshold: controls.maxError / Math.min(first.fx, first.fy, second.fx, second.fy),
    refinementIterations: 0,
    minParallaxDeg: Math.min(controls.pointAngle, 0.1),
    maxReprojectionErrorPx: controls.maxError,
    // The homography of the pair is the base's.
    homographyThresholdPx: controls.maxError,
    homographyIterations: 1,
  });
  return {
    ...result,
    ...common,
    inlierRows: Uint32Array.from(result.inlierRows, (k) => rows[k]!),
  };
}
