import { estimateHomography } from '../geometry/homography.ts';
import {
  verifyImagePairWithHomography,
  type ImagePairVerification,
} from '../geometry/image-pair.ts';
import type { Camera } from '../geometry/camera.ts';
import {
  eightPoint,
  normalizePixels,
  projectEssential,
  random,
  sampleIndices,
} from '../sfm/two-view.ts';
import { pairSeed } from './identity.ts';
import type { ImageFeatures, PairMatches } from './types.ts';

/**
 * The pair work of the focal search that does not depend on the focal (docs/auto-focal.md,
 * section 8). A trial of the essential-matrix search draws eight rows and solves the normalized
 * eight-point system. Hartley's normalization removes the scale of the coordinates, so the draw
 * and the solution are the same at every focal, up to rounding. Only the step after it depends
 * on the focal: `E(f) = K(f)ᵀ F K(f)`, projected to the nearest essential matrix, and then the
 * scoring of the trial. The base of a pair holds the matches, their homography and the
 * eight-point solutions of its trials. The search derives each grid point's verification from it.
 */
export interface PairBase {
  readonly pair: PairMatches;
  /** The raw correspondences [x1, y1, x2, y2] of the pair's matches. */
  readonly raw: Float64Array;
  readonly homographyInliers: number;
  /** The cameras that the solutions are expressed in. */
  readonly reference: readonly [Camera, Camera];
  /** The eight-point solution of trial `k` in the coordinates of `reference`; null when degenerate. */
  readonly trial: (k: number) => Float64Array | null;
}

/** The controls of the base and of its derivation: the mapper's own numerical controls. */
export interface PairBaseControls {
  readonly seed: number;
  readonly iterations: number;
  readonly maxError: number;
  readonly pointAngle: number;
}

const MIN_ROWS = 8;

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
 * trials are drawn lazily, in the order and from the generator that `reconstructTwoView` uses for
 * the seed of the pair, so that trial `k` is the trial `k` of a search at any focal.
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
  const points = n >= MIN_ROWS ? normalizePixels(raw, first, second) : new Float64Array(0);
  const rng = random(seed);
  const trials: (Float64Array | null)[] = [];
  const trial = (k: number): Float64Array | null => {
    while (trials.length <= k) {
      const rows = sampleIndices(n, MIN_ROWS, rng);
      let solution: Float64Array | null = null;
      try {
        solution = eightPoint(points, rows);
      } catch {
        /* a degenerate sample is normal in RANSAC */
      }
      trials.push(solution);
    }
    return trials[k]!;
  };
  return {
    pair,
    raw,
    homographyInliers: homography?.inlierCount ?? 0,
    reference: [first, second],
    trial,
  };
}

/**
 * The verification of the pair at the cameras `first` and `second`, in the shape of
 * `verifyImagePair`: its consensus search takes the solutions of the base, scaled to the cameras
 * and projected to essential matrices, and scores them as the full search does. The count of
 * the pair's homography inliers is the base's.
 */
export function verificationFromBase(
  base: PairBase,
  first: Camera,
  second: Camera,
  controls: PairBaseControls,
): ImagePairVerification {
  const [ref1, ref2] = base.reference;
  // x₂ᵀ F x₁ = 0 in coordinates scaled by f/f_ref: rows and columns of F scale by the inverse.
  const left = [first.fx / ref1.fx, first.fy / ref1.fy, 1],
    right = [second.fx / ref2.fx, second.fy / ref2.fy, 1];
  return verifyImagePairWithHomography(base.raw, first, second, base.homographyInliers, {
    iterations: controls.iterations,
    seed: pairSeed(controls.seed, base.pair.first, base.pair.second),
    threshold: controls.maxError / Math.min(first.fx, first.fy, second.fx, second.fy),
    refinementIterations: 0,
    minParallaxDeg: Math.min(controls.pointAngle, 0.1),
    maxReprojectionErrorPx: controls.maxError,
    homographyThresholdPx: controls.maxError,
    hypotheses: (k) => {
      const F = base.trial(k);
      if (!F) return undefined;
      const scaled = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) scaled[i * 3 + j] = F[i * 3 + j]! * right[i]! * left[j]!;
      return projectEssential(scaled);
    },
  });
}
