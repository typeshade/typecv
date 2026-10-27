/**
 * The defaults that a geometric operator and its GPU twin share, each written once
 * (docs/library-structure.md, section 5, phase 2). The CPU operator and the GPU operator read the
 * same entry; their bounds stay their own. The mapper's defaults are `sfm/config/defaults.ts`.
 */

/** The RANSAC seed of every operator and of the mapper (`seed` of their options). */
export const DEFAULT_SEED = 17;

/** `estimatePosePnP` and `estimatePosePnPGpu`; the inlier minimum is `pnpMinInliers`. */
export const PNP_DEFAULTS = {
  iterations: 1000,
  thresholdPx: 3,
  confidence: 0.999,
  seed: DEFAULT_SEED,
  refinementIterations: 20,
} as const;

/** The default `minInliers` of PnP over `n` correspondences. */
export const pnpMinInliers = (n: number): number => Math.min(6, n);

/** `triangulateTrackDetailed` and `triangulateTracksGpu`. */
export const TRIANGULATION_DEFAULTS = {
  maxErrorPx: 3,
  minAngleDeg: 0.5,
  maxPairHypotheses: 32,
} as const;

/** `estimateHomography` and `estimateHomographyGpu`. */
export const HOMOGRAPHY_DEFAULTS = {
  thresholdPx: 2,
  iterations: 500,
  confidence: 0.999,
  seed: DEFAULT_SEED,
} as const;

/** The homography trials of a pair verification: `min(iterations, 500)` (`ImagePairOptions`). */
export const imagePairHomographyIterations = (iterations: number): number =>
  Math.min(iterations, 500);
