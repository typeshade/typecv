/**
 * The epipolar guide of the lab's guided matching (`?guided=1`).
 *
 * The guide is an essential matrix from the plain matches of a pair. The guided matcher then
 * keeps the candidates within a few pixels of each query's epipolar line. The mapper verifies
 * the chosen matches of every pair with the full RANSAC (`ransacIterations` hypotheses and the
 * homography test), so that verification classifies the pair. The guide is only a band for the
 * matcher, and a short RANSAC finds it: the plain matches pass the ratio test and the mutual
 * check, so most of them are inliers. Before 2026-10-20 the guide was a second full
 * verification, and each pair paid for two.
 */

/**
 * RANSAC hypotheses of the guide. For a 5-point sample, 50 draws give 0.999 confidence of an
 * all-inlier draw at 66 % inliers and 0.99 at 59 %.
 */
export const GUIDE_RANSAC_ITERATIONS = 50;

type PinholeCamera = { readonly fx: number; readonly fy: number };

/**
 * The `estimateEssentialRansacGpu` options of the guide. It is a short RANSAC on E and nothing
 * else: no pose recovery, no homography test, no refinement. Only the mapper's verification
 * classifies the pair, so the guide needs a band and no pose.
 */
export function guideEssentialOptions(
  first: PinholeCamera,
  second: PinholeCamera,
  maxErrorPx: number,
  ransacIterations: number,
  seed: number,
) {
  const iterations = Math.max(1, Math.min(GUIDE_RANSAC_ITERATIONS, ransacIterations));
  return {
    iterations,
    // The guide always draws its full count. Adaptive stopping would make the band depend on
    // the order of the draws.
    minIterations: iterations,
    seed,
    threshold: maxErrorPx / Math.min(first.fx, first.fy, second.fx, second.fy),
  };
}

/**
 * The essential matrix of the guide from the plain matches of a pair, row-major, or undefined
 * when the RANSAC finds no model (`GeometryEstimationError`). Any other error is rethrown.
 * `estimate` is `estimateEssentialRansacGpu` bound to a device.
 */
export async function guideEssential(
  estimate: (
    options: ReturnType<typeof guideEssentialOptions>,
  ) => Promise<{ E: ArrayLike<number> }>,
  options: ReturnType<typeof guideEssentialOptions>,
): Promise<ArrayLike<number> | undefined> {
  try {
    return (await estimate(options)).E;
  } catch (error) {
    if (error instanceof Error && error.name === 'GeometryEstimationError') return undefined;
    throw error;
  }
}

/** Guided matching runs on a pair with at least this many plain match entries (8 matches). */
export const GUIDED_MIN_MATCH_ENTRIES = 16;

/**
 * The guided trigger: guided matching (`?guided=1`) runs on a pair of the SIFT front end whose
 * plain matches have at least `GUIDED_MIN_MATCH_ENTRIES` entries.
 */
export function guidedMatchingRuns(
  sift: boolean,
  guidedMatching: boolean,
  matchEntries: number,
): boolean {
  return sift && guidedMatching && matchEntries >= GUIDED_MIN_MATCH_ENTRIES;
}

/** The guided matches replace the plain ones only when they are more. */
export function preferGuided(plainMatches: number, guidedMatches: number | undefined): boolean {
  return guidedMatches !== undefined && guidedMatches > plainMatches;
}
