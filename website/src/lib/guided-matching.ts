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
 * The `verifyImagePairGpu` options of the guide. It is a short RANSAC on E with no refinement.
 * The guide skips the homography test (one hypothesis is the minimum the options accept),
 * because only the mapper's verification classifies the pair.
 */
export function guideVerificationOptions(
  first: PinholeCamera,
  second: PinholeCamera,
  maxErrorPx: number,
  ransacIterations: number,
  seed: number,
) {
  const iterations = Math.max(1, Math.min(GUIDE_RANSAC_ITERATIONS, ransacIterations));
  return {
    iterations,
    seed,
    threshold: maxErrorPx / Math.min(first.fx, first.fy, second.fx, second.fy),
    refinementIterations: 0,
    maxReprojectionErrorPx: maxErrorPx,
    homographyThresholdPx: maxErrorPx,
    homographyIterations: 1,
  };
}

/**
 * E = [t]× R in row-major order, with x2ᵀ E x1 = 0 for normalized coordinates and
 * X2 = R X1 + t. `R` is row-major.
 */
export function essentialFromPose(R: ArrayLike<number>, t: ArrayLike<number>): Float64Array {
  const tx = [0, -t[2]!, t[1]!, t[2]!, 0, -t[0]!, -t[1]!, t[0]!, 0];
  const E = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      E[r * 3 + c] = tx[r * 3]! * R[c]! + tx[r * 3 + 1]! * R[3 + c]! + tx[r * 3 + 2]! * R[6 + c]!;
  return E;
}

/** The guided matches replace the plain ones only when they are more. */
export function preferGuided(plainMatches: number, guidedMatches: number | undefined): boolean {
  return guidedMatches !== undefined && guidedMatches > plainMatches;
}
