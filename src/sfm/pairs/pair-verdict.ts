import type { ImagePairVerification } from '../../geometry/image-pair.ts';
import type { PairMatches } from '../../reconstruction/types.ts';
import type { IncrementalControls, PairVerification, SeedCandidate } from '../mapper/mapper.ts';
import { selectPairRows } from '../tracks/tracking.ts';

/**
 * @internal The verdict on one pair from its verification: the inlier matches, the diagnostic row
 * and, for a pair that can seed the map, its candidate. `verifyPairs` and the focal search share it.
 */
export function judgePair(
  pair: PairMatches,
  verification: ImagePairVerification,
  controls: Pick<IncrementalControls, 'initialAngle' | 'minInitial'>,
): { accepted?: PairMatches; diagnostic: PairVerification; candidate?: SeedCandidate } {
  const { initialAngle, minInitial } = controls;
  const base = { first: pair.first, second: pair.second, matches: verification.invertibleCount },
    result = verification.geometry,
    hCount = verification.homographyInlierCount;
  if (!result)
    return {
      diagnostic: {
        ...base,
        inliers: 0,
        homographyInliers: hCount,
        positiveDepthCount: 0,
        medianAngleDeg: 0,
        seedEligible: false,
        reason: verification.reason,
      },
    };
  const accepted = selectPairRows(pair, verification.inlierRows);
  const dominant =
    hCount >= 0.85 * verification.invertibleCount && hCount >= 0.9 * result.inlierCount;
  const eligible =
    !dominant &&
    result.diagnostics.medianParallaxDeg >= initialAngle &&
    result.positiveDepthCount >= minInitial;
  return {
    accepted,
    diagnostic: {
      ...base,
      inliers: result.inlierCount,
      homographyInliers: hCount,
      positiveDepthCount: result.positiveDepthCount,
      ...(result.diagnostics.directionCompatibleCount
        ? { directionCompatibleCount: result.diagnostics.directionCompatibleCount }
        : {}),
      medianAngleDeg: result.diagnostics.medianParallaxDeg,
      seedEligible: eligible,
      reason: eligible
        ? undefined
        : dominant
          ? 'Homography-dominated initialization'
          : result.positiveDepthCount < minInitial
            ? 'Insufficient initial points'
            : 'Insufficient initial parallax',
    },
    candidate: eligible
      ? {
          pair: accepted,
          result,
          score: result.positiveDepthCount * Math.min(10, result.diagnostics.medianParallaxDeg),
        }
      : undefined,
  };
}
