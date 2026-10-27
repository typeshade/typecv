/**
 * The failure reasons of two-view geometry and pair verification, one spelling on every tier
 * (docs/library-structure.md, section 5, phase 2). A reason is a message that callers read:
 * `isCheiralityFailure` tests its prefix, and the lab's records keep it.
 */

/** The prefix of a cheirality failure, which a pair verification retries. */
export const CHEIRALITY_FAILURE_PREFIX = 'Cheirality failed';

export const TwoViewFailure = {
  noGeometry: 'Could not find a valid two-view geometry',
  cheirality: `${CHEIRALITY_FAILURE_PREFIX}: low parallax, planar scene, or invalid matching`,
  lowParallax: 'Low parallax: insufficient baseline for stable reconstruction',
  fewReliablePoints: 'Insufficient reliable points after parallax and reprojection filtering',
  gpuRefinement: 'GPU two-view numerical refinement or median selection failed',
  gpuRecovery: 'GPU two-view recovery failed',
  fewInvertible: 'Insufficient invertible correspondences',
} as const;

/** The reason of each status of the recovery kernel (`recovery.shade.ts`); 0 is success. */
export const RECOVERY_FAILURES: readonly string[] = [
  '',
  TwoViewFailure.noGeometry,
  TwoViewFailure.cheirality,
  TwoViewFailure.lowParallax,
  TwoViewFailure.fewReliablePoints,
  TwoViewFailure.gpuRefinement,
];
