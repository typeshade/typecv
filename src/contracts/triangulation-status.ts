/**
 * The status of a track triangulation, as the CPU oracle (`triangulateTrackDetailed`) returns it
 * and as `triangulation.shade.ts` writes it in `statuses` (docs/library-structure.md, section 5,
 * phase 2). The kernel keeps its own literals until it exports them (section 4, rule 6).
 */
export const TriangulationStatus = {
  accepted: 0,
  insufficientObservations: 1,
  noConsensus: 3,
  lowParallax: 4,
  depthSensitivity: 5,
} as const;
