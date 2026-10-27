/**
 * The rules of the lab's feature stage (`docs/library-structure.md`, section 3.4, `features`). The
 * feature minimum of a frame: the SfM worker (`processFrame`) and the multi-video job
 * (`MultiVideoJob.frame`) call it, and each rejects the frame in its own way. The feature cap of a
 * frame: both workers derive it when they start. This module has no DOM and no worker globals, so
 * a unit test runs it in Node.
 */

/** A frame with fewer detected features than this is rejected ("Too few detected features"). */
export const MIN_FEATURES = 8;

/** True when a frame's count of detected features is under `MIN_FEATURES`. */
export function tooFewFeatures(count: number): boolean {
  return count < MIN_FEATURES;
}

/** The SIFT front end's limit on the features of a frame: the feature cap stops here. */
export const MAX_FEATURE_CAP = 16384;

/**
 * The feature cap of a frame at `scale` (`analysisScale` of `working-budget.ts`): `base` features
 * at 640 px, grown with the pixel count up to `MAX_FEATURE_CAP`. A base of 4096 gives 4096 at
 * 640 px and 16384 at 1280 px. The SfM worker passes the `maxFeatures` of its controls, and the
 * multi-video worker 4096 when the query sets no cap.
 */
export function featureCap(base: number, scale: number): number {
  return Math.min(MAX_FEATURE_CAP, Math.round(base * scale));
}
