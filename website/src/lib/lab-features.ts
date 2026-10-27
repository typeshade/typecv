/**
 * The rule of the lab's feature stage (`docs/library-structure.md`, section 3.4, `features`): the
 * feature minimum of a frame. The SfM worker (`processFrame`) and the multi-video job
 * (`MultiVideoJob.frame`) call it, and each rejects the frame in its own way. This module has no
 * DOM and no worker globals, so a unit test runs it in Node.
 */

/** A frame with fewer detected features than this is rejected ("Too few detected features"). */
export const MIN_FEATURES = 8;

/** True when a frame's count of detected features is under `MIN_FEATURES`. */
export function tooFewFeatures(count: number): boolean {
  return count < MIN_FEATURES;
}
