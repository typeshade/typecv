import {
  isInverseDepthTracker,
  type DepthTrackingResult,
  type InverseDepthTracking,
} from '../depth/inverse-depth-tracker.ts';
import {
  depthTrackingInput,
  publishDepth,
  recordDeferredDepth,
} from '../depth/depth-publication.ts';
import { copyReconstruction } from '../model/copy.ts';
import { ReconstructionInitializationError } from './errors.ts';
import type { ImageFeatures } from '../../contracts/scene.ts';
import type { CameraObservationMask } from '../../contracts/observation-mask.ts';
import type { IncrementalOptions, IncrementalResult } from '../mapper/mapper.ts';

/**
 * The depth post-step of a driver (docs/library-structure.md, section 3.3, `sfm/protocol`). A run
 * with a depth tracker (`depthTracking`) gives its completed map to the tracker. The driver then
 * publishes the admitted geometry, and only then calls `onReconstruction`. A run that does not
 * initialize records the deferred depth of its tracks. The CPU drivers use `withDepthStep`. The
 * device driver updates the tracker asynchronously with its own bundle adjustment and uses the
 * parts.
 */

/** @internal Throw a RangeError when `depthTracking` is set and is not a depth tracker. */
export function checkDepthTracker(options: Pick<IncrementalOptions, 'depthTracking'>): void {
  const tracker = options.depthTracking;
  if (tracker !== undefined && !isInverseDepthTracker(tracker))
    throw new RangeError('Invalid depth tracker');
}

/** @internal Publish the tracker's update of a completed run, then call `onReconstruction` with a copy. */
export function publishTrackedDepth<T extends IncrementalResult>(
  result: T,
  depth: DepthTrackingResult,
  options: Pick<IncrementalOptions, 'onReconstruction'>,
): T {
  const published = publishDepth(result, depth);
  options.onReconstruction?.(copyReconstruction(published.reconstruction));
  return published;
}

/** @internal A run that did not initialize records the deferred depth of its tracks. */
export function recordFailedDepth(
  tracker: InverseDepthTracking | undefined,
  error: unknown,
  images: readonly ImageFeatures[],
  masks: readonly CameraObservationMask[] | undefined,
): void {
  if (tracker && error instanceof ReconstructionInitializationError)
    recordDeferredDepth(tracker, error, images, masks);
}

/**
 * @internal A synchronous run with the depth post-step. With a tracker, the run's own
 * `onReconstruction` is off: the mapper's intermediate maps are not the tracker's admitted map.
 */
export function withDepthStep(
  images: readonly ImageFeatures[],
  options: IncrementalOptions,
  run: (options: IncrementalOptions) => IncrementalResult,
): IncrementalResult {
  checkDepthTracker(options);
  const tracker = options.depthTracking;
  try {
    const result = run(tracker ? { ...options, onReconstruction: undefined } : options);
    if (!tracker) return result;
    return publishTrackedDepth(
      result,
      tracker.update(depthTrackingInput(result, images, options.cameraObservationMasks)),
      options,
    );
  } catch (error) {
    recordFailedDepth(tracker, error, images, options.cameraObservationMasks);
    throw error;
  }
}
