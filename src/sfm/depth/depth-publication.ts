import type { IncrementalResult } from '../mapper/mapper.ts';
import type {
  InverseDepthTracking,
  DepthTrackingInput,
  DepthTrackingResult,
} from './inverse-depth-tracker.ts';
import type { ImageFeatures } from '../../reconstruction/types.ts';
import type { CameraObservationMask } from '../tracks/observation-selection.ts';
import type { ReconstructionTracking } from '../tracks/tracking.ts';
import { projectPoint } from '../../geometry/pose.ts';

export function depthTrackingInput(
  result: IncrementalResult,
  images: readonly ImageFeatures[],
  masks?: readonly CameraObservationMask[],
): DepthTrackingInput {
  return {
    tracks: result.tracking?.tracks ?? result.tracks,
    images,
    reconstruction: result.reconstruction,
    cameraObservationMasks: masks,
  };
}
/** Only the completed update publishes finite admitted geometry. Existing transient solver seeds
 * retain the mapper's native triangulation/PnP checks; they are not reported as depth posteriors.
 */
export function publishDepth<T extends IncrementalResult>(
  result: T,
  depth: DepthTrackingResult,
): T {
  const reconstruction = depth.reconstruction ?? result.reconstruction,
    views = new Map(reconstruction.views.map((v) => [v.id, v]));
  let error = 0,
    count = 0;
  for (const p of reconstruction.landmarks)
    for (const o of p.observations) {
      const v = views.get(o.viewId)!,
        q = projectPoint(v.camera, v.pose, p.xyz);
      if (q) {
        error += Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]);
        count++;
      }
    }
  return {
    ...result,
    tracks: depth.snapshot.tracks.map((t) => ({
      id: t.id,
      observations: t.observations.map((o) => ({ viewId: o.viewId, featureId: o.featureId })),
    })),
    ...(result.tracking
      ? {
          tracking: {
            ...result.tracking,
            tracks: depth.snapshot.tracks.map((t) => ({
              id: t.id,
              observations: t.observations.map((o) => ({
                viewId: o.viewId,
                featureId: o.featureId,
              })),
            })),
            untriangulatedTrackIds: depth.snapshot.tracks
              .filter((t) => t.status !== 'finite')
              .map((t) => t.id),
            depthDeferredTrackIds: depth.snapshot.tracks
              .filter((t) => t.status === 'estimating')
              .map((t) => t.id),
          },
        }
      : {}),
    reconstruction,
    inverseDepth: depth.snapshot,
    diagnostics: {
      ...result.diagnostics,
      meanReprojectionErrorPx: count ? error / count : Infinity,
    },
  };
}
export function recordDeferredDepth(
  tracker: InverseDepthTracking,
  error: { readonly tracking?: ReconstructionTracking },
  images: readonly ImageFeatures[],
  masks?: readonly CameraObservationMask[],
) {
  const depth = tracker.update({
    tracks: error.tracking?.tracks ?? [],
    images,
    cameraObservationMasks: masks,
  });
  Object.assign(error, { inverseDepth: depth.snapshot });
}
