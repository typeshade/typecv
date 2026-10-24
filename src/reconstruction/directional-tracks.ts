import { cameraProjection, unprojectPixel } from '../geometry/camera.ts';
import type { BundleDirection } from './bundle-directions.ts';
import type { FeatureTrack, ImageFeatures, SparseReconstruction } from './types.ts';
import type { TriangulationObservation } from '../geometry/multiview.ts';
import { bearingSpreadDeg, relativeRotationControls } from '../geometry/rotation.ts';
import type { RegisteredView } from './types.ts';

const automaticSupport = relativeRotationControls();

/** @internal Reuse the rotation estimator's minimum consensus and calibrated spatial-spread gate. */
export function supportedBundleDirections(
  directions: readonly BundleDirection[],
  views: readonly RegisteredView[],
): BundleDirection[] {
  const byId = new Map(views.map((v) => [v.id, v]));
  let selected = [...directions];
  while (selected.length >= automaticSupport.minInliers) {
    const bearings = new Map<number, number[][]>();
    for (const d of selected)
      for (const o of d.observations) {
        const b = unprojectPixel(byId.get(o.viewId)!.camera, o.pixel);
        if (!b) continue;
        if (!bearings.has(o.viewId)) bearings.set(o.viewId, []);
        bearings.get(o.viewId)!.push([b[0], b[1], 1]);
      }
    const supported = new Set(
        [...bearings]
          .filter(
            ([, b]) =>
              b.length >= automaticSupport.minInliers &&
              bearingSpreadDeg(b) >= automaticSupport.minSpreadDeg,
          )
          .map(([id]) => id),
      ),
      next = selected.filter((d) => d.observations.every((o) => supported.has(o.viewId)));
    if (next.length === selected.length) return selected;
    selected = next;
  }
  return [];
}

/** @internal World-bearing seed shared by the object and scene protocols; never estimates depth. */
export function fitBundleDirection(
  observations: readonly TriangulationObservation[],
  maxErrorPx: number,
): Float64Array | undefined {
  if (observations.length < 2) return undefined;
  const direction = new Float64Array(3);
  for (const {
    camera,
    pose: { R },
    pixel,
  } of observations) {
    const bearing = unprojectPixel(camera, pixel);
    if (!bearing) return undefined;
    const norm = Math.hypot(...bearing, 1);
    for (let k = 0; k < 3; k++)
      direction[k] += (R[k]! * bearing[0] + R[3 + k]! * bearing[1] + R[6 + k]!) / norm;
  }
  const norm = Math.hypot(...direction);
  if (!(norm > 1e-12)) return undefined;
  for (let k = 0; k < 3; k++) direction[k] /= norm;
  for (const {
    camera,
    pose: { R },
    pixel,
  } of observations) {
    const q = Float64Array.from(
        { length: 3 },
        (_, k) =>
          R[3 * k]! * direction[0]! + R[3 * k + 1]! * direction[1]! + R[3 * k + 2]! * direction[2]!,
      ),
      projected = cameraProjection(camera, q);
    if (
      !projected ||
      Math.hypot(projected.pixel[0] - pixel[0], projected.pixel[1] - pixel[1]) > maxErrorPx
    )
      return undefined;
  }
  return direction;
}

/**
 * @internal Fit unresolved tracks to world bearings, conditional on the current cameras.
 * Every participating registered observation must fit; this is not semantic sky detection,
 * a static-scene decision or proof of infinity. Recompute after poses/observations change;
 * ordinary triangulation remains free to resolve a track once observed parallax allows it.
 * Host-side f64 seed/selection uses already available keypoints; BA residuals run on its backend.
 */
export function pendingBundleDirections(
  model: SparseReconstruction,
  tracks: readonly FeatureTrack[],
  images: ReadonlyMap<number, ImageFeatures>,
  maxErrorPx: number,
): BundleDirection[] {
  const finite = new Set(model.landmarks.map((p) => p.trackId)),
    views = new Map(model.views.map((v) => [v.id, v])),
    result: BundleDirection[] = [];
  for (const track of tracks) {
    if (finite.has(track.id)) continue;
    const observations = track.observations
      .filter((o) => views.has(o.viewId))
      .map((o) => {
        const image = images.get(o.viewId)!;
        return {
          ...o,
          pixel: [
            image.keypoints[2 * o.featureId]!,
            image.keypoints[2 * o.featureId + 1]!,
          ] as const,
        };
      });
    // A final-map hypothesis needs repeat support beyond one two-view compatibility test.
    // Shorter tracks remain pending; callers with an explicit hypothesis can still use raw BA.
    if (observations.length < 3) continue;
    const direction = fitBundleDirection(
      observations.map((o) => ({ ...views.get(o.viewId)!, pixel: o.pixel })),
      maxErrorPx,
    );
    if (direction) result.push({ trackId: track.id, direction, observations });
  }
  return supportedBundleDirections(result, model.views);
}
