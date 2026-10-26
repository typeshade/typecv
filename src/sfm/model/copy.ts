import type { SparseReconstruction } from '../../reconstruction/types.ts';

/** @internal Copy all numerical state crossing a session or observer boundary. */
export function copyReconstruction(model: SparseReconstruction): SparseReconstruction {
  return {
    views: model.views.map((view) => ({
      ...view,
      camera: { ...view.camera },
      pose: { R: view.pose.R.slice(), t: view.pose.t.slice() },
    })),
    landmarks: model.landmarks.map((point) => ({
      ...point,
      xyz: point.xyz.slice(),
      observations: point.observations.map((observation) => ({
        ...observation,
        pixel: [...observation.pixel] as [number, number],
      })),
    })),
  };
}
