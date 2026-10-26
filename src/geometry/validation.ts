import { validateCamera } from './camera.ts';
import { validatePose } from './pose.ts';
import type { RegisteredView, SparseReconstruction } from '../contracts/scene.ts';

export function validateReconstruction(input: SparseReconstruction): void {
  const views = new Map<number, RegisteredView>();
  for (const v of input.views) {
    if (!Number.isSafeInteger(v.id) || v.id < 0 || views.has(v.id))
      throw new RangeError('Invalid/duplicate registered view id');
    validateCamera(v.camera);
    validatePose(v.pose);
    views.set(v.id, v);
  }
  const points = new Set<number>(),
    ownedFeatures = new Set<string>();
  for (const p of input.landmarks) {
    if (
      !Number.isSafeInteger(p.trackId) ||
      p.trackId < 0 ||
      points.has(p.trackId) ||
      p.xyz.length !== 3 ||
      !p.xyz.every(Number.isFinite)
    )
      throw new RangeError('Invalid/duplicate landmark');
    points.add(p.trackId);
    const seen = new Set<number>();
    if (p.observations.length < 2)
      throw new RangeError('Landmarks require at least two observations');
    for (const o of p.observations) {
      const key = `${o.viewId}:${o.featureId}`;
      if (
        !views.has(o.viewId) ||
        seen.has(o.viewId) ||
        !Number.isSafeInteger(o.featureId) ||
        o.featureId < 0 ||
        o.pixel.length !== 2 ||
        !o.pixel.every(Number.isFinite) ||
        ownedFeatures.has(key)
      )
        throw new RangeError('Invalid or multiply owned landmark observation');
      seen.add(o.viewId);
      ownedFeatures.add(key);
    }
  }
}
