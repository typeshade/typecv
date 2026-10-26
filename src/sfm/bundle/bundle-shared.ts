import { clonePose } from '../../geometry/pose.ts';
import type { SparseReconstruction } from '../../reconstruction/types.ts';
import type { BundleAdjustmentOptions } from './bundle-adjustment.ts';
import { projectBundlePoint } from './bundle-directions.ts';

/** @internal The LM stop rule shared by the CPU and GPU solvers. */
export function bundleStallTolerance(options: BundleAdjustmentOptions): number {
  const tolerance = options.stallTolerance ?? 1e-9;
  if (!Number.isFinite(tolerance) || tolerance < 0 || tolerance >= 1)
    throw new RangeError('stallTolerance must be in [0, 1)');
  return tolerance;
}
function clone(input: SparseReconstruction): SparseReconstruction {
  return {
    views: input.views.map((v) => ({ ...v, camera: { ...v.camera }, pose: clonePose(v.pose) })),
    landmarks: input.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.slice(),
      observations: p.observations.map((o) => ({ ...o, pixel: [o.pixel[0], o.pixel[1]] as const })),
    })),
  };
}
function metrics(
  input: SparseReconstruction,
  huber: number,
  finitePoints = input.landmarks.length,
): { cost: number; rmse: number; observations: number } {
  const views = new Map(input.views.map((v) => [v.id, v]));
  let cost = 0,
    squares = 0,
    observations = 0;
  for (let i = 0; i < input.landmarks.length; i++) {
    const p = input.landmarks[i]!;
    for (const o of p.observations) {
      const v = views.get(o.viewId)!,
        pixel = projectBundlePoint(v.camera, v.pose, p.xyz, i >= finitePoints);
      if (!pixel) return { cost: Infinity, rmse: Infinity, observations };
      const e = Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]);
      cost += e <= huber ? e * e : 2 * huber * e - huber * huber;
      squares += e * e;
      observations++;
    }
  }
  return { cost, rmse: Math.sqrt(squares / observations), observations };
}

/** @internal Shared objective and snapshot helpers for the joint focal reference. */
export { clone as cloneBundleReconstruction, metrics as bundleMetrics };
