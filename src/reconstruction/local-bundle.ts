import { cameraCenter } from '../geometry/pose.ts';
import type { SparseReconstruction } from './types.ts';

/** @internal A local problem keeps every observation of each selected landmark. */
export interface LocalBundleProblem {
  readonly reconstruction: SparseReconstruction;
  readonly variableViewIds: readonly number[];
  readonly anchorViewId: number;
  readonly scaleViewId: number;
}

/** @internal Return undefined when the local graph cannot retain a fixed, connected gauge. */
export function extractLocalBundle(
  model: SparseReconstruction,
  variableViewIds: readonly number[],
): LocalBundleProblem | undefined {
  const variable = new Set(variableViewIds);
  if (!variable.size || variable.size !== variableViewIds.length) return undefined;
  const landmarks = model.landmarks.filter((p) =>
      p.observations.some((o) => variable.has(o.viewId)),
    ),
    neighbors = new Map<number, number[]>();
  for (const p of landmarks) {
    const first = p.observations[0]!.viewId;
    for (const o of p.observations) {
      if (!neighbors.has(o.viewId)) neighbors.set(o.viewId, []);
      if (o.viewId !== first) {
        neighbors.get(o.viewId)!.push(first);
        neighbors.get(first)!.push(o.viewId);
      }
    }
  }
  if (variableViewIds.some((id) => !neighbors.has(id))) return undefined;
  const views = model.views.filter((v) => neighbors.has(v.id)),
    boundary = views.filter((v) => !variable.has(v.id));
  if (views.length !== neighbors.size || boundary.length < 2) return undefined;
  const centers = boundary.map((v) => cameraCenter(v.pose));
  let anchor = 0,
    scale = 1,
    baseline = 0;
  for (let i = 0; i < centers.length; i++)
    for (let j = i + 1; j < centers.length; j++) {
      const distance = Math.hypot(...centers[i]!.map((x, k) => x - centers[j]![k]!));
      if (distance > baseline) {
        baseline = distance;
        anchor = i;
        scale = j;
      }
    }
  // Match the solver's nonzero-baseline guard; both gauge cameras remain fixed.
  if (!(baseline > 1e-10)) return undefined;
  const anchorViewId = boundary[anchor]!.id,
    connected = new Set([anchorViewId]),
    pending = [anchorViewId];
  for (let i = 0; i < pending.length; i++)
    for (const id of neighbors.get(pending[i]!)!)
      if (!connected.has(id)) {
        connected.add(id);
        pending.push(id);
      }
  if (connected.size !== views.length) return undefined;
  return {
    reconstruction: { views, landmarks },
    variableViewIds,
    anchorViewId,
    scaleViewId: boundary[scale]!.id,
  };
}

/** @internal Preserve fixed cameras, unrelated points and the full observation set. */
export function mergeLocalBundle(
  model: SparseReconstruction,
  problem: LocalBundleProblem,
  refined: SparseReconstruction,
): SparseReconstruction {
  const variable = new Set(problem.variableViewIds),
    selected = new Set(problem.reconstruction.landmarks.map((p) => p.trackId)),
    views = new Map(refined.views.map((v) => [v.id, v])),
    points = new Map(
      refined.landmarks.filter((p) => selected.has(p.trackId)).map((p) => [p.trackId, p]),
    );
  return {
    views: model.views.map((v) => (variable.has(v.id) ? { ...v, pose: views.get(v.id)!.pose } : v)),
    landmarks: model.landmarks.map((p) => {
      const point = points.get(p.trackId);
      return point ? { ...p, xyz: point.xyz } : p;
    }),
  };
}
