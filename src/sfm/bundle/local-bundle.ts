import { cameraCenter } from '../../geometry/pose.ts';
import type { BundleAdjustmentOptions } from './bundle-adjustment.ts';
import type { SparseReconstruction } from '../../contracts/scene.ts';

/** A window/filter can disconnect observations of cameras already registered in one map. */
export interface BundleAdjustmentPartition {
  readonly components: readonly (readonly number[])[];
  /** Carried poses held fixed: at least two per optimized component; no alignment is inferred. */
  readonly fixedViewIds: readonly number[];
  /** No points, no nonzero fixed baseline, or insufficient fixed boundary cameras. */
  readonly deferredViewIds: readonly number[];
  /** Disconnected components cannot independently refine a focal shared across the map. */
  readonly sharedFocalDeferred: boolean;
}

/** @internal Plan connected solves within an existing map, preserving its carried coordinates. */
export function planConnectedBundles(
  model: SparseReconstruction,
  options: BundleAdjustmentOptions,
): {
  problems: { reconstruction: SparseReconstruction; options: BundleAdjustmentOptions }[];
  partition?: BundleAdjustmentPartition;
} {
  const parent = new Map(model.views.map((v) => [v.id, v.id]));
  const root = (id: number): number => {
    let r = id;
    while (parent.get(r)! !== r) r = parent.get(r)!;
    while (id !== r) {
      const next = parent.get(id)!;
      parent.set(id, r);
      id = next;
    }
    return r;
  };
  for (const p of model.landmarks) {
    const first = root(p.observations[0]!.viewId);
    for (const o of p.observations) parent.set(root(o.viewId), first);
  }
  const groups = new Map<number, number[]>();
  for (const v of model.views) {
    const r = root(v.id);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r)!.push(v.id);
  }
  // Preserve the original problem/options and its numerical order for connected input.
  if (groups.size === 1) return { problems: [{ reconstruction: model, options }] };
  const points = new Map<number, SparseReconstruction['landmarks'][number][]>();
  for (const p of model.landmarks) {
    const r = root(p.observations[0]!.viewId);
    if (!points.has(r)) points.set(r, []);
    points.get(r)!.push(p);
  }
  const byId = new Map(model.views.map((v) => [v.id, v])),
    variable = options.variableViewIds && new Set(options.variableViewIds),
    problems: { reconstruction: SparseReconstruction; options: BundleAdjustmentOptions }[] = [],
    fixedViewIds: number[] = [],
    deferredViewIds: number[] = [];
  for (const [r, ids] of groups) {
    const landmarks = points.get(r) ?? [],
      boundary = ids.filter((id) => !variable?.has(id));
    let gauge: [number, number] | undefined,
      baseline = 1e-10;
    for (let i = 0; i < boundary.length; i++)
      for (let j = i + 1; j < boundary.length; j++) {
        const a = boundary[i]!,
          b = boundary[j]!,
          ac = cameraCenter(byId.get(a)!.pose),
          bc = cameraCenter(byId.get(b)!.pose),
          distance = Math.hypot(...ac.map((x, k) => x - bc[k]!));
        if (distance > baseline) {
          baseline = distance;
          gauge = [a, b];
        }
      }
    if (!landmarks.length || !gauge) {
      deferredViewIds.push(...ids);
      continue;
    }
    const [anchorViewId, scaleViewId] = gauge;
    fixedViewIds.push(
      ...ids.filter(
        (id) => id === anchorViewId || id === scaleViewId || (variable && !variable.has(id)),
      ),
    );
    problems.push({
      reconstruction: { views: ids.map((id) => byId.get(id)!), landmarks },
      options: {
        ...options,
        anchorViewId,
        scaleViewId,
        variableViewIds: ids.filter(
          (id) => id !== anchorViewId && id !== scaleViewId && (!variable || variable.has(id)),
        ),
        refineSharedFocal: false,
        focalGroups: undefined,
      },
    });
  }
  return {
    problems,
    partition: {
      components: [...groups.values()],
      fixedViewIds,
      deferredViewIds,
      sharedFocalDeferred: options.refineSharedFocal ?? false,
    },
  };
}

/** @internal Replace solved values only; retain isolated/deferred cameras and unrelated points. */
export function mergeBundleRefinements(
  model: SparseReconstruction,
  refined: readonly SparseReconstruction[],
): SparseReconstruction {
  const views = new Map(refined.flatMap((m) => m.views.map((v) => [v.id, v] as const))),
    points = new Map(refined.flatMap((m) => m.landmarks.map((p) => [p.trackId, p] as const)));
  return {
    views: model.views.map((v) => views.get(v.id) ?? v),
    landmarks: model.landmarks.map((p) => points.get(p.trackId) ?? p),
  };
}

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
