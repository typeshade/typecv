import { cameraCenter, projectPoint } from '../../geometry/pose.ts';
import { cameraAtRatio } from '../focal/focal-search.ts';
import { copyReconstruction } from '../mapper/mapper.ts';
import { mergeBundleRefinements, planConnectedBundles } from '../bundle/local-bundle.ts';
import {
  validateReconstruction,
  type BundleAdjustmentOptions,
  type BundleAdjustmentResult,
} from '../bundle/bundle-adjustment.ts';
import type { SparseReconstruction } from '../../reconstruction/types.ts';

/** Whole-output metrics: unlike mapper inlier diagnostics, every exported observation counts. */
export function measureHistory(model: SparseReconstruction, huberPx = 1.5) {
  const views = new Map(model.views.map((v) => [v.id, v]));
  let error = 0,
    cost = 0,
    observations = 0;
  for (const point of model.landmarks)
    for (const o of point.observations) {
      const view = views.get(o.viewId)!,
        pixel = projectPoint(view.camera, view.pose, point.xyz),
        e = pixel ? Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]) : Infinity;
      error += e;
      cost += e <= huberPx ? e * e : 2 * huberPx * e - huberPx * huberPx;
      observations++;
    }
  return { observations, cost, meanErrorPx: observations ? error / observations : Infinity };
}

/** Finish an archive against its complete observation set, not just the last local window.
 * A shared physical focal is an optional *solver initialization*, never a post-hoc overwrite.
 * A candidate must improve the same complete robust objective before it can be published.
 */
export async function refineHistory(
  model: SparseReconstruction,
  options: BundleAdjustmentOptions & { readonly commonFocalRatios?: readonly number[] },
  solve: (
    model: SparseReconstruction,
    options: BundleAdjustmentOptions,
  ) => Promise<BundleAdjustmentResult>,
) {
  validateReconstruction(model);
  const original = copyReconstruction(model),
    huber = options.huberPx ?? 1.5,
    before = measureHistory(original, huber),
    reports: BundleAdjustmentResult['diagnostics'][] = [];
  let reconstruction = original,
    deferred = false;
  if (!Number.isFinite(before.cost)) deferred = true;
  else if (
    (options.iterations ?? 8) > 0 &&
    original.landmarks.length &&
    original.views.length >= 2
  ) {
    const anchorViewId = options.anchorViewId ?? original.views[0]!.id,
      anchor = original.views.find((v) => v.id === anchorViewId)!;
    let scaleViewId = options.scaleViewId,
      baseline = 1e-10;
    if (scaleViewId === undefined) {
      const origin = cameraCenter(anchor.pose);
      for (const view of original.views) {
        const center = cameraCenter(view.pose),
          distance = Math.hypot(...center.map((x, i) => x - origin[i]!));
        if (distance > baseline) {
          baseline = distance;
          scaleViewId = view.id;
        }
      }
    }
    if (scaleViewId === undefined) deferred = true;
    else {
      const fullOptions = { ...options, anchorViewId, scaleViewId },
        partition = planConnectedBundles(original, fullOptions).partition,
        ratios: readonly (number | undefined)[] =
          partition || !options.commonFocalRatios?.length ? [undefined] : options.commonFocalRatios;
      // Disconnected components cannot jointly identify a physical focal. In that case
      // keep their carried intrinsics and use fixed-boundary solves only.
      deferred ||= partition?.sharedFocalDeferred ?? false;
      let bestCost = before.cost;
      for (const ratio of ratios) {
        if (ratio !== undefined && !(Number.isFinite(ratio) && ratio > 0))
          throw new RangeError('History focal seeds must be finite and positive');
        const initial =
            ratio === undefined
              ? original
              : {
                  ...original,
                  views: original.views.map((v) => ({
                    ...v,
                    camera: cameraAtRatio(v.camera, ratio),
                  })),
                },
          plan = planConnectedBundles(initial, fullOptions),
          refined: SparseReconstruction[] = [];
        for (const problem of plan.problems) {
          try {
            const result = await solve(problem.reconstruction, problem.options);
            reports.push(result.diagnostics);
            refined.push(result.reconstruction);
          } catch (error) {
            // A mandatory resident floor can exceed a device budget. Preserve the
            // unsolved geometry; cancellations, device loss and other faults propagate.
            if (!(
              error instanceof RangeError &&
              /working budget|stay resident|binding limit/.test(error.message)
            ))
              throw error;
            deferred = true;
          }
        }
        deferred ||= (plan.partition?.deferredViewIds.length ?? 0) > 0;
        const merged = mergeBundleRefinements(original, refined),
          candidate = {
            ...merged,
            // A numerical solve owns coordinates, not admission or feature identity.
            landmarks: original.landmarks.map((point, i) => ({
              ...point,
              xyz: merged.landmarks[i]!.xyz,
            })),
          },
          after = measureHistory(candidate, huber);
        if (Number.isFinite(after.cost) && after.cost <= bestCost) {
          bestCost = after.cost;
          reconstruction = candidate;
        }
      }
    }
  }
  return {
    reconstruction,
    reports,
    deferred,
    before,
    after: measureHistory(reconstruction, huber),
  };
}

/** Carry an accepted archive solve back to native local track ids before another update. */
export function refinedHistoryWindow(local: SparseReconstruction, full: SparseReconstruction) {
  const views = new Map(full.views.map((v) => [v.id, v])),
    points = new Map(
      full.landmarks.flatMap((p) =>
        p.observations.map((o) => [o.viewId + ':' + o.featureId, p] as const),
      ),
    );
  return copyReconstruction({
    views: local.views.map((v) => views.get(v.id) ?? v),
    landmarks: local.landmarks.map((p) => {
      const solved = p.observations
        .map((o) => points.get(o.viewId + ':' + o.featureId))
        .find(Boolean);
      return solved ? { ...p, xyz: solved.xyz } : p;
    }),
  });
}
