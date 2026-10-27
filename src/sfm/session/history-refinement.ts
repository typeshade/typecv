import { cameraCenter, clonePose, projectPoint, transformPoint } from '../../geometry/pose.ts';
import { unprojectPixel } from '../../geometry/camera.ts';
import { refineTrackPoint, triangulateLinear } from '../../geometry/multiview.ts';
import { cameraAtRatio } from '../focal/focal-search.ts';
import { copyReconstruction } from '../model/copy.ts';
import { mergeBundleRefinements, planConnectedBundles } from '../bundle/local-bundle.ts';
import {
  type BundleAdjustmentOptions,
  type BundleAdjustmentResult,
} from '../bundle/bundle-adjustment.ts';
import { validateReconstruction } from '../../geometry/validation.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../../contracts/scene.ts';
import { observationKey } from '../../contracts/keys.ts';

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

/**
 * @internal The mapper input that the archive's tracks give: one keypoint for each observation
 * (in landmark order) and one match in each pair of a landmark's views, for the pairs with at
 * least `minMatches` matches. The cameras are the archive's. The rebuild of a restart focal
 * (`refineHistory`, `rebuildAtRatio`) maps these tracks again.
 */
export function historyTracks(
  model: SparseReconstruction,
  minMatches = 16,
): { images: ImageFeatures[]; pairs: PairMatches[] } {
  const rank = new Map(model.views.map((v, i) => [v.id, i])),
    keypoints = model.views.map((): number[] => []),
    matches = new Map<string, number[]>();
  for (const point of model.landmarks) {
    const track = point.observations
      .map((o) => {
        const view = rank.get(o.viewId)!,
          list = keypoints[view]!;
        list.push(o.pixel[0], o.pixel[1]);
        return { view, index: list.length / 2 - 1 };
      })
      .sort((a, b) => a.view - b.view);
    for (let a = 0; a < track.length; a++)
      for (let b = a + 1; b < track.length; b++) {
        const key = `${track[a]!.view}:${track[b]!.view}`;
        let list = matches.get(key);
        if (!list) matches.set(key, (list = []));
        list.push(track[a]!.index, track[b]!.index);
      }
  }
  const pairs: PairMatches[] = [];
  for (const [key, list] of matches) {
    if (list.length / 2 < minMatches) continue;
    const [first, second] = key.split(':').map(Number) as [number, number];
    pairs.push({
      first: model.views[first]!.id,
      second: model.views[second]!.id,
      matches: Uint32Array.from(list),
    });
  }
  pairs.sort(
    (a, b) => rank.get(a.second)! - rank.get(b.second)! || rank.get(a.first)! - rank.get(b.first)!,
  );
  return {
    images: model.views.map((v, i) => ({
      id: v.id,
      camera: { ...v.camera },
      keypoints: Float64Array.from(keypoints[i]!),
    })),
    pairs,
  };
}

/**
 * The archive with the poses of `rebuilt` and the camera at `ratio`, and each landmark
 * triangulated again from its own observations with those poses. A landmark that does not land
 * in front of all its cameras (an outlier of the rebuilt geometry) is placed on the ray of one of
 * its observations at the median depth of the others, the first such point in front of all its
 * cameras: the bundle adjustment requires that, and its robust loss then judges the landmark.
 * Undefined when the rebuild did not register every view of the archive or a landmark has no
 * such point: the restart then has no map at its focal.
 */
function reseeded(
  original: SparseReconstruction,
  rebuilt: SparseReconstruction,
  ratio: number,
): SparseReconstruction | undefined {
  const posed = new Map(rebuilt.views.map((v) => [v.id, v]));
  if (!original.views.every((v) => posed.has(v.id))) return undefined;
  const views = original.views.map((v) => ({
      ...v,
      camera: cameraAtRatio(v.camera, ratio),
      pose: clonePose(posed.get(v.id)!.pose),
    })),
    byId = new Map(views.map((v) => [v.id, v])),
    tracks = original.landmarks.map((point) =>
      point.observations.map((o) => ({
        camera: byId.get(o.viewId)!.camera,
        pose: byId.get(o.viewId)!.pose,
        pixel: o.pixel,
      })),
    ),
    inFront = (observations: (typeof tracks)[number], X: Float64Array) =>
      observations.every((o) => projectPoint(o.camera, o.pose, X) !== undefined),
    points = tracks.map((observations) => {
      const initial = triangulateLinear(observations),
        X = initial && refineTrackPoint(initial, observations);
      return X && inFront(observations, X) ? X : undefined;
    }),
    depths = points
      .flatMap((X, i) => (X ? [transformPoint(tracks[i]![0]!.pose, X)[2]!] : []))
      .sort((a, b) => a - b),
    depth = depths[Math.floor(depths.length / 2)];
  const landmarks = [];
  for (const [i, point] of original.landmarks.entries()) {
    let X = points[i];
    for (const o of tracks[i]!) {
      if (X || depth === undefined) break;
      const ray = unprojectPixel(o.camera, o.pixel);
      if (!ray) continue;
      // The camera point (x·d, y·d, d) in world coordinates: Rᵀ(Xc − t).
      const c = [
          ray[0] * depth - o.pose.t[0]!,
          ray[1] * depth - o.pose.t[1]!,
          depth - o.pose.t[2]!,
        ],
        R = o.pose.R,
        candidate = Float64Array.from(
          { length: 3 },
          (_, k) => R[k]! * c[0]! + R[k + 3]! * c[1]! + R[k + 6]! * c[2]!,
        );
      if (inFront(tracks[i]!, candidate)) X = candidate;
    }
    if (!X) return undefined;
    landmarks.push({ ...point, xyz: X });
  }
  return { views, landmarks };
}

/** Finish an archive against its complete observation set, not just the last local window.
 * A shared physical focal is an optional *solver initialization*, never a post-hoc overwrite.
 * A candidate must improve the same complete robust objective before it can be published.
 * With `rebuildAtRatio`, every restart focal after the first (the current calibration) starts
 * from a map rebuilt at that focal, not from the archive's poses and points: a map built at a
 * wrong focal is a different local minimum (docs/auto-focal.md 2.3, item 3; section 10).
 */
export async function refineHistory(
  model: SparseReconstruction,
  options: BundleAdjustmentOptions & {
    readonly commonFocalRatios?: readonly number[];
    /** @internal A map of the archive's views built at `ratio`; undefined when none builds. */
    readonly rebuildAtRatio?: (ratio: number) => Promise<SparseReconstruction | undefined>;
  },
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
      const { rebuildAtRatio, ...solveOptions } = options,
        fullOptions = { ...solveOptions, anchorViewId, scaleViewId },
        partition = planConnectedBundles(original, fullOptions).partition,
        ratios: readonly (number | undefined)[] =
          partition || !options.commonFocalRatios?.length ? [undefined] : options.commonFocalRatios;
      // Disconnected components cannot jointly identify a physical focal. In that case
      // keep their carried intrinsics and use fixed-boundary solves only.
      deferred ||= partition?.sharedFocalDeferred ?? false;
      let bestCost = before.cost;
      for (const [index, ratio] of ratios.entries()) {
        if (ratio !== undefined && !(Number.isFinite(ratio) && ratio > 0))
          throw new RangeError('History focal seeds must be finite and positive');
        let initial: SparseReconstruction | undefined =
          ratio === undefined
            ? original
            : {
                ...original,
                views: original.views.map((v) => ({
                  ...v,
                  camera: cameraAtRatio(v.camera, ratio),
                })),
              };
        const fromRebuild = ratio !== undefined && index > 0 && !!rebuildAtRatio;
        if (fromRebuild) {
          const rebuilt = await rebuildAtRatio!(ratio);
          initial = rebuilt && reseeded(original, rebuilt, ratio);
          // No map at this focal: the restart has nothing to start from.
          if (!initial) continue;
        }
        const plan = planConnectedBundles(initial, fullOptions),
          refined: SparseReconstruction[] = [];
        let refused = false;
        for (const problem of plan.problems) {
          try {
            const result = await solve(problem.reconstruction, problem.options);
            reports.push(result.diagnostics);
            refined.push(result.reconstruction);
          } catch (error) {
            // A rebuilt start that the solver refuses (for example a point behind a camera)
            // is no candidate; the other starts still compete.
            if (fromRebuild && error instanceof RangeError) {
              refused = true;
              break;
            }
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
        if (refused) continue;
        deferred ||= (plan.partition?.deferredViewIds.length ?? 0) > 0;
        // A rebuilt start is in its own frame: an unsolved part keeps the rebuilt geometry.
        const merged = mergeBundleRefinements(fromRebuild ? initial : original, refined),
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
        p.observations.map((o) => [observationKey(o.viewId, o.featureId), p] as const),
      ),
    );
  return copyReconstruction({
    views: local.views.map((v) => views.get(v.id) ?? v),
    landmarks: local.landmarks.map((p) => {
      const solved = p.observations
        .map((o) => points.get(observationKey(o.viewId, o.featureId)))
        .find(Boolean);
      return solved ? { ...p, xyz: solved.xyz } : p;
    }),
  });
}
