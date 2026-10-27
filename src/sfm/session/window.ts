import type { ImageFeatures, PairMatches, SparseReconstruction } from '../../contracts/scene.ts';
import { pairKey } from '../../contracts/keys.ts';
import { cameraCenter } from '../../geometry/pose.ts';

// The window policies of `IncrementalSfmSession.add` (docs/library-structure.md, section 3.3,
// `sfm/session`; phase 6). Each is a function of its inputs; the session applies the result.

/** @internal What `evictWindow` keeps of a window that outgrew its size. */
export interface WindowEviction {
  readonly images: ImageFeatures[];
  readonly edges: PairMatches[];
  /** The gauge views that the window keeps besides its newest views. */
  readonly anchors: readonly number[] | undefined;
  /** The window's model without the evicted views, when the session has a model and anchors. */
  readonly model?: SparseReconstruction;
}

/**
 * @internal The gauge views of a window that evicts: the first view while the session has no
 * model, else the farthest pair of registered recent views that share at least 12 landmarks, else
 * the initial pair of the last result.
 */
export function chooseAnchors(
  images: readonly ImageFeatures[],
  recentIds: ReadonlySet<number>,
  model: SparseReconstruction | undefined,
  initialPair: readonly number[] | undefined,
): readonly number[] | undefined {
  let anchors: readonly number[] | undefined = initialPair;
  // Retain the first reference while waiting for sufficient initial parallax.
  if (!model) anchors = [images[0]!.id];
  if (model) {
    // Shared-landmark counts of every registered pair, from one pass over the observations.
    const shared = new Map<string, number>();
    for (const point of model.landmarks) {
      const ids = point.observations.map((o) => o.viewId).sort((a, b) => a - b);
      for (let i = 0; i < ids.length; i++)
        for (let j = i + 1; j < ids.length; j++) {
          const key = pairKey(ids[i]!, ids[j]!);
          shared.set(key, (shared.get(key) ?? 0) + 1);
        }
    }
    const common = (a: number, b: number) => shared.get(pairKey(a, b)) ?? 0;
    // Move the local gauge to existing registered cameras; their world poses are preserved.
    const candidates = model.views.filter((view) => recentIds.has(view.id));
    let best = 0;
    for (let i = 0; i < candidates.length; i++)
      for (let j = i + 1; j < candidates.length; j++) {
        const a = candidates[i]!,
          b = candidates[j]!;
        const ac = cameraCenter(a.pose),
          bc = cameraCenter(b.pose);
        const distance = Math.hypot(...ac.map((v, k) => v - bc[k]!));
        if (distance > best && distance > 1e-8 && common(a.id, b.id) >= 12) {
          anchors = [a.id, b.id];
          best = distance;
        }
      }
  }
  return anchors;
}

/**
 * @internal The kept edges, followed by an edge from each anchor to each kept view that has none
 * and shares at least 8 landmarks with it (16 match entries), in anchor and view order.
 */
export function carryAnchorEdges(
  anchors: readonly number[],
  images: readonly ImageFeatures[],
  edges: readonly PairMatches[],
  model: SparseReconstruction,
): PairMatches[] {
  const carried = edges.slice();
  const existing = new Set(carried.map((pair) => pairKey(pair.first, pair.second)));
  // Carry real shared observations across an evicted chain; verify these edges normally.
  for (const anchor of anchors)
    for (const view of images) {
      if (view.id === anchor) continue;
      const key = pairKey(anchor, view.id);
      if (existing.has(key)) continue;
      const matches: number[] = [];
      for (const point of model.landmarks) {
        let a: number | undefined, b: number | undefined;
        for (const o of point.observations) {
          if (o.viewId === anchor) a = o.featureId;
          else if (o.viewId === view.id) b = o.featureId;
        }
        if (a !== undefined && b !== undefined) matches.push(a, b);
      }
      if (matches.length >= 16) {
        carried.push({ first: anchor, second: view.id, matches: Uint32Array.from(matches) });
        existing.add(key);
      }
    }
  return carried;
}

/**
 * @internal The window after an `add`: undefined while it holds at most `windowSize` views, else
 * its newest `windowSize - 2` views and the anchors, the edges among them, the carried edges and
 * the model without the evicted views.
 */
export function evictWindow(
  images: readonly ImageFeatures[],
  edges: readonly PairMatches[],
  windowSize: number,
  model: SparseReconstruction | undefined,
  initialPair: readonly number[] | undefined,
): WindowEviction | undefined {
  if (!(images.length > windowSize)) return undefined;
  const recent = images.slice(-(windowSize - 2));
  const recentIds = new Set(recent.map((view) => view.id));
  const anchors = chooseAnchors(images, recentIds, model, initialPair);
  const keep = new Set([...recentIds, ...(anchors ?? [])]);
  const kept = images.filter((view) => keep.has(view.id));
  const keptEdges = edges.filter((pair) => keep.has(pair.first) && keep.has(pair.second));
  if (!(model && anchors)) return { images: kept, edges: keptEdges, anchors };
  return {
    images: kept,
    edges: carryAnchorEdges(anchors, kept, keptEdges, model),
    anchors,
    model: {
      views: model.views.filter((view) => keep.has(view.id)),
      landmarks: model.landmarks
        .map((point) => ({
          ...point,
          observations: point.observations.filter((o) => keep.has(o.viewId)),
        }))
        .filter((point) => point.observations.length >= 2),
    },
  };
}
