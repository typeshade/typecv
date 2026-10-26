import { validateCamera } from '../../geometry/camera.ts';
import type {
  FeatureObservation,
  FeatureTrack,
  ImageFeatures,
  PairMatches,
} from '../../reconstruction/types.ts';

export function validateFeatures(
  views: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
): Map<number, ImageFeatures> {
  const byId = new Map<number, ImageFeatures>();
  for (const v of views) {
    if (!Number.isSafeInteger(v.id) || v.id < 0 || byId.has(v.id))
      throw new RangeError('View ids must be unique nonnegative safe integers');
    validateCamera(v.camera);
    if (v.keypoints.length % 2 || !v.keypoints.every(Number.isFinite))
      throw new RangeError('Keypoints must be finite packed pixel pairs');
    byId.set(v.id, v);
  }
  for (const pair of pairs) {
    const a = byId.get(pair.first),
      b = byId.get(pair.second);
    if (!a || !b || a === b || pair.matches.length % 2) throw new RangeError('Invalid match pair');
    if (
      pair.distances &&
      (pair.distances.length !== pair.matches.length / 2 ||
        !pair.distances.every((v) => Number.isFinite(v) && v >= 0))
    )
      throw new RangeError('Invalid match distances');
    for (let i = 0; i < pair.matches.length; i += 2)
      if (
        pair.matches[i]! >= a.keypoints.length / 2 ||
        pair.matches[i + 1]! >= b.keypoints.length / 2
      )
        throw new RangeError('Feature index outside view');
  }
  return byId;
}

/** Deterministic conflict-safe union: no track may contain two features from one image. */
export function buildTracks(
  views: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
): {
  tracks: FeatureTrack[];
  conflictingEdges: number;
  redundantEdges: number;
} {
  validateFeatures(views, pairs);
  const nodes: FeatureObservation[] = [],
    lookup = new Map<string, number>(),
    parent: number[] = [],
    members: Map<number, number>[] = [];
  const node = (viewId: number, featureId: number) => {
    const key = `${viewId}:${featureId}`;
    let id = lookup.get(key);
    if (id === undefined) {
      id = nodes.length;
      lookup.set(key, id);
      nodes.push({ viewId, featureId });
      parent.push(id);
      members.push(new Map([[viewId, featureId]]));
    }
    return id;
  };
  const root = (i: number): number => {
    let r = i;
    while (parent[r] !== r) r = parent[r]!;
    while (parent[i] !== i) {
      const n = parent[i]!;
      parent[i] = r;
      i = n;
    }
    return r;
  };
  const edges: { a: FeatureObservation; b: FeatureObservation; distance: number }[] = [];
  for (const pair of pairs)
    for (let i = 0; i < pair.matches.length; i += 2) {
      let a = { viewId: pair.first, featureId: pair.matches[i]! },
        b = { viewId: pair.second, featureId: pair.matches[i + 1]! };
      if (a.viewId > b.viewId) [a, b] = [b, a];
      edges.push({ a, b, distance: pair.distances?.[i / 2] ?? 0 });
    }
  edges.sort(
    (a, b) =>
      a.distance - b.distance ||
      a.a.viewId - b.a.viewId ||
      a.a.featureId - b.a.featureId ||
      a.b.viewId - b.b.viewId ||
      a.b.featureId - b.b.featureId,
  );
  let conflictingEdges = 0,
    redundantEdges = 0;
  for (const e of edges) {
    let a = root(node(e.a.viewId, e.a.featureId)),
      b = root(node(e.b.viewId, e.b.featureId));
    if (a === b) {
      redundantEdges++;
      continue;
    }
    if ([...members[a]!.keys()].some((id) => members[b]!.has(id))) {
      conflictingEdges++;
      continue;
    }
    if (members[a]!.size < members[b]!.size) [a, b] = [b, a];
    parent[b] = a;
    for (const [v, f] of members[b]!) members[a]!.set(v, f);
    members[b]!.clear();
  }
  const groups = new Map<number, FeatureObservation[]>();
  for (let i = 0; i < nodes.length; i++) {
    const r = root(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r)!.push(nodes[i]!);
  }
  const arrays = [...groups.values()]
    .filter((g) => g.length >= 2)
    .map((g) => g.sort((a, b) => a.viewId - b.viewId || a.featureId - b.featureId));
  arrays.sort((a, b) => a[0]!.viewId - b[0]!.viewId || a[0]!.featureId - b[0]!.featureId);
  return {
    tracks: arrays.map((observations, id) => ({ id, observations })),
    conflictingEdges,
    redundantEdges,
  };
}
