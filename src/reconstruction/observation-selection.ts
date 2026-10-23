import { validateFeatures } from './tracks.ts';
import type { ImageFeatures, PairMatches } from './types.ts';

export interface CameraObservationMask {
  readonly viewId: number;
  /** 1 permits a background hypothesis; 0 separates the feature from camera/static-map estimation. */
  readonly values: Uint8Array;
  readonly source: 'manual' | 'geometry' | 'learned' | 'ground-truth';
}
export interface CameraObservationSelection {
  readonly masks: readonly CameraObservationMask[];
  /** Original rows, not geometrically verified object tracks or motion classifications. */
  readonly separatedPairs: readonly PairMatches[];
  readonly inputMatchCount: number;
  readonly backgroundMatchCount: number;
  /** Masks for future/evicted/missing views; they were not applied in this map/window. */
  readonly unappliedViewIds: readonly number[];
}
export interface ObservationExclusion {
  /** Image-edge coordinates in [0,1], as in the existing gameplay HUD exclusion convention. */
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
}
/** @internal Session evidence must not alias a result or a caller's arrays. */
export function copyCameraSelection(
  selection: CameraObservationSelection,
): CameraObservationSelection {
  return {
    ...selection,
    masks: snapshotCameraMasks(selection.masks)!,
    separatedPairs: selection.separatedPairs.map((p) => ({
      ...p,
      matches: p.matches.slice(),
      distances: p.distances?.slice(),
    })),
    unappliedViewIds: [...selection.unappliedViewIds],
  };
}
/** RGB and feature ids stay unchanged. Padding is an explicit descriptor-support assumption. */
export function cameraMaskFromRegions(
  image: ImageFeatures,
  width: number,
  height: number,
  regions: readonly ObservationExclusion[],
  source: CameraObservationMask['source'],
  paddingPx = 0,
): CameraObservationMask {
  if (
    ![width, height].every((x) => Number.isSafeInteger(x) && x > 0) ||
    !Number.isFinite(paddingPx) ||
    paddingPx < 0
  )
    throw new RangeError('Invalid camera mask image dimensions/padding');
  for (const r of regions)
    if (
      !r ||
      ![r.left, r.top, r.right, r.bottom].every(Number.isFinite) ||
      r.left < 0 ||
      r.top < 0 ||
      r.right > 1 ||
      r.bottom > 1 ||
      r.left >= r.right ||
      r.top >= r.bottom
    )
      throw new RangeError('Invalid normalized observation exclusion');
  if (image.keypoints.length % 2 || !image.keypoints.every(Number.isFinite))
    throw new RangeError('Invalid mask keypoints');
  const values = Uint8Array.from({ length: image.keypoints.length / 2 }, (_, i) => {
    const x = image.keypoints[2 * i]! + 0.5,
      y = image.keypoints[2 * i + 1]! + 0.5;
    return regions.some(
      (r) =>
        x >= r.left * width - paddingPx &&
        x < r.right * width + paddingPx &&
        y >= r.top * height - paddingPx &&
        y < r.bottom * height + paddingPx,
    )
      ? 0
      : 1;
  });
  return snapshotCameraMasks([{ viewId: image.id, values, source }])![0]!;
}
/** @internal Snapshot before matching, session updates or any queued GPU operation. */
export function snapshotCameraMasks(masks: readonly CameraObservationMask[] | undefined) {
  if (masks === undefined) return undefined;
  if (!Array.isArray(masks)) throw new RangeError('cameraObservationMasks must be an array');
  const seen = new Set<number>();
  return masks.map((m: CameraObservationMask) => {
    if (
      !m ||
      !Number.isSafeInteger(m.viewId) ||
      m.viewId < 0 ||
      seen.has(m.viewId) ||
      !(m.values instanceof Uint8Array) ||
      m.values.some((x) => x !== 0 && x !== 1) ||
      !['manual', 'geometry', 'learned', 'ground-truth'].includes(m.source)
    )
      throw new RangeError('Invalid camera observation mask');
    seen.add(m.viewId);
    return { ...m, values: m.values.slice() };
  });
}
const key = (a: number, b: number) => `${Math.min(a, b)}:${Math.max(a, b)}`;
/** @internal Filter before E/H/R consensus; retained ids and distances always refer to original features. */
export function selectCameraObservations(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  masks: readonly CameraObservationMask[] | undefined,
): {
  pairs: readonly PairMatches[];
  selection?: CameraObservationSelection;
  changedPairKeys: ReadonlySet<string>;
} {
  const changedPairKeys = new Set<string>();
  if (!masks?.length) return { pairs, changedPairKeys };
  const byId = validateFeatures(images, pairs),
    copied = snapshotCameraMasks(masks)!,
    active = new Map(copied.filter((m) => byId.has(m.viewId)).map((m) => [m.viewId, m]));
  for (const [id, m] of active)
    if (m.values.length !== byId.get(id)!.keypoints.length / 2)
      throw new RangeError('Camera observation mask must address every original feature');
  let inputMatchCount = 0,
    backgroundMatchCount = 0;
  const separatedPairs: PairMatches[] = [];
  const filtered = pairs.map((p) => {
    const used: number[] = [],
      apart: number[] = [],
      keptDistances: number[] = [],
      apartDistances: number[] = [];
    for (let i = 0; i < p.matches.length; i += 2) {
      const a = p.matches[i]!,
        b = p.matches[i + 1]!,
        keep = (active.get(p.first)?.values[a] ?? 1) && (active.get(p.second)?.values[b] ?? 1);
      (keep ? used : apart).push(a, b);
      if (p.distances) (keep ? keptDistances : apartDistances).push(p.distances[i / 2]!);
    }
    inputMatchCount += p.matches.length / 2;
    backgroundMatchCount += used.length / 2;
    if (!apart.length) return p;
    changedPairKeys.add(key(p.first, p.second));
    separatedPairs.push({
      first: p.first,
      second: p.second,
      matches: Uint32Array.from(apart),
      ...(p.distances ? { distances: Float64Array.from(apartDistances) } : {}),
    });
    return {
      first: p.first,
      second: p.second,
      matches: Uint32Array.from(used),
      ...(p.distances ? { distances: Float64Array.from(keptDistances) } : {}),
    };
  });
  return {
    pairs: filtered,
    changedPairKeys,
    selection: {
      masks: copied,
      separatedPairs,
      inputMatchCount,
      backgroundMatchCount,
      unappliedViewIds: copied.filter((m) => !byId.has(m.viewId)).map((m) => m.viewId),
    },
  };
}
