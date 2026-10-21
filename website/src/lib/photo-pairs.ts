/**
 * Which earlier views the lab matches a new frame against, and which features it keeps resident.
 *
 * A video arrives in time order. A window of the first frame and a few recent frames covers the
 * pairs that overlap. Photographs have no order, so the same window drops most pairs: twelve
 * photos gave 38 of 66 pairs, and the mapper registered 9 of 12.
 *
 * The photo rule:
 *
 * 1. While fewer than `PHOTO_ALL_PAIRS_MAX` frames came before, match the new frame against
 *    every retained view.
 * 2. After that, match it against the retrieval candidates: the top `PHOTO_RETRIEVAL_K` views by
 *    the similarity of the VLAD-32 global descriptor, the `PHOTO_ORDER_NEIGHBOURS` latest views
 *    (the order of the set) and the first view.
 *
 * `PHOTO_ALL_PAIRS_MAX` equals the session window (`LAB_OPTIONS.windowSize`, 32). The session
 * evicts older views, so a pair with an evicted view has no use. Below the window the cost is
 * N (N - 1) / 2 matchings, which is 66 for 12 photos and 496 for 32.
 */

/** The same two values as `Profile` in `sfm.worker.ts`. */
export type PairProfile = 'video' | 'photo';

/** A photo set up to this size is matched in all pairs. It equals the session window. */
export const PHOTO_ALL_PAIRS_MAX = 32;
/** Retrieved neighbours of a new photo above that size. */
export const PHOTO_RETRIEVAL_K = 10;
/** Latest views kept as candidates above that size (the order of the set, if any). */
export const PHOTO_ORDER_NEIGHBOURS = 3;

export interface PartnerCandidate {
  readonly id: number;
  /** The global descriptor of the view, when it has one. */
  readonly global?: Float32Array;
}

/** The ids whose resident features survive after a frame is registered. */
export function retainedFeatureIds(
  profile: PairProfile,
  references: readonly number[],
  ids: readonly number[],
): Set<number> {
  // Photographs keep every view the session still holds: any of them may overlap the next photo.
  if (profile === 'photo') return new Set(ids);
  return new Set([...references, ...ids.slice(-3)]);
}

/** True when `partnerIds` needs the global descriptors of the frame and its candidates. */
export function needsGlobals(profile: PairProfile, seen: number): boolean {
  return profile === 'photo' && seen >= PHOTO_ALL_PAIRS_MAX;
}

const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

/**
 * The retained views that a new frame is matched against. `seen` is the number of frames that
 * came before it, retained or not. `global` is the global descriptor of the new frame.
 */
export function partnerIds(
  profile: PairProfile,
  candidates: readonly PartnerCandidate[],
  seen = candidates.length,
  global?: Float32Array,
): number[] {
  if (profile !== 'photo' || seen < PHOTO_ALL_PAIRS_MAX) return candidates.map((c) => c.id);
  const sorted = [...candidates].sort((a, b) => a.id - b.id);
  const chosen = new Set<number>();
  if (sorted.length) chosen.add(sorted[0]!.id);
  for (const c of sorted.slice(-PHOTO_ORDER_NEIGHBOURS)) chosen.add(c.id);
  if (global) {
    const ranked = sorted
      .filter((c) => c.global)
      .map((c) => ({ id: c.id, s: dot(global, c.global!) }))
      .sort((a, b) => b.s - a.s || a.id - b.id);
    for (const r of ranked.slice(0, PHOTO_RETRIEVAL_K)) chosen.add(r.id);
  }
  return sorted.filter((c) => chosen.has(c.id)).map((c) => c.id);
}
