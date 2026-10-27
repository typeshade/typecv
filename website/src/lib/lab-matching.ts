/**
 * The rules of the lab's matching stage (`docs/library-structure.md`, section 3.4, `matching`):
 * which pairs go to the mapper, which pairs the KLT tracks cover, and which feature sets the SfM
 * worker keeps for the next frames (descriptor retention).
 *
 * A match array holds two entries for each match: the feature index in each frame. So a minimum of
 * 16 entries is 8 matches. The minimums differ by path, and each keeps its value and its unit
 * (`docs/library-structure.md`, section 7, item 5): a descriptor pair needs 8 matches, a
 * precomputed pair 16 and a tracker-only pair 16 (`MIN_PAIR_MATCHES` of `klt-join.ts`).
 *
 * This module has no DOM and no worker globals, so a unit test runs it in Node.
 */
import { KLT_COVER_MATCHES } from './klt-join.ts';

/** A descriptor pair goes to the mapper with at least this many match entries (8 matches). */
export const PAIR_MIN_MATCH_ENTRIES = 16;

/**
 * A pair of precomputed matches (the keypoints and raw matches of another front end, the COLMAP
 * sample's SIFT) goes to the mapper with at least this many match entries (16 matches).
 */
export const PRECOMPUTED_MIN_MATCH_ENTRIES = 32;

/** True when a descriptor pair with `matchEntries` match entries goes to the mapper. */
export function admitsPair(matchEntries: number): boolean {
  return matchEntries >= PAIR_MIN_MATCH_ENTRIES;
}

/** True when a precomputed pair with `matchEntries` match entries goes to the mapper. */
export function admitsPrecomputedPair(matchEntries: number): boolean {
  return matchEntries >= PRECOMPUTED_MIN_MATCH_ENTRIES;
}

/**
 * True when the tracker matches of a pair cover it (`?kltMatching=tracks`, `docs/klt-tracking.md`
 * 3.6): `matchEntries` entries hold at least `KLT_COVER_MATCHES` matches. A covered pair skips the
 * descriptor matching.
 */
export function tracksCoverPair(matchEntries: number): boolean {
  return matchEntries / 2 >= KLT_COVER_MATCHES;
}

/** A keyframe becomes a reference when this many seconds have passed since the last reference. */
export const REFERENCE_SPACING_SECONDS = 0.5;

/** The worker keeps this many references: the first one, a long baseline, and the most recent. */
export const MAX_REFERENCES = 3;

/** Besides the references, the worker keeps the features of this many newest frames. */
export const ADJACENT_KEYFRAMES = 3;

/**
 * The references after the keyframe `frame` is registered. The references whose features are no
 * longer held go. `frame` becomes a reference when no reference is held, or when
 * `REFERENCE_SPACING_SECONDS` have passed since the last one. Above `MAX_REFERENCES`, the first
 * reference and the most recent ones stay.
 */
export function retainedReferences(
  references: readonly number[],
  features: ReadonlyMap<number, { readonly frame: { readonly time: number } }>,
  frame: { readonly id: number; readonly time: number },
): number[] {
  const kept = references.filter((id) => features.has(id));
  const reference = features.get(kept.at(-1) ?? -1);
  if (!reference || frame.time - reference.frame.time >= REFERENCE_SPACING_SECONDS)
    kept.push(frame.id);
  if (kept.length > MAX_REFERENCES) kept.splice(1, kept.length - MAX_REFERENCES);
  return kept;
}

/**
 * The frames whose features the worker keeps: the references and the `ADJACENT_KEYFRAMES` newest
 * of `ids`, in the order of `ids`.
 */
export function retainedFeatureIds(
  references: readonly number[],
  ids: Iterable<number>,
): Set<number> {
  return new Set([...references, ...[...ids].slice(-ADJACENT_KEYFRAMES)]);
}
