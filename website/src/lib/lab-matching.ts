/**
 * The rules of the lab's matching stage (`docs/library-structure.md`, section 3.4, `matching`):
 * which pairs go to the mapper, which pairs the KLT tracks cover, and which feature sets the SfM
 * worker keeps for the next frames (descriptor retention). Beside them, the camera of a frame,
 * which guided matching and the mapper read, and the match lines of the overlay.
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

/** The camera of an analysis frame: pinhole intrinsics in analysis pixels and two radial terms. */
export type FrameCamera = {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
};

/**
 * The camera of a frame in analysis pixels: the frame's own calibration (a sample's), else a
 * camera with the focal `focalRatio` times the source width, scaled to the analysis frame, the
 * principal point at the centre and no distortion. The SfM worker passes the focal search's
 * estimate once it ran, else the assumed focal; the multi-video job passes the focal ratio of the
 * video.
 */
export function frameCamera(
  frame: {
    readonly width: number;
    readonly height: number;
    readonly sourceWidth?: number;
    readonly sourceHeight?: number;
    readonly camera?: FrameCamera;
  },
  focalRatio: number,
): FrameCamera {
  const sourceWidth = frame.sourceWidth ?? frame.width;
  const sourceHeight = frame.sourceHeight ?? frame.height;
  const sourceFocal = sourceWidth * focalRatio;
  return (
    frame.camera ?? {
      fx: (sourceFocal * frame.width) / sourceWidth,
      fy: (sourceFocal * frame.height) / sourceHeight,
      cx: (frame.width - 1) / 2,
      cy: (frame.height - 1) / 2,
      k1: 0,
      k2: 0,
    }
  );
}

/** The overlay draws the lines of at most this many match entries (80 matches). */
export const MATCH_LINE_ENTRIES = 160;

/**
 * The candidate match lines between the last matched frame and this one, for the overlay: four
 * numbers for each match, the earlier frame's point scaled to this frame and this frame's point,
 * in this frame's pixels. The lines of the first `MATCH_LINE_ENTRIES` match entries are drawn.
 */
export function matchLines(
  otherPoints: Float64Array,
  otherFrame: { readonly width: number; readonly height: number },
  points: Float64Array,
  frame: { readonly width: number; readonly height: number },
  matches: ArrayLike<number>,
): number[] {
  const lines: number[] = [];
  for (let i = 0; i < Math.min(matches.length, MATCH_LINE_ENTRIES); i += 2) {
    const a = matches[i] * 2,
      b = matches[i + 1] * 2;
    lines.push(
      (otherPoints[a] * frame.width) / otherFrame.width,
      (otherPoints[a + 1] * frame.height) / otherFrame.height,
      points[b],
      points[b + 1],
    );
  }
  return lines;
}
