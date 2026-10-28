/**
 * The worker's side of the KLT join (`docs/klt-tracking.md` 3.3 and 3.5): merge the matches that
 * the tracker emits into the mapper's pairs, and read the verdict of the pair verification on
 * them. The functions take plain arrays and maps so that a test runs them without a device.
 */

/** A pair as the mapper session takes it. */
export interface MapperPair {
  first: number;
  second: number;
  /** Feature index pairs: the index in `first`, the index in `second`. */
  matches: Uint32Array;
  distances?: Float64Array;
  /** `klt` for a pair of tracker matches only, `merged` with descriptor matches too, absent for descriptor matches only. */
  source?: 'klt' | 'merged';
  /** The matches of the pair that came from the tracker and were not descriptor matches already. */
  kltMatches?: number;
}

/** What `KltTracker.join` returns for one earlier keyframe. */
export interface TrackerPair {
  first: number;
  second: number;
  matches: Uint32Array;
  /** The id of the track behind each match. */
  tracks: Uint32Array;
}

/** A pair needs this many matches to go to the mapper, as a descriptor pair does. */
export const MIN_PAIR_MATCHES = 16;

/** The key of one tracker match: pair and feature indexes. */
export const matchKey = (first: number, second: number, a: number, b: number) =>
  `${first}:${second}:${a}:${b}`;

export interface MergedPairs {
  pairs: MapperPair[];
  /** The key (`matchKey`) of each tracker match that the pairs hold, with its track id. */
  tracked: Map<string, number>;
  /** Tracker matches added to the pairs, tracker matches the descriptor matches already held, and matches left out. */
  added: number;
  duplicate: number;
  conflicting: number;
}

/**
 * Merge the tracker's matches into the descriptor pairs (section 3.3). A match found by both is
 * one match and keeps the descriptor distance. A tracker match that disagrees with a descriptor
 * match on either feature is left out: the descriptor match stays. A tracker match that is new
 * carries a distance of 0. A pair that only the tracker found goes to the mapper when it has
 * `MIN_PAIR_MATCHES` matches. Pairs of descriptor matches only pass unchanged.
 */
export function mergeTrackerPairs(
  descriptor: readonly MapperPair[],
  tracker: readonly TrackerPair[],
  minMatches = MIN_PAIR_MATCHES,
): MergedPairs {
  const byFirst = new Map(descriptor.map((pair) => [pair.first, pair])),
    out = new Map<number, MapperPair>(),
    tracked = new Map<string, number>();
  let added = 0,
    duplicate = 0,
    conflicting = 0;
  for (const pair of descriptor) out.set(pair.first, pair);
  for (const t of tracker) {
    const base = byFirst.get(t.first),
      forward = new Map<number, number>(),
      backward = new Map<number, number>();
    if (base)
      for (let i = 0; i < base.matches.length; i += 2) {
        forward.set(base.matches[i]!, base.matches[i + 1]!);
        backward.set(base.matches[i + 1]!, base.matches[i]!);
      }
    const rows: number[] = [],
      distances: number[] = [];
    let fresh = 0;
    for (let r = 0; r < t.tracks.length; r++) {
      const a = t.matches[2 * r]!,
        b = t.matches[2 * r + 1]!;
      if (forward.get(a) === b) {
        duplicate++;
        tracked.set(matchKey(t.first, t.second, a, b), t.tracks[r]!);
      } else if (forward.has(a) || backward.has(b)) {
        conflicting++;
      } else {
        rows.push(a, b);
        distances.push(0);
        tracked.set(matchKey(t.first, t.second, a, b), t.tracks[r]!);
        fresh++;
      }
    }
    if (base) {
      if (!fresh) continue;
      out.set(t.first, {
        ...base,
        matches: Uint32Array.from([...base.matches, ...rows]),
        distances: Float64Array.from([
          ...(base.distances ?? new Float64Array(base.matches.length / 2)),
          ...distances,
        ]),
        source: 'merged',
        kltMatches: fresh,
      });
      added += fresh;
    } else if (rows.length / 2 >= minMatches) {
      out.set(t.first, {
        first: t.first,
        second: t.second,
        matches: Uint32Array.from(rows),
        distances: Float64Array.from(distances),
        source: 'klt',
        kltMatches: fresh,
      });
      added += fresh;
    } else {
      // Too few for a pair of their own: the mapper never sees them, so the tracks get no verdict.
      for (let r = 0; r < rows.length; r += 2)
        tracked.delete(matchKey(t.first, t.second, rows[r]!, rows[r + 1]!));
    }
  }
  return {
    pairs: [...out.values()].sort((a, b) => a.first - b.first || a.second - b.second),
    tracked,
    added,
    duplicate,
    conflicting,
  };
}

/**
 * The verdict of the pair verification on the tracker's matches (section 3.5, item 6): for each
 * track, the matches that a verified pair of the frame `second` kept and the ones it dropped. A
 * pair that failed verification is absent from `verified` and gives no verdict.
 */
export function trackerVerdicts(
  tracked: ReadonlyMap<string, number>,
  verified: readonly { first: number; second: number; matches: ArrayLike<number> }[] | undefined,
  second: number,
): Map<number, { inlier: number; outlier: number }> {
  const out = new Map<number, { inlier: number; outlier: number }>();
  if (!verified) return out;
  for (const pair of verified) {
    if (pair.second !== second) continue;
    const kept = new Set<string>();
    for (let i = 0; i < pair.matches.length; i += 2)
      kept.add(matchKey(pair.first, pair.second, pair.matches[i]!, pair.matches[i + 1]!));
    const prefix = `${pair.first}:${pair.second}:`;
    for (const [key, track] of tracked) {
      if (!key.startsWith(prefix)) continue;
      const entry = out.get(track) ?? { inlier: 0, outlier: 0 };
      if (kept.has(key)) entry.inlier++;
      else entry.outlier++;
      out.set(track, entry);
    }
  }
  return out;
}

/** The fewest tracks that decide the keyframe gate of `kltGate` (`docs/klt-tracking.md` 3.6). */
export const KLT_GATE_MIN_TRACKS = 64;
/** A pair the tracks cover skips the descriptor matching with at least this many matches (`kltMatching`). */
export const KLT_COVER_MATCHES = 64;

/** The positions of the alive tracks at a keyframe, by track id. */
export function trackAnchor(tracks: {
  count: number;
  id: ArrayLike<number>;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
}): Map<number, [number, number]> {
  const out = new Map<number, [number, number]>();
  for (let k = 0; k < tracks.count; k++) out.set(tracks.id[k]!, [tracks.x[k]!, tracks.y[k]!]);
  return out;
}

/**
 * The median displacement in pixels of the tracks that are alive now and were alive at the anchor
 * (section 3.6), and how many there are. The median is Infinity when fewer than `minTracks` are
 * common, so a caller reads "not enough evidence" as "no decision".
 */
export function medianTrackDisplacement(
  anchor: ReadonlyMap<number, readonly [number, number]>,
  tracks: { count: number; id: ArrayLike<number>; x: ArrayLike<number>; y: ArrayLike<number> },
  minTracks = KLT_GATE_MIN_TRACKS,
): { median: number; tracks: number } {
  const distances: number[] = [];
  for (let k = 0; k < tracks.count; k++) {
    const from = anchor.get(tracks.id[k]!);
    if (from) distances.push(Math.hypot(tracks.x[k]! - from[0], tracks.y[k]! - from[1]));
  }
  if (distances.length < minTracks) return { median: Infinity, tracks: distances.length };
  distances.sort((a, b) => a - b);
  return { median: distances[distances.length >> 1]!, tracks: distances.length };
}
