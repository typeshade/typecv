import {
  relativeRotationControls,
  type RelativeRotationOptions,
  type RelativeRotationResult,
} from '../geometry/rotation.ts';
import type { FeatureTrack, PairMatches } from './types.ts';
import type { Camera } from '../geometry/camera.ts';

export interface PairRotationEvidence {
  readonly first: number;
  readonly second: number;
  /** Supplied calibrations used for this hypothesis, before any later focal BA. */
  readonly firstCamera: Camera;
  readonly secondCamera: Camera;
  readonly rotation: RelativeRotationResult;
}

/** Optional 2D evidence, separate from map registration and triangulated landmarks. */
export interface ReconstructionTracking {
  /** Epipolar inliers plus rotation-compatible observations. Not all are structure seed edges. */
  readonly pairs: readonly PairMatches[];
  /** Relative hypotheses for seed-ineligible pairs only; no world pose or camera center. */
  readonly rotations: readonly PairRotationEvidence[];
  readonly tracks: readonly FeatureTrack[];
  /** Local track ids without an accepted 3D landmark; not a claim that they have valid depth. */
  readonly untriangulatedTrackIds: readonly number[];
}

/** @internal Snapshot and validate the opt-in before asynchronous work starts. */
export function rotationTrackingControls(
  value: boolean | RelativeRotationOptions | undefined,
): Required<RelativeRotationOptions> | undefined {
  if (value === undefined || value === false) return undefined;
  if (value === true) return relativeRotationControls();
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new RangeError('rotationTracking must be a boolean or relative-rotation options');
  return relativeRotationControls(value);
}

/** @internal Preserve feature ownership and match distances when retaining raw inlier rows. */
export function selectPairRows(pair: PairMatches, rows: Uint32Array): PairMatches {
  const matches = new Uint32Array(rows.length * 2),
    distances = new Float64Array(rows.length);
  rows.forEach((row, i) => {
    matches[2 * i] = pair.matches[2 * row]!;
    matches[2 * i + 1] = pair.matches[2 * row + 1]!;
    distances[i] = pair.distances?.[row] ?? 0;
  });
  return { first: pair.first, second: pair.second, matches, distances };
}

/** @internal Union the two verified roles without duplicating a correspondence. */
export function mergeTrackingPair(
  raw: PairMatches,
  structure: PairMatches | undefined,
  rotation: RelativeRotationResult | undefined,
): PairMatches | undefined {
  if (!rotation) return structure;
  const tracking = selectPairRows(raw, rotation.inlierRows);
  if (!structure) return tracking;
  const matches: number[] = [],
    distances: number[] = [],
    seen = new Set<string>();
  for (const pair of [structure, tracking])
    for (let i = 0; i < pair.matches.length; i += 2) {
      const reversed = pair.first !== raw.first,
        a = pair.matches[i + (reversed ? 1 : 0)]!,
        b = pair.matches[i + (reversed ? 0 : 1)]!,
        key = `${a}:${b}`;
      if (seen.has(key)) continue;
      seen.add(key);
      matches.push(a, b);
      distances.push(pair.distances?.[i / 2] ?? 0);
    }
  return {
    first: raw.first,
    second: raw.second,
    matches: Uint32Array.from(matches),
    distances: Float64Array.from(distances),
  };
}

/** @internal Local graph ids can change as observations arrive; identify a track through its observations. */
export function trackingEvidence(
  pairs: readonly PairMatches[],
  rotations: readonly PairRotationEvidence[],
  tracks: readonly FeatureTrack[],
  triangulated: Iterable<number> = [],
): ReconstructionTracking {
  const points = new Set(triangulated);
  return {
    pairs,
    rotations,
    tracks,
    untriangulatedTrackIds: tracks.filter((t) => !points.has(t.id)).map((t) => t.id),
  };
}

/** @internal A caller may mutate a returned snapshot without changing the next session update. */
export function copyTracking(evidence: ReconstructionTracking): ReconstructionTracking {
  return {
    pairs: evidence.pairs.map((p) => ({
      ...p,
      matches: p.matches.slice(),
      distances: p.distances?.slice(),
    })),
    rotations: evidence.rotations.map((p) => ({
      ...p,
      firstCamera: { ...p.firstCamera },
      secondCamera: { ...p.secondCamera },
      rotation: {
        ...p.rotation,
        R: p.rotation.R.slice(),
        inlierRows: p.rotation.inlierRows.slice(),
        spreadDeg: [...p.rotation.spreadDeg] as [number, number],
      },
    })),
    tracks: evidence.tracks.map((t) => ({
      id: t.id,
      observations: t.observations.map((o) => ({ ...o })),
    })),
    untriangulatedTrackIds: [...evidence.untriangulatedTrackIds],
  };
}
