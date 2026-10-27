import { validateCamera } from '../../geometry/camera.ts';
import type { Camera } from '../../contracts/camera.ts';
import { validatePose } from '../../geometry/pose.ts';
import { buildTracks, validateFeatures } from './tracks.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  PairMatches,
  RegisteredView,
} from '../../contracts/scene.ts';
import { observationKey } from '../../contracts/keys.ts';

export const POSED_EVIDENCE_VERSION = 'typecv-posed-pair-evidence-v1' as const;
const representation = 'f64-le-hex-v1' as const;
interface Controls {
  readonly maxErrorPx: number;
  readonly minAngleDeg: number;
  readonly maxPairHypotheses: number;
  readonly refinementIterations: number;
}
interface Limits {
  readonly maxViews: number;
  readonly maxPairs: number;
  readonly maxMatchRows: number;
}
export interface PosedEvidenceInput {
  readonly images: readonly ImageFeatures[];
  readonly views: readonly RegisteredView[];
  readonly pairs: readonly PairMatches[];
  readonly controls: Controls;
  /** Explicit admission bounds, not a policy for truncating source evidence. */
  readonly limits: Limits;
}
interface FrozenPose {
  readonly R: readonly number[];
  readonly t: readonly number[];
}
interface SnapshotView {
  readonly id: number;
  readonly featureCount: number;
  readonly camera: Camera;
  readonly registered?: { readonly camera: Camera; readonly pose: FrozenPose };
}
interface SourceObservation {
  readonly viewId: number;
  readonly featureId: number;
  readonly pixel: readonly [number, number];
}
interface SourceRow {
  readonly pairIndex: number;
  readonly matchRow: number;
  readonly first: SourceObservation;
  readonly second: SourceObservation;
  readonly distance?: number;
}
interface PreparedObservation extends SourceObservation {
  readonly camera: Camera;
  readonly pose: FrozenPose;
}
export interface PreparedPosedEvidence {
  readonly version: typeof POSED_EVIDENCE_VERSION;
  readonly representation: typeof representation;
  /** Full numerical identity; no hash-only equality or mutable input aliases. */
  readonly identity: string;
  readonly controls: Controls;
  readonly limits: Limits;
  readonly source: {
    readonly views: readonly SnapshotView[];
    readonly pairs: readonly {
      readonly first: number;
      readonly second: number;
      readonly matchCount: number;
      readonly hasDistances: boolean;
    }[];
    /** Includes every original row, even when either view has no accepted pose. */
    readonly rows: readonly SourceRow[];
  };
  /** Original pair/match order, with exactly two fixed observations in each row. */
  readonly rows: readonly {
    readonly sourceRow: number;
    readonly observations: readonly [PreparedObservation, PreparedObservation];
  }[];
}
export interface PosedPairProof {
  readonly version: typeof POSED_EVIDENCE_VERSION;
  readonly inputIdentity: string;
  readonly observationsPerRow: 2;
  /** One actual numerical status per prepared row, in its unchanged order. */
  readonly statuses: ArrayLike<number>;
}
export interface EvidenceExclusions {
  readonly currentOwned: ReadonlySet<string>;
  readonly archiveOwned: ReadonlySet<string>;
  readonly rejected: ReadonlySet<string>;
}
export interface PosedEvidenceGraph {
  readonly authority: 'candidate-only';
  readonly evidence: PreparedPosedEvidence;
  readonly rows: readonly {
    readonly sourceRow: number;
    readonly status: number | null;
    readonly candidate: boolean;
    readonly excluded: {
      readonly currentOwned: boolean;
      readonly archiveOwned: boolean;
      readonly rejected: boolean;
    };
  }[];
  readonly graph: {
    readonly tracks: readonly FeatureTrack[];
    readonly conflictingEdges: number;
    readonly redundantEdges: number;
  };
}

function freeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}
function identity(value: unknown): string {
  const bytes = new DataView(new ArrayBuffer(8));
  return JSON.stringify(value, (_, number: unknown) => {
    if (typeof number !== 'number') return number;
    bytes.setFloat64(0, number, true);
    return Array.from({ length: 8 }, (_, i) =>
      bytes.getUint8(i).toString(16).padStart(2, '0'),
    ).join('');
  });
}
function copyCamera(camera: Camera): Camera {
  return {
    fx: camera.fx,
    fy: camera.fy,
    cx: camera.cx,
    cy: camera.cy,
    k1: camera.k1 ?? 0,
    k2: camera.k2 ?? 0,
  };
}

/** Pure preparation: validate before copying and retain complete bounded raw evidence. */
export function preparePosedEvidence(input: PosedEvidenceInput): PreparedPosedEvidence {
  const { limits, controls } = input;
  if (
    ![limits.maxViews, limits.maxPairs, limits.maxMatchRows].every(
      (n) => Number.isSafeInteger(n) && n > 0,
    ) ||
    input.images.length > limits.maxViews ||
    input.views.length > limits.maxViews ||
    input.pairs.length > limits.maxPairs
  )
    throw new RangeError('Posed evidence exceeds valid explicit view/pair bounds');
  let count = 0;
  for (const pair of input.pairs) {
    if (
      !Number.isSafeInteger(pair.matches.length) ||
      pair.matches.length < 0 ||
      pair.matches.length % 2
    )
      throw new RangeError('Invalid posed evidence match count');
    count += pair.matches.length / 2;
    if (!Number.isSafeInteger(count) || count > limits.maxMatchRows)
      throw new RangeError('Posed evidence exceeds match-row bound');
  }
  if (
    !Number.isFinite(controls.maxErrorPx) ||
    controls.maxErrorPx <= 0 ||
    !Number.isFinite(controls.minAngleDeg) ||
    controls.minAngleDeg < 0 ||
    controls.minAngleDeg >= 180 ||
    !Number.isSafeInteger(controls.maxPairHypotheses) ||
    controls.maxPairHypotheses < 1 ||
    !Number.isSafeInteger(controls.refinementIterations) ||
    controls.refinementIterations < 0
  )
    throw new RangeError('Invalid posed evidence numerical controls');
  const byId = validateFeatures(input.images, input.pairs);
  for (const image of input.images)
    for (const coordinate of image.keypoints)
      if (!Number.isFinite(coordinate)) throw new RangeError('Invalid posed evidence pixel');
  for (const pair of input.pairs)
    for (const featureId of pair.matches)
      if (!Number.isSafeInteger(featureId) || featureId < 0)
        throw new RangeError('Invalid posed evidence feature identity');
  const posed = new Map<number, RegisteredView>();
  for (const view of input.views) {
    if (!byId.has(view.id) || posed.has(view.id))
      throw new RangeError('Invalid posed view identity');
    validateCamera(view.camera);
    validatePose(view.pose);
    posed.set(view.id, view);
  }
  const views: SnapshotView[] = input.images.map((image) => {
    const view = posed.get(image.id);
    return {
      id: image.id,
      featureCount: image.keypoints.length / 2,
      camera: copyCamera(image.camera),
      ...(view
        ? {
            registered: {
              camera: copyCamera(view.camera),
              pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
            },
          }
        : {}),
    };
  });
  const snapshots = new Map(views.map((view) => [view.id, view]));
  const sourceRows: SourceRow[] = [],
    rows: PreparedPosedEvidence['rows'][number][] = [];
  const observation = (viewId: number, featureId: number): SourceObservation => {
    const pixels = byId.get(viewId)!.keypoints;
    return { viewId, featureId, pixel: [pixels[featureId * 2]!, pixels[featureId * 2 + 1]!] };
  };
  input.pairs.forEach((pair, pairIndex) => {
    const first = snapshots.get(pair.first)!.registered,
      second = snapshots.get(pair.second)!.registered;
    for (let matchRow = 0; matchRow < pair.matches.length / 2; matchRow++) {
      const sourceRow = sourceRows.length,
        a = observation(pair.first, pair.matches[matchRow * 2]!),
        b = observation(pair.second, pair.matches[matchRow * 2 + 1]!);
      sourceRows.push({
        pairIndex,
        matchRow,
        first: a,
        second: b,
        ...(pair.distances ? { distance: pair.distances[matchRow]! } : {}),
      });
      if (first && second)
        rows.push({
          sourceRow,
          observations: [
            { ...a, ...first },
            { ...b, ...second },
          ],
        });
    }
  });
  const payload = {
    version: POSED_EVIDENCE_VERSION,
    representation,
    controls: {
      maxErrorPx: controls.maxErrorPx,
      minAngleDeg: controls.minAngleDeg,
      maxPairHypotheses: controls.maxPairHypotheses,
      refinementIterations: controls.refinementIterations,
    },
    limits: {
      maxViews: limits.maxViews,
      maxPairs: limits.maxPairs,
      maxMatchRows: limits.maxMatchRows,
    },
    source: {
      views,
      pairs: input.pairs.map((pair) => ({
        first: pair.first,
        second: pair.second,
        matchCount: pair.matches.length / 2,
        hasDistances: Boolean(pair.distances),
      })),
      rows: sourceRows,
    },
  };
  return freeze({ ...payload, identity: identity(payload), rows });
}

/**
 * Build only candidate associations. Status0/4 proves compatibility for exactly two
 * inputs; it never admits depth or observations. Final robust multiview masks remain
 * the authority. Ownership changes, conflicts and missing/status1/status3 proofs do
 * not delete any original raw row. A geometry change requires fresh preparation.
 */
export function buildPosedEvidenceGraph(
  prepared: PreparedPosedEvidence,
  current: PosedEvidenceInput,
  proof: PosedPairProof | undefined,
  exclusions: EvidenceExclusions,
): PosedEvidenceGraph {
  const evidence = preparePosedEvidence(current);
  if (
    prepared.version !== evidence.version ||
    prepared.representation !== evidence.representation ||
    prepared.identity !== evidence.identity
  )
    throw new RangeError('Stale posed evidence input identity');
  let statuses: (number | null)[] = evidence.rows.map(() => null);
  if (proof) {
    if (
      proof.version !== evidence.version ||
      proof.inputIdentity !== evidence.identity ||
      proof.observationsPerRow !== 2 ||
      proof.statuses.length !== evidence.rows.length
    )
      throw new RangeError('Invalid posed pair proof binding/shape');
    statuses = Array.from({ length: evidence.rows.length }, (_, index) => proof.statuses[index]!);
    if (!statuses.every((status) => status === 0 || status === 1 || status === 3 || status === 4))
      throw new RangeError('Invalid posed pair proof status');
  }
  const pairRows = evidence.source.pairs.map(() => ({
    matches: [] as number[],
    distances: [] as number[],
  }));
  const decisions = evidence.rows.map((row, index) => {
    const source = evidence.source.rows[row.sourceRow]!,
      keys = row.observations.map((o) => observationKey(o.viewId, o.featureId));
    const excluded = {
      currentOwned: keys.some((key) => exclusions.currentOwned.has(key)),
      archiveOwned: keys.some((key) => exclusions.archiveOwned.has(key)),
      rejected: keys.some((key) => exclusions.rejected.has(key)),
    };
    const status = statuses[index] ?? null,
      candidate = !Object.values(excluded).some(Boolean) && (status === 0 || status === 4);
    if (candidate) {
      pairRows[source.pairIndex]!.matches.push(source.first.featureId, source.second.featureId);
      pairRows[source.pairIndex]!.distances.push(source.distance ?? 0);
    }
    return { sourceRow: row.sourceRow, status, candidate, excluded };
  });
  const pairs = evidence.source.pairs.map((pair, index) => ({
    first: pair.first,
    second: pair.second,
    matches: Uint32Array.from(pairRows[index]!.matches),
    ...(pair.hasDistances ? { distances: Float64Array.from(pairRows[index]!.distances) } : {}),
  }));
  return freeze({
    authority: 'candidate-only' as const,
    evidence,
    rows: decisions,
    graph: buildTracks(current.images, pairs),
  });
}
