import type { PairVerification } from '../../../src/reconstruction/mapper.ts';
import type { FocalSearchFailure } from '../../../src/reconstruction/focal-search.ts';

/** Optional bounded geometry evidence on a failed v6 run, without images or descriptors. */
export interface InitializationFailureRecord {
  readonly pairs: readonly PairVerification[];
  readonly omittedPairs: number;
  readonly focalSearch?: FocalSearchFailure;
}

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const text = (value: unknown, limit: number): value is string =>
  typeof value === 'string' && value.length <= limit;

function pairValid(value: unknown): value is PairVerification {
  return (
    object(value) &&
    count(value.first) &&
    count(value.second) &&
    value.first !== value.second &&
    count(value.matches) &&
    count(value.inliers) &&
    value.inliers <= value.matches &&
    count(value.homographyInliers) &&
    value.homographyInliers <= value.matches &&
    (value.positiveDepthCount === undefined ||
      (count(value.positiveDepthCount) && value.positiveDepthCount <= value.matches)) &&
    finite(value.medianAngleDeg) &&
    value.medianAngleDeg >= 0 &&
    value.medianAngleDeg <= 180 &&
    typeof value.seedEligible === 'boolean' &&
    (value.reason === undefined || text(value.reason, 160))
  );
}

function focalFailureValid(value: unknown): value is FocalSearchFailure {
  return (
    object(value) &&
    count(value.omittedAttempts) &&
    Array.isArray(value.attempts) &&
    value.attempts.length <= 64 &&
    value.attempts.every(
      (a) =>
        object(a) &&
        finite(a.ratio) &&
        a.ratio > 0 &&
        count(a.views) &&
        a.views >= 2 &&
        count(a.pairs) &&
        count(a.eligiblePairs) &&
        a.eligiblePairs <= a.pairs &&
        (a.bestPair === undefined || pairValid(a.bestPair)) &&
        Array.isArray(a.reasons) &&
        a.reasons.length <= 16 &&
        a.reasons.every((r) => object(r) && text(r.reason, 160) && count(r.pairs)) &&
        a.reasons.reduce((sum, r) => sum + r.pairs, 0) === a.pairs &&
        text(a.failure, 256),
    )
  );
}

export function initializationFailureValid(value: unknown): value is InitializationFailureRecord {
  return (
    object(value) &&
    Array.isArray(value.pairs) &&
    value.pairs.length <= 32 &&
    value.pairs.every(pairValid) &&
    count(value.omittedPairs) &&
    (value.focalSearch === undefined || focalFailureValid(value.focalSearch))
  );
}

/** Keep useful geometric diagnostics when the worker converts an Error into a run record. */
export function initializationFailureRecord(
  error: unknown,
): InitializationFailureRecord | undefined {
  if (
    !object(error) ||
    error.name !== 'ReconstructionInitializationError' ||
    !Array.isArray(error.pairVerification)
  )
    return undefined;
  const verified = error.pairVerification.filter(pairValid);
  const pairs = verified
    .sort((a, b) => Number(b.seedEligible) - Number(a.seedEligible) || b.inliers - a.inliers)
    .slice(0, 32)
    .map(copyPair);
  const search = focalFailureValid(error.focalSearch) ? error.focalSearch : undefined;
  return {
    pairs,
    omittedPairs: verified.length - pairs.length,
    ...(search
      ? {
          focalSearch: {
            omittedAttempts: search.omittedAttempts,
            attempts: search.attempts.map((a) => ({
              ratio: a.ratio,
              views: a.views,
              pairs: a.pairs,
              eligiblePairs: a.eligiblePairs,
              ...(a.bestPair ? { bestPair: copyPair(a.bestPair) } : {}),
              reasons: a.reasons.map((r) => ({ reason: r.reason, pairs: r.pairs })),
              failure: a.failure,
            })),
          },
        }
      : {}),
  };
}

function copyPair(pair: PairVerification): PairVerification {
  return {
    first: pair.first,
    second: pair.second,
    matches: pair.matches,
    inliers: pair.inliers,
    homographyInliers: pair.homographyInliers,
    medianAngleDeg: pair.medianAngleDeg,
    seedEligible: pair.seedEligible,
    ...(pair.positiveDepthCount === undefined
      ? {}
      : { positiveDepthCount: pair.positiveDepthCount }),
    ...(pair.reason === undefined ? {} : { reason: pair.reason }),
  };
}
