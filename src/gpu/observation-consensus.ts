/** @internal Original image coordinates used only to choose an independent patch witness. */
export interface ConsensusFrame {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  readonly keypoints: ArrayLike<number>;
  readonly validMask?: Uint8Array;
  readonly shotId?: number | string;
}

/** @internal A primary image measurement, before ownership or feature arrays are changed. */
export interface ObservationConsensusCandidate {
  readonly id: number;
  readonly group: number | undefined;
  readonly sourceViewId: number;
  readonly sourceFeatureId: number;
  readonly targetViewId: number;
  readonly measuredPixel: readonly [number, number];
}

/** @internal The callback must independently validate tracking, patches and ambiguity. */
export interface ConsensusMeasurement {
  readonly pixel: readonly [number, number];
  readonly ncc: number;
  readonly forwardBackwardErrorPx: number;
}

export interface ObservationConsensusResult {
  readonly id: number;
  readonly status: 'accepted' | 'disagreement' | 'measurementFailure' | 'unavailable';
  readonly anchorViewId?: number;
  readonly anchorFeatureId?: number;
  readonly measurement?: ConsensusMeasurement;
  readonly disagreementPx?: number;
}

export interface ObservationConsensusOptions {
  readonly maxDisagreementPx?: number;
  /** Conservative source/target footprint. The callback checks its actual warped footprint. */
  readonly halfWindow?: number;
  /** Bound callback arrays even if a caller supplies an unusually large candidate set. */
  readonly batchSize?: number;
}

function inside(frame: ConsensusFrame, x: number, y: number, hw: number): boolean {
  if (!(x >= hw && y >= hw && x + hw < frame.width - 1 && y + hw < frame.height - 1)) return false;
  if (frame.validMask)
    for (let j = Math.floor(y - hw); j <= Math.ceil(y + hw); j++)
      for (let i = Math.floor(x - hw); i <= Math.ceil(x + hw); i++)
        if (!frame.validMask[j * frame.width + i]) return false;
  return true;
}

/**
 * Find at most one independent ORIGINAL patch witness for each primary measurement.
 * Trusted membership and selected coordinates are snapshotted before the first await;
 * newly propagated descendants cannot corroborate their own ancestor in this call.
 * Initial target coordinates are measured pixels, never projections or accepted evidence.
 * Two-view tracks remain explicitly unavailable rather than failing a three-view test.
 */
export async function corroborateObservations<Frame extends ConsensusFrame>(
  frames: readonly Frame[],
  trustedGroups: readonly ReadonlyMap<number, number>[],
  candidates: readonly ObservationConsensusCandidate[],
  measure: (
    anchor: Frame,
    target: Frame,
    seeds: Float32Array,
    initialTargets: Float32Array,
    featureIds: readonly number[],
  ) => Promise<readonly (ConsensusMeasurement | null)[]>,
  options: ObservationConsensusOptions = {},
): Promise<ObservationConsensusResult[]> {
  const maxError = options.maxDisagreementPx ?? 0.5,
    hw = options.halfWindow ?? 0,
    batchSize = options.batchSize ?? 1024;
  if (
    !Number.isFinite(maxError) ||
    maxError <= 0 ||
    !Number.isSafeInteger(hw) ||
    hw < 0 ||
    hw > 31 ||
    !Number.isSafeInteger(batchSize) ||
    batchSize < 1 ||
    batchSize > 8192
  )
    throw new RangeError('Invalid observation consensus bounds');
  const indices = new Map(frames.map((frame, i) => [frame.id, i]));
  if (indices.size !== frames.length) throw new RangeError('Duplicate consensus frame id');
  if (new Set(candidates.map((candidate) => candidate.id)).size !== candidates.length)
    throw new RangeError('Duplicate consensus candidate id');
  const trusted = trustedGroups.map((group) => new Map(group));
  type Planned = {
    index: number;
    anchorFeatureId: number;
    seed: readonly [number, number];
    primary: readonly [number, number];
  };
  const groups = new Map<string, { anchor: Frame; target: Frame; plans: Planned[] }>(),
    results: ObservationConsensusResult[] = candidates.map((candidate) => ({
      id: candidate.id,
      status: 'unavailable',
    }));
  const sameInterval = (a: number, b: number, source: Frame) => {
    for (let i = Math.min(a, b); i <= Math.max(a, b); i++) {
      const frame = frames[i]!;
      if (
        frame.shotId !== source.shotId ||
        frame.width !== source.width ||
        frame.height !== source.height
      )
        return false;
    }
    return true;
  };
  candidates.forEach((candidate, index) => {
    const sourceIndex = indices.get(candidate.sourceViewId),
      targetIndex = indices.get(candidate.targetViewId),
      members = candidate.group === undefined ? undefined : trusted[candidate.group];
    if (sourceIndex === undefined || targetIndex === undefined || !members) return;
    const source = frames[sourceIndex]!,
      target = frames[targetIndex]!,
      sourceFeature = candidate.sourceFeatureId;
    if (
      sourceIndex === targetIndex ||
      !Number.isSafeInteger(sourceFeature) ||
      sourceFeature < 0 ||
      2 * sourceFeature + 1 >= source.keypoints.length ||
      !inside(
        source,
        source.keypoints[2 * sourceFeature]!,
        source.keypoints[2 * sourceFeature + 1]!,
        hw,
      ) ||
      !inside(target, candidate.measuredPixel[0], candidate.measuredPixel[1], hw) ||
      !sameInterval(sourceIndex, targetIndex, source)
    )
      return;
    let bestIndex: number | undefined,
      bestFeature = -1,
      bestBaseline = -1,
      bestTargetBaseline = -1;
    for (const [viewId, featureId] of members) {
      const anchorIndex = indices.get(viewId);
      if (
        anchorIndex === undefined ||
        anchorIndex === sourceIndex ||
        anchorIndex === targetIndex ||
        !Number.isSafeInteger(featureId) ||
        featureId < 0
      )
        continue;
      const anchor = frames[anchorIndex]!;
      if (
        2 * featureId + 1 >= anchor.keypoints.length ||
        !inside(
          anchor,
          anchor.keypoints[2 * featureId]!,
          anchor.keypoints[2 * featureId + 1]!,
          hw,
        ) ||
        !sameInterval(anchorIndex, targetIndex, source)
      )
        continue;
      const baseline = Math.abs(anchorIndex - sourceIndex),
        targetBaseline = Math.abs(anchorIndex - targetIndex);
      if (
        baseline > bestBaseline ||
        (baseline === bestBaseline && targetBaseline > bestTargetBaseline) ||
        (baseline === bestBaseline &&
          targetBaseline === bestTargetBaseline &&
          anchorIndex < bestIndex!)
      ) {
        bestIndex = anchorIndex;
        bestFeature = featureId;
        bestBaseline = baseline;
        bestTargetBaseline = targetBaseline;
      }
    }
    if (bestIndex === undefined) return;
    const anchor = frames[bestIndex]!,
      key = `${anchor.id}:${target.id}`;
    let batch = groups.get(key);
    if (!batch) {
      batch = { anchor, target, plans: [] };
      groups.set(key, batch);
    }
    batch.plans.push({
      index,
      anchorFeatureId: bestFeature,
      seed: [anchor.keypoints[2 * bestFeature]!, anchor.keypoints[2 * bestFeature + 1]!],
      primary: [candidate.measuredPixel[0], candidate.measuredPixel[1]],
    });
  });
  for (const batch of groups.values())
    for (let start = 0; start < batch.plans.length; start += batchSize) {
      const plans = batch.plans.slice(start, start + batchSize),
        seeds = Float32Array.from(plans.flatMap((plan) => [...plan.seed])),
        initials = Float32Array.from(plans.flatMap((plan) => [...plan.primary])),
        measurements = await measure(
          batch.anchor,
          batch.target,
          seeds,
          initials,
          plans.map((plan) => plan.anchorFeatureId),
        );
      if (measurements.length !== plans.length)
        throw new RangeError('Consensus callback returned a different measurement count');
      plans.forEach((plan, i) => {
        const measurement = measurements[i],
          base = {
            id: results[plan.index]!.id,
            anchorViewId: batch.anchor.id,
            anchorFeatureId: plan.anchorFeatureId,
          };
        if (
          !measurement ||
          !Number.isFinite(measurement.ncc) ||
          !Number.isFinite(measurement.forwardBackwardErrorPx) ||
          measurement.forwardBackwardErrorPx < 0 ||
          !inside(batch.target, measurement.pixel[0], measurement.pixel[1], hw)
        ) {
          results[plan.index] = { ...base, status: 'measurementFailure' };
          return;
        }
        const disagreementPx = Math.hypot(
          measurement.pixel[0] - plan.primary[0],
          measurement.pixel[1] - plan.primary[1],
        );
        results[plan.index] = {
          ...base,
          status: disagreementPx <= maxError ? 'accepted' : 'disagreement',
          measurement: {
            pixel: [measurement.pixel[0], measurement.pixel[1]],
            ncc: measurement.ncc,
            forwardBackwardErrorPx: measurement.forwardBackwardErrorPx,
          },
          disagreementPx,
        };
      });
    }
  return results;
}
