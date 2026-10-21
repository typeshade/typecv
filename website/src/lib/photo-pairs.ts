/**
 * Which earlier views the lab matches a new frame against, and which features it keeps resident.
 *
 * A video arrives in time order, so a window of the first frame and a few recent frames covers
 * the pairs that overlap. Photographs have no order, so the same window drops most pairs.
 */
import type { Profile } from '../workers/sfm.worker.ts';

export interface PartnerCandidate {
  readonly id: number;
}

/** The ids whose resident features survive after a frame is registered. */
export function retainedFeatureIds(
  profile: Profile,
  references: readonly number[],
  ids: readonly number[],
): Set<number> {
  return new Set([...references, ...ids.slice(-3)]);
}

/** The retained views that a new frame is matched against. */
export function partnerIds(profile: Profile, candidates: readonly PartnerCandidate[]): number[] {
  return candidates.map((c) => c.id);
}
