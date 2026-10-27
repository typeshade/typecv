import { buildTracks } from '../tracks/tracks.ts';
import { selectCameraObservations } from '../tracks/observation-selection.ts';
import {
  trackingEvidence,
  type PairRotationEvidence,
  type ReconstructionTracking,
} from '../tracks/tracking.ts';
import { ReconstructionInitializationError } from '../protocol/errors.ts';
import { pairKey } from '../../contracts/keys.ts';
import type { FeatureTrack, ImageFeatures, PairMatches } from '../../contracts/scene.ts';
import type { CameraObservationSelection } from '../tracks/observation-selection.ts';
import type { IncrementalControls, IncrementalOptions, PairVerification } from './mapper.ts';

/** @internal The verdicts of a run's pair verification (`verifyPairs`) that the run's reports read. */
export interface RunVerification {
  readonly verified: readonly PairMatches[];
  readonly diagnostics: readonly PairVerification[];
  readonly trackingPairs: readonly PairMatches[];
  readonly rotations: readonly PairRotationEvidence[];
}

/**
 * @internal The input of a run after the camera masks, which apply to the pairs before consensus
 * (`selectCameraObservations`). With `preVerified`, the pre-verified rows of the pairs whose
 * matches the masks changed go, so those pairs are verified again (`incrementalSteps`).
 * `sceneSteps` reads no pre-verified rows and keeps its options.
 */
export function maskedInput(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions,
  { preVerified }: { readonly preVerified: boolean },
): {
  readonly pairs: readonly PairMatches[];
  readonly options: IncrementalOptions;
  readonly selection?: CameraObservationSelection;
} {
  const selected = selectCameraObservations(images, inputPairs, options.cameraObservationMasks);
  return {
    pairs: selected.pairs,
    options:
      preVerified && selected.changedPairKeys.size
        ? {
            ...options,
            preVerified: options.preVerified?.filter(
              (p) => !selected.changedPairKeys.has(pairKey(p.verified.first, p.verified.second)),
            ),
          }
        : options,
    selection: selected.selection,
  };
}

/**
 * @internal The seed of a new map: the candidate of `initialPair` in either order, or else the
 * first candidate, which is the best (`verifyPairs` sorts them). Undefined when no candidate fits.
 */
export function pickSeed<Candidate extends { readonly pair: PairMatches }>(
  candidates: readonly Candidate[],
  initialPair: readonly [number, number] | undefined,
): Candidate | undefined {
  return initialPair
    ? candidates.find(
        (c) =>
          (c.pair.first === initialPair[0] && c.pair.second === initialPair[1]) ||
          (c.pair.first === initialPair[1] && c.pair.second === initialPair[0]),
      )
    : candidates[0];
}

/**
 * @internal The calibrated 2D and rotation evidence of a run (`trackingEvidence`): over the
 * rotation-tracking rows with rotation tracking, and over the verified pairs otherwise. The
 * depth-deferred tracks count only with a depth gate (`maxRelativeDepthStdDev`).
 */
export function runTracking(
  controls: Pick<IncrementalControls, 'rotationTracking' | 'depth'>,
  verification: RunVerification,
  tracks: readonly FeatureTrack[],
  triangulated?: Iterable<number>,
  depthDeferred?: Iterable<number>,
): ReconstructionTracking {
  return trackingEvidence(
    controls.rotationTracking ? verification.trackingPairs : verification.verified,
    verification.rotations,
    tracks,
    triangulated,
    controls.depth.maxRelativeDepthStdDev ? depthDeferred : undefined,
  );
}

/**
 * @internal The error of a run that found no seed pair: the pair verdicts and, when the run keeps
 * the tracking evidence, the evidence over the tracks of its pairs.
 */
export function noSeedError(
  images: readonly ImageFeatures[],
  controls: Pick<IncrementalControls, 'rotationTracking' | 'depth'>,
  verification: RunVerification,
  keepTracking: boolean,
  selection: CameraObservationSelection | undefined,
): ReconstructionInitializationError {
  return new ReconstructionInitializationError(
    verification.diagnostics,
    undefined,
    undefined,
    keepTracking
      ? runTracking(
          controls,
          verification,
          buildTracks(
            images,
            controls.rotationTracking ? verification.trackingPairs : verification.verified,
          ).tracks,
        )
      : undefined,
    selection,
  );
}

/**
 * @internal The error of a run whose seed pair triangulated fewer landmarks than `minInitialPoints`:
 * the pair verdicts and, when the run keeps the tracking evidence, the evidence with no
 * triangulated track.
 */
export function fewLandmarksError(
  controls: Pick<IncrementalControls, 'rotationTracking' | 'depth'>,
  verification: RunVerification,
  tracks: readonly FeatureTrack[],
  keepTracking: boolean,
  depthDeferred: Iterable<number>,
  selection: CameraObservationSelection | undefined,
): ReconstructionInitializationError {
  return new ReconstructionInitializationError(
    verification.diagnostics,
    'Initial tracks did not produce enough reliable landmarks',
    undefined,
    keepTracking ? runTracking(controls, verification, tracks, [], depthDeferred) : undefined,
    selection,
  );
}
