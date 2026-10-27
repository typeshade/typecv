import type { BundleAdjustmentResult } from '../bundle/bundle-adjustment.ts';
import type { BundleAdjustmentPartition } from '../bundle/local-bundle.ts';
import type { CameraObservationSelection } from '../tracks/observation-selection.ts';
import type { ReconstructionTracking } from '../tracks/tracking.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  PairMatches,
  SparseReconstruction,
} from '../../contracts/scene.ts';
import type { RegistrationFailure } from './registration.ts';
import type { IncrementalResult, PairVerification } from './mapper.ts';

/** @internal What a finished run reports (`mapperResult`). */
export interface MapperOutcome {
  readonly selection?: CameraObservationSelection;
  readonly reconstruction: SparseReconstruction;
  readonly graph: { readonly tracks: readonly FeatureTrack[]; readonly conflictingEdges: number };
  readonly images: readonly ImageFeatures[];
  readonly registered: (viewId: number) => boolean;
  readonly failure: ReadonlyMap<number, RegistrationFailure>;
  readonly initialPair: readonly [number, number];
  readonly pairVerification: readonly PairVerification[];
  readonly registrationOrder: readonly number[];
  readonly bundleAdjustments: readonly BundleAdjustmentResult['diagnostics'][];
  readonly bundlePartitions: readonly BundleAdjustmentPartition[];
  readonly meanReprojectionErrorPx: number;
  /** The tracking evidence, when the run keeps it. */
  readonly tracking?: ReconstructionTracking;
  /**
   * The verified pairs. `incrementalSteps` reports them. `sceneSteps` does not; phase 8a of
   * docs/library-structure.md, section 5, brings them.
   */
  readonly verifiedPairs?: readonly PairMatches[];
}

/**
 * @internal The result of a finished run. A view that did not register gives the reason of its
 * last failure, or no connection to the reconstructed tracks.
 */
export function mapperResult(run: MapperOutcome): IncrementalResult {
  return {
    ...(run.selection ? { observationSelection: run.selection } : {}),
    reconstruction: run.reconstruction,
    tracks: run.graph.tracks,
    unregistered: run.images
      .filter((v) => !run.registered(v.id))
      .map((v) => ({
        viewId: v.id,
        reason: run.failure.get(v.id)?.reason ?? 'Disconnected from reconstructed tracks',
      })),
    diagnostics: {
      initialPair: run.initialPair,
      pairVerification: run.pairVerification,
      conflictingTrackEdges: run.graph.conflictingEdges,
      registrationOrder: run.registrationOrder,
      bundleAdjustments: run.bundleAdjustments,
      ...(run.bundlePartitions.length ? { bundlePartitions: run.bundlePartitions } : {}),
      meanReprojectionErrorPx: run.meanReprojectionErrorPx,
    },
    ...(run.verifiedPairs ? { verifiedPairs: run.verifiedPairs } : {}),
    ...(run.tracking ? { tracking: run.tracking } : {}),
  };
}
