import type { InverseDepthSnapshot } from '../depth/inverse-depth-tracker.ts';
import type { FocalSearchFailure } from '../focal/focal-search.ts';
import type { PairVerification } from '../mapper/mapper.ts';
import type { CameraObservationSelection } from '../tracks/observation-selection.ts';
import type { ReconstructionTracking } from '../tracks/tracking.ts';

export class ReconstructionInitializationError extends Error {
  declare readonly inverseDepth?: InverseDepthSnapshot;
  readonly pairVerification: readonly PairVerification[];
  readonly focalSearch?: FocalSearchFailure;
  declare readonly tracking?: ReconstructionTracking;
  declare readonly observationSelection?: CameraObservationSelection;
  constructor(
    pairVerification: readonly PairVerification[],
    message = 'No stable initial pair: check calibration, homography dominance, parallax and verified matches',
    focalSearch?: FocalSearchFailure,
    tracking?: ReconstructionTracking,
    observationSelection?: CameraObservationSelection,
  ) {
    super(message);
    this.name = 'ReconstructionInitializationError';
    this.pairVerification = pairVerification;
    this.focalSearch = focalSearch;
    if (tracking) this.tracking = tracking;
    if (observationSelection) this.observationSelection = observationSelection;
  }
}
/** @internal Driver failures must propagate through per-view numerical rejection handlers. */
export class ReconstructionOperationError extends Error {
  constructor(cause: unknown) {
    super(
      `Reconstruction operation failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
    this.name = 'ReconstructionOperationError';
  }
}
