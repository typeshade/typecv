import { identityPose, projectPoint } from '../../geometry/pose.ts';
import type { InverseDepthTracking, InverseDepthSnapshot } from '../depth/inverse-depth-tracker.ts';
import { drive } from '../protocol/drive.ts';
import { withDepthStep } from '../protocol/depth-step.ts';
import {
  verifyImagePair,
  type ImagePairVerification,
  type ImagePairGeometry,
} from '../../geometry/image-pair.ts';
import { estimatePosePnP } from '../../geometry/pnp.ts';
import { estimateRelativeRotation, type RelativeRotationOptions } from '../../geometry/rotation.ts';
import {
  mergeTrackingPair,
  type PairRotationEvidence,
  type ReconstructionTracking,
} from '../tracks/tracking.ts';
import {
  assessTrackDepth,
  depthSensitivityControls,
  triangulateTrackDetailed,
} from '../../geometry/multiview.ts';
import type {
  TriangulationObservation,
  DetailedTrackTriangulation,
  DepthSensitivityOptions,
} from '../../geometry/multiview.ts';
import { bundleAdjust } from '../bundle/bundle-adjustment.ts';
import { validateReconstruction } from '../../geometry/validation.ts';
import { pendingBundleDirections } from '../bundle/directional-tracks.ts';
import type { BundleAdjustmentResult } from '../bundle/bundle-adjustment.ts';
import {
  extractLocalBundle,
  mergeLocalBundle,
  planConnectedBundles,
  mergeBundleRefinements,
} from '../bundle/local-bundle.ts';
import type { BundleAdjustmentPartition } from '../bundle/local-bundle.ts';
import { pairSeed } from '../model/identity.ts';
import { buildTracks, validateFeatures } from '../tracks/tracks.ts';
import type { CameraObservationSelection } from '../tracks/observation-selection.ts';
import type { CameraObservationMask } from '../../contracts/observation-mask.ts';
import {
  applyFocal,
  focalSearchControls,
  focalSearchSteps,
  type FocalSearchOptions,
  type FocalSearchResult,
} from '../focal/focal-search.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  Landmark,
  LandmarkObservation,
  PairMatches,
  RegisteredView,
  SparseReconstruction,
} from '../../contracts/scene.ts';
import { judgePair } from '../pairs/pair-verdict.ts';
import { incrementalControls } from '../config/incremental-controls.ts';
import {
  ReconstructionInitializationError,
  ReconstructionOperationError,
} from '../protocol/errors.ts';
import { copyReconstruction } from '../model/copy.ts';
import { imagePairHomographyIterations } from '../../geometry/operator-defaults.ts';
import { MAPPER_DEFAULTS, bundleHuberPx } from '../config/defaults.ts';
import { pairKey } from '../../contracts/keys.ts';
import { TriangulationStatus } from '../../contracts/triangulation-status.ts';
import {
  fewLandmarksError,
  maskedInput,
  noSeedError,
  pickSeed,
  runTracking,
} from './initialization.ts';
import { indexTracks, resumeMap } from './track-index.ts';
import {
  pnpFailure,
  rankViews,
  retryAfterRecovery,
  retryWaits,
  tooFewCorrespondences,
  type RegistrationFailure,
} from './registration.ts';
import { mapperResult } from './result.ts';

/** @internal */
export { judgePair } from '../pairs/pair-verdict.ts';
/** @internal */
export { incrementalControls } from '../config/incremental-controls.ts';
export {
  DEFAULT_BUNDLE_STALL_TOLERANCE,
  DEFAULT_GLOBAL_BUNDLE_RATIO,
} from '../config/incremental-controls.ts';
export { ReconstructionInitializationError } from '../protocol/errors.ts';
/** @internal */
export { ReconstructionOperationError } from '../protocol/errors.ts';
/** @internal */
export { copyReconstruction } from '../model/copy.ts';

export interface IncrementalOptions extends DepthSensitivityOptions {
  /** Opt-in persistent raw-observation depth state, applied at completed map/session updates.
   * Callbacks publish only the admitted completed map; intermediate solver seeds remain internal.
   */
  readonly depthTracking?: InverseDepthTracking;
  /** Experimental finite/directional separation with bearing-only BA and focal scoring. Default false. */
  readonly separateDirectionalObservations?: boolean;
  /** Explicit per-feature selection, applied before consensus; separate rows and provenance are retained. */
  readonly cameraObservationMasks?: readonly CameraObservationMask[];
  /** Existing CPU/TypeShade two-view pose refinement; 0 (legacy) by default, maximum 100. */
  readonly pairRefinementIterations?: number;
  /**
   * Calibrated rotation hypotheses for seed-ineligible pairs, enabled by default when
   * focalSearch is off. Set false for the original epipolar-only track graph. Preserve 2D tracks
   * without a camera center or new depth. Uses the existing f64 CPU math on every driver.
   * Requires supplied calibration; cannot be combined with focalSearch in this first stage.
   */
  readonly rotationTracking?: boolean | RelativeRotationOptions;
  /** Independent snapshots after initialization, each registration and final refinement. */
  readonly onReconstruction?: (model: SparseReconstruction) => void;
  readonly seed?: number;
  readonly ransacIterations?: number;
  readonly maxErrorPx?: number;
  readonly minInitialAngleDeg?: number;
  readonly minTriangulationAngleDeg?: number;
  readonly minPnPInliers?: number;
  readonly minInitialPoints?: number;
  readonly bundleIterations?: number;
  readonly finalBundleIterations?: number;
  /** Calibrate one shared focal scale during the final global bundle adjustment. */
  readonly refineSharedFocal?: boolean;
  /** Variable cameras of the local BA that follows each registration. */
  readonly localBundleSize?: number;
  /**
   * Restrict local BA to the landmarks the camera window observes, with all their retained
   * observations and observing cameras; `false` keeps every landmark variable in a local pass.
   */
  readonly localBundleSubset?: boolean;
  /**
   * Run a global BA after a registration when the registered views reach this multiple of their
   * count at the last global BA (COLMAP's `ba_global_images_ratio`). `Infinity` turns it off.
   */
  readonly globalBundleViewRatio?: number;
  /** The same rule for the landmark count (COLMAP's `ba_global_points_ratio`). */
  readonly globalBundlePointRatio?: number;
  /** Also run a global BA whenever the registered-view count is a multiple of this. Off by default. */
  readonly globalBundleFrequency?: number;
  /**
   * End the LM loop of each BA when an accepted step lowers the robust cost by less than this
   * fraction of the cost.
   */
  readonly bundleStallTolerance?: number;
  /** Explicit seed still undergoes all geometric/degeneracy checks. */
  readonly initialPair?: readonly [number, number];
  /**
   * Pairs that a view graph already verified (docs/multi-video.md 3.4). Each entry replaces the
   * `verifyPair` run of the input pair with the same two ids. A pair without an entry is
   * verified as before.
   */
  readonly preVerified?: readonly PreVerifiedPair[];
  /**
   * The camera group of each registered view id, for the final BA with `refineSharedFocal`: one
   * focal scale for each group (docs/multi-video.md 2.3, item 4). Without it, all views share one.
   */
  readonly focalGroups?: ReadonlyMap<number, number>;
  /**
   * Estimate one shared focal before mapping (docs/auto-focal.md): the supplied `fx` and `fy`
   * are replaced by `ratio × width` (width `2·cx + 1`), with `fy / fx` kept. The principal point
   * and the distortion stay. Ignored when a run resumes a map.
   */
  readonly focalSearch?: boolean | FocalSearchOptions;
}
export interface PairVerification {
  readonly first: number;
  readonly second: number;
  readonly matches: number;
  readonly inliers: number;
  readonly homographyInliers: number;
  readonly positiveDepthCount?: number;
  /** E inliers compatible with rotation within the pixel budget; not proof of infinite world depth. */
  readonly directionCompatibleCount?: number;
  readonly medianAngleDeg: number;
  readonly seedEligible: boolean;
  readonly reason?: string;
  /** Present only when the opt-in rotation fit was attempted. Does not imply zero translation. */
  readonly rotationInliers?: number;
  readonly rotationErrorPx?: number;
}
/**
 * The verdict of one pair that was verified before the mapper ran, as `verifyPairs` would have
 * produced it: the inlier matches, the diagnostic row and, for a pair that can seed a map, the
 * geometry.
 */
export interface PreVerifiedPair {
  /** The inlier matches, as `verifyPairs` keeps them. */
  readonly verified: PairMatches;
  readonly verification: PairVerification;
  /** Required when `verification.seedEligible` is true. */
  readonly geometry?: ImagePairGeometry;
  /**
   * The pair failed verification: `verifyPairs` keeps its diagnostic row and no inlier matches,
   * and `verified` only names the pair.
   */
  readonly rejected?: boolean;
}
export interface IncrementalResult {
  readonly inverseDepth?: InverseDepthSnapshot;
  readonly observationSelection?: CameraObservationSelection;
  /** Calibrated 2D/rotation evidence when enabled; never a source of full poses or depth. */
  readonly tracking?: ReconstructionTracking;
  readonly reconstruction: SparseReconstruction;
  readonly tracks: readonly FeatureTrack[];
  readonly unregistered: readonly { viewId: number; reason: string }[];
  readonly diagnostics: {
    readonly initialPair: readonly [number, number];
    readonly pairVerification: readonly PairVerification[];
    readonly conflictingTrackEdges: number;
    readonly registrationOrder: readonly number[];
    readonly bundleAdjustments: readonly BundleAdjustmentResult['diagnostics'][];
    /** Optimization partitions of an existing map, with held poses and deferred constraints. */
    readonly bundlePartitions?: readonly BundleAdjustmentPartition[];
    readonly meanReprojectionErrorPx: number;
    /** The focal search, when `focalSearch` ran. */
    readonly focalSearch?: FocalSearchResult;
  };
  /**
   * The pairs that passed verification, each with the matches that the epipolar model kept (the
   * inlier rows), in id order. A pair that failed verification is absent. A caller that feeds
   * tracks into the pairs reads the verdict on its matches here (`docs/klt-tracking.md` 3.5).
   */
  readonly verifiedPairs?: readonly PairMatches[];
}

/** Numerical requests shared by the synchronous reference and asynchronous device driver. */
export type ReconstructionJob =
  | { kind: 'verifyPair'; args: Parameters<typeof verifyImagePair> }
  | { kind: 'pnp'; args: Parameters<typeof estimatePosePnP> }
  | {
      kind: 'triangulation';
      observations: readonly (readonly TriangulationObservation[])[];
      options: Parameters<typeof triangulateTrackDetailed>[1];
      /** Track of each group, for diagnostics; drivers ignore it. */
      trackIds: readonly number[];
    }
  | { kind: 'bundle'; args: Parameters<typeof bundleAdjust> }
  | {
      kind: 'observations';
      model: SparseReconstruction;
      maxError: number;
      depth?: DepthSensitivityOptions;
      /**
       * `continuation` checks a new view's observation appended last to each landmark; `filter`
       * checks every observation after a bundle pass; `measure` measures session progress without
       * changing its observation set; `final` filters and measures the finished model.
       */
      stage: 'continuation' | 'filter' | 'measure' | 'final';
    };
function* request<T, J extends ReconstructionJob = ReconstructionJob>(
  job: J,
): Generator<J, T, unknown> {
  return (yield job) as T;
}
/** @internal The one request pair verification makes; both mappers share it. */
export type PairVerificationJob = Extract<ReconstructionJob, { kind: 'verifyPair' }>;
/** @internal */
export interface ObservationEvaluation {
  readonly depthRejected?: Uint8Array;
  readonly masks: readonly Uint8Array[];
  readonly meanErrorPx: number;
}
function evaluateCpuObservations(
  model: SparseReconstruction,
  maxError: number,
  options: DepthSensitivityOptions = {},
): ObservationEvaluation {
  const depth = depthSensitivityControls(options),
    rejected = new Uint8Array(model.landmarks.length);
  const views = new Map(model.views.map((v) => [v.id, v]));
  let total = 0,
    count = 0;
  const masks = model.landmarks.map((p, i) => {
    const errors: number[] = [];
    const mask = Uint8Array.from(p.observations, (o) => {
      const v = views.get(o.viewId)!,
        q = projectPoint(v.camera, v.pose, p.xyz);
      if (!q) {
        errors.push(Infinity);
        return 0;
      }
      const e = Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]);
      errors.push(e);
      if (e > maxError) return 0;
      return 1;
    });
    if (depth.maxRelativeDepthStdDev) {
      const observations = p.observations
        .filter((_, j) => mask[j])
        .map((o) => ({ ...views.get(o.viewId)!, pixel: o.pixel }));
      const sensitivity = assessTrackDepth(p.xyz, observations, depth.pixelNoiseStdDevPx);
      if (
        observations.length >= 2 &&
        (!sensitivity || sensitivity.maxRelativeDepthStdDev > depth.maxRelativeDepthStdDev)
      ) {
        rejected[i] = 1;
        mask.fill(0);
      }
    }
    p.observations.forEach((_, j) => {
      if (mask[j]) {
        total += errors[j]!;
        count++;
      }
    });
    return mask;
  });
  return {
    masks,
    meanErrorPx: count ? total / count : Infinity,
    ...(depth.maxRelativeDepthStdDev ? { depthRejected: rejected } : {}),
  };
}
/** @internal Synchronous CPU execution of one numerical request. */
export function executeReconstructionJob(job: ReconstructionJob): unknown {
  switch (job.kind) {
    case 'verifyPair':
      return verifyImagePair(...job.args);
    case 'pnp':
      return estimatePosePnP(...job.args);
    case 'triangulation':
      return job.observations.map((o) => triangulateTrackDetailed(o, job.options));
    case 'bundle':
      return bundleAdjust(...job.args);
    case 'observations':
      return evaluateCpuObservations(job.model, job.maxError, job.depth);
  }
}
/** Calibrated, fixed-intrinsics incremental sparse SfM. Raw edges are verified before tracks. */
export function reconstructIncremental(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
): IncrementalResult {
  return executeCpuMapper(images, options, (controls) =>
    incrementalSteps(images, inputPairs, controls),
  );
}

/**
 * Refine an accepted connected map after adding measured coordinates/edges. Feature indices
 * and original pixel measurements must remain stable. New edges undergo normal verification;
 * the original seed gauge and existing map initialize refinement instead of choosing a new seed.
 */
export function refineIncremental(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  initial: IncrementalResult,
  options: IncrementalOptions = {},
): IncrementalResult {
  const model = refinementModel(images, inputPairs, initial, options);
  return executeCpuMapper(
    images,
    { ...options, initialPair: initial.diagnostics.initialPair },
    (controls) => incrementalSteps(images, inputPairs, controls, model, undefined, true, 2, true),
  );
}

/** @internal Validate ownership and snapshot the accepted map before any asynchronous work. */
export function refinementModel(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  initial: IncrementalResult,
  options: IncrementalOptions,
): SparseReconstruction {
  const byId = validateFeatures(images, pairs),
    model = initial.reconstruction,
    anchor = initial.diagnostics.initialPair;
  validateReconstruction(model);
  if (
    anchor.length !== 2 ||
    anchor[0] === anchor[1] ||
    !anchor.every((id) => model.views.some((v) => v.id === id)) ||
    model.views.some((v) => !byId.has(v.id)) ||
    (options.initialPair && options.initialPair.some((id, i) => id !== anchor[i]))
  )
    throw new RangeError('Refinement must retain all registered views and the original seed gauge');
  for (const point of model.landmarks)
    for (const o of point.observations) {
      const image = byId.get(o.viewId)!;
      if (
        o.featureId >= image.keypoints.length / 2 ||
        Math.abs(image.keypoints[2 * o.featureId]! - o.pixel[0]) > 1e-9 ||
        Math.abs(image.keypoints[2 * o.featureId + 1]! - o.pixel[1]) > 1e-9
      )
        throw new RangeError(
          'Refinement requires stable feature indices and original pixel measurements',
        );
    }
  return copyReconstruction(model);
}

function executeCpuMapper(
  images: readonly ImageFeatures[],
  options: IncrementalOptions,
  create: (
    controls: IncrementalOptions,
  ) => Generator<ReconstructionJob, IncrementalResult, unknown>,
): IncrementalResult {
  return withDepthStep(images, options, (controls) =>
    drive(create(controls), executeReconstructionJob, 'pass'),
  );
}

export { MAX_REGISTRATION_ATTEMPTS } from './registration.ts';

/** @internal Common control flow; drivers select numerical execution explicitly. */
/**
 * Calibrated, fixed-intrinsics incremental sparse SfM. Raw pair edges are verified before
 * track construction. Landmarks persist across registrations: a new view continues the
 * tracks it observes, triangulates only the tracks it completes, and bundle adjustment
 * refines the retained points. Tracks that failed are retried after the map has grown.
 */
/** @internal The validated numerical controls of the incremental mapper. */
export interface IncrementalControls {
  readonly separateDirectionalObservations: boolean;
  readonly pairRefinementIterations: number;
  readonly depth: DepthSensitivityOptions;
  readonly rotationTracking?: Required<RelativeRotationOptions>;
  readonly seed: number;
  readonly iterations: number;
  readonly maxError: number;
  readonly initialAngle: number;
  readonly pointAngle: number;
  readonly minPnP: number;
  readonly minInitial: number;
  readonly bundleIterations: number;
  readonly finalBundle: number;
  readonly localSize: number;
  readonly localSubset: boolean;
  readonly globalViewRatio: number;
  readonly globalPointRatio: number;
  /** 0 when off. */
  readonly globalFrequency: number;
  readonly stallTolerance: number;
}

/** The model size at the last global BA, which the schedule compares the model to. */
export interface BundleSchedule {
  views: number;
  points: number;
}
/**
 * @internal Whether the BA after a registration is global: the model has grown by the view or
 * point ratio since the last global BA, or the view count meets the optional frequency. A local
 * BA runs otherwise. The initial, recovery and final passes are always global.
 */
export function globalBundleDue(
  views: number,
  points: number,
  last: Readonly<BundleSchedule>,
  controls: Pick<IncrementalControls, 'globalViewRatio' | 'globalPointRatio' | 'globalFrequency'>,
): boolean {
  return (
    views >= last.views * controls.globalViewRatio ||
    points >= last.points * controls.globalPointRatio ||
    (controls.globalFrequency > 0 && views % controls.globalFrequency === 0)
  );
}

/**
 * @internal Index the pre-verified pairs by pair key. Throws RangeError when an entry names no
 * input pair, repeats a pair, disagrees with its own diagnostic or lacks the geometry of a seed.
 */
export function preVerifiedByKey(
  inputPairs: readonly PairMatches[],
  entries: readonly PreVerifiedPair[] | undefined,
): Map<string, PreVerifiedPair> | undefined {
  if (!entries?.length) return undefined;
  const inputs = new Set(inputPairs.map((p) => pairKey(p.first, p.second)));
  const out = new Map<string, PreVerifiedPair>();
  for (const e of entries) {
    const key = pairKey(e.verified.first, e.verified.second);
    if (
      !inputs.has(key) ||
      out.has(key) ||
      pairKey(e.verification.first, e.verification.second) !== key ||
      (e.verification.seedEligible && !e.geometry)
    )
      throw new RangeError('Invalid pre-verified pair');
    out.set(key, e);
  }
  return out;
}
/** @internal A verified pair that may seed the map, with its score. */
export interface SeedCandidate {
  readonly pair: PairMatches;
  readonly result: ImagePairGeometry;
  readonly score: number;
}
/**
 * @internal Verify every pair in id order with a seed from the two ids of the pair, keep the inlier edges and
 * score the pairs that may seed the map, best first.
 */
export function* verifyPairs(
  byId: Map<number, ImageFeatures>,
  inputPairs: readonly PairMatches[],
  controls: IncrementalControls,
  preVerified?: ReadonlyMap<string, PreVerifiedPair>,
): Generator<
  PairVerificationJob,
  {
    verified: PairMatches[];
    diagnostics: PairVerification[];
    candidates: SeedCandidate[];
    trackingPairs: PairMatches[];
    rotations: PairRotationEvidence[];
  },
  unknown
> {
  const { seed, iterations, maxError, pointAngle, rotationTracking } = controls;
  const ordered = [...inputPairs].sort(
    (a, b) =>
      Math.min(a.first, a.second) - Math.min(b.first, b.second) ||
      Math.max(a.first, a.second) - Math.max(b.first, b.second),
  );
  const verified: PairMatches[] = [],
    diagnostics: PairVerification[] = [],
    candidates: SeedCandidate[] = [],
    trackingPairs: PairMatches[] = [],
    rotations: PairRotationEvidence[] = [];
  for (const pair of ordered) {
    const cached = preVerified?.get(pairKey(pair.first, pair.second)),
      sameDirectionalMode = cached?.geometry
        ? (cached.geometry.diagnostics.directionCompatibleCount !== undefined) ===
          controls.separateDirectionalObservations
        : !controls.separateDirectionalObservations,
      known = controls.pairRefinementIterations || !sameDirectionalMode ? undefined : cached,
      a = byId.get(pair.first)!,
      b = byId.get(pair.second)!;
    let raw: Float64Array | undefined;
    const pixels = () => {
      if (raw) return raw;
      raw = new Float64Array(pair.matches.length * 2);
      for (let k = 0; k < pair.matches.length; k += 2) {
        const ai = pair.matches[k]!,
          bi = pair.matches[k + 1]!;
        raw.set(
          [
            a.keypoints[ai * 2]!,
            a.keypoints[ai * 2 + 1]!,
            b.keypoints[bi * 2]!,
            b.keypoints[bi * 2 + 1]!,
          ],
          k * 2,
        );
      }
      return raw;
    };
    let judged: ReturnType<typeof judgePair>;
    if (known) {
      // Reuse the E/H verdict. The explicit rotation opt-in may still fit the raw 2D bearings.
      judged = {
        accepted: known.rejected ? undefined : known.verified,
        diagnostic: known.verification,
        candidate: known.verification.seedEligible
          ? {
              pair: known.verified,
              result: known.geometry!,
              score:
                known.geometry!.positiveDepthCount *
                Math.min(10, known.geometry!.diagnostics.medianParallaxDeg),
            }
          : undefined,
      };
    } else {
      const verification = yield* request<ImagePairVerification, PairVerificationJob>({
        kind: 'verifyPair',
        args: [
          pixels(),
          a.camera,
          b.camera,
          {
            iterations,
            separateDirectionalObservations: controls.separateDirectionalObservations,
            seed: pairSeed(seed, pair.first, pair.second),
            threshold: maxError / Math.min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy),
            refinementIterations: controls.pairRefinementIterations,
            minParallaxDeg: Math.min(pointAngle, 0.1),
            maxReprojectionErrorPx: maxError,
            homographyThresholdPx: maxError,
            homographyIterations: imagePairHomographyIterations(iterations),
          },
        ],
      });
      judged = judgePair(pair, verification, controls);
    }
    if (rotationTracking) {
      // A valid E seed already supplies structure evidence. A rotation-compatible pair can
      // retain 2D observations but can never promote a rejected pair to a structure seed.
      const rotation = !judged.candidate
        ? estimateRelativeRotation(pixels(), a.camera, b.camera, {
            ...rotationTracking,
            seed: pairSeed(rotationTracking.seed, pair.first, pair.second),
          })
        : undefined;
      if (!judged.candidate)
        judged = {
          ...judged,
          diagnostic: {
            ...judged.diagnostic,
            rotationInliers: rotation?.inlierRows.length ?? 0,
            ...(rotation ? { rotationErrorPx: rotation.meanReprojectionErrorPx } : {}),
          },
        };
      if (rotation)
        rotations.push({
          first: pair.first,
          second: pair.second,
          firstCamera: { ...a.camera },
          secondCamera: { ...b.camera },
          rotation,
        });
      const tracking = mergeTrackingPair(pair, judged.accepted, rotation);
      if (tracking) trackingPairs.push(tracking);
    }
    diagnostics.push(judged.diagnostic);
    if (judged.accepted) verified.push(judged.accepted);
    if (judged.candidate) candidates.push(judged.candidate);
  }
  candidates.sort(
    (a, b) => b.score - a.score || a.pair.first - b.pair.first || a.pair.second - b.pair.second,
  );
  return { verified, diagnostics, candidates, trackingPairs, rotations };
}

export function* incrementalSteps(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
  resume?: SparseReconstruction,
  /** The size at the last global BA, carried across resumed runs; updated in place. */
  schedule?: BundleSchedule,
  /** @internal Session progress preserves the local BA's observation set until a global pass or finish(). */
  finalize = true,
  /** @internal Minimum registered views for a streaming focal estimate, relaxed by finish(). */
  minimumFocalViews = 2,
  /** @internal Explicit batch refinement may complete observations in already registered views. */
  completeRegisteredObservations = false,
): Generator<ReconstructionJob, IncrementalResult, unknown> {
  const selected = maskedInput(images, inputPairs, options, { preVerified: true });
  inputPairs = selected.pairs;
  options = selected.options;
  const search = focalSearchControls(options.focalSearch);
  if (search && !resume) {
    incrementalControls(images, inputPairs, options);
    let found: FocalSearchResult;
    try {
      found = yield* focalSearchSteps(
        images,
        inputPairs,
        options,
        search,
        false,
        minimumFocalViews,
        incrementalSteps,
      );
    } catch (error) {
      if (error instanceof ReconstructionInitializationError && selected.selection)
        throw new ReconstructionInitializationError(
          error.pairVerification,
          error.message,
          error.focalSearch,
          error.tracking,
          selected.selection,
        );
      throw error;
    }
    const result = yield* incrementalSteps(
      applyFocal(images, found.ratio),
      inputPairs,
      // Search inputs did not promise calibrated bearings. Keep the calibrated default
      // disabled after the search too, including its final full-map reconstruction.
      { ...options, focalSearch: undefined, rotationTracking: false },
      undefined,
      schedule,
      finalize,
    );
    return {
      ...result,
      ...(selected.selection ? { observationSelection: selected.selection } : {}),
      diagnostics: { ...result.diagnostics, focalSearch: found },
    };
  }
  const { byId, controls } = incrementalControls(images, inputPairs, options);
  const keepTracking = !!(
      controls.rotationTracking ||
      controls.depth.maxRelativeDepthStdDev ||
      controls.separateDirectionalObservations
    ),
    depthDeferred = new Set<number>();
  const {
    iterations,
    maxError,
    seed,
    pointAngle,
    minPnP,
    minInitial,
    bundleIterations,
    finalBundle,
    localSize,
    localSubset,
    stallTolerance,
  } = controls;
  const verification = yield* verifyPairs(
    byId,
    inputPairs,
    controls,
    preVerifiedByKey(inputPairs, options.preVerified),
  );
  const { verified, diagnostics, candidates, trackingPairs } = verification;
  const initial = pickSeed(candidates, options.initialPair);
  if (!initial && !resume)
    throw noSeedError(images, controls, verification, keepTracking, selected.selection);
  const anchorPair = resume ? options.initialPair! : [initial!.pair.first, initial!.pair.second];
  const graph = buildTracks(images, controls.rotationTracking ? trackingPairs : verified),
    first = byId.get(anchorPair[0]!)!,
    second = byId.get(anchorPair[1]!)!;
  const { viewTracks, trackOfObservation, pixelOf } = indexTracks(images, graph.tracks, byId);

  // Mutable map state. Landmarks persist across registrations and are keyed by track.
  const resumed = resume ? resumeMap(resume, trackOfObservation) : undefined;
  const views: RegisteredView[] = resumed
    ? resumed.views
    : [
        { id: first.id, camera: { ...first.camera }, pose: identityPose() },
        {
          id: second.id,
          camera: { ...second.camera },
          pose: { R: initial!.result.R.slice(), t: initial!.result.t.slice() },
        },
      ];
  const registered = new Map(views.map((v) => [v.id, v])),
    landmarks = resumed?.landmarks ?? new Map<number, Landmark>(),
    // Track id -> registered view count when triangulation last failed or the landmark was dropped.
    triedAt = new Map<number, number>();
  const model = (): SparseReconstruction => ({ views, landmarks: [...landmarks.values()] });
  const publish = () => options.onReconstruction?.(copyReconstruction(model()));
  const order = views.map((view) => view.id),
    bundleReports: BundleAdjustmentResult['diagnostics'][] = [],
    bundlePartitions: BundleAdjustmentPartition[] = [];
  const registeredObservations = (trackId: number): LandmarkObservation[] =>
    graph.tracks[trackId]!.observations.flatMap((o) =>
      registered.has(o.viewId)
        ? [{ viewId: o.viewId, featureId: o.featureId, pixel: pixelOf(o.viewId, o.featureId) }]
        : [],
    );
  /** Triangulate tracks without a landmark that have two registered observations and no attempt at this map size. */
  const triangulate = function* (
    trackIds: Iterable<number>,
  ): Generator<ReconstructionJob, void, unknown> {
    const groups: { trackId: number; observations: LandmarkObservation[] }[] = [];
    for (const trackId of [...new Set(trackIds)].sort((a, b) => a - b)) {
      if (landmarks.has(trackId) || (triedAt.get(trackId) ?? -1) >= views.length) continue;
      const observations = registeredObservations(trackId);
      if (observations.length >= 2) groups.push({ trackId, observations });
    }
    if (!groups.length) return;
    const results = yield* request<readonly DetailedTrackTriangulation[]>({
      kind: 'triangulation',
      observations: groups.map((g) =>
        g.observations.map((o) => {
          const v = registered.get(o.viewId)!;
          return { camera: v.camera, pose: v.pose, pixel: o.pixel };
        }),
      ),
      options: { maxErrorPx: maxError, minAngleDeg: pointAngle, ...controls.depth },
      trackIds: groups.map((g) => g.trackId),
    });
    groups.forEach((g, i) => {
      const detailed = results[i]!,
        result = detailed.track;
      if (detailed.status === TriangulationStatus.depthSensitivity) depthDeferred.add(g.trackId);
      if (result) {
        depthDeferred.delete(g.trackId);
        landmarks.set(g.trackId, {
          trackId: g.trackId,
          xyz: result.xyz,
          observations: g.observations.filter((_, j) => result.inliers[j]),
        });
      } else triedAt.set(g.trackId, views.length);
    });
  };
  /** Append a newly registered view's observation to each landmark of a track it observes, when it reprojects. */
  const continueTracks = function* (viewId: number): Generator<ReconstructionJob, void, unknown> {
    const candidates: Landmark[] = [];
    for (const { trackId, featureId } of viewTracks.get(viewId)!) {
      const point = landmarks.get(trackId);
      if (!point || point.observations.some((o) => o.viewId === viewId)) continue;
      candidates.push({
        ...point,
        observations: [
          ...point.observations,
          { viewId, featureId, pixel: pixelOf(viewId, featureId) },
        ],
      });
    }
    if (!candidates.length) return;
    const evaluation = yield* request<ObservationEvaluation>({
      kind: 'observations',
      model: { views, landmarks: candidates },
      maxError,
      depth: controls.depth,
      stage: 'continuation',
    });
    candidates.forEach((candidate, i) => {
      const mask = evaluation.masks[i]!;
      if (mask[mask.length - 1]) landmarks.set(candidate.trackId, candidate);
    });
  };
  /** Drop rejected observations and points below two observations; report inliers lost with a point. */
  const applyEvaluation = (current: SparseReconstruction, evaluation: ObservationEvaluation) => {
    let removedInlier = false;
    current.landmarks.forEach((p, i) => {
      if (evaluation.depthRejected?.[i]) depthDeferred.add(p.trackId);
      const observations = p.observations.filter((_, j) => evaluation.masks[i]![j]);
      if (observations.length >= 2) {
        if (observations.length !== p.observations.length)
          landmarks.set(p.trackId, { ...p, observations });
      } else {
        landmarks.delete(p.trackId);
        triedAt.set(p.trackId, views.length);
        removedInlier ||= observations.length > 0;
      }
    });
    return removedInlier;
  };
  /** A dropped landmark may be re-triangulated after the map grows. */
  const filter = function* (): Generator<ReconstructionJob, void, unknown> {
    if (!landmarks.size) return;
    const current = model();
    const evaluation = yield* request<ObservationEvaluation>({
      kind: 'observations',
      model: current,
      maxError,
      depth: controls.depth,
      stage: 'filter',
    });
    applyEvaluation(current, evaluation);
  };
  // Keep local BA's observation ownership; only defer entire depth-unstable points.
  const deferSensitive = (current: SparseReconstruction, evaluation: ObservationEvaluation) => {
    current.landmarks.forEach((p, i) => {
      if (!evaluation.depthRejected?.[i] && evaluation.masks[i]!.reduce((n, x) => n + x, 0) >= 2)
        return;
      landmarks.delete(p.trackId);
      triedAt.set(p.trackId, views.length);
      if (evaluation.depthRejected?.[i]) depthDeferred.add(p.trackId);
    });
  };
  const adjust = function* (
    steps: number,
    local = false,
    refineSharedFocal = false,
    includeDirections = false,
  ): Generator<ReconstructionJob, void, unknown> {
    if (!steps || !landmarks.size) return;
    const current = model(),
      window = local
        ? views
            .filter((v) => v.id !== first.id && v.id !== second.id)
            .slice(-localSize)
            .map((v) => v.id)
        : undefined;
    const subset = local && localSubset ? extractLocalBundle(current, window!) : undefined;
    const problem = subset?.reconstruction ?? current;
    const plan = planConnectedBundles(problem, {
      iterations: steps,
      anchorViewId: subset?.anchorViewId ?? first.id,
      scaleViewId: subset?.scaleViewId ?? second.id,
      huberPx: bundleHuberPx(maxError),
      variableViewIds: window,
      refineSharedFocal,
      focalGroups: refineSharedFocal ? options.focalGroups : undefined,
      stallTolerance,
    });
    if (plan.partition) bundlePartitions.push(plan.partition);
    const refinements: SparseReconstruction[] = [];
    for (const part of plan.problems) {
      // First give finite-only mapping and final retriangulation a chance to establish depth.
      // A low-parallax finite track must not become a direction prior that absorbs later parallax.
      const directions =
        includeDirections && controls.separateDirectionalObservations
          ? pendingBundleDirections(part.reconstruction, graph.tracks, byId, maxError)
          : [];
      const result = yield* request<BundleAdjustmentResult>({
        kind: 'bundle',
        args: [
          part.reconstruction,
          directions.length
            ? { ...part.options, directionalConstraints: directions }
            : part.options,
        ],
      });
      refinements.push(result.reconstruction);
      bundleReports.push(result.diagnostics);
    }
    if (!refinements.length) return;
    const solved = plan.partition ? mergeBundleRefinements(problem, refinements) : refinements[0]!;
    const refined = subset ? mergeLocalBundle(current, subset, solved) : solved;
    for (const view of refined.views) {
      const index = views.findIndex((v) => v.id === view.id);
      views[index] = view;
      registered.set(view.id, view);
    }
    for (const point of refined.landmarks) landmarks.set(point.trackId, point);
    // Local extraction changes no observation ownership; other passes filter explicitly outside BA.
    if (subset) {
      if (controls.depth.maxRelativeDepthStdDev && landmarks.size) {
        const current = model();
        const evaluation = yield* request<ObservationEvaluation>({
          kind: 'observations',
          model: current,
          maxError,
          depth: controls.depth,
          stage: 'measure',
        });
        deferSensitive(current, evaluation);
      }
      return;
    }
    yield* filter();
    if (!local) {
      last.views = views.length;
      last.points = landmarks.size;
    }
  };
  // Without a carried schedule, the map as it starts counts as globally adjusted.
  const last: BundleSchedule = schedule ?? { views: views.length, points: landmarks.size };
  // A session window can evict views and points; the ratios then hold against the smaller map.
  last.views = Math.min(last.views, views.length);
  last.points = Math.min(last.points, landmarks.size);
  // A batch refinement can add an observation to an already registered image. It needs
  // the same image/geometry checks as continuation after registering a new image.
  if (resume && completeRegisteredObservations)
    for (const view of views) yield* continueTracks(view.id);
  yield* triangulate(viewTracks.get(first.id)!.map((entry) => entry.trackId));
  if (landmarks.size < minInitial)
    throw fewLandmarksError(
      controls,
      verification,
      graph.tracks,
      keepTracking,
      depthDeferred,
      selected.selection,
    );
  if (!resume) yield* adjust(bundleIterations);
  publish();
  const failure = new Map<number, RegistrationFailure>();
  // Map size at which a recovery pass (global retriangulation and refinement) last ran.
  let recoveredAt = -1;
  /** Unregistered views with the landmarks their tracks reach, the best supported first. */
  const rank = () =>
    rankViews(
      images,
      (id) => registered.has(id),
      viewTracks,
      (trackId) => landmarks.get(trackId),
    );
  const estimatePose = function* (
    view: ImageFeatures,
    correspondences: readonly { featureId: number; point: Landmark }[],
    minInliers: number,
  ): Generator<ReconstructionJob, ReturnType<typeof estimatePosePnP>, unknown> {
    const X = Float64Array.from(correspondences.flatMap((c) => [...c.point.xyz])),
      pixels = Float64Array.from(correspondences.flatMap((c) => pixelOf(view.id, c.featureId)));
    return yield* request<ReturnType<typeof estimatePosePnP>>({
      kind: 'pnp',
      args: [
        X,
        pixels,
        view.camera,
        { iterations, seed: seed + order.length, thresholdPx: maxError, minInliers },
      ],
    });
  };
  /** Add a view at a pose and continue and triangulate the tracks it observes. */
  const addView = function* (
    view: ImageFeatures,
    pose: ReturnType<typeof estimatePosePnP>['pose'],
  ): Generator<ReconstructionJob, void, unknown> {
    const registeredView: RegisteredView = { id: view.id, camera: { ...view.camera }, pose };
    views.push(registeredView);
    registered.set(view.id, registeredView);
    order.push(view.id);
    failure.delete(view.id);
    yield* continueTracks(view.id);
    yield* triangulate(viewTracks.get(view.id)!.map((entry) => entry.trackId));
  };
  /** Bundle adjustment and publication after a registration. */
  const settle = function* (): Generator<ReconstructionJob, void, unknown> {
    const global = globalBundleDue(views.length, landmarks.size, last, controls);
    if (global) yield* triangulate(graph.tracks.map((track) => track.id));
    yield* adjust(bundleIterations, !global);
    publish();
  };
  /**
   * A map that stalls may hold views that match each other strongly but reach the map through a
   * few landmarks: the tracks they share have one registered view, so nothing triangulates them
   * until one of the views registers. Try each failed view with half the PnP consensus. Keep the
   * registration only when its tracks then triangulate at least `minPnP` new landmarks, which a
   * wrong pose cannot do. Roll back otherwise.
   */
  const registerWeak = function* (): Generator<ReconstructionJob, boolean, unknown> {
    const weakMinimum = Math.max(6, Math.ceil(minPnP / 2));
    for (const { view, correspondences } of rank()) {
      if (correspondences.length < weakMinimum) continue;
      let pose: ReturnType<typeof estimatePosePnP>;
      try {
        pose = yield* estimatePose(view, correspondences, weakMinimum);
      } catch (error) {
        if (error instanceof ReconstructionOperationError) throw error;
        continue;
      }
      const savedLandmarks = new Map(landmarks),
        savedTried = new Map(triedAt),
        savedFailure = failure.get(view.id);
      yield* addView(view, pose.pose);
      let fresh = 0;
      for (const [trackId, point] of landmarks)
        if (!savedLandmarks.has(trackId) && point.observations.some((o) => o.viewId === view.id))
          fresh++;
      if (fresh >= minPnP) {
        yield* settle();
        return true;
      }
      views.pop();
      registered.delete(view.id);
      order.pop();
      landmarks.clear();
      for (const [trackId, point] of savedLandmarks) landmarks.set(trackId, point);
      triedAt.clear();
      for (const [trackId, size] of savedTried) triedAt.set(trackId, size);
      if (savedFailure) failure.set(view.id, savedFailure);
    }
    return false;
  };
  while (views.length < images.length) {
    const support = rank();
    let added = false;
    for (const { view, correspondences } of support) {
      const previous = failure.get(view.id);
      if (correspondences.length < minPnP) {
        failure.set(
          view.id,
          tooFewCorrespondences(correspondences.length, minPnP, views.length, previous),
        );
        continue;
      }
      // Retry a failed registration once per map growth, a bounded number of times.
      if (retryWaits(previous, views.length)) continue;
      let pose: ReturnType<typeof estimatePosePnP>;
      try {
        pose = yield* estimatePose(view, correspondences, minPnP);
      } catch (error) {
        if (error instanceof ReconstructionOperationError) throw error;
        failure.set(view.id, pnpFailure(error, views.length, previous));
        continue;
      }
      yield* addView(view, pose.pose);
      yield* settle();
      added = true;
      break;
    }
    if (added) continue;
    // No view registered at this map size. One recovery pass retriangulates the tracks tried at a
    // smaller map or dropped by a filter, refines the whole map and gives the failed views one more
    // attempt against the refined landmarks. A second pass at the same size would repeat it.
    if (recoveredAt === views.length) {
      if (yield* registerWeak()) continue;
      break;
    }
    recoveredAt = views.length;
    yield* triangulate(graph.tracks.map((track) => track.id));
    yield* adjust(bundleIterations);
    publish();
    retryAfterRecovery(failure);
  }
  yield* triangulate(graph.tracks.map((track) => track.id));
  yield* adjust(
    finalBundle,
    false,
    options.refineSharedFocal ?? MAPPER_DEFAULTS.refineSharedFocal,
    finalize,
  );
  let finalModel = model();
  let finalEvaluation = yield* request<ObservationEvaluation>({
    kind: 'observations',
    model: finalModel,
    maxError,
    depth: controls.depth,
    stage: finalize ? 'final' : 'measure',
  });
  const removedInlier = finalize && applyEvaluation(finalModel, finalEvaluation);
  if (!finalize && controls.depth.maxRelativeDepthStdDev)
    deferSensitive(finalModel, finalEvaluation);
  if (finalize || controls.depth.maxRelativeDepthStdDev) finalModel = model();
  // The first mean includes inliers of points that may now have fewer than two observations.
  // Re-measure only when that changes the population; ordinary filtering needs no extra dispatch.
  let meanErrorPx = finalEvaluation.meanErrorPx;
  if (!landmarks.size) meanErrorPx = Infinity;
  else if (removedInlier) {
    finalEvaluation = yield* request<ObservationEvaluation>({
      kind: 'observations',
      model: finalModel,
      maxError,
      depth: controls.depth,
      stage: 'final',
    });
    meanErrorPx = finalEvaluation.meanErrorPx;
  }
  publish();
  return mapperResult({
    selection: selected.selection,
    reconstruction: finalModel,
    graph,
    images,
    registered: (id) => registered.has(id),
    failure,
    initialPair: [first.id, second.id],
    pairVerification: diagnostics,
    registrationOrder: order,
    bundleAdjustments: bundleReports,
    bundlePartitions,
    meanReprojectionErrorPx: meanErrorPx,
    tracking: keepTracking
      ? runTracking(
          controls,
          verification,
          graph.tracks,
          finalModel.landmarks.map((p) => p.trackId),
          depthDeferred,
        )
      : undefined,
    verifiedPairs: verified,
  });
}
