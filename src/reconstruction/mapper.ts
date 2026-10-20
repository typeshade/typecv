import { identityPose, projectPoint } from '../geometry/pose.ts';
import {
  verifyImagePair,
  type ImagePairVerification,
  type ImagePairGeometry,
} from '../geometry/image-pair.ts';
import { estimatePosePnP } from '../geometry/pnp.ts';
import { triangulateTrack } from '../geometry/multiview.ts';
import type { TriangulationObservation, TrackTriangulation } from '../geometry/multiview.ts';
import { bundleAdjust } from './bundle-adjustment.ts';
import type { BundleAdjustmentResult } from './bundle-adjustment.ts';
import { extractLocalBundle, mergeLocalBundle } from './local-bundle.ts';
import { buildTracks, validateFeatures } from './tracks.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  Landmark,
  LandmarkObservation,
  PairMatches,
  RegisteredView,
  SparseReconstruction,
} from './types.ts';

export interface IncrementalOptions {
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
}
export interface PairVerification {
  readonly first: number;
  readonly second: number;
  readonly matches: number;
  readonly inliers: number;
  readonly homographyInliers: number;
  readonly medianAngleDeg: number;
  readonly seedEligible: boolean;
  readonly reason?: string;
}
export class ReconstructionInitializationError extends Error {
  readonly pairVerification: readonly PairVerification[];
  constructor(
    pairVerification: readonly PairVerification[],
    message = 'No stable initial pair: check calibration, homography dominance, parallax and verified matches',
  ) {
    super(message);
    this.name = 'ReconstructionInitializationError';
    this.pairVerification = pairVerification;
  }
}
export interface IncrementalResult {
  readonly reconstruction: SparseReconstruction;
  readonly tracks: readonly FeatureTrack[];
  readonly unregistered: readonly { viewId: number; reason: string }[];
  readonly diagnostics: {
    readonly initialPair: readonly [number, number];
    readonly pairVerification: readonly PairVerification[];
    readonly conflictingTrackEdges: number;
    readonly registrationOrder: readonly number[];
    readonly bundleAdjustments: readonly BundleAdjustmentResult['diagnostics'][];
    readonly meanReprojectionErrorPx: number;
  };
}

/** @internal Numerical requests shared by the synchronous reference and asynchronous device driver. */
export type ReconstructionJob =
  | { kind: 'verifyPair'; args: Parameters<typeof verifyImagePair> }
  | { kind: 'pnp'; args: Parameters<typeof estimatePosePnP> }
  | {
      kind: 'triangulation';
      observations: readonly (readonly TriangulationObservation[])[];
      options: Parameters<typeof triangulateTrack>[1];
      /** Track of each group, for diagnostics; drivers ignore it. */
      trackIds: readonly number[];
    }
  | { kind: 'bundle'; args: Parameters<typeof bundleAdjust> }
  | {
      kind: 'observations';
      model: SparseReconstruction;
      maxError: number;
      /**
       * `continuation` checks a new view's observation appended last to each landmark; `filter`
       * checks every observation after a bundle pass; `final` reports the finished model's error.
       */
      stage: 'continuation' | 'filter' | 'final';
    };
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
function* request<T, J extends ReconstructionJob = ReconstructionJob>(
  job: J,
): Generator<J, T, unknown> {
  return (yield job) as T;
}
/** @internal The one request pair verification makes; both mappers share it. */
export type PairVerificationJob = Extract<ReconstructionJob, { kind: 'verifyPair' }>;
/** @internal */
export interface ObservationEvaluation {
  readonly masks: readonly Uint8Array[];
  readonly meanErrorPx: number;
}
function evaluateCpuObservations(
  model: SparseReconstruction,
  maxError: number,
): ObservationEvaluation {
  const views = new Map(model.views.map((v) => [v.id, v]));
  let total = 0,
    count = 0;
  const masks = model.landmarks.map((p) =>
    Uint8Array.from(p.observations, (o) => {
      const v = views.get(o.viewId)!,
        q = projectPoint(v.camera, v.pose, p.xyz);
      if (!q) return 0;
      const e = Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]);
      if (e > maxError) return 0;
      total += e;
      count++;
      return 1;
    }),
  );
  return { masks, meanErrorPx: count ? total / count : Infinity };
}
/** @internal Synchronous CPU execution of one numerical request. */
export function executeReconstructionJob(job: ReconstructionJob): unknown {
  switch (job.kind) {
    case 'verifyPair':
      return verifyImagePair(...job.args);
    case 'pnp':
      return estimatePosePnP(...job.args);
    case 'triangulation':
      return job.observations.map((o) => triangulateTrack(o, job.options));
    case 'bundle':
      return bundleAdjust(...job.args);
    case 'observations':
      return evaluateCpuObservations(job.model, job.maxError);
  }
}
/** Calibrated, fixed-intrinsics incremental sparse SfM. Raw edges are verified before tracks. */
export function reconstructIncremental(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
): IncrementalResult {
  const machine = incrementalSteps(images, inputPairs, options);
  let step = machine.next();
  while (!step.done) {
    let result: unknown;
    try {
      result = executeReconstructionJob(step.value);
    } catch (error) {
      step = machine.throw(error);
      continue;
    }
    step = machine.next(result);
  }
  return step.value;
}

/** A registered view is retried after each PnP failure only while the map has grown, at most this often. */
export const MAX_REGISTRATION_ATTEMPTS = 3;

/** @internal Common control flow; drivers select numerical execution explicitly. */
/**
 * Calibrated, fixed-intrinsics incremental sparse SfM. Raw pair edges are verified before
 * track construction. Landmarks persist across registrations: a new view continues the
 * tracks it observes, triangulates only the tracks it completes, and bundle adjustment
 * refines the retained points. Tracks that failed are retried after the map has grown.
 */
/** @internal The validated numerical controls of the incremental mapper. */
export interface IncrementalControls {
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
/** @internal Validate the inputs and options; throws RangeError on an invalid one. */
export function incrementalControls(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions,
): { byId: Map<number, ImageFeatures>; controls: IncrementalControls } {
  const byId = validateFeatures(images, inputPairs),
    seed = options.seed ?? 17,
    iterations = options.ransacIterations ?? 750,
    maxError = options.maxErrorPx ?? 3;
  const initialAngle = options.minInitialAngleDeg ?? 1,
    pointAngle = options.minTriangulationAngleDeg ?? 0.5,
    minPnP = options.minPnPInliers ?? 12,
    minInitial = options.minInitialPoints ?? 12;
  const bundleIterations = options.bundleIterations ?? 8,
    finalBundle = options.finalBundleIterations ?? 25;
  const localSize = options.localBundleSize ?? 6,
    globalViewRatio = options.globalBundleViewRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalPointRatio = options.globalBundlePointRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalFrequency = options.globalBundleFrequency ?? 0,
    stallTolerance = options.bundleStallTolerance ?? DEFAULT_BUNDLE_STALL_TOLERANCE;
  if (
    images.length < 2 ||
    !Number.isSafeInteger(seed) ||
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    !Number.isFinite(maxError) ||
    maxError <= 0 ||
    !Number.isFinite(initialAngle) ||
    initialAngle <= 0 ||
    initialAngle >= 180 ||
    !Number.isFinite(pointAngle) ||
    pointAngle < 0 ||
    pointAngle >= 180 ||
    !Number.isSafeInteger(minPnP) ||
    minPnP < 6 ||
    !Number.isSafeInteger(minInitial) ||
    minInitial < 8 ||
    !Number.isSafeInteger(bundleIterations) ||
    bundleIterations < 0 ||
    !Number.isSafeInteger(finalBundle) ||
    finalBundle < 0 ||
    !Number.isSafeInteger(localSize) ||
    localSize < 1 ||
    !(globalViewRatio > 1) ||
    !(globalPointRatio > 1) ||
    !Number.isSafeInteger(globalFrequency) ||
    globalFrequency < (options.globalBundleFrequency === undefined ? 0 : 1) ||
    !Number.isFinite(stallTolerance) ||
    stallTolerance < 0 ||
    stallTolerance >= 1
  )
    throw new RangeError('Invalid incremental reconstruction inputs/options');
  if (options.localBundleSubset !== undefined && typeof options.localBundleSubset !== 'boolean')
    throw new RangeError('localBundleSubset must be a boolean');
  if (
    options.initialPair &&
    (!byId.has(options.initialPair[0]) ||
      !byId.has(options.initialPair[1]) ||
      options.initialPair[0] === options.initialPair[1])
  )
    throw new RangeError('Invalid initial pair');
  const pairKeys = new Set<string>();
  for (const p of inputPairs) {
    const key = [p.first, p.second].sort((a, b) => a - b).join(':');
    if (pairKeys.has(key)) throw new RangeError('Duplicate image pair');
    pairKeys.add(key);
  }
  return {
    byId,
    controls: {
      seed,
      iterations,
      maxError,
      initialAngle,
      pointAngle,
      minPnP,
      minInitial,
      bundleIterations,
      finalBundle,
      localSize,
      localSubset: options.localBundleSubset ?? true,
      globalViewRatio,
      globalPointRatio,
      globalFrequency,
      stallTolerance,
    },
  };
}

/** COLMAP's `ba_global_images_ratio` and `ba_global_points_ratio`: 10 % growth. */
export const DEFAULT_GLOBAL_BUNDLE_RATIO = 1.1;
/** The default `bundleStallTolerance` of the mapper's BA passes. */
export const DEFAULT_BUNDLE_STALL_TOLERANCE = 1e-4;

/** @internal The model size at the last global BA, which the schedule compares the model to. */
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
/** @internal A verified pair that may seed the map, with its score. */
export interface SeedCandidate {
  readonly pair: PairMatches;
  readonly result: ImagePairGeometry;
  readonly score: number;
}
/**
 * @internal Verify every pair in id order with a stable seed per pair, keep the inlier edges and
 * score the pairs that may seed the map, best first.
 */
export function* verifyPairs(
  byId: Map<number, ImageFeatures>,
  inputPairs: readonly PairMatches[],
  controls: IncrementalControls,
): Generator<
  PairVerificationJob,
  { verified: PairMatches[]; diagnostics: PairVerification[]; candidates: SeedCandidate[] },
  unknown
> {
  const { seed, iterations, maxError, initialAngle, pointAngle, minInitial } = controls;
  const ordered = [...inputPairs].sort(
    (a, b) =>
      Math.min(a.first, a.second) - Math.min(b.first, b.second) ||
      Math.max(a.first, a.second) - Math.max(b.first, b.second),
  );
  const verified: PairMatches[] = [],
    diagnostics: PairVerification[] = [],
    candidates: SeedCandidate[] = [];
  for (let pi = 0; pi < ordered.length; pi++) {
    const pair = ordered[pi]!,
      a = byId.get(pair.first)!,
      b = byId.get(pair.second)!;
    const raw = new Float64Array(pair.matches.length * 2);
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
    const verification = yield* request<ImagePairVerification, PairVerificationJob>({
      kind: 'verifyPair',
      args: [
        raw,
        a.camera,
        b.camera,
        {
          iterations,
          seed: seed + pi,
          threshold: maxError / Math.min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy),
          refinementIterations: 0,
          minParallaxDeg: Math.min(pointAngle, 0.1),
          maxReprojectionErrorPx: maxError,
          homographyThresholdPx: maxError,
          homographyIterations: Math.min(iterations, 500),
        },
      ],
    });
    const base = { first: pair.first, second: pair.second, matches: verification.invertibleCount },
      result = verification.geometry,
      hCount = verification.homographyInlierCount;
    if (!result) {
      diagnostics.push({
        ...base,
        inliers: 0,
        homographyInliers: hCount,
        medianAngleDeg: 0,
        seedEligible: false,
        reason: verification.reason,
      });
      continue;
    }
    const matched: number[] = [],
      distances: number[] = [];
    for (const source of verification.inlierRows) {
      matched.push(pair.matches[2 * source]!, pair.matches[2 * source + 1]!);
      distances.push(pair.distances?.[source] ?? 0);
    }
    const accepted = {
      first: pair.first,
      second: pair.second,
      matches: Uint32Array.from(matched),
      distances: Float64Array.from(distances),
    };
    verified.push(accepted);
    const dominant =
      hCount >= 0.85 * verification.invertibleCount && hCount >= 0.9 * result.inlierCount;
    const eligible =
      !dominant &&
      result.diagnostics.medianParallaxDeg >= initialAngle &&
      result.positiveDepthCount >= minInitial;
    diagnostics.push({
      ...base,
      inliers: result.inlierCount,
      homographyInliers: hCount,
      medianAngleDeg: result.diagnostics.medianParallaxDeg,
      seedEligible: eligible,
      reason: eligible
        ? undefined
        : dominant
          ? 'Homography-dominated initialization'
          : result.positiveDepthCount < minInitial
            ? 'Insufficient initial points'
            : 'Insufficient initial parallax',
    });
    if (eligible)
      candidates.push({
        pair: accepted,
        result,
        score: result.positiveDepthCount * Math.min(10, result.diagnostics.medianParallaxDeg),
      });
  }

  candidates.sort(
    (a, b) => b.score - a.score || a.pair.first - b.pair.first || a.pair.second - b.pair.second,
  );
  return { verified, diagnostics, candidates };
}

export function* incrementalSteps(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
  resume?: SparseReconstruction,
  /** The size at the last global BA, carried across resumed runs; updated in place. */
  schedule?: BundleSchedule,
): Generator<ReconstructionJob, IncrementalResult, unknown> {
  const { byId, controls } = incrementalControls(images, inputPairs, options);
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
  const { verified, diagnostics, candidates } = yield* verifyPairs(byId, inputPairs, controls);
  const initial = options.initialPair
    ? candidates.find(
        (c) =>
          (c.pair.first === options.initialPair![0] && c.pair.second === options.initialPair![1]) ||
          (c.pair.first === options.initialPair![1] && c.pair.second === options.initialPair![0]),
      )
    : candidates[0];
  if (!initial && !resume) throw new ReconstructionInitializationError(diagnostics);
  const anchorPair = resume ? options.initialPair! : [initial!.pair.first, initial!.pair.second];
  const graph = buildTracks(images, verified),
    first = byId.get(anchorPair[0]!)!,
    second = byId.get(anchorPair[1]!)!;
  // Per-view index of the track graph: which tracks a view observes, through which feature.
  const viewTracks = new Map<number, { trackId: number; featureId: number }[]>(),
    trackOfObservation = new Map<string, number>();
  for (const view of images) viewTracks.set(view.id, []);
  for (const track of graph.tracks)
    for (const o of track.observations) {
      viewTracks.get(o.viewId)!.push({ trackId: track.id, featureId: o.featureId });
      trackOfObservation.set(`${o.viewId}:${o.featureId}`, track.id);
    }
  const pixelOf = (viewId: number, featureId: number): [number, number] => {
    const k = byId.get(viewId)!.keypoints;
    return [k[2 * featureId]!, k[2 * featureId + 1]!];
  };

  // Mutable map state. Landmarks persist across registrations and are keyed by track.
  const views: RegisteredView[] = resume
    ? resume.views.map((v) => ({
        ...v,
        camera: { ...v.camera },
        pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
      }))
    : [
        { id: first.id, camera: { ...first.camera }, pose: identityPose() },
        {
          id: second.id,
          camera: { ...second.camera },
          pose: { R: initial!.result.R.slice(), t: initial!.result.t.slice() },
        },
      ];
  const registered = new Map(views.map((v) => [v.id, v])),
    landmarks = new Map<number, Landmark>(),
    // Track id -> registered view count when triangulation last failed or the landmark was dropped.
    triedAt = new Map<number, number>();
  if (resume)
    for (const point of resume.landmarks) {
      // Track ids are rebuilt from the current edges; adopt a landmark through its observations.
      const trackId = point.observations
        .map((o) => trackOfObservation.get(`${o.viewId}:${o.featureId}`))
        .find((id) => id !== undefined);
      if (trackId === undefined) continue;
      const observations = point.observations.filter(
        (o) =>
          registered.has(o.viewId) &&
          trackOfObservation.get(`${o.viewId}:${o.featureId}`) === trackId,
      );
      const existing = landmarks.get(trackId);
      if (
        observations.length >= 2 &&
        (!existing || existing.observations.length < observations.length)
      )
        landmarks.set(trackId, { trackId, xyz: point.xyz.slice(), observations });
    }
  const model = (): SparseReconstruction => ({ views, landmarks: [...landmarks.values()] });
  const publish = () => options.onReconstruction?.(copyReconstruction(model()));
  const order = views.map((view) => view.id),
    bundleReports: BundleAdjustmentResult['diagnostics'][] = [];
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
    const results = yield* request<readonly (TrackTriangulation | undefined)[]>({
      kind: 'triangulation',
      observations: groups.map((g) =>
        g.observations.map((o) => {
          const v = registered.get(o.viewId)!;
          return { camera: v.camera, pose: v.pose, pixel: o.pixel };
        }),
      ),
      options: { maxErrorPx: maxError, minAngleDeg: pointAngle },
      trackIds: groups.map((g) => g.trackId),
    });
    groups.forEach((g, i) => {
      const result = results[i];
      if (result)
        landmarks.set(g.trackId, {
          trackId: g.trackId,
          xyz: result.xyz,
          observations: g.observations.filter((_, j) => result.inliers[j]),
        });
      else triedAt.set(g.trackId, views.length);
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
      stage: 'continuation',
    });
    candidates.forEach((candidate, i) => {
      const mask = evaluation.masks[i]!;
      if (mask[mask.length - 1]) landmarks.set(candidate.trackId, candidate);
    });
  };
  /** Drop observations that no longer reproject; a landmark below two observations may be re-triangulated later. */
  const filter = function* (): Generator<ReconstructionJob, void, unknown> {
    if (!landmarks.size) return;
    const current = model();
    const evaluation = yield* request<ObservationEvaluation>({
      kind: 'observations',
      model: current,
      maxError,
      stage: 'filter',
    });
    current.landmarks.forEach((p, i) => {
      const observations = p.observations.filter((_, j) => evaluation.masks[i]![j]);
      if (observations.length >= 2) {
        if (observations.length !== p.observations.length)
          landmarks.set(p.trackId, { ...p, observations });
      } else {
        landmarks.delete(p.trackId);
        triedAt.set(p.trackId, views.length);
      }
    });
  };
  const adjust = function* (
    steps: number,
    local = false,
    refineSharedFocal = false,
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
    const result = yield* request<BundleAdjustmentResult>({
      kind: 'bundle',
      args: [
        subset?.reconstruction ?? current,
        {
          iterations: steps,
          anchorViewId: subset?.anchorViewId ?? first.id,
          scaleViewId: subset?.scaleViewId ?? second.id,
          huberPx: maxError / 2,
          variableViewIds: window,
          refineSharedFocal,
          stallTolerance,
        },
      ],
    });
    const refined = subset
      ? mergeLocalBundle(current, subset, result.reconstruction)
      : result.reconstruction;
    for (const view of refined.views) {
      const index = views.findIndex((v) => v.id === view.id);
      views[index] = view;
      registered.set(view.id, view);
    }
    for (const point of refined.landmarks) landmarks.set(point.trackId, point);
    bundleReports.push(result.diagnostics);
    // Local extraction changes no observation ownership; other passes filter explicitly outside BA.
    if (subset) return;
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
  yield* triangulate(viewTracks.get(first.id)!.map((entry) => entry.trackId));
  if (landmarks.size < minInitial)
    throw new ReconstructionInitializationError(
      diagnostics,
      'Initial tracks did not produce enough reliable landmarks',
    );
  if (!resume) yield* adjust(bundleIterations);
  publish();
  const failure = new Map<number, { reason: string; at: number; attempts: number }>();
  // Map size at which a recovery pass (global retriangulation and refinement) last ran.
  let recoveredAt = -1;
  while (views.length < images.length) {
    const support = images
      .filter((v) => !registered.has(v.id))
      .map((view) => ({
        view,
        correspondences: viewTracks.get(view.id)!.flatMap((entry) => {
          const point = landmarks.get(entry.trackId);
          return point ? [{ featureId: entry.featureId, point }] : [];
        }),
      }))
      .sort((a, b) => b.correspondences.length - a.correspondences.length || a.view.id - b.view.id);
    let added = false;
    for (const { view, correspondences } of support) {
      const previous = failure.get(view.id);
      if (correspondences.length < minPnP) {
        failure.set(view.id, {
          reason: `Only ${correspondences.length} 2D–3D correspondences; need ${minPnP}`,
          at: views.length,
          attempts: previous?.attempts ?? 0,
        });
        continue;
      }
      // Retry a failed registration once per map growth, a bounded number of times.
      if (
        previous &&
        (previous.at === views.length || previous.attempts >= MAX_REGISTRATION_ATTEMPTS)
      )
        continue;
      const X = Float64Array.from(correspondences.flatMap((c) => [...c.point.xyz])),
        pixels = Float64Array.from(correspondences.flatMap((c) => pixelOf(view.id, c.featureId)));
      let pose: ReturnType<typeof estimatePosePnP>;
      try {
        pose = yield* request<ReturnType<typeof estimatePosePnP>>({
          kind: 'pnp',
          args: [
            X,
            pixels,
            view.camera,
            {
              iterations,
              seed: seed + order.length,
              thresholdPx: maxError,
              minInliers: minPnP,
            },
          ],
        });
      } catch (error) {
        if (error instanceof ReconstructionOperationError) throw error;
        failure.set(view.id, {
          reason: (error as Error).message,
          at: views.length,
          attempts: (previous?.attempts ?? 0) + 1,
        });
        continue;
      }
      const registeredView: RegisteredView = {
        id: view.id,
        camera: { ...view.camera },
        pose: pose.pose,
      };
      views.push(registeredView);
      registered.set(view.id, registeredView);
      order.push(view.id);
      failure.delete(view.id);
      yield* continueTracks(view.id);
      yield* triangulate(viewTracks.get(view.id)!.map((entry) => entry.trackId));
      const global = globalBundleDue(views.length, landmarks.size, last, controls);
      if (global) yield* triangulate(graph.tracks.map((track) => track.id));
      yield* adjust(bundleIterations, !global);
      publish();
      added = true;
      break;
    }
    if (added) continue;
    // No view registered at this map size. One recovery pass retriangulates the tracks tried at a
    // smaller map or dropped by a filter, refines the whole map and gives the failed views one more
    // attempt against the refined landmarks. A second pass at the same size would repeat it.
    if (recoveredAt === views.length) break;
    recoveredAt = views.length;
    yield* triangulate(graph.tracks.map((track) => track.id));
    yield* adjust(bundleIterations);
    publish();
    for (const [id, entry] of failure) failure.set(id, { ...entry, at: -1 });
  }
  yield* triangulate(graph.tracks.map((track) => track.id));
  yield* adjust(finalBundle, false, options.refineSharedFocal ?? false);
  const finalModel = model();
  const finalEvaluation = yield* request<ObservationEvaluation>({
    kind: 'observations',
    model: finalModel,
    maxError,
    stage: 'final',
  });
  publish();
  return {
    reconstruction: finalModel,
    tracks: graph.tracks,
    unregistered: images
      .filter((v) => !registered.has(v.id))
      .map((v) => ({
        viewId: v.id,
        reason: failure.get(v.id)?.reason ?? 'Disconnected from reconstructed tracks',
      })),
    diagnostics: {
      initialPair: [first.id, second.id],
      pairVerification: diagnostics,
      conflictingTrackEdges: graph.conflictingEdges,
      registrationOrder: order,
      bundleAdjustments: bundleReports,
      meanReprojectionErrorPx: finalEvaluation.meanErrorPx,
    },
  };
}

/** @internal Copy all numerical state crossing a session or observer boundary. */
export function copyReconstruction(model: SparseReconstruction): SparseReconstruction {
  return {
    views: model.views.map((view) => ({
      ...view,
      camera: { ...view.camera },
      pose: { R: view.pose.R.slice(), t: view.pose.t.slice() },
    })),
    landmarks: model.landmarks.map((point) => ({
      ...point,
      xyz: point.xyz.slice(),
      observations: point.observations.map((observation) => ({
        ...observation,
        pixel: [...observation.pixel] as [number, number],
      })),
    })),
  };
}
