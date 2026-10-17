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
  readonly localBundleSize?: number;
  /** Restrict local BA point work to the camera window while retaining all selected observations. */
  readonly localBundleSubset?: boolean;
  readonly globalBundleFrequency?: number;
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
    }
  | { kind: 'bundle'; args: Parameters<typeof bundleAdjust> }
  | { kind: 'observations'; model: SparseReconstruction; maxError: number };
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
function* request<T>(job: ReconstructionJob): Generator<ReconstructionJob, T, unknown> {
  return (yield job) as T;
}
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
function executeCpu(job: ReconstructionJob): unknown {
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
      result = executeCpu(step.value);
    } catch (error) {
      step = machine.throw(error);
      continue;
    }
    step = machine.next(result);
  }
  return step.value;
}

/** @internal Common control flow; drivers select numerical execution explicitly. */
/** Calibrated, fixed-intrinsics incremental sparse SfM. Raw pair edges are verified before track construction. */
export function* incrementalSteps(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
  resume?: SparseReconstruction,
): Generator<ReconstructionJob, IncrementalResult, unknown> {
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
    globalFrequency = options.globalBundleFrequency ?? 3;
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
    !Number.isSafeInteger(globalFrequency) ||
    globalFrequency < 1
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
  const ordered = [...inputPairs].sort(
    (a, b) =>
      Math.min(a.first, a.second) - Math.min(b.first, b.second) ||
      Math.max(a.first, a.second) - Math.max(b.first, b.second),
  );
  const verified: PairMatches[] = [],
    diagnostics: PairVerification[] = [],
    candidates: { pair: PairMatches; result: ImagePairGeometry; score: number }[] = [];
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
    const verification = yield* request<ImagePairVerification>({
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
  let model: SparseReconstruction = resume ?? {
    views: [
      { id: first.id, camera: { ...first.camera }, pose: identityPose() },
      {
        id: second.id,
        camera: { ...second.camera },
        pose: { R: initial!.result.R.slice(), t: initial!.result.t.slice() },
      },
    ],
    landmarks: [],
  };
  const publish = () => options.onReconstruction?.(copyReconstruction(model));
  const order = model.views.map((view) => view.id),
    bundleReports: BundleAdjustmentResult['diagnostics'][] = [];
  const updateLandmarks = function* (): Generator<ReconstructionJob, void, unknown> {
    const registered = new Map(model.views.map((v) => [v.id, v]));
    const groups: { trackId: number; observations: LandmarkObservation[] }[] = [];
    for (const track of graph.tracks) {
      const observations: LandmarkObservation[] = track.observations
        .filter((o) => registered.has(o.viewId))
        .map((o) => {
          const v = byId.get(o.viewId)!;
          return {
            ...o,
            pixel: [v.keypoints[2 * o.featureId]!, v.keypoints[2 * o.featureId + 1]!],
          };
        });
      if (observations.length >= 2) groups.push({ trackId: track.id, observations });
    }
    const results = groups.length
      ? yield* request<readonly (TrackTriangulation | undefined)[]>({
          kind: 'triangulation',
          observations: groups.map((g) =>
            g.observations.map((o) => {
              const v = registered.get(o.viewId)!;
              return { camera: v.camera, pose: v.pose, pixel: o.pixel };
            }),
          ),
          options: { maxErrorPx: maxError, minAngleDeg: pointAngle },
        })
      : [];
    const landmarks: Landmark[] = [];
    groups.forEach((g, i) => {
      const result = results[i];
      if (result)
        landmarks.push({
          trackId: g.trackId,
          xyz: result.xyz,
          observations: g.observations.filter((_, j) => result.inliers[j]),
        });
    });
    model = { ...model, landmarks };
  };
  const adjust = function* (
    steps: number,
    local = false,
    refineSharedFocal = false,
  ): Generator<ReconstructionJob, void, unknown> {
    if (!steps || !model.landmarks.length) return;
    const window = local
      ? model.views
          .filter((v) => v.id !== first.id && v.id !== second.id)
          .slice(-localSize)
          .map((v) => v.id)
      : undefined;
    const subset =
      local && options.localBundleSubset ? extractLocalBundle(model, window!) : undefined;
    const result = yield* request<BundleAdjustmentResult>({
      kind: 'bundle',
      args: [
        subset?.reconstruction ?? model,
        {
          iterations: steps,
          anchorViewId: subset?.anchorViewId ?? first.id,
          scaleViewId: subset?.scaleViewId ?? second.id,
          huberPx: maxError / 2,
          variableViewIds: window,
          refineSharedFocal,
        },
      ],
    });
    model = subset ? mergeLocalBundle(model, subset, result.reconstruction) : result.reconstruction;
    bundleReports.push(result.diagnostics);
    // Local extraction changes no observation ownership; global passes still perform filtering.
    if (subset) return;
    // Filtering is explicit and occurs outside BA, whose observation set is immutable.
    const evaluation = yield* request<ObservationEvaluation>({
      kind: 'observations',
      model,
      maxError,
    });
    model = {
      ...model,
      landmarks: model.landmarks
        .map((p, i) => ({
          ...p,
          observations: p.observations.filter((_, j) => evaluation.masks[i]![j]),
        }))
        .filter((p) => p.observations.length >= 2),
    };
  };
  yield* updateLandmarks();
  if (model.landmarks.length < minInitial)
    throw new ReconstructionInitializationError(
      diagnostics,
      'Initial tracks did not produce enough reliable landmarks',
    );
  if (!resume) yield* adjust(bundleIterations);
  publish();
  const failure = new Map<number, string>();
  while (model.views.length < images.length) {
    const registered = new Set(model.views.map((v) => v.id)),
      pointByTrack = new Map(model.landmarks.map((p) => [p.trackId, p]));
    const available = images
      .filter((v) => !registered.has(v.id))
      .map((view) => ({
        view,
        correspondences: graph.tracks.flatMap((track) => {
          const observation = track.observations.find((o) => o.viewId === view.id),
            point = pointByTrack.get(track.id);
          return observation && point ? [{ observation, point }] : [];
        }),
      }))
      .sort((a, b) => b.correspondences.length - a.correspondences.length || a.view.id - b.view.id);
    let added = false;
    for (const { view, correspondences } of available) {
      if (correspondences.length < minPnP) {
        failure.set(
          view.id,
          `Only ${correspondences.length} 2D–3D correspondences; need ${minPnP}`,
        );
        continue;
      }
      const X = Float64Array.from(correspondences.flatMap((c) => [...c.point.xyz])),
        pixels = Float64Array.from(
          correspondences.flatMap((c) => {
            const i = c.observation.featureId;
            return [view.keypoints[2 * i]!, view.keypoints[2 * i + 1]!];
          }),
        );
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
        failure.set(view.id, (error as Error).message);
        continue;
      }
      const registeredView: RegisteredView = {
        id: view.id,
        camera: { ...view.camera },
        pose: pose.pose,
      };
      model = { ...model, views: [...model.views, registeredView] };
      order.push(view.id);
      failure.delete(view.id);
      yield* updateLandmarks();
      yield* adjust(bundleIterations, model.views.length % globalFrequency !== 0);
      publish();
      added = true;
      break;
    }
    if (!added) break;
  }
  yield* updateLandmarks();
  yield* adjust(finalBundle, false, options.refineSharedFocal ?? false);
  const registered = new Map(model.views.map((v) => [v.id, v]));
  const finalEvaluation = yield* request<ObservationEvaluation>({
    kind: 'observations',
    model,
    maxError,
  });
  publish();
  return {
    reconstruction: model,
    tracks: graph.tracks,
    unregistered: images
      .filter((v) => !registered.has(v.id))
      .map((v) => ({
        viewId: v.id,
        reason: failure.get(v.id) ?? 'Disconnected from reconstructed tracks',
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
