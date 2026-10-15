import { unprojectPixel } from '../geometry/camera.ts';
import { identityPose, projectPoint } from '../geometry/pose.ts';
import { estimateHomography } from '../geometry/homography.ts';
import { estimatePosePnP } from '../geometry/pnp.ts';
import { triangulateTrack } from '../geometry/multiview.ts';
import { reconstructTwoView } from '../sfm/two-view.ts';
import type { TwoViewResult } from '../sfm/two-view.ts';
import { bundleAdjust } from './bundle-adjustment.ts';
import type { BundleAdjustmentResult } from './bundle-adjustment.ts';
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
  readonly seed?: number;
  readonly ransacIterations?: number;
  readonly maxErrorPx?: number;
  readonly minInitialAngleDeg?: number;
  readonly minTriangulationAngleDeg?: number;
  readonly minPnPInliers?: number;
  readonly minInitialPoints?: number;
  readonly bundleIterations?: number;
  readonly finalBundleIterations?: number;
  readonly localBundleSize?: number;
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
  constructor(pairVerification: readonly PairVerification[]) {
    super(
      'No stable initial pair: check calibration, homography dominance, parallax and verified matches',
    );
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

/** Calibrated, fixed-intrinsics incremental sparse SfM. Raw pair edges are verified before track construction. */
export function reconstructIncremental(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
): IncrementalResult {
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
    candidates: { pair: PairMatches; result: TwoViewResult; score: number }[] = [];
  for (let pi = 0; pi < ordered.length; pi++) {
    const pair = ordered[pi]!,
      a = byId.get(pair.first)!,
      b = byId.get(pair.second)!;
    const packed: number[] = [],
      indices: number[] = [];
    for (let k = 0; k < pair.matches.length; k += 2) {
      const ai = pair.matches[k]!,
        bi = pair.matches[k + 1]!,
        x = unprojectPixel(a.camera, [a.keypoints[ai * 2]!, a.keypoints[ai * 2 + 1]!]),
        y = unprojectPixel(b.camera, [b.keypoints[bi * 2]!, b.keypoints[bi * 2 + 1]!]);
      if (!x || !y) continue;
      packed.push(
        x[0] * a.camera.fx + a.camera.cx,
        x[1] * a.camera.fy + a.camera.cy,
        y[0] * b.camera.fx + b.camera.cx,
        y[1] * b.camera.fy + b.camera.cy,
      );
      indices.push(k / 2);
    }
    const pixels = Float64Array.from(packed),
      base = { first: pair.first, second: pair.second, matches: indices.length };
    if (indices.length < 8) {
      diagnostics.push({
        ...base,
        inliers: 0,
        homographyInliers: 0,
        medianAngleDeg: 0,
        seedEligible: false,
        reason: 'Insufficient invertible correspondences',
      });
      continue;
    }
    const H = estimateHomography(pixels, {
      thresholdPx: maxError,
      iterations: Math.min(iterations, 500),
      seed: seed + pi,
    });
    try {
      const result = reconstructTwoView(pixels, a.camera, b.camera, {
        iterations,
        seed: seed + pi,
        threshold: maxError / Math.min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy),
        refinementIterations: 0,
        minParallaxDeg: Math.min(pointAngle, 0.1),
        maxReprojectionErrorPx: maxError,
      });
      const matched: number[] = [],
        distances: number[] = [];
      for (let i = 0; i < indices.length; i++)
        if (result.inliers[i]) {
          const source = indices[i]!;
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
      const hCount = H?.inlierCount ?? 0,
        dominant = hCount >= 0.85 * indices.length && hCount >= 0.9 * result.inlierCount;
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
    } catch (error) {
      diagnostics.push({
        ...base,
        inliers: 0,
        homographyInliers: H?.inlierCount ?? 0,
        medianAngleDeg: 0,
        seedEligible: false,
        reason: (error as Error).message,
      });
    }
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
  if (!initial) throw new ReconstructionInitializationError(diagnostics);
  const graph = buildTracks(images, verified),
    first = byId.get(initial.pair.first)!,
    second = byId.get(initial.pair.second)!;
  let model: SparseReconstruction = {
    views: [
      { id: first.id, camera: { ...first.camera }, pose: identityPose() },
      {
        id: second.id,
        camera: { ...second.camera },
        pose: { R: initial.result.R.slice(), t: initial.result.t.slice() },
      },
    ],
    landmarks: [],
  };
  const order = [first.id, second.id],
    bundleReports: BundleAdjustmentResult['diagnostics'][] = [];
  const updateLandmarks = () => {
    const registered = new Map(model.views.map((v) => [v.id, v])),
      landmarks: Landmark[] = [];
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
      if (observations.length < 2) continue;
      const result = triangulateTrack(
        observations.map((o) => {
          const v = registered.get(o.viewId)!;
          return { camera: v.camera, pose: v.pose, pixel: o.pixel };
        }),
        { maxErrorPx: maxError, minAngleDeg: pointAngle },
      );
      if (result)
        landmarks.push({
          trackId: track.id,
          xyz: result.xyz,
          observations: observations.filter((_, i) => result.inliers[i]),
        });
    }
    model = { ...model, landmarks };
  };
  const adjust = (steps: number, local = false) => {
    if (!steps || !model.landmarks.length) return;
    const window = local
      ? model.views
          .filter((v) => v.id !== first.id && v.id !== second.id)
          .slice(-localSize)
          .map((v) => v.id)
      : undefined;
    const result = bundleAdjust(model, {
      iterations: steps,
      anchorViewId: first.id,
      scaleViewId: second.id,
      huberPx: maxError / 2,
      variableViewIds: window,
    });
    model = result.reconstruction;
    bundleReports.push(result.diagnostics);
    // Filtering is explicit and occurs outside BA, whose observation set is immutable.
    model = {
      ...model,
      landmarks: model.landmarks
        .map((p) => ({
          ...p,
          observations: p.observations.filter((o) => {
            const v = model.views.find((v) => v.id === o.viewId)!,
              q = projectPoint(v.camera, v.pose, p.xyz);
            return q && Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]) <= maxError;
          }),
        }))
        .filter((p) => p.observations.length >= 2),
    };
  };
  updateLandmarks();
  if (model.landmarks.length < minInitial)
    throw new Error('Initial tracks did not produce enough reliable landmarks');
  adjust(bundleIterations);
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
        pose = estimatePosePnP(X, pixels, view.camera, {
          iterations,
          seed: seed + order.length,
          thresholdPx: maxError,
          minInliers: minPnP,
        });
      } catch (error) {
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
      updateLandmarks();
      adjust(bundleIterations, model.views.length % globalFrequency !== 0);
      added = true;
      break;
    }
    if (!added) break;
  }
  updateLandmarks();
  adjust(finalBundle);
  const registered = new Map(model.views.map((v) => [v.id, v]));
  let totalError = 0,
    totalObs = 0;
  for (const p of model.landmarks)
    for (const o of p.observations) {
      const v = registered.get(o.viewId)!,
        q = projectPoint(v.camera, v.pose, p.xyz)!;
      totalError += Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]);
      totalObs++;
    }
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
      meanReprojectionErrorPx: totalObs ? totalError / totalObs : Infinity,
    },
  };
}
