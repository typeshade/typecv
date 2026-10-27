import { drive } from '../protocol/drive.ts';
import { withDepthStep } from '../protocol/depth-step.ts';
import { identityPose, projectPoint } from '../../geometry/pose.ts';
import type { Pose, Camera } from '../../contracts/camera.ts';
import { estimatePosePnP } from '../../geometry/pnp.ts';
import type { PnPOptions, PnPResult } from '../../geometry/pnp.ts';
import { assessTrackDepth, triangulateTrackDetailed } from '../../geometry/multiview.ts';
import type { DepthSensitivityOptions } from '../../geometry/multiview.ts';
import { bundleAdjust } from '../bundle/bundle-adjustment.ts';
import type { BundleAdjustmentResult } from '../bundle/bundle-adjustment.ts';
import { fitBundleDirection, supportedBundleDirections } from '../bundle/directional-tracks.ts';
import {
  extractLocalBundle,
  mergeLocalBundle,
  planConnectedBundles,
  mergeBundleRefinements,
} from '../bundle/local-bundle.ts';
import type { BundleAdjustmentPartition } from '../bundle/local-bundle.ts';
import { buildTracks } from '../tracks/tracks.ts';
import { trackingEvidence } from '../tracks/tracking.ts';
import { selectCameraObservations } from '../tracks/observation-selection.ts';
import { ALIVE, HostScene, PIXEL_STRIDE } from '../scene/scene.ts';
import {
  MAX_REGISTRATION_ATTEMPTS,
  globalBundleDue,
  executeReconstructionJob,
  verifyPairs,
} from './mapper.ts';
import {
  ReconstructionInitializationError,
  ReconstructionOperationError,
} from '../protocol/errors.ts';
import { incrementalControls } from '../config/incremental-controls.ts';
import type {
  BundleSchedule,
  IncrementalOptions,
  IncrementalResult,
  PairVerificationJob,
} from './mapper.ts';
import type {
  ImageFeatures,
  Landmark,
  PairMatches,
  SparseReconstruction,
} from '../../contracts/scene.ts';
import { MAPPER_DEFAULTS, bundleHuberPx, maxErrorOfHuberPx } from '../config/defaults.ts';
import { observationKey } from '../../contracts/keys.ts';
import { TriangulationStatus } from '../../contracts/triangulation-status.ts';

/** A triangulation result in the point channel that defers a depth: the status, negated. */
const DEPTH_DEFERRED_POINT = -TriangulationStatus.depthSensitivity;

/** One track to triangulate: its registered observations as camera indices, feature ids and pixels. */
export interface TriangulationGroup {
  readonly trackId: number;
  readonly cameras: Uint32Array;
  readonly features: Uint32Array;
  readonly pixels: Float64Array;
}
export interface SceneFilterResult {
  readonly depthRejectedTracks?: readonly number[];
  readonly observationsRemoved: number;
  readonly pointsRemoved: number;
  /** For each point index before the filter, its index after it, or -1 when removed. */
  readonly pointRemap: Int32Array;
}
export interface SceneBundleResult {
  readonly diagnostics: readonly BundleAdjustmentResult['diagnostics'][];
  readonly partition?: BundleAdjustmentPartition;
  /** Whether the local subset was extracted; the mapper filters after a full problem only. */
  readonly subset: boolean;
}
/**
 * @internal The index-based requests of the scene mapper (`docs/gpu-resident-scene.md`, Mapper
 * protocol). The driver owns the scene. A request carries indices and small typed arrays; the
 * object model crosses the boundary in `load` (once, on resume) and `snapshot` (on request).
 */
export type SceneRequest =
  | PairVerificationJob
  /** Replace the scene with a model: camera i is `views[i]`, point j is `landmarks[j]`. */
  | { kind: 'load'; model: SparseReconstruction }
  /** Append a camera; the result is its index, the next free one. */
  | { kind: 'register'; viewId: number; camera: Camera; pose: Pose }
  /**
   * Append an observation of each candidate point by `camera` that reprojects within
   * `maxErrorPx`; the result flags the accepted candidates.
   */
  | {
      kind: 'continue';
      camera: number;
      points: Uint32Array;
      features: Uint32Array;
      pixels: Float64Array;
      maxErrorPx: number;
    }
  /** Append a point with its inlier observations for each accepted group; the result is the point index of each group, or -1. */
  | {
      kind: 'triangulate';
      groups: readonly TriangulationGroup[];
      maxErrorPx: number;
      minAngleDeg: number;
      depth?: DepthSensitivityOptions;
    }
  /** Drop the observations beyond `maxErrorPx`, the points left with fewer than two, and compact. */
  | { kind: 'filter'; maxErrorPx: number; depth?: DepthSensitivityOptions; depthOnly?: boolean }
  /** Pose of a new view from its candidate points and pixels; throws `GeometryEstimationError`. */
  | { kind: 'pnp'; camera: Camera; points: Uint32Array; pixels: Float64Array; options: PnPOptions }
  /** Refine the scene in place; `anchor` and `scale` are camera indices, `variable` the local window. */
  | {
      kind: 'bundle';
      iterations: number;
      anchor: number;
      scale: number;
      huberPx: number;
      variable?: Uint32Array;
      subset: boolean;
      refineSharedFocal: boolean;
      stallTolerance: number;
      /** Pending observed tracks for a final bearing-only BA, after finite retriangulation. */
      directionalGroups?: readonly TriangulationGroup[];
    }
  /** Mean reprojection error in pixels of the alive observations within `maxErrorPx`. */
  | { kind: 'measure'; maxErrorPx: number }
  | { kind: 'snapshot' };

function* request<T>(job: SceneRequest): Generator<SceneRequest, T, unknown> {
  return (yield job) as T;
}
function driverError(message: string): ReconstructionOperationError {
  return new ReconstructionOperationError(new Error(message));
}

/**
 * The incremental mapper over the scene protocol: the same control flow as `incrementalSteps`,
 * with the map held by the driver and addressed by camera and point index. The host keeps the
 * view-to-camera and track-to-point maps, the track graph and the per-view feature index.
 */
export function* sceneSteps(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
  resume?: SparseReconstruction,
  /** The size at the last global BA, carried across resumed runs; updated in place. */
  schedule?: BundleSchedule,
): Generator<SceneRequest, IncrementalResult, unknown> {
  const selected = selectCameraObservations(images, inputPairs, options.cameraObservationMasks);
  inputPairs = selected.pairs;
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
  const { verified, diagnostics, candidates, trackingPairs, rotations } = yield* verifyPairs(
    byId,
    inputPairs,
    controls,
  );
  const initial = options.initialPair
    ? candidates.find(
        (c) =>
          (c.pair.first === options.initialPair![0] && c.pair.second === options.initialPair![1]) ||
          (c.pair.first === options.initialPair![1] && c.pair.second === options.initialPair![0]),
      )
    : candidates[0];
  if (!initial && !resume)
    throw new ReconstructionInitializationError(
      diagnostics,
      undefined,
      undefined,
      keepTracking
        ? trackingEvidence(
            controls.rotationTracking ? trackingPairs : verified,
            rotations,
            buildTracks(images, controls.rotationTracking ? trackingPairs : verified).tracks,
          )
        : undefined,
      selected.selection,
    );
  const anchorPair = resume ? options.initialPair! : [initial!.pair.first, initial!.pair.second];
  const graph = buildTracks(images, controls.rotationTracking ? trackingPairs : verified),
    first = byId.get(anchorPair[0]!)!,
    second = byId.get(anchorPair[1]!)!;
  const viewTracks = new Map<number, { trackId: number; featureId: number }[]>(),
    trackOfObservation = new Map<string, number>();
  for (const view of images) viewTracks.set(view.id, []);
  for (const track of graph.tracks)
    for (const o of track.observations) {
      viewTracks.get(o.viewId)!.push({ trackId: track.id, featureId: o.featureId });
      trackOfObservation.set(observationKey(o.viewId, o.featureId), track.id);
    }
  const pixelOf = (viewId: number, featureId: number): [number, number] => {
    const k = byId.get(viewId)!.keypoints;
    return [k[2 * featureId]!, k[2 * featureId + 1]!];
  };

  // Index-based host state: no pose, point or observation lives here.
  const cameraOf = new Map<number, number>(),
    cameraIds: number[] = [],
    pointOf = new Map<number, number>(),
    triedAt = new Map<number, number>();
  let trackOfPoint: number[] = [];
  const bundleReports: BundleAdjustmentResult['diagnostics'][] = [],
    bundlePartitions: BundleAdjustmentPartition[] = [];
  const register = function* (
    view: ImageFeatures,
    pose: Pose,
  ): Generator<SceneRequest, void, unknown> {
    const index = yield* request<number>({
      kind: 'register',
      viewId: view.id,
      camera: { ...view.camera },
      pose,
    });
    if (index !== cameraIds.length) throw driverError('Camera registered out of order');
    cameraIds.push(view.id);
    cameraOf.set(view.id, index);
  };
  if (resume) {
    // Track ids are rebuilt from the current edges; adopt a landmark through its observations.
    const views = resume.views.map((v) => ({
        ...v,
        camera: { ...v.camera },
        pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
      })),
      registered = new Set(views.map((v) => v.id)),
      adopted = new Map<number, Landmark>();
    for (const point of resume.landmarks) {
      const trackId = point.observations
        .map((o) => trackOfObservation.get(observationKey(o.viewId, o.featureId)))
        .find((id) => id !== undefined);
      if (trackId === undefined) continue;
      const observations = point.observations.filter(
        (o) =>
          registered.has(o.viewId) &&
          trackOfObservation.get(observationKey(o.viewId, o.featureId)) === trackId,
      );
      const existing = adopted.get(trackId);
      if (
        observations.length >= 2 &&
        (!existing || existing.observations.length < observations.length)
      )
        adopted.set(trackId, { trackId, xyz: point.xyz.slice(), observations });
    }
    const model = { views, landmarks: [...adopted.values()] };
    yield* request<void>({ kind: 'load', model });
    views.forEach((v, i) => {
      cameraIds.push(v.id);
      cameraOf.set(v.id, i);
    });
    trackOfPoint = model.landmarks.map((p) => p.trackId);
    trackOfPoint.forEach((trackId, j) => pointOf.set(trackId, j));
  } else {
    yield* register(first, identityPose());
    yield* register(second, { R: initial!.result.R.slice(), t: initial!.result.t.slice() });
  }
  const publish = function* (): Generator<SceneRequest, void, unknown> {
    if (!options.onReconstruction) return;
    options.onReconstruction(yield* request<SparseReconstruction>({ kind: 'snapshot' }));
  };
  /** Triangulate tracks without a point that have two registered observations and no attempt at this map size. */
  const triangulate = function* (
    trackIds: Iterable<number>,
  ): Generator<SceneRequest, void, unknown> {
    const groups: TriangulationGroup[] = [];
    for (const trackId of [...new Set(trackIds)].sort((a, b) => a - b)) {
      if (pointOf.has(trackId) || (triedAt.get(trackId) ?? -1) >= cameraIds.length) continue;
      const observations = graph.tracks[trackId]!.observations.filter((o) =>
        cameraOf.has(o.viewId),
      );
      if (observations.length >= 2)
        groups.push({
          trackId,
          cameras: Uint32Array.from(observations, (o) => cameraOf.get(o.viewId)!),
          features: Uint32Array.from(observations, (o) => o.featureId),
          pixels: Float64Array.from(observations.flatMap((o) => pixelOf(o.viewId, o.featureId))),
        });
    }
    if (!groups.length) return;
    const points = yield* request<Int32Array>({
      kind: 'triangulate',
      groups,
      maxErrorPx: maxError,
      minAngleDeg: pointAngle,
      depth: controls.depth,
    });
    if (points.length !== groups.length) throw driverError('Triangulation result size');
    groups.forEach((g, i) => {
      const point = points[i]!;
      if (point < 0) {
        if (point === DEPTH_DEFERRED_POINT) depthDeferred.add(g.trackId);
        triedAt.set(g.trackId, cameraIds.length);
        return;
      }
      depthDeferred.delete(g.trackId);
      if (point !== trackOfPoint.length) throw driverError('Point appended out of order');
      trackOfPoint.push(g.trackId);
      pointOf.set(g.trackId, point);
    });
  };
  /** Observe each point of a track the new view observes, when the view reprojects it. */
  const continueTracks = function* (viewId: number): Generator<SceneRequest, void, unknown> {
    const points: number[] = [],
      features: number[] = [],
      pixels: number[] = [];
    for (const { trackId, featureId } of viewTracks.get(viewId)!) {
      const point = pointOf.get(trackId);
      if (point === undefined) continue;
      points.push(point);
      features.push(featureId);
      pixels.push(...pixelOf(viewId, featureId));
    }
    if (!points.length) return;
    yield* request<Uint8Array>({
      kind: 'continue',
      camera: cameraOf.get(viewId)!,
      points: Uint32Array.from(points),
      features: Uint32Array.from(features),
      pixels: Float64Array.from(pixels),
      maxErrorPx: maxError,
    });
  };
  /** Drop what no longer reprojects; a track that lost its point may be re-triangulated later. */
  const filter = function* (depthOnly = false): Generator<SceneRequest, void, unknown> {
    if (!trackOfPoint.length) return;
    const result = yield* request<SceneFilterResult>({
      kind: 'filter',
      maxErrorPx: maxError,
      depth: controls.depth,
      depthOnly,
    });
    for (const track of result.depthRejectedTracks ?? []) depthDeferred.add(track);
    if (result.pointRemap.length !== trackOfPoint.length) throw driverError('Filter remap size');
    const next: number[] = [];
    for (let i = 0; i < trackOfPoint.length; i++) {
      const trackId = trackOfPoint[i]!,
        to = result.pointRemap[i]!;
      if (to < 0) {
        pointOf.delete(trackId);
        triedAt.set(trackId, cameraIds.length);
        continue;
      }
      if (to !== next.length) throw driverError('Filter renumbered out of order');
      next.push(trackId);
      pointOf.set(trackId, to);
    }
    trackOfPoint = next;
  };
  const adjust = function* (
    steps: number,
    local = false,
    refineSharedFocal = false,
    includeDirections = false,
  ): Generator<SceneRequest, void, unknown> {
    if (!steps || !trackOfPoint.length) return;
    const window = local
      ? cameraIds.filter((id) => id !== first.id && id !== second.id).slice(-localSize)
      : undefined;
    const directionalGroups: TriangulationGroup[] = [];
    if (includeDirections && controls.separateDirectionalObservations)
      for (const track of graph.tracks) {
        if (pointOf.has(track.id)) continue;
        const observations = track.observations.filter((o) => cameraOf.has(o.viewId));
        if (observations.length >= 3)
          directionalGroups.push({
            trackId: track.id,
            cameras: Uint32Array.from(observations, (o) => cameraOf.get(o.viewId)!),
            features: Uint32Array.from(observations, (o) => o.featureId),
            pixels: Float64Array.from(observations.flatMap((o) => pixelOf(o.viewId, o.featureId))),
          });
      }
    const result = yield* request<SceneBundleResult>({
      kind: 'bundle',
      iterations: steps,
      anchor: cameraOf.get(first.id)!,
      scale: cameraOf.get(second.id)!,
      huberPx: bundleHuberPx(maxError),
      variable: window ? Uint32Array.from(window, (id) => cameraOf.get(id)!) : undefined,
      subset: local && localSubset,
      refineSharedFocal,
      stallTolerance,
      ...(directionalGroups.length ? { directionalGroups } : {}),
    });
    bundleReports.push(...result.diagnostics);
    if (result.partition) bundlePartitions.push(result.partition);
    if (!result.diagnostics.length) return;
    // Local extraction changes no observation ownership; other passes filter explicitly outside BA.
    if (result.subset) {
      if (controls.depth.maxRelativeDepthStdDev) yield* filter(true);
      return;
    }
    yield* filter();
    if (!local) {
      last.views = cameraIds.length;
      last.points = trackOfPoint.length;
    }
  };
  // Without a carried schedule, the map as it starts counts as globally adjusted.
  const last: BundleSchedule = schedule ?? {
    views: cameraIds.length,
    points: trackOfPoint.length,
  };
  last.views = Math.min(last.views, cameraIds.length);
  last.points = Math.min(last.points, trackOfPoint.length);
  yield* triangulate(viewTracks.get(first.id)!.map((entry) => entry.trackId));
  if (trackOfPoint.length < minInitial)
    throw new ReconstructionInitializationError(
      diagnostics,
      'Initial tracks did not produce enough reliable landmarks',
      undefined,
      keepTracking
        ? trackingEvidence(
            controls.rotationTracking ? trackingPairs : verified,
            rotations,
            graph.tracks,
            [],
            controls.depth.maxRelativeDepthStdDev ? depthDeferred : undefined,
          )
        : undefined,
      selected.selection,
    );
  if (!resume) yield* adjust(bundleIterations);
  yield* publish();
  const failure = new Map<number, { reason: string; at: number; attempts: number }>();
  let recoveredAt = -1;
  while (cameraIds.length < images.length) {
    const support = images
      .filter((v) => !cameraOf.has(v.id))
      .map((view) => ({
        view,
        correspondences: viewTracks.get(view.id)!.flatMap((entry) => {
          const point = pointOf.get(entry.trackId);
          return point === undefined ? [] : [{ featureId: entry.featureId, point }];
        }),
      }))
      .sort((a, b) => b.correspondences.length - a.correspondences.length || a.view.id - b.view.id);
    let added = false;
    for (const { view, correspondences } of support) {
      const previous = failure.get(view.id);
      if (correspondences.length < minPnP) {
        failure.set(view.id, {
          reason: `Only ${correspondences.length} 2D–3D correspondences; need ${minPnP}`,
          at: cameraIds.length,
          attempts: previous?.attempts ?? 0,
        });
        continue;
      }
      if (
        previous &&
        (previous.at === cameraIds.length || previous.attempts >= MAX_REGISTRATION_ATTEMPTS)
      )
        continue;
      let pose: PnPResult;
      try {
        pose = yield* request<PnPResult>({
          kind: 'pnp',
          camera: view.camera,
          points: Uint32Array.from(correspondences, (c) => c.point),
          pixels: Float64Array.from(correspondences.flatMap((c) => pixelOf(view.id, c.featureId))),
          options: {
            iterations,
            seed: seed + cameraIds.length,
            thresholdPx: maxError,
            minInliers: minPnP,
          },
        });
      } catch (error) {
        if (error instanceof ReconstructionOperationError) throw error;
        failure.set(view.id, {
          reason: (error as Error).message,
          at: cameraIds.length,
          attempts: (previous?.attempts ?? 0) + 1,
        });
        continue;
      }
      yield* register(view, pose.pose);
      failure.delete(view.id);
      yield* continueTracks(view.id);
      yield* triangulate(viewTracks.get(view.id)!.map((entry) => entry.trackId));
      const global = globalBundleDue(cameraIds.length, trackOfPoint.length, last, controls);
      if (global) yield* triangulate(graph.tracks.map((track) => track.id));
      yield* adjust(bundleIterations, !global);
      yield* publish();
      added = true;
      break;
    }
    if (added) continue;
    if (recoveredAt === cameraIds.length) break;
    recoveredAt = cameraIds.length;
    yield* triangulate(graph.tracks.map((track) => track.id));
    yield* adjust(bundleIterations);
    yield* publish();
    for (const [id, entry] of failure) failure.set(id, { ...entry, at: -1 });
  }
  yield* triangulate(graph.tracks.map((track) => track.id));
  yield* adjust(
    finalBundle,
    false,
    options.refineSharedFocal ?? MAPPER_DEFAULTS.refineSharedFocal,
    true,
  );
  if (controls.depth.maxRelativeDepthStdDev) yield* filter();
  const meanReprojectionErrorPx = yield* request<number>({ kind: 'measure', maxErrorPx: maxError });
  const reconstruction = yield* request<SparseReconstruction>({ kind: 'snapshot' });
  options.onReconstruction?.(yield* request<SparseReconstruction>({ kind: 'snapshot' }));
  return {
    ...(selected.selection ? { observationSelection: selected.selection } : {}),
    reconstruction,
    tracks: graph.tracks,
    ...(keepTracking
      ? {
          tracking: trackingEvidence(
            controls.rotationTracking ? trackingPairs : verified,
            rotations,
            graph.tracks,
            reconstruction.landmarks.map((p) => p.trackId),
            controls.depth.maxRelativeDepthStdDev ? depthDeferred : undefined,
          ),
        }
      : {}),
    unregistered: images
      .filter((v) => !cameraOf.has(v.id))
      .map((v) => ({
        viewId: v.id,
        reason: failure.get(v.id)?.reason ?? 'Disconnected from reconstructed tracks',
      })),
    diagnostics: {
      initialPair: [first.id, second.id],
      pairVerification: diagnostics,
      conflictingTrackEdges: graph.conflictingEdges,
      registrationOrder: cameraIds,
      bundleAdjustments: bundleReports,
      ...(bundlePartitions.length ? { bundlePartitions } : {}),
      meanReprojectionErrorPx,
    },
  };
}

/** @internal The CPU driver: the scene protocol over a `HostScene`, one request at a time. */
export class HostSceneDriver {
  scene: HostScene;
  constructor(scene = new HostScene()) {
    this.scene = scene;
  }
  execute(job: SceneRequest): unknown {
    const scene = this.scene;
    switch (job.kind) {
      case 'verifyPair':
        return executeReconstructionJob(job);
      case 'load':
        this.scene = HostScene.fromReconstruction(job.model);
        return undefined;
      case 'register':
        return scene.registerCamera(job.viewId, job.camera, job.pose);
      case 'continue': {
        const camera = scene.camera(job.camera),
          pose = scene.pose(job.camera),
          accepted = new Uint8Array(job.points.length);
        for (let i = 0; i < job.points.length; i++) {
          const q = projectPoint(camera, pose, scene.point(job.points[i]!));
          if (!q) continue;
          const x = job.pixels[2 * i]!,
            y = job.pixels[2 * i + 1]!;
          if (Math.hypot(q[0] - x, q[1] - y) > job.maxErrorPx) continue;
          scene.addObservation(job.camera, job.points[i]!, job.features[i]!, x, y);
          accepted[i] = 1;
        }
        return accepted;
      }
      case 'triangulate': {
        const points = new Int32Array(job.groups.length).fill(-1);
        job.groups.forEach((group, g) => {
          const observations = Array.from(group.cameras, (c, k) => ({
            camera: scene.camera(c),
            pose: scene.pose(c),
            pixel: [group.pixels[2 * k]!, group.pixels[2 * k + 1]!] as const,
          }));
          const detailed = triangulateTrackDetailed(observations, {
            maxErrorPx: job.maxErrorPx,
            minAngleDeg: job.minAngleDeg,
            ...job.depth,
          });
          const result = detailed.track;
          if (detailed.status === TriangulationStatus.depthSensitivity)
            points[g] = DEPTH_DEFERRED_POINT;
          if (!result) return;
          const point = scene.addPoint(group.trackId, result.xyz);
          for (let k = 0; k < observations.length; k++)
            if (result.inliers[k])
              scene.addObservation(
                group.cameras[k]!,
                point,
                group.features[k]!,
                observations[k]!.pixel[0],
                observations[k]!.pixel[1],
              );
          points[g] = point;
        });
        return points;
      }
      case 'filter': {
        if (!job.depthOnly) this.evaluate(job.maxErrorPx, true);
        const rejected: number[] = [];
        if (job.depth?.maxRelativeDepthStdDev) {
          const model = scene.snapshot(),
            views = new Map(model.views.map((v) => [v.id, v]));
          for (const p of model.landmarks) {
            const observations = p.observations
                .filter((o) => {
                  const v = views.get(o.viewId)!,
                    q = projectPoint(v.camera, v.pose, p.xyz);
                  return q && Math.hypot(q[0] - o.pixel[0], q[1] - o.pixel[1]) <= job.maxErrorPx;
                })
                .map((o) => ({
                  ...views.get(o.viewId)!,
                  pixel: o.pixel,
                })),
              sensitivity = assessTrackDepth(p.xyz, observations, job.depth.pixelNoiseStdDevPx);
            if (observations.length < 2 && job.depthOnly) {
              scene.removePoint(scene.pointIndex(p.trackId)!);
              continue;
            }
            if (
              observations.length >= 2 &&
              (!sensitivity ||
                sensitivity.maxRelativeDepthStdDev > job.depth.maxRelativeDepthStdDev)
            ) {
              rejected.push(p.trackId);
              scene.removePoint(scene.pointIndex(p.trackId)!);
            }
          }
        }
        for (let p = 0; p < scene.pointCount; p++)
          if (scene.pointAlive(p) && scene.pointObservations[p]! < 2) scene.removePoint(p);
        return {
          ...scene.compact(),
          ...(job.depth?.maxRelativeDepthStdDev ? { depthRejectedTracks: rejected } : {}),
        } satisfies SceneFilterResult;
      }
      case 'pnp': {
        const X = new Float64Array(job.points.length * 3);
        for (let i = 0; i < job.points.length; i++) X.set(scene.point(job.points[i]!), 3 * i);
        return estimatePosePnP(X, job.pixels, job.camera, job.options);
      }
      case 'bundle': {
        const model = scene.snapshot(),
          window = job.variable ? Array.from(job.variable, (c) => scene.cameraIds[c]!) : undefined,
          subset = job.subset && window ? extractLocalBundle(model, window) : undefined;
        const problem = subset?.reconstruction ?? model;
        const plan = planConnectedBundles(problem, {
          iterations: job.iterations,
          anchorViewId: subset?.anchorViewId ?? scene.cameraIds[job.anchor]!,
          scaleViewId: subset?.scaleViewId ?? scene.cameraIds[job.scale]!,
          huberPx: job.huberPx,
          variableViewIds: window,
          refineSharedFocal: job.refineSharedFocal,
          stallTolerance: job.stallTolerance,
        });
        const results = plan.problems.map((part) => {
          const views = new Map(part.reconstruction.views.map((v) => [v.id, v])),
            directionalConstraints = (job.directionalGroups ?? []).flatMap((g) => {
              const observations = Array.from(g.cameras, (c, i) => ({
                  viewId: scene.cameraIds[c]!,
                  featureId: g.features[i]!,
                  pixel: [g.pixels[2 * i]!, g.pixels[2 * i + 1]!] as const,
                })).filter((o) => views.has(o.viewId)),
                direction = fitBundleDirection(
                  observations.map((o) => ({ ...views.get(o.viewId)!, pixel: o.pixel })),
                  maxErrorOfHuberPx(job.huberPx),
                );
              return direction ? [{ trackId: g.trackId, direction, observations }] : [];
            });
          const supported = supportedBundleDirections(
            directionalConstraints,
            part.reconstruction.views,
          );
          return bundleAdjust(
            part.reconstruction,
            supported.length
              ? { ...part.options, directionalConstraints: supported }
              : part.options,
          );
        });
        const solved = mergeBundleRefinements(
          problem,
          results.map((r) => r.reconstruction),
        );
        const refined = subset ? mergeLocalBundle(model, subset, solved) : solved;
        for (const view of refined.views) {
          const index = scene.cameraIndex(view.id)!;
          scene.setPose(index, view.pose);
          scene.setCamera(index, view.camera);
        }
        for (const point of refined.landmarks)
          scene.setPoint(scene.pointIndex(point.trackId)!, point.xyz);
        return {
          diagnostics: results.map((r) => r.diagnostics),
          partition: plan.partition,
          subset: !!subset,
        } satisfies SceneBundleResult;
      }
      case 'measure':
        return this.evaluate(job.maxErrorPx, false);
      case 'snapshot':
        return scene.snapshot();
    }
  }
  /** Mean error of the alive observations within `maxErrorPx`; with `drop`, flag the others dead. */
  private evaluate(maxErrorPx: number, drop: boolean): number {
    const scene = this.scene;
    let total = 0,
      count = 0;
    const cameras = Array.from({ length: scene.cameraCount }, (_, c) => ({
      camera: scene.camera(c),
      pose: scene.pose(c),
    }));
    for (let i = 0; i < scene.observationCount; i++) {
      if (!(scene.obsFlags[i]! & ALIVE)) continue;
      const view = cameras[scene.obsCamera[i]!]!,
        q = projectPoint(view.camera, view.pose, scene.point(scene.obsPoint[i]!));
      const e = q
        ? Math.hypot(
            q[0] - scene.obsPixel[i * PIXEL_STRIDE]!,
            q[1] - scene.obsPixel[i * PIXEL_STRIDE + 1]!,
          )
        : Infinity;
      if (e > maxErrorPx) {
        if (drop) scene.removeObservation(i);
        continue;
      }
      total += e;
      count++;
    }
    return count ? total / count : Infinity;
  }
}
/** Incremental sparse SfM over the scene protocol on the CPU: the same result as `reconstructIncremental`. */
export function reconstructIncrementalScene(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions = {},
): IncrementalResult {
  return withDepthStep(images, options, (controls) => {
    const driver = new HostSceneDriver();
    return drive(sceneSteps(images, inputPairs, controls), (job) => driver.execute(job), 'pass');
  });
}
