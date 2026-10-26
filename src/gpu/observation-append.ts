import { projectPoint } from '../geometry/pose.ts';
import { validateReconstruction } from '../reconstruction/bundle-adjustment.ts';
import { validateFeatures } from '../reconstruction/tracks.ts';
import type { ImageFeatures, SparseReconstruction } from '../reconstruction/types.ts';
import type {
  ExpandedObservationEdge,
  ObservationExpansionResult,
} from './observation-expansion.ts';

export interface ObservationAppendOptions {
  /** Fixed-model reprojection gate in the input images' pixel coordinates; no implicit resize. */
  readonly maxErrorPx: number;
}
export type ObservationAppendRejection =
  | 'unavailableWitness'
  | 'nonIndependentWitness'
  | 'sourceNotOriginal'
  | 'witnessNotOriginalPoint'
  | 'unregisteredTarget'
  | 'targetOwned'
  | 'alreadyObserved'
  | 'targetViewConflict'
  | 'nonpositiveProjection'
  | 'reprojection'
  | 'conflictingTargetView'
  | 'conflictingTargetFeature'
  | 'redundant';
export interface ObservationAppendResult {
  readonly reconstruction: SparseReconstruction;
  /** Indices refer to expansion.newEdges, retaining its complete image-measurement provenance. */
  readonly admitted: readonly {
    readonly edgeIndex: number;
    readonly trackId: number;
    readonly errorPx: number;
  }[];
  readonly diagnostics: {
    readonly candidates: number;
    readonly appended: number;
    readonly rejected: Readonly<Record<ObservationAppendRejection, number>>;
  };
}

const key = (viewId: number, featureId: number) => `${viewId}:${featureId}`;
const pixelValid = (pixel: readonly number[]) =>
  pixel?.length === 2 && Number.isFinite(pixel[0]) && Number.isFinite(pixel[1]);

/**
 * Append measured expansion candidates to existing finite points without solving or filtering
 * the map. Both source and independent witness must belong to that same ORIGINAL point.
 * Newly admitted observations never establish ownership for another admission in this call.
 *
 * Supply the immutable images used to start expansion and that call's returned images/edges.
 * Original feature indices/pixels must remain a prefix. A join uses targetPixel, never a pose
 * projection or the KLT endpoint in place of the retained feature. Malformed lineage throws;
 * well-formed but unqualified candidates are counted and deferred. Competing eligible claims
 * are all deferred; exact duplicate claims append once. No poses, XYZ, track IDs, original
 * observations, pair matches, descriptors or feature coordinates are changed or created.
 *
 * Image corroboration and fixed-model reprojection do not certify correct scene geometry.
 * This opt-in helper does not verify expansion.candidatePairs for use by another mapper.
 */
export function appendExpandedObservations(
  model: SparseReconstruction,
  originalImages: readonly ImageFeatures[],
  expansion: Pick<ObservationExpansionResult, 'images' | 'newEdges'>,
  options: ObservationAppendOptions,
): ObservationAppendResult {
  if (!Number.isFinite(options.maxErrorPx) || !(options.maxErrorPx > 0))
    throw new RangeError('Observation append requires a positive finite pixel threshold');
  validateReconstruction(model);
  const original = validateFeatures(originalImages, []),
    expanded = validateFeatures(expansion.images, []),
    views = new Map(model.views.map((v) => [v.id, v]));
  if (original.size !== expanded.size || model.views.some((v) => !original.has(v.id)))
    throw new RangeError('Observation append requires unchanged image identities');
  for (const image of originalImages) {
    const next = expanded.get(image.id);
    if (
      !next ||
      next.keypoints.length < image.keypoints.length ||
      image.keypoints.some((value, i) => next.keypoints[i] !== value) ||
      (['fx', 'fy', 'cx', 'cy', 'k1', 'k2'] as const).some(
        (k) => (image.camera[k] ?? 0) !== (next.camera[k] ?? 0),
      )
    )
      throw new RangeError('Observation append requires an unchanged original image prefix');
  }
  const owners = new Map<string, number>();
  model.landmarks.forEach((point, pointIndex) => {
    for (const observation of point.observations) {
      const image = original.get(observation.viewId)!;
      if (
        observation.featureId >= image.keypoints.length / 2 ||
        Math.abs(image.keypoints[2 * observation.featureId]! - observation.pixel[0]) > 1e-9 ||
        Math.abs(image.keypoints[2 * observation.featureId + 1]! - observation.pixel[1]) > 1e-9
      )
        throw new RangeError('Observation append requires original feature measurements');
      owners.set(key(observation.viewId, observation.featureId), pointIndex);
    }
  });
  const validateFeature = (
    images: Map<number, ImageFeatures>,
    viewId: number,
    featureId: number,
  ) => {
    const image = images.get(viewId);
    if (
      !Number.isSafeInteger(viewId) ||
      viewId < 0 ||
      !image ||
      !Number.isSafeInteger(featureId) ||
      featureId < 0 ||
      featureId >= image.keypoints.length / 2
    )
      throw new RangeError('Observation append edge has an invalid feature identity');
    return image;
  };
  // Validate the complete batch before making a candidate reconstruction. Status strings
  // describe caller-supplied expansion evidence; structural checks cannot authenticate RGB.
  for (const edge of expansion.newEdges) {
    const target = validateFeature(expanded, edge.viewId, edge.featureId),
      before = original.get(edge.viewId)!,
      createdFeature = edge.featureId >= before.keypoints.length / 2;
    validateFeature(original, edge.sourceViewId, edge.sourceFeatureId);
    if (
      edge.geometry !== 'candidate' ||
      typeof edge.createdFeature !== 'boolean' ||
      edge.createdFeature !== createdFeature ||
      !['none', 'rotation', 'finite', 'surface'].includes(edge.prediction) ||
      (edge.patchWarp !== undefined &&
        (edge.patchWarp.length !== 4 || !edge.patchWarp.every(Number.isFinite))) ||
      (edge.surfaceSupport !== undefined &&
        (!Number.isSafeInteger(edge.surfaceSupport.points) ||
          edge.surfaceSupport.points < 1 ||
          !Number.isFinite(edge.surfaceSupport.maxRelativeDepthResidual) ||
          edge.surfaceSupport.maxRelativeDepthResidual < 0)) ||
      !pixelValid(edge.targetPixel) ||
      !pixelValid(edge.measuredPixel) ||
      edge.targetPixel[0] !== target.keypoints[2 * edge.featureId] ||
      edge.targetPixel[1] !== target.keypoints[2 * edge.featureId + 1] ||
      ![edge.ncc, edge.forwardBackwardErrorPx, edge.uniquenessMargin].every(Number.isFinite) ||
      edge.forwardBackwardErrorPx < 0 ||
      edge.uniquenessMargin < 0
    )
      throw new RangeError('Observation append requires measured expansion edge provenance');
    const witness = edge.corroboration;
    if (witness?.status === 'verified') {
      validateFeature(original, witness.viewId, witness.featureId);
      if (
        !pixelValid(witness.measuredPixel) ||
        ![witness.disagreementPx, witness.ncc, witness.forwardBackwardErrorPx].every(
          Number.isFinite,
        ) ||
        witness.disagreementPx < 0 ||
        witness.forwardBackwardErrorPx < 0
      )
        throw new RangeError('Observation append requires finite witness measurements');
    } else if (witness !== undefined && witness.status !== 'unavailable')
      throw new RangeError('Invalid observation append witness status');
  }
  const rejected: Record<ObservationAppendRejection, number> = {
    unavailableWitness: 0,
    nonIndependentWitness: 0,
    sourceNotOriginal: 0,
    witnessNotOriginalPoint: 0,
    unregisteredTarget: 0,
    targetOwned: 0,
    alreadyObserved: 0,
    targetViewConflict: 0,
    nonpositiveProjection: 0,
    reprojection: 0,
    conflictingTargetView: 0,
    conflictingTargetFeature: 0,
    redundant: 0,
  };
  type Candidate = {
    edge: ExpandedObservationEdge;
    edgeIndex: number;
    pointIndex: number;
    errorPx: number;
  };
  const eligible: Candidate[] = [],
    unique = new Set<string>(),
    pointViews = new Map<string, Set<number>>(),
    features = new Map<string, Set<number>>();
  expansion.newEdges.forEach((edge, edgeIndex) => {
    const witness = edge.corroboration;
    if (witness?.status !== 'verified') {
      rejected.unavailableWitness++;
      return;
    }
    if (new Set([edge.viewId, edge.sourceViewId, witness.viewId]).size !== 3) {
      rejected.nonIndependentWitness++;
      return;
    }
    const pointIndex = owners.get(key(edge.sourceViewId, edge.sourceFeatureId));
    if (pointIndex === undefined) {
      rejected.sourceNotOriginal++;
      return;
    }
    if (owners.get(key(witness.viewId, witness.featureId)) !== pointIndex) {
      rejected.witnessNotOriginalPoint++;
      return;
    }
    const view = views.get(edge.viewId);
    if (!view) {
      rejected.unregisteredTarget++;
      return;
    }
    const owner = owners.get(key(edge.viewId, edge.featureId));
    if (owner !== undefined && owner !== pointIndex) {
      rejected.targetOwned++;
      return;
    }
    const previous = model.landmarks[pointIndex]!.observations.find(
      (o) => o.viewId === edge.viewId,
    );
    if (previous) {
      rejected[previous.featureId === edge.featureId ? 'alreadyObserved' : 'targetViewConflict']++;
      return;
    }
    const projected = projectPoint(view.camera, view.pose, model.landmarks[pointIndex]!.xyz);
    if (!projected) {
      rejected.nonpositiveProjection++;
      return;
    }
    const errorPx = Math.hypot(
      projected[0] - edge.targetPixel[0],
      projected[1] - edge.targetPixel[1],
    );
    if (!(errorPx <= options.maxErrorPx)) {
      rejected.reprojection++;
      return;
    }
    const claim = `${pointIndex}:${edge.viewId}:${edge.featureId}`;
    if (unique.has(claim)) {
      rejected.redundant++;
      return;
    }
    unique.add(claim);
    eligible.push({ edge, edgeIndex, pointIndex, errorPx });
    const pv = key(pointIndex, edge.viewId),
      feature = key(edge.viewId, edge.featureId);
    if (!pointViews.has(pv)) pointViews.set(pv, new Set());
    pointViews.get(pv)!.add(edge.featureId);
    if (!features.has(feature)) features.set(feature, new Set());
    features.get(feature)!.add(pointIndex);
  });
  const reconstruction: SparseReconstruction = {
    views: model.views.map((v) => ({
      ...v,
      camera: { ...v.camera },
      pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
    })),
    landmarks: model.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.slice(),
      observations: p.observations.map((o) => ({ ...o, pixel: [...o.pixel] as const })),
    })),
  };
  const admitted: ObservationAppendResult['admitted'][number][] = [];
  for (const candidate of eligible) {
    const { edge, edgeIndex, pointIndex, errorPx } = candidate;
    if (pointViews.get(key(pointIndex, edge.viewId))!.size > 1) {
      rejected.conflictingTargetView++;
      continue;
    }
    if (features.get(key(edge.viewId, edge.featureId))!.size > 1) {
      rejected.conflictingTargetFeature++;
      continue;
    }
    const point = reconstruction.landmarks[pointIndex]!;
    (point.observations as Array<(typeof point.observations)[number]>).push({
      viewId: edge.viewId,
      featureId: edge.featureId,
      pixel: [...edge.targetPixel] as const,
    });
    admitted.push({ edgeIndex, trackId: point.trackId, errorPx });
  }
  return {
    reconstruction,
    admitted,
    diagnostics: { candidates: expansion.newEdges.length, appended: admitted.length, rejected },
  };
}
