import { observationJacobian, projectPoint, type Pose } from '../../geometry/pose.ts';
import type { Camera } from '../../geometry/camera.ts';
import type { LandmarkObservation, SparseReconstruction } from '../../reconstruction/types.ts';
import type { BundleAdjustmentOptions, BundleAdjustmentResult } from './bundle-adjustment.ts';

/**
 * A world-frame bearing hypothesis with unresolved depth. The optimizer uses q = R d;
 * translation has no effect. Compatibility does not prove exact infinity or a static scene.
 * These constraints supplement an already connected finite map, never establish its gauge.
 */
export interface BundleDirection {
  readonly trackId: number;
  readonly direction: Float64Array;
  readonly observations: readonly LandmarkObservation[];
}

/** @internal Append direction vectors only inside the solver's existing point-block storage. */
export function appendBundleDirections(
  input: SparseReconstruction,
  options: BundleAdjustmentOptions,
): SparseReconstruction {
  if (options.directionalConstraints === undefined) return input;
  if (!Array.isArray(options.directionalConstraints))
    throw new RangeError('directionalConstraints must be an array');
  const constraints: readonly BundleDirection[] = options.directionalConstraints;
  const ids = new Set(input.landmarks.map((p) => p.trackId)),
    views = new Set(input.views.map((v) => v.id)),
    features = new Set(
      input.landmarks.flatMap((p) => p.observations.map((o) => `${o.viewId}:${o.featureId}`)),
    );
  const extra = constraints.map((p) => {
    const norm = p.direction instanceof Float64Array ? Math.hypot(...p.direction) : NaN;
    if (
      !Number.isSafeInteger(p.trackId) ||
      p.trackId < 0 ||
      ids.has(p.trackId) ||
      !(p.direction instanceof Float64Array) ||
      p.direction.length !== 3 ||
      !Number.isFinite(norm) ||
      norm <= 1e-12 ||
      !Array.isArray(p.observations) ||
      p.observations.length < 2
    )
      throw new RangeError('Invalid or multiply owned directional constraint');
    ids.add(p.trackId);
    const seen = new Set<number>();
    for (const o of p.observations) {
      const key = `${o.viewId}:${o.featureId}`;
      if (
        !views.has(o.viewId) ||
        seen.has(o.viewId) ||
        !Number.isSafeInteger(o.featureId) ||
        o.featureId < 0 ||
        o.pixel.length !== 2 ||
        !o.pixel.every(Number.isFinite) ||
        features.has(key)
      )
        throw new RangeError('Invalid or multiply owned directional observation');
      seen.add(o.viewId);
      features.add(key);
    }
    return {
      trackId: p.trackId,
      xyz: p.direction.map((x) => x / norm),
      observations: p.observations,
    };
  });
  return extra.length ? { views: input.views, landmarks: [...input.landmarks, ...extra] } : input;
}

/** @internal Directions cross result boundaries separately, never as finite XYZ landmarks. */
export function separateBundleDirections<T extends BundleAdjustmentResult>(
  result: T,
  finitePoints: number,
): T {
  if (result.reconstruction.landmarks.length === finitePoints) return result;
  const directions = result.reconstruction.landmarks.slice(finitePoints);
  return {
    ...result,
    reconstruction: {
      views: result.reconstruction.views,
      landmarks: result.reconstruction.landmarks.slice(0, finitePoints),
    },
    directionalConstraints: directions.map((p) => ({
      trackId: p.trackId,
      direction: p.xyz,
      observations: p.observations,
    })),
    diagnostics: {
      ...result.diagnostics,
      directionalObservations: directions.reduce((n, p) => n + p.observations.length, 0),
      pointParameters: 3 * finitePoints + 2 * directions.length,
    },
  };
}

function directionPose(pose: Pose, directional: boolean): Pose {
  return directional ? { R: pose.R, t: new Float64Array(3) } : pose;
}

/** @internal Reuse the existing camera model and pose Jacobian, with exactly zero d(pixel)/dt. */
export function bundleObservationJacobian(
  camera: Camera,
  pose: Pose,
  point: Float64Array,
  directional: boolean,
) {
  const jac = observationJacobian(camera, directionPose(pose, directional), point);
  if (jac && directional) for (const at of [3, 4, 5, 9, 10, 11]) jac.pose[at] = 0;
  return jac;
}

/** @internal Evaluate directions without introducing a large finite surrogate or cancellation. */
export function projectBundlePoint(
  camera: Camera,
  pose: Pose,
  point: Float64Array,
  directional: boolean,
) {
  return projectPoint(camera, directionPose(pose, directional), point);
}

/** @internal Direction vectors have their own radial gauge; world similarity never moves them. */
export function normalizeBundlePoint(
  xyz: Float64Array,
  directional: boolean,
  origin: Float64Array,
  scale: number,
): Float64Array {
  if (directional) {
    const norm = Math.hypot(...xyz);
    return xyz.map((x) => x / norm);
  }
  return xyz.map((x, i) => origin[i]! + scale * (x - origin[i]!));
}
