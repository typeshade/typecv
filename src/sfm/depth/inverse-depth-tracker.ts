import { DepthTrackHistory } from './depth-tracks.ts';
import {
  fitInverseDepth,
  initializeInverseDepth,
  inverseDepthPoint,
  inverseRangeUncertainty,
  type AnchoredInverseDepth,
} from '../../geometry/inverse-depth.ts';
import { unprojectPixel, cameraProjection } from '../../geometry/camera.ts';
import { inverseDepthJacobian } from '../../geometry/inverse-depth.ts';
import { cameraCenter } from '../../geometry/pose.ts';
import type { Pose } from '../../contracts/camera.ts';
import {
  bundleAdjustInverseDepth,
  type InverseDepthBundleOptions,
  type InverseDepthBundleResult,
  type InverseDepthConstraint,
} from './inverse-depth-bundle.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  RegisteredView,
  SparseReconstruction,
} from '../../contracts/scene.ts';
import type { CameraObservationMask } from '../../contracts/observation-mask.ts';
import { snapshotCameraMasks } from '../tracks/observation-selection.ts';
import { observationKey } from '../../contracts/keys.ts';

export interface InverseDepthTrackingOptions extends InverseDepthBundleOptions {
  readonly maxRelativeStddev?: number;
  readonly credibleSigma?: number;
  readonly minObservations?: number;
  readonly maxErrorPx?: number;
}
export interface DepthTrackState {
  readonly id: number;
  readonly observations: readonly {
    readonly viewId: number;
    readonly featureId: number;
    readonly pixel: readonly [number, number];
  }[];
  readonly status: 'bearing-only' | 'estimating' | 'finite';
  readonly reason:
    | 'translation-unavailable'
    | 'depth-unobservable'
    | 'joint-uncertainty-unavailable'
    | 'insufficient-observations'
    | 'uncertain-depth'
    | 'inconsistent-observations'
    | 'validated';
  readonly referenceViewId: number;
  readonly bearing: Float64Array;
  /** Covariance of pinhole (a,b), NOT inverse depth or world-camera orientation. */
  readonly bearingCovariance?: Float64Array;
  readonly bearingCovarianceConditionalOn?:
    'fixed-intrinsics-anchor-observation' | 'joint-bundle-linearization';
  readonly estimate?: AnchoredInverseDepth;
  /** Covariance of (a,b,inverse Z) in the fixed reference chart, under `uncertainty` below. */
  readonly covariance?: Float64Array;
  readonly inverseRange?: { readonly mean: number; readonly stddev?: number };
  readonly maxErrorPx?: number;
}
export interface InverseDepthSnapshot {
  readonly revision: number;
  readonly tracks: readonly DepthTrackState[];
  readonly uncertainty?: InverseDepthBundleResult['uncertainty'];
  readonly assumptions: {
    readonly pixelNoiseStdDevPx: number;
    readonly huberPx: number;
    readonly maxRelativeStddev: number;
    readonly credibleSigma: number;
    readonly minObservations: number;
  };
  readonly diagnostics: {
    readonly bearingOnly: number;
    readonly estimating: number;
    readonly finite: number;
    readonly uniqueObservations: number;
    readonly backend: string;
    readonly solver?: InverseDepthBundleResult['diagnostics'];
  };
}
export interface DepthTrackingInput {
  readonly tracks: readonly FeatureTrack[];
  readonly images: readonly ImageFeatures[];
  readonly reconstruction?: SparseReconstruction;
  readonly cameraObservationMasks?: readonly CameraObservationMask[];
}
export interface DepthTrackingResult {
  readonly reconstruction?: SparseReconstruction;
  readonly snapshot: InverseDepthSnapshot;
}
/** Structural protocol shared by the separately emitted CPU and GPU declaration trees. */
export interface InverseDepthTracking {
  readonly protocol: 'typecv-inverse-depth-tracker-v1';
  snapshot(): InverseDepthSnapshot | undefined;
  update(input: DepthTrackingInput): DepthTrackingResult;
  updateAsync(
    input: DepthTrackingInput,
    refine: (
      model: SparseReconstruction,
      constraints: readonly InverseDepthConstraint[],
      options: InverseDepthBundleOptions,
    ) => Promise<InverseDepthBundleResult>,
  ): Promise<DepthTrackingResult>;
}

/**
 * Persistent raw-observation estimator. Each update rebuilds its likelihood at current poses/K;
 * previous fitted points/covariances are NEVER added as independent measurements. Opt-in.
 * A reference chart is fixed during BA; its view id does not assert an exact physical center.
 */
export class InverseDepthTracker implements InverseDepthTracking {
  /** Cross-bundle identity: the public CPU and GPU bundles own separate constructor objects. */
  readonly protocol = 'typecv-inverse-depth-tracker-v1';
  private history = new DepthTrackHistory();
  private images = new Map<number, ImageFeatures>();
  private views = new Map<number, RegisteredView>();
  private masks = new Map<number, CameraObservationMask>();
  private current?: InverseDepthSnapshot;
  private busy = false;
  private beforeUpdate?: InverseDepthSnapshot;
  readonly options: Readonly<InverseDepthTrackingOptions>;
  constructor(options: InverseDepthTrackingOptions = {}) {
    this.options = {
      iterations: 4,
      maxRelativeStddev: 0.2,
      credibleSigma: 3,
      minObservations: 3,
      maxErrorPx: 3,
      ...options,
      variableViewIds: options.variableViewIds?.slice(),
      focalGroups: options.focalGroups && new Map(options.focalGroups),
    };
    if (
      !(this.options.maxRelativeStddev! > 0 && Number.isFinite(this.options.maxRelativeStddev)) ||
      !(this.options.credibleSigma! > 0 && Number.isFinite(this.options.credibleSigma)) ||
      !Number.isSafeInteger(this.options.minObservations) ||
      this.options.minObservations! < 2 ||
      !(this.options.maxErrorPx! > 0 && Number.isFinite(this.options.maxErrorPx))
    )
      throw new RangeError('Invalid persistent depth admission controls');
    for (const value of [this.options.pixelNoiseStdDevPx ?? 1, this.options.huberPx ?? 3])
      if (!(value > 0 && Number.isFinite(value)))
        throw new RangeError('Invalid persistent depth noise model');
    if (
      !Number.isSafeInteger(this.options.iterations) ||
      this.options.iterations! < 0 ||
      this.options.iterations! > 100
    )
      throw new RangeError('Invalid persistent depth iterations');
  }
  snapshot(): InverseDepthSnapshot | undefined {
    const snapshot = this.busy ? this.beforeUpdate : this.current;
    return snapshot && structuredClone(snapshot);
  }

  private begin() {
    const old = {
      history: this.history,
      images: this.images,
      views: this.views,
      masks: this.masks,
      current: this.current,
    };
    this.beforeUpdate = this.current;
    this.history = this.history.fork();
    this.images = new Map(this.images);
    this.views = new Map(this.views);
    this.masks = new Map(this.masks);
    return () => {
      this.history = old.history;
      this.images = old.images;
      this.views = old.views;
      this.masks = old.masks;
      this.current = old.current;
    };
  }

  update(input: DepthTrackingInput): DepthTrackingResult {
    if (this.busy) throw new Error('Await the previous depth update');
    this.busy = true;
    const rollback = this.begin();
    try {
      const prepared = this.prepare(input),
        result = prepared.model
          ? bundleAdjustInverseDepth(prepared.model, prepared.constraints, this.options)
          : undefined;
      return this.finish(input, prepared.states, result);
    } catch (error) {
      rollback();
      throw error;
    } finally {
      this.busy = false;
      this.beforeUpdate = undefined;
    }
  }
  async updateAsync(
    input: DepthTrackingInput,
    refine: (
      model: SparseReconstruction,
      constraints: readonly InverseDepthConstraint[],
      options: InverseDepthBundleOptions,
    ) => Promise<InverseDepthBundleResult>,
  ): Promise<DepthTrackingResult> {
    if (this.busy) throw new Error('Await the previous depth update');
    this.busy = true;
    const rollback = this.begin();
    try {
      const prepared = this.prepare(input),
        result = prepared.model
          ? await refine(prepared.model, prepared.constraints, this.options)
          : undefined;
      return this.finish(input, prepared.states, result);
    } catch (error) {
      rollback();
      throw error;
    } finally {
      this.busy = false;
      this.beforeUpdate = undefined;
    }
  }
  private prepare(input: DepthTrackingInput) {
    const reusable = this.rebaseHistory(input.reconstruction);
    const excluded = new Set<string>();
    for (const mask of snapshotCameraMasks(input.cameraObservationMasks) ?? [])
      this.masks.set(mask.viewId, mask);
    for (const mask of this.masks.values())
      mask.values.forEach((v, featureId) => {
        if (!v) excluded.add(observationKey(mask.viewId, featureId));
      });
    const availableImages = new Map(this.images);
    for (const image of input.images) availableImages.set(image.id, image);
    const tracks = this.history.update(input.tracks, [...availableImages.values()], excluded);
    for (const v of input.images)
      this.images.set(v.id, { ...v, camera: { ...v.camera }, keypoints: v.keypoints.slice() });
    for (const v of input.reconstruction?.views ?? [])
      this.views.set(v.id, {
        ...v,
        camera: { ...v.camera },
        pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
      });
    if (this.options.refineSharedFocal) {
      const ratios = new Map<number, number>();
      for (const v of input.reconstruction?.views ?? []) {
        const base = this.images.get(v.id),
          group = this.options.focalGroups?.get(v.id) ?? 0;
        if (base && !ratios.has(group)) ratios.set(group, v.camera.fx / base.camera.fx);
      }
      // A shared log-focal parameter scales fixed per-view base calibrations consistently,
      // including archived views; previous focal estimates are seeds, never extra priors.
      for (const [id, v] of this.views) {
        const base = this.images.get(id),
          ratio = ratios.get(this.options.focalGroups?.get(id) ?? 0);
        if (base && ratio)
          this.views.set(id, {
            ...v,
            camera: { ...base.camera, fx: base.camera.fx * ratio, fy: base.camera.fy * ratio },
          });
      }
    }
    const states: DepthTrackState[] = [],
      constraints: InverseDepthConstraint[] = [],
      active = new Set(this.views.keys());
    // Stable ids inside the solver are independent of the mapper's current track numbering.
    const byTrackId = new Map(tracks.map((t) => [t.id, t]));
    const owner = new Map(
        tracks.flatMap((t) =>
          t.observations.map((o) => [observationKey(o.viewId, o.featureId), t.id] as const),
        ),
      ),
      finite =
        input.reconstruction?.landmarks.flatMap((p) => {
          const id = owner.get(
            observationKey(p.observations[0]!.viewId, p.observations[0]!.featureId),
          );
          const observations =
            byTrackId.get(id!)?.observations.filter((o) => active.has(o.viewId)) ?? [];
          return id !== undefined && observations.length >= 2
            ? [{ ...p, trackId: id, observations }]
            : [];
        }) ?? [],
      finiteIds = new Set(finite.map((p) => p.trackId));
    if (reusable)
      for (const previous of this.current?.tracks ?? []) {
        if (previous.status !== 'finite' || !previous.estimate || finiteIds.has(previous.id))
          continue;
        const raw = byTrackId.get(previous.id),
          observations = raw?.observations.filter((o) => active.has(o.viewId));
        const xyz = inverseDepthPoint(previous.estimate);
        if (xyz && observations && observations.length >= 2) {
          finite.push({ trackId: previous.id, xyz, observations });
          finiteIds.add(previous.id);
        }
      }
    for (const t of tracks) {
      const registered = t.observations.filter(
          (o) => active.has(o.viewId) && this.views.has(o.viewId),
        ),
        first = registered[0] ?? t.observations[0]!,
        image = this.images.get(first.viewId)!,
        camera = this.views.get(first.viewId)?.camera ?? image.camera,
        ray = unprojectPixel(camera, first.pixel);
      if (!ray) continue;
      const bearing = Float64Array.of(ray[0], ray[1], 1),
        length = Math.hypot(...bearing);
      for (let i = 0; i < 3; i++) bearing[i] /= length;
      let state: DepthTrackState = {
        ...t,
        referenceViewId: first.viewId,
        bearing,
        status: 'bearing-only',
        reason: 'translation-unavailable',
      };
      const J = cameraProjection(camera, [ray[0], ray[1], 1])!.jacobian,
        determinant = J[0]! * J[4]! - J[1]! * J[3]!,
        variance = (this.options.pixelNoiseStdDevPx ?? 1) ** 2;
      if (determinant !== 0) {
        const a = J[4]! / determinant,
          b = -J[1]! / determinant,
          c = -J[3]! / determinant,
          d = J[0]! / determinant;
        state = {
          ...state,
          bearingCovariance: Float64Array.of(
            variance * (a * a + b * b),
            variance * (a * c + b * d),
            variance * (a * c + b * d),
            variance * (c * c + d * d),
          ),
          bearingCovarianceConditionalOn: 'fixed-intrinsics-anchor-observation',
        };
      }
      if (registered.length >= 2) {
        const fitted = fitInverseDepth(
            registered.map((o) => ({ ...this.views.get(o.viewId)!, pixel: o.pixel })),
            this.options,
          ),
          estimate = fitted.covariance
            ? fitted.point
            : initializeInverseDepth({ ...this.views.get(first.viewId)!, pixel: first.pixel });
        let estimateError = 0;
        for (const o of registered) {
          const v = this.views.get(o.viewId)!,
            j = inverseDepthJacobian(v.camera, v.pose, estimate);
          estimateError = Math.max(
            estimateError,
            j ? Math.hypot(j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]) : Infinity,
          );
        }
        state = {
          ...state,
          estimate,
          inverseRange: inverseRangeUncertainty(estimate),
          maxErrorPx: estimateError,
          status: fitted.covariance ? 'estimating' : 'bearing-only',
          reason:
            estimateError > this.options.maxErrorPx!
              ? 'inconsistent-observations'
              : fitted.covariance
                ? 'joint-uncertainty-unavailable'
                : 'depth-unobservable',
        };
        if (!finiteIds.has(t.id) && estimateError <= this.options.maxErrorPx!)
          constraints.push({
            trackId: t.id,
            referenceViewId: first.viewId,
            point: estimate,
            directionOnly: !fitted.covariance,
            observations: registered,
          });
      }
      states.push(state);
    }
    const model =
      input.reconstruction && finite.length && active.size >= 2
        ? { views: [...this.views.values()], landmarks: finite }
        : undefined;
    return { model, constraints, states };
  }

  /** Align coordinate gauges through shared registered cameras. This is an initialization
   * change of coordinates, never a repeated camera measurement or a zero-motion constraint.
   * Without a shared observed baseline, old poses are withheld while raw tracks survive.
   */
  private rebaseHistory(model?: SparseReconstruction): boolean {
    if (!model || !this.views.size) return false;
    const common = model.views.filter((v) => this.views.has(v.id)),
      first = common[0];
    if (!first) {
      this.views.clear();
      return false;
    }
    const old = this.views.get(first.id)!,
      oldC = cameraCenter(old.pose),
      newC = cameraCenter(first.pose),
      Q = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++)
          Q[3 * i + j] += first.pose.R[3 * k + i]! * old.pose.R[3 * k + j]!;
    const candidates = common
      .slice(1)
      .map((v) => ({
        oldDistance: Math.hypot(
          ...cameraCenter(this.views.get(v.id)!.pose).map((x, i) => x - oldC[i]!),
        ),
        newDistance: Math.hypot(...cameraCenter(v.pose).map((x, i) => x - newC[i]!)),
      }))
      .filter((c) => c.oldDistance > 1e-10 && c.newDistance > 1e-10)
      .sort((a, b) => b.oldDistance - a.oldDistance);
    if (!candidates.length) {
      this.views.clear();
      return false;
    }
    const scale = candidates[0]!.newDistance / candidates[0]!.oldDistance,
      rotate = (x: ArrayLike<number>) =>
        Float64Array.from(
          { length: 3 },
          (_, i) => Q[3 * i]! * x[0]! + Q[3 * i + 1]! * x[1]! + Q[3 * i + 2]! * x[2]!,
        ),
      qOld = rotate(oldC),
      shift = newC.map((x, i) => x - scale * qOld[i]!);
    const transform = (pose: Pose): Pose => {
      const R = new Float64Array(9),
        C = rotate(cameraCenter(pose)).map((x, i) => scale * x + shift[i]!);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          for (let k = 0; k < 3; k++) R[3 * i + j] += pose.R[3 * i + k]! * Q[3 * j + k]!;
      return {
        R,
        t: Float64Array.from(
          { length: 3 },
          (_, i) => -R[3 * i]! * C[0]! - R[3 * i + 1]! * C[1]! - R[3 * i + 2]! * C[2]!,
        ),
      };
    };
    for (const [id, v] of this.views) this.views.set(id, { ...v, pose: transform(v.pose) });
    if (this.current)
      this.current = {
        ...this.current,
        tracks: this.current.tracks.map((s) => {
          if (!s.estimate) return s;
          const parameters = s.estimate.parameters.slice();
          parameters[2] /= scale;
          const covariance = s.covariance?.map(
            (x, i) => x / ((Math.floor(i / 3) === 2 ? scale : 1) * (i % 3 === 2 ? scale : 1)),
          );
          return {
            ...s,
            estimate: { reference: transform(s.estimate.reference), parameters },
            covariance,
          };
        }),
      };
    return true;
  }
  private finish(
    input: DepthTrackingInput,
    states: DepthTrackState[],
    result?: InverseDepthBundleResult,
  ): DepthTrackingResult {
    const estimates = new Map(result?.estimates.map((p) => [p.trackId, p])),
      views = new Map(
        (result?.reconstruction.views ?? input.reconstruction?.views ?? []).map((v) => [v.id, v]),
      );
    const updated = states.map((s): DepthTrackState => {
      const p = estimates.get(s.id);
      if (!p) return s;
      const inverseRange = inverseRangeUncertainty(p.point, p.covariance),
        xyz = inverseDepthPoint(p.point);
      let error = 0,
        count = 0;
      for (const o of s.observations.filter((o) => views.has(o.viewId))) {
        const v = views.get(o.viewId)!,
          j = inverseDepthJacobian(v.camera, v.pose, p.point);
        error = Math.max(
          error,
          j ? Math.hypot(j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]) : Infinity,
        );
        count++;
      }
      const sigma = inverseRange.stddev,
        relative =
          sigma !== undefined && inverseRange.mean > 0 ? sigma / inverseRange.mean : Infinity,
        reason =
          error > this.options.maxErrorPx!
            ? 'inconsistent-observations'
            : !p.covariance
              ? p.directionOnly || p.informationDeferred
                ? 'depth-unobservable'
                : 'joint-uncertainty-unavailable'
              : count < this.options.minObservations!
                ? 'insufficient-observations'
                : !xyz ||
                    !(inverseRange.mean > this.options.credibleSigma! * sigma!) ||
                    relative > this.options.maxRelativeStddev!
                  ? 'uncertain-depth'
                  : 'validated';
      return {
        ...s,
        referenceViewId: p.referenceViewId ?? s.referenceViewId,
        bearing: Float64Array.of(p.point.parameters[0]!, p.point.parameters[1]!, 1).map(
          (x) => x / Math.hypot(p.point.parameters[0]!, p.point.parameters[1]!, 1),
        ),
        estimate: p.point,
        covariance: p.covariance,
        bearingCovariance: p.bearingCovariance,
        bearingCovarianceConditionalOn: p.bearingCovariance
          ? 'joint-bundle-linearization'
          : undefined,
        inverseRange,
        maxErrorPx: error,
        status: reason === 'validated' ? 'finite' : p.directionOnly ? 'bearing-only' : 'estimating',
        reason,
      };
    });
    for (const v of result?.reconstruction.views ?? []) this.views.set(v.id, v);
    const snapshot: InverseDepthSnapshot = {
      revision: (this.current?.revision ?? 0) + 1,
      tracks: updated,
      ...(result ? { uncertainty: result.uncertainty } : {}),
      assumptions: {
        pixelNoiseStdDevPx: this.options.pixelNoiseStdDevPx ?? 1,
        huberPx: this.options.huberPx ?? 3,
        maxRelativeStddev: this.options.maxRelativeStddev!,
        credibleSigma: this.options.credibleSigma!,
        minObservations: this.options.minObservations!,
      },
      diagnostics: {
        bearingOnly: updated.filter((s) => s.status === 'bearing-only').length,
        estimating: updated.filter((s) => s.status === 'estimating').length,
        finite: updated.filter((s) => s.status === 'finite').length,
        uniqueObservations: updated.reduce((n, s) => n + s.observations.length, 0),
        backend: result?.diagnostics.backend ?? 'cpu-bearing',
        ...(result ? { solver: { ...result.diagnostics } } : {}),
      },
    };
    const landmarks = updated.flatMap((s) => {
      if (s.status !== 'finite' || !s.estimate) return [];
      const observations = s.observations.filter((o) => views.has(o.viewId));
      return [{ trackId: s.id, xyz: inverseDepthPoint(s.estimate)!, observations }];
    });
    this.current = structuredClone(snapshot);
    return {
      snapshot: structuredClone(snapshot),
      ...(input.reconstruction
        ? { reconstruction: structuredClone({ views: [...views.values()], landmarks }) }
        : {}),
    };
  }
}

/** @internal Accept the same state object across core/GPU bundle boundaries. */
export function isInverseDepthTracker(value: unknown): value is InverseDepthTracking {
  const t = value as InverseDepthTracking | undefined;
  return (
    !!t &&
    t.protocol === 'typecv-inverse-depth-tracker-v1' &&
    typeof t.update === 'function' &&
    typeof t.updateAsync === 'function' &&
    typeof t.snapshot === 'function'
  );
}
