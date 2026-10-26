import { bundleAdjustJointFocal } from './joint-focal.ts';
import { validateCamera } from '../../geometry/camera.ts';
import {
  cameraCenter,
  clonePose,
  transformPoint,
  updatePose,
  validatePose,
} from '../../geometry/pose.ts';
import { inverse3, solveSystem } from '../../geometry/numerics.ts';
import type { Landmark, RegisteredView, SparseReconstruction } from '../../reconstruction/types.ts';
import {
  appendBundleDirections,
  separateBundleDirections,
  bundleObservationJacobian,
  projectBundlePoint,
  normalizeBundlePoint,
  type BundleDirection,
} from './bundle-directions.ts';

export interface BundleAdjustmentOptions {
  /** Optional unresolved world-bearing constraints. They cannot connect or initialize a finite map. */
  readonly directionalConstraints?: readonly BundleDirection[];
  readonly iterations?: number;
  readonly huberPx?: number;
  readonly anchorViewId?: number;
  readonly scaleViewId?: number;
  /** Optional local camera window. Other poses stay fixed; points still use all their observations. */
  readonly variableViewIds?: readonly number[];
  /** Jointly optimize one positive focal scale shared by all views with poses and points. */
  readonly refineSharedFocal?: boolean;
  /**
   * With `refineSharedFocal`: the camera group of each view id. The BA refines one focal scale
   * for each group. Every view needs a group. Without it, all views share one scale.
   */
  readonly focalGroups?: ReadonlyMap<number, number>;
  /**
   * End the LM loop when an accepted step lowers the robust cost by less than this fraction of
   * the cost (at least 1). The default, 1e-9, ends it only at convergence.
   */
  readonly stallTolerance?: number;
}

/** @internal The LM stop rule shared by the CPU and GPU solvers. */
export function bundleStallTolerance(options: BundleAdjustmentOptions): number {
  const tolerance = options.stallTolerance ?? 1e-9;
  if (!Number.isFinite(tolerance) || tolerance < 0 || tolerance >= 1)
    throw new RangeError('stallTolerance must be in [0, 1)');
  return tolerance;
}
export interface BundleAdjustmentResult {
  readonly reconstruction: SparseReconstruction;
  /** Optimized unresolved bearings, kept separate from finite XYZ. Present only when supplied. */
  readonly directionalConstraints?: readonly BundleDirection[];
  readonly diagnostics: {
    readonly initialCost: number;
    readonly finalCost: number;
    readonly initialRmsePx: number;
    readonly finalRmsePx: number;
    readonly acceptedSteps: number;
    readonly rejectedSteps: number;
    readonly observations: number;
    readonly baseline: number;
    /** Observations in the bearing-only factors, included in the aggregate cost/RMS above. */
    readonly directionalObservations?: number;
    /** Point-block degrees of freedom (3 per finite point, 2 per bearing), when directions participate. */
    readonly pointParameters?: number;
  };
}
export function validateReconstruction(input: SparseReconstruction): void {
  const views = new Map<number, RegisteredView>();
  for (const v of input.views) {
    if (!Number.isSafeInteger(v.id) || v.id < 0 || views.has(v.id))
      throw new RangeError('Invalid/duplicate registered view id');
    validateCamera(v.camera);
    validatePose(v.pose);
    views.set(v.id, v);
  }
  const points = new Set<number>(),
    ownedFeatures = new Set<string>();
  for (const p of input.landmarks) {
    if (
      !Number.isSafeInteger(p.trackId) ||
      p.trackId < 0 ||
      points.has(p.trackId) ||
      p.xyz.length !== 3 ||
      !p.xyz.every(Number.isFinite)
    )
      throw new RangeError('Invalid/duplicate landmark');
    points.add(p.trackId);
    const seen = new Set<number>();
    if (p.observations.length < 2)
      throw new RangeError('Landmarks require at least two observations');
    for (const o of p.observations) {
      const key = `${o.viewId}:${o.featureId}`;
      if (
        !views.has(o.viewId) ||
        seen.has(o.viewId) ||
        !Number.isSafeInteger(o.featureId) ||
        o.featureId < 0 ||
        o.pixel.length !== 2 ||
        !o.pixel.every(Number.isFinite) ||
        ownedFeatures.has(key)
      )
        throw new RangeError('Invalid or multiply owned landmark observation');
      seen.add(o.viewId);
      ownedFeatures.add(key);
    }
  }
}
function clone(input: SparseReconstruction): SparseReconstruction {
  return {
    views: input.views.map((v) => ({ ...v, camera: { ...v.camera }, pose: clonePose(v.pose) })),
    landmarks: input.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.slice(),
      observations: p.observations.map((o) => ({ ...o, pixel: [o.pixel[0], o.pixel[1]] as const })),
    })),
  };
}
function metrics(
  input: SparseReconstruction,
  huber: number,
  finitePoints = input.landmarks.length,
): { cost: number; rmse: number; observations: number } {
  const views = new Map(input.views.map((v) => [v.id, v]));
  let cost = 0,
    squares = 0,
    observations = 0;
  for (let i = 0; i < input.landmarks.length; i++) {
    const p = input.landmarks[i]!;
    for (const o of p.observations) {
      const v = views.get(o.viewId)!,
        pixel = projectBundlePoint(v.camera, v.pose, p.xyz, i >= finitePoints);
      if (!pixel) return { cost: Infinity, rmse: Infinity, observations };
      const e = Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]);
      cost += e <= huber ? e * e : 2 * huber * e - huber * huber;
      squares += e * e;
      observations++;
    }
  }
  return { cost, rmse: Math.sqrt(squares / observations), observations };
}
interface PointBlock {
  V: Float64Array;
  b: Float64Array;
  W: Map<number, Float64Array>;
  inverse?: Float64Array;
}

/** Joint pose/point LM with point-block Schur elimination and a dense reduced camera solve.
 * Fixed camera intrinsics. First pose and the initial inter-camera baseline define the gauge.
 * Input data and the fixed observation set are preserved. Intended for small/medium scenes.
 */
function bundleAdjustFixedIntrinsics(
  input: SparseReconstruction,
  options: BundleAdjustmentOptions = {},
): BundleAdjustmentResult {
  validateReconstruction(input);
  const iterations = options.iterations ?? 20,
    huber = options.huberPx ?? 2,
    stall = bundleStallTolerance(options);
  if (input.views.length < 2 || !input.landmarks.length)
    throw new RangeError('Bundle adjustment needs two views and landmarks');
  if (!Number.isSafeInteger(iterations) || iterations < 0 || !Number.isFinite(huber) || huber <= 0)
    throw new RangeError('Invalid bundle adjustment options');
  const anchorId = options.anchorViewId ?? input.views[0]!.id,
    scaleId = options.scaleViewId ?? input.views[1]!.id;
  const anchor = input.views.find((v) => v.id === anchorId),
    scaleCamera = input.views.find((v) => v.id === scaleId);
  if (!anchor || !scaleCamera || anchorId === scaleId)
    throw new RangeError('Invalid bundle adjustment gauge cameras');
  const origin = cameraCenter(anchor.pose),
    scaleCenter = cameraCenter(scaleCamera.pose),
    baseline = Math.hypot(...origin.map((x, i) => x - scaleCenter[i]!));
  if (!(baseline > 1e-10))
    throw new RangeError('Bundle adjustment gauge requires a nonzero baseline');
  const connected = new Set<number>([anchorId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const p of input.landmarks)
      if (p.observations.some((o) => connected.has(o.viewId)))
        for (const o of p.observations)
          if (!connected.has(o.viewId)) {
            connected.add(o.viewId);
            grew = true;
          }
  }
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('Bundle adjustment requires one connected observation graph');
  const finitePoints = input.landmarks.length;
  input = appendBundleDirections(input, options);
  let model = clone(input),
    initial = metrics(model, huber, finitePoints),
    current = initial,
    lambda = 1e-3,
    acceptedSteps = 0,
    rejectedSteps = 0;
  if (!Number.isFinite(initial.cost))
    throw new RangeError(
      'Bundle adjustment observations must initially project in front of their cameras',
    );
  const variable =
    options.variableViewIds === undefined ? undefined : new Set(options.variableViewIds);
  if (
    variable &&
    (variable.size !== options.variableViewIds!.length ||
      [...variable].some((id) => id === anchorId || !input.views.some((v) => v.id === id)))
  )
    throw new RangeError('Invalid variable camera window');
  const fixed = input.views.filter((v) => v.id === anchorId || (variable && !variable.has(v.id)));
  if (fixed.length > 1 && variable?.has(scaleId))
    throw new RangeError('Local BA scale camera must be fixed');
  const active = input.views.filter((v) => v.id !== anchorId && (!variable || variable.has(v.id))),
    indices = new Map(active.map((v, i) => [v.id, i])),
    m = active.length * 6;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const A = new Float64Array(m * m),
      b = new Float64Array(m),
      blocks: PointBlock[] = [],
      views = new Map(model.views.map((v) => [v.id, v]));
    for (let pointIndex = 0; pointIndex < model.landmarks.length; pointIndex++) {
      const point = model.landmarks[pointIndex]!;
      const V = new Float64Array(9),
        bp = new Float64Array(3),
        W = new Map<number, Float64Array>();
      for (const o of point.observations) {
        const view = views.get(o.viewId)!,
          jac = bundleObservationJacobian(
            view.camera,
            view.pose,
            point.xyz,
            pointIndex >= finitePoints,
          )!;
        const r = [jac.pixel[0] - o.pixel[0], jac.pixel[1] - o.pixel[1]],
          norm = Math.hypot(...r),
          weight = norm <= huber ? 1 : huber / norm,
          ci = indices.get(o.viewId);
        let cross: Float64Array | undefined;
        if (ci !== undefined) {
          cross = new Float64Array(18);
          W.set(ci, cross);
        }
        for (let row = 0; row < 2; row++) {
          for (let k = 0; k < 3; k++) {
            bp[k] -= weight * jac.point[row * 3 + k]! * r[row]!;
            for (let l = 0; l < 3; l++)
              V[k * 3 + l] += weight * jac.point[row * 3 + k]! * jac.point[row * 3 + l]!;
          }
          if (ci !== undefined)
            for (let k = 0; k < 6; k++) {
              const i = ci * 6 + k;
              b[i] -= weight * jac.pose[row * 6 + k]! * r[row]!;
              for (let l = 0; l < 6; l++)
                A[i * m + ci * 6 + l] += weight * jac.pose[row * 6 + k]! * jac.pose[row * 6 + l]!;
              for (let l = 0; l < 3; l++)
                cross![k * 3 + l] += weight * jac.pose[row * 6 + k]! * jac.point[row * 3 + l]!;
            }
        }
      }
      for (let k = 0; k < 3; k++) V[k * 3 + k] += lambda * Math.max(V[k * 3 + k]!, 1e-10);
      blocks.push({ V, b: bp, W, inverse: inverse3(V) });
    }
    if (blocks.some((p) => !p.inverse)) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    for (let k = 0; k < m; k++) A[k * m + k] += lambda * Math.max(A[k * m + k]!, 1e-10);
    // S = A - W V^-1 W^T; reduced RHS = b_camera - W V^-1 b_point.
    for (const block of blocks) {
      const inv = block.inverse!,
        temp = new Map<number, Float64Array>();
      for (const [ci, W] of block.W) {
        const T = new Float64Array(18);
        for (let i = 0; i < 6; i++)
          for (let j = 0; j < 3; j++)
            for (let k = 0; k < 3; k++) T[i * 3 + j] += W[i * 3 + k]! * inv[k * 3 + j]!;
        temp.set(ci, T);
        for (let i = 0; i < 6; i++)
          for (let k = 0; k < 3; k++) b[ci * 6 + i] -= T[i * 3 + k]! * block.b[k]!;
      }
      for (const [ci, T] of temp)
        for (const [cj, W] of block.W)
          for (let i = 0; i < 6; i++)
            for (let j = 0; j < 6; j++) {
              let value = 0;
              for (let k = 0; k < 3; k++) value += T[i * 3 + k]! * W[j * 3 + k]!;
              A[(ci * 6 + i) * m + cj * 6 + j] -= value;
            }
    }
    const delta = solveSystem(A, b);
    if (!delta) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    const newViews: RegisteredView[] = model.views.map((v) => {
      const i = indices.get(v.id);
      return i === undefined
        ? v
        : { ...v, pose: updatePose(v.pose, delta.subarray(i * 6, i * 6 + 6)) };
    });
    const newPoints: Landmark[] = model.landmarks.map((p, i) => {
      const block = blocks[i]!,
        rhs = block.b.slice();
      for (const [ci, W] of block.W)
        for (let j = 0; j < 3; j++)
          for (let k = 0; k < 6; k++) rhs[j] -= W[k * 3 + j]! * delta[ci * 6 + k]!;
      const dp = Float64Array.from({ length: 3 }, (_, j) =>
        [0, 1, 2].reduce((s, k) => s + block.inverse![j * 3 + k]! * rhs[k]!, 0),
      );
      return { ...p, xyz: p.xyz.map((v, j) => v + dp[j]!) };
    });
    const candidateScale = cameraCenter(newViews.find((v) => v.id === scaleId)!.pose),
      newBaseline = Math.hypot(...origin.map((x, i) => x - candidateScale[i]!));
    if (!(newBaseline > 1e-12)) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    const s = fixed.length === 1 ? baseline / newBaseline : 1;
    // Similarity normalization fixes the scale null direction without changing any projection.
    const candidate: SparseReconstruction = {
      views: newViews.map((v) => {
        if (v.id === anchorId) return v;
        const ro = transformPoint({ R: v.pose.R, t: new Float64Array(3) }, origin);
        return { ...v, pose: { R: v.pose.R, t: v.pose.t.map((t, i) => s * t + (s - 1) * ro[i]!) } };
      }),
      landmarks: newPoints.map((p, i) => ({
        ...p,
        xyz: normalizeBundlePoint(p.xyz, i >= finitePoints, origin, s),
      })),
    };
    const next = metrics(candidate, huber, finitePoints);
    if (next.cost < current.cost) {
      const improvement = current.cost - next.cost;
      model = candidate;
      current = next;
      lambda = Math.max(1e-10, lambda / 3);
      acceptedSteps++;
      if (improvement < stall * Math.max(1, current.cost)) break;
    } else {
      lambda *= 10;
      rejectedSteps++;
      if (lambda > 1e12) break;
    }
  }
  return separateBundleDirections(
    {
      reconstruction: model,
      diagnostics: {
        initialCost: initial.cost,
        finalCost: current.cost,
        initialRmsePx: initial.rmse,
        finalRmsePx: current.rmse,
        acceptedSteps,
        rejectedSteps,
        observations: initial.observations,
        baseline,
      },
    },
    finitePoints,
  );
}

/** Optional shared focal is jointly optimized with poses and points in the same LM step. */
export function bundleAdjust(
  input: SparseReconstruction,
  options: BundleAdjustmentOptions = {},
): BundleAdjustmentResult {
  return options.refineSharedFocal
    ? bundleAdjustJointFocal(input, options)
    : bundleAdjustFixedIntrinsics(input, options);
}

/** @internal Shared objective and snapshot helpers for the joint focal reference. */
export { clone as cloneBundleReconstruction, metrics as bundleMetrics };
