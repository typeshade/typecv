import { SparseSymmetric, factorSparseSchur, type SchurFactor } from './sparse-schur.ts';
import { type BundleAdjustmentOptions } from '../bundle/bundle-adjustment.ts';
import { validateReconstruction } from '../bundle/validation.ts';
import type {
  LandmarkObservation,
  SparseReconstruction,
  RegisteredView,
} from '../../reconstruction/types.ts';
import { cameraCenter, updatePose, validatePose } from '../../geometry/pose.ts';
import {
  initializeInverseDepth,
  inverseDepthCovariance,
  inverseDepthJacobian,
  inverseDepthPoint,
  rescaleInverseDepth,
  type AnchoredInverseDepth,
  type InverseDepthLinearization,
} from '../../geometry/inverse-depth.ts';

export interface InverseDepthConstraint {
  readonly trackId: number;
  readonly point: AnchoredInverseDepth;
  readonly referenceViewId?: number;
  readonly observations: readonly LandmarkObservation[];
  /** Rho is unobservable in the supplied observations. Two bearing parameters only. */
  readonly directionOnly?: boolean;
}
export interface InverseDepthBundleOptions extends Pick<
  BundleAdjustmentOptions,
  | 'iterations'
  | 'huberPx'
  | 'anchorViewId'
  | 'scaleViewId'
  | 'variableViewIds'
  | 'refineSharedFocal'
  | 'focalGroups'
> {
  readonly pixelNoiseStdDevPx?: number;
  /** Bound host reduced-system memory/solve size; fail explicitly rather than discard correlations. */
  readonly maxCameraParameters?: number;
  /** Auto uses predicted camera-graph fill; local tracks use sparse selected inversion. */
  readonly solver?: 'auto' | 'dense' | 'sparse';
  readonly maxSchurEntries?: number;
  readonly maxFactorEntries?: number;
  /** Optional GPU measurement batch bound; does not change the objective or uncertainty. */
  readonly measurementBatchSize?: number;
}
export interface InverseDepthMeasurement {
  readonly view: RegisteredView;
  readonly point: AnchoredInverseDepth;
}
export interface InverseDepthMeasurementJob {
  readonly measurements: readonly InverseDepthMeasurement[];
}
export interface InverseDepthBundleResult {
  readonly reconstruction: SparseReconstruction;
  readonly estimates: readonly (InverseDepthConstraint & {
    readonly covariance?: Float64Array;
    readonly bearingCovariance?: Float64Array;
    readonly informationDeferred?: true;
  })[];
  readonly uncertainty: {
    readonly poses: 'joint' | 'fixed-exterior';
    readonly intrinsics: 'joint-focal' | 'fixed';
    readonly anchorViewId: number;
    readonly scaleViewId: number;
    readonly observable: boolean;
    readonly approximation: 'local-robust-Gaussian';
  };
  readonly diagnostics: {
    readonly initialCost: number;
    readonly finalCost: number;
    readonly acceptedSteps: number;
    readonly observations: number;
    readonly cameraParameters: number;
    readonly backend: string;
    readonly measurementBatches?: number;
    readonly measurementBatchSize?: number;
    readonly linearSolver: 'dense' | 'sparse';
    readonly schurEntries: number;
    readonly factorEntries: number;
    readonly cameraCovarianceEntries: number;
    readonly pointCrossEntries: number;
    readonly denseEquivalentCrossEntries: number;
    readonly inputObservations: number;
    readonly deferredTrackIds: readonly number[];
  };
}
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  [0, 1, 2].reduce((s, i) => s + a[i]! * b[i]!, 0);
function positiveInverse(A: Float64Array, n: number): Float64Array | undefined {
  if (!n) return new Float64Array(0);
  const d = Float64Array.from({ length: n }, (_, i) => Math.sqrt(A[i * n + i]!));
  if (d.some((v) => !(v > 0 && Number.isFinite(v)))) return undefined;
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let a = A[i * n + j]! / (d[i]! * d[j]!);
      for (let k = 0; k < j; k++) a -= L[i * n + k]! * L[j * n + k]!;
      if (i === j) {
        if (!(a > 1e-10)) return undefined;
        L[i * n + i] = Math.sqrt(a);
      } else L[i * n + j] = a / L[j * n + j]!;
    }
  const inv = new Float64Array(n * n);
  for (let c = 0; c < n; c++) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = i === c ? 1 : 0;
      for (let j = 0; j < i; j++) v -= L[i * n + j]! * x[j]!;
      x[i] = v / L[i * n + i]!;
    }
    for (let i = n - 1; i >= 0; i--) {
      let v = x[i]!;
      for (let j = i + 1; j < n; j++) v -= L[j * n + i]! * x[j]!;
      x[i] = v / L[i * n + i]!;
    }
    for (let i = 0; i < n; i++) inv[i * n + c] = x[i]! / (d[i]! * d[c]!);
  }
  return inv;
}
const multiplyVector = (A: Float64Array, b: Float64Array, n: number) =>
  Float64Array.from({ length: n }, (_, i) => {
    let v = 0;
    for (let j = 0; j < n; j++) v += A[i * n + j]! * b[j]!;
    return v;
  });

interface CameraBlock {
  offset: number;
  dimension: number;
  basis: Float64Array;
}
function cameraBlocks(
  model: SparseReconstruction,
  active: ReadonlySet<number>,
  scaleId: number | undefined,
  origin: Float64Array,
) {
  const blocks = new Map<number, CameraBlock>();
  let n = 0;
  for (const v of model.views)
    if (active.has(v.id)) {
      const dim = v.id === scaleId ? 5 : 6,
        basis = new Float64Array(6 * dim);
      for (let i = 0; i < 3; i++) basis[i * dim + i] = 1;
      if (dim === 6) for (let i = 3; i < 6; i++) basis[i * dim + i] = 1;
      else {
        const world = cameraCenter(v.pose).map((x, i) => x - origin[i]!),
          length = Math.hypot(...world),
          d = Float64Array.from(
            { length: 3 },
            (_, i) => dot(v.pose.R.subarray(3 * i, 3 * i + 3), world) / length,
          ),
          axis = [0, 1, 2].sort((a, b) => Math.abs(d[a]!) - Math.abs(d[b]!))[0]!,
          u = new Float64Array(3);
        u[axis] = 1;
        const along = d[axis]!;
        for (let i = 0; i < 3; i++) u[i] -= along * d[i]!;
        const norm = Math.hypot(...u);
        for (let i = 0; i < 3; i++) u[i] /= norm;
        const w = [
          d[1]! * u[2]! - d[2]! * u[1]!,
          d[2]! * u[0]! - d[0]! * u[2]!,
          d[0]! * u[1]! - d[1]! * u[0]!,
        ];
        for (let i = 0; i < 3; i++) {
          basis[(i + 3) * dim + 3] = u[i]!;
          basis[(i + 3) * dim + 4] = w[i]!;
        }
      }
      blocks.set(v.id, { offset: n, dimension: dim, basis });
      n += dim;
    }
  return { blocks, n };
}
interface PointBlock {
  V: Float64Array;
  b: Float64Array;
  W: Float64Array;
  d: number;
  inverse?: Float64Array;
  T?: Float64Array;
  indices: number[];
  deferred?: boolean;
}
interface Equations {
  A: SparseSymmetric;
  b: Float64Array;
  points: PointBlock[];
  cost: number;
}

/** Numerical jobs are shared by CPU and the TypeShade driver; only raw measurements enter H. */
export function* inverseDepthBundleSteps(
  input: SparseReconstruction,
  constraints: readonly InverseDepthConstraint[] = [],
  options: InverseDepthBundleOptions = {},
): Generator<
  InverseDepthMeasurementJob,
  InverseDepthBundleResult,
  readonly (InverseDepthLinearization | undefined)[]
> {
  input = structuredClone(input);
  constraints = structuredClone(constraints);
  options = {
    ...options,
    variableViewIds: options.variableViewIds?.slice(),
    focalGroups: options.focalGroups && new Map(options.focalGroups),
  };
  validateReconstruction(input);
  const iterations = options.iterations ?? 8,
    huber = options.huberPx ?? 3,
    sigma = options.pixelNoiseStdDevPx ?? 1,
    anchorId = options.anchorViewId ?? input.views[0]?.id,
    scaleId = options.scaleViewId ?? input.views.find((v) => v.id !== anchorId)?.id,
    limit = options.maxCameraParameters ?? 1536,
    maxSchurEntries = options.maxSchurEntries ?? 300000,
    maxFactorEntries = options.maxFactorEntries ?? 600000;
  if (
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    iterations > 100 ||
    !(huber > 0 && Number.isFinite(huber) && sigma > 0 && Number.isFinite(sigma)) ||
    !Number.isSafeInteger(limit) ||
    limit < 1
  )
    throw new RangeError('Invalid inverse-depth bundle controls');
  if (
    ![maxSchurEntries, maxFactorEntries, options.measurementBatchSize ?? 1].every(
      (v) => Number.isSafeInteger(v) && v > 0,
    ) ||
    !['auto', 'dense', 'sparse'].includes(options.solver ?? 'auto')
  )
    throw new RangeError('Invalid inverse-depth sparse/batch controls');
  const anchor = input.views.find((v) => v.id === anchorId),
    scale = input.views.find((v) => v.id === scaleId);
  if (!anchor || !scale || anchorId === scaleId || !input.landmarks.length)
    throw new RangeError('Inverse-depth BA needs a connected finite map and gauge');
  const origin = cameraCenter(anchor.pose),
    baseline = Math.hypot(...cameraCenter(scale.pose).map((x, i) => x - origin[i]!));
  if (!(baseline > 1e-10))
    throw new RangeError('Inverse-depth BA cannot initialize from pure rotation');
  const connected = new Set<number>([anchorId!]);
  for (let iteration = 0; iteration < input.views.length; iteration++)
    for (const p of input.landmarks)
      if (p.observations.some((o) => connected.has(o.viewId)))
        for (const o of p.observations) connected.add(o.viewId);
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('Inverse-depth BA requires one connected finite observation graph');
  const active = new Set(
    options.variableViewIds ?? input.views.filter((v) => v.id !== anchorId).map((v) => v.id),
  );
  if (active.has(anchorId!) || [...active].some((id) => !input.views.some((v) => v.id === id)))
    throw new RangeError('Invalid inverse-depth variable cameras');
  const fixed = input.views.filter((v) => !active.has(v.id)),
    scaleFree = fixed.length === 1 && active.has(scaleId!),
    viewMap = new Map(input.views.map((v) => [v.id, v])),
    ids = new Set(input.landmarks.map((p) => p.trackId)),
    owned = new Set(
      input.landmarks.flatMap((p) => p.observations.map((o) => `${o.viewId}:${o.featureId}`)),
    );
  const points: InverseDepthConstraint[] = input.landmarks.map((p) => ({
    trackId: p.trackId,
    observations: p.observations,
    referenceViewId: p.observations[0]!.viewId,
    point: initializeInverseDepth(
      { ...viewMap.get(p.observations[0]!.viewId)!, pixel: p.observations[0]!.pixel },
      p.xyz,
    ),
  }));
  for (const p of constraints) {
    validatePose(p.point.reference);
    if (p.directionOnly && p.point.parameters[2] !== 0)
      throw new RangeError('Bearing-only constraints cannot assert finite inverse depth');
    if (
      ids.has(p.trackId) ||
      !Number.isSafeInteger(p.trackId) ||
      p.trackId < 0 ||
      p.observations.length < 2 ||
      p.point.parameters.length !== 3 ||
      !p.point.parameters.every(Number.isFinite)
    )
      throw new RangeError('Invalid inverse-depth constraint');
    const seen = new Set<number>();
    ids.add(p.trackId);
    for (const o of p.observations) {
      const key = `${o.viewId}:${o.featureId}`;
      if (
        !Number.isSafeInteger(o.featureId) ||
        o.featureId < 0 ||
        !viewMap.has(o.viewId) ||
        seen.has(o.viewId) ||
        owned.has(key) ||
        !o.pixel.every(Number.isFinite)
      )
        throw new RangeError('Multiply owned/invalid inverse-depth observation');
      seen.add(o.viewId);
      owned.add(key);
    }
    points.push({
      ...p,
      point: {
        reference: { R: p.point.reference.R.slice(), t: p.point.reference.t.slice() },
        parameters: p.point.parameters.slice(),
      },
    });
  }
  let model: SparseReconstruction = {
      views: input.views.map((v) => ({
        ...v,
        camera: { ...v.camera },
        pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
      })),
      landmarks: input.landmarks,
    },
    currentPoints = points;
  const groups = new Map<number, number>();
  if (options.refineSharedFocal)
    for (const v of model.views) {
      const g = options.focalGroups ? options.focalGroups.get(v.id) : 0;
      if (g === undefined || !Number.isSafeInteger(g) || g < 0)
        throw new RangeError('Invalid inverse-depth focal groups');
      if (!groups.has(g)) groups.set(g, groups.size);
    }
  let lambda = 1e-3,
    acceptedSteps = 0,
    initialCost = 0,
    finalCost = 0;
  const job = (): InverseDepthMeasurementJob => {
    const views = new Map(model.views.map((v) => [v.id, v]));
    return {
      measurements: currentPoints.flatMap((p) =>
        p.observations.map((o) => ({ view: views.get(o.viewId)!, point: p.point })),
      ),
    };
  };
  let last: { eq: Equations; m: number } | undefined;
  const plannedParameters =
    cameraBlocks(model, active, scaleFree ? scaleId : undefined, origin).n + groups.size;
  if (plannedParameters > limit)
    throw new RangeError('Inverse-depth camera parameter budget exceeded');
  const linearSolver = chooseSchurSolver(
    currentPoints,
    active,
    groups,
    options,
    plannedParameters,
    maxFactorEntries,
  );
  if ((linearSolver === 'dense' ? plannedParameters ** 2 : plannedParameters) > maxFactorEntries)
    throw new RangeError('Schur factor budget exceeded before measurement allocation');
  const factorize = (S: SparseSymmetric): SchurFactor | undefined =>
    linearSolver === 'sparse'
      ? factorSparseSchur(S, maxFactorEntries)
      : denseSchurFactor(S, maxFactorEntries);
  for (let iteration = 0; iteration <= iterations; iteration++) {
    const cameras = cameraBlocks(model, active, scaleFree ? scaleId : undefined, origin),
      m = cameras.n + groups.size;
    if (m > limit) throw new RangeError('Inverse-depth camera parameter budget exceeded');
    const measurements = yield job(),
      A = new SparseSymmetric(m, maxSchurEntries, linearSolver === 'dense'),
      bc = new Float64Array(m),
      blocks: PointBlock[] = [];
    // Reuse scalar camera/cross scratch; only the compact point blocks survive this pass.
    const columns = new Int32Array(7),
      values = new Float64Array(7),
      cross = new Float64Array(3 * m),
      touched = new Uint8Array(m);
    let rowIndex = 0,
      cost = 0;
    for (const p of currentPoints) {
      const d = p.directionOnly ? 2 : 3,
        V = new Float64Array(d * d),
        bp = new Float64Array(d),
        firstRow = rowIndex;
      // Test point information before contributing any camera terms. The second pass reuses
      // already computed Jacobians; it neither reevaluates nor adds another measurement.
      for (const o of p.observations) {
        const j = measurements[rowIndex++];
        if (!j) throw new RangeError('Inverse-depth bundle projection is behind a camera');
        const residual = [j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]],
          norm = Math.hypot(...residual),
          weight = (norm <= huber ? 1 : huber / norm) / sigma ** 2;
        for (let row = 0; row < 2; row++)
          for (let k = 0; k < d; k++) {
            bp[k] -= weight * j.point[3 * row + k]! * residual[row]!;
            for (let l = 0; l < d; l++)
              V[k * d + l] += weight * j.point[3 * row + k]! * j.point[3 * row + l]!;
          }
      }
      const observable = d === 3 ? inverseDepthCovariance(V) : positiveInverse(V, d),
        activeIndices: number[] = [];
      if (observable) {
        let at = firstRow;
        for (const o of p.observations) {
          const j = measurements[at++]!,
            residual = [j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]],
            norm = Math.hypot(...residual),
            weight = (norm <= huber ? 1 : huber / norm) / sigma ** 2,
            camera = cameras.blocks.get(o.viewId);
          cost += norm <= huber ? norm * norm : 2 * huber * norm - huber * huber;
          for (let row = 0; row < 2; row++) {
            let count = 0;
            if (camera)
              for (let k = 0; k < camera.dimension; k++) {
                let value = 0;
                for (let l = 0; l < 6; l++)
                  value += j.pose[6 * row + l]! * camera.basis[l * camera.dimension + k]!;
                columns[count] = camera.offset + k;
                values[count++] = value;
              }
            if (groups.size) {
              const g = options.focalGroups?.get(o.viewId) ?? 0;
              columns[count] = cameras.n + groups.get(g)!;
              values[count++] = j.focal[row]!;
            }
            for (let column = 0; column < count; column++) {
              const i = columns[column]!,
                a = values[column]!;
              if (!touched[i]) {
                touched[i] = 1;
                activeIndices.push(i);
              }
              for (let k = 0; k < d; k++) cross[3 * i + k] += weight * a * j.point[3 * row + k]!;
              bc[i] -= weight * a * residual[row]!;
              for (let other = 0; other < count; other++) {
                const k = columns[other]!;
                if (i <= k) A.add(i, k, weight * a * values[other]!);
              }
            }
          }
        }
      }
      const indices = activeIndices
          .filter((i) => {
            for (let k = 0; k < d; k++) if (cross[3 * i + k] !== 0) return true;
            return false;
          })
          .sort((a, b) => a - b),
        W = new Float64Array(indices.length * d);
      for (let row = 0; row < indices.length; row++)
        for (let k = 0; k < d; k++) W[row * d + k] = cross[3 * indices[row]! + k]!;
      for (const i of activeIndices) {
        touched[i] = 0;
        cross.fill(0, 3 * i, 3 * i + 3);
      }
      blocks.push({ V, b: bp, W, d, indices, deferred: !observable });
    }
    const eq = { A, b: bc, points: blocks, cost };
    last = { eq, m };
    if (!iteration) initialCost = cost;
    finalCost = cost;
    if (iteration === iterations) break;
    const reduced = reduce(eq, m, lambda);
    if (!reduced) {
      lambda *= 10;
      continue;
    }
    const factor = factorize(reduced.S);
    if (!factor) {
      lambda *= 10;
      continue;
    }
    const delta = factor.solve(reduced.b),
      nextViews = model.views.map((v) => {
        const c = cameras.blocks.get(v.id),
          step = new Float64Array(6);
        if (c)
          for (let i = 0; i < 6; i++)
            for (let k = 0; k < c.dimension; k++)
              step[i] += c.basis[i * c.dimension + k]! * delta[c.offset + k]!;
        const focal = groups.size
          ? Math.exp(delta[cameras.n + groups.get(options.focalGroups?.get(v.id) ?? 0)!]!)
          : 1;
        return {
          ...v,
          camera: { ...v.camera, fx: v.camera.fx * focal, fy: v.camera.fy * focal },
          pose: c ? updatePose(v.pose, step) : v.pose,
        };
      });
    let nextPoints = currentPoints.map((p, i) => {
      const block = reduced.points[i]!,
        rhs = block.b.slice();
      if (block.deferred) return p;
      for (let row = 0; row < block.indices.length; row++)
        for (let j = 0; j < block.d; j++)
          rhs[j] -= block.W[row * block.d + j]! * delta[block.indices[row]!]!;
      const dp = multiplyVector(block.inverse!, rhs, block.d),
        parameters = p.point.parameters.map((x, k) => x + (k < block.d ? dp[k]! : 0));
      return { ...p, point: { ...p.point, parameters } };
    });
    let candidateViews = nextViews;
    if (scaleFree) {
      const length = Math.hypot(
          ...cameraCenter(nextViews.find((v) => v.id === scaleId)!.pose).map(
            (x, i) => x - origin[i]!,
          ),
        ),
        scale = baseline / length;
      if (!(scale > 0 && Number.isFinite(scale))) {
        lambda *= 10;
        continue;
      }
      candidateViews = nextViews.map((v) => {
        if (v.id === anchorId) return v;
        const ro = [0, 1, 2].map((i) => dot(v.pose.R.subarray(3 * i, 3 * i + 3), origin));
        return {
          ...v,
          pose: { R: v.pose.R, t: v.pose.t.map((t, i) => scale * t + (scale - 1) * ro[i]!) },
        };
      });
      nextPoints = nextPoints.map((p) => ({
        ...p,
        point: rescaleInverseDepth(p.point, scale, undefined, origin).point,
      }));
    }
    const nextMap = new Map(candidateViews.map((v) => [v.id, v]));
    let nextCost = 0;
    for (const [pointIndex, p] of nextPoints.entries())
      for (const o of reduced.points[pointIndex]!.deferred ? [] : p.observations) {
        const v = nextMap.get(o.viewId)!,
          j = inverseDepthJacobian(v.camera, v.pose, p.point);
        if (!j) {
          nextCost = Infinity;
          break;
        }
        const e = Math.hypot(j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]);
        nextCost += e <= huber ? e * e : 2 * huber * e - huber * huber;
      }
    if (nextPoints.slice(0, input.landmarks.length).some((p) => !(p.point.parameters[2]! > 0)))
      nextCost = Infinity;
    if (nextCost < cost) {
      model = { ...model, views: candidateViews };
      currentPoints = nextPoints;
      lambda = Math.max(1e-12, lambda / 3);
      acceptedSteps++;
    } else lambda *= 10;
  }
  const undamped = reduce(last!.eq, last!.m, 0),
    cameraFactor = undamped && factorize(undamped.S),
    estimates = currentPoints.map((p, index) => {
      const block = undamped?.points[index];
      let covariance: Float64Array | undefined, bearingCovariance: Float64Array | undefined;
      if (block && !block.deferred && cameraFactor) {
        const marginal = pointMarginal(block, cameraFactor);
        if (block.d === 3) {
          covariance = marginal;
          bearingCovariance = Float64Array.of(
            marginal[0]!,
            marginal[1]!,
            marginal[3]!,
            marginal[4]!,
          );
        } else bearingCovariance = marginal;
      }
      return {
        ...p,
        ...(covariance ? { covariance } : {}),
        ...(bearingCovariance ? { bearingCovariance } : {}),
        ...(last!.eq.points[index]!.deferred ? { informationDeferred: true as const } : {}),
      };
    });
  const landmarks = input.landmarks.flatMap((p, i) =>
    estimates[i]!.informationDeferred
      ? []
      : [{ ...p, xyz: inverseDepthPoint(estimates[i]!.point)! }],
  );
  if (landmarks.some((p) => !p.xyz))
    throw new RangeError('Joint BA invalidated a finite input point');
  return {
    reconstruction: { views: model.views, landmarks },
    estimates,
    uncertainty: {
      poses: fixed.length === 1 ? 'joint' : 'fixed-exterior',
      intrinsics: groups.size ? 'joint-focal' : 'fixed',
      anchorViewId: anchorId!,
      scaleViewId: scaleId!,
      observable: !!cameraFactor,
      approximation: 'local-robust-Gaussian',
    },
    diagnostics: {
      initialCost,
      finalCost,
      acceptedSteps,
      observations: currentPoints.reduce(
        (n, p, i) => n + (last!.eq.points[i]!.deferred ? 0 : p.observations.length),
        0,
      ),
      cameraParameters: last!.m,
      backend: 'cpu',
      linearSolver,
      schurEntries: undamped?.S.entries ?? 0,
      factorEntries: cameraFactor?.entries ?? 0,
      cameraCovarianceEntries: cameraFactor?.selectedEntries ?? 0,
      pointCrossEntries: last!.eq.points.reduce((n, p) => n + p.W.length, 0),
      denseEquivalentCrossEntries: last!.eq.points.reduce((n, p) => n + last!.m * p.d, 0),
      inputObservations: currentPoints.reduce((n, p) => n + p.observations.length, 0),
      deferredTrackIds: currentPoints
        .filter((_, i) => last!.eq.points[i]!.deferred)
        .map((p) => p.trackId),
    },
  };
}

function reduce(eq: Equations, m: number, lambda: number) {
  const S = eq.A.copy(),
    b = eq.b.slice(),
    points: PointBlock[] = [];
  for (let i = 0; i < m; i++) S.add(i, i, lambda * Math.max(S.get(i, i), 1e-10));
  for (const block of eq.points) {
    if (block.deferred) {
      points.push(block);
      continue;
    }
    const V = block.V.slice();
    for (let i = 0; i < block.d; i++)
      V[i * block.d + i] += lambda * Math.max(V[i * block.d + i]!, 1e-10);
    const inverse = block.d === 3 ? inverseDepthCovariance(V) : positiveInverse(V, block.d);
    if (!inverse) return undefined;
    const T = new Float64Array(block.W.length);
    for (let row = 0; row < block.indices.length; row++)
      for (let j = 0; j < block.d; j++)
        for (let k = 0; k < block.d; k++)
          T[row * block.d + j] += block.W[row * block.d + k]! * inverse[k * block.d + j]!;
    for (let row = 0; row < block.indices.length; row++) {
      const i = block.indices[row]!;
      for (let k = 0; k < block.d; k++) b[i] -= T[row * block.d + k]! * block.b[k]!;
      for (let col = 0; col < block.indices.length; col++)
        if (i <= block.indices[col]!) {
          // Dense reference historically factors the lower triangle; retain its summation order.
          for (let k = 0; k < block.d; k++)
            S.add(i, block.indices[col]!, -T[col * block.d + k]! * block.W[row * block.d + k]!);
        }
    }
    points.push({ ...block, inverse, T });
  }
  return { S, b, points };
}
/** Estimate camera-block fill from observation ownership, including global focal couplings. */
function chooseSchurSolver(
  points: readonly InverseDepthConstraint[],
  active: ReadonlySet<number>,
  groups: ReadonlyMap<number, number>,
  options: InverseDepthBundleOptions,
  parameters: number,
  maxFactorEntries: number,
): 'dense' | 'sparse' {
  if (options.solver && options.solver !== 'auto') return options.solver;
  if (parameters <= 96) return 'dense';
  if (parameters * parameters > maxFactorEntries) return 'sparse';
  const cameras = new Map([...active].map((id, index) => [id, index])),
    n = active.size + groups.size,
    edges = new Set<number>(),
    threshold = (0.35 * n * (n + 1)) / 2;
  for (const p of points) {
    const involved = new Set<number>();
    for (const o of p.observations) {
      const c = cameras.get(o.viewId);
      if (c !== undefined) involved.add(c);
      if (groups.size)
        involved.add(active.size + groups.get(options.focalGroups?.get(o.viewId) ?? 0)!);
    }
    const indices = [...involved];
    for (let a = 0; a < indices.length; a++)
      for (let b = a; b < indices.length; b++)
        edges.add(Math.min(indices[a]!, indices[b]!) * n + Math.max(indices[a]!, indices[b]!));
    if (edges.size >= threshold) return 'dense';
  }
  return 'sparse';
}
function denseSchurFactor(S: SparseSymmetric, maxEntries: number): SchurFactor | undefined {
  if (S.size * S.size > maxEntries) throw new RangeError('Dense Schur factor budget exceeded');
  const inverse = positiveInverse(S.dense(), S.size);
  return (
    inverse && {
      storage: 'dense',
      entries: inverse.length,
      selectedEntries: inverse.length,
      solve: (rhs) => multiplyVector(inverse, rhs, S.size),
      covariance: (i, j) => inverse[i * S.size + j]!,
    }
  );
}
function pointMarginal(block: PointBlock, factor: SchurFactor) {
  const result = block.inverse!.slice(),
    d = block.d;
  if (factor.storage === 'dense') {
    // Each output keeps exactly the original (a,b) accumulation/product order. Cache the
    // shared covariance lookup and first multiply instead of repeating them for every i,j.
    for (let a = 0; a < block.indices.length; a++)
      for (let b = 0; b < block.indices.length; b++) {
        const c = factor.covariance(block.indices[a]!, block.indices[b]!);
        for (let i = 0; i < d; i++) {
          const left = block.T![a * d + i]! * c;
          for (let j = 0; j < d; j++) result[i * d + j] += left * block.T![b * d + j]!;
        }
      }
    return result;
  }
  for (let a = 0; a < block.indices.length; a++)
    for (let b = a; b < block.indices.length; b++) {
      const c = factor.covariance(block.indices[a]!, block.indices[b]!);
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) {
          const value =
            block.T![a * d + i]! * block.T![b * d + j]! +
            (a === b ? 0 : block.T![b * d + i]! * block.T![a * d + j]!);
          result[i * d + j] += c * value;
        }
    }
  return result;
}
export function linearizeInverseDepthJob(job: InverseDepthMeasurementJob) {
  return job.measurements.map((o) => inverseDepthJacobian(o.view.camera, o.view.pose, o.point));
}
export function bundleAdjustInverseDepth(
  input: SparseReconstruction,
  constraints: readonly InverseDepthConstraint[] = [],
  options: InverseDepthBundleOptions = {},
) {
  const steps = inverseDepthBundleSteps(input, constraints, options);
  let next = steps.next();
  while (!next.done) next = steps.next(linearizeInverseDepthJob(next.value));
  return next.value;
}
