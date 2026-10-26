import { type BundleAdjustmentOptions, type BundleAdjustmentResult } from './bundle-adjustment.ts';
import { validateReconstruction } from '../../geometry/validation.ts';
import { cloneBundleReconstruction, bundleMetrics, bundleStallTolerance } from './bundle-shared.ts';
import { cameraCenter, transformPoint, updatePose } from '../../geometry/pose.ts';
import { inverse3, solveSystem } from '../../geometry/numerics.ts';
import type { SparseReconstruction } from '../../contracts/scene.ts';
import {
  appendBundleDirections,
  separateBundleDirections,
  bundleObservationJacobian,
  normalizeBundlePoint,
} from './bundle-directions.ts';

interface PointNormal {
  rhs: Float64Array;
  cross: Map<number, Float64Array>;
  inverse?: Float64Array;
}

/**
 * The dense index of each view's focal group, from 0, in the order of the group numbers. Without
 * `groups` every view is in group 0. Throws RangeError when a view has no group.
 */
export function focalGroupIndex(
  input: SparseReconstruction,
  groups: ReadonlyMap<number, number> | undefined,
): Map<number, number> {
  if (!groups) return new Map(input.views.map((v) => [v.id, 0]));
  const numbers = input.views.map((v) => groups.get(v.id));
  if (numbers.some((g) => g === undefined || !Number.isSafeInteger(g)))
    throw new RangeError('Every view needs a focal group');
  const rank = new Map(
    [...new Set(numbers as number[])].sort((a, b) => a - b).map((g, i) => [g, i]),
  );
  return new Map(input.views.map((v) => [v.id, rank.get(groups.get(v.id)!)!]));
}

/** CPU reference: one global log-focal coordinate jointly eliminated with pose/point LM. */
export function bundleAdjustJointFocal(
  input: SparseReconstruction,
  options: BundleAdjustmentOptions = {},
): BundleAdjustmentResult {
  validateReconstruction(input);
  const iterations = options.iterations ?? 20,
    huber = options.huberPx ?? 2,
    stall = bundleStallTolerance(options);
  if (
    input.views.length < 2 ||
    !input.landmarks.length ||
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    !Number.isFinite(huber) ||
    huber <= 0
  )
    throw new RangeError('Invalid joint focal bundle scene/options');
  const anchorId = options.anchorViewId ?? input.views[0]!.id,
    scaleId = options.scaleViewId ?? input.views[1]!.id,
    anchor = input.views.find((v) => v.id === anchorId),
    scaleCamera = input.views.find((v) => v.id === scaleId);
  if (!anchor || !scaleCamera || anchorId === scaleId)
    throw new RangeError('Invalid bundle adjustment gauge cameras');
  const origin = cameraCenter(anchor.pose),
    scaleCenter = cameraCenter(scaleCamera.pose),
    baseline = Math.hypot(...origin.map((x, i) => x - scaleCenter[i]!));
  if (!(baseline > 1e-10))
    throw new RangeError('Bundle adjustment gauge requires a nonzero baseline');
  const connected = new Set([anchorId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const point of input.landmarks)
      if (point.observations.some((o) => connected.has(o.viewId)))
        for (const o of point.observations)
          if (!connected.has(o.viewId)) {
            connected.add(o.viewId);
            grew = true;
          }
  }
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('Bundle adjustment requires one connected observation graph');
  const finitePoints = input.landmarks.length;
  input = appendBundleDirections(input, options);
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
    cameraRows = new Map(active.map((v, i) => [v.id, i * 6])),
    focal = active.length * 6,
    // One log-focal coordinate for each camera group. Without groups, one for all views.
    groupOf = focalGroupIndex(input, options.focalGroups),
    groups = Math.max(...groupOf.values()) + 1,
    n = focal + groups;
  // The reference deliberately assembles a dense reduced system, never a dense point system.
  // Keep its allocation arithmetic safe; normal scenes use the matrix-free GPU implementation.
  if (!Number.isSafeInteger(n * n) || n * n > 16 * 1024 ** 2)
    throw new RangeError('Joint focal CPU reduced system is too large');
  let model = cloneBundleReconstruction(input),
    initial = bundleMetrics(model, huber, finitePoints),
    current = initial,
    lambda = 1e-3,
    acceptedSteps = 0,
    rejectedSteps = 0;
  if (!Number.isFinite(initial.cost))
    throw new RangeError(
      'Bundle adjustment observations must initially project in front of their cameras',
    );
  for (let iteration = 0; iteration < iterations; iteration++) {
    const A = new Float64Array(n * n),
      rhs = new Float64Array(n),
      pointNormals: PointNormal[] = [],
      views = new Map(model.views.map((v) => [v.id, v]));
    for (let pointIndex = 0; pointIndex < model.landmarks.length; pointIndex++) {
      const point = model.landmarks[pointIndex]!;
      const V = new Float64Array(9),
        bp = new Float64Array(3),
        cross = new Map<number, Float64Array>();
      for (const observation of point.observations) {
        const view = views.get(observation.viewId)!,
          jac = bundleObservationJacobian(
            view.camera,
            view.pose,
            point.xyz,
            pointIndex >= finitePoints,
          )!;
        const r = [jac.pixel[0] - observation.pixel[0], jac.pixel[1] - observation.pixel[1]],
          norm = Math.hypot(...r),
          w = norm <= huber ? 1 : huber / norm,
          base = cameraRows.get(view.id),
          focalRow = focal + groupOf.get(view.id)!,
          rows =
            base === undefined
              ? [focalRow]
              : [base, base + 1, base + 2, base + 3, base + 4, base + 5, focalRow];
        for (const row of rows) if (!cross.has(row)) cross.set(row, new Float64Array(3));
        for (let axis = 0; axis < 2; axis++) {
          // d(pixel)/d(log scale) includes distortion and the original fx/fy aspect ratio.
          const J = rows.map((row) =>
            row === focalRow
              ? jac.pixel[axis]! - (axis === 0 ? view.camera.cx : view.camera.cy)
              : jac.pose[axis * 6 + row - base!]!,
          );
          for (let i = 0; i < 3; i++) {
            bp[i] -= w * jac.point[axis * 3 + i]! * r[axis]!;
            for (let j = 0; j < 3; j++)
              V[i * 3 + j] += w * jac.point[axis * 3 + i]! * jac.point[axis * 3 + j]!;
          }
          for (let i = 0; i < rows.length; i++) {
            const row = rows[i]!,
              Wi = cross.get(row)!;
            rhs[row] -= w * J[i]! * r[axis]!;
            for (let j = 0; j < rows.length; j++) A[row * n + rows[j]!] += w * J[i]! * J[j]!;
            for (let j = 0; j < 3; j++) Wi[j] += w * J[i]! * jac.point[axis * 3 + j]!;
          }
        }
      }
      for (let i = 0; i < 3; i++) V[i * 3 + i] += lambda * Math.max(V[i * 3 + i]!, 1e-10);
      pointNormals.push({ rhs: bp, cross, inverse: inverse3(V) });
    }
    if (pointNormals.some((p) => !p.inverse)) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    for (let i = 0; i < n; i++) A[i * n + i] += lambda * Math.max(A[i * n + i]!, 1e-10);
    for (const point of pointNormals) {
      const temp = new Map<number, Float64Array>();
      for (const [row, W] of point.cross) {
        const t = new Float64Array(3);
        for (let i = 0; i < 3; i++)
          for (let j = 0; j < 3; j++) t[i] += W[j]! * point.inverse![j * 3 + i]!;
        temp.set(row, t);
        for (let i = 0; i < 3; i++) rhs[row] -= t[i]! * point.rhs[i]!;
      }
      for (const [row, t] of temp)
        for (const [column, W] of point.cross)
          for (let i = 0; i < 3; i++) A[row * n + column] -= t[i]! * W[i]!;
    }
    const step = solveSystem(A, rhs);
    if (!step || !step.every(Number.isFinite)) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    const focalScales = Array.from({ length: groups }, (_, g) => Math.exp(step[focal + g]!));
    if (focalScales.some((scale) => !Number.isFinite(scale) || scale <= 0)) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    const newViews = model.views.map((view) => {
      const row = cameraRows.get(view.id),
        focalScale = focalScales[groupOf.get(view.id)!]!;
      return {
        ...view,
        camera: {
          ...view.camera,
          fx: view.camera.fx * focalScale,
          fy: view.camera.fy * focalScale,
        },
        pose: row === undefined ? view.pose : updatePose(view.pose, step.subarray(row, row + 6)),
      };
    });
    const newPoints = model.landmarks.map((point, i) => {
      const block = pointNormals[i]!,
        b = block.rhs.slice();
      for (const [row, W] of block.cross) for (let j = 0; j < 3; j++) b[j] -= W[j]! * step[row]!;
      const dp = Float64Array.from({ length: 3 }, (_, j) =>
        [0, 1, 2].reduce((v, k) => v + block.inverse![j * 3 + k]! * b[k]!, 0),
      );
      return { ...point, xyz: point.xyz.map((v, j) => v + dp[j]!) };
    });
    const center = cameraCenter(newViews.find((v) => v.id === scaleId)!.pose),
      newBaseline = Math.hypot(...origin.map((x, i) => x - center[i]!));
    if (
      !(newBaseline > 1e-12) ||
      newViews.some(
        (v) =>
          !Number.isFinite(v.camera.fx) ||
          !Number.isFinite(v.camera.fy) ||
          v.camera.fx <= 0 ||
          v.camera.fy <= 0,
      )
    ) {
      lambda *= 10;
      rejectedSteps++;
      continue;
    }
    const s = fixed.length === 1 ? baseline / newBaseline : 1;
    const candidate: SparseReconstruction = {
      views: newViews.map((view) => {
        if (view.id === anchorId) return view;
        const ro = transformPoint({ R: view.pose.R, t: new Float64Array(3) }, origin);
        return {
          ...view,
          pose: { R: view.pose.R, t: view.pose.t.map((t, i) => s * t + (s - 1) * ro[i]!) },
        };
      }),
      landmarks: newPoints.map((point, i) => ({
        ...point,
        xyz: normalizeBundlePoint(point.xyz, i >= finitePoints, origin, s),
      })),
    };
    const next = bundleMetrics(candidate, huber, finitePoints);
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
