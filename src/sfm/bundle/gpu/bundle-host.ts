import { validateReconstruction } from '../../../geometry/validation.ts';
import type { SparseReconstruction } from '../../../contracts/scene.ts';
import { appendBundleDirections } from '../bundle-directions.ts';
import type { ShadeDevice } from '../../../device/gpu/device.ts';
import type { BundleResidency } from '../../../device/gpu/budget-plan.ts';
import { validateGpuCamera } from '../../../geometry/gpu/bounds.ts';
import { cameraRecord, poseRecord } from '../../../contracts/camera-record.ts';
import { BUNDLE_DEFAULTS } from '../../config/defaults.ts';
import { MODEL_STRIDE, PIXEL_STRIDE, POINT_STRIDE, POSE_STRIDE } from '../../scene/scene.ts';
import { BUNDLE_STATE, type GpuBundleAdjustmentOptions } from './bundle-shared.ts';

// The host half that the two GPU bundle adjustment routes share verbatim: `bundleAdjustGpuFixed`
// (`bundle.ts`) and `bundleAdjustJointFocalGpu` (`joint-focal.ts`) (docs/library-structure.md,
// section 3.3, `sfm/bundle`; phase 6). What differs between the routes stays in each route.

/**
 * @internal The words of the `blocks` array per point and per camera, on both routes: 13 per
 * point, and 36 + 36 + 6 per camera (the block, its inverse and the right-hand side). Per
 * observation the fixed route holds 42 words and the shared-focal route 45. The kernels read the
 * same layout (`pointBlock`, `cameraBlock`, `inverseBlock` and `cameraRhs` in `bundle.shade.ts`
 * and `joint-focal.shade.ts`), and they keep their own numbers until R03 (section 4, rule 6).
 */
export const BUNDLE_BLOCK_WORDS = { point: 13, camera: 78 } as const;

/**
 * @internal The checked problem of a GPU bundle adjustment: the scene with its bearing
 * directions, its sizes, the solver options with their defaults, the gauge and the variable
 * cameras. A bad problem throws before any GPU work. The default of `maxCgIterations` is the
 * route's own (`defaultMaxCg`).
 */
export function bundleProblem(
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions,
  defaultMaxCg: (cameras: number) => number,
) {
  validateReconstruction(input);
  const finiteLandmarks = input.landmarks,
    finitePoints = finiteLandmarks.length;
  input = appendBundleDirections(input, options);
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((n, p) => n + p.observations.length, 0);
  if (![C, P, M].every((n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff))
    throw new RangeError('GPU BA scene dimensions exceed integer index bounds');
  const iterations = options.iterations ?? BUNDLE_DEFAULTS.iterations,
    huber = options.huberPx ?? BUNDLE_DEFAULTS.huberPx,
    maxCg = options.maxCgIterations ?? defaultMaxCg(C),
    tolerance = options.cgRelativeTolerance ?? BUNDLE_DEFAULTS.cgRelativeTolerance;
  if (
    C < 2 ||
    finitePoints < 1 ||
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    iterations > 100 ||
    !Number.isFinite(huber) ||
    huber < 1e-12 ||
    huber > 1e6 ||
    !Number.isSafeInteger(maxCg) ||
    maxCg < 1 ||
    maxCg > 256 ||
    !Number.isFinite(tolerance) ||
    tolerance < 1e-12 ||
    tolerance > 1e-2
  )
    throw new RangeError('Invalid GPU BA scene/options');
  const byId = new Map(input.views.map((v, i) => [v.id, i])),
    anchorId = options.anchorViewId ?? input.views[0]!.id,
    scaleId = options.scaleViewId ?? input.views[1]!.id;
  if (!byId.has(anchorId) || !byId.has(scaleId) || anchorId === scaleId)
    throw new RangeError('Invalid GPU BA gauge cameras');
  const variable =
    options.variableViewIds === undefined ? undefined : new Set(options.variableViewIds);
  if (
    variable &&
    (variable.size !== options.variableViewIds!.length ||
      [...variable].some((id) => id === anchorId || !byId.has(id)))
  )
    throw new RangeError('Invalid GPU BA variable camera window');
  const active = input.views.map((v) => v.id !== anchorId && (!variable || variable.has(v.id))),
    fixed = active.filter((x) => !x).length;
  if (fixed > 1 && variable?.has(scaleId))
    throw new RangeError('Local GPU BA scale camera must be fixed');
  const connected = new Set([anchorId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const p of finiteLandmarks)
      if (p.observations.some((o) => connected.has(o.viewId)))
        for (const o of p.observations)
          if (!connected.has(o.viewId)) {
            connected.add(o.viewId);
            grew = true;
          }
  }
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('GPU BA requires one connected observation graph');
  return {
    input,
    finitePoints,
    C,
    P,
    M,
    iterations,
    huber,
    maxCg,
    tolerance,
    byId,
    anchorId,
    scaleId,
    active,
    fixed,
  };
}

/**
 * @internal The sections of the index array, in words: the camera and the point of each
 * observation, the first observation of each point, the first entry of each camera's list, the
 * observations of each camera, and the flag of each variable camera. The kernels read the same
 * layout (`pointStart`, `cameraStart`, `cameraObservation` and `variable`).
 */
export function bundleIndexLayout(C: number, P: number, M: number) {
  return {
    observationCamera: 0,
    observationPoint: M,
    pointStart: 2 * M,
    cameraStart: 2 * M + P + 1,
    cameraObservations: 2 * M + P + C + 2,
    cameraActive: 3 * M + P + C + 2,
    size: 3 * M + P + 2 * C + 2,
  } as const;
}

/** @internal The lengths of the arrays that both routes size alike, in elements. */
export function bundleArraySizes(C: number, P: number, M: number) {
  return {
    sceneSize: (POSE_STRIDE + MODEL_STRIDE) * C + POINT_STRIDE * P + PIXEL_STRIDE * M,
    trialSize: POSE_STRIDE * C + POINT_STRIDE * P,
    indexSize: bundleIndexLayout(C, P, M).size,
    cameraScratchSize: 36 * C,
    stateSize: BUNDLE_STATE.size,
    partialsSize: 3 * Math.max(1, Math.ceil(M / 64)),
  };
}

/** @internal The first observation of each point (`P + 1` offsets) and the longest track. */
export function bundlePointStarts(input: SparseReconstruction, P: number) {
  const pointStarts = new Uint32Array(P + 1);
  let longestTrack = 1;
  for (let p = 0; p < P; p++) {
    const n = input.landmarks[p]!.observations.length;
    pointStarts[p + 1] = pointStarts[p]! + n;
    longestTrack = Math.max(longestTrack, n);
  }
  return { pointStarts, longestTrack };
}

/** @internal A copy of the scene's metadata, apart from the mutable numerical arrays. */
export function snapshotBundleModel(input: SparseReconstruction): SparseReconstruction {
  return {
    views: input.views.map((v) => ({
      ...v,
      camera: { ...v.camera },
      pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
    })),
    landmarks: input.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.slice(),
      observations: p.observations.map((o) => ({ ...o, pixel: [...o.pixel] as const })),
    })),
  };
}

/**
 * @internal The scene array (poses, points, camera models, pixels), the index array of
 * `bundleIndexLayout` and the observations of each camera. Each camera is validated here.
 */
export function packBundleScene(
  model: SparseReconstruction,
  byId: ReadonlyMap<number, number>,
  active: readonly boolean[],
  sizes: { C: number; P: number; M: number; sceneSize: number; indexSize: number },
) {
  const { C, P, M, sceneSize, indexSize } = sizes,
    layout = bundleIndexLayout(C, P, M);
  const source = new Float64Array(sceneSize),
    indices = new Uint32Array(indexSize),
    cameraLists: number[][] = Array.from({ length: C }, () => []);
  for (let c = 0; c < C; c++) {
    const v = model.views[c]!,
      k = validateGpuCamera(v.camera);
    source.set(poseRecord(v.pose), c * POSE_STRIDE);
    source.set(cameraRecord(k), POSE_STRIDE * C + POINT_STRIDE * P + c * MODEL_STRIDE);
    indices[layout.cameraActive + c] = Number(active[c]);
  }
  let obs = 0;
  for (let p = 0; p < P; p++) {
    const point = model.landmarks[p]!;
    source.set(point.xyz, POSE_STRIDE * C + p * POINT_STRIDE);
    indices[layout.pointStart + p] = obs;
    for (const o of point.observations) {
      const c = byId.get(o.viewId)!;
      indices[layout.observationCamera + obs] = c;
      indices[layout.observationPoint + obs] = p;
      source.set(o.pixel, (POSE_STRIDE + MODEL_STRIDE) * C + POINT_STRIDE * P + obs * PIXEL_STRIDE);
      cameraLists[c]!.push(obs);
      obs++;
    }
  }
  indices[layout.pointStart + P] = obs;
  let cursor = 0;
  for (let c = 0; c < C; c++) {
    indices[layout.cameraStart + c] = cursor;
    indices.set(cameraLists[c]!, layout.cameraObservations + cursor);
    cursor += cameraLists[c]!.length;
  }
  indices[layout.cameraStart + C] = cursor;
  return { source, indices, cameraLists };
}

/**
 * @internal The controls of the windows of a residency plan: `full` covers the whole problem,
 * `windowControl(k, sweep)` window `k` of a sweep (1: PCG, 2: update, 0: assembly) and
 * `costControl(k)` cost window `k`.
 */
export function bundleWindowControls<T extends object>(
  control: T,
  plan: BundleResidency,
  M: number,
  P: number,
  streamed: boolean,
) {
  const full = {
    ...control,
    obsBase: 0,
    obsEnd: M,
    pointBase: 0,
    pointEnd: P,
    obsCapacity: plan.obsCapacity,
    pointCapacity: plan.pointCapacity,
    first: 1,
    last: 1,
    sweep: 0,
    streamed: Number(streamed),
    base: 0,
  };
  const windowControl = (k: number, sweep: number) => ({
    ...full,
    ...plan.windows[k]!,
    first: Number(k === 0),
    last: Number(k === plan.windows.length - 1),
    sweep,
  });
  const costControl = (k: number) => ({ ...full, ...plan.costWindows[k]! });
  return { full, windowControl, costControl };
}

/**
 * @internal The workgroups of `n` elements, the piece of the dispatch limit, and the steps of a
 * dispatch over `n` elements that exceeds the dispatch limit, split into pieces: piece `base`.
 */
export function bundlePieces<T extends { readonly base: number }>(
  device: Pick<ShadeDevice, 'maxKernelElements'>,
  full: T,
) {
  const groups = (n: number) => Math.ceil(n / 64);
  const piece = Math.max(64, device.maxKernelElements - (device.maxKernelElements % 64));
  function* pieces(name: string, n: number, run: (control: T, count: number) => void) {
    for (let at = 0; at < n; at += piece)
      yield [name, () => run({ ...full, base: at }, groups(Math.min(piece, n - at)))] as const;
  }
  return { groups, piece, pieces };
}

/**
 * @internal The output's refusals: a gauge without a baseline, and an observation that does not
 * project in front of its camera at the start.
 */
export function checkBundleOutput(values: Float64Array, base: number): void {
  if (values[base + BUNDLE_STATE.inputError] === 1)
    throw new RangeError('GPU BA gauge requires a nonzero baseline');
  if (values[base + BUNDLE_STATE.inputError] === 2)
    throw new RangeError('GPU BA observations must initially project in front of their cameras');
}

/** @internal The pose of camera `c` from the output, or its input pose when it did not change. */
export function bundleOutputPose(
  values: Float64Array,
  view: SparseReconstruction['views'][number],
  c: number,
  changed: boolean,
  active: readonly boolean[],
) {
  return changed && active[c]
    ? {
        R: values.slice(c * POSE_STRIDE, c * POSE_STRIDE + 9),
        t: values.slice(c * POSE_STRIDE + 9, (c + 1) * POSE_STRIDE),
      }
    : view.pose;
}

/** @internal The landmarks with the output's points, or the input points when nothing changed. */
export function bundleOutputLandmarks(
  model: SparseReconstruction,
  values: Float64Array,
  C: number,
  changed: boolean,
) {
  return model.landmarks.map((p, i) => ({
    ...p,
    xyz: changed
      ? values.slice(C * POSE_STRIDE + i * POINT_STRIDE, C * POSE_STRIDE + (i + 1) * POINT_STRIDE)
      : p.xyz,
  }));
}

/** @internal The diagnostics and solver counts of the output. */
export function bundleOutputReport(values: Float64Array, base: number, M: number) {
  return {
    diagnostics: {
      initialCost: values[base + BUNDLE_STATE.initialCost]!,
      finalCost: values[base + BUNDLE_STATE.cost]!,
      initialRmsePx: values[base + BUNDLE_STATE.initialRmsePx]!,
      finalRmsePx: values[base + BUNDLE_STATE.finalRmsePx]!,
      acceptedSteps: values[base + BUNDLE_STATE.accepted]!,
      rejectedSteps: values[base + BUNDLE_STATE.rejected]!,
      observations: M,
      baseline: values[base + BUNDLE_STATE.baseline]!,
    },
    cgIterations: values[base + BUNDLE_STATE.cgSteps]!,
    linearSolverFailures: values[base + BUNDLE_STATE.failedSolves]!,
  };
}
