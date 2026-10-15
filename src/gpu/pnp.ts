import type { Resident } from 'typeshade/runtime';
import { GeometryEstimationError } from '../geometry/errors.ts';
import type { Camera } from '../geometry/camera.ts';
import { validateCamera } from '../geometry/camera.ts';
import type { PnPOptions, PnPResult } from '../geometry/pnp.ts';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import { solveP3p } from './p3p.shade.ts';
import {
  normalizePnP,
  initializePnP,
  samplePnP,
  scorePnP,
  scorePnPOracle,
  selectPnP,
  consensusMask,
  beginPoseRefinement,
  poseNormals,
  poseNormalsOracle,
  poseStep,
  poseTrialCosts,
  acceptPoseStep,
  finishPoseRefinement,
  summarizePnP,
} from './pnp.shade.ts';

export interface GpuGeometryExecution {
  readonly backend: 'webgpu' | 'cpu';
  readonly report: ShadePlanReport;
  readonly applicationUploadBytes: number;
  readonly applicationReadbackBytes: number;
  readonly cpuStages: readonly string[];
}
export interface GpuPnPOptions extends PnPOptions {
  readonly batchSize?: number;
  readonly minIterations?: number;
}
export interface GpuPnPResult extends PnPResult {
  readonly candidatesEvaluated: number;
  readonly refinementAcceptedSteps: number;
  readonly refinementRejectedSteps: number;
  readonly execution: GpuGeometryExecution;
}

/** @internal Host validation only; numerical camera inversion/projection is in shader code. */
export function validateGpuCamera(camera: Camera): Camera {
  validateCamera(camera);
  if (
    camera.fx < 1e-6 ||
    camera.fy < 1e-6 ||
    camera.fx > 1e6 ||
    camera.fy > 1e6 ||
    Math.max(Math.abs(camera.cx), Math.abs(camera.cy)) > 1e9 ||
    Math.max(Math.abs(camera.k1 ?? 0), Math.abs(camera.k2 ?? 0)) > 1
  )
    throw new RangeError('Camera exceeds the supported GPU geometry bounds');
  return { ...camera, k1: camera.k1 ?? 0, k2: camera.k2 ?? 0 };
}
/** @internal */
export function finiteGpuArray(array: Float64Array): boolean {
  return (
    array instanceof Float64Array &&
    array.every(
      (x) => Number.isFinite(x) && Math.abs(x) <= 1e9 && (x === 0 || Math.abs(x) >= 1e-30),
    )
  );
}

/** GPU P3P RANSAC, including sampling, radial inversion, consensus selection and pose LM. */
export function estimatePosePnPGpu(
  device: ShadeDevice,
  worldPoints: Float64Array,
  pixels: Float64Array,
  camera: Camera,
  options: GpuPnPOptions = {},
): Promise<GpuPnPResult> {
  const k = validateGpuCamera({ ...camera }),
    n = worldPoints.length / 3;
  const maximum = options.iterations ?? 1000,
    threshold = options.thresholdPx ?? 3,
    confidence = options.confidence ?? 0.999,
    seed = options.seed ?? 17,
    minimum = options.minInliers ?? Math.min(6, n),
    refinement = options.refinementIterations ?? 20,
    minTrials = options.minIterations ?? Math.min(64, maximum),
    requestedBatch = options.batchSize ?? 32;
  if (
    !Number.isInteger(n) ||
    n < 4 ||
    pixels.length !== 2 * n ||
    !finiteGpuArray(worldPoints) ||
    !finiteGpuArray(pixels) ||
    !Number.isSafeInteger(maximum) ||
    maximum < 1 ||
    maximum > 100000 ||
    !Number.isFinite(threshold) ||
    threshold < 1e-12 ||
    threshold > 1e6 ||
    !(confidence > 0 && confidence < 1) ||
    !Number.isSafeInteger(seed) ||
    !Number.isSafeInteger(minimum) ||
    minimum < 4 ||
    minimum > n ||
    !Number.isSafeInteger(refinement) ||
    refinement < 0 ||
    refinement > 100 ||
    !Number.isSafeInteger(minTrials) ||
    minTrials < 1 ||
    minTrials > maximum ||
    !Number.isSafeInteger(requestedBatch) ||
    requestedBatch < 1
  )
    throw new RangeError('Invalid GPU PnP correspondences/options');
  const blocks = Math.ceil(n / 64),
    normalCount = Math.ceil(n / 32),
    maxGroups = device.maxKernelElements / 64;
  // XYZ/pixels/normalized observations/mask, normal blocks, trial costs, persistent solve state.
  const fixedBytes = n * 108 + normalCount * 44 * 8 + blocks * 8 + 480;
  const perTrialBytes = 552 + 4 * blocks * 2 * 8;
  const batchSize = Math.min(
    requestedBatch,
    device.maxModelsPerBatch,
    maximum,
    Math.floor((device.workingBytes - fixedBytes) / perTrialBytes),
    Math.floor(device.bindingBytes / 384),
    Math.floor(device.bindingBytes / (4 * blocks * 16)),
    Math.floor(maxGroups / (4 * blocks)),
  );
  if (
    normalCount > maxGroups ||
    n * 64 > device.bindingBytes ||
    normalCount * 44 * 8 > device.bindingBytes ||
    batchSize < 1
  )
    throw new RangeError('GPU PnP exceeds the working/storage/dispatch budget');
  const X = worldPoints.slice(),
    uv = pixels.slice();
  return device.enqueue(async () => {
    const before = device.statistics;
    const owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (length: number, values?: Float64Array) => {
      const b = device.allocateF64(length, values);
      owned.push(b);
      return b;
    };
    const integer = (length: number) => {
      const b = device.allocateU32(length);
      owned.push(b);
      return b;
    };
    try {
      const xyz = wide(X.length, X),
        pixels = wide(uv.length, uv),
        observations = wide(n * 8),
        problems = wide(batchSize * 18),
        poses = wide(batchSize * 48),
        valid = integer(batchSize * 4),
        counts = integer(batchSize),
        status = integer(batchSize),
        partials = wide(batchSize * 4 * blocks * 2),
        state = integer(8),
        best = wide(16),
        working = wide(12),
        candidate = wide(12),
        mask = integer(n),
        lm = wide(4),
        normalBlocks = wide(normalCount * 44),
        trialCosts = wide(blocks);
      const baseControl = {
        camera: { fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy, k1: k.k1!, k2: k.k2! },
        threshold,
        seed: seed >>> 0,
        start: 0,
        trials: batchSize,
        maximum,
        minimum,
        minTrials,
        blocks,
        logFailure: Math.log1p(-confidence),
      };
      const poseBuffers = { best, working, candidate };
      const control = baseControl;
      const report = await device.plan(() => {
        normalizePnP({ control, xyz, pixels, observations }, Math.ceil(n / 64));
        initializePnP({ control, state, best, observations }, 1);
        for (let start = 0; start < maximum; start += batchSize) {
          const batchControl = {
            ...baseControl,
            start,
            trials: Math.min(batchSize, maximum - start),
          };
          samplePnP(
            { control: batchControl, state, observations, problems },
            Math.ceil(batchSize / 32),
          );
          solveP3p({ problems, poses, valid, counts, status }, Math.ceil(batchSize / 32));
          const bindings = { control: batchControl, observations, poses, valid, partials };
          if (device.backend === 'webgpu') scorePnP(bindings, batchSize * 4 * blocks);
          else scorePnPOracle(bindings, Math.ceil((batchSize * 4 * blocks) / 32));
          selectPnP(
            { control: batchControl, state, best, observations, poses, valid, partials },
            1,
          );
        }
        for (let pass = 0; pass < (refinement ? 3 : 0); pass++) {
          consensusMask({ control, state, mask, observations, ...poseBuffers }, Math.ceil(n / 64));
          beginPoseRefinement({ control, state, best, working, lm }, 1);
          for (let iteration = 0; iteration < refinement; iteration++) {
            const bindings = { control, mask, lm, observations, normalBlocks, ...poseBuffers };
            if (device.backend === 'webgpu') poseNormals(bindings, normalCount);
            else poseNormalsOracle(bindings, Math.ceil(normalCount / 32));
            poseStep({ normalBlocks, lm, ...poseBuffers }, 1);
            poseTrialCosts(
              { control, mask, lm, observations, trialCosts, ...poseBuffers },
              Math.ceil(blocks / 32),
            );
            acceptPoseStep({ lm, state, candidate, working, trialCosts }, 1);
          }
          finishPoseRefinement({ control, state, mask, observations, ...poseBuffers }, 1);
        }
        consensusMask({ control, state, mask, observations, ...poseBuffers }, Math.ceil(n / 64));
        summarizePnP({ control, mask, observations, ...poseBuffers }, 1);
      });
      // No intermediate candidate, mask, score or convergence reads occurred.
      const summary = await device.read(best, 16),
        flags = await device.read(mask, n),
        metadata = await device.read(state, 8);
      if (metadata[7]) throw new RangeError('Cannot invert camera distortion for GPU PnP pixel');
      if (metadata[3]! < minimum)
        throw new GeometryEstimationError(
          'GPU PnP could not register camera: insufficient nondegenerate consensus',
        );
      const after = device.statistics;
      successful = true;
      return {
        pose: { R: summary.slice(0, 9), t: summary.slice(9, 12) },
        inliers: Uint8Array.from(flags),
        inlierCount: metadata[3]!,
        meanErrorPx: summary[14]! / metadata[3]!,
        iterations: metadata[1]!,
        candidatesEvaluated: metadata[2]!,
        refinementAcceptedSteps: metadata[5]!,
        refinementRejectedSteps: metadata[6]!,
        execution: {
          backend: device.backend,
          report,
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          cpuStages: [],
        },
      };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}
