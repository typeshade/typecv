import type { Resident } from 'typeshade/runtime';
import type { Camera } from '../geometry/camera.ts';
import { ShadeDevice } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import {
  preparePair,
  undistortPair,
  initializePair,
  generatePairModels,
  scorePairModels,
  scorePairModelsOracle,
  selectPairModel,
  classifyPair,
} from './consensus.shade.ts';
import {
  decomposeEssential,
  triangulatePoseCandidates,
  chooseTwoViewPose,
  seedTwoViewPoints,
  beginTwoViewRefinement,
  twoViewNormals,
  twoViewNormalsOracle,
  twoViewStep,
  twoViewTrial,
  acceptTwoViewStep,
  commitTwoViewPoints,
  finishTwoViewRefinement,
  validateTwoViewRefinement,
  finishTwoViewPoints,
  summarizeTwoView,
} from './recovery.shade.ts';

/** @internal */
export interface PairEstimationOptions {
  kind: 0 | 1;
  iterations: number;
  minIterations: number;
  threshold: number;
  confidence: number;
  seed: number;
  batchSize?: number;
  recovery?: { iterations: number; huber: number; minAngle: number; maxError: number };
}
/** @internal */
export interface PairEstimationResult {
  model: Float64Array;
  mask: Uint8Array;
  count: number;
  msac: number;
  trials: number;
  batches: number;
  hypotheses: number;
  recovery?: { info: Float64Array; points: Float64Array; valid: Uint8Array };
  execution: GpuGeometryExecution;
}
const identityCamera = { fx: 1, fy: 1, cx: 0, cy: 0, k1: 0, k2: 0 };

/** @internal Sampling, proposals, scoring, stopping and all inlier refits stay resident. */
export function estimatePairGpu(
  device: ShadeDevice,
  input: Float64Array,
  first: Camera,
  second: Camera,
  options: PairEstimationOptions,
): Promise<PairEstimationResult> {
  const a = validateGpuCamera({ ...first }),
    b = validateGpuCamera({ ...second }),
    n = input.length / 4,
    minimum = options.kind === 0 ? 8 : 4,
    maximum = options.iterations,
    requestedBatch = options.batchSize ?? 32;
  if (
    !Number.isInteger(n) ||
    n < minimum ||
    !finiteGpuArray(input) ||
    !Number.isSafeInteger(maximum) ||
    maximum < 1 ||
    maximum > 100000 ||
    !Number.isSafeInteger(options.minIterations) ||
    options.minIterations < 1 ||
    options.minIterations > maximum ||
    !Number.isFinite(options.threshold) ||
    options.threshold < 1e-12 ||
    options.threshold > 1e6 ||
    !(options.confidence > 0 && options.confidence < 1) ||
    !Number.isSafeInteger(options.seed) ||
    !Number.isSafeInteger(requestedBatch) ||
    requestedBatch < 1
  )
    throw new RangeError('Invalid GPU pair estimation inputs/options');
  const pixels = input.slice(),
    controls = { ...options },
    blocks = Math.ceil(n / 64),
    normalCount = Math.ceil(n / 32),
    recoveryBytes = options.recovery ? 564 * n + 960 + 352 * normalCount : 0,
    fixedBytes = 72 * n + 200 + recoveryBytes,
    perModelBytes = 148 + blocks * 16,
    batchSize = Math.min(
      requestedBatch,
      device.maxModelsPerBatch,
      maximum,
      Math.floor((device.workingBytes - fixedBytes) / perModelBytes),
      Math.floor(device.bindingBytes / 144),
      Math.floor(device.bindingBytes / (blocks * 16)),
      Math.floor(device.maxKernelElements / (64 * blocks)),
    );
  if (
    n > device.maxKernelElements ||
    n * 32 > device.bindingBytes ||
    batchSize < 1 ||
    (options.recovery &&
      (128 * n > device.bindingBytes ||
        240 * n > device.bindingBytes ||
        normalCount * 352 > device.bindingBytes ||
        n > device.maxKernelElements / 16))
  )
    throw new RangeError('GPU pair estimation exceeds working/storage/dispatch budget');
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (size: number, source?: Float64Array) => {
      const buffer = device.allocateF64(size, source);
      owned.push(buffer);
      return buffer;
    };
    const integer = (size: number) => {
      const buffer = device.allocateU32(size);
      owned.push(buffer);
      return buffer;
    };
    try {
      const raw = wide(pixels.length, pixels),
        coords = wide(pixels.length),
        invertible = integer(n),
        models = wide(batchSize * 18),
        valid = integer(batchSize),
        partials = wide(batchSize * blocks * 2),
        state = integer(8),
        best = wide(21),
        mask = integer(n);
      const base = {
        first: { ...identityCamera, ...a },
        second: { ...identityCamera, ...b },
        kind: controls.kind,
        threshold: controls.threshold,
        seed: controls.seed >>> 0,
        start: 0,
        trials: batchSize,
        maximum,
        minTrials: controls.minIterations,
        blocks,
        refit: 0,
        logFailure: Math.log1p(-controls.confidence),
      };
      const recovery = controls.recovery
        ? {
            control: {
              first: base.first,
              second: base.second,
              threshold: base.threshold,
              iterations: controls.recovery.iterations,
              huber: controls.recovery.huber,
              minAngle: controls.recovery.minAngle,
              maxError: controls.recovery.maxError,
            },
            pixels: raw,
            coords,
            invertible,
            essential: best,
            mask,
            poses: wide(48),
            cloud: wide(16 * n),
            info: wide(72),
            points: wide(3 * n),
            trialPoints: wide(3 * n),
            fixed: integer(n),
            flags: integer(n),
            pointBlocks: wide(30 * n),
            normalBlocks: wide(normalCount * 44),
            metrics: wide(4 * n),
            angles: wide(n),
          }
        : undefined;
      const report = await device.plan(() => {
        preparePair({ control: base, pixels: raw, coords, invertible }, blocks);
        initializePair({ control: base, invertible, state, best }, 1);
        const step = (control: typeof base) => {
          generatePairModels(
            { control, coords, invertible, models, valid, state, mask },
            Math.ceil(batchSize / 8),
          );
          const bindings = { control, coords, invertible, models, valid, partials };
          if (device.backend === 'webgpu') scorePairModels(bindings, control.trials * blocks);
          else scorePairModelsOracle(bindings, Math.ceil((control.trials * blocks) / 8));
          selectPairModel({ control, state, models, valid, partials, best, invertible }, 1);
        };
        for (let start = 0; start < maximum; start += batchSize)
          step({ ...base, start, trials: Math.min(batchSize, maximum - start) });
        for (let pass = 0; pass < (controls.kind === 0 ? 4 : 3); pass++) {
          classifyPair({ control: base, state, coords, invertible, best, mask }, blocks);
          step({ ...base, trials: 1, refit: 1 });
        }
        classifyPair({ control: base, state, coords, invertible, best, mask }, blocks);
        if (recovery) {
          const r = recovery;
          decomposeEssential(selectBindings(r, ['essential', 'poses', 'info']), 1);
          triangulatePoseCandidates(
            selectBindings(r, ['coords', 'mask', 'poses', 'cloud', 'info']),
            Math.ceil(n / 4),
          );
          chooseTwoViewPose(selectBindings(r, ['info', 'mask', 'cloud', 'essential', 'poses']), 1);
          seedTwoViewPoints(
            selectBindings(r, [
              'mask',
              'fixed',
              'points',
              'metrics',
              'info',
              'cloud',
              'control',
              'pixels',
            ]),
            blocks,
          );
          beginTwoViewRefinement(selectBindings(r, ['info', 'metrics', 'mask', 'control']), 1);
          for (let iteration = 0; iteration < r.control.iterations; iteration++) {
            const normals = selectBindings(r, [
              'mask',
              'info',
              'fixed',
              'points',
              'control',
              'pixels',
              'pointBlocks',
              'normalBlocks',
            ]);
            if (device.backend === 'webgpu') twoViewNormals(normals, normalCount);
            else twoViewNormalsOracle(normals, Math.ceil(normalCount / 8));
            twoViewStep(selectBindings(r, ['info', 'normalBlocks']), 1);
            twoViewTrial(
              selectBindings(r, [
                'mask',
                'metrics',
                'info',
                'fixed',
                'pointBlocks',
                'points',
                'trialPoints',
                'control',
                'pixels',
              ]),
              blocks,
            );
            acceptTwoViewStep(selectBindings(r, ['info', 'mask', 'metrics']), 1);
            commitTwoViewPoints(
              selectBindings(r, ['mask', 'info', 'fixed', 'points', 'trialPoints']),
              blocks,
            );
          }
          if (r.control.iterations > 0)
            validateTwoViewRefinement(
              selectBindings(r, ['fixed', 'info', 'coords', 'pixels', 'control', 'metrics']),
              Math.ceil(n / 16),
            );
          finishTwoViewRefinement(selectBindings(r, ['info', 'mask', 'metrics']), 1);
          finishTwoViewPoints(
            selectBindings(r, [
              'mask',
              'flags',
              'points',
              'metrics',
              'info',
              'invertible',
              'control',
              'coords',
              'pixels',
            ]),
            Math.ceil(n / 16),
          );
          summarizeTwoView(
            selectBindings(r, ['control', 'info', 'mask', 'metrics', 'flags', 'angles']),
            1,
          );
        }
      });
      // All public snapshots follow the final dispatch, including convergence decisions.
      const values = recovery ? undefined : await device.read(best, 21),
        recoveryResult = recovery
          ? {
              info: (await device.read(recovery.info, 72)).slice(),
              points: (await device.read(recovery.points, 3 * n)).slice(),
              valid: Uint8Array.from(await device.read(recovery.flags, n)),
            }
          : undefined,
        flags = await device.read(mask, n),
        metadata = await device.read(state, 8),
        after = device.statistics;
      successful = true;
      return {
        model: recoveryResult ? recoveryResult.info.slice(57, 66) : values!.slice(0, 9),
        mask: Uint8Array.from(flags),
        count: recoveryResult ? recoveryResult.info[69]! : metadata[3]!,
        msac: values?.[20] ?? NaN,
        trials: metadata[1]!,
        batches: metadata[5]!,
        hypotheses: metadata[2]!,
        recovery: recoveryResult,
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
      for (const buffer of owned) successful ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}

export interface GpuUndistortionResult {
  readonly pixels: Float64Array;
  /** Original correspondence index for each retained row. */
  readonly indices: Uint32Array;
  readonly execution: GpuGeometryExecution;
}

/** Invert two radial cameras on the device, then compact the returned valid rows. */
export function undistortCorrespondencesGpu(
  device: ShadeDevice,
  input: Float64Array,
  first: Camera,
  second: Camera,
): Promise<GpuUndistortionResult> {
  const a = validateGpuCamera({ ...first }),
    b = validateGpuCamera({ ...second }),
    n = input.length / 4;
  if (!Number.isInteger(n) || !n || !finiteGpuArray(input))
    throw new RangeError('Invalid GPU undistortion correspondences');
  if (68 * n > device.workingBytes || 32 * n > device.bindingBytes || n > device.maxKernelElements)
    throw new RangeError('GPU undistortion exceeds working/storage/dispatch budget');
  const pixels = input.slice();
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    try {
      const raw = device.allocateF64(pixels.length, pixels);
      owned.push(raw);
      const coords = device.allocateF64(pixels.length);
      owned.push(coords);
      const invertible = device.allocateU32(n);
      owned.push(invertible);
      const report = await device.plan(() => {
        undistortPair(
          {
            control: {
              first: { ...identityCamera, ...a },
              second: { ...identityCamera, ...b },
              kind: 0,
              threshold: 1,
              seed: 0,
              start: 0,
              trials: 0,
              maximum: 0,
              minTrials: 0,
              blocks: 0,
              refit: 0,
              logFailure: 0,
            },
            pixels: raw,
            coords,
            invertible,
          },
          Math.ceil(n / 64),
        );
      });
      const values = await device.read(coords, n * 4),
        flags = await device.read(invertible, n),
        indices: number[] = [],
        packed: number[] = [];
      for (let i = 0; i < n; i++)
        if (flags[i]) {
          indices.push(i);
          packed.push(...values.subarray(4 * i, 4 * i + 4));
        }
      const after = device.statistics;
      successful = true;
      return {
        pixels: Float64Array.from(packed),
        indices: Uint32Array.from(indices),
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
      for (const buffer of owned) successful ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}

function selectBindings<T, K extends keyof T>(source: T, keys: readonly K[]): Pick<T, K> {
  return Object.fromEntries(keys.map((key) => [key, source[key]])) as Pick<T, K>;
}
