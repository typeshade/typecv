import type { Resident } from 'typeshade/runtime';
import type { Camera } from '../geometry/camera.ts';
import {
  preparePairModels,
  appendPairModelRows,
  initializePairModelSvd,
  sweepPairModelSvd,
  summarizePairModelWork,
  snapshotPairModelBudget,
  finishPairModels,
} from './pair-models.shade.ts';
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
  snapshotTwoViewActivity,
  commitTwoViewPoints,
  finishTwoViewRefinement,
  validateTwoViewRefinement,
  finishTwoViewPoints,
  summarizeTwoView,
} from './recovery.shade.ts';
import {
  prepareImagePair,
  scanPairRows,
  scanPairRowsOracle,
  scanPairBlocks,
  compactImagePair,
  finishPairPreparation,
  preserveHomography,
  scatterPairConsensus,
} from './pair-preparation.shade.ts';

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
  verification?: { homographyIterations: number; homographyThreshold: number };
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
  verification?: {
    invertibleCount: number;
    homographyInlierCount: number;
    homographyIterations: number;
  };
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
    minimum = options.verification ? 1 : options.kind === 0 ? 8 : 4,
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
    requestedBatch < 1 ||
    (options.verification &&
      (!options.recovery ||
        options.kind !== 0 ||
        !Number.isSafeInteger(options.verification.homographyIterations) ||
        options.verification.homographyIterations < 1 ||
        options.verification.homographyIterations > 100000 ||
        !Number.isFinite(options.verification.homographyThreshold) ||
        options.verification.homographyThreshold < 1e-12 ||
        options.verification.homographyThreshold > 1e6))
  )
    throw new RangeError('Invalid GPU pair estimation inputs/options');
  const pixels = input.slice(),
    controls = { ...options },
    blocks = Math.ceil(n / 64),
    normalCount = Math.ceil(n / 32),
    recoveryBytes = options.recovery ? 564 * n + 964 + 352 * normalCount : 0,
    verificationBytes = options.verification ? 108 * n + 16 + blocks * 8 : 0,
    checkpointBytes = device.solverCheckpoints || device.boundedSubmissions ? 4 : 0,
    fixedBytes = 72 * n + 200 + recoveryBytes + verificationBytes + checkpointBytes,
    perModelBytes = 148 + blocks * 16 + (device.boundedSubmissions ? 1396 : 0),
    batchSize = Math.min(
      requestedBatch,
      device.maxModelsPerBatch,
      maximum,
      Math.floor((device.workingBytes - fixedBytes) / perModelBytes),
      Math.floor(device.bindingBytes / (device.boundedSubmissions ? 1344 : 144)),
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
        state = integer(controls.verification ? 12 : 8),
        best = wide(21),
        mask = integer(n);
      const modelWork = device.boundedSubmissions ? wide(batchSize * 168) : undefined;
      const modelState = device.boundedSubmissions ? integer(batchSize * 12) : undefined;
      const modelProgress =
        device.boundedSubmissions || device.solverCheckpoints ? integer(1) : undefined;
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
        verification: controls.verification ? 1 : 0,
        rowStart: 0,
        rowCount: 16,
      };
      const verification = controls.verification
        ? {
            prepared: wide(n * 8),
            ideal: wide(n * 4),
            prefix: integer(n),
            sourceRows: integer(n),
            outputMask: integer(n),
            blockOffsets: integer(blocks * 2),
          }
        : undefined;
      const recovery = controls.recovery
        ? {
            control: {
              first: verification ? { ...base.first, k1: 0, k2: 0 } : base.first,
              second: verification ? { ...base.second, k1: 0, k2: 0 } : base.second,
              threshold: base.threshold,
              iterations: controls.recovery.iterations,
              huber: controls.recovery.huber,
              minAngle: controls.recovery.minAngle,
              maxError: controls.recovery.maxError,
            },
            pixels: verification?.ideal ?? raw,
            coords,
            invertible,
            essential: best,
            mask,
            poses: wide(48),
            cloud: wide(16 * n),
            info: wide(72),
            points: wide(3 * n),
            trialPoints: wide(3 * n),
            activity: integer(1),
            fixed: integer(n),
            flags: integer(n),
            pointBlocks: wide(30 * n),
            normalBlocks: wide(normalCount * 44),
            metrics: wide(4 * n),
            angles: wide(n),
          }
        : undefined;
      const report = await device.planSteps(
        (function* () {
          function* consensus(settings: typeof base, data: Resident<Float64Array>) {
            let budget = settings.maximum;
            yield [
              'initializePair',
              () => {
                initializePair({ control: settings, invertible, state, best }, 1);
              },
            ] as const;
            function* step(control: typeof base) {
              if (modelWork && modelState && modelProgress) {
                const bindings = {
                  control,
                  coords: data,
                  invertible,
                  models,
                  valid,
                  state,
                  mask,
                  modelWork,
                  modelState,
                };
                const groups = Math.ceil(batchSize / 8);
                yield [
                  'preparePairModels',
                  () => {
                    preparePairModels(bindings, groups);
                  },
                ] as const;
                let remaining = true;
                const readProgress = async () => {
                  const word = (await device.read(modelProgress, 1))[0]!;
                  remaining = (word & 1) !== 0;
                  budget = word >>> 1;
                };
                yield [
                  'summarizePairModelWork',
                  () => {
                    summarizePairModelWork({ state, modelState, modelProgress }, 1);
                  },
                  readProgress,
                ] as const;
                if (remaining) {
                  const rows = control.refit ? n : control.kind === 0 ? 8 : 4;
                  for (let rowStart = 0; rowStart < rows; rowStart += 16) {
                    const chunk = { ...bindings, control: { ...control, rowStart } };
                    yield [
                      'appendPairModelRows',
                      () => {
                        appendPairModelRows(
                          selectBindings(chunk, [
                            'control',
                            'coords',
                            'invertible',
                            'valid',
                            'mask',
                            'modelWork',
                            'modelState',
                          ]),
                          groups,
                        );
                      },
                    ] as const;
                  }
                  yield [
                    'initializePairModelSvd',
                    () => {
                      initializePairModelSvd({ valid, modelWork, modelState }, groups);
                    },
                  ] as const;
                  if (remaining) {
                    yield [
                      'sweepPairModelSvd',
                      () => {
                        sweepPairModelSvd({ valid, modelWork, modelState }, groups);
                      },
                    ] as const;
                    yield [
                      'finishPairModels',
                      () => {
                        finishPairModels({ control, valid, models, modelWork, modelState }, groups);
                      },
                    ] as const;
                  }
                }
              } else
                yield [
                  'generatePairModels',
                  () => {
                    generatePairModels(
                      { control, coords: data, invertible, models, valid, state, mask },
                      Math.ceil(batchSize / 8),
                    );
                  },
                ] as const;
              const bindings = { control, coords: data, invertible, models, valid, partials };
              if (device.backend === 'webgpu')
                yield [
                  'scorePairModels',
                  () => {
                    scorePairModels(bindings, control.trials * blocks);
                  },
                ] as const;
              else
                yield [
                  'scorePairModelsOracle',
                  () => {
                    scorePairModelsOracle(bindings, Math.ceil((control.trials * blocks) / 8));
                  },
                ] as const;
              yield [
                'selectPairModel',
                () => {
                  selectPairModel({ control, state, models, valid, partials, best, invertible }, 1);
                },
              ] as const;
            }
            for (let start = 0; start < Math.min(settings.maximum, budget); start += batchSize) {
              yield* step({
                ...settings,
                start,
                trials: Math.min(batchSize, settings.maximum - start),
              });
              if (
                device.solverCheckpoints &&
                !device.boundedSubmissions &&
                settings.refit === 0 &&
                start + batchSize < Math.min(settings.maximum, budget)
              ) {
                yield [
                  'snapshotPairModelBudget',
                  () => snapshotPairModelBudget({ state, modelProgress: modelProgress! }, 1),
                  async () => {
                    budget = (await device.read(modelProgress!, 1))[0]!;
                  },
                ] as const;
              }
            }
            for (let pass = 0; pass < (settings.kind === 0 ? 4 : 3); pass++) {
              yield [
                'classifyPair',
                () => {
                  classifyPair(
                    { control: settings, state, coords: data, invertible, best, mask },
                    blocks,
                  );
                },
              ] as const;
              yield* step({ ...settings, trials: 1, refit: 1 });
            }
            yield [
              'classifyPair',
              () => {
                classifyPair(
                  { control: settings, state, coords: data, invertible, best, mask },
                  blocks,
                );
              },
            ] as const;
          }
          if (verification) {
            const v = verification;
            yield [
              'prepareImagePair',
              () => {
                prepareImagePair(
                  {
                    control: { first: base.first, second: base.second },
                    pixels: raw,
                    prepared: v.prepared,
                    prefix: v.prefix,
                  },
                  blocks,
                );
              },
            ] as const;
            const scan = { prefix: v.prefix, blockOffsets: v.blockOffsets };
            if (device.backend === 'webgpu')
              yield [
                'scanPairRows',
                () => {
                  scanPairRows(scan, blocks);
                },
              ] as const;
            else
              yield [
                'scanPairRowsOracle',
                () => {
                  scanPairRowsOracle(scan, Math.ceil(blocks / 8));
                },
              ] as const;
            yield [
              'scanPairBlocks',
              () => {
                scanPairBlocks({ blockOffsets: v.blockOffsets, state }, 1);
              },
            ] as const;
            yield [
              'compactImagePair',
              () => {
                compactImagePair(
                  {
                    prepared: v.prepared,
                    prefix: v.prefix,
                    blockOffsets: v.blockOffsets,
                    coords,
                    ideal: v.ideal,
                    sourceRows: v.sourceRows,
                  },
                  blocks,
                );
              },
            ] as const;
            yield [
              'finishPairPreparation',
              () => {
                finishPairPreparation({ state, invertible, outputMask: v.outputMask }, blocks);
              },
            ] as const;
            const h = controls.verification!;
            yield* consensus(
              {
                ...base,
                kind: 1,
                maximum: h.homographyIterations,
                minTrials: Math.min(32, h.homographyIterations),
                threshold: h.homographyThreshold,
                logFailure: Math.log1p(-0.999),
              },
              v.ideal,
            );
            yield [
              'preserveHomography',
              () => {
                preserveHomography({ state }, 1);
              },
            ] as const;
          } else
            yield [
              'preparePair',
              () => {
                preparePair({ control: base, pixels: raw, coords, invertible }, blocks);
              },
            ] as const;
          yield* consensus(base, coords);
          if (recovery) {
            const r = recovery;
            yield [
              'decomposeEssential',
              () => {
                decomposeEssential(selectBindings(r, ['essential', 'poses', 'info']), 1);
              },
            ] as const;
            yield [
              'triangulatePoseCandidates',
              () => {
                triangulatePoseCandidates(
                  selectBindings(r, ['coords', 'mask', 'poses', 'cloud', 'info']),
                  Math.ceil(n / 4),
                );
              },
            ] as const;
            yield [
              'chooseTwoViewPose',
              () => {
                chooseTwoViewPose(
                  selectBindings(r, ['info', 'mask', 'cloud', 'essential', 'poses']),
                  1,
                );
              },
            ] as const;
            yield [
              'seedTwoViewPoints',
              () => {
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
              },
            ] as const;
            yield [
              'beginTwoViewRefinement',
              () => {
                beginTwoViewRefinement(
                  selectBindings(r, ['info', 'metrics', 'mask', 'control']),
                  1,
                );
              },
            ] as const;
            let refinementActive = true;
            const readRecoveryActivity = async () => {
              refinementActive = (await device.read(r.activity, 1))[0] !== 0;
            };
            if (device.solverCheckpoints && r.control.iterations > 0)
              yield [
                'snapshotTwoViewActivity',
                () => {
                  snapshotTwoViewActivity(selectBindings(r, ['info', 'activity']), 1);
                },
                readRecoveryActivity,
              ] as const;
            for (
              let iteration = 0;
              iteration < r.control.iterations && refinementActive;
              iteration++
            ) {
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
              if (device.backend === 'webgpu')
                yield [
                  'twoViewNormals',
                  () => {
                    twoViewNormals(normals, normalCount);
                  },
                ] as const;
              else
                yield [
                  'twoViewNormalsOracle',
                  () => {
                    twoViewNormalsOracle(normals, Math.ceil(normalCount / 8));
                  },
                ] as const;
              yield [
                'twoViewStep',
                () => {
                  twoViewStep(selectBindings(r, ['info', 'normalBlocks']), 1);
                },
              ] as const;
              yield [
                'twoViewTrial',
                () => {
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
                },
              ] as const;
              yield [
                'acceptTwoViewStep',
                () => {
                  acceptTwoViewStep(selectBindings(r, ['info', 'mask', 'metrics']), 1);
                },
              ] as const;
              yield [
                'commitTwoViewPoints',
                () => {
                  commitTwoViewPoints(
                    selectBindings(r, ['mask', 'info', 'fixed', 'points', 'trialPoints']),
                    blocks,
                  );
                },
              ] as const;
              if (
                device.solverCheckpoints &&
                (iteration + 1) % 4 === 0 &&
                iteration + 1 < r.control.iterations
              )
                yield [
                  'snapshotTwoViewActivity',
                  () => {
                    snapshotTwoViewActivity(selectBindings(r, ['info', 'activity']), 1);
                  },
                  readRecoveryActivity,
                ] as const;
            }
            if (r.control.iterations > 0)
              yield [
                'validateTwoViewRefinement',
                () => {
                  validateTwoViewRefinement(
                    selectBindings(r, ['fixed', 'info', 'coords', 'pixels', 'control', 'metrics']),
                    Math.ceil(n / 16),
                  );
                },
              ] as const;
            yield [
              'finishTwoViewRefinement',
              () => {
                finishTwoViewRefinement(selectBindings(r, ['info', 'mask', 'metrics']), 1);
              },
            ] as const;
            yield [
              'finishTwoViewPoints',
              () => {
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
              },
            ] as const;
            yield [
              'summarizeTwoView',
              () => {
                summarizeTwoView(
                  selectBindings(r, ['control', 'info', 'mask', 'metrics', 'flags', 'angles']),
                  1,
                );
              },
            ] as const;
          }
          if (verification)
            yield [
              'scatterPairConsensus',
              () => {
                scatterPairConsensus(
                  {
                    mask,
                    sourceRows: verification.sourceRows,
                    outputMask: verification.outputMask,
                  },
                  blocks,
                );
              },
            ] as const;
        })(),
      );
      // All public snapshots follow the final dispatch, including convergence decisions.
      let values: Float64Array | undefined,
        recoveryResult: PairEstimationResult['recovery'],
        flags: Uint32Array,
        metadata: Uint32Array;
      if (verification) {
        // These independent reads share the same completed compute prefix. Keep this
        // job's buffers until every snapshot settles, even when an earlier one fails.
        const [info, output, status] = await Promise.allSettled([
          device.read(recovery!.info, 72),
          device.read(verification.outputMask, n),
          device.read(state, 12),
        ]);
        // Preserve the original read order's first rejection, including undefined.
        if (info.status === 'rejected') throw info.reason;
        if (output.status === 'rejected') throw output.reason;
        if (status.status === 'rejected') throw status.reason;
        recoveryResult = {
          info: info.value.slice(),
          points: new Float64Array(0),
          valid: new Uint8Array(0),
        };
        flags = output.value;
        metadata = status.value;
      } else {
        values = recovery ? undefined : await device.read(best, 21);
        recoveryResult = recovery
          ? {
              info: (await device.read(recovery.info, 72)).slice(),
              points: (await device.read(recovery.points, 3 * n)).slice(),
              valid: Uint8Array.from(await device.read(recovery.flags, n)),
            }
          : undefined;
        flags = await device.read(mask, n);
        metadata = await device.read(state, 8);
      }
      const after = device.statistics;
      if (verification && metadata[11])
        throw new RangeError('Undistorted pixel exceeds the supported GPU geometry bounds');
      if (verification && metadata[7] !== metadata[10])
        throw new Error('GPU pair compaction population mismatch');
      successful = true;
      return {
        verification: verification
          ? {
              invertibleCount: metadata[10]!,
              homographyInlierCount: metadata[8]!,
              homographyIterations: metadata[9]!,
            }
          : undefined,
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
              verification: 0,
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
