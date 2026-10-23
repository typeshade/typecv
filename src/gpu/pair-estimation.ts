import type { Resident } from 'typeshade/runtime';
import type { Camera } from '../geometry/camera.ts';
import {
  preparePairModels,
  appendPairModelRows,
  solvePairModelNull,
  summarizePairModelWork,
  snapshotPairModelBudget,
  planPairModelBatch,
  finishPairModels,
} from './pair-models.shade.ts';
import { ShadeDevice, type GpuStep, type ShadePlanReport } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import {
  preparePair,
  undistortPair,
  initializePair,
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
  recovery?: {
    iterations: number;
    huber: number;
    minAngle: number;
    maxError: number;
    separateDirectionalObservations?: boolean;
  };
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
/**
 * The batches of a pair search between two reads of the scheduling word. The GPU decides each
 * batch's work through the indirect workgroup counts (TypeShade change 0075), so the host reads
 * nothing between batches. It reads the adaptive budget after every fourth batch to leave the
 * search early: at most three batches past the budget are submitted, and each dispatches zero
 * workgroups but its three one-workgroup control kernels.
 */
const PROGRESS_INTERVAL = 4;
/**
 * Rows that one dispatch of `appendPairModelRows` appends to the model of a refit. A refit has
 * one model and appends all `n` rows to it, so the dispatches were `n / 16` for each of the eight
 * refits of a pair: 600 matches cost 300 dispatches. Each dispatch costs the host a check of every
 * array it binds, whatever its work. A search batch appends 8 rows, so it is one dispatch either
 * way (Amendment 8 of docs/multi-video.md).
 */
const REFIT_ROW_CHUNK = 128;
/**
 * Pairs that share their plans (`estimatePairsGpu`) read the scheduling word after this many
 * batches. One round trip serves all of them.
 */
const SHARED_PROGRESS_INTERVAL = 4;
/**
 * The models a search batch scores at least, to fill the device: the scoring kernel has one model
 * for each 8-lane workgroup, so 256 models are 32 workgroups of a latency-bound kernel whose
 * dispatch costs the same at 32 as at 256 (`docs/gpu-throughput.md`, item 2).
 */
const PAIR_SCORING_FLOOR = 256;
/** The offsets, in words, of the workgroup counts of `pair-models.shade.ts` (`counts`). */
const PREPARE_COUNT = 0,
  MODEL_COUNT = 3,
  SCORE_COUNT = 6,
  SCORE_ORACLE_COUNT = 9;
/** @internal Sampling, proposals, scoring, stopping and all inlier refits stay resident. */
function checkPairEstimation(
  device: ShadeDevice,
  input: Float64Array,
  first: Camera,
  second: Camera,
  options: PairEstimationOptions,
  share: number,
): () => PairEstimationTask {
  const a = validateGpuCamera({ ...first }),
    b = validateGpuCamera({ ...second }),
    n = input.length / 4,
    minimum = options.verification ? 1 : options.kind === 0 ? 8 : 4,
    maximum = options.iterations,
    requestedBatch = options.batchSize ?? 512;
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
    // The scheduling word, the twelve words of indirect workgroup counts, and the triangular
    // factor, null vector and state of each model: the model generation is a state machine on
    // every device, one dispatch per step.
    checkpointBytes = 4 + 48,
    perModelWords = 98,
    fixedBytes = 72 * n + 200 + recoveryBytes + verificationBytes + checkpointBytes,
    perModelBytes = 148 + blocks * 16 + 52 + perModelWords * 8,
    batchSize = Math.min(
      requestedBatch,
      device.maxModelsPerBatch,
      maximum,
      Math.floor((device.workingBytes / share - fixedBytes) / perModelBytes),
      Math.floor(device.bindingBytes / (perModelWords * 8)),
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
  return () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const release = () => {
      for (const buffer of owned) successful ? device.recycle(buffer) : device.drop(buffer);
    };
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
      const modelWork = wide(batchSize * perModelWords),
        modelState = integer(batchSize * 12),
        modelProgress = integer(1),
        counts = integer(12);
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
              separateDirectional: controls.recovery.separateDirectionalObservations ? 1 : 0,
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
      const scoringFloor = Math.min(batchSize, PAIR_SCORING_FLOOR);
      const steps = (function* () {
        function* consensus(settings: typeof base, data: Resident<Float64Array>) {
          let budget = settings.maximum;
          yield [
            'initializePair',
            () => {
              initializePair({ control: settings, invertible, state, best }, 1);
            },
          ] as const;
          // The models of a batch are made as a state machine, one short dispatch per step
          // (`docs/gpu-resident-scene.md`, step 2): the rows are appended sixteen at a time,
          // and one dispatch solves each model's triangular factor for its null vector, at a
          // fixed cost (no kernel iterates to convergence). The scoring of the batch is a plan
          // of its own (key `scorePairModels`) and runs as one dispatch: a batch is at most
          // `maxModelsPerBatch` trials.
          //
          // A search batch decides its work on the GPU (TypeShade change 0075):
          // `planPairModelBatch` writes the prepare's workgroup count, and
          // `summarizePairModelWork` those of the row appends, the null solve, the finish and
          // the scoring, zero for a batch past the adaptive budget or with no active model.
          // Those entries dispatch with `{ indirect: counts, offset }`, so the host reads
          // nothing between the kernels of a batch. A refit (`refit: 1`) always prepares and
          // scores one model, so only its row appends, null solve and finish are indirect.
          //
          // `timed` says the host knows the scoring runs: the first batch of a search, a refit,
          // and the first batch after a read of the budget that showed the search goes on. Only
          // such a scoring is timed for the rate of a trial. Another one runs alone, untimed:
          // the GPU may have set its count to zero, and a plan that dispatches nothing would
          // teach the device a rate far too fast.
          function* step(control: typeof base, timed: boolean) {
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
            const groups = Math.ceil(batchSize / 8),
              search = control.refit === 0,
              modelCount = { indirect: counts, offset: MODEL_COUNT };
            if (search)
              yield [
                'planPairModelBatch',
                () => {
                  planPairModelBatch({ control, state, modelState, counts }, 1);
                },
              ] as const;
            yield [
              'preparePairModels',
              () => {
                preparePairModels(
                  bindings,
                  search ? { indirect: counts, offset: PREPARE_COUNT } : groups,
                );
              },
            ] as const;
            yield [
              'summarizePairModelWork',
              () => {
                summarizePairModelWork({ control, state, modelState, counts }, 1);
              },
            ] as const;
            const rows = control.refit ? n : control.kind === 0 ? 8 : 4;
            // A refit appends every row of the pair to one model, in order, on one invocation. The
            // dispatches only bound the time of an invocation, so a refit takes more rows in one
            // (`REFIT_ROW_CHUNK`): the arithmetic and its order are the same.
            const rowChunk = control.refit ? REFIT_ROW_CHUNK : 16;
            for (let rowStart = 0; rowStart < rows; rowStart += rowChunk) {
              const chunk = { ...bindings, control: { ...control, rowStart, rowCount: rowChunk } };
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
                    modelCount,
                  );
                },
              ] as const;
            }
            yield [
              'solvePairModelNull',
              () => {
                solvePairModelNull({ valid, modelWork, modelState }, modelCount);
              },
            ] as const;
            yield [
              'finishPairModels',
              () => {
                finishPairModels({ control, valid, models, modelWork, modelState }, modelCount);
              },
            ] as const;
            const scoring = { control, coords: data, invertible, models, valid, partials },
              chunk = {
                key: 'scorePairModels',
                units: control.trials,
                ...(timed ? {} : { timed: false }),
                // The refit scores one model: it is no search batch, and it teaches no floor.
                ...(control.refit === 0 ? { minUnits: scoringFloor } : {}),
              };
            if (device.backend === 'webgpu')
              yield [
                'scorePairModels',
                () => {
                  scorePairModels(
                    scoring,
                    search ? { indirect: counts, offset: SCORE_COUNT } : control.trials * blocks,
                  );
                },
                undefined,
                {
                  ...chunk,
                  // Zero workgroups: the pipeline is made before the first timed scoring.
                  warm: { id: 'pair', run: () => scorePairModels(scoring, 0) },
                },
              ] as const;
            else
              yield [
                'scorePairModelsOracle',
                () => {
                  scorePairModelsOracle(
                    scoring,
                    search
                      ? { indirect: counts, offset: SCORE_ORACLE_COUNT }
                      : Math.ceil((control.trials * blocks) / 8),
                  );
                },
                undefined,
                chunk,
              ] as const;
            yield [
              'selectPairModel',
              () => {
                selectPairModel({ control, state, models, valid, partials, best, invertible }, 1);
              },
            ] as const;
          }
          // The host bounds the search by the requested maximum, as before, and by the budget
          // it read last. It reads the budget after every `PROGRESS_INTERVAL` batches; a batch
          // submitted after the GPU met the budget dispatches nothing.
          let timed = true;
          for (let start = 0, batch = 1; start < Math.min(settings.maximum, budget); batch++) {
            // The selection stops where a sequential search would (`selectPairModel`), so the
            // batch is a performance choice: as many trials as the time budget allows at the
            // scoring rate the device learned, at least the occupancy floor that fills the device
            // (`PAIR_SCORING_FLOOR` trials, shared by the streams of a job) and at most
            // `batchSize`. The CPU tier, which has no watchdog and no idle lanes, scores 64 at a
            // time.
            const trials = Math.min(
              settings.maximum - start,
              device.backend === 'cpu'
                ? Math.min(batchSize, 64)
                : Math.max(
                    Math.min(batchSize, 32),
                    Math.floor(
                      device.chunkSize(
                        'scorePairModels',
                        batchSize,
                        Math.min(batchSize, 32),
                        scoringFloor,
                      ) / share,
                    ),
                  ),
            );
            yield* step({ ...settings, start, trials }, timed);
            timed = false;
            start += trials;
            if (
              settings.refit === 0 &&
              batch % (share > 1 ? SHARED_PROGRESS_INTERVAL : PROGRESS_INTERVAL) === 0 &&
              start < Math.min(settings.maximum, budget)
            ) {
              // The adaptive budget shrank with the best model so far: read it, so that a
              // finished search submits no more batches. The next batch runs when the loop
              // goes on, so its scoring is timed.
              yield [
                'snapshotPairModelBudget',
                () => snapshotPairModelBudget({ state, modelProgress }, 1),
                async () => {
                  budget = (await device.read(modelProgress, 1))[0]!;
                },
              ] as const;
              timed = true;
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
            yield* step({ ...settings, trials: 1, refit: 1 }, true);
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
                selectBindings(r, [
                  'coords',
                  'mask',
                  'poses',
                  'cloud',
                  'info',
                  'control',
                  'pixels',
                ]),
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
                  'coords',
                ]),
                blocks,
              );
            },
          ] as const;
          yield [
            'beginTwoViewRefinement',
            () => {
              beginTwoViewRefinement(selectBindings(r, ['info', 'metrics', 'mask', 'control']), 1);
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
      })();
      const finish = async (report: ShadePlanReport): Promise<PairEstimationResult> => {
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
          // The reads share one round trip: the plan is complete before the first returns.
          const [bestRead, info, points, valid, maskRead, stateRead] = await Promise.all([
            recovery ? undefined : device.read(best, 21),
            recovery ? device.read(recovery.info, 72) : undefined,
            recovery ? device.read(recovery.points, 3 * n) : undefined,
            recovery ? device.read(recovery.flags, n) : undefined,
            device.read(mask, n),
            device.read(state, 8),
          ]);
          values = bestRead;
          recoveryResult = recovery
            ? { info: info!.slice(), points: points!.slice(), valid: Uint8Array.from(valid!) }
            : undefined;
          flags = maskRead;
          metadata = stateRead;
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
      };
      return { steps, finish, release };
    } catch (error) {
      successful = false;
      release();
      throw error;
    }
  };
}

/** A pair estimation that has its buffers: the steps to plan, then the reads that finish it. */
interface PairEstimationTask {
  readonly steps: Iterable<GpuStep>;
  readonly finish: (report: ShadePlanReport) => Promise<PairEstimationResult>;
  /** Give the buffers back (after a success) or destroy them (otherwise). */
  readonly release: () => void;
}

/** @internal Sampling, proposals, scoring, stopping and all inlier refits stay resident. */
export function estimatePairGpu(
  device: ShadeDevice,
  input: Float64Array,
  first: Camera,
  second: Camera,
  options: PairEstimationOptions,
): Promise<PairEstimationResult> {
  const start = checkPairEstimation(device, input, first, second, options, 1);
  return device.enqueue(async () => {
    const task = start();
    try {
      return await task.finish(await device.planSteps(task.steps));
    } finally {
      task.release();
    }
  });
}

/**
 * @internal Estimate many pairs in one job. The pairs share the plans: their steps run in rounds,
 * so a round holds one plan of the steps between two reads and one plan of the scoring chunks, and
 * each read of the rounds is awaited once for all. A pair returns what `estimatePairGpu` returns
 * for the same input: the search stops where a sequential search would (`selectPairModel`), so
 * the number of trials a batch holds is a performance choice. `report` of each result is the
 * report of the shared plans. A device with bounded submissions runs the pairs one by one.
 */
export function estimatePairsGpu(
  device: ShadeDevice,
  requests: readonly {
    readonly input: Float64Array;
    readonly first: Camera;
    readonly second: Camera;
    readonly options: PairEstimationOptions;
  }[],
): Promise<PairEstimationResult[]> {
  if (!requests.length) return Promise.resolve([]);
  const share = requests.length;
  if (device.boundedSubmissions)
    return requests.reduce<Promise<PairEstimationResult[]>>(
      (chain, r) =>
        chain.then(async (all) => [
          ...all,
          await estimatePairGpu(device, r.input, r.first, r.second, r.options),
        ]),
      Promise.resolve([]),
    );
  const starts = requests.map((r) =>
    checkPairEstimation(device, r.input, r.first, r.second, r.options, share),
  );
  return device.enqueue(async () => {
    const tasks: PairEstimationTask[] = [];
    try {
      for (const start of starts) tasks.push(start());
      const report = await device.planStepsMany(tasks.map((t) => t.steps));
      const settled = await Promise.allSettled(tasks.map((t) => t.finish(report)));
      const failed = settled.find((r) => r.status === 'rejected');
      if (failed) throw (failed as PromiseRejectedResult).reason;
      return settled.map((r) => (r as PromiseFulfilledResult<PairEstimationResult>).value);
    } finally {
      for (const task of tasks) task.release();
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
