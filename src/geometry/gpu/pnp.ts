import type { Resident } from 'typeshade/runtime';
import { GeometryEstimationError } from '../errors.ts';
import type { Camera } from '../camera.ts';
import { validateCamera } from '../camera.ts';
import type { PnPOptions, PnPResult } from '../pnp.ts';
import { ShadeDevice, type ShadePlanReport } from '../../device/gpu/device.ts';
import { solveP3p } from './p3p.shade.ts';
import {
  normalizePnP,
  initializePnP,
  samplePnP,
  scorePnP,
  scorePnPOracle,
  selectPnP,
  planPnPBatch,
  snapshotPnPBudget,
  consensusMask,
  beginPoseRefinement,
  poseNormals,
  poseNormalsOracle,
  poseStep,
  poseTrialCosts,
  acceptPoseStep,
  snapshotPoseActivity,
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
  /** The batches of trials the search selected from; the GPU skips a batch past the budget. */
  readonly hypothesisBatches: number;
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

/**
 * The batches of a PnP search between two reads of the scheduling word. The GPU decides each
 * batch's work through the indirect workgroup counts (TypeShade change 0075), so the host reads
 * nothing between batches. It reads the adaptive budget after every fourth batch to leave the
 * search early: at most three batches past the budget are submitted, and each dispatches zero
 * workgroups but its one-workgroup `planPnPBatch`.
 */
const PROGRESS_INTERVAL = 4;
/** The trials a scoring batch holds at least, to fill the device (`docs/gpu-throughput.md`, item 2). */
const PNP_SCORING_FLOOR = 512;
/** The offsets, in words, of the workgroup counts of `pnp.shade.ts` (`batchCounts`). */
const TRIAL_COUNT = 0,
  SCORE_COUNT = 3,
  SCORE_ORACLE_COUNT = 6,
  SELECT_COUNT = 9;

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
    requestedBatch = options.batchSize ?? 512;
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
  // XYZ/pixels/normalized observations/mask, normal blocks, trial costs, persistent solve state,
  // the scheduling word and the twelve words of indirect workgroup counts.
  const fixedBytes = n * 108 + normalCount * 44 * 8 + blocks * 8 + 488 + 4 + 48;
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
        state = integer(9),
        best = wide(16),
        working = wide(12),
        candidate = wide(12),
        activity = integer(1),
        mask = integer(n),
        lm = wide(4),
        normalBlocks = wide(normalCount * 44),
        trialCosts = wide(blocks),
        progress = integer(1),
        batchCounts = integer(12);
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
      const report = await device.planSteps(
        (function* () {
          yield [
            'normalizePnP',
            () => {
              normalizePnP({ control, xyz, pixels, observations }, Math.ceil(n / 64));
            },
          ] as const;
          yield [
            'initializePnP',
            () => {
              initializePnP({ control, state, best, observations }, 1);
            },
          ] as const;
          // A search batch decides its work on the GPU (TypeShade change 0075): `planPnPBatch`
          // writes the workgroup counts of the batch's sampling, solve, scoring and selection,
          // zero for a batch past the adaptive budget, and those entries dispatch with
          // `{ indirect: batchCounts, offset }`, so the host reads nothing between batches but
          // the scheduling word, after every `PROGRESS_INTERVAL` batches.
          //
          // The scoring of a batch is a plan of its own (key `scorePnP`). `timed` says the host
          // knows it runs: the first batch, and the first batch after a read of the budget that
          // showed the search goes on. Only such a scoring is timed for the rate of a trial.
          // Another one runs alone, untimed: the GPU may have set its count to zero, and a plan
          // that dispatches nothing would teach the device a rate far too fast.
          let budget = maximum,
            timed = true;
          for (let start = 0, batch = 1; start < Math.min(maximum, budget); batch++) {
            // The selection stops where a sequential search would (`selectPnP`), so the batch is
            // a performance choice: as many trials as the time budget allows at the scoring rate
            // the device learned, at least the occupancy floor and at most `batchSize`. The CPU tier, which has
            // no watchdog and no idle lanes, scores 64 at a time.
            const trials = Math.min(
              maximum - start,
              device.backend === 'cpu'
                ? Math.min(batchSize, 64)
                : device.chunkSize(
                    'scorePnP',
                    batchSize,
                    Math.min(batchSize, 32),
                    Math.min(batchSize, PNP_SCORING_FLOOR),
                  ),
            );
            const batchControl = { ...baseControl, start, trials };
            yield [
              'planPnPBatch',
              () => {
                planPnPBatch({ control: batchControl, state, batchCounts }, 1);
              },
            ] as const;
            const trialCount = { indirect: batchCounts, offset: TRIAL_COUNT };
            yield [
              'samplePnP',
              () => {
                samplePnP({ control: batchControl, state, observations, problems }, trialCount);
              },
            ] as const;
            yield [
              'solveP3p',
              () => {
                solveP3p({ problems, poses, valid, counts, status }, trialCount);
              },
            ] as const;
            const bindings = { control: batchControl, observations, poses, valid, partials },
              chunk = {
                key: 'scorePnP',
                units: trials,
                minUnits: Math.min(batchSize, PNP_SCORING_FLOOR),
                ...(timed ? {} : { timed: false }),
              };
            if (device.backend === 'webgpu')
              yield [
                'scorePnP',
                () => {
                  scorePnP(bindings, { indirect: batchCounts, offset: SCORE_COUNT });
                },
                undefined,
                {
                  ...chunk,
                  // Zero workgroups: the pipeline is made before the first timed scoring.
                  warm: { id: 'pnp', run: () => scorePnP(bindings, 0) },
                },
              ] as const;
            else
              yield [
                'scorePnPOracle',
                () => {
                  scorePnPOracle(bindings, { indirect: batchCounts, offset: SCORE_ORACLE_COUNT });
                },
                undefined,
                chunk,
              ] as const;
            yield [
              'selectPnP',
              () => {
                selectPnP(
                  { control: batchControl, state, best, observations, poses, valid, partials },
                  { indirect: batchCounts, offset: SELECT_COUNT },
                );
              },
            ] as const;
            timed = false;
            start += trials;
            if (batch % PROGRESS_INTERVAL === 0 && start < Math.min(maximum, budget)) {
              // The adaptive budget shrank with the best pose so far: read it, so that a finished
              // search submits no more batches. The next batch runs when the loop goes on, so its
              // scoring is timed.
              yield [
                'snapshotPnPBudget',
                () => {
                  snapshotPnPBudget({ state, progress }, 1);
                },
                async () => {
                  budget = (await device.read(progress, 1))[0]!;
                },
              ] as const;
              timed = true;
            }
          }
          for (let pass = 0; pass < (refinement ? 3 : 0); pass++) {
            yield [
              'consensusMask',
              () => {
                consensusMask(
                  { control, state, mask, observations, ...poseBuffers },
                  Math.ceil(n / 64),
                );
              },
            ] as const;
            yield [
              'beginPoseRefinement',
              () => {
                beginPoseRefinement({ control, state, best, working, lm }, 1);
              },
            ] as const;
            let poseActive = true;
            const readPoseActivity = async () => {
              poseActive = (await device.read(activity, 1))[0] !== 0;
            };
            // No read here: the refinement is active at its start unless the consensus is too small,
            // and every kernel of an inactive iteration is a guarded no-op (`lm[1] === 0`). The first
            // read comes after four iterations (Amendment 8 of docs/multi-video.md).
            for (let iteration = 0; iteration < refinement && poseActive; iteration++) {
              const bindings = { control, mask, lm, observations, normalBlocks, ...poseBuffers };
              if (device.backend === 'webgpu')
                yield [
                  'poseNormals',
                  () => {
                    poseNormals(bindings, normalCount);
                  },
                ] as const;
              else
                yield [
                  'poseNormalsOracle',
                  () => {
                    poseNormalsOracle(bindings, Math.ceil(normalCount / 32));
                  },
                ] as const;
              yield [
                'poseStep',
                () => {
                  poseStep({ normalBlocks, lm, ...poseBuffers }, 1);
                },
              ] as const;
              yield [
                'poseTrialCosts',
                () => {
                  poseTrialCosts(
                    { control, mask, lm, observations, trialCosts, ...poseBuffers },
                    Math.ceil(blocks / 32),
                  );
                },
              ] as const;
              yield [
                'acceptPoseStep',
                () => {
                  acceptPoseStep({ lm, state, candidate, working, trialCosts }, 1);
                },
              ] as const;
              if (
                device.solverCheckpoints &&
                (iteration + 1) % 4 === 0 &&
                iteration + 1 < refinement
              )
                yield [
                  'snapshotPoseActivity',
                  () => {
                    snapshotPoseActivity({ lm, activity }, 1);
                  },
                  readPoseActivity,
                ] as const;
            }
            yield [
              'finishPoseRefinement',
              () => {
                finishPoseRefinement({ control, state, mask, observations, ...poseBuffers }, 1);
              },
            ] as const;
          }
          yield [
            'consensusMask',
            () => {
              consensusMask(
                { control, state, mask, observations, ...poseBuffers },
                Math.ceil(n / 64),
              );
            },
          ] as const;
          yield [
            'summarizePnP',
            () => {
              summarizePnP({ control, mask, observations, ...poseBuffers }, 1);
            },
          ] as const;
        })(),
      );
      // No intermediate candidate, mask or score reads occurred: only the scheduling words.
      const [summary, flags, metadata] = await Promise.all([
        device.read(best, 16),
        device.read(mask, n),
        device.read(state, 9),
      ]);
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
        hypothesisBatches: metadata[8]!,
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
