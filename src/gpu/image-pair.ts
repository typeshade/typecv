import {
  resolveImagePairOptions,
  validateImagePairPixels,
  type ImagePairOptions,
  type ImagePairVerification,
} from '../geometry/image-pair.ts';
import type { Camera } from '../geometry/camera.ts';
import { finiteGpuArray, validateGpuCamera } from './pnp.ts';
import {
  estimatePairGpu,
  estimatePairGpuWavefront,
  type PairEstimationWaveRequest,
} from './pair-estimation.ts';
import { recoveryFailure, recoverySummary, type GpuConsensusExecution } from './two-view.ts';
import type { GpuGeometryExecution } from './pnp.ts';
import type { ShadeDevice } from './device.ts';

export interface GpuImagePairOptions extends ImagePairOptions {
  readonly batchSize?: number;
}
export interface GpuImagePairVerification extends ImagePairVerification {
  readonly execution: GpuConsensusExecution & {
    readonly homographyIterations: number;
  };
}
export interface GpuImagePairWaveRequest {
  readonly matches: Float64Array;
  readonly first: Camera;
  readonly second: Camera;
  readonly options?: GpuImagePairOptions;
}
export interface GpuImagePairWaveResult {
  readonly results: readonly GpuImagePairVerification[];
  readonly execution: GpuGeometryExecution;
  readonly waveSizes: readonly number[];
  readonly checkpointReadbacks: number;
  readonly pairCheckpointSnapshots: number;
}
interface PreparedImagePair {
  readonly pixels: Float64Array;
  readonly first: Camera;
  readonly second: Camera;
  readonly controls: ReturnType<typeof resolveImagePairOptions>;
  readonly batchSize?: number;
}
function prepareGpuImagePair(
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: GpuImagePairOptions,
): PreparedImagePair {
  validateImagePairPixels(matches);
  const pixels = matches.slice(),
    a = validateGpuCamera({ ...first }),
    b = validateGpuCamera({ ...second }),
    controls = resolveImagePairOptions(a, b, { ...options }),
    batchSize = options.batchSize;
  if (
    !finiteGpuArray(pixels) ||
    controls.iterations > 100000 ||
    controls.homographyIterations > 100000 ||
    controls.refinementIterations > 100 ||
    controls.huberPx < 1e-12 ||
    controls.huberPx > 1e6 ||
    controls.threshold < 1e-12 ||
    controls.threshold > 1e6 ||
    controls.maxError < 1e-12 ||
    controls.maxError > 1e6 ||
    controls.homographyThreshold < 1e-12 ||
    controls.homographyThreshold > 1e6 ||
    (batchSize !== undefined && (!Number.isSafeInteger(batchSize) || batchSize < 1))
  )
    throw new RangeError('Image pair exceeds the supported GPU geometry bounds');
  return { pixels, first: a, second: b, controls, batchSize };
}
function emptyPairVerification(device: ShadeDevice): GpuImagePairVerification {
  return {
    invertibleCount: 0,
    homographyInlierCount: 0,
    inlierRows: new Uint32Array(0),
    reason: 'Insufficient invertible correspondences',
    execution: {
      backend: device.backend,
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
      applicationUploadBytes: 0,
      applicationReadbackBytes: 0,
      cpuStages: [],
      hypothesisBatches: 0,
      hypothesesScored: 0,
      scoreDispatches: 0,
      scoreSubmissions: 0,
      homographyIterations: 0,
    },
  };
}
function finishPairVerification(
  result: Awaited<ReturnType<typeof estimatePairGpu>>,
): GpuImagePairVerification {
  const info = result.recovery!.info,
    details = result.verification!,
    reason =
      details.invertibleCount < 8
        ? 'Insufficient invertible correspondences'
        : recoveryFailure(info[56]!);
  const rows: number[] = [];
  if (!reason) for (let i = 0; i < result.mask.length; i++) if (result.mask[i]) rows.push(i);
  return {
    invertibleCount: details.invertibleCount,
    homographyInlierCount: details.invertibleCount < 8 ? 0 : details.homographyInlierCount,
    inlierRows: Uint32Array.from(rows),
    geometry: reason ? undefined : recoverySummary(info, result.trials),
    reason,
    execution: {
      ...result.execution,
      hypothesisBatches: result.batches,
      hypothesesScored: result.hypotheses,
      scoreDispatches: result.execution.report.dispatches,
      scoreSubmissions: result.execution.report.submissions,
      homographyIterations: details.homographyIterations,
    },
  };
}
function toPairEstimationRequest(pair: PreparedImagePair): PairEstimationWaveRequest {
  const { controls, pixels, first, second, batchSize } = pair;
  return {
    input: pixels,
    first,
    second,
    options: {
      kind: 0,
      iterations: controls.iterations,
      minIterations: controls.minIterations,
      threshold: controls.threshold,
      confidence: controls.confidence,
      seed: controls.seed,
      batchSize,
      recovery: {
        iterations: controls.refinementIterations,
        huber: controls.huberPx,
        minAngle: controls.minParallax,
        maxError: controls.maxError,
      },
      verification: {
        homographyThreshold: controls.homographyThreshold,
        homographyIterations: controls.homographyIterations,
      },
    },
  };
}

/** One resident pipeline: radial inversion, stable compaction, H/E fits and relative-pose checks. */
export async function verifyImagePairGpu(
  device: ShadeDevice,
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: GpuImagePairOptions = {},
): Promise<GpuImagePairVerification> {
  const pair = prepareGpuImagePair(matches, first, second, options);
  if (!pair.pixels.length) return device.enqueue(async () => emptyPairVerification(device));
  const request = toPairEstimationRequest(pair),
    result = await estimatePairGpu(
      device,
      request.input,
      request.first,
      request.second,
      request.options,
    );
  return finishPairVerification(result);
}

/** Verify several independent mapper edges while sharing adaptive RANSAC checkpoint reads. */
export async function verifyImagePairsGpuWavefront(
  device: ShadeDevice,
  requests: readonly GpuImagePairWaveRequest[],
): Promise<GpuImagePairWaveResult> {
  const prepared = requests.map((request) =>
    prepareGpuImagePair(request.matches, request.first, request.second, request.options ?? {}),
  );
  const active: number[] = [],
    estimates: PairEstimationWaveRequest[] = [];
  prepared.forEach((pair, index) => {
    if (pair.pixels.length) {
      active.push(index);
      estimates.push(toPairEstimationRequest(pair));
    }
  });
  if (!estimates.length)
    return {
      results: prepared.map(() => emptyPairVerification(device)),
      execution: {
        backend: device.backend,
        report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
        applicationUploadBytes: 0,
        applicationReadbackBytes: 0,
        cpuStages: [],
      },
      waveSizes: [],
      checkpointReadbacks: 0,
      pairCheckpointSnapshots: 0,
    };
  const wave = await estimatePairGpuWavefront(device, estimates),
    results = prepared.map(() => emptyPairVerification(device));
  wave.results.forEach((result, index) => {
    results[active[index]!] = finishPairVerification(result);
  });
  return {
    results,
    execution: wave.execution,
    waveSizes: wave.waveSizes,
    checkpointReadbacks: wave.checkpointReadbacks,
    pairCheckpointSnapshots: wave.pairCheckpointSnapshots,
  };
}
