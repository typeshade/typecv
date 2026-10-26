import {
  resolveImagePairOptions,
  validateImagePairPixels,
  type ImagePairOptions,
  type ImagePairVerification,
  CHEIRALITY_RETRIES,
  CHEIRALITY_RETRY_SEED_STEP,
  isCheiralityFailure,
} from '../image-pair.ts';
import type { Camera } from '../camera.ts';
import { finiteGpuArray, validateGpuCamera } from './bounds.ts';
import { estimatePairGpu, estimatePairsGpu } from './pair-estimation.ts';
import { recoveryFailure, recoverySummary, type GpuConsensusExecution } from './two-view.ts';
import type { ShadeDevice } from '../../device/gpu/device.ts';

export interface GpuImagePairOptions extends ImagePairOptions {
  readonly batchSize?: number;
}
export interface GpuImagePairVerification extends ImagePairVerification {
  readonly execution: GpuConsensusExecution & {
    readonly homographyIterations: number;
  };
}

/** One pair to verify: the arguments of `verifyImagePairGpu`. */
export interface GpuImagePairRequest {
  readonly matches: Float64Array;
  readonly first: Camera;
  readonly second: Camera;
  readonly options?: GpuImagePairOptions;
}

type Checked = {
  pixels: Float64Array;
  a: Camera;
  b: Camera;
  controls: ReturnType<typeof resolveImagePairOptions>;
  batchSize: number | undefined;
};

/** The validation and the resolved controls of one pair. */
function checkRequest(r: GpuImagePairRequest): Checked {
  validateImagePairPixels(r.matches);
  const options = r.options ?? {};
  const pixels = r.matches.slice(),
    a = validateGpuCamera({ ...r.first }),
    b = validateGpuCamera({ ...r.second }),
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
  return { pixels, a, b, controls, batchSize };
}

const estimateRequest = (c: Checked, seed: number) => ({
  input: c.pixels,
  first: c.a,
  second: c.b,
  options: {
    kind: 0 as const,
    iterations: c.controls.iterations,
    minIterations: c.controls.minIterations,
    threshold: c.controls.threshold,
    confidence: c.controls.confidence,
    seed,
    batchSize: c.batchSize,
    recovery: {
      separateDirectionalObservations: c.controls.separateDirectionalObservations,
      iterations: c.controls.refinementIterations,
      huber: c.controls.huberPx,
      minAngle: c.controls.minParallax,
      maxError: c.controls.maxError,
    },
    verification: {
      homographyThreshold: c.controls.homographyThreshold,
      homographyIterations: c.controls.homographyIterations,
    },
  },
});

type Estimate = Awaited<ReturnType<typeof estimatePairGpu>>;

const failureOf = (result: Estimate) =>
  result.verification!.invertibleCount < 8
    ? 'Insufficient invertible correspondences'
    : recoveryFailure(result.recovery!.info[56]!);

function finishRequest(result: Estimate): GpuImagePairVerification {
  const info = result.recovery!.info,
    details = result.verification!,
    reason = failureOf(result);
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

const emptyVerification = (device: ShadeDevice): GpuImagePairVerification => ({
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
});

/** One resident pipeline: radial inversion, stable compaction, H/E fits and relative-pose checks. */
export async function verifyImagePairGpu(
  device: ShadeDevice,
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: GpuImagePairOptions = {},
): Promise<GpuImagePairVerification> {
  const c = checkRequest({ matches, first, second, options });
  if (!c.pixels.length) return device.enqueue(async () => emptyVerification(device));
  const estimate = (seed: number) => {
    const r = estimateRequest(c, seed);
    return estimatePairGpu(device, r.input, r.first, r.second, r.options);
  };
  let result = await estimate(c.controls.seed);
  // The same bounded retry as the CPU verifier: a near-planar pair's best-scoring essential
  // matrix can fail cheirality on one seed and pass on the next.
  for (
    let attempt = 1;
    attempt <= CHEIRALITY_RETRIES && isCheiralityFailure(failureOf(result));
    attempt++
  )
    result = await estimate(c.controls.seed + attempt * CHEIRALITY_RETRY_SEED_STEP);
  return finishRequest(result);
}

/**
 * Verify many pairs in one job (docs/multi-video.md, Amendment 8). The pairs share the plans and
 * the reads, `maxPairs` at a time (`estimatePairsGpu`). Each result equals the result of
 * `verifyImagePairGpu` on the same arguments, except `execution.report`, which is the report of
 * the shared plans. A pair that fails the cheirality test retries with the next seed in a later
 * job, as the single form does.
 */
export async function verifyImagePairsGpu(
  device: ShadeDevice,
  requests: readonly GpuImagePairRequest[],
  maxPairs = 16,
): Promise<GpuImagePairVerification[]> {
  if (!Number.isSafeInteger(maxPairs) || maxPairs < 1)
    throw new RangeError('maxPairs must be a positive integer');
  const checked = requests.map(checkRequest);
  const out: GpuImagePairVerification[] = new Array(requests.length);
  const results = new Array<Estimate>(requests.length);
  const todo: number[] = [];
  checked.forEach((c, i) => {
    if (c.pixels.length) todo.push(i);
    else out[i] = emptyVerification(device);
  });
  let pending = todo.map((i) => ({ index: i, attempt: 0 }));
  while (pending.length) {
    const retry: typeof pending = [];
    for (let k = 0; k < pending.length; k += maxPairs) {
      const batch = pending.slice(k, k + maxPairs);
      const done = await estimatePairsGpu(
        device,
        batch.map(({ index, attempt }) =>
          estimateRequest(
            checked[index]!,
            checked[index]!.controls.seed + attempt * CHEIRALITY_RETRY_SEED_STEP,
          ),
        ),
      );
      batch.forEach(({ index, attempt }, j) => {
        results[index] = done[j]!;
        if (attempt < CHEIRALITY_RETRIES && isCheiralityFailure(failureOf(done[j]!)))
          retry.push({ index, attempt: attempt + 1 });
      });
    }
    pending = retry;
  }
  for (const i of todo) out[i] = finishRequest(results[i]!);
  return out;
}
