import { GeometryEstimationError } from '../errors.ts';
import type { Camera } from '../../contracts/camera.ts';
import type { ImagePairGeometry } from '../image-pair.ts';
import { resolveTwoViewInput, type TwoViewOptions, type TwoViewResult } from '../two-view.ts';
import { ShadeDevice, type ShadePlanReport } from '../../device/gpu/device.ts';
import { estimatePairGpu, RECOVERY_INFO } from './pair-estimation.ts';
import { RECOVERY_FAILURES, TwoViewFailure } from '../../contracts/two-view-failures.ts';

export interface GpuTwoViewOptions extends TwoViewOptions {
  /** Adaptive stopping is evaluated after each batch. Default: 32, bounded by device capacity. */
  readonly batchSize?: number;
}
export interface GpuConsensusExecution {
  readonly backend: 'webgpu' | 'cpu';
  readonly report: ShadePlanReport;
  readonly applicationUploadBytes: number;
  readonly hypothesisBatches: number;
  readonly hypothesesScored: number;
  /** All initialization dispatches, including proposals, refits and optional recovery. */
  readonly scoreDispatches: number;
  readonly scoreSubmissions: number;
  /** Application arrays only; excludes uniforms, runtime staging and shader compilation. */
  readonly applicationReadbackBytes: number;
  readonly cpuStages: readonly string[];
}
export interface GpuEssentialConsensus {
  readonly E: Float64Array;
  readonly inliers: Uint8Array;
  readonly inlierCount: number;
  /** Sum of min(Sampson² / threshold², 1), computed with TypeShade f64 arithmetic. */
  readonly msacCost: number;
  readonly ransacIterations: number;
  readonly execution: GpuConsensusExecution;
}
export interface GpuTwoViewResult extends TwoViewResult {
  readonly execution: GpuConsensusExecution;
}

/** GPU eight-point RANSAC, including normalization, sampling, SVD and consensus refits. */
export async function estimateEssentialRansacGpu(
  device: ShadeDevice,
  matchesPixels: Float64Array,
  K1: Camera,
  K2: Camera,
  options: GpuTwoViewOptions = {},
): Promise<GpuEssentialConsensus> {
  const pixels = matchesPixels.slice(),
    a = { ...K1 },
    b = { ...K2 };
  const controls = resolveTwoViewInput(pixels, a, b, options);
  const result = await estimatePairGpu(device, pixels, a, b, {
    kind: 0,
    iterations: controls.iterations,
    minIterations: controls.minIterations,
    threshold: controls.threshold,
    confidence: controls.confidence,
    seed: controls.seed,
    batchSize: options.batchSize,
  });
  if (result.count < 8) throw new GeometryEstimationError(TwoViewFailure.noGeometry);
  return {
    E: result.model,
    inliers: result.mask,
    inlierCount: result.count,
    msacCost: result.msac,
    ransacIterations: result.trials,
    execution: {
      ...result.execution,
      hypothesisBatches: result.batches,
      hypothesesScored: result.hypotheses,
      scoreDispatches: result.execution.report.dispatches,
      scoreSubmissions: result.execution.report.submissions,
    },
  };
}

/** Resident GPU consensus, cheirality, joint two-camera LM and final quality classification. */
export async function reconstructTwoViewGpu(
  device: ShadeDevice,
  matchesPixels: Float64Array,
  K1: Camera,
  K2: Camera,
  options: GpuTwoViewOptions = {},
): Promise<GpuTwoViewResult> {
  const pixels = matchesPixels.slice(),
    a = { ...K1 },
    b = { ...K2 },
    opts = { ...options };
  const controls = resolveTwoViewInput(pixels, a, b, opts);
  if (controls.refinementIterations > 100 || controls.huberPx > 1e6 || controls.maxError > 1e6)
    throw new RangeError('GPU two-view refinement exceeds supported bounds');
  const result = await estimatePairGpu(device, pixels, a, b, {
    kind: 0,
    iterations: controls.iterations,
    minIterations: controls.minIterations,
    threshold: controls.threshold,
    confidence: controls.confidence,
    seed: controls.seed,
    batchSize: opts.batchSize,
    recovery: {
      separateDirectionalObservations: controls.separateDirectionalObservations,
      iterations: controls.refinementIterations,
      huber: controls.huberPx,
      minAngle: controls.minParallax,
      maxError: controls.maxError,
    },
  });
  const recovered = result.recovery!,
    info = recovered.info;
  const reason = recoveryFailure(info[RECOVERY_INFO.status]!);
  if (reason) throw new GeometryEstimationError(reason);
  for (let i = 0; i < recovered.valid.length; i++)
    if (!recovered.valid[i]) recovered.points.fill(NaN, i * 3, i * 3 + 3);
  return {
    E: result.model,
    ...recoverySummary(info, result.trials),
    inliers: result.mask,
    points: recovered.points,
    validPoints: recovered.valid,
    execution: {
      ...result.execution,
      hypothesisBatches: result.batches,
      hypothesesScored: result.hypotheses,
      scoreDispatches: result.execution.report.dispatches,
      scoreSubmissions: result.execution.report.submissions,
    },
  };
}

/** @internal */
export function recoveryFailure(status: number): string | undefined {
  if (status === 0) return undefined;
  return RECOVERY_FAILURES[status] ?? TwoViewFailure.gpuRecovery;
}
/** @internal Device results only; this function performs no geometry computation. */
export function recoverySummary(info: Float64Array, trials: number): ImagePairGeometry {
  return {
    R: info.slice(0, 9),
    t: info.slice(9, 12),
    inlierCount: info[RECOVERY_INFO.inlierCount]!,
    positiveDepthCount: info[54]!,
    meanReprojectionErrorPx: info[55]!,
    diagnostics: {
      ransacIterations: trials,
      refinementApplied: !!info[52],
      ...(info[70] ? { directionCompatibleCount: info[66]! } : {}),
      medianParallaxDeg: info[53]!,
      meanReprojectionErrorBeforePx: info[45]!,
      meanReprojectionErrorAfterPx: info[44]!,
    },
  };
}
