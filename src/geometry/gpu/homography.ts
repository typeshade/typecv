import type { ShadeDevice } from '../../device/gpu/device.ts';
import type { GpuGeometryExecution } from './execution.ts';
import { estimatePairGpu } from './pair-estimation.ts';

export interface GpuHomographyOptions {
  readonly thresholdPx?: number;
  readonly iterations?: number;
  readonly minIterations?: number;
  readonly seed?: number;
  readonly confidence?: number;
  readonly batchSize?: number;
}
export interface GpuHomographyResult {
  readonly H: Float64Array;
  readonly inliers: Uint8Array;
  readonly inlierCount: number;
  readonly iterations: number;
  readonly execution: GpuGeometryExecution;
}
/** Normalized DLT, symmetric transfer RANSAC and consensus refits, all in TypeShade. */
export async function estimateHomographyGpu(
  device: ShadeDevice,
  matches: Float64Array,
  options: GpuHomographyOptions = {},
): Promise<GpuHomographyResult | undefined> {
  const iterations = options.iterations ?? 500;
  const unit = { fx: 1, fy: 1, cx: 0, cy: 0 };
  const result = await estimatePairGpu(device, matches, unit, unit, {
    kind: 1,
    iterations,
    minIterations: options.minIterations ?? Math.min(32, iterations),
    threshold: options.thresholdPx ?? 2,
    confidence: options.confidence ?? 0.999,
    seed: options.seed ?? 17,
    batchSize: options.batchSize,
  });
  if (result.count < 4) return undefined;
  return {
    H: result.model,
    inliers: result.mask,
    inlierCount: result.count,
    iterations: result.trials,
    execution: result.execution,
  };
}
