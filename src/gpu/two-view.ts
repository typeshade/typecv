import { sample, seededRandom } from '../geometry/numerics.ts';
import {
  estimateEssential,
  normalizePixels,
  recoverTwoViewGeometry,
  resolveTwoViewInput,
  type CameraIntrinsics,
  type TwoViewOptions,
  type TwoViewResult,
} from '../sfm/two-view.ts';
import { ShadeDevice } from './device.ts';
import { EpipolarSession, type EpipolarScore } from './epipolar.ts';

export interface GpuTwoViewOptions extends TwoViewOptions {
  /** Adaptive stopping is evaluated after each batch. Default: device capacity. */
  readonly batchSize?: number;
}
export interface GpuConsensusExecution {
  readonly backend: 'webgpu' | 'cpu';
  readonly hypothesisBatches: number;
  readonly hypothesesScored: number;
  /** Includes consensus/refit scoring; excludes the one-time normalization plan. */
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
  /** Sum of min(Sampson² / threshold², 1), computed with f32 device arithmetic. */
  readonly msacCost: number;
  readonly ransacIterations: number;
  readonly execution: GpuConsensusExecution;
}
export interface GpuTwoViewResult extends TwoViewResult {
  readonly execution: GpuConsensusExecution;
}

/**
 * Hybrid RANSAC: f64 eight-point proposals, device-resident f32 dense scoring,
 * hierarchical reduction and winner selection. No automatic backend fallback.
 */
export async function estimateEssentialRansacGpu(
  device: ShadeDevice,
  matchesPixels: Float64Array,
  K1: CameraIntrinsics,
  K2: CameraIntrinsics,
  options: GpuTwoViewOptions = {},
): Promise<GpuEssentialConsensus> {
  // Snapshot mutable caller inputs before the first await.
  const pixels = matchesPixels.slice(),
    a = { ...K1 },
    b = { ...K2 };
  const controls = resolveTwoViewInput(pixels, a, b, options);
  const requestedBatchSize = options.batchSize;
  if (
    requestedBatchSize !== undefined &&
    (!Number.isSafeInteger(requestedBatchSize) || requestedBatchSize < 1)
  )
    throw new RangeError('batchSize must be a positive safe integer');
  const points = normalizePixels(pixels, a, b),
    n = points.length / 4;
  const rng = seededRandom(controls.seed);
  const session = await EpipolarSession.create(device, pixels, a, b);
  let batches = 0,
    hypotheses = 0,
    dispatches = 0,
    submissions = 0,
    readbacks = 0;
  const score = async (models: Float64Array, readMask = false): Promise<EpipolarScore> => {
    const result = await session.score(models, controls.threshold, { readMask });
    dispatches += result.report.dispatches;
    submissions += result.report.submissions;
    readbacks += 12 + (readMask ? n * 4 : 0);
    return result;
  };
  try {
    const batchSize = Math.min(requestedBatchSize ?? session.capacity, session.capacity);
    let budget = controls.iterations,
      trials = 0;
    let bestE: Float64Array | undefined,
      bestCount = 0,
      bestCost = Infinity;
    while (trials < budget) {
      const attempted = Math.min(batchSize, budget - trials),
        proposals: Float64Array[] = [];
      for (let i = 0; i < attempted; i++) {
        trials++;
        try {
          proposals.push(estimateEssential(points, sample(n, 8, rng)));
        } catch {
          /* Rejected minimal samples still consume the hard trial cap. */
        }
      }
      if (!proposals.length) continue;
      const packed = new Float64Array(proposals.length * 9);
      proposals.forEach((E, i) => packed.set(E, i * 9));
      const result = await score(packed);
      batches++;
      hypotheses += proposals.length;
      if (
        result.inlierCount > bestCount ||
        (result.inlierCount === bestCount && result.msacCost < bestCost)
      ) {
        bestE = proposals[result.bestIndex]!;
        bestCount = result.inlierCount;
        bestCost = result.msacCost;
        const probability = (bestCount / n) ** 8;
        const required =
          probability >= 1
            ? 1
            : Math.ceil(Math.log1p(-controls.confidence) / Math.log1p(-probability));
        budget = Math.min(budget, Math.max(controls.minIterations, required));
      }
    }
    if (!bestE || bestCount < 8) throw new Error('Could not find a valid two-view geometry');
    let selected = await score(bestE, true);
    // Only winner masks cross the host boundary, for the small f64 refit solver.
    for (let pass = 0; pass < 4; pass++) {
      const indices: number[] = [];
      for (let i = 0; i < n; i++) if (selected.mask![i]) indices.push(i);
      let refined: Float64Array;
      try {
        refined = estimateEssential(points, indices);
      } catch {
        break;
      }
      const candidate = await score(refined, true);
      if (
        candidate.inlierCount < selected.inlierCount ||
        (candidate.inlierCount === selected.inlierCount && candidate.msacCost >= selected.msacCost)
      )
        break;
      bestE = refined;
      selected = candidate;
    }
    return {
      E: bestE,
      inliers: selected.mask!,
      inlierCount: selected.inlierCount,
      msacCost: selected.msacCost,
      ransacIterations: trials,
      execution: {
        backend: device.backend,
        hypothesisBatches: batches,
        hypothesesScored: hypotheses,
        scoreDispatches: dispatches,
        scoreSubmissions: submissions,
        applicationReadbackBytes: readbacks,
        cpuStages: ['eight-point proposals', 'adaptive stopping', 'inlier refit'],
      },
    };
  } finally {
    await session.dispose();
  }
}

/** GPU consensus followed by the shared f64 pose/point recovery and refinement. */
export async function reconstructTwoViewGpu(
  device: ShadeDevice,
  matchesPixels: Float64Array,
  K1: CameraIntrinsics,
  K2: CameraIntrinsics,
  options: GpuTwoViewOptions = {},
): Promise<GpuTwoViewResult> {
  const pixels = matchesPixels.slice(),
    a = { ...K1 },
    b = { ...K2 },
    opts = { ...options };
  const controls = resolveTwoViewInput(pixels, a, b, opts);
  const consensus = await estimateEssentialRansacGpu(device, pixels, a, b, opts);
  const result = recoverTwoViewGeometry(normalizePixels(pixels, a, b), a, b, controls, {
    E: consensus.E,
    mask: consensus.inliers,
    count: consensus.inlierCount,
    trials: consensus.ransacIterations,
  });
  return {
    ...result,
    execution: {
      ...consensus.execution,
      cpuStages: [
        ...consensus.execution.cpuStages,
        'pose recovery',
        'triangulation',
        'nonlinear pose/point refinement',
        'final consensus classification',
      ],
    },
  };
}
