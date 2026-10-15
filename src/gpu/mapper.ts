import {
  incrementalSteps,
  ReconstructionOperationError,
  type IncrementalOptions,
  type IncrementalResult,
  type ReconstructionJob,
} from '../reconstruction/mapper.ts';
import type { ImageFeatures, PairMatches } from '../reconstruction/types.ts';
import { GeometryEstimationError } from '../geometry/errors.ts';
import { ShadeDevice } from './device.ts';
import { estimatePosePnPGpu, type GpuGeometryExecution } from './pnp.ts';
import { triangulateTracksGpu } from './triangulation.ts';
import { evaluateObservationsGpu } from './observations.ts';
import { bundleAdjustGpu } from './bundle.ts';
import { reconstructTwoViewGpu, type GpuConsensusExecution } from './two-view.ts';

export interface GpuIncrementalResult extends IncrementalResult {
  readonly execution: {
    readonly backend: 'webgpu' | 'cpu';
    readonly stages: readonly {
      readonly operation: 'pnp' | 'triangulation' | 'observations' | 'bundle';
      readonly execution: GpuGeometryExecution;
    }[];
    readonly pairConsensus: readonly GpuConsensusExecution[];
    readonly gpuBundleAdjustments: number;
    readonly cpuStages: readonly string[];
  };
}

/**
 * Hybrid incremental mapper: GPU PnP, track triangulation, filtering and joint BA.
 * Pair initialization still uses the hybrid two-view path.
 * No device failure is converted to a CPU retry or an ordinary rejected view.
 */
export async function reconstructIncrementalGpu(
  device: ShadeDevice,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions = {},
): Promise<GpuIncrementalResult> {
  // The driver can suspend between operations, so capture all mutable caller data first.
  const views = images.map((v) => ({
    ...v,
    camera: { ...v.camera },
    keypoints: v.keypoints.slice(),
  }));
  const edges = pairs.map((p) => ({
    ...p,
    matches: p.matches.slice(),
    distances: p.distances?.slice(),
  }));
  const controls = {
    ...options,
    initialPair: options.initialPair ? ([...options.initialPair] as const) : undefined,
  };
  const stages: {
      operation: 'pnp' | 'triangulation' | 'observations' | 'bundle';
      execution: GpuGeometryExecution;
    }[] = [],
    pairConsensus: GpuConsensusExecution[] = [];
  let gpuBundleAdjustments = 0;
  const execute = async (job: ReconstructionJob): Promise<unknown> => {
    switch (job.kind) {
      case 'pnp': {
        const result = await estimatePosePnPGpu(device, ...job.args);
        stages.push({ operation: 'pnp', execution: result.execution });
        return result;
      }
      case 'triangulation': {
        const result = await triangulateTracksGpu(device, job.observations, job.options);
        stages.push({ operation: 'triangulation', execution: result.execution });
        return result.tracks;
      }
      case 'observations': {
        const result = await evaluateObservationsGpu(device, job.model, job.maxError);
        stages.push({ operation: 'observations', execution: result.execution });
        return result;
      }
      case 'twoView': {
        const [pixels, a, b, opts] = job.args;
        // Pair coordinates have already been undistorted by the common mapper.
        const first = { fx: a.fx, fy: a.fy, cx: a.cx, cy: a.cy },
          second = { fx: b.fx, fy: b.fy, cx: b.cx, cy: b.cy };
        const result = await reconstructTwoViewGpu(device, pixels, first, second, opts);
        pairConsensus.push(result.execution);
        return result;
      }
      case 'bundle': {
        const result = await bundleAdjustGpu(device, ...job.args);
        gpuBundleAdjustments++;
        stages.push({ operation: 'bundle', execution: result.execution });
        return result;
      }
    }
  };
  const machine = incrementalSteps(views, edges, controls);
  let step = machine.next();
  while (!step.done) {
    let value: unknown;
    try {
      value = await execute(step.value);
    } catch (error) {
      step = machine.throw(
        error instanceof GeometryEstimationError ? error : new ReconstructionOperationError(error),
      );
      continue;
    }
    step = machine.next(value);
  }
  return {
    ...step.value,
    execution: {
      backend: device.backend,
      stages,
      pairConsensus,
      gpuBundleAdjustments,
      cpuStages: [
        'pair undistortion',
        'homography verification',
        'eight-point proposal/refit',
        'two-view pose/point recovery',
        'track graph and mapper control',
      ],
    },
  };
}
