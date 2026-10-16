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
import type { GpuConsensusExecution } from './two-view.ts';
import { verifyImagePairGpu } from './image-pair.ts';

type GpuMapperOperation = 'pairVerification' | 'pnp' | 'triangulation' | 'observations' | 'bundle';
export interface GpuIncrementalOptions extends IncrementalOptions {
  readonly onProgress?: (event: {
    operation: ReconstructionJob['kind'];
    index: number;
    phase: 'start' | 'complete';
  }) => void;
}

export interface GpuIncrementalResult extends IncrementalResult {
  readonly execution: {
    readonly backend: 'webgpu' | 'cpu';
    readonly stages: readonly {
      readonly operation: GpuMapperOperation;
      /** Wall time including first-use pipeline compilation and numerical readback. */
      readonly elapsedMs: number;
      readonly execution: GpuGeometryExecution;
    }[];
    readonly pairConsensus: readonly GpuConsensusExecution[];
    readonly gpuBundleAdjustments: number;
    readonly cpuStages: readonly string[];
  };
}

/**
 * Incremental mapper with TypeShade numerical geometry throughout initialization,
 * registration, triangulation, filtering and joint BA. Graph/control remains host-side.
 * No device failure is converted to a CPU retry or an ordinary rejected view.
 */
export async function reconstructIncrementalGpu(
  device: ShadeDevice,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: GpuIncrementalOptions = {},
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
      operation: GpuMapperOperation;
      elapsedMs: number;
      execution: GpuGeometryExecution;
    }[] = [],
    pairConsensus: GpuConsensusExecution[] = [];
  let gpuBundleAdjustments = 0;
  let operationIndex = 0;
  const execute = async (job: ReconstructionJob): Promise<unknown> => {
    const started = performance.now();
    switch (job.kind) {
      case 'verifyPair': {
        const result = await verifyImagePairGpu(device, ...job.args);
        stages.push({
          operation: 'pairVerification',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        if (result.geometry) pairConsensus.push(result.execution);
        return result;
      }
      case 'pnp': {
        const result = await estimatePosePnPGpu(device, ...job.args);
        stages.push({
          operation: 'pnp',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result;
      }
      case 'triangulation': {
        const result = await triangulateTracksGpu(device, job.observations, job.options);
        stages.push({
          operation: 'triangulation',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result.tracks;
      }
      case 'observations': {
        const result = await evaluateObservationsGpu(device, job.model, job.maxError);
        stages.push({
          operation: 'observations',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result;
      }
      case 'bundle': {
        const result = await bundleAdjustGpu(device, ...job.args);
        gpuBundleAdjustments++;
        stages.push({
          operation: 'bundle',
          elapsedMs: performance.now() - started,
          execution: result.execution,
        });
        return result;
      }
    }
  };
  const machine = incrementalSteps(views, edges, controls);
  let step = machine.next();
  while (!step.done) {
    let value: unknown;
    const event = { operation: step.value.kind, index: ++operationIndex };
    try {
      options.onProgress?.({ ...event, phase: 'start' });
      value = await execute(step.value);
      options.onProgress?.({ ...event, phase: 'complete' });
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
      cpuStages: ['track graph, result indexing, data packing and mapper control'],
    },
  };
}
