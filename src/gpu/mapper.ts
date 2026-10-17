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
import { validateFeatures } from '../reconstruction/tracks.ts';
import {
  matchFloatDescriptorsGpu,
  type GpuResidentFloatDescriptors,
  type GpuFloatMatchOptions,
  type GpuFloatMatches,
} from './float-matching.ts';

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

/** Keypoint coordinates use the camera's pixel coordinate system and descriptor row order. */
export interface GpuFloatImageFeatures extends ImageFeatures {
  readonly descriptors: GpuResidentFloatDescriptors;
}

export interface GpuFloatImagePair {
  readonly first: number;
  readonly second: number;
}

export interface GpuFloatIncrementalOptions extends GpuIncrementalOptions {
  readonly matching?: GpuFloatMatchOptions;
  readonly onMatchProgress?: (event: {
    readonly first: number;
    readonly second: number;
    readonly index: number;
    readonly total: number;
    readonly phase: 'start' | 'complete';
  }) => void;
}

export interface GpuFloatIncrementalResult extends GpuIncrementalResult {
  readonly matching: {
    /** Matching wall time; excludes extraction and subsequent mapper operations. */
    readonly elapsedMs: number;
    readonly pairs: readonly {
      readonly first: number;
      readonly second: number;
      readonly matches: number;
      readonly report: GpuFloatMatches['report'];
      readonly execution: GpuFloatMatches['execution'];
    }[];
  };
}

/**
 * Opt-in float-descriptor adapter for externally extracted features (for example ALIKED).
 * Borrows descriptors from the same TypeShade runtime/device until this promise settles:
 * callers must not mutate, recycle or dispose them meanwhile. Descriptors are neither read
 * to the host nor disposed here. Host coordinates, cameras and controls are snapshotted
 * before matching starts. Explicit image pairs avoid an implicit quadratic view graph.
 * All resulting edges undergo the existing mapper's geometric verification and filtering.
 */
export async function reconstructIncrementalFloatGpu(
  device: ShadeDevice,
  images: readonly GpuFloatImageFeatures[],
  pairs: readonly GpuFloatImagePair[],
  options: GpuFloatIncrementalOptions = {},
): Promise<GpuFloatIncrementalResult> {
  const views = images.map((image) => ({
    id: image.id,
    camera: { ...image.camera },
    keypoints: image.keypoints.slice(),
    descriptors: { ...image.descriptors },
  }));
  const edges = pairs.map((pair) => ({ first: pair.first, second: pair.second }));
  const controls = {
    ...options,
    matching: options.matching ? { ...options.matching } : undefined,
    initialPair: options.initialPair ? ([...options.initialPair] as const) : undefined,
  };
  const byId = validateFeatures(views, []);
  if (views.length < 2) throw new RangeError('At least two float-feature views are required');
  for (const view of views) {
    const { count, dimensions, elements } = view.descriptors;
    if (
      !Number.isSafeInteger(count) ||
      count < 0 ||
      count !== view.keypoints.length / 2 ||
      !Number.isSafeInteger(dimensions) ||
      dimensions < 1 ||
      !Number.isSafeInteger(elements) ||
      elements !== count * dimensions ||
      !view.descriptors.descriptors ||
      typeof view.descriptors.descriptors.read !== 'function' ||
      typeof view.descriptors.descriptors.write !== 'function'
    )
      throw new RangeError('Float descriptor rows must correspond to keypoints');
  }
  const features = new Map(views.map((view) => [view.id, view]));
  const seen = new Set<string>();
  for (const pair of edges) {
    if (!byId.has(pair.first) || !byId.has(pair.second) || pair.first === pair.second)
      throw new RangeError('Invalid float-feature image pair');
    const key = [pair.first, pair.second].sort((a, b) => a - b).join(':');
    if (seen.has(key)) throw new RangeError('Duplicate float-feature image pair');
    seen.add(key);
    const first = features.get(pair.first)!,
      second = features.get(pair.second)!;
    if (first.descriptors.dimensions !== second.descriptors.dimensions)
      throw new RangeError('Float descriptor dimensions must agree within each pair');
  }
  const matched: PairMatches[] = [],
    diagnostics: GpuFloatIncrementalResult['matching']['pairs'][number][] = [];
  const started = performance.now();
  for (let index = 0; index < edges.length; index++) {
    const pair = edges[index]!,
      event = { ...pair, index: index + 1, total: edges.length };
    controls.onMatchProgress?.({ ...event, phase: 'start' });
    const result = await matchFloatDescriptorsGpu(
      device,
      features.get(pair.first)!.descriptors,
      features.get(pair.second)!.descriptors,
      controls.matching,
    );
    matched.push({ ...pair, matches: result.matches, distances: result.distances });
    diagnostics.push({
      ...pair,
      matches: result.matches.length / 2,
      report: result.report,
      execution: result.execution,
    });
    controls.onMatchProgress?.({ ...event, phase: 'complete' });
  }
  const elapsedMs = performance.now() - started;
  const result = await reconstructIncrementalGpu(device, views, matched, controls);
  return { ...result, matching: { elapsedMs, pairs: diagnostics } };
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
