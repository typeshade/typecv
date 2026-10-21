/**
 * The multi-video pipeline on a device (docs/multi-video.md, step 6): canonical view ids (2.4),
 * the view graph (section 3), the components (4.1) and the mapper on each component (4.2).
 *
 * The lab's worker extracts the features of each keyframe as the frames arrive. This module takes
 * the finished keyframes of all videos and does everything that needs the whole run. It is
 * exported from the internal entry only (`src/gpu/internal.ts`).
 */
import type { Camera } from '../geometry/camera.ts';
import { assignViewIds } from '../reconstruction/identity.ts';
import { componentInput } from '../reconstruction/component-mapper.ts';
import { ReconstructionInitializationError } from '../reconstruction/mapper.ts';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import {
  buildViewGraph,
  dequantizeDescriptors,
  quantizeDescriptors,
  type ViewGraph,
  type ViewGraphKeyframe,
  type ViewGraphOptions,
} from '../reconstruction/view-graph.ts';
import type { ShadeDevice } from './device.ts';
import { gpuViewGraphBackend } from './view-graph.ts';
import { globalDescriptorGpu } from './global-descriptor-gpu.ts';
import type { GpuResidentSift } from './sift.ts';
import { reconstructIncrementalGpu, type GpuIncrementalOptions } from './mapper.ts';

/** A keyframe as the worker holds it after feature extraction. The input order is arbitrary. */
export interface MultiVideoKeyframe {
  readonly videoId: string;
  readonly frameKey: string;
  /** Time in the video, in seconds. */
  readonly time: number;
  readonly camera: Camera;
  /** Packed x, y pixels, one pair for each feature. */
  readonly keypoints: Float64Array;
  /** RootSIFT as 8-bit values (3.5): `keypoints.length / 2` rows of 128. */
  readonly descriptors: Uint8Array;
  /** The L2-normalized global descriptor (3.2). */
  readonly global: Float32Array;
}

export interface MultiVideoOptions {
  /**
   * `true` (the default): `viewId` is the content rank of 2.4. `false` is the debug switch
   * `?canonical=0`: `viewId` is the position in the input list, so the pair seeds, the candidate
   * ties and the registration ties follow the arrival order. It exists to show that the shuffle
   * gate of 7.1 can fail.
   */
  readonly canonical?: boolean;
  readonly graph?: ViewGraphOptions;
  /**
   * Options of the mapper on each component. The WebGPU bundle adjustment has no per-group focal
   * kernel (Amendment 4), so the run uses `focalPerVideo: false`: one shared focal.
   */
  readonly mapper?: Omit<GpuIncrementalOptions, 'preVerified' | 'focalGroups'>;
  /** The stage that starts: `viewGraph`, then `component <index>`. */
  readonly onProgress?: (stage: string) => void;
}

export interface MultiVideoComponent {
  readonly index: number;
  readonly viewIds: readonly number[];
  readonly videoIds: readonly string[];
  /** False for a component under `minComponentViews`: it is not reconstructed. */
  readonly reconstructed: boolean;
  readonly reconstruction?: SparseReconstruction;
  readonly meanReprojectionErrorPx?: number;
  readonly registrationOrder?: readonly number[];
  /** The mapper could not initialize this component. */
  readonly error?: string;
  readonly ms: number;
}

export interface MultiVideoResult {
  /** The keyframes with their view ids, in input order. */
  readonly keyframes: readonly ViewGraphKeyframe[];
  readonly graph: ViewGraph;
  readonly components: readonly MultiVideoComponent[];
  readonly canonical: boolean;
  readonly timing: {
    readonly globalDescriptorMs?: number;
    readonly reconstructionMs: number;
  };
}

/** A keyframe whose float descriptors are decoded from the 8-bit store when the graph reads them. */
function graphKeyframe(k: MultiVideoKeyframe, viewId: number): ViewGraphKeyframe {
  const stored = k.descriptors;
  return {
    viewId,
    videoId: k.videoId,
    frameKey: k.frameKey,
    time: k.time,
    camera: k.camera,
    keypoints: k.keypoints,
    global: k.global,
    get descriptors() {
      return dequantizeDescriptors(stored);
    },
  };
}

export async function reconstructMultiVideoGpu(
  device: ShadeDevice,
  input: readonly MultiVideoKeyframe[],
  options: MultiVideoOptions = {},
): Promise<MultiVideoResult> {
  const canonical = options.canonical ?? true;
  const ids = canonical ? assignViewIds(input) : input.map((_, i) => i);
  const keyframes = input.map((k, i) => graphKeyframe(k, ids[i]!));
  options.onProgress?.('viewGraph');
  const graph = await buildViewGraph(keyframes, gpuViewGraphBackend(device), options.graph);
  const started = performance.now();
  const components: MultiVideoComponent[] = [];
  for (const c of graph.components) {
    const base = { index: c.index, viewIds: c.viewIds, videoIds: c.videoIds };
    if (!c.reconstructed) {
      components.push({ ...base, reconstructed: false, ms: 0 });
      continue;
    }
    options.onProgress?.('component ' + c.index);
    const t = performance.now();
    const prepared = componentInput(graph, keyframes, c.index, {
      ...options.mapper,
      focalPerVideo: false,
    });
    try {
      const result = await reconstructIncrementalGpu(device, prepared.images, prepared.pairs, {
        ...options.mapper,
        ...prepared.options,
      } as GpuIncrementalOptions);
      components.push({
        ...base,
        reconstructed: true,
        reconstruction: result.reconstruction,
        meanReprojectionErrorPx: result.diagnostics.meanReprojectionErrorPx,
        registrationOrder: result.diagnostics.registrationOrder,
        ms: performance.now() - t,
      });
    } catch (error) {
      if (!(error instanceof ReconstructionInitializationError)) throw error;
      components.push({
        ...base,
        reconstructed: true,
        error: error.message,
        ms: performance.now() - t,
      });
    }
  }
  return {
    keyframes,
    graph,
    components,
    canonical,
    timing: { reconstructionMs: performance.now() - started },
  };
}

/**
 * The 8-bit descriptors (3.5) and the global descriptor (3.2) of one keyframe, from its resident
 * SIFT result. The descriptors are read back once. `vocabulary` is `rootsift-32-v1` for VLAD.
 */
export async function describeKeyframeGpu(
  device: ShadeDevice,
  sift: GpuResidentSift,
  vocabulary: Float32Array,
): Promise<{ descriptors: Uint8Array; global: Float32Array }> {
  const n = sift.count * 128;
  const host = n
    ? Float32Array.from(
        (await device.enqueue(() => device.read(sift.descriptors.descriptors, n))).subarray(0, n),
      )
    : new Float32Array(0);
  return {
    descriptors: quantizeDescriptors(host),
    global: await globalDescriptorGpu(device, host, vocabulary),
  };
}
