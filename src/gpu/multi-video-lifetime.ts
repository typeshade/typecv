/** Private, consuming host ownership for the worker's multi-video pipeline. No pixel or feature
 * bytes are modified. The ordinary multi-video API retains its immutable input/result contract. */
import type { ViewGraphEdge, ViewGraphKeyframe } from '../reconstruction/view-graph.ts';
import { dequantizeDescriptors } from '../reconstruction/view-graph.ts';

// Keep this pure ownership helper independent of the GPU entry and its shader imports, so
// source-level ownership tests and core typechecking do not traverse the device runtime.
type OwnedFeatureInput = Pick<
  ViewGraphKeyframe,
  'videoId' | 'frameKey' | 'time' | 'camera' | 'keypoints' | 'global'
> & { readonly descriptors: Uint8Array };

const EMPTY_POINTS = new Float64Array(0);
const EMPTY_GLOBAL = new Float32Array(0);
type MutableFrame = Omit<ViewGraphKeyframe, 'keypoints' | 'global'> & {
  keypoints: Float64Array;
  global: Float32Array;
};

/** Unique backing allocations, including the full buffer of a borrowed subview. This measures
 * reachable typed-array roots owned here, not RSS, GPU storage, or references held by a caller. */
export function backingBytes(arrays: Iterable<ArrayBufferView>): number {
  const roots = new Set<ArrayBufferLike>();
  for (const array of arrays) roots.add(array.buffer);
  let bytes = 0;
  for (const root of roots) bytes += root.byteLength;
  return bytes;
}

export class OwnedMultiVideoFeatures {
  readonly keyframes: MutableFrame[];
  private readonly descriptors = new Map<number, Uint8Array>();
  constructor(input: OwnedFeatureInput[], ids: readonly number[]) {
    if (input.length !== ids.length) throw new RangeError('Keyframe ids must match the input');
    this.keyframes = input.map((frame, i) => {
      const viewId = ids[i]!;
      this.descriptors.set(viewId, frame.descriptors);
      const descriptors = this.descriptors;
      return {
        viewId,
        videoId: frame.videoId,
        frameKey: frame.frameKey,
        time: frame.time,
        camera: frame.camera,
        keypoints: frame.keypoints,
        global: frame.global,
        get descriptors() {
          const stored = descriptors.get(viewId);
          if (!stored) throw new Error('Released multi-video descriptors');
          return dequantizeDescriptors(stored);
        },
      };
    });
    // The array is exclusively owned. A caller retaining any input record/array retains that
    // allocation independently; the pipeline never mutates such an array's bytes.
    input.length = 0;
  }
  get bytes(): number {
    return backingBytes([
      ...this.descriptors.values(),
      ...this.keyframes.flatMap((frame) => [frame.keypoints, frame.global]),
    ]);
  }
  /** Graph retrieval/matching is complete. Mapper inputs read neither local nor global SIFT. */
  releaseDescriptors(): void {
    this.descriptors.clear();
    for (const frame of this.keyframes) frame.global = EMPTY_GLOBAL;
  }
  /** Final component callbacks have returned; no future mapper reads these feature positions. */
  releaseKeypoints(viewIds: readonly number[]): void {
    const done = new Set(viewIds);
    for (const frame of this.keyframes) if (done.has(frame.viewId)) frame.keypoints = EMPTY_POINTS;
  }
  clear(): void {
    this.releaseDescriptors();
    for (const frame of this.keyframes) frame.keypoints = EMPTY_POINTS;
  }
}

/** The accepted match rows, excluding small geometry/diagnostic records and object overhead. */
export function graphMatchBytes(edges: readonly ViewGraphEdge[]): number {
  return backingBytes(edges.flatMap((edge) => [edge.matches, edge.distances]));
}
