/**
 * The mapper on one component of the view graph (docs/multi-video.md 4.2 and step 5).
 *
 * The component's keyframes and verified edges go to `incrementalSteps` in one call. The edges
 * arrive as `preVerified` rows, so the mapper runs no second verification. The inputs are sorted
 * by canonical view id, and every tie in the mapper breaks by that id (4.4).
 */
import {
  incrementalSteps,
  executeReconstructionJob,
  type IncrementalOptions,
  type IncrementalResult,
  type PreVerifiedPair,
} from '../mapper/mapper.ts';
import type { ImageFeatures, PairMatches } from '../../contracts/scene.ts';
import type { ViewGraph, ViewGraphEdge, ViewGraphKeyframe } from './view-graph.ts';

export interface ComponentOptions extends IncrementalOptions {
  /**
   * One focal group for each video (2.3, item 4). It matters when `refineSharedFocal` is on.
   * Default true. The WebGPU bundle adjustment has no per-group form yet, so a GPU caller sets
   * this to false.
   */
  readonly focalPerVideo?: boolean;
}

/** The reusable verdict of one edge of the view graph. */
export function preVerifiedOf(edge: ViewGraphEdge): PreVerifiedPair {
  return {
    verified: {
      first: edge.first,
      second: edge.second,
      matches: edge.matches,
      distances: edge.distances,
    },
    verification: edge.diagnostic,
    geometry: edge.geometry,
  };
}

/**
 * The focal group of each keyframe: the rank of its `videoId` in the sorted list of the
 * `videoId`s of the given keyframes. The rank does not depend on the input order.
 */
export function focalGroupsOf(keyframes: readonly ViewGraphKeyframe[]): Map<number, number> {
  const videos = [...new Set(keyframes.map((k) => k.videoId))].sort((a, b) =>
    a < b ? -1 : a > b ? 1 : 0,
  );
  const rank = new Map(videos.map((v, i) => [v, i]));
  return new Map(keyframes.map((k) => [k.viewId, rank.get(k.videoId)!]));
}

/** The inputs of `incrementalSteps` for one component, sorted by view id. */
export function componentInput(
  graph: ViewGraph,
  keyframes: readonly ViewGraphKeyframe[],
  componentIndex: number,
  options: ComponentOptions = {},
): { images: ImageFeatures[]; pairs: PairMatches[]; options: IncrementalOptions } {
  const component = graph.components[componentIndex];
  if (!component) throw new RangeError('Unknown component');
  if (!component.reconstructed) throw new RangeError('The component is too small to reconstruct');
  const byViewId = new Map(keyframes.map((k) => [k.viewId, k]));
  const members = new Set(component.viewIds);
  const frames = [...members]
    .sort((a, b) => a - b)
    .map((id) => {
      const frame = byViewId.get(id);
      if (!frame) throw new RangeError('A component view has no keyframe');
      return frame;
    });
  const edges = graph.edges.filter((e) => members.has(e.first) && members.has(e.second));
  const { focalPerVideo = true, ...mapper } = options;
  return {
    images: frames.map((f) => ({ id: f.viewId, camera: f.camera, keypoints: f.keypoints })),
    pairs: edges.map((e) => preVerifiedOf(e).verified),
    options: {
      ...mapper,
      preVerified: edges.map(preVerifiedOf),
      focalGroups: focalPerVideo ? focalGroupsOf(frames) : undefined,
    },
  };
}

/** Reconstruct one component on the CPU. A device driver runs `componentInput` instead. */
export function reconstructComponent(
  graph: ViewGraph,
  keyframes: readonly ViewGraphKeyframe[],
  componentIndex: number,
  options: ComponentOptions = {},
): IncrementalResult {
  const input = componentInput(graph, keyframes, componentIndex, options);
  const machine = incrementalSteps(input.images, input.pairs, input.options);
  let step = machine.next();
  while (!step.done) step = machine.next(executeReconstructionJob(step.value));
  return step.value;
}
