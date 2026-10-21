/**
 * The page side of a multi-video run (`docs/multi-video.md`, 6.1 to 6.4): the options a run takes
 * from the query string, the rows of the upload list, the tags of a video, the n × n matrix and
 * the run-record fields. This module has no DOM and no worker globals, so a unit test runs it in
 * Node. The worker's part is `multi-video-job.ts`.
 */
import type {
  RunRecordComponent,
  RunRecordMultiVideoOptions,
  RunRecordVideo,
  RunRecordViewGraph,
} from './run-record.ts';

/** The defaults of 2.2, 3.3 and 3.4. The record names them in `options.multiVideo`. */
export const MULTI_VIDEO_DEFAULTS = {
  descriptor: 'vlad-32-v1',
  k: 10,
  minMatches: 30,
  minEdgeInliers: 25,
  /** 2.2: the cap per video. */
  maxKeyframesPerVideo: 60,
  /** 10: the cap of a run. */
  maxKeyframes: 800,
} as const;

export interface MultiVideoQuery {
  /** `?canonical=0` assigns view ids by arrival order (7.1, "prove the instrument"). */
  readonly canonical: boolean;
  /** `?maxKeyframes=` caps the keyframes of each video (default 60). */
  readonly maxKeyframesPerVideo: number;
  /** `?maxFeatures=` caps the SIFT features of a keyframe; undefined takes the lab's cap. */
  readonly maxFeatures?: number;
}

export function multiVideoQuery(search = ''): MultiVideoQuery {
  const q = new URLSearchParams(search);
  const integer = (name: string, min: number, max: number) => {
    const raw = q.get(name);
    const v = raw === null ? NaN : Number(raw);
    return Number.isInteger(v) && v >= min && v <= max ? v : undefined;
  };
  return {
    canonical: q.get('canonical') !== '0',
    maxKeyframesPerVideo:
      integer('maxKeyframes', 2, 400) ?? MULTI_VIDEO_DEFAULTS.maxKeyframesPerVideo,
    maxFeatures: integer('maxFeatures', 64, 16384),
  };
}

/**
 * The positions to keep when a video has more than `cap` keyframes (2.2): the first and the last
 * keyframe, and the others thinned evenly in time. Positions are in time order.
 */
export function thinnedPositions(count: number, cap: number): number[] {
  if (!Number.isInteger(count) || count < 0) throw new RangeError('count');
  if (!Number.isInteger(cap) || cap < 2) throw new RangeError('cap');
  if (count <= cap) return Array.from({ length: count }, (_, i) => i);
  return Array.from({ length: cap }, (_, i) => Math.round((i * (count - 1)) / (cap - 1)));
}

/** The tags of a video row (6.3). */
export type VideoTag = RunRecordVideo['tags'][number];

/** The colors of the components, one hue each. The viewer marks the cameras with them. */
export const COMPONENT_COLORS = [
  '#1677ff',
  '#fa8c16',
  '#52c41a',
  '#eb2f96',
  '#722ed1',
  '#13c2c2',
  '#faad14',
  '#a0d911',
] as const;
export const componentColor = (component: number): string =>
  COMPONENT_COLORS[
    ((component % COMPONENT_COLORS.length) + COMPONENT_COLORS.length) % COMPONENT_COLORS.length
  ]!;

/** What the page knows of one video of the upload list. */
export interface VideoRowState {
  readonly videoId: string;
  readonly keyframes: number;
  readonly registeredViews: number;
  /** Set when the file has the `videoId` of an earlier row (2.1): it is dropped. */
  readonly duplicateOf?: string;
}

/** One row of the graph result, as the worker's `graph` message carries it. */
export interface GraphVideo {
  readonly videoId: string;
  readonly tag: 'connected' | 'not connected' | 'alone';
  readonly component: number;
  readonly share: number;
}

/** A row marked `duplicate` when an earlier row has its `videoId` (2.1). */
export function markDuplicates<T extends { videoId: string }>(
  rows: readonly T[],
): (T & { duplicateOf?: string })[] {
  const first = new Map<string, number>();
  return rows.map((row, i) => {
    const earlier = first.get(row.videoId);
    if (earlier === undefined) {
      first.set(row.videoId, i);
      return { ...row };
    }
    return { ...row, duplicateOf: rows[earlier]!.videoId };
  });
}

/**
 * The tags of one video after a run (6.3): `component` (with its number), `alone`,
 * `not connected` and `rotation`. `rotation` marks a video with at least three verified edges
 * inside it and none that can seed a map: every one was homography-dominated (5, pure-rotation
 * clip).
 */
export function tagsOf(
  video: GraphVideo | undefined,
  internal: { edges: number; seedEligible: number },
): VideoTag[] {
  if (!video) return [];
  const tags: VideoTag[] = [];
  if (video.tag === 'connected') tags.push('component');
  else if (video.tag === 'alone') tags.push('alone');
  else tags.push('not connected');
  if (internal.edges >= 3 && internal.seedEligible === 0) tags.push('rotation');
  return tags;
}

/** The accepted edges as the job summarizes them: one row for each pair of videos. */
export interface VideoPairEdges {
  readonly a: string;
  readonly b: string;
  readonly edges: number;
  readonly inliers: number;
}

/**
 * The n × n matrix of 6.3: cell (i, j) holds the cross-video inliers between the videos `order[i]`
 * and `order[j]`. The diagonal is 0 and the matrix is symmetric.
 */
export function inlierMatrix(
  order: readonly string[],
  pairs: readonly VideoPairEdges[],
): number[][] {
  const index = new Map(order.map((id, i) => [id, i]));
  const m = order.map(() => order.map(() => 0));
  for (const p of pairs) {
    const i = index.get(p.a),
      j = index.get(p.b);
    if (i === undefined || j === undefined || i === j) continue;
    m[i]![j]! += p.inliers;
    m[j]![i]! += p.inliers;
  }
  return m;
}

/** The shade of a matrix cell in [0, 1]: the cell over the largest off-diagonal cell. */
export function matrixShade(value: number, max: number): number {
  return max > 0 && value > 0 ? Math.min(1, Math.max(0.12, value / max)) : 0;
}

export interface MultiVideoSummary {
  readonly videos: readonly (Omit<RunRecordVideo, 'tags' | 'component' | 'registeredViews'> & {
    readonly tags: readonly VideoTag[];
    readonly component: number | null;
    readonly registeredViews: number;
  })[];
  readonly components: readonly RunRecordComponent[];
  readonly crossVideoEdges: { edges: number; inliers: number };
  readonly viewGraph: RunRecordViewGraph;
  readonly notConnected: readonly string[];
}

/** The `options.multiVideo` block of a run. */
export function multiVideoOptions(input: {
  query: MultiVideoQuery;
  minEdgeInliers?: number;
  minMatches?: number;
  k?: number;
}): RunRecordMultiVideoOptions {
  return {
    descriptor: MULTI_VIDEO_DEFAULTS.descriptor,
    k: input.k ?? MULTI_VIDEO_DEFAULTS.k,
    minMatches: input.minMatches ?? MULTI_VIDEO_DEFAULTS.minMatches,
    minEdgeInliers: input.minEdgeInliers ?? MULTI_VIDEO_DEFAULTS.minEdgeInliers,
    maxKeyframesPerVideo: input.query.maxKeyframesPerVideo,
    growth: 'incremental',
    canonical: input.query.canonical,
    // The WebGPU bundle adjustment has no per-group focal kernel (Amendment 4): one shared focal.
    focalPerVideo: false,
  };
}

/** What the page saw of one video while it fed the frames to the job. */
export interface VideoFacts {
  /** The frames sent to the job, kept or not. */
  readonly frames: number;
  /** The analysis size of the first frame. */
  readonly width: number;
  readonly height: number;
  /** The keyframes after the cap, known when the video ended. */
  readonly keyframes?: number;
}

/**
 * The `input.videos` list of a run that failed before the worker's summary (6.4). It lists each
 * video that got a `videoId`, with what the page saw and the views that the maps held at the
 * failure. The tags stay empty, since the tags need the finished graph.
 * @param scenes the map of each component: the final one, or else the last partial one.
 */
export function partialVideos(input: {
  rows: readonly {
    videoId?: string;
    sampleId?: string;
    durationSeconds?: number;
    focal: number;
    keyframes?: number;
  }[];
  facts: ReadonlyMap<string, VideoFacts>;
  scenes: ReadonlyMap<number, { views: readonly { videoId?: string }[] }>;
}): MultiVideoSummary['videos'][number][] {
  const registered = new Map<string, number>();
  const componentOf = new Map<string, number>();
  for (const [component, scene] of input.scenes)
    for (const view of scene.views) {
      if (!view.videoId) continue;
      registered.set(view.videoId, (registered.get(view.videoId) ?? 0) + 1);
      if (!componentOf.has(view.videoId)) componentOf.set(view.videoId, component);
    }
  const out: MultiVideoSummary['videos'][number][] = [];
  for (const row of input.rows) {
    if (!row.videoId || !/^[0-9a-f]{16}$/.test(row.videoId)) continue;
    const fact = input.facts.get(row.videoId);
    const views = registered.get(row.videoId) ?? 0;
    const keyframes = Math.max(fact?.keyframes ?? row.keyframes ?? 0, views);
    out.push({
      videoId: row.videoId,
      label: row.sampleId ?? null,
      durationSeconds: row.durationSeconds ?? 0,
      keyframes,
      frames: Math.max(fact?.frames ?? 0, keyframes),
      width: fact?.width ?? 0,
      height: fact?.height ?? 0,
      registeredViews: views,
      focalRatio: row.focal,
      component: componentOf.get(row.videoId) ?? null,
      tags: [],
    });
  }
  return out.slice(0, 64);
}
