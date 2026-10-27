/**
 * The `multiVideo` job of the SfM worker (`docs/multi-video.md`, 6.2). The worker file only wires
 * messages to this class, so a unit test runs the job in Node with a fake feature front end.
 *
 * Messages in: `video` (the start of a video), `frame`, `endVideo` and `finish`. Messages out,
 * through `post`: `frameComplete`, `features`, `videoComplete`, `graph`, `model` and `result`.
 * The messages carry data, not text (6.2).
 */
import { packLandmarks, type PackedScene } from './sfm-scene.ts';
import { PhaseGate } from './keyframe-gate.ts';
import { tooFewFeatures } from './lab-features.ts';
import { frameCamera } from './lab-matching.ts';
import {
  createDenseJob,
  denseFailure,
  denseGpuTimestampFields,
  endDense,
  runDense,
  type DenseFusion,
  type DenseImage,
  type DenseJob,
  type DenseRunContext,
} from './dense-run.ts';
import { frameKeyOf } from '../../../src/reconstruction/identity.ts';
import { reportGpuMs } from './run-record.ts';
import { VideoActivityGraph } from './multi-video-activity.ts';
import type { ViewGraphActivity } from '../../../src/reconstruction/view-graph.ts';
import { backingBytes } from '../../../src/gpu/multi-video-lifetime.ts';
import {
  MULTI_VIDEO_DEFAULTS,
  inlierMatrix,
  tagsOf,
  thinnedPositions,
  type MultiVideoQuery,
  type MultiVideoSummary,
  type VideoPairEdges,
} from './multi-video.ts';
import { LAB_OPTIONS } from './lab-options.ts';

/** A decoded or sampled analysis frame, as the page posts it. */
export interface JobFrame {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
  sourceWidth?: number;
  sourceHeight?: number;
  image?: string;
  /** The final frame of an uploaded video: the phase gate always keeps it. */
  last?: boolean;
  /** Known intrinsics in analysis pixels. */
  camera?: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
}

export interface JobVideo {
  /** The content hash of 2.4, computed by the page. */
  videoId: string;
  /** The sample id, or null for an upload. */
  label: string | null;
  durationSeconds: number;
  /** Focal length over image width (2.3). */
  focalRatio: number;
}

/** The part of the internal runtime entry (`/runtime/internal/gpu.js`) that the job calls. */
export interface JobApi {
  extractSiftGpuResident(
    device: any,
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
    options: { maxFeatures: number; firstOctave: -1 | 0; clahe: boolean },
  ): Promise<
    {
      count: number;
      readKeypoints(): Promise<Float64Array>;
      dispose(): Promise<void>;
    } & Record<string, any>
  >;
  describeKeyframeGpu(
    device: any,
    sift: any,
    vocabulary: Float32Array,
  ): Promise<{ descriptors: Uint8Array; global: Float32Array }>;
  reconstructMultiVideoGpu(device: any, keyframes: any[], options: any): Promise<any>;
  /** Consumes its private array and retires feature/match roots after each final reader. */
  reconstructMultiVideoOwnedGpu?(device: any, keyframes: any[], options: any): Promise<any>;
  /** The dense stage (Amendment 9): present when the run asked for it. */
  reconstructDenseGpu?(
    device: any,
    scene: any,
    images: Map<number, any>,
    options: any,
  ): Promise<any>;
  cleanDenseCloudGpu?(device: any, points: any, options?: any): Promise<any>;
  meshDenseCloudGpu?(device: any, points: any): Promise<any>;
}

/** What a run with the dense MVS needs of the worker (`docs/multi-video.md`, Amendment 9). */
export interface JobDenseOptions {
  /** Clean the fused cloud of a component (`docs/dense-mvs.md` 3.12). */
  clean: boolean;
  /** Mesh the cleaned cloud of a component (3.13). */
  mesh: boolean;
  /** The fusion (3.10 or 10.4). Default `DENSE_LAB_FUSION`. */
  fusion?: DenseFusion;
  /** The MVS options, `LAB_OPTIONS.dense` of `lib/lab-options.ts`. */
  options: Readonly<Record<string, unknown>>;
  /** The options of `cleanDenseCloudGpu`. */
  cleanOptions?: unknown;
  /** The kernel that started last on the dense device, for a failure. */
  lastKernel: () => string;
}

export interface JobOptions {
  query: MultiVideoQuery;
  maxFeatures: number;
  firstOctave: -1 | 0;
  clahe?: boolean;
  /** The mapper's fixed controls: the lab preset (`LAB_MAPPER`, `src/sfm/config/lab-preset.ts`). */
  mapper: Record<string, unknown>;
  /** The graph controls that the lab changes from the defaults of 3.3 and 3.4. */
  graph?: Record<string, unknown>;
  /** The interval of the partial maps, in ms. Default 500. */
  snapshotIntervalMs?: number;
  /** Set for a run with the dense MVS: the job keeps each keyframe's RGBA (Amendment 9). */
  dense?: JobDenseOptions;
}

interface Keyframe {
  videoId: string;
  frameKey: string;
  time: number;
  frameId: number;
  camera: { fx: number; fy: number; cx: number; cy: number; k1?: number; k2?: number };
  keypoints: Float64Array;
  descriptors: Uint8Array;
  global: Float32Array;
  /** RGB bytes under each keypoint, for the point cloud. */
  colors: Uint8Array;
  width: number;
  height: number;
  sourceWidth?: number;
  sourceHeight?: number;
  image?: string;
  /** The position in the time order of its video, after the cap. */
  sequence: number;
  /** The RGBA at the analysis size, kept only for a run with the dense MVS. */
  rgba?: Uint8ClampedArray;
}

/** A reconstructed component that waits for the dense stage. */
interface DenseComponent {
  index: number;
  model: { views: any[]; landmarks: any[] };
}

interface VideoState {
  info: JobVideo;
  gate: PhaseGate;
  keyframes: Keyframe[];
  frames: number;
  /** The analysis size of the first frame; the frames of one video share it. */
  width: number;
  height: number;
  skipped: number;
  dropped: number;
  ended: boolean;
}

function sampleColors(frame: JobFrame, points: Float64Array): Uint8Array {
  const colors = new Uint8Array((points.length / 2) * 3);
  for (let i = 0; i < points.length / 2; i++) {
    const x = Math.max(0, Math.min(frame.width - 1, Math.round(points[i * 2]!)));
    const y = Math.max(0, Math.min(frame.height - 1, Math.round(points[i * 2 + 1]!)));
    const offset = (y * frame.width + x) * 4;
    for (let c = 0; c < 3; c++) colors[i * 3 + c] = frame.pixels[offset + c]!;
  }
  return colors;
}

/** Map snapshots go to the page at most this often (the single-video worker's interval). */
export const SNAPSHOT_INTERVAL_MS = LAB_OPTIONS.snapshotIntervalMs;

interface Snapshot {
  component: number;
  registered: number;
  reconstruction: unknown;
}

/**
 * Pass the latest snapshot to `send` at most once in `intervalMs`. A snapshot that arrives too
 * early waits and goes out when the interval ends, unless a later one replaces it. `cancel`
 * drops the one that waits.
 */
export class Snapshots {
  private last = -Infinity;
  private pending: Snapshot | undefined;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private readonly intervalMs: number;
  private readonly send: (snapshot: Snapshot) => void;
  constructor(intervalMs: number, send: (snapshot: Snapshot) => void) {
    this.intervalMs = intervalMs;
    this.send = send;
  }
  offer(snapshot: Snapshot): void {
    const wait = this.intervalMs - (performance.now() - this.last);
    if (wait <= 0) {
      this.cancel();
      this.last = performance.now();
      this.send(snapshot);
      return;
    }
    this.pending = snapshot;
    this.timer ??= setTimeout(() => {
      this.timer = undefined;
      const waiting = this.pending;
      this.pending = undefined;
      if (waiting) {
        this.last = performance.now();
        this.send(waiting);
      }
    }, wait);
  }
  cancel(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.pending = undefined;
  }
}

export class MultiVideoJob {
  private readonly videos = new Map<string, VideoState>();
  /** The videos in the order they arrived: the list order of the page. */
  private readonly order: string[] = [];
  private stageMs = { features: 0, globalDescriptor: 0 };
  /** The GPU time of the two frame stages in ms, from the device's per-kernel table. */
  private stageGpuMs = { features: 0, globalDescriptor: 0 };
  private featureCalls = 0;
  private readonly featureCounts: number[] = [];

  private readonly api: JobApi;
  private readonly device: any;
  private readonly vocabulary: Float32Array;
  private readonly options: JobOptions;
  private readonly post: (message: Record<string, unknown>, transfer?: Transferable[]) => void;
  /** Only pending models and registered pixels survive the sparse-to-dense handoff. */
  private denseComponents: DenseComponent[] = [];
  private denseInfo = new Map<number, DenseImage>();
  private denseCurrent:
    { job: DenseJob; context: DenseRunContext; model: DenseComponent['model'] } | undefined;
  private denseCancelled = false;
  private denseFailed: { message: string; reason: string } | undefined;

  constructor(
    api: JobApi,
    device: any,
    vocabulary: Float32Array,
    options: JobOptions,
    post: (message: Record<string, unknown>, transfer?: Transferable[]) => void,
  ) {
    this.api = api;
    this.device = device;
    this.vocabulary = vocabulary;
    this.options = options;
    this.post = post;
  }

  /** The start of one video (6.2): the job makes its phase gate. */
  video(info: JobVideo): void {
    if (this.videos.has(info.videoId)) throw new RangeError('Duplicate videoId ' + info.videoId);
    this.videos.set(info.videoId, {
      info,
      gate: new PhaseGate(),
      keyframes: [],
      frames: 0,
      width: 0,
      height: 0,
      skipped: 0,
      dropped: 0,
      ended: false,
    });
    this.order.push(info.videoId);
  }

  /** One frame: the gate, then SIFT, the 8-bit store and the global descriptor of a keyframe. */
  async frame(videoId: string, frame: JobFrame): Promise<void> {
    const video = this.videos.get(videoId);
    if (!video || video.ended) throw new RangeError('Unknown or finished video');
    const started = performance.now();
    video.frames++;
    if (video.frames === 1) {
      video.width = frame.width;
      video.height = frame.height;
    }
    const total = [...this.videos.values()].reduce((n, v) => n + v.keyframes.length, 0);
    if (total >= MULTI_VIDEO_DEFAULTS.maxKeyframes)
      throw new RangeError(
        'The run reached ' +
          MULTI_VIDEO_DEFAULTS.maxKeyframes +
          ' keyframes. Remove a video or lower maxKeyframes.',
      );
    const decision = video.gate.decide(
      frame.pixels,
      frame.width,
      frame.height,
      frame.time,
      frame.last === true,
    );
    const done = (extra: Record<string, unknown>) =>
      this.post({
        type: 'frameComplete',
        videoId,
        frameId: frame.id,
        time: frame.time,
        elapsedMs: performance.now() - started,
        ...extra,
      });
    if (decision.action === 'skip') {
      video.gate.discard();
      video.skipped++;
      done({ keyframe: false, skipped: 'Not a keyframe: ' + decision.reason });
      return;
    }
    // A weak correlation peak (`fallback`) leaves the legacy rule to decide in the single-video
    // path. The multi-video job has no legacy rule: it keeps the frame.
    const frameKey = await frameKeyOf(frame.pixels);
    const t0 = performance.now();
    const g0 = this.gpuNs();
    const sift = await this.api.extractSiftGpuResident(
      this.device,
      frame.pixels,
      frame.width,
      frame.height,
      {
        maxFeatures: this.options.maxFeatures,
        firstOctave: this.options.firstOctave,
        clahe: this.options.clahe === true,
      },
    );
    this.stageMs.features += performance.now() - t0;
    this.stageGpuMs.features += (this.gpuNs() - g0) / 1e6;
    this.featureCalls++;
    try {
      if (tooFewFeatures(sift.count)) {
        video.gate.discard();
        video.skipped++;
        done({ keyframe: false, skipped: 'Too few detected features' });
        return;
      }
      const keypoints = await sift.readKeypoints();
      const t1 = performance.now();
      const g1 = this.gpuNs();
      const described = await this.api.describeKeyframeGpu(this.device, sift, this.vocabulary);
      this.stageMs.globalDescriptor += performance.now() - t1;
      this.stageGpuMs.globalDescriptor += (this.gpuNs() - g1) / 1e6;
      video.gate.accept();
      this.featureCounts.push(sift.count);
      video.keyframes.push({
        videoId,
        frameKey,
        time: frame.time,
        frameId: frame.id,
        camera: frameCamera(frame, video.info.focalRatio),
        keypoints,
        descriptors: described.descriptors,
        global: described.global,
        colors: sampleColors(frame, keypoints),
        width: frame.width,
        height: frame.height,
        sourceWidth: frame.sourceWidth,
        sourceHeight: frame.sourceHeight,
        image: frame.image,
        sequence: 0,
        rgba: this.options.dense ? frame.pixels : undefined,
      });
      this.post({
        type: 'features',
        videoId,
        frameId: frame.id,
        time: frame.time,
        points: keypoints,
      });
      done({ keyframe: true, features: sift.count });
    } finally {
      await sift.dispose();
    }
  }

  /**
   * The end of a video. Above the cap of 2.2 the job keeps the first and the last keyframe and
   * thins the others evenly in time. SIFT already ran for the dropped ones: a video above the cap
   * pays for features it drops, and the job holds no pixels.
   */
  endVideo(videoId: string): void {
    const video = this.videos.get(videoId);
    if (!video) throw new RangeError('Unknown video');
    const cap = this.options.query.maxKeyframesPerVideo;
    const sorted = [...video.keyframes].sort((a, b) => a.time - b.time);
    const keep = new Set(thinnedPositions(sorted.length, cap));
    video.dropped = sorted.length - keep.size;
    video.keyframes = sorted.filter((_, i) => keep.has(i));
    video.keyframes.forEach((k, i) => (k.sequence = i));
    video.ended = true;
    this.post({
      type: 'videoComplete',
      videoId,
      frames: video.frames,
      skipped: video.skipped,
      keyframes: video.keyframes.length,
      dropped: video.dropped,
      keptFrameIds: video.keyframes.map((frame) => frame.frameId),
    });
  }

  /** The finish of 6.2: ids, view graph, components, the mapper on each, the models. */
  async finish(): Promise<void> {
    const input = this.order.flatMap((id) => this.videos.get(id)!.keyframes);
    const inputFeatureBytes = () =>
      backingBytes(
        input.flatMap((frame) => [frame.keypoints, frame.descriptors, frame.global, frame.colors]),
      );
    const featuresBeforeSfMBytes = inputFeatureBytes();
    const query = this.options.query;
    const graphOptions = {
      retrievalK: MULTI_VIDEO_DEFAULTS.k,
      minMatches: MULTI_VIDEO_DEFAULTS.minMatches,
      minEdgeInliers: MULTI_VIDEO_DEFAULTS.minEdgeInliers,
      ...this.options.graph,
    };
    const started = performance.now();
    // The scene of a keyframe for the viewer, by canonical view id. It is known after the ids.
    let info = new Map<number, Keyframe>();
    let activityVideoOf = new Map<number, string>();
    const activity = new VideoActivityGraph();
    let lastActivity = -Infinity;
    let activityStage = '';
    const onActivity = (event: ViewGraphActivity) => {
      activity.update(event, activityVideoOf);
      const now = performance.now();
      // Send complete cumulative snapshots, so throttling never loses a completed pair.
      if (event.stage === activityStage && now - lastActivity < 200) return;
      activityStage = event.stage;
      lastActivity = now;
      this.post({ type: 'graphActivity', ...activity.snapshot() });
    };
    const posted = new Set<number>();
    const snapshots = new Snapshots(this.options.snapshotIntervalMs ?? SNAPSHOT_INTERVAL_MS, (s) =>
      this.post({
        type: 'model',
        partial: true,
        component: s.component,
        registered: s.registered,
        reconstruction: this.packedScene(s.reconstruction, info),
      }),
    );
    let finishedViews = 0,
      plannedViews = 0,
      currentViews = 0;
    const postStage = (event: Record<string, unknown>) =>
      this.post({ type: 'multiStage', ...event });
    const onEvent = (e: any) => {
      // The canonical ids, in input order: the scenes of the partial maps need the keyframes.
      if (e.stage === 'keyframes') {
        info = new Map(e.viewIds.map((id: number, i: number) => [id, input[i]!]));
        activityVideoOf = new Map([...info].map(([id, frame]) => [id, frame.videoId]));
      } else if (e.stage === 'candidates')
        postStage({ stage: 'candidates', done: e.total, total: e.total, final: true });
      else if (e.stage === 'matching' || e.stage === 'verification')
        postStage({ stage: e.stage, done: e.done, total: e.total });
      else if (e.stage === 'components')
        postStage({ stage: 'components', final: true, detail: e.edges + ' edges' });
      else if (e.stage === 'plan') {
        plannedViews = (e.sizes as number[]).reduce((n, v) => n + v, 0);
        postStage({
          stage: 'components',
          done: e.sizes.length,
          total: e.sizes.length,
          final: true,
        });
        postStage({ stage: 'reconstruction', done: 0, total: plannedViews });
      } else if (e.stage === 'reconstruction') {
        currentViews = e.registered;
        postStage({
          stage: 'reconstruction',
          done: finishedViews + currentViews,
          total: plannedViews,
        });
        snapshots.offer({
          component: e.component,
          registered: e.registered,
          reconstruction: e.reconstruction,
        });
      } else if (e.stage === 'componentDone') {
        snapshots.cancel();
        finishedViews += e.registered;
        currentViews = 0;
        postStage({ stage: 'reconstruction', done: finishedViews, total: plannedViews });
        posted.add(e.component);
        this.post({
          type: 'model',
          component: e.component,
          reconstruction: this.scene(e.reconstruction, info),
          meanReprojectionErrorPx: e.meanReprojectionErrorPx ?? null,
        });
      }
    };
    // Where the reconstruction time goes: the calls, the time and the submissions of each kind.
    const operations: Record<
      string,
      { calls: number; elapsedMs: number; submissions: number; gpuMs?: number }
    > = {};
    let reconstructionGpuMs: number | undefined;
    const reconstruct = this.api.reconstructMultiVideoOwnedGpu ?? this.api.reconstructMultiVideoGpu;
    let result: any;
    try {
      result = await reconstruct(
        this.device,
        this.api.reconstructMultiVideoOwnedGpu ? input.slice() : input,
        {
          canonical: query.canonical,
          graph: { ...graphOptions, onActivity },
          mapper: {
            ...this.options.mapper,
            onStage: (s: any) => {
              const o = (operations[s.operation] ??= { calls: 0, elapsedMs: 0, submissions: 0 });
              o.calls++;
              o.elapsedMs += s.elapsedMs;
              o.submissions += s.execution?.report?.submissions ?? 0;
              const gpuMs = reportGpuMs(s.execution?.report);
              if (gpuMs !== undefined) {
                o.gpuMs = (o.gpuMs ?? 0) + gpuMs;
                reconstructionGpuMs = (reconstructionGpuMs ?? 0) + gpuMs;
              }
            },
          },
          onProgress: (stage: string) => this.post({ type: 'progress', detail: stage }),
          onEvent,
          onRelease: ({ stage, viewIds }: { stage: string; viewIds: readonly number[] }) => {
            for (const id of viewIds) {
              const frame = info.get(id);
              if (!frame) continue;
              if (stage === 'descriptors') {
                frame.descriptors = new Uint8Array(0);
                frame.global = new Float32Array(0);
              } else {
                // componentDone has synchronously posted the final colored model. Later scenes do
                // not refer across disconnected components, so neither positions nor colors remain
                // numerical/display inputs for this component.
                frame.keypoints = new Float64Array(0);
                frame.colors = new Uint8Array(0);
              }
            }
          },
        },
      );
    } finally {
      // A failed mapper must not leave a delayed partial-map message retaining the failed run.
      snapshots.cancel();
    }
    const graph = result.graph;
    info = new Map<number, Keyframe>(
      result.keyframes.map((k: any, i: number) => [k.viewId, input[i]!]),
    );
    const videoOf = new Map<number, string>(
      result.keyframes.map((k: any) => [k.viewId, k.videoId]),
    );
    // Cross-video edges, one row for each pair of videos; edges inside a video, for the tags.
    const pairs = new Map<string, VideoPairEdges>();
    const internal = new Map<string, { edges: number; seedEligible: number }>();
    let crossEdges = 0,
      crossInliers = 0;
    for (const e of graph.edges) {
      const a = videoOf.get(e.first)!,
        b = videoOf.get(e.second)!;
      if (a === b) {
        const s = internal.get(a) ?? { edges: 0, seedEligible: 0 };
        s.edges++;
        if (e.seedEligible) s.seedEligible++;
        internal.set(a, s);
        continue;
      }
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const key = lo + ':' + hi;
      const row = pairs.get(key) ?? { a: lo, b: hi, edges: 0, inliers: 0 };
      pairs.set(key, { ...row, edges: row.edges + 1, inliers: row.inliers + e.inliers });
      crossEdges++;
      crossInliers += e.inliers;
    }
    const matrixOrder = [...this.order].sort();
    this.post({
      type: 'graph',
      canonical: result.canonical,
      stats: graph.stats,
      timing: graph.timing,
      components: graph.components.map((c: any) => ({
        index: c.index,
        keyframes: c.viewIds.length,
        videos: c.videoIds,
        reconstructed: c.reconstructed,
      })),
      videos: graph.videos,
      matrix: { order: matrixOrder, cells: inlierMatrix(matrixOrder, [...pairs.values()]) },
    });
    const registeredByVideo = new Map<string, number>();
    for (const c of result.components) {
      if (c.reconstruction && !posted.has(c.index))
        this.post({
          type: 'model',
          component: c.index,
          reconstruction: this.scene(c.reconstruction, info),
          meanReprojectionErrorPx: c.meanReprojectionErrorPx ?? null,
          videos: c.videoIds,
        });
      else if (c.error) this.post({ type: 'model', component: c.index, error: c.error });
      for (const v of c.reconstruction?.views ?? []) {
        const id = videoOf.get(v.id)!;
        registeredByVideo.set(id, (registeredByVideo.get(id) ?? 0) + 1);
      }
    }
    this.keepDenseInput(input, result.components, info);
    const tagged = new Map<string, any>(graph.videos.map((v: any) => [v.videoId, v]));
    const summary: MultiVideoSummary = {
      videos: this.order.map((id) => {
        const v = this.videos.get(id)!;
        const g = tagged.get(id);
        return {
          videoId: id,
          label: v.info.label,
          durationSeconds: v.info.durationSeconds,
          keyframes: v.keyframes.length,
          frames: v.frames,
          width: v.width,
          height: v.height,
          registeredViews: registeredByVideo.get(id) ?? 0,
          focalRatio: v.info.focalRatio,
          component: g && g.tag !== 'not connected' ? g.component : null,
          tags: tagsOf(g, internal.get(id) ?? { edges: 0, seedEligible: 0 }),
        };
      }),
      components: result.components
        .filter((c: any) => c.reconstructed)
        .map((c: any) => ({
          keyframes: c.viewIds.length,
          registeredViews: c.reconstruction?.views.length ?? 0,
          landmarks: c.reconstruction?.landmarks.length ?? 0,
          videos: c.videoIds.length,
          meanReprojectionErrorPx: Number.isFinite(c.meanReprojectionErrorPx)
            ? c.meanReprojectionErrorPx
            : null,
        })),
      crossVideoEdges: { edges: crossEdges, inliers: crossInliers },
      viewGraph: {
        nodes: graph.stats.keyframes,
        candidatePairs: graph.stats.candidatePairs,
        matchedPairs: graph.stats.verifiedPairs,
        verifiedEdges: graph.stats.edges,
        exhaustivePairs: graph.stats.exhaustivePairs,
      },
      notConnected: graph.videos
        .filter((v: any) => v.tag === 'not connected')
        .map((v: any) => v.videoId),
    };
    // `gpuMs` of a stage is absent when the job cannot attribute GPU time to it: the view-graph
    // stages run inside one library call, and a device without `timestamp-query` has no time.
    const stage = (ms: number, calls: number, gpuMs?: number) => ({
      calls,
      elapsedMs: ms,
      dispatches: 0,
      submissions: 0,
      ...(gpuMs !== undefined && gpuMs > 0 ? { gpuMs: Math.round(gpuMs * 1e3) / 1e3 } : {}),
    });
    const g = graph.timing;
    // The fallback runtime has no early release callback. All final scenes and summary rows are
    // now complete, so it can retire the same roots before Dense starts as well.
    for (const frame of input) {
      frame.keypoints = new Float64Array(0);
      frame.descriptors = new Uint8Array(0);
      frame.global = new Float32Array(0);
      frame.colors = new Uint8Array(0);
    }
    const featuresAfterSfMBytes = inputFeatureBytes();
    this.post({
      type: 'result',
      summary,
      canonical: result.canonical,
      // For the debug hook of the shuffle gate: identity and registration of every keyframe.
      keyframes: result.keyframes.map((k: any) => ({
        viewId: k.viewId,
        videoId: k.videoId,
        frameKey: k.frameKey,
        time: k.time,
      })),
      components: result.components.map((c: any) => ({
        index: c.index,
        viewIds: c.viewIds,
        registrationOrder: c.registrationOrder ?? [],
        poses: (c.reconstruction?.views ?? []).map((v: any) => ({
          id: v.id,
          R: Array.from(v.pose.R),
          t: Array.from(v.pose.t),
        })),
        error: c.error,
      })),
      featureCounts: this.featureCounts,
      operations,
      keyframeCount: input.length,
      hostRetention: {
        featuresBeforeSfMBytes,
        featuresAfterSfMBytes,
        ...(result.hostRetention ?? {}),
      },
      ...(result.descriptorCache ? { descriptorCache: result.descriptorCache } : {}),
      ...this.gpuFields(),
      stages: {
        features: stage(this.stageMs.features, this.featureCalls, this.stageGpuMs.features),
        globalDescriptor: stage(
          this.stageMs.globalDescriptor,
          this.featureCounts.length,
          this.stageGpuMs.globalDescriptor,
        ),
        retrieval: stage(g.retrievalMs, 1),
        pairMatching: stage(g.pairMatchingMs, graph.stats.candidatePairs),
        pairVerification: stage(g.pairVerificationMs, graph.stats.verifiedPairs),
        components: stage(g.componentsMs, 1),
        reconstruction: stage(
          result.timing.reconstructionMs,
          result.components.length,
          reconstructionGpuMs,
        ),
      },
      totalMs: performance.now() - started,
    });
    // Metadata/JPEGs have crossed to the page. Dense owns distinct image-only records, so it
    // cannot accidentally retain feature arrays through a Keyframe or a video cache.
    for (const video of this.videos.values()) video.keyframes = [];
  }

  /** The summed `gpuTime` of every group the device ran, in ns. 0 without `timestamp-query`. */
  private gpuNs(): number {
    try {
      let ns = 0;
      for (const row of this.device.kernelStatistics?.rows ?? []) ns += row.gpuNs;
      return ns;
    } catch {
      return 0;
    }
  }

  /**
   * The device's per-kernel table for the record's `timing.gpu` (`gpuRecord` in
   * `run-record/assemble.ts`).
   */
  gpuFields(device: any = this.device): { gpuKernels: unknown[]; gpuDropped: number } {
    try {
      const { rows, dropped } = device.kernelStatistics;
      return { gpuKernels: rows, gpuDropped: dropped };
    } catch {
      return { gpuKernels: [], gpuDropped: 0 };
    }
  }

  /**
   * Keep only each reconstructed component's registered pixels for Dense. The cache deliberately
   * has no reference to SIFT, colors, thumbnails or full keyframe records.
   */
  private keepDenseInput(input: Keyframe[], components: any[], info: Map<number, Keyframe>): void {
    if (!this.options.dense) return;
    this.denseComponents = components
      .filter((c) => c.reconstruction)
      .map((c) => ({ index: c.index, model: c.reconstruction }));
    for (const c of this.denseComponents)
      for (const v of c.model.views) {
        const k = info.get(v.id);
        if (k?.rgba) this.denseInfo.set(v.id, { rgba: k.rgba, width: k.width, height: k.height });
      }
    for (const k of input) k.rgba = undefined;
  }

  /** The worker has ended (including a sparse failure). Drop every remaining host input root. */
  releaseInputs(): void {
    for (const video of this.videos.values()) video.keyframes = [];
    this.denseInfo.clear();
    this.denseComponents.length = 0;
    this.denseCurrent = undefined;
  }

  /** Logical worker-owned input backing buffers. This excludes posted messages, caller-retained
   * arrays, JS object/string overhead, GPU allocations and driver/browser RSS. Category bytes
   * can overlap through subviews; `retainedArrayBufferBytes` deduplicates all categories. */
  hostInputs(): {
    featureBytes: number;
    rgbaBytes: number;
    sparseBytes: number;
    retainedArrayBufferBytes: number;
    pendingComponents: number;
    pendingObservations: number;
    activeObservations: number;
  } {
    const features: ArrayBufferView[] = [];
    const pixels: ArrayBufferView[] = [...this.denseInfo.values()].map((image) => image.rgba);
    for (const video of this.videos.values())
      for (const frame of video.keyframes) {
        features.push(frame.keypoints, frame.descriptors, frame.global, frame.colors);
        if (frame.rgba) pixels.push(frame.rgba);
      }
    const sparse: ArrayBufferView[] = [];
    let pendingObservations = 0;
    for (const { model } of this.denseComponents) {
      for (const view of model.views) sparse.push(view.pose.R, view.pose.t);
      for (const point of model.landmarks) {
        sparse.push(point.xyz);
        pendingObservations += point.observations.length;
      }
    }
    let activeObservations = 0;
    if (this.denseCurrent) {
      const { model, job } = this.denseCurrent;
      for (const image of job.rgba.values()) pixels.push(image.rgba);
      for (const view of model.views) sparse.push(view.pose.R, view.pose.t);
      for (const point of model.landmarks) {
        sparse.push(point.xyz);
        activeObservations += point.observations.length;
      }
    }
    return {
      featureBytes: backingBytes(features),
      rgbaBytes: backingBytes(pixels),
      sparseBytes: backingBytes(sparse),
      retainedArrayBufferBytes: backingBytes([...features, ...pixels, ...sparse]),
      pendingComponents: this.denseComponents.length,
      pendingObservations,
      activeObservations,
    };
  }

  /** Stop the dense stage: the current component keeps its points, the others start no work. */
  cancelDense(): void {
    this.denseCancelled = true;
    this.denseCurrent?.job.abort.abort();
  }

  /**
   * The dense device was lost: the running component ends as `failed`, and so does each later one,
   * without work. The sparse maps stay (Amendment 9, item 5).
   */
  failDense(message: string, reason: string): void {
    this.denseFailed ??= { message, reason };
    const current = this.denseCurrent;
    if (current?.job.running)
      endDense(current.context, {
        failure: denseFailure(current.job, message, this.options.dense!.lastKernel(), reason),
      });
  }

  /**
   * The dense stage of a multi-video run (`docs/multi-video.md`, Amendment 9): `runDense` of
   * `dense-run.ts` once for each reconstructed component, in index order, on `device`. Every
   * message carries its `component`. `denseDone` ends the stage.
   */
  async runDense(device: any): Promise<void> {
    const dense = this.options.dense;
    if (!dense) return;
    const api = this.api;
    while (this.denseComponents.length) {
      // Consume the pending queue: a completed component's full sparse observation graph is
      // not retained for all the following components.
      const { index, model } = this.denseComponents.shift()!;
      const job = createDenseJob(dense.clean, dense.mesh, dense.fusion);
      const context: DenseRunContext = {
        dense: job,
        component: index,
        options: dense.options,
        statistics: () => device.statistics,
        reconstruct: (scene, images, options) =>
          api.reconstructDenseGpu!(device, scene, images, options),
        clean: api.cleanDenseCloudGpu
          ? (points) => api.cleanDenseCloudGpu!(device, points, dense.cleanOptions)
          : undefined,
        mesh: api.meshDenseCloudGpu
          ? (points) => api.meshDenseCloudGpu!(device, points)
          : undefined,
        // Display previews are not drawn; lossless diagnostic artifacts still reach the record.
        post: (message, transfer) => {
          if (message.type !== 'denseDepth') this.post({ ...message, component: index }, transfer);
        },
        finalFields: () => {
          const timestampFields = denseGpuTimestampFields(device);
          // A lost or disposed device may refuse the read.
          try {
            return { chunks: { ...(device?.chunkRates ?? {}) }, ...timestampFields };
          } catch {
            return { chunks: {}, ...timestampFields };
          }
        },
        lastKernel: dense.lastKernel,
      };
      this.denseCurrent = { job, context, model };
      if (this.denseFailed) {
        endDense(context, {
          failure: denseFailure(job, this.denseFailed.message, '', this.denseFailed.reason),
        });
      } else if (this.denseCancelled) {
        job.abort.abort();
        endDense(context, { cancelled: true });
      } else {
        if (model.views.length < 2) job.skipped = 'Fewer than two registered views';
        for (const view of model.views) {
          const k = this.denseInfo.get(view.id);
          if (k?.rgba) job.rgba.set(view.id, { rgba: k.rgba, width: k.width, height: k.height });
        }
        await runDense(context, model);
      }
      // The component is done: its pixels are no longer needed.
      for (const view of model.views) {
        this.denseInfo.delete(view.id);
      }
      this.denseCurrent = undefined;
    }
    this.denseCurrent = undefined;
    this.denseComponents = [];
    this.denseInfo = new Map();
    // The worker opens a different device for Dense. Its table is cumulative across all
    // components; send it once so the page can add it to the sparse device's final table.
    this.post({ type: 'denseDone', ...(device ? this.gpuFields(device) : {}) });
  }

  /** A growing map for the viewer: the views, and the landmarks as flat typed arrays. */
  private packedScene(model: any, info: Map<number, Keyframe>): PackedScene {
    const { views, landmarks } = this.scene(model, info);
    return { views, packed: packLandmarks(landmarks) };
  }

  /** A model as the viewer reads it: the mapper's views with their frame data, and colors. */
  private scene(model: any, info: Map<number, Keyframe>) {
    const colorOf = new Map<number, [number, number, number]>();
    return {
      views: model.views.map((view: any) => {
        const k = info.get(view.id)!;
        return {
          ...view,
          time: k.time,
          sequence: k.sequence,
          width: k.width,
          height: k.height,
          image: k.image,
          videoId: k.videoId,
        };
      }),
      landmarks: model.landmarks.map((point: any) => {
        let color = colorOf.get(point.trackId);
        if (!color)
          for (const o of point.observations) {
            const k = info.get(o.viewId);
            if (k && o.featureId * 3 + 2 < k.colors.length) {
              color = [
                k.colors[o.featureId * 3]! / 255,
                k.colors[o.featureId * 3 + 1]! / 255,
                k.colors[o.featureId * 3 + 2]! / 255,
              ];
              break;
            }
          }
        return { ...point, color: color ?? [0.7, 0.7, 0.7] };
      }),
    };
  }
}
