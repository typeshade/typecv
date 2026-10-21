/**
 * The `multiVideo` job of the SfM worker (`docs/multi-video.md`, 6.2). The worker file only wires
 * messages to this class, so a unit test runs the job in Node with a fake feature front end.
 *
 * Messages in: `video` (the start of a video), `frame`, `endVideo` and `finish`. Messages out,
 * through `post`: `frameComplete`, `features`, `videoComplete`, `graph`, `model` and `result`.
 * The messages carry data, not text (6.2).
 */
import { PhaseGate } from './keyframe-gate.ts';
import { frameKeyOf } from '../../../src/reconstruction/identity.ts';
import {
  MULTI_VIDEO_DEFAULTS,
  inlierMatrix,
  tagsOf,
  thinnedPositions,
  type MultiVideoQuery,
  type MultiVideoSummary,
  type VideoPairEdges,
} from './multi-video.ts';

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
}

export interface JobOptions {
  query: MultiVideoQuery;
  maxFeatures: number;
  firstOctave: -1 | 0;
  clahe?: boolean;
  /** The mapper's fixed controls (`LAB_OPTIONS` of the worker). */
  mapper: Record<string, unknown>;
  /** The graph controls that the lab changes from the defaults of 3.3 and 3.4. */
  graph?: Record<string, unknown>;
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
}

interface VideoState {
  info: JobVideo;
  gate: PhaseGate;
  keyframes: Keyframe[];
  frames: number;
  skipped: number;
  dropped: number;
  ended: boolean;
}

function cameraOf(frame: JobFrame, focalRatio: number) {
  const sourceWidth = frame.sourceWidth ?? frame.width;
  const sourceHeight = frame.sourceHeight ?? frame.height;
  const sourceFocal = sourceWidth * focalRatio;
  return (
    frame.camera ?? {
      fx: (sourceFocal * frame.width) / sourceWidth,
      fy: (sourceFocal * frame.height) / sourceHeight,
      cx: (frame.width - 1) / 2,
      cy: (frame.height - 1) / 2,
      k1: 0,
      k2: 0,
    }
  );
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

export class MultiVideoJob {
  private readonly videos = new Map<string, VideoState>();
  /** The videos in the order they arrived: the list order of the page. */
  private readonly order: string[] = [];
  private stageMs = { features: 0, globalDescriptor: 0 };
  private featureCalls = 0;
  private readonly featureCounts: number[] = [];

  private readonly api: JobApi;
  private readonly device: any;
  private readonly vocabulary: Float32Array;
  private readonly options: JobOptions;
  private readonly post: (message: Record<string, unknown>) => void;

  constructor(
    api: JobApi,
    device: any,
    vocabulary: Float32Array,
    options: JobOptions,
    post: (message: Record<string, unknown>) => void,
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
    this.featureCalls++;
    try {
      if (sift.count < 8) {
        video.gate.discard();
        video.skipped++;
        done({ keyframe: false, skipped: 'Too few detected features' });
        return;
      }
      const keypoints = await sift.readKeypoints();
      const t1 = performance.now();
      const described = await this.api.describeKeyframeGpu(this.device, sift, this.vocabulary);
      this.stageMs.globalDescriptor += performance.now() - t1;
      video.gate.accept();
      this.featureCounts.push(sift.count);
      video.keyframes.push({
        videoId,
        frameKey,
        time: frame.time,
        frameId: frame.id,
        camera: cameraOf(frame, video.info.focalRatio),
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
    });
  }

  /** The finish of 6.2: ids, view graph, components, the mapper on each, the models. */
  async finish(): Promise<void> {
    const input = this.order.flatMap((id) => this.videos.get(id)!.keyframes);
    const query = this.options.query;
    const graphOptions = {
      retrievalK: MULTI_VIDEO_DEFAULTS.k,
      minMatches: MULTI_VIDEO_DEFAULTS.minMatches,
      minEdgeInliers: MULTI_VIDEO_DEFAULTS.minEdgeInliers,
      ...this.options.graph,
    };
    const started = performance.now();
    const result = await this.api.reconstructMultiVideoGpu(this.device, input, {
      canonical: query.canonical,
      graph: graphOptions,
      mapper: this.options.mapper,
      onProgress: (stage: string) => this.post({ type: 'progress', detail: stage }),
    });
    const graph = result.graph;
    const info = new Map<number, Keyframe>(
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
      if (c.reconstruction)
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
    const stage = (ms: number, calls: number) => ({
      calls,
      elapsedMs: ms,
      dispatches: 0,
      submissions: 0,
    });
    const g = graph.timing;
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
      keyframeCount: input.length,
      stages: {
        features: stage(this.stageMs.features, this.featureCalls),
        globalDescriptor: stage(this.stageMs.globalDescriptor, this.featureCounts.length),
        retrieval: stage(g.retrievalMs, 1),
        pairMatching: stage(g.pairMatchingMs, graph.stats.candidatePairs),
        pairVerification: stage(g.pairVerificationMs, graph.stats.verifiedPairs),
        components: stage(g.componentsMs, 1),
        reconstruction: stage(result.timing.reconstructionMs, result.components.length),
      },
      totalMs: performance.now() - started,
    });
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
