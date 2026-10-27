/**
 * The page's builders of a whole run record (`docs/lab-run-records.md`): one video
 * (`buildRunRecord`) and a multi-video run (`buildMultiRunRecord`), from the worker's messages,
 * the page's part of the run and the browser's environment. The lab page calls them, and the
 * golden records of `website/run-record-golden.test.mjs` hold them in Node.
 *
 * This module sits outside the barrel `run-record.ts`. The collector, the scripts and the root
 * typecheck load the barrel, and the builders import modules of the page:
 * `lab-options.ts` and `multi-video.ts`.
 * Through the barrel, a type import of `multi-video.ts` would also close a cycle.
 */
import { LAB_OPTIONS, MULTI_VIDEO_RECORD_OPTIONS } from '../lab-options.ts';
import type { stageTimings } from '../multi-video-stages.ts';
import { multiVideoOptions, type multiVideoQuery, type MultiVideoSummary } from '../multi-video.ts';
import type { VideoSource } from '../sample-source.ts';
import type { PartialScene, Scene } from '../sfm-scene.ts';
import type { VideoRange } from '../video-range.ts';
import {
  chunkRecord,
  denseRecord,
  gpuRecord,
  sparsePerformanceRecord,
  sumDenseRecords,
  summarizeStages,
} from './assemble.ts';
import {
  RUN_RECORD_FORMAT,
  RUN_RECORD_MAX_FEATURES,
  RUN_RECORD_MAX_FRAMES,
  RUN_RECORD_MAX_NOTE,
  type RunRecord,
  type RunRecordDense,
} from './schema.ts';

/**
 * The browser globals that `buildRunRecord` reads for the record's `client` block. The page
 * passes its own `navigator`, `screen` and `devicePixelRatio`; a Node test passes fixed values.
 */
export interface RunRecordEnvironment {
  readonly navigator: Navigator;
  readonly screen: Screen;
  readonly devicePixelRatio: number;
}

/**
 * Build the record of one run from the worker's result and the page's environment. A failed run
 * passes the worker's partial outcome (no reconstruction; the last snapshot's counts) and the
 * failure.
 */
export function buildRunRecord(
  environment: RunRecordEnvironment,
  input: {
    result: any;
    failure?: RunRecord['failure'];
    adapter: any;
    commit: string;
    kind: RunRecord['input']['kind'];
    sampleName: string | null;
    frames: readonly {
      width: number;
      height: number;
      sourceWidth?: number;
      sourceHeight?: number;
    }[];
    durationSeconds: number;
    range?: VideoRange;
    /** Set for an uploaded video only. */
    videoSampleFps?: number;
    /** Set for an uploaded video only: the path that decoded it. */
    videoDecode?: 'worker' | 'main-thread';
    /** Set for a video sample that stores a source video: the input its frames came from. */
    videoSource?: VideoSource | 'mixed';
    benchmark?: { name: string };
    reference: RunRecord['reference'];
    note: string;
    mainThread?: RunRecord['timing']['mainThread'];
    /** The worker's `denseResult`, for a run with the dense MVS. */
    dense?: any;
  },
): RunRecord {
  const { navigator, screen, devicePixelRatio } = environment;
  const { result, adapter } = input,
    dense = denseRecord(input.dense),
    total = result.stages.find((stage: any) => stage.operation === 'runTotal'),
    first = input.frames[0],
    options = result.options ?? {};
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    format: RUN_RECORD_FORMAT,
    createdAt: new Date().toISOString(),
    site: { commit: input.commit, runtime: '/runtime/gpu.js' },
    client: {
      userAgent: navigator.userAgent.slice(0, 512),
      platform: String((navigator as any).userAgentData?.platform ?? '').slice(0, 64),
      language: navigator.language.slice(0, 32),
      hardwareConcurrency: navigator.hardwareConcurrency ?? 0,
      deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
      screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio },
    },
    gpu: {
      vendor: String(adapter?.vendor ?? '').slice(0, 128),
      architecture: String(adapter?.architecture ?? '').slice(0, 128),
      device: String(adapter?.device ?? '').slice(0, 128),
      description: String(adapter?.description ?? '').slice(0, 512),
      fallbackAdapter: adapter?.fallbackAdapter === true,
      limits: {
        maxStorageBufferBindingSize: adapter?.limits?.maxStorageBufferBindingSize ?? 0,
        maxBufferSize: adapter?.limits?.maxBufferSize ?? 0,
        maxComputeWorkgroupStorageSize: adapter?.limits?.maxComputeWorkgroupStorageSize ?? 0,
        maxComputeInvocationsPerWorkgroup: adapter?.limits?.maxComputeInvocationsPerWorkgroup ?? 0,
        maxStorageBuffersPerShaderStage: adapter?.limits?.maxStorageBuffersPerShaderStage ?? 0,
      },
      features: (Array.isArray(adapter?.features) ? adapter.features : [])
        .slice(0, RUN_RECORD_MAX_FEATURES)
        .map((name: unknown) => String(name).slice(0, 64)),
    },
    input: {
      kind: input.kind,
      name: input.kind === 'sample' ? input.sampleName : null,
      frames: input.frames.length,
      width: first?.width ?? 0,
      height: first?.height ?? 0,
      sourceWidth: first?.sourceWidth ?? first?.width ?? 0,
      sourceHeight: first?.sourceHeight ?? first?.height ?? 0,
      durationSeconds: Number.isFinite(input.durationSeconds) ? input.durationSeconds : 0,
      ...(input.range ? { range: input.range } : {}),
    },
    options: {
      frontEnd: options.frontEnd === 'sift' ? 'sift' : 'corner',
      focalRatio: options.focalRatio ?? 0,
      windowSize: options.windowSize ?? 0,
      ransacIterations: options.ransacIterations ?? 0,
      maxErrorPx: options.maxErrorPx ?? 0,
      bundleIterations: options.bundleIterations ?? 0,
      finalBundleIterations: options.finalBundleIterations ?? 0,
      refineSharedFocal: options.refineSharedFocal === true,
      ...(options.focalSearch === true ? { focalSearch: true } : {}),
      maxFeatures: options.maxFeatures ?? 0,
      matchRatio: options.matchRatio ?? 0,
      maxDistance: options.maxDistance ?? 0,
      boundedSubmissions: options.boundedSubmissions === true,
      fp64Flavor: options.fp64Flavor === 'integer' ? 'integer' : 'float',
      guidedMatching: options.guidedMatching === true,
      ...(options.guidedMatching === true && options.guidedKernel === 'tiled'
        ? { guidedKernel: 'tiled' as const }
        : {}),
      siftFirstOctave: options.siftFirstOctave === -1 ? -1 : 0,
      clahe: options.clahe === true,
      ...(input.videoSampleFps ? { videoSampleFps: input.videoSampleFps } : {}),
      ...(input.videoDecode ? { videoDecode: input.videoDecode } : {}),
      ...(input.videoSource ? { videoSource: input.videoSource } : {}),
      ...(options.keyframeGate === 'phase' || options.keyframeGate === 'legacy'
        ? { keyframeGate: options.keyframeGate as 'phase' | 'legacy' }
        : {}),
      ...(options.features === 'orb' ? { features: 'orb' as const } : {}),
      ...(typeof result.keyframes === 'number' ? { keyframes: result.keyframes } : {}),
      ...(typeof options.klt === 'boolean' ? { klt: options.klt } : {}),
      ...(options.klt && options.kltSetup ? { kltSetup: options.kltSetup } : {}),
    },
    result: {
      registeredViews: result.reconstruction?.views.length ?? result.latest?.views ?? 0,
      landmarks: result.reconstruction?.landmarks.length ?? result.latest?.landmarks ?? 0,
      observations:
        result.reconstruction?.landmarks.reduce(
          (sum: number, point: any) => sum + point.observations.length,
          0,
        ) ??
        result.latest?.observations ??
        0,
      meanReprojectionErrorPx: Number.isFinite(result.diagnostics?.meanReprojectionErrorPx)
        ? result.diagnostics.meanReprojectionErrorPx
        : null,
      unregistered: result.unregistered?.length ?? 0,
      matchedPairs: result.matchedPairs ?? 0,
      ...(result.guided
        ? { guided: { pairs: result.guided.pairs, addedMatches: result.guided.addedMatches } }
        : {}),
      featureCounts: (result.featureCounts ?? []).slice(0, 2048),
      ...sparsePerformanceRecord(result),
      ...(result.klt ? { klt: result.klt } : {}),
      ...(result.focalSearch
        ? {
            focalSearch: {
              ratio: result.focalSearch.ratio,
              gridRatio: result.focalSearch.gridRatio,
              views: result.focalSearch.views,
            },
          }
        : {}),
    },
    timing: {
      totalMs: total?.elapsedMs ?? 0,
      reusedDevice: total?.reusedDevice === true,
      stages: result.summary ?? summarizeStages(result.stages ?? []),
      frameMs: (
        result.frameMs ??
        (result.stages ?? [])
          .filter((stage: any) => stage.operation === 'keyframe')
          .map((stage: any) => stage.elapsedMs)
      ).slice(0, 2048),
      // The dense MVS's kernels and chunk rates follow the sparse run's (5.3).
      kernels: (input.dense?.kernels ?? result.kernels ?? [])
        .slice(0, 256)
        .map((name: unknown) => String(name).slice(0, 64)),
      chunks: chunkRecord(input.dense?.chunks ?? result.chunks),
      // The dense MVS's table is the whole job's: it covers the sparse run too.
      ...gpuRecord(
        input.dense?.gpuKernels ?? result.gpuKernels,
        input.dense?.gpuDropped ?? result.gpuDropped,
      ),
      budgetMs: Number.isFinite(result.budgetMs) && result.budgetMs > 0 ? result.budgetMs : null,
      ...(input.mainThread ? { mainThread: input.mainThread } : {}),
    },
    transfers: {
      applicationUploadBytes: total?.applicationUploadBytes ?? 0,
      applicationReadbackBytes: total?.applicationReadbackBytes ?? 0,
    },
    benchmark: input.benchmark
      ? { name: input.benchmark.name, pass: total?.reusedDevice ? 'warm' : 'cold' }
      : null,
    reference: input.reference,
    failure: input.failure ?? null,
    note: input.note.slice(0, RUN_RECORD_MAX_NOTE),
    ...(dense ? { dense } : {}),
  };
}

/**
 * The record of a multi-video run (`docs/multi-video.md` 6.4): the single-video record built from
 * the sums of the run, plus the optional multi-video fields. `result` is the worker's `result`
 * message; it is undefined for a run that failed before it.
 */
export function buildMultiRunRecord(
  environment: RunRecordEnvironment,
  input: {
    result: any;
    graph: any;
    models: ReadonlyMap<number, Scene>;
    meanErrors: ReadonlyMap<number, number | null>;
    adapter: any;
    commit: string;
    rows: readonly { videoId?: string; sampleId?: string; focal: number; range?: VideoRange }[];
    query: ReturnType<typeof multiVideoQuery>;
    maxFeatures: number;
    firstOctave: -1 | 0;
    frames: number;
    first?: { width: number; height: number; sourceWidth?: number; sourceHeight?: number };
    durationSeconds: number;
    videoDecode?: 'worker' | 'main-thread';
    videoSource?: VideoSource | 'mixed';
    videoSampleFps?: number;
    mainThread?: RunRecord['timing']['mainThread'];
    setName?: string | null;
    note: string;
    failure?: RunRecord['failure'];
    /**
     * The dense MVS of each component, by component index (`docs/multi-video.md`, Amendment 9), and
     * the chunk rates the dense device reported last. Absent for a run without the dense MVS.
     */
    dense?: { blocks: ReadonlyMap<number, RunRecordDense>; chunks?: unknown };
    /**
     * A failed run has no worker `result`. It passes what the page saw: the time of the stages so
     * far, the videos it fed, and the maps that existed at the failure (the final map of a
     * component, or else its last partial map).
     */
    partial?: {
      stages: ReturnType<typeof stageTimings>;
      totalMs: number;
      videos: MultiVideoSummary['videos'][number][];
      scenes: ReadonlyMap<number, Scene>;
      /** The device's per-kernel table up to the failure, from the worker's `error` message. */
      gpuKernels?: unknown[];
      gpuDropped?: number;
    };
  },
): RunRecord {
  const { models } = input;
  const result =
    input.result ??
    (input.partial && {
      stages: input.partial.stages,
      totalMs: input.partial.totalMs,
      gpuKernels: input.partial.gpuKernels,
      gpuDropped: input.partial.gpuDropped,
    });
  const scenes = [...(input.partial?.scenes ?? models).values()];
  const registered = scenes.reduce((n, m) => n + m.views.length, 0);
  const landmarks = scenes.reduce((n, m) => n + m.landmarks.length, 0);
  const observations = scenes.reduce(
    (n, m) =>
      n +
      ((m as Partial<PartialScene>).observationCount ??
        m.landmarks.reduce((k, p) => k + p.observations.length, 0)),
    0,
  );
  const keyframes: number = result?.keyframeCount ?? 0;
  const pseudo = {
    stages: [
      {
        operation: 'runTotal',
        elapsedMs: result?.totalMs ?? 0,
        reusedDevice: false,
        applicationUploadBytes: 0,
        applicationReadbackBytes: 0,
      },
    ],
    summary: result?.stages ?? {},
    // The numbers of `LAB_OPTIONS` (`lib/lab-options.ts`) that a multi-video map uses, and the
    // literals of a multi-video record (`MULTI_VIDEO_RECORD_OPTIONS`).
    options: {
      frontEnd: 'sift',
      focalRatio: input.rows[0]?.focal ?? 0,
      windowSize: MULTI_VIDEO_RECORD_OPTIONS.windowSize,
      ransacIterations: LAB_OPTIONS.ransacIterations,
      maxErrorPx: LAB_OPTIONS.maxErrorPx,
      bundleIterations: LAB_OPTIONS.bundleIterations,
      finalBundleIterations: LAB_OPTIONS.finalBundleIterations,
      refineSharedFocal: LAB_OPTIONS.refineSharedFocal,
      maxFeatures: input.maxFeatures,
      matchRatio: LAB_OPTIONS.matchRatio,
      maxDistance: LAB_OPTIONS.maxDistance,
      siftFirstOctave: input.firstOctave,
      clahe: MULTI_VIDEO_RECORD_OPTIONS.clahe,
      keyframeGate: MULTI_VIDEO_RECORD_OPTIONS.keyframeGate,
    },
    keyframes,
    latest: { views: registered, landmarks, observations },
    diagnostics: { meanReprojectionErrorPx: input.meanErrors.get(0) ?? Number.NaN },
    unregistered: Array.from({ length: Math.max(0, keyframes - registered) }),
    matchedPairs: input.graph?.stats.verifiedPairs ?? 0,
    featureCounts: result?.featureCounts ?? [],
    ...sparsePerformanceRecord(result),
    frameMs: [],
    kernels: [],
    chunks: input.dense?.chunks ?? {},
    gpuKernels: result?.gpuKernels,
    gpuDropped: result?.gpuDropped,
  };
  const allSamples = input.rows.length > 0 && input.rows.every((r) => r.sampleId);
  const base = buildRunRecord(environment, {
    result: pseudo,
    failure: input.failure,
    adapter: input.adapter,
    commit: input.commit,
    kind: allSamples ? 'sample' : 'video',
    sampleName: allSamples ? (input.setName ?? null) : null,
    frames: Array.from({ length: Math.min(input.frames, RUN_RECORD_MAX_FRAMES) }, () => ({
      width: input.first?.width ?? 0,
      height: input.first?.height ?? 0,
      sourceWidth: input.first?.sourceWidth,
      sourceHeight: input.first?.sourceHeight,
    })),
    durationSeconds: input.durationSeconds,
    videoSampleFps: input.videoSampleFps,
    videoDecode: input.videoDecode,
    videoSource: input.videoSource,
    reference: null,
    note: input.note,
    mainThread: input.mainThread,
  });
  const summary = input.result?.summary;
  const videos = (summary?.videos ?? input.partial?.videos ?? []).map((v: any) => {
    const range = input.rows.find((row) => row.videoId === v.videoId)?.range;
    return {
      ...v,
      // The typed name of an upload is never sent; a lab sample is public (the record, 6.4).
      label: v.label ?? null,
      ...(range ? { range } : {}),
    };
  });
  // The summary lists the reconstructed components in the order of the graph's components.
  const reconstructed: number[] = (input.graph?.components ?? [])
    .filter((c: any) => c.reconstructed)
    .map((c: any) => c.index);
  const denseBlocks = input.dense?.blocks;
  const components = summary
    ? summary.components.map((c: any, i: number) => {
        const block = denseBlocks?.get(reconstructed[i]!);
        return block ? { ...c, dense: block } : c;
      })
    : undefined;
  const denseSum = denseBlocks?.size
    ? sumDenseRecords([...denseBlocks.entries()].sort((a, b) => a[0] - b[0]).map(([, d]) => d))
    : undefined;
  return {
    ...base,
    input: { ...base.input, ...(videos.length ? { videos } : {}) },
    options: { ...base.options, multiVideo: multiVideoOptions({ query: input.query }) },
    ...(denseSum ? { dense: denseSum } : {}),
    result: {
      ...base.result,
      ...(summary
        ? {
            components,
            crossVideoEdges: summary.crossVideoEdges,
            viewGraph: summary.viewGraph,
            notConnected: summary.notConnected,
          }
        : {}),
    },
  };
}
