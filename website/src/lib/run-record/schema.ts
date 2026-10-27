import type { GpuLossContext } from '../gpu-loss-context.ts';
import type { VideoRange } from '../video-range.ts';
import type { InitializationFailureRecord } from '../initialization-failure.ts';

/**
 * A run record is the measurement a visitor can send from the SfM lab after a run:
 * environment, input shape, options, result counts and timings. It holds no video,
 * image, file name or network address. The lab builds it; `/api/records` validates and
 * stores it. The same validator runs in both places.
 */
export const RUN_RECORD_FORMAT = 'typecv-run-record-v6' as const;
/** The formats the validator accepts: the current one, v5 (no GPU time per kernel, no adapter
 *  features), v4 (no `dense`) and v3 (no `dense`, no `timing.chunks` and no `timing.budgetMs`) from
 *  a page loaded before the next one was deployed. */
export const RUN_RECORD_FORMATS = [
  'typecv-run-record-v3',
  'typecv-run-record-v4',
  'typecv-run-record-v5',
  RUN_RECORD_FORMAT,
] as const;
export const RUN_RECORD_MAX_CHUNK_KEYS = 32;
export const RUN_RECORD_MAX_KERNELS = 256;
/** The rows of `timing.gpu`: the device's collector keeps at most this many call lists. */
export const RUN_RECORD_MAX_GPU_ROWS = 160;
export const RUN_RECORD_MAX_GPU_NAME = 256;
export const RUN_RECORD_MAX_FEATURES = 64;
export const RUN_RECORD_MAX_BYTES = 256 * 1024;
export const RUN_RECORD_MAX_NOTE = 500;
export const RUN_RECORD_MAX_FRAMES = 2048;
/** The result files a run record can declare, in the names of the lab's downloads. */
export const RESULT_FILE_NAMES = [
  'typecv-reconstruction.json',
  'typecv-reconstruction.ply',
  'typecv-dense.ply',
  'typecv-mesh.ply',
] as const;
export type ResultFileBaseName =
  | (typeof RESULT_FILE_NAMES)[number]
  | `typecv-reconstruction-c${string}.json`
  | `typecv-reconstruction-c${string}.ply`
  | `typecv-dense-c${string}.ply`
  | `typecv-mesh-c${string}.ply`
  | `typecv-depths-c${string}.json`
  | `typecv-depths-c${string}-${string}.tar.gz`;
export type ResultFileName = ResultFileBaseName | `${ResultFileBaseName}.part-${string}`;
export const RESULT_FILE_MAX_COUNT = 1024;
export const RESULT_FILES_MAX_BYTES = 4 * 1024 ** 3;
export const RESULT_FILE_PART_BYTES = 16 * 1024 ** 2;
export function resultFileBaseNameValid(name: unknown): name is ResultFileBaseName {
  return (
    typeof name === 'string' &&
    ((RESULT_FILE_NAMES as readonly string[]).includes(name) ||
      /^typecv-(?:reconstruction-c[0-9]{3}\.(?:json|ply)|(?:dense|mesh)-c[0-9]{3}\.ply)$/.test(
        name,
      ) ||
      /^typecv-depths-c[0-9]{3}(?:-[0-9]{4}\.tar\.gz|\.json)$/.test(name))
  );
}
export function resultFileNameValid(name: unknown): name is ResultFileName {
  if (resultFileBaseNameValid(name)) return true;
  if (typeof name !== 'string') return false;
  const match = /^(.*)\.part-([0-9]{4})$/.exec(name);
  return !!match && resultFileBaseNameValid(match[1]);
}
/** The largest result file the collector stores. */
export const RESULT_FILE_MAX_BYTES = 50 * 1024 * 1024;

export interface RunRecordStage {
  readonly calls: number;
  readonly elapsedMs: number;
  readonly dispatches: number;
  readonly submissions: number;
  /**
   * v6. The GPU time of the stage's plans in ms, summed from the groups that carry a time. Absent
   * when no group did (the device has no `timestamp-query`) and in earlier records.
   */
  readonly gpuMs?: number;
}
/**
 * v6. One row of `timing.gpu`: a fused call list (the kernel names joined by `+`), the groups of
 * that list the run executed, the dispatches they made and their GPU time in ms. `gpuMs` is null
 * when no group of the list carried a time. `timedGroups` is how many did (at most `groups`).
 */
export interface RunRecordKernelTime {
  readonly name: string;
  readonly groups: number;
  readonly dispatches: number;
  readonly timedGroups: number;
  readonly gpuMs: number | null;
}
/** What the TypeShade device measured of one chunk key (change 0074), as
 *  `ShadeDevice.chunkRates` reports it: `chunks` counts the timed chunks, at most
 *  `chunkTimingSamples` (8) per key on a device. */
/** The page's main thread during a run: the Long Tasks API and `requestAnimationFrame` gaps. */
export interface RunRecordMainThread {
  /** Tasks over 50 ms, their total, the longest and the lab's stage when it ended. */
  readonly longTasks: number;
  readonly longTaskMs: number;
  readonly longestTaskMs: number;
  readonly longestTaskStage: string;
  /** Animation frames more than 100 ms apart, their total, the longest and the stage. */
  readonly frameGaps: number;
  readonly frameGapMs: number;
  readonly longestGapMs: number;
  readonly longestGapStage: string;
  /**
   * The time the tab was hidden during the run, in ms. A hidden tab paints no frames: a frame gap
   * that overlaps it is not counted. Absent in earlier records, where a long gap may be a hidden tab.
   */
  readonly hiddenMs?: number;
}

export type RunRecordFile = {
  readonly name: ResultFileName;
  /** Unknown only when preparation failed and `skipped` explains the failure. */
  readonly bytes?: number;
  /** Lowercase hex SHA-256; absent when skipped before its bytes could be read. */
  readonly sha256?: string;
  /** Why the page did not upload the file; absent when it uploads it. */
  readonly skipped?: string;
  /** Concatenate the named original's parts by offset; each part has its own digest. */
  readonly part?: {
    readonly original: ResultFileBaseName;
    readonly totalBytes: number;
    readonly offset: number;
  };
};

/** The stages of the dense MVS whose elapsed time a record holds (`docs/dense-mvs.md` 5.3). */
export const RUN_RECORD_DENSE_STAGES = ['photometric', 'geometric', 'fusion'] as const;

/** Unique backing-buffer roots retained by each named owner. Worker and library scopes overlap:
 * these counters must not be added to one another, GPU memory, or browser/process RSS. */
export interface RunRecordHostRetention {
  readonly featuresBeforeSfMBytes: number;
  readonly featuresAfterSfMBytes: number;
  readonly featuresBeforeGraphBytes: number;
  readonly featuresAfterGraphBytes: number;
  readonly featuresAfterMappingBytes: number;
  readonly graphMatchesBeforeMappingBytes: number;
  readonly graphMatchesAfterMappingBytes: number;
}

/** Immutable-descriptor cache payload/work counters. Excludes uncached live inputs, pooled
 * arrays, driver storage and process RSS. Entries/bytes include retired active cache leases. */
export interface RunRecordDescriptorCache {
  readonly hits: number;
  readonly misses: number;
  readonly uncached: number;
  readonly decodes: number;
  readonly decodedBytes: number;
  readonly applicationUploadBytes: number;
  readonly evictions: number;
  readonly entries: number;
  readonly residentBytes: number;
  readonly peakEntries: number;
  readonly peakResidentBytes: number;
  readonly limitBytes: number;
}

/** Per-component measured work and host-array lifetimes, not process/browser RSS. */
export interface RunRecordDensePerformance {
  /** Unique backing bytes for owned depth/normal/confidence maps; excludes RGB and GPU memory. */
  readonly planarHostMaps?: {
    readonly inputBytes: number;
    readonly peakRetainedBytes: number;
    readonly peakExtraOriginalViews: number;
    readonly retiredOriginalViews: number;
    readonly reorderedViews: number;
  };
  readonly tsdf?: {
    readonly blocks: number;
    readonly passes: number;
    readonly peakBytes: number;
    readonly uploadBytes: number;
    /** Extra host channels only; excludes borrowed inputs and runtime snapshots. */
    readonly hostViewStagingBytes?: number;
    readonly allocationPixels?: number;
    readonly reusedBoxTables?: number;
    readonly inlineVoteResetGroups?: number;
    /** Complete view payload admitted since confirmed completions. The effective limit can
     * expand to fit one oversized view; excludes other arrays, native allocations and RSS. */
    readonly queuedViewUploads?: {
      readonly limitBytes: number;
      readonly peakSubmittedBytes: number;
      readonly drainCount: number;
    };
    /** Synchronous wall sections, including upload enqueue; not CPU time or a total partition. */
    readonly hostTimings?: {
      readonly volumePlanningMs: number;
      readonly boxPlanningMs: number;
      readonly boxTableMs: number;
      readonly viewUploadMs: number;
      readonly sortingMs: number;
    };
  };
}

/**
 * The dense MVS of a v5 record (`docs/dense-mvs.md` 5.3), present when the run asked for it.
 * `status` is `completed`, `cancelled` (the visitor stopped it), `failed` (`failure` says why) or
 * `skipped` (`skipped` says why: more keyframes than the dense layout holds, or no kept image).
 */
export interface RunRecordDense {
  readonly status: 'completed' | 'cancelled' | 'failed' | 'skipped';
  readonly skipped: string;
  /** The MVS options of the lab (`LAB_OPTIONS.dense` of `lib/lab-options.ts`). */
  readonly options: {
    readonly maxSources: number;
    readonly bestSources: number;
    readonly windowRadius: number;
    readonly windowStep: number;
    readonly photometricIterations: number;
    readonly geometricIterations: number;
    readonly minConsistentViews: number;
    /** The cleanup of the fused cloud ran (section 3.12). Absent in a record before the cleanup. */
    readonly denseClean?: boolean;
    /** The fusion (3.10 or the TSDF of 10.4). Absent in a record before the TSDF: the pixel one. */
    readonly fusion?: 'pixel' | 'tsdf';
    /** The mesh of the final cloud was on (section 3.13). Absent in a record before the mesh. */
    readonly denseMesh?: boolean;
    readonly planarDepth?: boolean;
    readonly boundaryAware?: boolean;
    readonly minFiniteGain?: number;
    readonly tsdfVisibility?: 'ray-count' | 'view-weighted';
    /** Timestamp queries enabled on the actual Dense device, independent of whether any plan
     * produced a timed group. Absent when unavailable; preserves an observed disabled false. */
    readonly gpuTimestamps?: boolean;
  };
  /** Depth pixels filled by planes verified against two original source maps. */
  readonly planarFilled?: number;
  /** Absent in older records and multi-component sums; inspect each component's dense block. */
  readonly performance?: RunRecordDensePerformance;
  /**
   * The mesh of the final cloud (`docs/dense-mvs.md` 3.13): the counts and the time of each stage.
   * Absent when the mesh was off or did not run. `error` is empty unless the mesh failed.
   */
  readonly mesh?: {
    readonly tier: 'cpu' | 'webgpu';
    readonly points: number;
    readonly voxelSize: number;
    readonly radius: number;
    readonly nodes: number;
    readonly cubes: number;
    readonly vertices: number;
    readonly faces: number;
    readonly latticeMs: number;
    readonly fieldMs: number;
    readonly cubesMs: number;
    readonly verticesMs: number;
    readonly error: string;
  };
  /**
   * The cleanup of the fused cloud (`docs/dense-mvs.md` 3.12): the point count before and after
   * each stage, the thresholds it used and the time of each stage. Absent when the cleanup was off
   * or did not run. `error` is empty unless the cleanup failed, and the raw cloud then stays.
   */
  readonly clean?: {
    readonly tier: 'cpu' | 'webgpu';
    readonly before: number;
    readonly afterOutliers: number;
    readonly afterVoxel: number;
    readonly after: number;
    readonly outlierThreshold: number;
    readonly voxelSize: number;
    readonly eps: number;
    readonly clusters: number;
    readonly keptClusters: number;
    readonly outliersMs: number;
    readonly voxelMs: number;
    readonly clusterMs: number;
    readonly error: string;
  };
  /** The views with a depth map, and the registered views without one (no source or too few
   *  landmarks). Both 0 for a run that did not complete. */
  readonly views: number;
  readonly viewsWithoutDepth: number;
  /** The fused points: of the whole run, or posted before the cancel or the failure. */
  readonly points: number;
  /** The wall time of the MVS and of each stage in ms. A stage that did not end is 0. */
  readonly totalMs: number;
  readonly stageMs: Readonly<Record<(typeof RUN_RECORD_DENSE_STAGES)[number], number>>;
  /**
   * The device's bytes (4.2 and 4.3): `liveBytes` and `pooledBytes` before the first MVS
   * allocation, the active bytes (`liveBytes` less `pooledBytes`) at the depth stage's peak and
   * at the fusion stage, and the depth stage's formula: (16 N + 4) P before Amendment 3 of
   * `docs/dense-mvs.md`, and the bytes of its residency plan after it (the larger of its two
   * passes). Null when the stage was not reached.
   */
  readonly memory: {
    readonly liveBytesBefore: number;
    readonly pooledBytesBefore: number;
    readonly depthStageBytes: number | null;
    readonly depthStageFormulaBytes: number | null;
    readonly fusionStageBytes: number | null;
  };
  /** Null unless `status` is `failed`: the error, the stage, the kernel that started last and
   *  the device-loss reason. */
  readonly failure: {
    readonly message: string;
    readonly stage: string;
    readonly kernel: string;
    readonly reason: string;
    readonly context?: GpuLossContext;
  } | null;
}

/**
 * The KLT tracker of a run with `?klt=1` (`docs/klt-tracking.md` 5.2). Counts and times from
 * `KltTracker.stats()` and from the worker's merge. The times are in ms, summed over the run.
 */
export interface RunRecordKlt {
  /** Frames pushed and keyframes joined. */
  readonly pushes: number;
  readonly joins: number;
  /** Tracks alive at the end of the run. */
  readonly alive: number;
  /** Tracks that ended, by reason: alive (always 0), out of frame, low structure, no convergence,
   *  forward-backward, residual, epipolar, age. */
  readonly ended: readonly number[];
  /** Tracks that joined a keypoint, SIFT keypoints seeded, fill-in points added, and picks the
   *  host dropped for an alive track closer than the minimum distance. */
  readonly joined: number;
  readonly seeded: number;
  readonly filledIn: number;
  readonly filtered: number;
  /** Matches the joins emitted; the ones added to the pairs after the merge with the descriptor
   *  matches; the ones the descriptor matches held already; the ones left out as conflicting. */
  readonly matches: number;
  readonly pairMatches: number;
  readonly duplicateMatches: number;
  readonly conflictingMatches: number;
  /** Pairs that held tracker matches, and pairs that only the tracker found. */
  readonly pairs: number;
  readonly trackerOnlyPairs: number;
  /** Tracks removed on the verdict of the pair verification. */
  readonly rejected: number;
  /** Tracks by frames lived since the seed: under 2, 2, 3 to 4, 5 to 8, 9 to 16, 17 or more. */
  readonly lengths: readonly number[];
  readonly meanLived: number;
  /** Tracking and check, selection, and join (with its selection), in ms. */
  readonly msPush: number;
  readonly msSelect: number;
  readonly msJoin: number;
}

/** The most videos, components and not-connected videos a record lists (`docs/multi-video.md` 6.4). */
export const RUN_RECORD_MAX_VIDEOS = 64;
/** The tags of a video row (6.3). `component N` is `component` with `component` set. */
export const RUN_RECORD_VIDEO_TAGS = [
  'component',
  'alone',
  'not connected',
  'rotation',
  'duplicate',
] as const;

/** One video of a multi-video run (`docs/multi-video.md` 6.4, `input.videos`). */
export interface RunRecordVideo {
  /** The content hash of 2.4: 16 hex digits. */
  readonly videoId: string;
  /** The sample id for a lab sample; null for an upload (the typed name is never sent). */
  readonly label: string | null;
  readonly durationSeconds: number;
  /** The selected source-time interval; absent when the whole video was analysed. */
  readonly range?: VideoRange;
  readonly keyframes: number;
  /** Keyframes of this video in a registered view. */
  readonly registeredViews: number;
  /** The frames of this video that the run read, kept or not. Absent in earlier records. */
  readonly frames?: number;
  /** The analysis size of this video in pixels. Absent in earlier records. */
  readonly width?: number;
  readonly height?: number;
  readonly focalRatio: number;
  /** The component of the video (0-based, the rank of 4.1), or null without one. */
  readonly component: number | null;
  readonly tags: readonly (typeof RUN_RECORD_VIDEO_TAGS)[number][];
}

/** `options.multiVideo` (6.4). */
export interface RunRecordMultiVideoOptions {
  /** The global descriptor: `vlad-32-v1` by default. */
  readonly descriptor: string;
  readonly k: number;
  readonly minMatches: number;
  readonly minEdgeInliers: number;
  readonly maxKeyframesPerVideo: number;
  readonly growth: 'incremental' | 'submaps';
  /** `?canonical=0` runs with view ids in arrival order. Absent means canonical. */
  readonly canonical?: boolean;
  /** One focal group for each video. The WebGPU path runs one shared focal (Amendment 4). */
  readonly focalPerVideo?: boolean;
}

/** One component of a multi-video result (6.4, `result.components`). */
export interface RunRecordComponent {
  readonly keyframes: number;
  readonly registeredViews: number;
  readonly landmarks: number;
  readonly videos: number;
  readonly meanReprojectionErrorPx: number | null;
  /**
   * The dense MVS of this component (`docs/multi-video.md`, Amendment 9). The record's `dense`
   * block is the sum of these. Absent when the run did not ask for the dense MVS.
   */
  readonly dense?: RunRecordDense;
}

/** `result.viewGraph` (6.4): the graph against exhaustive matching. */
export interface RunRecordViewGraph {
  readonly nodes: number;
  readonly candidatePairs: number;
  readonly matchedPairs: number;
  readonly verifiedEdges: number;
  readonly exhaustivePairs: number;
}

export interface RunRecordChunk {
  readonly msPerUnit: number;
  readonly chunks: number;
  readonly longestMs: number;
  readonly lastUnits: number;
}
export interface RunRecord {
  readonly format: (typeof RUN_RECORD_FORMATS)[number];
  /** When the lab built the record, ISO 8601. */
  readonly createdAt: string;
  readonly site: { readonly commit: string; readonly runtime: string };
  readonly client: {
    readonly userAgent: string;
    readonly platform: string;
    readonly language: string;
    readonly hardwareConcurrency: number;
    readonly deviceMemoryGb: number | null;
    readonly screen: { readonly width: number; readonly height: number; readonly dpr: number };
  };
  readonly gpu: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly fallbackAdapter: boolean;
    readonly limits: {
      readonly maxStorageBufferBindingSize: number;
      readonly maxBufferSize: number;
      /** v6. The compute limits of the adapter; 0 when the page could not read them. */
      readonly maxComputeWorkgroupStorageSize?: number;
      readonly maxComputeInvocationsPerWorkgroup?: number;
      readonly maxStorageBuffersPerShaderStage?: number;
    };
    /** v6. The adapter's feature names (`shader-f16`, `subgroups`, `timestamp-query`, ...), sorted. */
    readonly features?: readonly string[];
  };
  readonly input: {
    readonly kind: 'sample' | 'prepared' | 'video';
    /** The sample's name only; uploaded inputs carry no name. */
    readonly name: string | null;
    readonly frames: number;
    readonly width: number;
    readonly height: number;
    readonly sourceWidth: number;
    readonly sourceHeight: number;
    readonly durationSeconds: number;
    readonly range?: VideoRange;
    /** A multi-video run: its videos (6.4). Absent for one video and in earlier records. */
    readonly videos?: readonly RunRecordVideo[];
  };
  readonly options: {
    /** `corner`: FAST/BRIEF pyramid features; `sift`: scale-space SIFT with f32 descriptors. */
    readonly frontEnd: 'corner' | 'sift';
    readonly focalRatio: number;
    readonly windowSize: number;
    readonly ransacIterations: number;
    readonly maxErrorPx: number;
    readonly bundleIterations: number;
    readonly finalBundleIterations: number;
    readonly refineSharedFocal: boolean;
    /** The automatic focal estimate ran (`docs/auto-focal.md`); `focalRatio` was then only the
     *  start. Absent in records before 2026-10-21 and for a run with a typed focal. */
    readonly focalSearch?: boolean;
    readonly maxFeatures: number;
    readonly matchRatio: number;
    readonly maxDistance: number;
    /** The device fences every submission (the lab's `?bounded=1` diagnostic mode). */
    readonly boundedSubmissions: boolean;
    /** The flavor of TypeShade's emulated double the kernels ran (change 0073): the device's
     *  choice from the adapter, or the lab's `?fp64=` override. */
    readonly fp64Flavor: 'float' | 'integer';
    /** Guided matching (the lab's `?guided=1`). Absent in records before 2026-10-20 (off). */
    readonly guidedMatching?: boolean;
    /** The WebGPU kernel of guided matching (the lab's `?guided=tiled`); absent means scalar. */
    readonly guidedKernel?: 'tiled' | 'scalar';
    /** SIFT's first octave: -1 when the frames were upsampled 2× first (the lab's `?octave=-1`).
     *  Absent in records before 2026-10-20, which ran octave 0. */
    readonly siftFirstOctave?: -1 | 0;
    readonly clahe?: boolean;
    /** Analysis frames per second taken from an uploaded video before SIFT (the lab's `?fps=`).
     *  Absent for photo and sample inputs and in records before 2026-10-20 (every frame). */
    readonly videoSampleFps?: number;
    /** Where an uploaded video was decoded: a dedicated worker or the page's main thread.
     *  Absent for photo and sample inputs and in records before 2026-10-20 (main thread). */
    readonly videoDecode?: 'worker' | 'main-thread';
    /** Where the frames of a video sample came from: its source video decoded by byte range
     *  (`source`), its JPEG frame files (`frames`, the fallback when the browser cannot decode
     *  the codec), or both in one multi-video run (`mixed`). Absent for an uploaded video, a
     *  photo set, a sample that stores no source video and in records before 2026-10-21. */
    readonly videoSource?: 'source' | 'frames' | 'mixed';
    /** The keyframe gate of the run: `phase` (phase correlation before SIFT, `?gate=phase`) or
     *  `legacy` (the median match displacement after SIFT, `?gate=legacy`). Absent in records
     *  before 2026-10-21, which ran the legacy gate. */
    readonly keyframeGate?: 'phase' | 'legacy';
    /** The feature front end: `orb` for a run with `?features=orb` (FAST corners, rBRIEF
     *  descriptors, Hamming matching; `frontEnd` is `corner` then). Absent means SIFT, as in every
     *  record before 2026-10-21. */
    readonly features?: 'orb';
    /** Frames the gate kept as keyframes, of `input.frames`. Absent in records before 2026-10-21. */
    readonly keyframes?: number;
    /** KLT tracking (the lab's `?klt=1`, `docs/klt-tracking.md`): true when the worker tracked
     *  points and joined them to the keyframes. Absent in records before 2026-10-21 (off). */
    readonly klt?: boolean;
    /** A multi-video run (`docs/multi-video.md` 6.4). Absent for one video. */
    readonly multiVideo?: RunRecordMultiVideoOptions;
    /** The step 6 switches of a KLT run (`kltSetup` in `sift-quality.ts`). Absent when none was set. */
    readonly kltSetup?: {
      readonly capacity?: number;
      readonly fillIn?: true;
      readonly matching?: 'tracks';
      readonly gate?: true;
    };
  };
  /** For a run that did not complete: the counts of the last map snapshot, if any. */
  readonly result: {
    readonly registeredViews: number;
    readonly landmarks: number;
    readonly observations: number;
    readonly meanReprojectionErrorPx: number | null;
    readonly unregistered: number;
    readonly matchedPairs: number;
    /** With guided matching: the pairs matched again and the matches the guide added. */
    readonly guided?: { readonly pairs: number; readonly addedMatches: number };
    readonly featureCounts: readonly number[];
    /** Optional scoped lifetimes/work from the multi-video pipeline; absent in older runs. */
    readonly hostRetention?: RunRecordHostRetention;
    readonly descriptorCache?: RunRecordDescriptorCache;
    /** The focal search's estimate (focal / width): after the parabola (`gridRatio`), after the
     *  joint refinement on its subset (`ratio`), and the views of the subset. */
    readonly focalSearch?: {
      readonly ratio: number;
      readonly gridRatio: number;
      readonly views: number;
    };
    /** Present for a run that tracked with KLT (`options.klt`). */
    readonly klt?: RunRecordKlt;
    /** A multi-video run: one entry for each component (6.4). Absent for one video. */
    readonly components?: readonly RunRecordComponent[];
    /** The accepted edges between two videos and the sum of their inliers (6.4). */
    readonly crossVideoEdges?: { readonly edges: number; readonly inliers: number };
    readonly viewGraph?: RunRecordViewGraph;
    /** The `videoId`s that join no component of two or more videos (6.4). */
    readonly notConnected?: readonly string[];
  };
  readonly timing: {
    readonly totalMs: number;
    readonly reusedDevice: boolean;
    readonly stages: Readonly<Record<string, RunRecordStage>>;
    /** Per analysis frame, in processing order. */
    readonly frameMs: readonly number[];
    /** The GPU kernels the run compiled or ran, in first-use order; the last one started last. */
    readonly kernels: readonly string[];
    /**
     * v6. The GPU time of each fused call list the run executed, in first-use order, at most
     * `RUN_RECORD_MAX_GPU_ROWS` rows, from the device's collector (`ShadeDevice.statistics.kernels`).
     * The rows count groups and dispatches on a device without `timestamp-query`, and then every
     * `gpuMs` is null. For a run with the dense MVS the rows cover the sparse run and the MVS.
     */
    readonly gpu?: readonly RunRecordKernelTime[];
    /** v6. True when at least one group of the run carried a GPU time (`timestamp-query`). */
    readonly gpuTimestamps?: boolean;
    /** v6. Call lists the collector did not keep because it was full. */
    readonly gpuDropped?: number;
    /**
     * Absent in v3. What the device measured of each chunk key (`scorePairModels`, `scorePnP`,
     * `bundleCg`, `bundleCachedCg`, `triangulateTracks`) when the run completed or failed. The device keeps these over its life, so
     * on a reused device (`reusedDevice`) they include the earlier runs on it.
     */
    readonly chunks?: Readonly<Record<string, RunRecordChunk>>;
    /** Absent in v3. The device's time budget per submission in ms, null without a device. */
    readonly budgetMs?: number | null;
    /**
     * Absent in v3 and in a v4 record from a page deployed before 2026-10-21. What the page's
     * main thread did during the run (`website/src/lib/main-thread.ts`): long tasks and frame
     * gaps, with the lab's stage at the longest of each. The SfM job runs in a worker, so a
     * stall here is the page's own work or the browser's, not a kernel.
     */
    readonly mainThread?: RunRecordMainThread;
  };
  /** Absent in v3 and v4, and in a v5 record of a run without the dense MVS. */
  readonly dense?: RunRecordDense;
  /**
   * Absent when the run sent no result files. The files that the page declared with the record:
   * the collector stores a file by `PUT /api/records/<id>/files/<name>` only when its bytes and
   * SHA-256 match the entry. `skipped` is the reason the page did not upload the file.
   */
  readonly files?: readonly RunRecordFile[];
  /** A bounded attachment-planning summary when its manifest could not fit beside this record. */
  readonly filePreparationFailure?: {
    readonly reason: string;
    readonly requested: number;
  };
  readonly transfers: {
    readonly applicationUploadBytes: number;
    readonly applicationReadbackBytes: number;
  };
  readonly benchmark: { readonly name: string; readonly pass: 'cold' | 'warm' } | null;
  /**
   * Agreement with the reference cameras a sample ships (the COLMAP model of the
   * south-building sample): the lab's registered cameras similarity-aligned to the
   * reference on their common views. Null when the input has no reference.
   */
  readonly reference: {
    readonly name: string;
    /** `aligned`, or the reason no alignment was possible. */
    readonly status: string;
    readonly commonViews: number;
    readonly medianRotationDeg: number | null;
    readonly maxRotationDeg: number | null;
    /** Camera center error as a fraction of the reference trajectory's radius. */
    readonly medianCenterOfRadius: number | null;
    readonly maxCenterOfRadius: number | null;
  } | null;
  /**
   * Null for a completed run. Otherwise the error that ended it: the worker's message, the
   * mapper operation and the last kernel in the current request (empty while idle or before
   * its first kernel), the device-loss reason the browser gave (`destroyed`, `unknown`, or empty) and the
   * number of frames whose features were extracted before the failure.
   */
  readonly failure: {
    readonly message: string;
    readonly operation: string;
    readonly kernel: string;
    readonly reason: string;
    readonly framesProcessed: number;
    readonly context?: GpuLossContext;
    readonly initialization?: InitializationFailureRecord;
  } | null;
  readonly note: string;
}
