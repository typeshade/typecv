import { gpuLossContextValid, type GpuLossContext } from './gpu-loss-context.ts';
import { videoRangeValid, type VideoRange } from './video-range.ts';
import {
  initializationFailureValid,
  type InitializationFailureRecord,
} from './initialization-failure.ts';

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
export type ResultFileName =
  | (typeof RESULT_FILE_NAMES)[number]
  | `typecv-depths-c${string}.json`
  | `typecv-depths-c${string}-${string}.tar.gz`;
export const RESULT_FILE_MAX_COUNT = 1024;
export const RESULT_FILES_MAX_BYTES = 4 * 1024 ** 3;
export function resultFileNameValid(name: unknown): name is ResultFileName {
  return (
    typeof name === 'string' &&
    ((RESULT_FILE_NAMES as readonly string[]).includes(name) ||
      /^typecv-depths-c[0-9]{3}(?:-[0-9]{4}\.tar\.gz|\.json)$/.test(name))
  );
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
  readonly bytes: number;
  /** Lowercase hex SHA-256 of the file. */
  readonly sha256: string;
  /** Why the page did not upload the file; absent when it uploads it. */
  readonly skipped?: string;
};

/** The stages of the dense MVS whose elapsed time a record holds (`docs/dense-mvs.md` 5.3). */
export const RUN_RECORD_DENSE_STAGES = ['photometric', 'geometric', 'fusion'] as const;

/**
 * The dense MVS of a v5 record (`docs/dense-mvs.md` 5.3), present when the run asked for it.
 * `status` is `completed`, `cancelled` (the visitor stopped it), `failed` (`failure` says why) or
 * `skipped` (`skipped` says why: more keyframes than the dense layout holds, or no kept image).
 */
export interface RunRecordDense {
  readonly status: 'completed' | 'cancelled' | 'failed' | 'skipped';
  readonly skipped: string;
  /** The MVS options of the lab (`LAB_OPTIONS.dense` of the worker). */
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
    readonly tsdfVisibility?: 'ray-count' | 'view-weighted';
  };
  /** Depth pixels filled by planes verified against two original source maps. */
  readonly planarFilled?: number;
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

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
const text = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.length <= max;

const positive = (value: unknown): value is number => finite(value) && value > 0;
function chunksValid(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const entries = Object.entries(value as Record<string, RunRecordChunk>);
  return (
    entries.length <= RUN_RECORD_MAX_CHUNK_KEYS &&
    entries.every(
      ([key, rate]) =>
        text(key, 64) &&
        key.length > 0 &&
        rate &&
        typeof rate === 'object' &&
        finite(rate.msPerUnit) &&
        rate.msPerUnit >= 0 &&
        count(rate.chunks) &&
        finite(rate.longestMs) &&
        rate.longestMs >= 0 &&
        count(rate.lastUnits),
    )
  );
}

function cleanValid(value: unknown): boolean {
  if (value === undefined) return true;
  const c = value as NonNullable<RunRecordDense['clean']>;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    !!c &&
    typeof c === 'object' &&
    (c.tier === 'cpu' || c.tier === 'webgpu') &&
    count(c.before) &&
    count(c.afterOutliers) &&
    count(c.afterVoxel) &&
    count(c.after) &&
    ms(c.outlierThreshold) &&
    ms(c.voxelSize) &&
    ms(c.eps) &&
    count(c.clusters) &&
    count(c.keptClusters) &&
    ms(c.outliersMs) &&
    ms(c.voxelMs) &&
    ms(c.clusterMs) &&
    text(c.error, 200)
  );
}

function meshValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as NonNullable<RunRecordDense['mesh']>;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    !!m &&
    typeof m === 'object' &&
    (m.tier === 'cpu' || m.tier === 'webgpu') &&
    count(m.points) &&
    ms(m.voxelSize) &&
    ms(m.radius) &&
    count(m.nodes) &&
    count(m.cubes) &&
    count(m.vertices) &&
    count(m.faces) &&
    ms(m.latticeMs) &&
    ms(m.fieldMs) &&
    ms(m.cubesMs) &&
    ms(m.verticesMs) &&
    text(m.error, 200)
  );
}

function kltSetupValid(value: unknown): boolean {
  const k = value as NonNullable<RunRecord['options']['kltSetup']> | null;
  return (
    typeof k === 'object' &&
    k !== null &&
    (k.capacity === undefined ||
      (Number.isInteger(k.capacity) && k.capacity >= 1 && k.capacity <= 8192)) &&
    (k.fillIn === undefined || k.fillIn === true) &&
    (k.matching === undefined || k.matching === 'tracks') &&
    (k.gate === undefined || k.gate === true)
  );
}

function kltValid(value: unknown): boolean {
  const k = value as RunRecordKlt;
  if (!k || typeof k !== 'object') return false;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    count(k.pushes) &&
    count(k.joins) &&
    count(k.alive) &&
    Array.isArray(k.ended) &&
    k.ended.length === 8 &&
    k.ended.every(count) &&
    count(k.joined) &&
    count(k.seeded) &&
    count(k.filledIn) &&
    count(k.filtered) &&
    count(k.matches) &&
    count(k.pairMatches) &&
    count(k.duplicateMatches) &&
    count(k.conflictingMatches) &&
    count(k.pairs) &&
    count(k.trackerOnlyPairs) &&
    count(k.rejected) &&
    Array.isArray(k.lengths) &&
    k.lengths.length === 6 &&
    k.lengths.every(count) &&
    ms(k.meanLived) &&
    ms(k.msPush) &&
    ms(k.msSelect) &&
    ms(k.msJoin)
  );
}

function denseValid(value: unknown): boolean {
  const d = value as RunRecordDense;
  if (!d || typeof d !== 'object') return false;
  const o = d.options,
    m = d.memory,
    f = d.failure;
  const bytes = (v: unknown) => v === null || count(v);
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    ['completed', 'cancelled', 'failed', 'skipped'].includes(d.status) &&
    text(d.skipped, 200) &&
    !!o &&
    typeof o === 'object' &&
    count(o.maxSources) &&
    count(o.bestSources) &&
    count(o.windowRadius) &&
    count(o.windowStep) &&
    count(o.photometricIterations) &&
    count(o.geometricIterations) &&
    count(o.minConsistentViews) &&
    (o.denseClean === undefined || typeof o.denseClean === 'boolean') &&
    (o.fusion === undefined || o.fusion === 'pixel' || o.fusion === 'tsdf') &&
    (o.denseMesh === undefined || typeof o.denseMesh === 'boolean') &&
    (o.planarDepth === undefined || typeof o.planarDepth === 'boolean') &&
    (o.tsdfVisibility === undefined ||
      o.tsdfVisibility === 'ray-count' ||
      o.tsdfVisibility === 'view-weighted') &&
    (d.planarFilled === undefined || count(d.planarFilled)) &&
    cleanValid(d.clean) &&
    meshValid(d.mesh) &&
    count(d.views) &&
    count(d.viewsWithoutDepth) &&
    count(d.points) &&
    ms(d.totalMs) &&
    !!d.stageMs &&
    typeof d.stageMs === 'object' &&
    Object.keys(d.stageMs).length === RUN_RECORD_DENSE_STAGES.length &&
    RUN_RECORD_DENSE_STAGES.every((stage) => ms(d.stageMs[stage])) &&
    !!m &&
    typeof m === 'object' &&
    count(m.liveBytesBefore) &&
    count(m.pooledBytesBefore) &&
    bytes(m.depthStageBytes) &&
    bytes(m.depthStageFormulaBytes) &&
    bytes(m.fusionStageBytes) &&
    (d.status === 'failed'
      ? !!f &&
        typeof f === 'object' &&
        text(f.message, 2000) &&
        text(f.stage, 64) &&
        text(f.kernel, 64) &&
        text(f.reason, 64) &&
        (f.context === undefined || gpuLossContextValid(f.context))
      : f === null)
  );
}

/** Return the reason a value is not a run record, or undefined when it is one. */
function mainThreadValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as Record<string, unknown>;
  const measure = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  const stage = (v: unknown) => typeof v === 'string' && v.length <= 128;
  return (
    !!m &&
    typeof m === 'object' &&
    Number.isSafeInteger(m.longTasks) &&
    (m.longTasks as number) >= 0 &&
    measure(m.longTaskMs) &&
    measure(m.longestTaskMs) &&
    stage(m.longestTaskStage) &&
    Number.isSafeInteger(m.frameGaps) &&
    (m.frameGaps as number) >= 0 &&
    measure(m.frameGapMs) &&
    measure(m.longestGapMs) &&
    stage(m.longestGapStage) &&
    (m.hiddenMs === undefined || measure(m.hiddenMs))
  );
}

function filesValid(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value) || value.length > RESULT_FILE_MAX_COUNT) return false;
  const names = new Set<unknown>();
  for (const file of value) {
    if (!file || typeof file !== 'object') return false;
    const f = file as Record<string, unknown>;
    if (
      !resultFileNameValid(f.name) ||
      names.has(f.name) ||
      !Number.isSafeInteger(f.bytes) ||
      (f.bytes as number) < 0 ||
      typeof f.sha256 !== 'string' ||
      !/^[0-9a-f]{64}$/.test(f.sha256) ||
      (f.skipped !== undefined && (!text(f.skipped, 200) || f.skipped === ''))
    )
      return false;
    names.add(f.name);
  }
  return value.reduce((sum, f) => sum + (f.skipped ? 0 : f.bytes), 0) <= RESULT_FILES_MAX_BYTES;
}

export function runRecordProblem(value: unknown): string | undefined {
  const r = value as RunRecord;
  if (!r || typeof r !== 'object') return 'not an object';
  if (!(RUN_RECORD_FORMATS as readonly unknown[]).includes(r.format)) return 'format';
  if (!text(r.createdAt, 40) || Number.isNaN(Date.parse(r.createdAt))) return 'createdAt';
  if (!r.site || !text(r.site.commit, 64) || !text(r.site.runtime, 200)) return 'site';
  const c = r.client;
  if (
    !c ||
    !text(c.userAgent, 512) ||
    !text(c.platform, 64) ||
    !text(c.language, 32) ||
    !count(c.hardwareConcurrency) ||
    !(c.deviceMemoryGb === null || finite(c.deviceMemoryGb)) ||
    !c.screen ||
    !count(c.screen.width) ||
    !count(c.screen.height) ||
    !finite(c.screen.dpr)
  )
    return 'client';
  const g = r.gpu;
  if (
    !g ||
    !text(g.vendor, 128) ||
    !text(g.architecture, 128) ||
    !text(g.device, 128) ||
    !text(g.description, 512) ||
    typeof g.fallbackAdapter !== 'boolean' ||
    !g.limits ||
    !count(g.limits.maxStorageBufferBindingSize) ||
    !count(g.limits.maxBufferSize)
  )
    return 'gpu';
  if (!gpuV6Valid(r.format === RUN_RECORD_FORMAT, g)) return 'gpu';
  const i = r.input;
  if (
    !i ||
    !['sample', 'prepared', 'video'].includes(i.kind) ||
    !(i.name === null || text(i.name, 128)) ||
    !count(i.frames) ||
    i.frames > RUN_RECORD_MAX_FRAMES ||
    !count(i.width) ||
    !count(i.height) ||
    !count(i.sourceWidth) ||
    !count(i.sourceHeight) ||
    !finite(i.durationSeconds) ||
    i.durationSeconds < 0 ||
    (i.range !== undefined && !videoRangeValid(i.range, i.durationSeconds)) ||
    !videosValid(i.videos)
  )
    return 'input';
  const o = r.options;
  if (
    !o ||
    !finite(o.focalRatio) ||
    !count(o.windowSize) ||
    !count(o.ransacIterations) ||
    !finite(o.maxErrorPx) ||
    !count(o.bundleIterations) ||
    !count(o.finalBundleIterations) ||
    typeof o.refineSharedFocal !== 'boolean' ||
    (o.focalSearch !== undefined && typeof o.focalSearch !== 'boolean') ||
    !count(o.maxFeatures) ||
    !finite(o.matchRatio) ||
    !finite(o.maxDistance) ||
    o.maxDistance < 0 ||
    !['corner', 'sift'].includes(o.frontEnd) ||
    typeof o.boundedSubmissions !== 'boolean' ||
    !['float', 'integer'].includes(o.fp64Flavor) ||
    (o.guidedMatching !== undefined && typeof o.guidedMatching !== 'boolean') ||
    (o.guidedKernel !== undefined && o.guidedKernel !== 'tiled' && o.guidedKernel !== 'scalar') ||
    (o.siftFirstOctave !== undefined && o.siftFirstOctave !== -1 && o.siftFirstOctave !== 0) ||
    (o.clahe !== undefined && typeof o.clahe !== 'boolean') ||
    (o.videoSampleFps !== undefined && (!finite(o.videoSampleFps) || o.videoSampleFps <= 0)) ||
    (o.videoDecode !== undefined &&
      o.videoDecode !== 'worker' &&
      o.videoDecode !== 'main-thread') ||
    (o.videoSource !== undefined &&
      o.videoSource !== 'source' &&
      o.videoSource !== 'frames' &&
      o.videoSource !== 'mixed') ||
    (o.keyframeGate !== undefined && o.keyframeGate !== 'phase' && o.keyframeGate !== 'legacy') ||
    (o.features !== undefined && o.features !== 'orb') ||
    (o.keyframes !== undefined && !count(o.keyframes)) ||
    (o.klt !== undefined && typeof o.klt !== 'boolean') ||
    (o.kltSetup !== undefined && !kltSetupValid(o.kltSetup)) ||
    !multiVideoOptionsValid(o.multiVideo)
  )
    return 'options';
  const s = r.result;
  if (
    !s ||
    !count(s.registeredViews) ||
    !count(s.landmarks) ||
    !count(s.observations) ||
    !(s.meanReprojectionErrorPx === null || finite(s.meanReprojectionErrorPx)) ||
    !count(s.unregistered) ||
    !count(s.matchedPairs) ||
    !Array.isArray(s.featureCounts) ||
    s.featureCounts.length > RUN_RECORD_MAX_FRAMES ||
    !s.featureCounts.every(count) ||
    (s.klt !== undefined && !kltValid(s.klt)) ||
    (s.focalSearch !== undefined &&
      (!s.focalSearch ||
        !(finite(s.focalSearch.ratio) && s.focalSearch.ratio > 0) ||
        !(finite(s.focalSearch.gridRatio) && s.focalSearch.gridRatio > 0) ||
        !count(s.focalSearch.views))) ||
    !multiVideoResultValid(s)
  )
    return 'result';
  const t = r.timing;
  if (
    !t ||
    !finite(t.totalMs) ||
    typeof t.reusedDevice !== 'boolean' ||
    !t.stages ||
    typeof t.stages !== 'object' ||
    Object.keys(t.stages).length > 64 ||
    !Object.entries(t.stages).every(
      ([name, stage]) =>
        text(name, 64) &&
        stage &&
        count(stage.calls) &&
        finite(stage.elapsedMs) &&
        count(stage.dispatches) &&
        count(stage.submissions) &&
        (stage.gpuMs === undefined
          ? true
          : r.format === RUN_RECORD_FORMAT && finite(stage.gpuMs) && stage.gpuMs >= 0),
    ) ||
    !Array.isArray(t.frameMs) ||
    t.frameMs.length > RUN_RECORD_MAX_FRAMES ||
    !t.frameMs.every(finite) ||
    !Array.isArray(t.kernels) ||
    t.kernels.length > RUN_RECORD_MAX_KERNELS ||
    !t.kernels.every((name) => text(name, 64)) ||
    !gpuTimeValid(r.format === RUN_RECORD_FORMAT, t) ||
    (r.format !== 'typecv-run-record-v3'
      ? !chunksValid(t.chunks) ||
        !(t.budgetMs === null || positive(t.budgetMs)) ||
        !mainThreadValid(t.mainThread)
      : t.chunks !== undefined || t.budgetMs !== undefined || t.mainThread !== undefined)
  )
    return 'timing';
  const x = r.transfers;
  if (!x || !count(x.applicationUploadBytes) || !count(x.applicationReadbackBytes))
    return 'transfers';
  if (!(
    r.benchmark === null ||
    (r.benchmark &&
      text(r.benchmark.name, 64) &&
      (r.benchmark.pass === 'cold' || r.benchmark.pass === 'warm'))
  ))
    return 'benchmark';
  const f = r.reference;
  const measure = (value: unknown) => value === null || (finite(value) && (value as number) >= 0);
  if (!(
    f === null ||
    (f &&
      text(f.name, 128) &&
      text(f.status, 128) &&
      count(f.commonViews) &&
      measure(f.medianRotationDeg) &&
      measure(f.maxRotationDeg) &&
      measure(f.medianCenterOfRadius) &&
      measure(f.maxCenterOfRadius))
  ))
    return 'reference';
  const e = r.failure;
  if (!(
    e === null ||
    (e &&
      text(e.message, 2000) &&
      text(e.operation, 200) &&
      text(e.kernel, 64) &&
      text(e.reason, 64) &&
      count(e.framesProcessed) &&
      (e.context === undefined || gpuLossContextValid(e.context)) &&
      (e.initialization === undefined || initializationFailureValid(e.initialization)))
  ))
    return 'failure';
  if (!text(r.note, RUN_RECORD_MAX_NOTE)) return 'note';
  if (
    r.format === RUN_RECORD_FORMAT || r.format === 'typecv-run-record-v5'
      ? r.dense !== undefined && !denseValid(r.dense)
      : r.dense !== undefined
  )
    return 'dense';
  if (!filesValid(r.files)) return 'files';
  return undefined;
}

/** `gpu.features` and the compute limits: required in a v6 record, absent before it. */
function gpuV6Valid(current: boolean, g: RunRecord['gpu']): boolean {
  const limits = g.limits;
  const added = [
    limits.maxComputeWorkgroupStorageSize,
    limits.maxComputeInvocationsPerWorkgroup,
    limits.maxStorageBuffersPerShaderStage,
  ];
  if (!current) return g.features === undefined && added.every((v) => v === undefined);
  return (
    Array.isArray(g.features) &&
    g.features.length <= RUN_RECORD_MAX_FEATURES &&
    g.features.every((name) => text(name, 64)) &&
    added.every(count)
  );
}

/** `timing.gpu`, `gpuTimestamps` and `gpuDropped`: required in a v6 record, absent before it. */
function gpuTimeValid(current: boolean, t: RunRecord['timing']): boolean {
  if (!current)
    return t.gpu === undefined && t.gpuTimestamps === undefined && t.gpuDropped === undefined;
  return (
    Array.isArray(t.gpu) &&
    t.gpu.length <= RUN_RECORD_MAX_GPU_ROWS &&
    t.gpu.every(
      (row) =>
        !!row &&
        typeof row === 'object' &&
        text(row.name, RUN_RECORD_MAX_GPU_NAME) &&
        row.name.length > 0 &&
        count(row.groups) &&
        count(row.dispatches) &&
        count(row.timedGroups) &&
        row.timedGroups <= row.groups &&
        (row.gpuMs === null ? row.timedGroups === 0 : finite(row.gpuMs) && row.gpuMs >= 0),
    ) &&
    typeof t.gpuTimestamps === 'boolean' &&
    count(t.gpuDropped)
  );
}

const VIDEO_ID = /^[0-9a-f]{16}$/;

function videosValid(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value) || value.length > RUN_RECORD_MAX_VIDEOS) return false;
  return (value as RunRecordVideo[]).every(
    (v) =>
      !!v &&
      typeof v === 'object' &&
      typeof v.videoId === 'string' &&
      VIDEO_ID.test(v.videoId) &&
      (v.label === null || (text(v.label, 64) && v.label.length > 0)) &&
      finite(v.durationSeconds) &&
      v.durationSeconds >= 0 &&
      (v.range === undefined || videoRangeValid(v.range, v.durationSeconds)) &&
      count(v.keyframes) &&
      count(v.registeredViews) &&
      v.registeredViews <= v.keyframes &&
      (v.frames === undefined || (count(v.frames) && v.frames >= v.keyframes)) &&
      (v.width === undefined || count(v.width)) &&
      (v.height === undefined || count(v.height)) &&
      positive(v.focalRatio) &&
      (v.component === null || count(v.component)) &&
      Array.isArray(v.tags) &&
      v.tags.length <= 5 &&
      v.tags.every((t) => (RUN_RECORD_VIDEO_TAGS as readonly unknown[]).includes(t)),
  );
}

function multiVideoOptionsValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as RunRecordMultiVideoOptions;
  return (
    !!m &&
    typeof m === 'object' &&
    text(m.descriptor, 32) &&
    m.descriptor.length > 0 &&
    count(m.k) &&
    count(m.minMatches) &&
    count(m.minEdgeInliers) &&
    count(m.maxKeyframesPerVideo) &&
    (m.growth === 'incremental' || m.growth === 'submaps') &&
    (m.canonical === undefined || typeof m.canonical === 'boolean') &&
    (m.focalPerVideo === undefined || typeof m.focalPerVideo === 'boolean')
  );
}

function multiVideoResultValid(s: RunRecord['result']): boolean {
  const components = s.components;
  if (
    components !== undefined &&
    !(
      Array.isArray(components) &&
      components.length <= RUN_RECORD_MAX_VIDEOS &&
      components.every(
        (c) =>
          !!c &&
          typeof c === 'object' &&
          count(c.keyframes) &&
          count(c.registeredViews) &&
          c.registeredViews <= c.keyframes &&
          count(c.landmarks) &&
          count(c.videos) &&
          (c.meanReprojectionErrorPx === null || finite(c.meanReprojectionErrorPx)) &&
          (c.dense === undefined || denseValid(c.dense)),
      )
    )
  )
    return false;
  const x = s.crossVideoEdges;
  if (x !== undefined && !(!!x && typeof x === 'object' && count(x.edges) && count(x.inliers)))
    return false;
  const g = s.viewGraph;
  if (
    g !== undefined &&
    !(
      !!g &&
      typeof g === 'object' &&
      count(g.nodes) &&
      count(g.candidatePairs) &&
      count(g.matchedPairs) &&
      count(g.verifiedEdges) &&
      count(g.exhaustivePairs)
    )
  )
    return false;
  const n = s.notConnected;
  return (
    n === undefined ||
    (Array.isArray(n) &&
      n.length <= RUN_RECORD_MAX_VIDEOS &&
      n.every((id) => typeof id === 'string' && VIDEO_ID.test(id)))
  );
}

/**
 * The GPU time of one report in ms, summed from the groups that carry a `gpuTime` (nanoseconds),
 * or undefined when no group does: a device without `timestamp-query` reports no time, not 0.
 */
export function reportGpuMs(report: unknown): number | undefined {
  const groups = (report as { groups?: readonly { gpuTime?: unknown }[] } | undefined)?.groups;
  if (!Array.isArray(groups)) return undefined;
  let ns = 0,
    timed = false;
  for (const group of groups)
    if (
      typeof group?.gpuTime === 'number' &&
      Number.isFinite(group.gpuTime) &&
      group.gpuTime >= 0
    ) {
      ns += group.gpuTime;
      timed = true;
    }
  return timed ? Math.round(ns / 1e3) / 1e3 : undefined;
}

/**
 * The record's `timing.gpu` from the rows the worker sends (`ShadeDevice.statistics.kernels`):
 * names cut to `RUN_RECORD_MAX_GPU_NAME` (rows that then share a name are summed), at most
 * `RUN_RECORD_MAX_GPU_ROWS` rows, `gpuMs` in ms with microsecond precision, null without a timed
 * group. `timestamps` is whether any group was timed.
 */
export function gpuRecord(
  rows: unknown,
  dropped: unknown = 0,
): { gpu: RunRecordKernelTime[]; gpuTimestamps: boolean; gpuDropped: number } {
  const merged = new Map<
    string,
    { groups: number; dispatches: number; timed: number; ns: number }
  >();
  let lost = count(dropped) ? dropped : 0;
  if (Array.isArray(rows))
    for (const row of rows) {
      if (!row || typeof row.name !== 'string' || row.name.length === 0) continue;
      const name = row.name.slice(0, RUN_RECORD_MAX_GPU_NAME);
      let entry = merged.get(name);
      if (!entry) {
        if (merged.size >= RUN_RECORD_MAX_GPU_ROWS) {
          lost += 1;
          continue;
        }
        entry = { groups: 0, dispatches: 0, timed: 0, ns: 0 };
        merged.set(name, entry);
      }
      const whole = (v: unknown) => (count(v) ? v : 0);
      entry.groups += whole(row.groups);
      entry.dispatches += whole(row.dispatches);
      const timed = Math.min(whole(row.timedGroups), whole(row.groups));
      entry.timed += timed;
      if (timed > 0 && finite(row.gpuNs) && row.gpuNs >= 0) entry.ns += row.gpuNs;
    }
  const gpu = [...merged].map(([name, e]) => ({
    name,
    groups: e.groups,
    dispatches: e.dispatches,
    timedGroups: Math.min(e.timed, e.groups),
    gpuMs: e.timed > 0 ? Math.round(e.ns / 1e3) / 1e3 : null,
  }));
  return { gpu, gpuTimestamps: gpu.some((row) => row.gpuMs !== null), gpuDropped: lost };
}

/** Collapse the lab's stage list into per-operation totals. */
export function summarizeStages(
  stages: readonly {
    operation: string;
    elapsedMs?: number;
    dispatches?: number;
    submissions?: number;
    gpuMs?: number;
  }[],
): Record<string, RunRecordStage> {
  const out: Record<string, RunRecordStage> = {};
  for (const stage of stages) {
    if (stage.operation === 'runTotal') continue;
    const entry = out[stage.operation] ?? { calls: 0, elapsedMs: 0, dispatches: 0, submissions: 0 };
    out[stage.operation] = {
      calls: entry.calls + 1,
      elapsedMs: entry.elapsedMs + (stage.elapsedMs ?? 0),
      dispatches: entry.dispatches + (stage.dispatches ?? 0),
      submissions: entry.submissions + (stage.submissions ?? 0),
      ...(entry.gpuMs !== undefined || stage.gpuMs !== undefined
        ? { gpuMs: Math.round(((entry.gpuMs ?? 0) + (stage.gpuMs ?? 0)) * 1e3) / 1e3 }
        : {}),
    };
  }
  return out;
}

/**
 * The record's `timing.chunks` from what a worker sends: at most `RUN_RECORD_MAX_CHUNK_KEYS` keys,
 * each with the four numbers of a chunk rate. Anything else is dropped.
 */
export function chunkRecord(rates: unknown): Record<string, RunRecordChunk> {
  const out: Record<string, RunRecordChunk> = {};
  if (!rates || typeof rates !== 'object') return out;
  for (const [key, rate] of Object.entries(rates as Record<string, any>)) {
    if (Object.keys(out).length >= RUN_RECORD_MAX_CHUNK_KEYS) break;
    if (!key || key.length > 64 || !rate || typeof rate !== 'object') continue;
    const entry = {
      msPerUnit: rate.msPerUnit,
      chunks: rate.chunks,
      longestMs: rate.longestMs,
      lastUnits: rate.lastUnits,
    };
    if (chunksValid({ [key]: entry })) out[key] = entry;
  }
  return out;
}

/** One short line per chunk key: `scorePairModels: 37 chunks, longest 212 ms, 0.41 ms/unit`. */
export function chunkLines(chunks: Readonly<Record<string, RunRecordChunk>> | undefined): string[] {
  return Object.entries(chunks ?? {}).map(
    ([key, rate]) =>
      key +
      ': ' +
      rate.chunks +
      (rate.chunks === 1 ? ' chunk' : ' chunks') +
      ', longest ' +
      Math.round(rate.longestMs) +
      ' ms, ' +
      Number(rate.msPerUnit.toPrecision(2)) +
      ' ms/unit',
  );
}

/**
 * The record's `dense` block from the worker's `denseResult` (5.3), or undefined without one.
 * The worker's numbers are bounded and rounded here, so that the validator accepts the block.
 */
export function denseRecord(message: any): RunRecordDense | undefined {
  if (!message || typeof message !== 'object') return undefined;
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const ms = (v: unknown) => (finite(v) && v >= 0 ? Math.round(v * 10) / 10 : 0);
  const bytes = (v: unknown) =>
    Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : null;
  const o = message.options ?? {},
    m = message.memory ?? {},
    views: any[] = Array.isArray(message.views) ? message.views : [];
  const status: RunRecordDense['status'] = message.failure
    ? 'failed'
    : message.cancelled
      ? 'cancelled'
      : message.skipped
        ? 'skipped'
        : 'completed';
  const stageMs = Object.fromEntries(
    RUN_RECORD_DENSE_STAGES.map((stage) => [stage, ms(message.timing?.stageMs?.[stage])]),
  ) as RunRecordDense['stageMs'];
  const failure = message.failure ?? {};
  return {
    status,
    skipped: status === 'skipped' ? String(message.skipped).slice(0, 200) : '',
    options: {
      maxSources: whole(o.maxSources),
      bestSources: whole(o.bestSources),
      windowRadius: whole(o.windowRadius),
      windowStep: whole(o.windowStep),
      photometricIterations: whole(o.photometricIterations),
      geometricIterations: whole(o.geometricIterations),
      minConsistentViews: whole(o.minConsistentViews),
      denseClean: o.denseClean === true,
      ...(o.fusion === 'pixel' || o.fusion === 'tsdf' ? { fusion: o.fusion } : {}),
      ...(typeof o.denseMesh === 'boolean' ? { denseMesh: o.denseMesh } : {}),
      ...(typeof o.planarDepth === 'boolean' ? { planarDepth: o.planarDepth } : {}),
      ...(o.fusion === 'tsdf' &&
      (o.tsdf?.visibility === 'ray-count' || o.tsdf?.visibility === 'view-weighted')
        ? { tsdfVisibility: o.tsdf.visibility }
        : {}),
    },
    ...(Number.isSafeInteger(message.planarFilled) && message.planarFilled >= 0
      ? { planarFilled: whole(message.planarFilled) }
      : {}),
    ...cleanRecord(message),
    ...meshRecord(message),
    views: views.filter((view) => view?.status === 'done').length,
    viewsWithoutDepth: views.filter((view) => view && view.status !== 'done').length,
    points: whole(message.points),
    totalMs: ms(message.timing?.totalMs),
    stageMs,
    memory: {
      liveBytesBefore: whole(m.liveBytesBefore),
      pooledBytesBefore: whole(m.pooledBytesBefore),
      depthStageBytes: bytes(m.depthStageBytes),
      depthStageFormulaBytes: bytes(m.depthStageFormulaBytes),
      fusionStageBytes: bytes(m.fusionStageBytes),
    },
    failure:
      status === 'failed'
        ? {
            message: String(failure.message ?? '').slice(0, 2000),
            stage: String(failure.stage ?? '').slice(0, 64),
            kernel: String(failure.kernel ?? '').slice(0, 64),
            reason: String(failure.reason ?? '').slice(0, 64),
            ...(gpuLossContextValid(failure.context) ? { context: failure.context } : {}),
          }
        : null,
  };
}

/**
 * The record's one `dense` block of a multi-video run: the sum of the blocks of its components
 * (`docs/multi-video.md`, Amendment 9, item 6). Counts and times add up. A byte count takes the
 * largest value, and the "before" counts those of the first component. The status is `failed` if
 * any component failed, else `cancelled` if any was cancelled, else `completed` if any completed,
 * else `skipped`. The thresholds of `clean` and `mesh` come from the component with the most points.
 */
export function sumDenseRecords(blocks: readonly RunRecordDense[]): RunRecordDense | undefined {
  const first = blocks[0];
  if (!first) return undefined;
  const sum = (pick: (d: RunRecordDense) => number) => blocks.reduce((n, d) => n + pick(d), 0);
  const round = (v: number) => Math.round(v * 10) / 10;
  const largest = (pick: (d: RunRecordDense) => number | null) =>
    blocks.reduce<number | null>((m, d) => {
      const v = pick(d);
      return v === null ? m : m === null ? v : Math.max(m, v);
    }, null);
  const failed = blocks.find((d) => d.status === 'failed');
  const status: RunRecordDense['status'] = failed
    ? 'failed'
    : blocks.some((d) => d.status === 'cancelled')
      ? 'cancelled'
      : blocks.some((d) => d.status === 'completed')
        ? 'completed'
        : 'skipped';
  const cleans = blocks.filter((d) => d.clean).map((d) => ({ d, c: d.clean! }));
  const meshes = blocks.filter((d) => d.mesh).map((d) => ({ d, m: d.mesh! }));
  const biggest = <T extends { d: RunRecordDense }>(list: T[]) =>
    list.reduce((a, b) => (b.d.points > a.d.points ? b : a));
  const clean = cleans.length
    ? (() => {
        const ref = biggest(cleans).c;
        const add = (pick: (c: NonNullable<RunRecordDense['clean']>) => number) =>
          cleans.reduce((n, { c }) => n + pick(c), 0);
        return {
          tier: cleans.every(({ c }) => c.tier === 'webgpu')
            ? ('webgpu' as const)
            : ('cpu' as const),
          before: add((c) => c.before),
          afterOutliers: add((c) => c.afterOutliers),
          afterVoxel: add((c) => c.afterVoxel),
          after: add((c) => c.after),
          outlierThreshold: ref.outlierThreshold,
          voxelSize: ref.voxelSize,
          eps: ref.eps,
          clusters: add((c) => c.clusters),
          keptClusters: add((c) => c.keptClusters),
          outliersMs: round(add((c) => c.outliersMs)),
          voxelMs: round(add((c) => c.voxelMs)),
          clusterMs: round(add((c) => c.clusterMs)),
          error: cleans.find(({ c }) => c.error)?.c.error ?? '',
        };
      })()
    : undefined;
  const mesh = meshes.length
    ? (() => {
        const ref = biggest(meshes).m;
        const add = (pick: (m: NonNullable<RunRecordDense['mesh']>) => number) =>
          meshes.reduce((n, { m }) => n + pick(m), 0);
        return {
          tier: meshes.every(({ m }) => m.tier === 'webgpu')
            ? ('webgpu' as const)
            : ('cpu' as const),
          points: add((m) => m.points),
          voxelSize: ref.voxelSize,
          radius: ref.radius,
          nodes: add((m) => m.nodes),
          cubes: add((m) => m.cubes),
          vertices: add((m) => m.vertices),
          faces: add((m) => m.faces),
          latticeMs: round(add((m) => m.latticeMs)),
          fieldMs: round(add((m) => m.fieldMs)),
          cubesMs: round(add((m) => m.cubesMs)),
          verticesMs: round(add((m) => m.verticesMs)),
          error: meshes.find(({ m }) => m.error)?.m.error ?? '',
        };
      })()
    : undefined;
  return {
    status,
    skipped:
      status === 'skipped' ? (blocks.find((d) => d.skipped)?.skipped ?? '').slice(0, 200) : '',
    options: first.options,
    ...(blocks.some((d) => d.planarFilled !== undefined)
      ? { planarFilled: sum((d) => d.planarFilled ?? 0) }
      : {}),
    ...(clean ? { clean } : {}),
    ...(mesh ? { mesh } : {}),
    views: sum((d) => d.views),
    viewsWithoutDepth: sum((d) => d.viewsWithoutDepth),
    points: sum((d) => d.points),
    totalMs: round(sum((d) => d.totalMs)),
    stageMs: Object.fromEntries(
      RUN_RECORD_DENSE_STAGES.map((stage) => [stage, round(sum((d) => d.stageMs[stage]))]),
    ) as RunRecordDense['stageMs'],
    memory: {
      liveBytesBefore: first.memory.liveBytesBefore,
      pooledBytesBefore: first.memory.pooledBytesBefore,
      depthStageBytes: largest((d) => d.memory.depthStageBytes),
      depthStageFormulaBytes: largest((d) => d.memory.depthStageFormulaBytes),
      fusionStageBytes: largest((d) => d.memory.fusionStageBytes),
    },
    failure: failed?.failure ?? null,
  };
}

/** The `clean` field of a dense record, or nothing when the cleanup neither ran nor failed. */
function cleanRecord(message: any): Pick<RunRecordDense, 'clean'> | Record<string, never> {
  const r = message.clean,
    error = typeof message.cleanError === 'string' ? message.cleanError.slice(0, 200) : '';
  if (!(r && typeof r === 'object') && !error) return {};
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const num = (v: unknown) => (finite(v) && v >= 0 ? v : 0);
  const ms = (v: unknown) => Math.round(num(v) * 10) / 10;
  const c = r ?? {};
  return {
    clean: {
      tier: c.tier === 'webgpu' ? 'webgpu' : 'cpu',
      before: whole(c.before),
      afterOutliers: whole(c.afterOutliers),
      afterVoxel: whole(c.afterVoxel),
      after: whole(c.after),
      outlierThreshold: num(c.outlierThreshold),
      voxelSize: num(c.voxelSize),
      eps: num(c.eps),
      clusters: whole(c.clusters),
      keptClusters: whole(c.keptClusters),
      outliersMs: ms(c.ms?.outliers),
      voxelMs: ms(c.ms?.voxel),
      clusterMs: ms(c.ms?.cluster),
      error,
    },
  };
}

/** The `mesh` field of a dense record, or nothing when the mesh neither ran nor failed. */
function meshRecord(message: any): Pick<RunRecordDense, 'mesh'> | Record<string, never> {
  const r = message.mesh,
    error = typeof message.meshError === 'string' ? message.meshError.slice(0, 200) : '';
  if (!(r && typeof r === 'object') && !error) return {};
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const num = (v: unknown) => (finite(v) && v >= 0 ? v : 0);
  const ms = (v: unknown) => Math.round(num(v) * 10) / 10;
  const m = r ?? {};
  return {
    mesh: {
      tier: m.tier === 'webgpu' ? 'webgpu' : 'cpu',
      points: whole(m.points),
      voxelSize: num(m.voxelSize),
      radius: num(m.radius),
      nodes: whole(m.nodes),
      cubes: whole(m.cubes),
      vertices: whole(m.vertices),
      faces: whole(m.faces),
      latticeMs: ms(m.ms?.lattice),
      fieldMs: ms(m.ms?.field),
      cubesMs: ms(m.ms?.cubes),
      verticesMs: ms(m.ms?.vertices),
      error,
    },
  };
}

/** One line for the page: `Dense MVS completed: 123456 points from 12 depth maps in 812 s …`. */
/** One line for a record's KLT block: the matches it added and the mean length of a track. */
export function kltLine(klt: RunRecordKlt | undefined): string | undefined {
  if (!klt) return undefined;
  return (
    'KLT ' +
    klt.matches +
    ' matches (' +
    klt.pairMatches +
    ' new), mean track ' +
    klt.meanLived.toFixed(1) +
    ' frames, ' +
    klt.rejected +
    ' rejected'
  );
}

export function denseLine(dense: RunRecordDense | undefined): string | undefined {
  if (!dense) return undefined;
  const stages = RUN_RECORD_DENSE_STAGES.map(
    (stage) => stage + ' ' + (dense.stageMs[stage] / 1000).toFixed(1) + ' s',
  ).join(', ');
  const before = (dense.memory.liveBytesBefore / 1024 ** 2).toFixed(1);
  return (
    'Dense MVS ' +
    dense.status +
    (dense.status === 'skipped' ? ' (' + dense.skipped + ')' : '') +
    ': ' +
    dense.points +
    ' points from ' +
    dense.views +
    ' depth maps in ' +
    (dense.totalMs / 1000).toFixed(1) +
    ' s (' +
    stages +
    '), ' +
    before +
    ' MiB live before' +
    (dense.clean
      ? dense.clean.error
        ? ', cleanup failed: ' + dense.clean.error
        : ', cleaned to ' + dense.clean.after + ' points'
      : '') +
    (dense.mesh
      ? dense.mesh.error
        ? ', mesh failed: ' + dense.mesh.error
        : ', mesh of ' + dense.mesh.faces + ' faces'
      : '')
  );
}
