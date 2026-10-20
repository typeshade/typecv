/**
 * A run record is the measurement a visitor can send from the SfM lab after a run:
 * environment, input shape, options, result counts and timings. It holds no video,
 * image, file name or network address. The lab builds it; `/api/records` validates and
 * stores it. The same validator runs in both places.
 */
export const RUN_RECORD_FORMAT = 'typecv-run-record-v5' as const;
/** The formats the validator accepts: the current one, v4 (no `dense`) and v3 (no `dense`, no
 *  `timing.chunks` and no `timing.budgetMs`) from a page loaded before the next one was deployed. */
export const RUN_RECORD_FORMATS = [
  'typecv-run-record-v3',
  'typecv-run-record-v4',
  RUN_RECORD_FORMAT,
] as const;
export const RUN_RECORD_MAX_CHUNK_KEYS = 32;
export const RUN_RECORD_MAX_KERNELS = 256;
export const RUN_RECORD_MAX_BYTES = 256 * 1024;
export const RUN_RECORD_MAX_NOTE = 500;
export const RUN_RECORD_MAX_FRAMES = 2048;

export interface RunRecordStage {
  readonly calls: number;
  readonly elapsedMs: number;
  readonly dispatches: number;
  readonly submissions: number;
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
}

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
    /** The mesh of the final cloud was on (section 3.13). Absent in a record before the mesh. */
    readonly denseMesh?: boolean;
  };
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
   * at the fusion stage, and the depth stage's formula, (16 N + 4) P. Null when the stage was not
   * reached.
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
  } | null;
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
    };
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
    /** The keyframe gate of the run: `phase` (phase correlation before SIFT, `?gate=phase`) or
     *  `legacy` (the median match displacement after SIFT, `?gate=legacy`). Absent in records
     *  before 2026-10-21, which ran the legacy gate. */
    readonly keyframeGate?: 'phase' | 'legacy';
    /** Frames the gate kept as keyframes, of `input.frames`. Absent in records before 2026-10-21. */
    readonly keyframes?: number;
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
   * mapper operation and the kernel that started last (empty when the failure was not in a
   * kernel), the device-loss reason the browser gave (`destroyed`, `unknown`, or empty) and the
   * number of frames whose features were extracted before the failure.
   */
  readonly failure: {
    readonly message: string;
    readonly operation: string;
    readonly kernel: string;
    readonly reason: string;
    readonly framesProcessed: number;
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
    (o.denseMesh === undefined || typeof o.denseMesh === 'boolean') &&
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
        text(f.reason, 64)
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
    stage(m.longestGapStage)
  );
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
    i.durationSeconds < 0
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
    !count(o.maxFeatures) ||
    !finite(o.matchRatio) ||
    !finite(o.maxDistance) ||
    o.maxDistance < 0 ||
    !['corner', 'sift'].includes(o.frontEnd) ||
    typeof o.boundedSubmissions !== 'boolean' ||
    !['float', 'integer'].includes(o.fp64Flavor) ||
    (o.guidedMatching !== undefined && typeof o.guidedMatching !== 'boolean') ||
    (o.siftFirstOctave !== undefined && o.siftFirstOctave !== -1 && o.siftFirstOctave !== 0) ||
    (o.clahe !== undefined && typeof o.clahe !== 'boolean') ||
    (o.videoSampleFps !== undefined && (!finite(o.videoSampleFps) || o.videoSampleFps <= 0)) ||
    (o.videoDecode !== undefined &&
      o.videoDecode !== 'worker' &&
      o.videoDecode !== 'main-thread') ||
    (o.keyframeGate !== undefined && o.keyframeGate !== 'phase' && o.keyframeGate !== 'legacy') ||
    (o.keyframes !== undefined && !count(o.keyframes))
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
    !s.featureCounts.every(count)
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
        count(stage.submissions),
    ) ||
    !Array.isArray(t.frameMs) ||
    t.frameMs.length > RUN_RECORD_MAX_FRAMES ||
    !t.frameMs.every(finite) ||
    !Array.isArray(t.kernels) ||
    t.kernels.length > RUN_RECORD_MAX_KERNELS ||
    !t.kernels.every((name) => text(name, 64)) ||
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
      count(e.framesProcessed))
  ))
    return 'failure';
  if (!text(r.note, RUN_RECORD_MAX_NOTE)) return 'note';
  if (
    r.format === RUN_RECORD_FORMAT
      ? r.dense !== undefined && !denseValid(r.dense)
      : r.dense !== undefined
  )
    return 'dense';
  return undefined;
}

/** Collapse the lab's stage list into per-operation totals. */
export function summarizeStages(
  stages: readonly {
    operation: string;
    elapsedMs?: number;
    dispatches?: number;
    submissions?: number;
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
      ...(typeof o.denseMesh === 'boolean' ? { denseMesh: o.denseMesh } : {}),
    },
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
          }
        : null,
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
