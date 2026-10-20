/**
 * A run record is the measurement a visitor can send from the SfM lab after a run:
 * environment, input shape, options, result counts and timings. It holds no video,
 * image, file name or network address. The lab builds it; `/api/records` validates and
 * stores it. The same validator runs in both places.
 */
export const RUN_RECORD_FORMAT = 'typecv-run-record-v4' as const;
/** The formats the validator accepts: the current one, and v3 (no `timing.chunks` and no
 *  `timing.budgetMs`) from a page loaded before v4 was deployed. */
export const RUN_RECORD_FORMATS = ['typecv-run-record-v3', RUN_RECORD_FORMAT] as const;
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
    /** Analysis frames per second taken from an uploaded video before SIFT (the lab's `?fps=`).
     *  Absent for photo and sample inputs and in records before 2026-10-20 (every frame). */
    readonly videoSampleFps?: number;
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
    (o.videoSampleFps !== undefined && (!finite(o.videoSampleFps) || o.videoSampleFps <= 0))
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
    (r.format === RUN_RECORD_FORMAT
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
