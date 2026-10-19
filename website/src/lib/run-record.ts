/**
 * A run record is the measurement a visitor can send from the SfM lab after a run:
 * environment, input shape, options, result counts and timings. It holds no video,
 * image, file name or network address. The lab builds it; `/api/records` validates and
 * stores it. The same validator runs in both places.
 */
export const RUN_RECORD_FORMAT = 'typecv-run-record-v1' as const;
export const RUN_RECORD_MAX_BYTES = 256 * 1024;
export const RUN_RECORD_MAX_NOTE = 500;
export const RUN_RECORD_MAX_FRAMES = 2048;

export interface RunRecordStage {
  readonly calls: number;
  readonly elapsedMs: number;
  readonly dispatches: number;
  readonly submissions: number;
}
export interface RunRecord {
  readonly format: typeof RUN_RECORD_FORMAT;
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
  };
  readonly result: {
    readonly registeredViews: number;
    readonly landmarks: number;
    readonly observations: number;
    readonly meanReprojectionErrorPx: number | null;
    readonly unregistered: number;
    readonly matchedPairs: number;
    readonly featureCounts: readonly number[];
  };
  readonly timing: {
    readonly totalMs: number;
    readonly reusedDevice: boolean;
    readonly stages: Readonly<Record<string, RunRecordStage>>;
    /** Per analysis frame, in processing order. */
    readonly frameMs: readonly number[];
  };
  readonly transfers: {
    readonly applicationUploadBytes: number;
    readonly applicationReadbackBytes: number;
  };
  readonly benchmark: { readonly name: string; readonly pass: 'cold' | 'warm' } | null;
  readonly note: string;
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
const text = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.length <= max;

/** Return the reason a value is not a run record, or undefined when it is one. */
export function runRecordProblem(value: unknown): string | undefined {
  const r = value as RunRecord;
  if (!r || typeof r !== 'object') return 'not an object';
  if (r.format !== RUN_RECORD_FORMAT) return 'format';
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
    !count(o.maxDistance)
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
    !t.frameMs.every(finite)
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
