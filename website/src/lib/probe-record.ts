/**
 * A probe record is the result of the triangulation probe of `/lab/probe/`
 * (`website/src/workers/probe.worker.ts`, `docs/lab-run-records.md`, "Probe"): the steps it ran
 * on the job of `tests/fixtures/south-building-pair-triangulation.json`, and the tracks its
 * bisection found. The probe builds it; `/api/records` validates and stores it next to the run
 * records. The same validator runs in both places.
 */
export const PROBE_RECORD_FORMAT = 'typecv-triangulation-probe-v1' as const;
/** The most steps a record holds; the probe runs at most 40. */
export const PROBE_RECORD_MAX_STEPS = 64;
/** The most tracks or sets a list of the bisection holds. */
export const PROBE_RECORD_MAX_SETS = 64;
/** The most tracks whose data a set carries: one workgroup of `triangulateTracks`. */
export const PROBE_RECORD_MAX_SET_DATA = 16;
export const PROBE_RECORD_MAX_MESSAGE = 2000;

export interface ProbeRange {
  readonly first: number;
  readonly end: number;
}
export interface ProbeOverride {
  readonly refinementIterations?: number;
  readonly maxPairHypotheses?: number;
}
/** What the device measured of key `triangulateTracks` (`ShadeDevice.chunkRates`). */
export interface ProbeChunk {
  readonly msPerUnit: number;
  readonly chunks: number;
  readonly longestMs: number;
  readonly lastUnits: number;
}
export type ProbeOutcome = 'completed' | 'lost' | 'timeout' | 'error' | 'open-failed';
/** One track of the fixture: its index, its two pixels (x1, y1, x2, y2) and the CPU status. */
export interface ProbeTrack {
  readonly index: number;
  readonly pixels: readonly number[];
  readonly expectedStatus: number;
}
export interface ProbeSet extends ProbeRange {
  readonly tracks: number;
  /** The data of each track, for a set of at most `PROBE_RECORD_MAX_SET_DATA` tracks. */
  readonly data: readonly ProbeTrack[] | null;
}
export interface ProbeLoss {
  readonly message: string;
  readonly reason: string;
  readonly lastKernel: string;
  readonly simulated: boolean;
  readonly chunksStarted: number;
  readonly chunksCompleted: number;
  readonly completedTracks: number;
  /** The chunk in flight at the loss; for a simulated loss, the chunk that triggered it. */
  readonly inFlight: ProbeRange | null;
}
export interface ProbeStep extends ProbeRange {
  readonly step: number;
  readonly kind: 'variant' | 'chunk' | 'half';
  readonly name: string;
  readonly override: ProbeOverride;
  readonly tracks: number;
  readonly outcome: ProbeOutcome;
  readonly elapsedMs: number | null;
  readonly fp64Flavor: 'float' | 'integer' | null;
  readonly chunks: ProbeChunk | null;
  readonly dispatches: number | null;
  readonly submissions: number | null;
  readonly accepted: number | null;
  readonly expectedAccepted: number;
  /** Whether the accepted count is within 8 of the fixture's `expectedStatus`, as the test asks. */
  readonly withinEight: boolean | null;
  readonly statusCounts: Readonly<Record<string, number>> | null;
  readonly statusMismatches: { readonly count: number; readonly first: readonly number[] } | null;
  readonly loss: ProbeLoss | null;
  readonly error: string | null;
}
export interface ProbeRecord {
  readonly format: typeof PROBE_RECORD_FORMAT;
  readonly startedAt: string;
  readonly finishedAt: string;
  readonly site: { readonly commit: string | null; readonly href: string };
  readonly client: { readonly userAgent: string; readonly hardwareConcurrency: number };
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly fallbackAdapter: boolean;
  } | null;
  readonly device: {
    readonly boundedSubmissions: boolean;
    readonly fp64Requested: 'auto' | 'float' | 'integer';
    readonly stepTimeoutMs: number;
    readonly maxSteps: number;
  };
  readonly simulatedLossTrack: number | null;
  readonly range: ProbeRange;
  readonly fixture: {
    readonly source: string;
    readonly tracks: number;
    readonly options: { readonly maxErrorPx: number; readonly minAngleDeg: number };
    readonly views: readonly {
      readonly camera: Readonly<Record<string, number>>;
      readonly pose: { readonly R: readonly number[]; readonly t: readonly number[] };
    }[];
    readonly expectedAccepted: number;
  };
  readonly variants: readonly {
    readonly name: string;
    readonly override: ProbeOverride;
    readonly outcome: ProbeOutcome;
    readonly elapsedMs: number | null;
    readonly chunks: ProbeChunk | null;
    readonly accepted: number | null;
    readonly expectedAccepted: number;
    readonly withinEight: boolean | null;
    readonly dispatches: number | null;
  }[];
  readonly bisection: {
    readonly override: ProbeOverride;
    readonly hangingTracks: readonly ProbeTrack[];
    readonly minimalSets: readonly ProbeSet[];
    readonly unresolved: readonly ProbeSet[];
    readonly steps: number;
  } | null;
  readonly aborted: string | null;
  /** After an abort: the query `?tracks=<first>-<last>` that goes on from the smallest lost set. */
  readonly resume: string | null;
  readonly steps: readonly ProbeStep[];
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
const text = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.length <= max;
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const orNull =
  <T>(check: (value: unknown) => boolean) =>
  (value: unknown): value is T | null =>
    value === null || check(value);
const list = (value: unknown, max: number, check: (item: any) => boolean) =>
  Array.isArray(value) && value.length <= max && value.every(check);
const time = (value: unknown) => text(value, 40) && !Number.isNaN(Date.parse(value));
const OUTCOMES = ['completed', 'lost', 'timeout', 'error', 'open-failed'];

const range = (value: any) => object(value) && count(value.first) && count(value.end);
const override = (value: any) =>
  object(value) &&
  Object.keys(value).every((key) => ['refinementIterations', 'maxPairHypotheses'].includes(key)) &&
  (value.refinementIterations === undefined || count(value.refinementIterations)) &&
  (value.maxPairHypotheses === undefined || count(value.maxPairHypotheses));
const chunk = (value: any) =>
  object(value) &&
  finite(value.msPerUnit) &&
  (value.msPerUnit as number) >= 0 &&
  count(value.chunks) &&
  finite(value.longestMs) &&
  (value.longestMs as number) >= 0 &&
  count(value.lastUnits);
const track = (value: any) =>
  object(value) &&
  count(value.index) &&
  list(value.pixels, 128, finite) &&
  count(value.expectedStatus);
const set = (value: any) =>
  range(value) &&
  count(value.tracks) &&
  (value.data === null || list(value.data, PROBE_RECORD_MAX_SET_DATA, track));
const loss = (value: any) =>
  object(value) &&
  text(value.message, PROBE_RECORD_MAX_MESSAGE) &&
  text(value.reason, 64) &&
  text(value.lastKernel, 64) &&
  typeof value.simulated === 'boolean' &&
  count(value.chunksStarted) &&
  count(value.chunksCompleted) &&
  count(value.completedTracks) &&
  (value.inFlight === null || range(value.inFlight));
const result = (value: any) =>
  OUTCOMES.includes(value.outcome) &&
  orNull(finite)(value.elapsedMs) &&
  orNull(chunk)(value.chunks) &&
  orNull(count)(value.accepted) &&
  count(value.expectedAccepted) &&
  orNull((v) => typeof v === 'boolean')(value.withinEight) &&
  orNull(count)(value.dispatches);
const variant = (value: any) =>
  object(value) && text(value.name, 128) && override(value.override) && result(value);
const step = (value: any) =>
  object(value) &&
  count(value.step) &&
  ['variant', 'chunk', 'half'].includes(value.kind as string) &&
  text(value.name, 128) &&
  override(value.override) &&
  range(value) &&
  count(value.tracks) &&
  result(value) &&
  (value.fp64Flavor === null || ['float', 'integer'].includes(value.fp64Flavor as string)) &&
  orNull(count)(value.submissions) &&
  (value.statusCounts === null ||
    (object(value.statusCounts) &&
      Object.keys(value.statusCounts).length <= 16 &&
      Object.entries(value.statusCounts).every(([key, n]) => text(key, 16) && count(n)))) &&
  (value.statusMismatches === null ||
    (object(value.statusMismatches) &&
      count(value.statusMismatches.count) &&
      list(value.statusMismatches.first, 32, count))) &&
  (value.loss === null || loss(value.loss)) &&
  orNull((v) => text(v, PROBE_RECORD_MAX_MESSAGE))(value.error);
const view = (value: any) =>
  object(value) &&
  object(value.camera) &&
  Object.keys(value.camera).length <= 16 &&
  Object.entries(value.camera).every(([key, n]) => text(key, 16) && finite(n)) &&
  object(value.pose) &&
  list(value.pose.R, 9, finite) &&
  list(value.pose.t, 3, finite);

/** Return the reason a value is not a probe record, or undefined when it is one. */
export function probeRecordProblem(value: unknown): string | undefined {
  const r = value as ProbeRecord;
  if (!object(r)) return 'not an object';
  if (r.format !== PROBE_RECORD_FORMAT) return 'format';
  if (!time(r.startedAt)) return 'startedAt';
  if (!time(r.finishedAt)) return 'finishedAt';
  if (!object(r.site) || !orNull((v) => text(v, 64))(r.site.commit) || !text(r.site.href, 512))
    return 'site';
  if (!object(r.client) || !text(r.client.userAgent, 512) || !count(r.client.hardwareConcurrency))
    return 'client';
  const a = r.adapter;
  if (!(
    a === null ||
    (object(a) &&
      text(a.vendor, 128) &&
      text(a.architecture, 128) &&
      text(a.device, 128) &&
      text(a.description, 512) &&
      typeof a.fallbackAdapter === 'boolean')
  ))
    return 'adapter';
  const d = r.device;
  if (
    !object(d) ||
    typeof d.boundedSubmissions !== 'boolean' ||
    !['auto', 'float', 'integer'].includes(d.fp64Requested) ||
    !count(d.stepTimeoutMs) ||
    !count(d.maxSteps) ||
    d.maxSteps > PROBE_RECORD_MAX_STEPS
  )
    return 'device';
  if (!orNull(count)(r.simulatedLossTrack)) return 'simulatedLossTrack';
  if (!range(r.range)) return 'range';
  const f = r.fixture;
  if (
    !object(f) ||
    !text(f.source, 200) ||
    !count(f.tracks) ||
    !object(f.options) ||
    !finite(f.options.maxErrorPx) ||
    !finite(f.options.minAngleDeg) ||
    !list(f.views, 8, view) ||
    !count(f.expectedAccepted)
  )
    return 'fixture';
  if (!list(r.variants, 8, variant)) return 'variants';
  const b = r.bisection;
  if (!(
    b === null ||
    (object(b) &&
      override(b.override) &&
      list(b.hangingTracks, PROBE_RECORD_MAX_SETS, track) &&
      list(b.minimalSets, PROBE_RECORD_MAX_SETS, set) &&
      list(b.unresolved, PROBE_RECORD_MAX_SETS, set) &&
      count(b.steps))
  ))
    return 'bisection';
  if (!orNull((v) => text(v, PROBE_RECORD_MAX_MESSAGE))(r.aborted)) return 'aborted';
  if (!(r.resume === null || (text(r.resume, 64) && /^\?tracks=\d+-\d+$/.test(r.resume))))
    return 'resume';
  if (!list(r.steps, PROBE_RECORD_MAX_STEPS, step)) return 'steps';
  return undefined;
}
