import type { RunRecordChunk, RunRecordKernelTime } from '../run-record/schema.ts';
import { chunksValid, count, finite, text } from '../run-record/validate.ts';

/**
 * A hardware-check record is the result of the check page `/lab/check/`
 * (`website/src/workers/hardware-check.worker.ts`, `docs/lab-run-records.md`, "Hardware check").
 * It holds the facts of the device, the flavor of the emulated double that the TypeShade device
 * selects beside the flavor that TypeCV's rule before change 0085 gives on the same signals, and
 * each case that the check ran on the device and on the CPU tier of the same browser: the
 * outcome, the digests, the agreement with the family's bound, the GPU time and the loss. The
 * page builds it; `/api/records` validates and stores it next to the run records. The same
 * validator runs in both places, and `scripts/compare-hardware-check.mjs` reads the record back.
 * It holds no image, file name or IP address.
 */
export const HARDWARE_CHECK_FORMAT = 'typecv-hardware-check-v1' as const;
/** The most cases a record holds. */
export const HARDWARE_CHECK_MAX_CASES = 32;
/** The most rows of GPU time a case holds (the call lists of `ShadeDevice.kernelStatistics`). */
export const HARDWARE_CHECK_MAX_GPU_ROWS = 64;
/** The most digest parts (the top-level fields of an answer) a case holds. */
export const HARDWARE_CHECK_MAX_PARTS = 16;
export const HARDWARE_CHECK_MAX_MESSAGE = 2000;
/** The visitor's statement of the device: the GPU model and the computer. */
export const HARDWARE_CHECK_MAX_NOTE = 500;
export const HARDWARE_CHECK_OUTCOMES = [
  'completed',
  'lost',
  'timeout',
  'error',
  'open-failed',
  'skipped',
] as const;
export type HardwareCheckOutcome = (typeof HARDWARE_CHECK_OUTCOMES)[number];
export type HardwareCheckFlavorName = 'float' | 'integer';

/** The identity strings of a WebGPU adapter (`GPUAdapterInfo`). */
export interface HardwareCheckIdentity {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}
/** What the page reads of the adapter WebGPU offers (`website/src/lib/adapter-info.ts`). */
export interface HardwareCheckAdapter extends HardwareCheckIdentity {
  readonly fallbackAdapter: boolean;
  readonly limits: Readonly<Record<string, number>>;
  readonly features: readonly string[];
}
/** The flavor of the emulated double (TypeShade changes 0073 and 0085). */
export interface HardwareCheckFlavor {
  /** The page's `?fp64=`, else `'auto'`. */
  readonly requested: 'auto' | HardwareCheckFlavorName;
  /** `ShadeDevice.fp64Flavor` of a device opened as the lab opens its device. */
  readonly selected: HardwareCheckFlavorName | null;
  /** The same of a device opened with the default options. */
  readonly selectedDefault: HardwareCheckFlavorName | null;
  /** The rule before change 0085 (`flavor.ts`) on the lab's path. */
  readonly previous: HardwareCheckFlavorName;
  /** The rule before change 0085 on the default path. */
  readonly previousRequested: HardwareCheckFlavorName;
  /** Whether both selections equal the rule: null when the flavor was forced or no device opened. */
  readonly same: boolean | null;
  /** The signals the rule reads. */
  readonly signals: {
    readonly deviceAdapterInfo: HardwareCheckIdentity | null;
    readonly requestedAdapterInfo: HardwareCheckIdentity | null;
    readonly userAgent: string;
  };
  /** `ShadeDevice.info.webgpu` of the lab's device: where TypeShade read the identity. */
  readonly info: {
    readonly provenance: 'device' | 'adapter' | 'unavailable';
    readonly vendor: string | null;
    readonly architecture: string | null;
    readonly device: string | null;
    readonly description: string | null;
  } | null;
  readonly error: string | null;
}
/** The bound of a case, from the WebGPU test of its family (`source`). */
export interface HardwareCheckBound {
  readonly kind: 'exact' | 'abs' | 'rel';
  readonly value: number;
  readonly source: string;
}
/** SHA-256 digests in the canonical form of `tests/typeshade/digest.mjs`. */
export interface HardwareCheckDigests {
  /** The case's input. */
  readonly input: string | null;
  /** The device's answer, first run. */
  readonly device: string | null;
  /** The device's answer, second run on the same device. */
  readonly deviceWarm: string | null;
  /** The CPU tier's answer in the same browser. */
  readonly cpu: string | null;
  readonly deviceParts: Readonly<Record<string, string>> | null;
  readonly cpuParts: Readonly<Record<string, string>> | null;
}
/** How far the device's answer is from the reference of the family's test. */
export interface HardwareCheckAgreement {
  readonly maxAbs: number | null;
  readonly maxRel: number | null;
  readonly mismatches: number;
  readonly compared: number;
  readonly problems: readonly string[];
}
export interface HardwareCheckLoss {
  readonly message: string;
  readonly reason: string;
  readonly lastKernel: string;
  /** Up to four uncaptured GPU errors of the device before the loss. */
  readonly uncaptured: readonly { readonly name: string; readonly message: string }[];
}
export interface HardwareCheckCase {
  readonly name: string;
  /** The kernel entries the case runs. */
  readonly family: readonly string[];
  readonly bound: HardwareCheckBound;
  readonly outcome: HardwareCheckOutcome;
  /** Why the case did not run; null when it ran. */
  readonly skipped: string | null;
  /** The wall time of the first run on the device (cold), of the second (warm) and of the CPU tier. */
  readonly coldMs: number | null;
  readonly warmMs: number | null;
  readonly cpuMs: number | null;
  /** Whether the two runs on the device gave the same digest. */
  readonly repeatable: boolean | null;
  readonly digests: HardwareCheckDigests;
  readonly agreement: HardwareCheckAgreement | null;
  /** Whether the device's answer is within the bound. */
  readonly pass: boolean | null;
  /** Counts of the case's work: pixels, points, partial workgroups. */
  readonly summary: Readonly<Record<string, number>>;
  /** The GPU time of each fused call list of the first run, as `timing.gpu` of a run record. */
  readonly gpu: readonly RunRecordKernelTime[];
  /** `ShadeDevice.chunkRates` after the first run. */
  readonly chunks: Readonly<Record<string, RunRecordChunk>> | null;
  readonly loss: HardwareCheckLoss | null;
  readonly error: string | null;
}
export interface HardwareCheckRecord {
  readonly format: typeof HARDWARE_CHECK_FORMAT;
  readonly startedAt: string;
  readonly finishedAt: string;
  /** The site commit (`/version.json`), the page address and the TypeShade pin of the runtime. */
  readonly site: {
    readonly commit: string | null;
    readonly href: string;
    readonly typeshade: string | null;
  };
  readonly client: {
    readonly userAgent: string;
    readonly platform: string;
    readonly language: string;
    readonly hardwareConcurrency: number;
    readonly deviceMemory: number | null;
  };
  readonly adapter: HardwareCheckAdapter | null;
  readonly flavor: HardwareCheckFlavor;
  readonly device: {
    readonly timeBudgetMs: number | null;
    readonly timestampQuery: boolean;
    /** `ShadeDevice.info.build.version`. */
    readonly typeshadeVersion: string | null;
  };
  readonly cases: readonly HardwareCheckCase[];
  /** Why the check stopped before its last case; null when it ran them all. */
  readonly aborted: string | null;
  /** After a device loss: the query `?cases=<name>,…` that runs the cases that did not run. */
  readonly resume: string | null;
  /** What the visitor says of the device: the GPU model and the computer. */
  readonly note: string;
}

/** The names of the cases that did not complete within their bound, or that differed between runs. */
export function failedCases(record: Pick<HardwareCheckRecord, 'cases'>): string[] {
  return record.cases
    .filter(
      (c) =>
        (c.outcome !== 'completed' && c.outcome !== 'skipped') ||
        c.pass === false ||
        c.repeatable === false,
    )
    .map((c) => c.name);
}

const object = (value: unknown): value is Record<string, any> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const orNull =
  (check: (value: unknown) => boolean) =>
  (value: unknown): boolean =>
    value === null || check(value);
const list = (value: unknown, max: number, check: (item: any) => boolean) =>
  Array.isArray(value) && value.length <= max && value.every(check);
const time = (value: unknown) => text(value, 40) && !Number.isNaN(Date.parse(value));
const hex = (value: unknown) => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const flavor = (value: unknown) => value === 'float' || value === 'integer';
const ms = (value: unknown) => finite(value) && value >= 0;
const name = (value: unknown) => text(value, 64) && value.length > 0;
const identity = (value: any) =>
  object(value) &&
  text(value.vendor, 512) &&
  text(value.architecture, 512) &&
  text(value.device, 512) &&
  text(value.description, 512);
const adapter = (value: any) =>
  identity(value) &&
  typeof value.fallbackAdapter === 'boolean' &&
  object(value.limits) &&
  Object.keys(value.limits).length <= 16 &&
  Object.entries(value.limits).every(([key, n]) => name(key) && count(n)) &&
  list(value.features, 64, (feature) => name(feature));
const deviceInfo = (value: any) =>
  object(value) &&
  ['device', 'adapter', 'unavailable'].includes(value.provenance) &&
  ['vendor', 'architecture', 'device', 'description'].every((key) =>
    orNull((v) => text(v, 512))(value[key]),
  );
const flavorValid = (value: any) =>
  object(value) &&
  ['auto', 'float', 'integer'].includes(value.requested) &&
  orNull(flavor)(value.selected) &&
  orNull(flavor)(value.selectedDefault) &&
  flavor(value.previous) &&
  flavor(value.previousRequested) &&
  (value.same === null || typeof value.same === 'boolean') &&
  object(value.signals) &&
  orNull(identity)(value.signals.deviceAdapterInfo) &&
  orNull(identity)(value.signals.requestedAdapterInfo) &&
  text(value.signals.userAgent, 512) &&
  orNull(deviceInfo)(value.info) &&
  orNull((v) => text(v, HARDWARE_CHECK_MAX_MESSAGE))(value.error);
const parts = (value: unknown) =>
  object(value) &&
  Object.keys(value).length <= HARDWARE_CHECK_MAX_PARTS &&
  Object.entries(value).every(([key, digest]) => name(key) && hex(digest));
const digests = (value: any) =>
  object(value) &&
  ['input', 'device', 'deviceWarm', 'cpu'].every((key) => orNull(hex)(value[key])) &&
  orNull(parts)(value.deviceParts) &&
  orNull(parts)(value.cpuParts);
const agreement = (value: any) =>
  object(value) &&
  orNull(ms)(value.maxAbs) &&
  orNull(ms)(value.maxRel) &&
  count(value.mismatches) &&
  count(value.compared) &&
  list(value.problems, 8, (problem) => text(problem, 200));
const gpuRow = (value: any) =>
  object(value) &&
  text(value.name, 256) &&
  value.name.length > 0 &&
  count(value.groups) &&
  count(value.dispatches) &&
  count(value.timedGroups) &&
  value.timedGroups <= value.groups &&
  orNull(ms)(value.gpuMs);
const loss = (value: any) =>
  object(value) &&
  text(value.message, HARDWARE_CHECK_MAX_MESSAGE) &&
  text(value.reason, 64) &&
  text(value.lastKernel, 64) &&
  list(
    value.uncaptured,
    4,
    (error: any) => object(error) && text(error.name, 64) && text(error.message, 400),
  );
const caseValid = (value: any) =>
  object(value) &&
  name(value.name) &&
  list(value.family, 16, name) &&
  object(value.bound) &&
  ['exact', 'abs', 'rel'].includes(value.bound.kind) &&
  ms(value.bound.value) &&
  text(value.bound.source, 200) &&
  HARDWARE_CHECK_OUTCOMES.includes(value.outcome) &&
  // A case that did not run says why, and a case that ran gives no such reason.
  (value.outcome === 'skipped'
    ? text(value.skipped, 500) && value.skipped.length > 0
    : value.skipped === null) &&
  orNull(ms)(value.coldMs) &&
  orNull(ms)(value.warmMs) &&
  orNull(ms)(value.cpuMs) &&
  (value.repeatable === null || typeof value.repeatable === 'boolean') &&
  digests(value.digests) &&
  orNull(agreement)(value.agreement) &&
  (value.pass === null || typeof value.pass === 'boolean') &&
  object(value.summary) &&
  Object.keys(value.summary).length <= 16 &&
  Object.entries(value.summary).every(([key, n]) => name(key) && finite(n)) &&
  list(value.gpu, HARDWARE_CHECK_MAX_GPU_ROWS, gpuRow) &&
  orNull(chunksValid)(value.chunks) &&
  orNull(loss)(value.loss) &&
  orNull((v) => text(v, HARDWARE_CHECK_MAX_MESSAGE))(value.error);

/** Return the reason a value is not a hardware-check record, or undefined when it is one. */
export function hardwareCheckProblem(value: unknown): string | undefined {
  const r = value as HardwareCheckRecord;
  if (!object(r)) return 'not an object';
  if (r.format !== HARDWARE_CHECK_FORMAT) return 'format';
  if (!time(r.startedAt)) return 'startedAt';
  if (!time(r.finishedAt)) return 'finishedAt';
  const s = r.site;
  if (
    !object(s) ||
    !orNull((v) => text(v, 64))(s.commit) ||
    !text(s.href, 512) ||
    !orNull((v) => text(v, 64))(s.typeshade)
  )
    return 'site';
  const c = r.client;
  if (
    !object(c) ||
    !text(c.userAgent, 512) ||
    !text(c.platform, 64) ||
    !text(c.language, 35) ||
    !count(c.hardwareConcurrency) ||
    !orNull(ms)(c.deviceMemory)
  )
    return 'client';
  if (!orNull(adapter)(r.adapter)) return 'adapter';
  if (!flavorValid(r.flavor)) return 'flavor';
  const d = r.device;
  if (
    !object(d) ||
    !orNull(ms)(d.timeBudgetMs) ||
    typeof d.timestampQuery !== 'boolean' ||
    !orNull((v) => text(v, 32))(d.typeshadeVersion)
  )
    return 'device';
  if (!list(r.cases, HARDWARE_CHECK_MAX_CASES, caseValid)) return 'cases';
  if (new Set(r.cases.map((x) => x.name)).size !== r.cases.length) return 'cases';
  if (!orNull((v) => text(v, HARDWARE_CHECK_MAX_MESSAGE))(r.aborted)) return 'aborted';
  if (!(
    r.resume === null ||
    (text(r.resume, 512) && /^\?cases=[a-z0-9-]+(,[a-z0-9-]+)*$/.test(r.resume))
  ))
    return 'resume';
  if (!text(r.note, HARDWARE_CHECK_MAX_NOTE)) return 'note';
  return undefined;
}
