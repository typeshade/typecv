/// <reference lib="webworker" />
import { adapterFacts } from '../lib/adapter-info';
import type { CheckApis, TriangulationFixture } from '../lib/hardware-check/cases';
import { previousFp64Flavor } from '../lib/hardware-check/flavor';
import {
  HARDWARE_CHECK_FORMAT,
  HARDWARE_CHECK_MAX_MESSAGE,
  HARDWARE_CHECK_MAX_NOTE,
  type HardwareCheckAdapter,
  type HardwareCheckFlavor,
  type HardwareCheckIdentity,
  type HardwareCheckRecord,
} from '../lib/hardware-check/record';
import { runCases } from '../lib/hardware-check/run';
/**
 * The hardware check of `/lab/check/` (`docs/lab-run-records.md`, "Hardware check"). It reads the
 * device's facts, compares the flavor of the emulated double that the TypeShade device selects
 * with TypeCV's rule before change 0085, and runs the cases of `website/src/lib/hardware-check/`
 * on the CPU tier and on the device. It posts its log, then the record. It uses the site's runtime
 * entries only (`/runtime/gpu.js` and `/runtime/internal/gpu.js`).
 */
type Request = {
  type: 'start';
  href: string;
  /** `/version.json`'s commit. */
  commit: string | null;
  /** `/runtime/version.json`'s TypeShade pin. */
  typeshade: string | null;
  note: string;
  /** The page's `?cases=<name>,…`: the cases to run; all of them when empty. */
  cases: string[];
};

const FIXTURE_URL = '/sample/probe/south-building-pair-triangulation.json';
/**
 * The options with which the lab's worker opens its device (`openOptions` in
 * `website/src/workers/sfm.worker.ts`), as far as they decide the flavor: a `GPUDevice` of
 * TypeCV's own with the adapter's limits, and the flavor `'auto'`. Phase 5c of
 * `docs/library-structure.md` moves the lab's device options into `lab-device`; this copy then
 * imports them from there.
 */
const LAB_DEVICE_OPTIONS = {
  backend: 'webgpu',
  fp64Flavor: 'auto',
  adapterLimits: true,
  boundedSubmissions: false,
  solverCheckpoints: true,
  maxModelsPerBatch: 1024,
} as const;

const log = (line: string) => self.postMessage({ type: 'log', line });
const describe = (error: unknown) =>
  (error instanceof Error ? error.stack || String(error) : String(error)).slice(
    0,
    HARDWARE_CHECK_MAX_MESSAGE,
  );
const cut = (value: unknown, max = 512) => String(value ?? '').slice(0, max);
const identity = (info: any): HardwareCheckIdentity | null =>
  info
    ? {
        vendor: cut(info.vendor),
        architecture: cut(info.architecture),
        device: cut(info.device),
        description: cut(info.description),
      }
    : null;
const load = (path: string) => import(/* @vite-ignore */ new URL(path, self.location.origin).href);

async function check(request: Request) {
  const startedAt = new Date().toISOString();
  const [gpu, internal] = await Promise.all([
    load('/runtime/gpu.js'),
    load('/runtime/internal/gpu.js'),
  ]);
  const userAgent = cut(navigator.userAgent);
  // The adapter that WebGPU offers, and the adapter info of a GPUDevice of the page's own: the
  // two signals of the rule before change 0085.
  let adapter: HardwareCheckAdapter | null = null,
    requestedAdapterInfo: HardwareCheckIdentity | null = null,
    deviceAdapterInfo: HardwareCheckIdentity | null = null;
  try {
    const requested = await navigator.gpu?.requestAdapter();
    if (requested) {
      const facts = adapterFacts(requested);
      adapter = {
        ...facts,
        ...identity(facts)!,
        features: facts.features.slice(0, 64).map((f) => cut(f, 64)),
      };
      requestedAdapterInfo = identity((requested as any).info);
      try {
        const own = await requested.requestDevice();
        deviceAdapterInfo = identity((own as any).adapterInfo);
        own.destroy();
      } catch {
        deviceAdapterInfo = null;
      }
    }
  } catch {
    adapter = null;
  }
  log(
    adapter
      ? `Adapter: ${[adapter.vendor, adapter.architecture, adapter.description].filter(Boolean).join(' ') || '(no identity)'}` +
          (adapter.fallbackAdapter ? ' (fallback)' : '')
      : 'No WebGPU adapter',
  );
  const flavor: { -readonly [K in keyof HardwareCheckFlavor]: HardwareCheckFlavor[K] } = {
    requested: 'auto',
    selected: null,
    selectedDefault: null,
    previous: previousFp64Flavor({ handedDevice: true, deviceAdapterInfo, userAgent }),
    previousRequested: previousFp64Flavor({
      handedDevice: false,
      requestedAdapterInfo,
      userAgent,
    }),
    same: null,
    signals: { deviceAdapterInfo, requestedAdapterInfo, userAgent },
    info: null,
    error: null,
  };
  let timeBudgetMs: number | null = null,
    typeshadeVersion: string | null = null;
  try {
    const lab = await gpu.ShadeDevice.open(LAB_DEVICE_OPTIONS);
    try {
      flavor.selected = lab.fp64Flavor;
      const info = lab.info?.webgpu;
      flavor.info = info
        ? {
            provenance: info.provenance,
            vendor: info.vendor === null ? null : cut(info.vendor),
            architecture: info.architecture === null ? null : cut(info.architecture),
            device: info.device === null ? null : cut(info.device),
            description: info.description === null ? null : cut(info.description),
          }
        : null;
      typeshadeVersion = lab.info?.build?.version ? cut(lab.info.build.version, 32) : null;
      timeBudgetMs = lab.timeBudgetMs ?? null;
    } finally {
      await lab.dispose();
    }
    const plain = await gpu.ShadeDevice.open({ backend: 'webgpu' });
    flavor.selectedDefault = plain.fp64Flavor;
    await plain.dispose();
  } catch (error) {
    flavor.error = describe(error);
  }
  flavor.same =
    flavor.selected === null || flavor.selectedDefault === null
      ? null
      : flavor.selected === flavor.previous && flavor.selectedDefault === flavor.previousRequested;
  log(
    `Flavor of the emulated double: ${flavor.selected ?? 'no device'} on the lab's path, ` +
      `${flavor.selectedDefault ?? 'no device'} on the default path; the rule before change ` +
      `0085 gives ${flavor.previous} and ${flavor.previousRequested}` +
      (flavor.same === null ? '' : flavor.same ? ': the same' : ': DIFFERENT'),
  );
  const fixture: TriangulationFixture = await (await fetch(FIXTURE_URL)).json();
  const api: CheckApis = { gpu, internal, fixture };
  const { cases, aborted, resume } = await runCases(api, { only: request.cases, log });
  const nav = navigator as Navigator & { deviceMemory?: number };
  const record: HardwareCheckRecord = {
    format: HARDWARE_CHECK_FORMAT,
    startedAt,
    finishedAt: new Date().toISOString(),
    site: { commit: request.commit, href: cut(request.href), typeshade: request.typeshade },
    client: {
      userAgent,
      platform: cut(nav.platform, 64),
      language: cut(nav.language, 35),
      hardwareConcurrency: Number.isSafeInteger(nav.hardwareConcurrency)
        ? nav.hardwareConcurrency
        : 0,
      deviceMemory: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    },
    adapter,
    flavor,
    device: {
      timeBudgetMs,
      timestampQuery: adapter?.features.includes('timestamp-query') ?? false,
      typeshadeVersion,
    },
    cases,
    aborted,
    resume,
    note: request.note.slice(0, HARDWARE_CHECK_MAX_NOTE),
  };
  if (resume) log(`To run the cases that did not run, open this page with ${resume}`);
  self.postMessage({ type: 'result', result: record });
}

self.onmessage = (event: MessageEvent<Request>) => {
  if (event.data.type !== 'start') return;
  check(event.data).catch((error) => self.postMessage({ type: 'error', message: describe(error) }));
};
