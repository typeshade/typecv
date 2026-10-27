/**
 * The runner of the hardware check's cases (`cases.ts`), shared by the check's worker
 * (`website/src/workers/hardware-check.worker.ts`), the CPU-tier digests of Node
 * (`tests/typeshade/hardware-check-cpu.test.mjs`) and the browser test
 * (`website/hardware-check-browser.test.mjs`). A case runs once on the CPU tier, the oracle, then
 * twice on one WebGPU device, cold and warm. Each run races the device's loss and a timeout, as
 * the triangulation probe does. After a loss or a timeout the cases that follow do not run, and
 * the record says how to run them (`resume`).
 */
import { digest, digestParts } from '../../../../tests/typeshade/digest.mjs';
import { gpuRecord } from '../run-record/assemble.ts';
import type { RunRecordChunk, RunRecordKernelTime } from '../run-record/schema.ts';
import { CASES, type CheckApis, type CheckCaseDefinition } from './cases.ts';
import {
  HARDWARE_CHECK_MAX_GPU_ROWS,
  HARDWARE_CHECK_MAX_MESSAGE,
  type HardwareCheckCase,
  type HardwareCheckLoss,
  type HardwareCheckOutcome,
} from './record.ts';

/** A run that neither completes nor loses the device in this time is recorded as a timeout. */
export const CHECK_STEP_TIMEOUT_MS = 180000;

export interface TierRun {
  readonly outcome: Exclude<HardwareCheckOutcome, 'skipped'>;
  /** The answer of each run that completed, in order. */
  readonly answers: readonly unknown[];
  readonly coldMs: number | null;
  readonly warmMs: number | null;
  readonly gpu: readonly RunRecordKernelTime[];
  readonly chunks: Readonly<Record<string, RunRecordChunk>> | null;
  readonly loss: HardwareCheckLoss | null;
  readonly error: string | null;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const describe = (error: unknown) =>
  (error instanceof Error ? error.stack || String(error) : String(error)).slice(
    0,
    HARDWARE_CHECK_MAX_MESSAGE,
  );
const chunkRates = (device: any): Record<string, RunRecordChunk> | null => {
  try {
    const entries = Object.entries(device.chunkRates ?? {}).slice(0, 16) as [string, any][];
    return entries.length
      ? Object.fromEntries(
          entries.map(([key, r]) => [
            key.slice(0, 64),
            {
              msPerUnit: r.msPerUnit,
              chunks: r.chunks,
              longestMs: r.longestMs,
              lastUnits: r.lastUnits,
            },
          ]),
        )
      : null;
  } catch {
    return null;
  }
};
const lossOf = (info: any, lastKernel: string): HardwareCheckLoss => ({
  message: String(info?.message ?? '').slice(0, HARDWARE_CHECK_MAX_MESSAGE),
  reason: String(info?.reason ?? '').slice(0, 64),
  lastKernel: (lastKernel || 'none').slice(0, 64),
  uncaptured: (info?.diagnostics?.recentErrors ?? []).slice(-4).map((e: any) => ({
    name: String(e?.name ?? '').slice(0, 64),
    message: String(e?.message ?? '').slice(0, 400),
  })),
});

/**
 * Run a case `runs` times on one device of `backend`. A family test that opens a device of its own
 * gets this one, so the device does not dispose itself until the runs end.
 */
export async function runOnTier(
  c: CheckCaseDefinition,
  api: CheckApis,
  backend: 'webgpu' | 'cpu',
  runs: number,
): Promise<TierRun> {
  const entry = c.entry === 'gpu' ? api.gpu : api.internal;
  let lastKernel = '';
  let device: any;
  try {
    device = await entry.ShadeDevice.open({
      backend,
      poolBytes: 0,
      onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
        if (phase === 'start') lastKernel = kernel;
      },
    });
  } catch (error) {
    return {
      outcome: 'open-failed',
      answers: [],
      coldMs: null,
      warmMs: null,
      gpu: [],
      chunks: null,
      loss: null,
      error: describe(error),
    };
  }
  const answers: unknown[] = [];
  const times: number[] = [];
  let outcome: TierRun['outcome'] = 'completed',
    gpu: RunRecordKernelTime[] = [],
    chunks: TierRun['chunks'] = null,
    loss: HardwareCheckLoss | null = null,
    failure: string | null = null;
  const lost: Promise<{ kind: 'lost'; info: any }> = device.lost
    ? device.lost.then((info: any) => ({ kind: 'lost' as const, info }))
    : new Promise(() => {});
  Object.defineProperty(device, 'dispose', {
    configurable: true,
    writable: true,
    value: async () => {},
  });
  try {
    for (let k = 0; k < runs && outcome === 'completed'; k++) {
      device.resetKernelStatistics?.();
      const started = performance.now();
      let timer: ReturnType<typeof setTimeout> | undefined;
      let result: any = await Promise.race([
        c.run(api, device).then(
          (answer) => ({ kind: 'completed' as const, answer }),
          (error) => ({ kind: 'error' as const, error }),
        ),
        lost,
        new Promise<{ kind: 'timeout' }>((resolve) => {
          timer = setTimeout(() => resolve({ kind: 'timeout' }), CHECK_STEP_TIMEOUT_MS);
        }),
      ]);
      clearTimeout(timer);
      times.push(Math.round(performance.now() - started));
      if (k === 0) {
        gpu = gpuRecord(device.kernelStatistics?.rows, device.kernelStatistics?.dropped).gpu.slice(
          0,
          HARDWARE_CHECK_MAX_GPU_ROWS,
        );
        chunks = chunkRates(device);
      }
      // An error that a device loss caused names the loss: wait a moment for it.
      if (result.kind === 'error') {
        const late = await Promise.race([lost, sleep(1000).then(() => undefined)]);
        if (late) result = late;
      }
      if (result.kind === 'completed') answers.push(result.answer);
      else if (result.kind === 'lost') {
        outcome = 'lost';
        loss = lossOf(result.info, lastKernel);
      } else if (result.kind === 'timeout') {
        outcome = 'timeout';
        loss = lossOf(
          {
            message: `No answer and no device loss in ${CHECK_STEP_TIMEOUT_MS} ms`,
            reason: 'timeout',
          },
          lastKernel,
        );
      } else {
        outcome = 'error';
        failure = describe(result.error);
      }
    }
  } finally {
    delete device.dispose;
    // A lost device may never finish its queue: wait for the disposal a short time only.
    const disposal = device.dispose().catch(() => {});
    await (outcome === 'completed' ? disposal : Promise.race([disposal, sleep(5000)]));
  }
  return {
    outcome,
    answers,
    coldMs: times[0] ?? null,
    warmMs: times[1] ?? null,
    gpu,
    chunks,
    loss,
    error: failure,
  };
}

/** The CPU tier's digests of a case: its input, its answer and the answer's parts. */
export async function cpuDigests(c: CheckCaseDefinition, api: CheckApis) {
  const cpu = await runOnTier(c, api, 'cpu', 1);
  if (cpu.outcome !== 'completed') throw new Error(`${c.name} on the CPU tier: ${cpu.error}`);
  return {
    input: await digest(c.input(api)),
    cpu: await digest(cpu.answers[0]),
    parts: (await digestParts(cpu.answers[0])).parts,
  };
}

const skippedCase = (c: CheckCaseDefinition, reason: string): HardwareCheckCase => ({
  name: c.name,
  family: c.family,
  bound: c.bound,
  outcome: 'skipped',
  skipped: reason,
  coldMs: null,
  warmMs: null,
  cpuMs: null,
  repeatable: null,
  digests: {
    input: null,
    device: null,
    deviceWarm: null,
    cpu: null,
    deviceParts: null,
    cpuParts: null,
  },
  agreement: null,
  pass: null,
  summary: {},
  gpu: [],
  chunks: null,
  loss: null,
  error: null,
});

/** Run one case on the CPU tier and on the device, and build its entry of the record. */
export async function runCase(c: CheckCaseDefinition, api: CheckApis): Promise<HardwareCheckCase> {
  const cpu = await runOnTier(c, api, 'cpu', 1);
  const device = await runOnTier(c, api, 'webgpu', 2);
  const [first, second] = device.answers;
  const reference = cpu.answers[0];
  const digests = {
    input: await digest(c.input(api)),
    device: first === undefined ? null : await digest(first),
    deviceWarm: second === undefined ? null : await digest(second),
    cpu: reference === undefined ? null : await digest(reference),
    deviceParts: first === undefined ? null : (await digestParts(first)).parts,
    cpuParts: reference === undefined ? null : (await digestParts(reference)).parts,
  };
  const compared =
    first !== undefined && reference !== undefined ? c.compare(api, first, reference) : null;
  const cpuError =
    cpu.outcome === 'completed'
      ? null
      : `The CPU tier: ${cpu.outcome}. ${cpu.error ?? ''}`.slice(0, HARDWARE_CHECK_MAX_MESSAGE);
  return {
    name: c.name,
    family: c.family,
    bound: c.bound,
    outcome: device.outcome,
    skipped: null,
    coldMs: device.coldMs,
    warmMs: device.warmMs,
    cpuMs: cpu.coldMs,
    repeatable:
      digests.device !== null && digests.deviceWarm !== null
        ? digests.device === digests.deviceWarm
        : null,
    digests,
    agreement: compared && {
      maxAbs: compared.maxAbs,
      maxRel: compared.maxRel,
      mismatches: compared.mismatches,
      compared: compared.compared,
      problems: compared.problems.map((p) => p.slice(0, 200)),
    },
    pass: compared ? compared.pass : null,
    summary: compared?.summary ?? {},
    gpu: device.gpu,
    chunks: device.chunks,
    loss: device.loss,
    error: device.error ?? cpuError,
  };
}

/**
 * Run the cases named in `only` (all of them when it is empty), in the order of `CASES`. After a
 * device loss or a timeout the cases that follow are skipped, and `resume` names them.
 */
export async function runCases(
  api: CheckApis,
  { only = [], log = () => {} }: { only?: readonly string[]; log?: (line: string) => void } = {},
): Promise<{ cases: HardwareCheckCase[]; aborted: string | null; resume: string | null }> {
  const selected = CASES.filter((c) => only.length === 0 || only.includes(c.name));
  const cases: HardwareCheckCase[] = [];
  let stopped: string | null = null;
  for (const c of selected) {
    if (stopped) {
      cases.push(skippedCase(c, `Not run after the device loss in ${stopped}`));
      continue;
    }
    log(`${c.name}: running on the CPU tier, then twice on the device`);
    const entry = await runCase(c, api);
    cases.push(entry);
    log(
      `${c.name}: ${entry.outcome}` +
        (entry.pass === null ? '' : entry.pass ? ', within the bound' : ', OUTSIDE THE BOUND') +
        (entry.repeatable === false ? ', the two runs differ' : '') +
        (entry.coldMs === null ? '' : `, ${entry.coldMs} ms cold, ${entry.warmMs ?? '?'} ms warm`) +
        (entry.loss ? ` (${entry.loss.reason}: ${entry.loss.message.split('\n')[0]})` : ''),
    );
    if (entry.outcome === 'lost' || entry.outcome === 'timeout') stopped = c.name;
  }
  const rest = cases.filter((c) => c.outcome === 'skipped').map((c) => c.name);
  return {
    cases,
    aborted: stopped
      ? `The device was lost or timed out in ${stopped}; the cases after it did not run`
      : null,
    resume: stopped && rest.length ? `?cases=${rest.join(',')}` : null,
  };
}
