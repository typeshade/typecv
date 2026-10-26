/// <reference lib="webworker" />
import {
  PROBE_RECORD_FORMAT,
  PROBE_RECORD_MAX_MESSAGE,
  type ProbeChunk,
  type ProbeOverride,
  type ProbeRange,
  type ProbeRecord,
  type ProbeStep,
} from '../lib/probe-record';
import {
  PROBE_MAX_FAILED_RELOADS,
  PROBE_MAX_RELOADS,
  type ProbeState,
  type ProbeTask,
} from '../lib/probe-plan';
/**
 * The triangulation probe of `/lab/probe/` (`docs/lab-run-records.md`, "Probe"). It runs the job of
 * `tests/fixtures/south-building-pair-triangulation.json` in the three variants of
 * `tests/typeshade/south-building-triangulation-webgpu.test.mjs`, each on a fresh `ShadeDevice`.
 * When a variant loses the device, it bisects the tracks with that variant's options to find the
 * track whose work hangs, before it runs the remaining variants. It runs a plan of tasks
 * (`website/src/lib/probe-plan.ts`) and posts its state after every step, so that the page can
 * reload itself when no new device opens after a loss and go on from the saved plan. It uses the
 * public GPU module only (`ShadeDevice`, `triangulateTracksGpu`).
 */
type Request = {
  type: 'start';
  /** The page's `?bounded=1`: fenced submissions, as in the lab. */
  bounded: boolean;
  /** The page's `?fp64=`: a forced flavor of the emulated double; the device picks otherwise. */
  fp64?: 'float' | 'integer';
  /**
   * The page's `?probe-fail=<track>`, a test hook only: a step whose tracks hold this index
   * reports a simulated loss when the chunk that holds the track completes, and the next device
   * then fails to open once, as on D3D12 after a loss.
   */
  simulatedLossTrack?: number;
  /**
   * The page's `?tracks=<first>-<last>`: run every step on these tracks only, to go on from a
   * set that an earlier probe named.
   */
  tracks?: { first: number; last: number };
  href: string;
  commit: string | null;
  /** The saved state to go on from after a reload; null for a new probe. */
  state: ProbeState | null;
  /** Whether the page can save the state and reload; without storage a loss of the adapter ends the probe. */
  canReload: boolean;
};
type Range = ProbeRange;
type Fixture = {
  source: string;
  options: { maxErrorPx: number; minAngleDeg: number };
  views: {
    camera: Record<string, number>;
    pose: { R: number[]; t: number[] };
  }[];
  pixels: number[][];
  expectedStatus: number[];
};

const FIXTURE_URL = '/sample/probe/south-building-pair-triangulation.json';
/** The most steps of one probe, across reloads: each loss costs the watchdog's 2 to 7 s. */
const MAX_STEPS = 40;
/** A step that neither completes nor loses the device in this time is recorded as a timeout. */
const STEP_TIMEOUT_MS = 180000;
/** The least chunk of `triangulateTracks`: one workgroup of tracks (`src/geometry/gpu/triangulation.ts`). */
const TRACKS_PER_WORKGROUP = 16;
/** The variants of the test file, from the simplest to the full kernel. */
const VARIANTS: { name: string; override: ProbeOverride }[] = [
  {
    name: 'no refinement, one pair hypothesis',
    override: { refinementIterations: 0, maxPairHypotheses: 1 },
  },
  { name: 'no refinement', override: { refinementIterations: 0 } },
  { name: 'the lab job', override: {} },
];

/** The four numbers of a chunk rate, as the probe record holds them; null without a rate. */
const chunkOf = (rate: any): ProbeChunk | null =>
  rate
    ? {
        msPerUnit: rate.msPerUnit,
        chunks: rate.chunks,
        longestMs: rate.longestMs,
        lastUnits: rate.lastUnits,
      }
    : null;
const log = (line: string) => self.postMessage({ type: 'log', line });
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const describeError = (error: unknown) =>
  error instanceof Error ? error.stack || String(error) : String(error);
const span = (range: Range) => range.end - range.first;
const label = (range: Range) =>
  span(range) === 1 ? `track ${range.first}` : `tracks ${range.first}..${range.end - 1}`;
const same = (a: Range, b: Range) => a.first === b.first && a.end === b.end;

async function adapterIdentity() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) return null;
    const info = (adapter as any).info ?? {};
    return {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      fallbackAdapter: (adapter as any).isFallbackAdapter === true,
    };
  } catch {
    return null;
  }
}

/** A step that opened no usable device: the page may reload and run the same task again. */
type NoDevice = { kind: 'no-device'; error: string };

async function probe(request: Request) {
  const api = await import(
    /* @vite-ignore */ new URL('/runtime/gpu.js', self.location.origin).href
  );
  const fixture: Fixture = await (await fetch(FIXTURE_URL)).json();
  const tracks = fixture.pixels.map((pixels) =>
    fixture.views.map((view, k) => ({
      camera: view.camera,
      pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
      pixel: [pixels[2 * k], pixels[2 * k + 1]],
    })),
  );
  const all: Range = request.tracks
    ? {
        first: Math.min(request.tracks.first, tracks.length - 1),
        end: Math.min(Math.max(request.tracks.first, request.tracks.last) + 1, tracks.length),
      }
    : { first: 0, end: tracks.length };
  const state: ProbeState = request.state ?? {
    version: 1,
    startedAt: new Date().toISOString(),
    range: all,
    adapter: null,
    steps: [],
    plan: VARIANTS.map((_, index) => ({ kind: 'variant', index })),
    bisection: null,
    aborted: null,
    reloads: 0,
    failedReloads: 0,
    simulatedOpenFailure: false,
  };
  // The adapter's identity is read before the first step: after a loss on D3D12 it is gone.
  state.adapter ??= await adapterIdentity();
  const save = () => self.postMessage({ type: 'state', state });
  const trackData = (index: number) => ({
    index,
    pixels: fixture.pixels[index],
    expectedStatus: fixture.expectedStatus[index],
  });

  /** The record of a step before it runs. */
  const blank = (
    kind: 'variant' | 'chunk' | 'half',
    name: string,
    range: Range,
    override: ProbeOverride,
  ): any => ({
    step: state.steps.length + 1,
    kind,
    name,
    override,
    first: range.first,
    end: range.end,
    tracks: span(range),
    outcome: 'error',
    elapsedMs: null,
    fp64Flavor: null,
    chunks: null,
    dispatches: null,
    submissions: null,
    accepted: null,
    expectedAccepted: fixture.expectedStatus.slice(range.first, range.end).filter((s) => s === 0)
      .length,
    withinEight: null,
    statusCounts: null,
    statusMismatches: null,
    loss: null,
    error: null,
  });
  /** One step: open a device, run the tracks of `range`, dispose. */
  async function step(
    kind: 'variant' | 'chunk' | 'half',
    name: string,
    range: Range,
    override: ProbeOverride,
  ): Promise<ProbeStep | NoDevice> {
    const options = { ...fixture.options, ...override };
    const record = blank(kind, name, range, override);
    if (state.simulatedOpenFailure) {
      state.simulatedOpenFailure = false;
      return { kind: 'no-device', error: 'Simulated open failure after a simulated loss' };
    }
    const progress = { lastKernel: '', chunksStarted: 0, chunksCompleted: 0, completedTracks: 0 };
    let simulate: (info: { message: string; reason: string; inFlight: Range }) => void = () => {};
    const simulated = new Promise<{ message: string; reason: string; inFlight: Range }>(
      (resolve) => (simulate = resolve),
    );
    let device: any;
    try {
      device = await api.ShadeDevice.open({
        backend: 'webgpu',
        fp64Flavor: request.fp64 ?? 'auto',
        boundedSubmissions: request.bounded,
        solverCheckpoints: true,
        maxModelsPerBatch: request.bounded ? 64 : 1024,
        onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
          if (phase === 'start') {
            progress.lastKernel = kernel;
            if (kernel === 'triangulateTracks') progress.chunksStarted++;
            return;
          }
          if (kernel !== 'triangulateTracks') return;
          const units = device?.chunkRates?.triangulateTracks?.lastUnits ?? 0,
            first = range.first + progress.completedTracks;
          progress.chunksCompleted++;
          progress.completedTracks += units;
          const target = request.simulatedLossTrack;
          if (target !== undefined && target >= first && target < first + units)
            simulate({
              message: `Simulated loss of the chunk that holds track ${target} (?probe-fail=)`,
              reason: 'simulated',
              inFlight: { first, end: first + units },
            });
        },
      });
    } catch (error) {
      return { kind: 'no-device', error: describeError(error).slice(0, PROBE_RECORD_MAX_MESSAGE) };
    }
    record.fp64Flavor = device.fp64Flavor;
    log(`${record.step}. ${name}, ${label(range)}: running`);
    const started = performance.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = api
      .triangulateTracksGpu(device, tracks.slice(range.first, range.end), options)
      .then(
        (result: any) => ({ kind: 'completed' as const, result }),
        (error: unknown) => ({ kind: 'error' as const, error }),
      );
    const lost: Promise<{ kind: 'lost'; info: any; simulated: boolean }> = device.lost
      ? device.lost.then((info: any) => ({ kind: 'lost' as const, info, simulated: false }))
      : new Promise(() => {});
    const outcome = await Promise.race([
      run,
      lost,
      simulated.then((info) => ({ kind: 'lost' as const, info, simulated: true })),
      new Promise<{ kind: 'timeout' }>((resolve) => {
        timer = setTimeout(() => resolve({ kind: 'timeout' }), STEP_TIMEOUT_MS);
      }),
    ]);
    clearTimeout(timer);
    record.elapsedMs = Math.round(performance.now() - started);
    let final: typeof outcome | Awaited<typeof lost> = outcome;
    // An error that a device loss caused names the loss: wait a moment for `lost`.
    if (outcome.kind === 'error') {
      const late = await Promise.race([lost, sleep(1000).then(() => undefined)]);
      if (late) final = late;
    }
    // A device lost before any kernel started was not usable: the same task runs again.
    if (final.kind === 'lost' && !final.simulated && !progress.lastKernel) {
      void device.dispose().catch(() => {});
      return {
        kind: 'no-device',
        error: ('Device lost before the first kernel: ' + String(final.info?.message)).slice(
          0,
          PROBE_RECORD_MAX_MESSAGE,
        ),
      };
    }
    let chunkRates: any = {};
    try {
      chunkRates = { ...(device.chunkRates ?? {}) };
    } catch {
      chunkRates = {};
    }
    record.chunks = chunkOf(chunkRates.triangulateTracks);
    if (final.kind === 'completed') {
      const result = final.result,
        status = [...result.status] as number[],
        counts: Record<string, number> = {},
        mismatches: number[] = [];
      for (const [h, s] of status.entries()) {
        counts[s] = (counts[s] ?? 0) + 1;
        if (s !== fixture.expectedStatus[range.first + h]) mismatches.push(range.first + h);
      }
      record.outcome = 'completed';
      record.chunks = chunkOf(result.execution.report.chunks?.triangulateTracks) ?? record.chunks;
      record.dispatches = result.execution.report.dispatches;
      record.submissions = result.execution.report.submissions;
      record.accepted = status.filter((s) => s === 0).length;
      record.withinEight = Math.abs(record.accepted - record.expectedAccepted) <= 8;
      record.statusCounts = counts;
      record.statusMismatches = { count: mismatches.length, first: mismatches.slice(0, 32) };
      log(
        `${record.step}. ${name}, ${label(range)}: completed in ${record.elapsedMs} ms, ` +
          `${record.accepted} accepted (${record.expectedAccepted} expected), ` +
          `${record.chunks?.chunks ?? '?'} chunks, ${record.dispatches} dispatches`,
      );
    } else if (final.kind === 'lost') {
      // The chunk in flight: it starts after the tracks of the completed chunks and holds at
      // most four times the last chunk (`ShadeDevice.chunkSize`), the first chunk one workgroup.
      const first = range.first + progress.completedTracks,
        last = record.chunks?.lastUnits ?? 0,
        most = progress.chunksCompleted ? 4 * last : TRACKS_PER_WORKGROUP,
        inFlight: Range | null = final.simulated
          ? final.info.inFlight
          : progress.lastKernel === 'triangulateTracks' && first < range.end
            ? { first, end: Math.min(range.end, first + most) }
            : null;
      record.outcome = 'lost';
      record.loss = {
        message: String(final.info?.message ?? '').slice(0, PROBE_RECORD_MAX_MESSAGE),
        reason: String(final.info?.reason ?? '').slice(0, 64),
        lastKernel: (progress.lastKernel || 'none').slice(0, 64),
        simulated: final.simulated,
        chunksStarted: progress.chunksStarted,
        chunksCompleted: progress.chunksCompleted,
        completedTracks: progress.completedTracks,
        inFlight,
      };
      // The test hook: after a simulated loss the next device fails to open once, as on D3D12.
      if (final.simulated) state.simulatedOpenFailure = true;
      log(
        `${record.step}. ${name}, ${label(range)}: device lost after ${record.elapsedMs} ms ` +
          `at kernel ${record.loss.lastKernel}, ${progress.chunksCompleted} chunks completed` +
          (inFlight ? `, chunk in flight ${label(inFlight)}` : '') +
          ` (${record.loss.reason}: ${record.loss.message.split('\n')[0]})`,
      );
    } else if (final.kind === 'timeout') {
      record.outcome = 'timeout';
      record.loss = {
        message: `No result and no device loss in ${STEP_TIMEOUT_MS} ms`,
        reason: 'timeout',
        lastKernel: (progress.lastKernel || 'none').slice(0, 64),
        simulated: false,
        chunksStarted: progress.chunksStarted,
        chunksCompleted: progress.chunksCompleted,
        completedTracks: progress.completedTracks,
        inFlight: null,
      };
      log(
        `${record.step}. ${name}, ${label(range)}: timed out at kernel ${record.loss.lastKernel}`,
      );
    } else {
      record.outcome = 'error';
      record.error = describeError(final.error).slice(0, PROBE_RECORD_MAX_MESSAGE);
      log(`${record.step}. ${name}, ${label(range)}: error (${String(final.error)})`);
    }
    // A lost device may never finish its queue: wait for the disposal a short time only. A
    // simulated loss leaves a working device, so its run is waited for.
    const disposal = device.dispose().catch(() => {});
    await (final.kind === 'lost' && !final.simulated
      ? Promise.race([disposal, sleep(5000)])
      : disposal);
    return record as ProbeStep;
  }

  /** The step a task runs, or null for a task that only decides. */
  const stepOf = (task: ProbeTask) => {
    const override = state.bisection?.override ?? {};
    if (task.kind === 'variant')
      return [
        'variant',
        VARIANTS[task.index].name,
        state.range,
        VARIANTS[task.index].override,
      ] as const;
    if (task.kind === 'chunk')
      return ['chunk', 'the chunk in flight', task.test, override] as const;
    if (task.kind === 'half') return ['half', 'a half', task.half, override] as const;
    return null;
  };
  /** The ranges the bisection had not resolved when the probe stopped. */
  const unresolvedOf = (plan: ProbeTask[]) => {
    const ranges: Range[] = [];
    for (const task of plan) {
      const range =
        task.kind === 'narrow' ? task.range : task.kind === 'variant' ? null : task.parent;
      if (range && !ranges.some((r) => same(r, range))) ranges.push(range);
    }
    return ranges;
  };

  while (state.plan.length && !state.aborted) {
    const task = state.plan[0];
    const bisection = state.bisection;
    // The tasks that decide: no step.
    if (task.kind === 'narrow') {
      state.plan.shift();
      const { range, inFlight } = task;
      if (span(range) === 1) {
        bisection!.hanging.push(range.first);
        log(`Hanging track: ${range.first}`);
      } else if (inFlight && span(inFlight) < span(range))
        // First the chunk that was in flight, when it is smaller than the range.
        state.plan.unshift({ kind: 'chunk', parent: range, test: inFlight });
      else {
        // Then the two halves, each on a fresh device; every half that loses is narrowed.
        const middle = range.first + Math.floor(span(range) / 2);
        state.plan.unshift(
          { kind: 'half', parent: range, half: { first: range.first, end: middle } },
          { kind: 'half', parent: range, half: { first: middle, end: range.end } },
          { kind: 'halves-done', parent: range, lost: [] },
        );
      }
      continue;
    }
    if (task.kind === 'halves-done') {
      state.plan.shift();
      if (!task.lost.length) {
        bisection!.minimal.push(task.parent);
        log(`${label(task.parent)} lost the device, and neither half did`);
      } else state.plan.unshift(...task.lost.map((lost) => ({ kind: 'narrow' as const, ...lost })));
      continue;
    }
    if (state.steps.length >= MAX_STEPS) {
      log(`The budget of ${MAX_STEPS} steps is spent`);
      break;
    }
    const [kind, name, range, override] = stepOf(task)!;
    const record = await step(kind, name, range, override);
    if (record.kind === 'no-device') {
      // On D3D12 the adapter is gone for the page after a loss: reload the page and go on.
      const afterLoss = state.steps.some((s) => s.outcome === 'lost');
      if (
        afterLoss &&
        request.canReload &&
        state.reloads < PROBE_MAX_RELOADS &&
        state.failedReloads < PROBE_MAX_FAILED_RELOADS
      ) {
        state.reloads++;
        state.failedReloads++;
        log(
          `No device for step ${state.steps.length + 1} (${record.error.split('\n')[0]}): reloading the page`,
        );
        self.postMessage({ type: 'reload', state });
        return;
      }
      state.steps.push({
        ...blank(kind, name, range, override),
        outcome: 'open-failed',
        error: record.error,
      });
      state.aborted = (
        `No WebGPU device opened at step ${state.steps.length}` +
        (state.reloads ? ` after ${state.reloads} reloads` : '') +
        ': ' +
        record.error
      ).slice(0, PROBE_RECORD_MAX_MESSAGE);
      log(
        `${state.steps.length}. ${name}, ${label(range)}: no device (${record.error.split('\n')[0]})`,
      );
      save();
      break;
    }
    state.failedReloads = 0;
    state.steps.push(record);
    state.plan.shift();
    if (record.outcome === 'timeout')
      state.aborted = `Step ${record.step} timed out: the device state is not known`;
    else if (record.outcome === 'lost') {
      const inFlight = record.loss?.inFlight ?? null;
      if (task.kind === 'variant') {
        // The first variant that loses starts the bisection with its options, before the
        // remaining variants run: the hanging track is found with the fewest reloads.
        if (!state.bisection) {
          state.bisection = {
            override: VARIANTS[task.index].override,
            hanging: [],
            minimal: [],
            unresolved: [],
          };
          state.plan.unshift({ kind: 'narrow', range: state.range, inFlight });
        }
      } else if (task.kind === 'chunk')
        state.plan.unshift({ kind: 'narrow', range: task.test, inFlight });
      else if (task.kind === 'half') {
        const done = state.plan.find(
          (t): t is Extract<ProbeTask, { kind: 'halves-done' }> =>
            t.kind === 'halves-done' && same(t.parent, task.parent),
        );
        done?.lost.push({ range: task.half, inFlight });
      }
    } else if (task.kind === 'chunk') {
      // The chunk alone did not lose the device: halve its parent instead.
      const middle = task.parent.first + Math.floor(span(task.parent) / 2);
      state.plan.unshift(
        { kind: 'half', parent: task.parent, half: { first: task.parent.first, end: middle } },
        { kind: 'half', parent: task.parent, half: { first: middle, end: task.parent.end } },
        { kind: 'halves-done', parent: task.parent, lost: [] },
      );
    }
    save();
  }

  // The result. The ranges the bisection did not reach are unresolved.
  const bisection = state.bisection;
  if (bisection) bisection.unresolved.push(...unresolvedOf(state.plan));
  const withData = (ranges: Range[]) =>
    ranges.map((range) => ({
      ...range,
      tracks: span(range),
      data:
        span(range) <= TRACKS_PER_WORKGROUP
          ? Array.from({ length: span(range) }, (_, k) => trackData(range.first + k))
          : null,
    }));
  const variants = VARIANTS.flatMap((variant) =>
    state.steps.filter((s) => s.kind === 'variant' && s.name === variant.name),
  );
  // After an aborted probe, the smallest set that lost the device: the probe opened again with
  // `?tracks=` goes on from it.
  const smallestLost = state.aborted
    ? state.steps.filter((s) => s.outcome === 'lost').sort((x, y) => x.tracks - y.tracks)[0]
    : undefined;
  const resume = smallestLost ? `?tracks=${smallestLost.first}-${smallestLost.end - 1}` : null;
  if (resume) log(`Aborted. To go on, open this page with ${resume}`);
  const result: ProbeRecord = {
    format: PROBE_RECORD_FORMAT,
    startedAt: state.startedAt,
    finishedAt: new Date().toISOString(),
    site: { commit: request.commit, href: request.href },
    client: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency },
    adapter: state.adapter,
    device: {
      boundedSubmissions: request.bounded,
      fp64Requested: request.fp64 ?? 'auto',
      stepTimeoutMs: STEP_TIMEOUT_MS,
      maxSteps: MAX_STEPS,
    },
    simulatedLossTrack: request.simulatedLossTrack ?? null,
    range: state.range,
    fixture: {
      source: fixture.source,
      tracks: tracks.length,
      options: fixture.options,
      views: fixture.views,
      expectedAccepted: fixture.expectedStatus.filter((s) => s === 0).length,
    },
    variants: variants.map((record) => ({
      name: record.name,
      override: record.override,
      outcome: record.outcome,
      elapsedMs: record.elapsedMs,
      chunks: record.chunks,
      accepted: record.accepted,
      expectedAccepted: record.expectedAccepted,
      withinEight: record.withinEight,
      dispatches: record.dispatches,
    })),
    bisection: bisection
      ? {
          override: bisection.override,
          hangingTracks: bisection.hanging.map(trackData),
          minimalSets: withData(bisection.minimal),
          unresolved: withData(bisection.unresolved),
          steps: state.steps.filter((s) => s.kind !== 'variant').length,
        }
      : null,
    aborted: state.aborted,
    resume,
    reloads: state.reloads,
    steps: state.steps,
  };
  self.postMessage({ type: 'result', result });
}

self.onmessage = (event: MessageEvent<Request>) => {
  if (event.data.type !== 'start') return;
  probe(event.data).catch((error) =>
    self.postMessage({ type: 'error', message: describeError(error) }),
  );
};
