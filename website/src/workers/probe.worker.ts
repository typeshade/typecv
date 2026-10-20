/// <reference lib="webworker" />
import {
  PROBE_RECORD_FORMAT,
  PROBE_RECORD_MAX_MESSAGE,
  type ProbeChunk,
  type ProbeRecord,
} from '../lib/probe-record';
/**
 * The triangulation probe of `/lab/probe/` (`docs/lab-run-records.md`, "Probe"). It runs the job of
 * `tests/fixtures/south-building-pair-triangulation.json` in the three variants of
 * `tests/typeshade/south-building-triangulation-webgpu.test.mjs`, each on a fresh `ShadeDevice`.
 * When the lab job loses the device, it bisects the tracks to find the track whose work hangs.
 * It uses the public GPU module only (`ShadeDevice`, `triangulateTracksGpu`) and sends nothing.
 */
type Request = {
  type: 'start';
  /** The page's `?bounded=1`: fenced submissions, as in the lab. */
  bounded: boolean;
  /** The page's `?fp64=`: a forced flavor of the emulated double; the device picks otherwise. */
  fp64?: 'float' | 'integer';
  /**
   * The page's `?probe-fail=<track>`, a test hook only: a step whose tracks hold this index
   * reports a simulated loss when the chunk that holds the track completes.
   */
  simulatedLossTrack?: number;
  /**
   * The page's `?tracks=<first>-<last>`: run every step on these tracks only, to go on from a
   * set that an earlier probe named after the browser refused a new device.
   */
  tracks?: { first: number; last: number };
  href: string;
  commit: string | null;
};
type Range = { first: number; end: number };
type Override = { refinementIterations?: number; maxPairHypotheses?: number };
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
/** The most steps of one probe: each loss costs the watchdog's 2 to 7 s. */
const MAX_STEPS = 40;
/** A step that neither completes nor loses the device in this time is recorded as a timeout. */
const STEP_TIMEOUT_MS = 180000;
/** The least chunk of `triangulateTracks`: one workgroup of tracks (`src/gpu/triangulation.ts`). */
const TRACKS_PER_WORKGROUP = 16;
/** The variants of the test file, from the simplest to the full kernel. */
const VARIANTS: { name: string; override: Override }[] = [
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

async function probe(request: Request) {
  const startedAt = new Date().toISOString();
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
  const steps: any[] = [];
  let aborted: string | null = null;
  const trackData = (index: number) => ({
    index,
    pixels: fixture.pixels[index],
    expectedStatus: fixture.expectedStatus[index],
  });

  /** One step: open a device, run the tracks of `range`, dispose. Null when the budget is spent. */
  async function step(
    kind: 'variant' | 'chunk' | 'half',
    name: string,
    range: Range,
    override: Override,
  ) {
    if (aborted || steps.length >= MAX_STEPS) return null;
    const options = { ...fixture.options, ...override };
    const record: any = {
      step: steps.length + 1,
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
    };
    steps.push(record);
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
      record.outcome = 'open-failed';
      record.error = describeError(error).slice(0, 2000);
      aborted = ('No WebGPU device opened at step ' + record.step + ': ' + record.error).slice(
        0,
        PROBE_RECORD_MAX_MESSAGE,
      );
      log(`${record.step}. ${name}, ${label(range)}: no device (${String(error)})`);
      return record;
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
        message: String(final.info?.message ?? '').slice(0, 2000),
        reason: String(final.info?.reason ?? '').slice(0, 64),
        lastKernel: (progress.lastKernel || 'none').slice(0, 64),
        simulated: final.simulated,
        chunksStarted: progress.chunksStarted,
        chunksCompleted: progress.chunksCompleted,
        completedTracks: progress.completedTracks,
        inFlight,
      };
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
      aborted = `Step ${record.step} timed out: the device state is not known`;
      log(
        `${record.step}. ${name}, ${label(range)}: timed out at kernel ${record.loss.lastKernel}`,
      );
    } else {
      record.outcome = 'error';
      record.error = describeError(final.error).slice(0, 2000);
      log(`${record.step}. ${name}, ${label(range)}: error (${String(final.error)})`);
    }
    // A lost device may never finish its queue: wait for the disposal a short time only. A
    // simulated loss leaves a working device, so its run is waited for.
    const disposal = device.dispose().catch(() => {});
    await (final.kind === 'lost' && !final.simulated
      ? Promise.race([disposal, sleep(5000)])
      : disposal);
    return record;
  }

  // 1. The three variants, in the order of the test file.
  const variants: any[] = [];
  for (const variant of VARIANTS) {
    const record = await step('variant', variant.name, all, variant.override);
    if (record) variants.push(record);
  }
  // 2. The bisection, when the lab job lost the device.
  const labJob = variants.find((record) => record.name === 'the lab job');
  let bisection: any = null;
  if (labJob?.outcome === 'lost') {
    const override = VARIANTS[2].override,
      hanging: number[] = [],
      minimal: Range[] = [],
      unresolved: Range[] = [];
    const lostIn = (record: any) => record?.outcome === 'lost';
    /** Narrow a range whose run lost the device, `record` being that run. */
    const narrow = async (range: Range, record: any): Promise<void> => {
      if (span(range) === 1) {
        hanging.push(range.first);
        log(`Hanging track: ${range.first}`);
        return;
      }
      if (aborted || steps.length >= MAX_STEPS) {
        unresolved.push(range);
        return;
      }
      // First the chunk that was in flight, when it is smaller than the range.
      const inFlight: Range | null = record?.loss?.inFlight ?? null;
      if (inFlight && span(inFlight) < span(range)) {
        const chunk = await step('chunk', 'the chunk in flight', inFlight, override);
        if (!chunk) {
          unresolved.push(range);
          return;
        }
        if (lostIn(chunk)) return narrow(inFlight, chunk);
      }
      // Then the two halves, each on a fresh device; every half that loses is narrowed.
      const middle = range.first + Math.floor(span(range) / 2),
        halves: Range[] = [
          { first: range.first, end: middle },
          { first: middle, end: range.end },
        ],
        records: any[] = [];
      for (const half of halves) records.push(await step('half', 'a half', half, override));
      if (records.some((r) => r === null)) {
        unresolved.push(range);
        return;
      }
      const losing = halves.filter((_, k) => lostIn(records[k]));
      if (!losing.length) {
        minimal.push(range);
        log(`${label(range)} lost the device, and neither half did`);
        return;
      }
      for (const [k, half] of halves.entries())
        if (lostIn(records[k])) await narrow(half, records[k]);
    };
    await narrow(all, labJob);
    const withData = (ranges: Range[]) =>
      ranges.map((range) => ({
        ...range,
        tracks: span(range),
        data:
          span(range) <= TRACKS_PER_WORKGROUP
            ? Array.from({ length: span(range) }, (_, k) => trackData(range.first + k))
            : null,
      }));
    bisection = {
      override,
      hangingTracks: hanging.map(trackData),
      minimalSets: withData(minimal),
      unresolved: withData(unresolved),
      steps: steps.length - variants.length,
    };
  }
  // After an aborted probe, the smallest set that lost the device and that no later step
  // resolved: the probe opened again with `?tracks=` goes on from it.
  const smallestLost = aborted
    ? steps.filter((record) => record.outcome === 'lost').sort((x, y) => x.tracks - y.tracks)[0]
    : undefined;
  const resume = smallestLost ? `?tracks=${smallestLost.first}-${smallestLost.end - 1}` : null;
  if (resume) log(`Aborted. To go on, restart the browser and open this page with ${resume}`);
  const result: ProbeRecord = {
    format: PROBE_RECORD_FORMAT,
    startedAt,
    finishedAt: new Date().toISOString(),
    site: { commit: request.commit, href: request.href },
    client: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency },
    adapter: await adapterIdentity(),
    device: {
      boundedSubmissions: request.bounded,
      fp64Requested: request.fp64 ?? 'auto',
      stepTimeoutMs: STEP_TIMEOUT_MS,
      maxSteps: MAX_STEPS,
    },
    simulatedLossTrack: request.simulatedLossTrack ?? null,
    range: all,
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
    bisection,
    aborted,
    resume,
    steps,
  };
  self.postMessage({ type: 'result', result });
}

self.onmessage = (event: MessageEvent<Request>) => {
  if (event.data.type !== 'start') return;
  probe(event.data).catch((error) =>
    self.postMessage({ type: 'error', message: describeError(error) }),
  );
};
