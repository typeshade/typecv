import test from 'node:test';
import assert from 'node:assert/strict';
import { handle } from './worker.mjs';
import {
  RUN_RECORD_FORMAT,
  chunkLines,
  chunkRecord,
  denseLine,
  denseRecord,
  runRecordProblem,
  summarizeStages,
} from './src/lib/run-record.ts';
import { PROBE_RECORD_FORMAT, probeRecordProblem } from './src/lib/probe-record.ts';

/** An in-memory KV with the subset of the binding the Worker uses. */
function fakeKv() {
  const store = new Map();
  return {
    store,
    async get(key) {
      return store.has(key) ? store.get(key).value : null;
    },
    async put(key, value, options = {}) {
      store.set(key, { value, metadata: options.metadata ?? null });
    },
    async list({ prefix = '', limit = 1000 } = {}) {
      const keys = [...store.keys()]
        .filter((key) => key.startsWith(prefix))
        .sort()
        .slice(0, limit)
        .map((name) => ({ name, metadata: store.get(name).metadata }));
      return { keys, list_complete: true, cursor: undefined };
    },
  };
}

export function sampleRecord(overrides = {}) {
  return {
    format: RUN_RECORD_FORMAT,
    createdAt: '2026-10-19T06:00:00.000Z',
    site: { commit: 'abc123', runtime: '/runtime/gpu.js' },
    client: {
      userAgent: 'test',
      platform: 'Linux',
      language: 'en',
      hardwareConcurrency: 4,
      deviceMemoryGb: null,
      screen: { width: 1440, height: 900, dpr: 1 },
    },
    gpu: {
      vendor: 'google',
      architecture: 'swiftshader',
      device: '',
      description: '',
      fallbackAdapter: true,
      limits: { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 },
    },
    input: {
      kind: 'sample',
      name: 'Pexels pyramid sample',
      frames: 8,
      width: 640,
      height: 360,
      sourceWidth: 640,
      sourceHeight: 360,
      durationSeconds: 20,
    },
    options: {
      frontEnd: 'corner',
      focalRatio: 1.2,
      windowSize: 32,
      ransacIterations: 500,
      maxErrorPx: 3,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      maxFeatures: 800,
      matchRatio: 0.82,
      maxDistance: 80,
      boundedSubmissions: false,
      fp64Flavor: 'float',
    },
    result: {
      registeredViews: 8,
      landmarks: 726,
      observations: 1895,
      meanReprojectionErrorPx: 0.29,
      unregistered: 0,
      matchedPairs: 18,
      featureCounts: [800, 800, 800, 800, 800, 800, 800, 800],
    },
    timing: {
      totalMs: 2411.9,
      reusedDevice: false,
      stages: {
        pairVerification: { calls: 18, elapsedMs: 1003.6, dispatches: 2412, submissions: 72 },
      },
      frameMs: [300, 280, 290, 310, 300, 295, 305, 330],
      kernels: ['pyramidBlur', 'fastCorners', 'pairModels'],
      chunks: {
        scorePairModels: { msPerUnit: 0.41, chunks: 37, longestMs: 212.4, lastUnits: 480 },
        bundleCg: { msPerUnit: 0.0021, chunks: 52, longestMs: 96.2, lastUnits: 41200 },
      },
      budgetMs: 250,
      mainThread: {
        longTasks: 4,
        longTaskMs: 3366.2,
        longestTaskMs: 2851.4,
        longestTaskStage: 'pnp · scorePnP (start)',
        frameGaps: 3,
        frameGapMs: 3362,
        longestGapMs: 2854.1,
        longestGapStage: 'pnp · scorePnP (start)',
      },
    },
    transfers: { applicationUploadBytes: 1234, applicationReadbackBytes: 567 },
    benchmark: { name: 'pyramid-8', pass: 'cold' },
    reference: null,
    failure: null,
    note: '',
    ...overrides,
  };
}

const post = (body, headers = {}) =>
  new Request('https://cv.typeshade.dev/api/records', {
    method: 'POST',
    headers: { origin: 'https://cv.typeshade.dev', 'cf-connecting-ip': '203.0.113.7', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });

test('the validator accepts the sample record and names the first broken field', () => {
  assert.equal(runRecordProblem(sampleRecord()), undefined);
  assert.equal(runRecordProblem(sampleRecord({ format: 'other' })), 'format');
  assert.equal(runRecordProblem(sampleRecord({ note: 'x'.repeat(501) })), 'note');
  assert.equal(runRecordProblem(sampleRecord({ gpu: null })), 'gpu');
  assert.equal(
    runRecordProblem(sampleRecord({ options: { ...sampleRecord().options, fp64Flavor: 'fast' } })),
    'options',
  );
  const failure = {
    message:
      'WebGPU device lost during bundle #12 at kernel bundleResidual: DXGI_ERROR_DEVICE_HUNG',
    operation: 'bundle #12',
    kernel: 'bundleResidual',
    reason: 'unknown',
    framesProcessed: 5,
  };
  assert.equal(runRecordProblem(sampleRecord({ failure })), undefined);
  assert.equal(runRecordProblem(sampleRecord({ failure: { ...failure, kernel: 7 } })), 'failure');
  assert.equal(runRecordProblem(sampleRecord({ failure: undefined })), 'failure');
  assert.equal(
    runRecordProblem(sampleRecord({ timing: { ...sampleRecord().timing, kernels: 'x' } })),
    'timing',
  );
  const reference = {
    name: 'COLMAP south-building',
    status: 'aligned',
    commonViews: 12,
    medianRotationDeg: 0.08,
    maxRotationDeg: 0.3,
    medianCenterOfRadius: 0.004,
    maxCenterOfRadius: 0.02,
  };
  assert.equal(runRecordProblem(sampleRecord({ reference })), undefined);
  assert.equal(
    runRecordProblem(
      sampleRecord({
        reference: {
          ...reference,
          status: 'fewer than three common cameras',
          commonViews: 2,
          medianRotationDeg: null,
          maxRotationDeg: null,
          medianCenterOfRadius: null,
          maxCenterOfRadius: null,
        },
      }),
    ),
    undefined,
  );
  assert.equal(runRecordProblem(sampleRecord({ reference: undefined })), 'reference');
  assert.equal(
    runRecordProblem(sampleRecord({ reference: { ...reference, medianRotationDeg: -1 } })),
    'reference',
  );
  assert.equal(
    runRecordProblem(sampleRecord({ timing: { ...sampleRecord().timing, frameMs: [NaN] } })),
    'timing',
  );
  assert.equal(RUN_RECORD_FORMAT, 'typecv-run-record-v5');
  // v4 and v5 carry the chunk rates and the budget; null budget means no device was opened.
  const timing = sampleRecord().timing;
  assert.equal(runRecordProblem(sampleRecord({ timing: { ...timing, chunks: {} } })), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ timing: { ...timing, budgetMs: null } })),
    undefined,
  );
  const { chunks: _chunks, budgetMs: _budget, mainThread: _main, ...v3Timing } = timing;
  assert.equal(runRecordProblem(sampleRecord({ timing: v3Timing })), 'timing');
  assert.equal(runRecordProblem(sampleRecord({ timing: { ...timing, budgetMs: 0 } })), 'timing');
  // The main-thread summary is optional in v4 (a page deployed before it existed) and bounded.
  const { mainThread: _absent, ...withoutMainThread } = timing;
  assert.equal(runRecordProblem(sampleRecord({ timing: withoutMainThread })), undefined);
  for (const bad of [
    { longTasks: -1 },
    { longTasks: 1.5 },
    { longestTaskMs: NaN },
    { longestTaskStage: 'x'.repeat(129) },
    { frameGaps: '3' },
    { longestGapMs: -2 },
  ])
    assert.equal(
      runRecordProblem(
        sampleRecord({ timing: { ...timing, mainThread: { ...timing.mainThread, ...bad } } }),
      ),
      'timing',
      JSON.stringify(bad),
    );
  assert.equal(runRecordProblem(sampleRecord({ timing: { ...timing, chunks: [] } })), 'timing');
  assert.equal(
    runRecordProblem(
      sampleRecord({
        timing: {
          ...timing,
          chunks: { bundleCg: { ...timing.chunks.bundleCg, chunks: 1.5 } },
        },
      }),
    ),
    'timing',
  );
  assert.equal(
    runRecordProblem(
      sampleRecord({
        timing: {
          ...timing,
          chunks: Object.fromEntries(
            Array.from({ length: 33 }, (_, i) => ['k' + i, timing.chunks.bundleCg]),
          ),
        },
      }),
    ),
    'timing',
  );
  // A v4 record from a page loaded before v5 still validates.
  assert.equal(runRecordProblem(sampleRecord({ format: 'typecv-run-record-v4' })), undefined);
  // A v3 record from a page loaded before v4 still validates, and carries no chunk fields.
  const v3 = sampleRecord({ format: 'typecv-run-record-v3', timing: v3Timing });
  assert.equal(runRecordProblem(v3), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ format: 'typecv-run-record-v3' })),
    'timing',
    'a v3 record with v4 fields is neither',
  );
  assert.equal(runRecordProblem(sampleRecord({ format: 'typecv-run-record-v2' })), 'format');
  assert.deepEqual(
    chunkRecord({
      scorePairModels: { msPerUnit: 0.41, chunks: 37, longestMs: 212.4, lastUnits: 480, x: 1 },
      bad: { msPerUnit: NaN, chunks: 1, longestMs: 1, lastUnits: 1 },
      none: null,
    }),
    { scorePairModels: { msPerUnit: 0.41, chunks: 37, longestMs: 212.4, lastUnits: 480 } },
  );
  assert.deepEqual(chunkRecord(undefined), {});
  assert.deepEqual(chunkLines(timing.chunks), [
    'scorePairModels: 37 chunks, longest 212 ms, 0.41 ms/unit',
    'bundleCg: 52 chunks, longest 96 ms, 0.0021 ms/unit',
  ]);
  assert.deepEqual(
    chunkLines({ bundleCachedCg: { msPerUnit: 12.34, chunks: 1, longestMs: 12.3, lastUnits: 1 } }),
    ['bundleCachedCg: 1 chunk, longest 12 ms, 12 ms/unit'],
  );
  assert.deepEqual(chunkLines(undefined), []);
  assert.deepEqual(
    summarizeStages([
      { operation: 'runTotal', elapsedMs: 10 },
      { operation: 'pnp', elapsedMs: 2, dispatches: 3, submissions: 1 },
      { operation: 'pnp', elapsedMs: 4, dispatches: 5, submissions: 2 },
    ]),
    { pnp: { calls: 2, elapsedMs: 6, dispatches: 8, submissions: 3 } },
  );
});

/** The worker's `denseResult` of a completed run (`website/src/lib/dense-run.ts`). */
const denseResult = {
  type: 'denseResult',
  cancelled: false,
  failure: null,
  skipped: '',
  points: 123456,
  views: [
    { id: 0, status: 'done', sources: [1], depthPixels: 200000, points: 100000 },
    { id: 1, status: 'done', sources: [0], depthPixels: 190000, points: 23456 },
    { id: 2, status: 'noSource', sources: [], depthPixels: 0, points: 0 },
  ],
  memory: {
    liveBytesBefore: 1048576,
    pooledBytesBefore: 524288,
    depthStageBytes: 9839360,
    depthStageFormulaBytes: 9837000,
    fusionStageBytes: 8200000,
  },
  timing: {
    totalMs: 812345.67,
    stageMs: { photometric: 500000, geometric: 300000, fusion: 12345 },
  },
  options: {
    maxSources: 6,
    bestSources: 3,
    windowRadius: 5,
    windowStep: 2,
    photometricIterations: 8,
    geometricIterations: 4,
    minConsistentViews: 3,
  },
  kernels: ['mvsPropagate'],
  chunks: {},
};

// Verifies: docs/dense-mvs.md step 6 (5.3, the run record v5 and its dense block).
test('the v5 record carries the dense block, which v3 and v4 cannot carry', () => {
  const dense = denseRecord(denseResult);
  assert.deepEqual(dense, {
    status: 'completed',
    skipped: '',
    options: denseResult.options,
    views: 2,
    viewsWithoutDepth: 1,
    points: 123456,
    totalMs: 812345.7,
    stageMs: { photometric: 500000, geometric: 300000, fusion: 12345 },
    memory: denseResult.memory,
    failure: null,
  });
  assert.equal(runRecordProblem(sampleRecord({ dense })), undefined);
  // A run without the dense MVS has no block.
  assert.equal(denseRecord(undefined), undefined);
  assert.equal(runRecordProblem(sampleRecord()), undefined);
  for (const format of ['typecv-run-record-v4', 'typecv-run-record-v3'])
    assert.equal(
      runRecordProblem(
        sampleRecord(
          format === 'typecv-run-record-v4'
            ? { format, dense }
            : {
                format,
                dense,
                timing: (({ chunks, budgetMs, mainThread, ...rest }) => rest)(
                  sampleRecord().timing,
                ),
              },
        ),
      ),
      'dense',
      format,
    );
  for (const bad of [
    { status: 'done' },
    { points: -1 },
    { views: 1.5 },
    { stageMs: { photometric: 1, geometric: 1 } },
    { stageMs: { photometric: NaN, geometric: 1, fusion: 1 } },
    { memory: { ...dense.memory, liveBytesBefore: -1 } },
    { memory: { ...dense.memory, depthStageBytes: 'x' } },
    { options: { ...dense.options, windowRadius: 2.5 } },
    { failure: { message: 'x', stage: '', kernel: '', reason: '' } },
    { skipped: 'x'.repeat(201) },
  ])
    assert.equal(
      runRecordProblem(sampleRecord({ dense: { ...dense, ...bad } })),
      'dense',
      JSON.stringify(bad),
    );
  // A cancelled run, a failed one and a skipped one.
  const cancelled = denseRecord({ ...denseResult, cancelled: true, points: 4000, views: [] });
  assert.equal(cancelled.status, 'cancelled');
  assert.equal(cancelled.views, 0);
  assert.equal(runRecordProblem(sampleRecord({ dense: cancelled })), undefined);
  const failed = denseRecord({
    ...denseResult,
    views: [],
    failure: { message: 'WebGPU device lost', stage: 'fusion', kernel: 'mvsFuseFlags', reason: '' },
  });
  assert.equal(failed.status, 'failed');
  assert.deepEqual(failed.failure, {
    message: 'WebGPU device lost',
    stage: 'fusion',
    kernel: 'mvsFuseFlags',
    reason: '',
  });
  assert.equal(runRecordProblem(sampleRecord({ dense: failed })), undefined);
  const skipped = denseRecord({
    type: 'denseResult',
    skipped: 'More keyframes than the 15 views the dense layout holds (section 4.2)',
    memory: { liveBytesBefore: 0, pooledBytesBefore: 0 },
    timing: { totalMs: 0.01, stageMs: {} },
    options: denseResult.options,
  });
  assert.equal(skipped.status, 'skipped');
  assert.equal(skipped.memory.depthStageBytes, null);
  assert.equal(runRecordProblem(sampleRecord({ dense: skipped })), undefined);
  assert.equal(
    denseLine(dense),
    'Dense MVS completed: 123456 points from 2 depth maps in 812.3 s (photometric 500.0 s, geometric 300.0 s, fusion 12.3 s), 1.0 MiB live before',
  );
  assert.equal(denseLine(undefined), undefined);
});

test('POST stores a valid record with server fields and metadata; reads need the token', async () => {
  const RECORDS = fakeKv(),
    env = { RECORDS, RECORDS_READ_TOKEN: 'secret' };
  const created = await handle(post(sampleRecord()), env);
  assert.equal(created.status, 201);
  const { id } = await created.json();
  assert.match(id, /^[0-9]{17}-[0-9a-f]{8}$/);
  const stored = JSON.parse(RECORDS.store.get('record/' + id).value);
  assert.equal(stored.server.id, id);
  assert.equal(stored.input.name, 'Pexels pyramid sample');
  assert.equal(RECORDS.store.get('record/' + id).metadata.registered, 8);
  assert.equal(RECORDS.store.get('record/' + id).metadata.budgetMs, 250);
  assert.equal(RECORDS.store.get('record/' + id).metadata.longestChunkMs, 212);
  assert.equal(RECORDS.store.get('record/' + id).metadata.longestTaskMs, 2851);
  assert.equal(RECORDS.store.get('record/' + id).metadata.longestGapMs, 2854);
  // A v5 record with the dense block (`docs/dense-mvs.md` 5.3) is stored whole.
  const denseStore = fakeKv();
  const withDense = await handle(post(sampleRecord({ dense: denseRecord(denseResult) })), {
    RECORDS: denseStore,
  });
  assert.equal(withDense.status, 201);
  const denseId = (await withDense.json()).id;
  assert.equal(JSON.parse(denseStore.store.get('record/' + denseId).value).dense.points, 123456);
  assert.equal(denseStore.store.get('record/' + denseId).metadata.denseStatus, 'completed');
  assert.equal(denseStore.store.get('record/' + denseId).metadata.densePoints, 123456);
  assert.equal(RECORDS.store.get('record/' + id).metadata.denseStatus, null);
  const anonymous = await handle(new Request('https://cv.typeshade.dev/api/records'), env);
  assert.equal(anonymous.status, 401);
  const listed = await handle(
    new Request('https://cv.typeshade.dev/api/records?limit=10', {
      headers: { authorization: 'Bearer secret' },
    }),
    env,
  );
  assert.equal(listed.status, 200);
  const page = await listed.json();
  assert.equal(page.records.length, 1);
  assert.equal(page.records[0].id, id);
  assert.equal(page.records[0].totalMs, 2412);
  const single = await handle(
    new Request('https://cv.typeshade.dev/api/records/' + id + '/', {
      headers: { authorization: 'Bearer secret' },
    }),
    env,
  );
  assert.equal(single.status, 200);
  assert.equal((await single.json()).server.id, id);
  const missing = await handle(
    new Request('https://cv.typeshade.dev/api/records/20260101000000000-00000000', {
      headers: { authorization: 'Bearer secret' },
    }),
    env,
  );
  assert.equal(missing.status, 404);
  // A v3 record from a page loaded before v4 is still stored, without the chunk metadata.
  const {
    chunks: _chunks,
    budgetMs: _budget,
    mainThread: _main,
    ...v3Timing
  } = sampleRecord().timing;
  const old = await handle(
    post(sampleRecord({ format: 'typecv-run-record-v3', timing: v3Timing })),
    env,
  );
  assert.equal(old.status, 201);
  const oldMeta = RECORDS.store.get('record/' + (await old.json()).id).metadata;
  assert.equal(oldMeta.budgetMs, null);
  assert.equal(oldMeta.longestChunkMs, null);
  assert.equal(oldMeta.longestTaskMs, null);
  assert.equal(oldMeta.longestGapMs, null);
});

test('POST rejects foreign origins, oversized bodies, invalid records and rate abuse', async () => {
  const RECORDS = fakeKv(),
    env = { RECORDS };
  assert.equal(
    (await handle(post(sampleRecord(), { origin: 'https://evil.example' }), env)).status,
    403,
  );
  assert.equal((await handle(post('x'.repeat(256 * 1024 + 1)), env)).status, 413);
  assert.equal((await handle(post('{'), env)).status, 400);
  const invalid = await handle(post(sampleRecord({ result: null })), env);
  assert.equal(invalid.status, 400);
  assert.equal((await invalid.json()).field, 'result');
  for (let i = 0; i < 30; i++) assert.equal((await handle(post(sampleRecord()), env)).status, 201);
  assert.equal((await handle(post(sampleRecord()), env)).status, 429);
  assert.equal(
    (await handle(post(sampleRecord(), { 'cf-connecting-ip': '198.51.100.1' }), env)).status,
    201,
  );
  assert.equal(
    (await handle(new Request('https://cv.typeshade.dev/api/records', { method: 'DELETE' }), env))
      .status,
    405,
  );
  assert.equal((await handle(new Request('https://cv.typeshade.dev/api/other'), env)).status, 404);
  const unconfigured = await handle(post(sampleRecord()), {});
  assert.equal(unconfigured.status, 503);
});

test('other paths fall through to the static assets binding', async () => {
  const seen = [];
  const env = {
    ASSETS: {
      fetch: async (request) => (seen.push(new URL(request.url).pathname), new Response('ok')),
    },
  };
  const response = await handle(new Request('https://cv.typeshade.dev/lab/'), env);
  assert.equal(await response.text(), 'ok');
  assert.deepEqual(seen, ['/lab/']);
});

test('the host adds a trailing slash to every path; both spellings reach the same route', async () => {
  const env = { RECORDS: fakeKv(), RECORDS_READ_TOKEN: 'secret' };
  const slashed = await handle(
    new Request('https://cv.typeshade.dev/api/records/', post(sampleRecord())),
    env,
  );
  assert.equal(slashed.status, 201);
  const { id } = await slashed.json();
  const one = await handle(
    new Request('https://cv.typeshade.dev/api/records/' + id + '/', {
      headers: { authorization: 'Bearer secret' },
    }),
    env,
  );
  assert.equal(one.status, 200);
  const other = await handle(new Request('https://cv.typeshade.dev/api/other/'), env);
  assert.equal(other.status, 404);
});

/** A probe record of `/lab/probe/` as the probe builds it, shortened to three steps. */
export function sampleProbeRecord(overrides = {}) {
  const view = (R, t) => ({
    camera: { fx: 533.27, fy: 533.27, cx: 319.5, cy: 239.5, k1: -0.0205, k2: 0 },
    pose: { R, t },
  });
  const chunks = { msPerUnit: 1.33, chunks: 9, longestMs: 4785.6, lastUnits: 64 };
  const lost = {
    step: 1,
    kind: 'variant',
    name: 'the lab job',
    override: {},
    first: 0,
    end: 1672,
    tracks: 1672,
    outcome: 'lost',
    elapsedMs: 5150,
    fp64Flavor: 'float',
    chunks,
    dispatches: null,
    submissions: null,
    accepted: null,
    expectedAccepted: 1666,
    withinEight: null,
    statusCounts: null,
    statusMismatches: null,
    loss: {
      message: 'WebGPU device was lost',
      reason: 'unknown',
      lastKernel: 'triangulateTracks',
      simulated: false,
      chunksStarted: 3,
      chunksCompleted: 2,
      completedTracks: 80,
      inFlight: { first: 80, end: 336 },
    },
    error: null,
  };
  const completed = {
    ...lost,
    step: 3,
    kind: 'half',
    name: 'a half',
    first: 201,
    end: 202,
    tracks: 1,
    outcome: 'completed',
    elapsedMs: 31,
    dispatches: 2,
    submissions: 2,
    accepted: 1,
    expectedAccepted: 1,
    withinEight: true,
    statusCounts: { 0: 1 },
    statusMismatches: { count: 0, first: [] },
    loss: null,
  };
  const track = { index: 200, pixels: [627.7158, 210.7954, 616.5777, 192.7957], expectedStatus: 0 };
  return {
    format: PROBE_RECORD_FORMAT,
    startedAt: '2026-10-20T02:37:20.327Z',
    finishedAt: '2026-10-20T02:38:25.040Z',
    site: { commit: '52d39bb', href: 'https://cv.typeshade.dev/lab/probe/' },
    client: { userAgent: 'test', hardwareConcurrency: 16 },
    adapter: {
      vendor: 'nvidia',
      architecture: 'blackwell',
      device: '',
      description: '',
      fallbackAdapter: false,
    },
    device: {
      boundedSubmissions: false,
      fp64Requested: 'auto',
      stepTimeoutMs: 180000,
      maxSteps: 40,
    },
    simulatedLossTrack: null,
    range: { first: 0, end: 1672 },
    fixture: {
      source: 'lab COLMAP-SIFT button, frames 0 and 1',
      tracks: 1672,
      options: { maxErrorPx: 3, minAngleDeg: 0.5 },
      views: [
        view([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]),
        view([1, 0, 0, 0, 1, 0, 0, 0, 1], [-0.87, -0.07, -0.5]),
      ],
      expectedAccepted: 1666,
    },
    variants: [
      {
        name: 'the lab job',
        override: {},
        outcome: 'lost',
        elapsedMs: 5150,
        chunks,
        accepted: null,
        expectedAccepted: 1666,
        withinEight: null,
        dispatches: null,
      },
    ],
    bisection: {
      override: {},
      hangingTracks: [track],
      minimalSets: [],
      unresolved: [{ first: 300, end: 304, tracks: 4, data: [track] }],
      steps: 2,
    },
    aborted: null,
    resume: null,
    steps: [
      lost,
      {
        ...lost,
        step: 2,
        kind: 'chunk',
        name: 'the chunk in flight',
        first: 200,
        end: 201,
        tracks: 1,
      },
      completed,
    ],
    ...overrides,
  };
}

test('the probe validator accepts the probe record and names the first broken field', () => {
  assert.equal(probeRecordProblem(sampleProbeRecord()), undefined);
  assert.equal(probeRecordProblem(sampleProbeRecord({ adapter: null })), undefined);
  const aborted = sampleProbeRecord({
    aborted: 'No WebGPU device opened at step 4: Error: no adapter',
    resume: '?tracks=80-335',
  });
  assert.equal(probeRecordProblem(aborted), undefined);
  assert.equal(probeRecordProblem(null), 'not an object');
  assert.equal(probeRecordProblem(sampleProbeRecord({ format: 'typecv-probe-v0' })), 'format');
  assert.equal(probeRecordProblem(sampleProbeRecord({ finishedAt: 'later' })), 'finishedAt');
  assert.equal(
    probeRecordProblem(sampleProbeRecord({ site: { commit: null, href: 'x'.repeat(513) } })),
    'site',
  );
  assert.equal(probeRecordProblem(sampleProbeRecord({ adapter: { vendor: 'nvidia' } })), 'adapter');
  const device = sampleProbeRecord().device;
  assert.equal(
    probeRecordProblem(sampleProbeRecord({ device: { ...device, fp64Requested: 'double' } })),
    'device',
  );
  const variant = sampleProbeRecord().variants[0];
  assert.equal(
    probeRecordProblem(sampleProbeRecord({ variants: [{ ...variant, outcome: 'hung' }] })),
    'variants',
  );
  assert.equal(
    probeRecordProblem(
      sampleProbeRecord({ variants: [{ ...variant, override: { refinementIterations: -1 } }] }),
    ),
    'variants',
  );
  const bisection = sampleProbeRecord().bisection;
  assert.equal(
    probeRecordProblem(
      sampleProbeRecord({
        bisection: {
          ...bisection,
          hangingTracks: [{ index: 200, pixels: 'x', expectedStatus: 0 }],
        },
      }),
    ),
    'bisection',
  );
  assert.equal(probeRecordProblem(sampleProbeRecord({ aborted: 'x'.repeat(2001) })), 'aborted');
  assert.equal(probeRecordProblem(sampleProbeRecord({ resume: '?tracks=all' })), 'resume');
  // `reloads` counts the page reloads after device losses; a record from before it has none.
  assert.equal(probeRecordProblem(sampleProbeRecord({ reloads: 9 })), undefined);
  assert.equal(probeRecordProblem(sampleProbeRecord({ reloads: -1 })), 'reloads');
  assert.equal(probeRecordProblem(sampleProbeRecord({ reloads: 65 })), 'reloads');
  const steps = sampleProbeRecord().steps;
  assert.equal(
    probeRecordProblem(sampleProbeRecord({ steps: Array.from({ length: 65 }, () => steps[2]) })),
    'steps',
  );
  assert.equal(
    probeRecordProblem(
      sampleProbeRecord({ steps: [{ ...steps[0], loss: { ...steps[0].loss, message: 7 } }] }),
    ),
    'steps',
  );
  // A probe record is not a run record, and a run record is not a probe record.
  assert.equal(runRecordProblem(sampleProbeRecord()), 'format');
  assert.equal(probeRecordProblem(sampleRecord()), 'format');
});

test('POST stores a probe record with probe metadata and rejects a malformed one', async () => {
  const RECORDS = fakeKv(),
    env = { RECORDS, RECORDS_READ_TOKEN: 'secret' };
  const created = await handle(post(sampleProbeRecord()), env);
  assert.equal(created.status, 201);
  const { id, receivedAt } = await created.json();
  const entry = RECORDS.store.get('record/' + id);
  assert.equal(JSON.parse(entry.value).server.id, id);
  assert.equal(JSON.parse(entry.value).bisection.hangingTracks[0].index, 200);
  assert.deepEqual(entry.metadata, {
    receivedAt,
    kind: 'probe',
    gpu: 'blackwell',
    fp64Flavor: 'float',
    boundedSubmissions: false,
    variants: ['lost'],
    steps: 3,
    hangingTracks: 1,
    hanging: [200],
    aborted: false,
  });
  const abortedRecord = sampleProbeRecord({
    adapter: null,
    bisection: null,
    aborted: 'No WebGPU device opened at step 2: Error: no adapter',
    resume: '?tracks=0-1671',
  });
  const aborted = await handle(post(abortedRecord), env);
  assert.equal(aborted.status, 201);
  const abortedMeta = RECORDS.store.get('record/' + (await aborted.json()).id).metadata;
  assert.equal(abortedMeta.gpu, '');
  assert.equal(abortedMeta.hangingTracks, 0);
  assert.deepEqual(abortedMeta.hanging, []);
  assert.equal(abortedMeta.aborted, true);
  const listed = await (
    await handle(
      new Request('https://cv.typeshade.dev/api/records?limit=10', {
        headers: { authorization: 'Bearer secret' },
      }),
      env,
    )
  ).json();
  assert.deepEqual(
    listed.records.map((record) => record.kind),
    ['probe', 'probe'],
  );
  const malformed = await handle(post(sampleProbeRecord({ steps: null })), env);
  assert.equal(malformed.status, 400);
  assert.deepEqual(await malformed.json(), { error: 'invalid record', field: 'steps' });
  // The run records keep their own validation and metadata.
  const run = await handle(post(sampleRecord()), env);
  assert.equal(run.status, 201);
  const runMeta = RECORDS.store.get('record/' + (await run.json()).id).metadata;
  assert.equal(runMeta.kind, undefined);
  assert.equal(runMeta.registered, 8);
});
