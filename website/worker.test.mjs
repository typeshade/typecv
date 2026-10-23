import test from 'node:test';
import assert from 'node:assert/strict';
import { handle, parseRange } from './worker.mjs';
import {
  RESULT_FILE_MAX_BYTES,
  RUN_RECORD_FORMAT,
  RUN_RECORD_MAX_GPU_NAME,
  RUN_RECORD_MAX_GPU_ROWS,
  gpuRecord,
  reportGpuMs,
  chunkLines,
  chunkRecord,
  denseLine,
  denseRecord,
  sumDenseRecords,
  runRecordProblem,
  summarizeStages,
} from './src/lib/run-record.ts';
import { PROBE_RECORD_FORMAT, probeRecordProblem } from './src/lib/probe-record.ts';
import { planFiles, postRecord } from './src/lib/send-record.ts';
import { olderRecord, sampleRecord } from './record-fixture.mjs';
import { initializationFailureRecord } from './src/lib/initialization-failure.ts';

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
  // Where a video sample's frames came from (docs/lab-run-records.md).
  const withSource = (videoSource) =>
    runRecordProblem(sampleRecord({ options: { ...sampleRecord().options, videoSource } }));
  for (const ok of ['source', 'frames', 'mixed']) assert.equal(withSource(ok), undefined, ok);
  assert.equal(withSource('url'), 'options');
  assert.equal(withSource(1), 'options');
  // The step 6 switches of a KLT run (docs/klt-tracking.md, "Step 6 record").
  const withSetup = (kltSetup) =>
    runRecordProblem(sampleRecord({ options: { ...sampleRecord().options, klt: true, kltSetup } }));
  assert.equal(
    withSetup({ capacity: 512, fillIn: true, matching: 'tracks', gate: true }),
    undefined,
  );
  assert.equal(withSetup({ capacity: 0 }), 'options');
  assert.equal(withSetup({ capacity: 9000 }), 'options');
  assert.equal(withSetup({ fillIn: false }), 'options');
  assert.equal(withSetup({ matching: 'both' }), 'options');
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
  assert.equal(RUN_RECORD_FORMAT, 'typecv-run-record-v6');
  // v4 to v6 carry the chunk rates and the budget; null budget means no device was opened.
  const timing = sampleRecord().timing;
  assert.equal(runRecordProblem(sampleRecord({ timing: { ...timing, chunks: {} } })), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ timing: { ...timing, budgetMs: null } })),
    undefined,
  );
  const {
    chunks: _chunks,
    budgetMs: _budget,
    mainThread: _main,
    ...v3Timing
  } = olderRecord('typecv-run-record-v3').timing;
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
    { hiddenMs: -1 },
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
  assert.equal(runRecordProblem(olderRecord('typecv-run-record-v4')), undefined);
  // A v3 record from a page loaded before v4 still validates, and carries no chunk fields.
  const v3 = olderRecord('typecv-run-record-v3', { timing: v3Timing });
  assert.equal(runRecordProblem(v3), undefined);
  assert.equal(
    runRecordProblem(olderRecord('typecv-run-record-v3')),
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

// Verifies: docs/dense-mvs.md 3.12 (the cleanup of the fused cloud in the dense block).
test('the dense block records the cleanup option and the counts of each stage', () => {
  const report = {
    tier: 'webgpu',
    before: 1000,
    afterOutliers: 980,
    afterVoxel: 400,
    after: 380,
    outlierThreshold: 0.25,
    voxelSize: 0.02,
    eps: 0.08,
    clusters: 3,
    keptClusters: 1,
    ms: { outliers: 12.34, voxel: 5, cluster: 7 },
  };
  const dense = denseRecord({
    ...denseResult,
    options: { ...denseResult.options, denseClean: true },
    clean: report,
    cleanError: '',
  });
  assert.equal(dense.options.denseClean, true);
  assert.deepEqual(dense.clean, {
    tier: 'webgpu',
    before: 1000,
    afterOutliers: 980,
    afterVoxel: 400,
    after: 380,
    outlierThreshold: 0.25,
    voxelSize: 0.02,
    eps: 0.08,
    clusters: 3,
    keptClusters: 1,
    outliersMs: 12.3,
    voxelMs: 5,
    clusterMs: 7,
    error: '',
  });
  assert.equal(runRecordProblem(sampleRecord({ dense })), undefined);
  assert.match(denseLine(dense), /cleaned to 380 points/);
  // Off: no block. A failed cleanup records its error.
  assert.equal(denseRecord({ ...denseResult, clean: null, cleanError: '' }).clean, undefined);
  const failed = denseRecord({ ...denseResult, clean: null, cleanError: 'device lost' });
  assert.equal(failed.clean.error, 'device lost');
  assert.equal(runRecordProblem(sampleRecord({ dense: failed })), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ dense: { ...dense, clean: { ...dense.clean, after: -1 } } })),
    'dense',
  );
  // A record from before the cleanup stays valid.
  const { clean, ...older } = dense;
  const { denseClean, ...options } = older.options;
  assert.equal(runRecordProblem(sampleRecord({ dense: { ...older, options } })), undefined);
});

// Verifies: docs/dense-mvs.md 3.13 (the mesh in the dense block, no format bump).
test('the dense block records the mesh option and the counts and times of the mesh', () => {
  const report = {
    tier: 'webgpu',
    points: 380,
    voxelSize: 0.05,
    radius: 0.125,
    nodes: 5000,
    cubes: 1200,
    vertices: 1400,
    faces: 2500,
    ms: { lattice: 10.04, field: 20, cubes: 30.26, vertices: 4 },
  };
  const dense = denseRecord({
    ...denseResult,
    options: { ...denseResult.options, denseMesh: true },
    mesh: report,
    meshError: '',
  });
  assert.equal(dense.options.denseMesh, true);
  assert.deepEqual(dense.mesh, {
    tier: 'webgpu',
    points: 380,
    voxelSize: 0.05,
    radius: 0.125,
    nodes: 5000,
    cubes: 1200,
    vertices: 1400,
    faces: 2500,
    latticeMs: 10,
    fieldMs: 20,
    cubesMs: 30.3,
    verticesMs: 4,
    error: '',
  });
  assert.equal(runRecordProblem(sampleRecord({ dense })), undefined);
  assert.match(denseLine(dense), /mesh of 2500 faces/);
  // Off: no block. A failed mesh records its error.
  assert.equal(denseRecord({ ...denseResult, mesh: null, meshError: '' }).mesh, undefined);
  const failed = denseRecord({ ...denseResult, mesh: null, meshError: 'device lost' });
  assert.equal(failed.mesh.error, 'device lost');
  assert.equal(runRecordProblem(sampleRecord({ dense: failed })), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ dense: { ...dense, mesh: { ...dense.mesh, faces: -1 } } })),
    'dense',
  );
  // A record from before the mesh stays valid, and is still format v5.
  const { mesh, ...older } = dense;
  const { denseMesh, ...options } = older.options;
  assert.equal(runRecordProblem(sampleRecord({ dense: { ...older, options } })), undefined);
});

// Verifies: docs/dense-mvs.md step 6 (5.3, the run record v5 and its dense block).
test('dense quality settings and completed depths survive records and component aggregation', () => {
  const dense = denseRecord({
    ...denseResult,
    planarFilled: 36,
    options: {
      ...denseResult.options,
      fusion: 'tsdf',
      planarDepth: true,
      tsdf: { visibility: 'view-weighted' },
    },
  });
  assert.equal(dense.options.planarDepth, true);
  assert.equal(dense.options.tsdfVisibility, 'view-weighted');
  assert.equal(dense.planarFilled, 36);
  assert.equal(sumDenseRecords([dense, dense]).planarFilled, 72);
  assert.equal(runRecordProblem(sampleRecord({ dense })), undefined);
  assert.ok(runRecordProblem(sampleRecord({ dense: { ...dense, planarFilled: -1 } })));
  assert.ok(
    runRecordProblem(
      sampleRecord({
        dense: { ...dense, options: { ...dense.options, tsdfVisibility: 'unknown' } },
      }),
    ),
  );
});

test('the v5 record carries the dense block, which v3 and v4 cannot carry', () => {
  const dense = denseRecord(denseResult);
  assert.deepEqual(dense, {
    status: 'completed',
    skipped: '',
    options: { ...denseResult.options, denseClean: false },
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
  for (const format of ['typecv-run-record-v4', 'typecv-run-record-v3']) {
    const older = olderRecord(format);
    const { chunks: _c, budgetMs: _b, mainThread: _m, ...v3Timing } = older.timing;
    assert.equal(
      runRecordProblem({
        ...older,
        dense,
        ...(format === 'typecv-run-record-v3' ? { timing: v3Timing } : {}),
      }),
      'dense',
      format,
    );
  }
  assert.equal(runRecordProblem(olderRecord('typecv-run-record-v5', { dense })), undefined);
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
  assert.equal(RECORDS.store.get('record/' + id).metadata.gpuMs, 866);
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
  } = olderRecord('typecv-run-record-v3').timing;
  const old = await handle(post(olderRecord('typecv-run-record-v3', { timing: v3Timing })), env);
  assert.equal(old.status, 201);
  const oldMeta = RECORDS.store.get('record/' + (await old.json()).id).metadata;
  assert.equal(oldMeta.budgetMs, null);
  assert.equal(oldMeta.longestChunkMs, null);
  assert.equal(oldMeta.longestTaskMs, null);
  assert.equal(oldMeta.longestGapMs, null);
  assert.equal(oldMeta.gpuMs, null);
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

/** An in-memory R2 bucket with the subset of the binding the Worker uses for result files. */
function fakeR2() {
  const store = new Map();
  return {
    store,
    async head(key) {
      return store.has(key) ? { key } : null;
    },
    async put(key, value) {
      store.set(key, new Uint8Array(value));
    },
    async get(key) {
      return store.has(key) ? { body: store.get(key) } : null;
    },
    async list({ prefix }) {
      return {
        objects: [...store.keys()]
          .filter((key) => key.startsWith(prefix))
          .map((key) => ({ key, size: store.get(key).byteLength })),
      };
    },
  };
}

const ORIGIN = 'https://cv.typeshade.dev';
const bytesOf = (text) => new TextEncoder().encode(text);
const put = (id, name, body, headers = {}) =>
  new Request(`${ORIGIN}/api/records/${id}/files/${name}`, {
    method: 'PUT',
    headers: { origin: ORIGIN, 'cf-connecting-ip': '203.0.113.9', ...headers },
    body,
  });
const get = (path, token = 'secret') =>
  new Request(ORIGIN + path, token ? { headers: { authorization: 'Bearer ' + token } } : {});

/** Post a record that declares the given files and return its id. */
async function postWithFiles(env, files, headers = {}) {
  const plan = await planFiles(files);
  const response = await handle(post(sampleRecord({ files: plan.entries }), headers), env);
  assert.equal(response.status, 201);
  return { id: (await response.json()).id, plan };
}

test('the run record validator takes an optional files list and names a broken one', async () => {
  const plan = await planFiles([{ name: 'typecv-reconstruction.json', data: '{"a":1}' }]);
  assert.equal(runRecordProblem(sampleRecord({ files: plan.entries })), undefined);
  assert.equal(runRecordProblem(sampleRecord()), undefined);
  const [entry] = plan.entries;
  for (const bad of [
    'x',
    [{ ...entry, name: 'evil.txt' }],
    [{ ...entry, sha256: 'ABC' }],
    [{ ...entry, bytes: -1 }],
    [{ ...entry, skipped: '' }],
    [entry, entry],
  ])
    assert.equal(runRecordProblem(sampleRecord({ files: bad })), 'files');
});

test('the upload plan declares size and SHA-256 and skips a file over the limit', async () => {
  const plan = await planFiles([
    { name: 'typecv-reconstruction.json', data: 'abc' },
    { name: 'typecv-dense.ply', data: new Uint8Array(RESULT_FILE_MAX_BYTES + 1) },
  ]);
  assert.equal(plan.entries[0].bytes, 3);
  assert.equal(
    plan.entries[0].sha256,
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  );
  assert.equal(plan.entries[0].skipped, undefined);
  assert.match(plan.entries[1].skipped, /larger than 50 MB/);
  assert.deepEqual(
    plan.uploads.map((upload) => upload.name),
    ['typecv-reconstruction.json'],
  );
});

test('PUT stores a declared file; the token reads the list and the bytes', async () => {
  const env = {
    RECORDS: fakeKv(),
    RECORD_FILES: fakeR2(),
    RECORDS_READ_TOKEN: 'secret',
  };
  const { id } = await postWithFiles(env, [
    { name: 'typecv-reconstruction.json', data: '{"a":1}' },
    { name: 'typecv-mesh.ply', data: 'mesh' },
  ]);
  const stored = await handle(put(id, 'typecv-reconstruction.json', bytesOf('{"a":1}')), env);
  assert.equal(stored.status, 201);
  assert.ok(env.RECORD_FILES.store.has(`records/${id}/typecv-reconstruction.json`));
  assert.equal((await handle(get(`/api/records/${id}/files/`, null), env)).status, 401);
  assert.equal((await handle(get(`/api/records/${id}/files/`, 'wrong'), env)).status, 401);
  const list = await (await handle(get(`/api/records/${id}/files/`), env)).json();
  assert.deepEqual(list.files, [{ name: 'typecv-reconstruction.json', bytes: 7 }]);
  const file = await handle(get(`/api/records/${id}/files/typecv-reconstruction.json`), env);
  assert.equal(await file.text(), '{"a":1}');
  assert.equal(
    (await handle(get(`/api/records/${id}/files/typecv-reconstruction.json`, null), env)).status,
    401,
  );
  assert.equal((await handle(get(`/api/records/${id}/files/typecv-mesh.ply`), env)).status, 404);
  assert.equal((await handle(get(`/api/records/${id}/files/other.txt`), env)).status, 404);
});

test('PUT refuses what the record did not declare', async () => {
  const env = { RECORDS: fakeKv(), RECORD_FILES: fakeR2() };
  const { id } = await postWithFiles(env, [{ name: 'typecv-reconstruction.json', data: 'abc' }]);
  const name = 'typecv-reconstruction.json';
  // A name outside the four.
  assert.equal((await handle(put(id, 'evil.html', bytesOf('abc')), env)).status, 400);
  // A name the record did not declare.
  assert.equal((await handle(put(id, 'typecv-mesh.ply', bytesOf('abc')), env)).status, 400);
  // Foreign origin.
  assert.equal(
    (await handle(put(id, name, bytesOf('abc'), { origin: 'https://evil.example' }), env)).status,
    403,
  );
  // Bytes that differ from the declared SHA-256 and size.
  assert.equal((await handle(put(id, name, bytesOf('abd')), env)).status, 400);
  assert.equal((await handle(put(id, name, bytesOf('abcd')), env)).status, 400);
  // Declared size over the limit.
  assert.equal(
    (
      await handle(
        put(id, name, bytesOf('abc'), { 'content-length': String(RESULT_FILE_MAX_BYTES + 1) }),
        env,
      )
    ).status,
    413,
  );
  // A record that does not exist, and a malformed id.
  assert.equal(
    (await handle(put('20260101000000000-deadbeef', name, bytesOf('abc')), env)).status,
    404,
  );
  assert.equal((await handle(put('../x', name, bytesOf('abc')), env)).status, 404);
  assert.equal(env.RECORD_FILES.store.size, 0);
  // The good file is stored once; a second PUT does not replace it.
  assert.equal((await handle(put(id, name, bytesOf('abc')), env)).status, 201);
  assert.equal((await handle(put(id, name, bytesOf('abc')), env)).status, 409);
  // GET on a record route with another method.
  assert.equal(
    (
      await handle(
        new Request(`${ORIGIN}/api/records/${id}/files/${name}`, { method: 'DELETE' }),
        env,
      )
    ).status,
    405,
  );
});

test('PUT refuses a file of a skipped entry and of a stale record', async () => {
  const env = { RECORDS: fakeKv(), RECORD_FILES: fakeR2() };
  const plan = await planFiles([{ name: 'typecv-dense.ply', data: 'abc' }]);
  const skipped = [{ ...plan.entries[0], skipped: 'too big' }];
  const post1 = await handle(post(sampleRecord({ files: skipped })), env);
  const skippedId = (await post1.json()).id;
  assert.equal((await handle(put(skippedId, 'typecv-dense.ply', bytesOf('abc')), env)).status, 400);
  const { id } = await postWithFiles(env, [{ name: 'typecv-dense.ply', data: 'abc' }]);
  const key = 'record/' + id,
    entry = env.RECORDS.store.get(key),
    record = JSON.parse(entry.value);
  record.server.receivedAt = new Date(Date.now() - 11 * 60_000).toISOString();
  env.RECORDS.store.set(key, { ...entry, value: JSON.stringify(record) });
  const stale = await handle(put(id, 'typecv-dense.ply', bytesOf('abc')), env);
  assert.equal(stale.status, 403);
  assert.equal((await stale.json()).error, 'record too old');
  assert.equal(env.RECORD_FILES.store.size, 0);
});

test('PUT is rate limited apart from the record POST', async () => {
  const env = { RECORDS: fakeKv(), RECORD_FILES: fakeR2() };
  const { id } = await postWithFiles(env, [{ name: 'typecv-dense.ply', data: 'abc' }]);
  let last;
  for (let i = 0; i < 121; i++)
    last = await handle(put(id, 'typecv-dense.ply', bytesOf('abc')), env);
  assert.equal(last.status, 429);
  assert.equal(
    (await handle(post(sampleRecord()), env)).status,
    201,
    'the record POST keeps its own count',
  );
});

test('without the file store the routes answer 503 and the config says so', async () => {
  const env = { RECORDS: fakeKv(), RECORDS_READ_TOKEN: 'secret' };
  assert.deepEqual(await (await handle(get('/api/config', null), env)).json(), { files: false });
  const { id } = await postWithFiles(env, [{ name: 'typecv-dense.ply', data: 'abc' }]);
  assert.equal((await handle(put(id, 'typecv-dense.ply', bytesOf('abc')), env)).status, 503);
  assert.equal((await handle(get(`/api/records/${id}/files/`), env)).status, 503);
  const withStore = { ...env, RECORD_FILES: fakeR2() };
  assert.deepEqual(await (await handle(get('/api/config/', null), withStore)).json(), {
    files: true,
  });
});

test('the client uploads the planned files and reports a refusal per file', async () => {
  const env = { RECORDS: fakeKv(), RECORD_FILES: fakeR2() };
  const files = [
    { name: 'typecv-reconstruction.json', data: '{"a":1}' },
    { name: 'typecv-mesh.ply', data: new Uint8Array([1, 2, 3]) },
    { name: 'typecv-dense.ply', data: new Uint8Array(RESULT_FILE_MAX_BYTES + 1) },
  ];
  const plan = await planFiles(files);
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (input, init = {}) => {
    const request = new Request(new URL(input, ORIGIN), {
      ...init,
      headers: { origin: ORIGIN, ...(init.headers ?? {}) },
    });
    // The second upload is refused by the server.
    if (request.method === 'PUT' && ++calls === 2)
      return new Response('{"error":"x"}', { status: 429 });
    return handle(request, env);
  };
  const progress = [];
  try {
    const state = await postRecord(sampleRecord(), false, plan, (done, total) =>
      progress.push([done, total]),
    );
    assert.equal(state.status, 'sent');
    assert.deepEqual(state.files.stored, ['typecv-reconstruction.json']);
    assert.deepEqual(
      state.files.failed.map((f) => f.name),
      ['typecv-dense.ply', 'typecv-mesh.ply'],
    );
    assert.match(state.files.failed[1].reason, /HTTP 429/);
    assert.deepEqual(progress, [
      [0, 2],
      [1, 2],
      [2, 2],
    ]);
    const stored = JSON.parse(env.RECORDS.store.get('record/' + state.id).value);
    assert.deepEqual(
      stored.files.map((f) => f.name),
      ['typecv-reconstruction.json', 'typecv-mesh.ply', 'typecv-dense.ply'],
    );
    // Without a plan the record carries no files and nothing is uploaded.
    const plain = await postRecord(sampleRecord(), false);
    assert.equal(plain.status, 'sent');
    assert.equal(plain.files, undefined);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('the validator accepts the multi-video fields of docs/multi-video.md 6.4 and names the broken one', () => {
  const base = sampleRecord();
  const video = {
    videoId: '0123456789abcdef',
    label: null,
    durationSeconds: 12.5,
    keyframes: 8,
    registeredViews: 7,
    focalRatio: 0.8,
    component: 0,
    tags: ['component'],
  };
  const multi = {
    input: {
      ...base.input,
      videos: [video, { ...video, videoId: 'fedcba9876543210', tags: ['alone'] }],
    },
    options: {
      ...base.options,
      multiVideo: {
        descriptor: 'vlad-32-v1',
        k: 10,
        minMatches: 30,
        minEdgeInliers: 25,
        maxKeyframesPerVideo: 60,
        growth: 'incremental',
        canonical: true,
        focalPerVideo: false,
      },
    },
    result: {
      ...base.result,
      components: [
        {
          keyframes: 16,
          registeredViews: 15,
          landmarks: 900,
          videos: 2,
          meanReprojectionErrorPx: 0.6,
        },
      ],
      crossVideoEdges: { edges: 12, inliers: 640 },
      viewGraph: {
        nodes: 16,
        candidatePairs: 80,
        matchedPairs: 50,
        verifiedEdges: 40,
        exhaustivePairs: 120,
      },
      notConnected: ['0123456789abcdef'],
    },
  };
  assert.equal(runRecordProblem(sampleRecord(multi)), undefined);
  // A record without the fields stays valid.
  assert.equal(runRecordProblem(sampleRecord()), undefined);
  const withInput = (videos) => sampleRecord({ ...multi, input: { ...multi.input, videos } });
  assert.equal(runRecordProblem(withInput([{ ...video, videoId: 'photo.mp4' }])), 'input');
  assert.equal(runRecordProblem(withInput([{ ...video, tags: ['mine'] }])), 'input');
  assert.equal(runRecordProblem(withInput([{ ...video, registeredViews: 9 }])), 'input');
  assert.equal(runRecordProblem(withInput(Array(65).fill(video))), 'input');
  const withOptions = (patch) =>
    sampleRecord({
      ...multi,
      options: { ...multi.options, multiVideo: { ...multi.options.multiVideo, ...patch } },
    });
  assert.equal(runRecordProblem(withOptions({ growth: 'merge' })), 'options');
  assert.equal(runRecordProblem(withOptions({ k: -1 })), 'options');
  const withResult = (patch) => sampleRecord({ ...multi, result: { ...multi.result, ...patch } });
  assert.equal(runRecordProblem(withResult({ notConnected: ['name.mp4'] })), 'result');
  assert.equal(
    runRecordProblem(withResult({ crossVideoEdges: { edges: 1.5, inliers: 2 } })),
    'result',
  );
  assert.equal(runRecordProblem(withResult({ components: [{ keyframes: 1 }] })), 'result');
  assert.equal(runRecordProblem(withResult({ viewGraph: { nodes: 1 } })), 'result');
  // The stage names of 6.4 fit the existing stage table.
  const stages = Object.fromEntries(
    ['globalDescriptor', 'retrieval', 'pairMatching', 'pairVerification', 'components'].map((n) => [
      n,
      { calls: 1, elapsedMs: 5, dispatches: 0, submissions: 0 },
    ]),
  );
  assert.equal(
    runRecordProblem(sampleRecord({ ...multi, timing: { ...base.timing, stages } })),
    undefined,
  );
});

test('the validator accepts the focal search of docs/auto-focal.md and names the broken one', () => {
  const base = sampleRecord();
  const searched = sampleRecord({
    options: { ...base.options, focalSearch: true },
    result: { ...base.result, focalSearch: { ratio: 0.78, gridRatio: 0.85, views: 5 } },
  });
  assert.equal(runRecordProblem(searched), undefined);
  assert.equal(
    runRecordProblem(sampleRecord({ options: { ...base.options, focalSearch: 'yes' } })),
    'options',
  );
  for (const focalSearch of [null, { ratio: 0, gridRatio: 0.8, views: 5 }, { ratio: 0.8 }])
    assert.equal(
      runRecordProblem(sampleRecord({ result: { ...base.result, focalSearch } })),
      'result',
    );
});

test('initialization evidence survives serialization and is bounded and validated by the collector', async () => {
  const pair = {
    first: 1,
    second: 2,
    matches: 80,
    inliers: 75,
    homographyInliers: 78,
    medianAngleDeg: 0.02,
    seedEligible: false,
    reason: 'Homography-dominated initialization',
    positiveDepthCount: 2,
  };
  const attempt = {
    ratio: 0.7,
    views: 5,
    pairs: 1,
    eligiblePairs: 0,
    bestPair: pair,
    reasons: [{ reason: pair.reason, pairs: 1 }],
    failure: 'ReconstructionInitializationError: No stable initial pair',
  };
  const error = Object.assign(new Error('No focal of the search built an initial map'), {
    name: 'ReconstructionInitializationError',
    pairVerification: Array.from({ length: 40 }, (_, i) => ({ ...pair, second: i + 2 })),
    focalSearch: { attempts: [attempt], omittedAttempts: 0 },
  });
  const initialization = initializationFailureRecord(error);
  assert.equal(initialization.pairs.length, 32);
  assert.equal(initialization.omittedPairs, 8);
  assert.deepEqual(initialization.focalSearch.attempts[0], attempt);
  assert.equal(initializationFailureRecord(new Error('Device lost')), undefined);
  const extended = { ...pair, pixels: [1, 2], descriptors: [3, 4] };
  const clean = initializationFailureRecord({
    ...error,
    pairVerification: [extended],
    focalSearch: {
      attempts: [{ ...attempt, bestPair: extended, pixels: [5, 6] }],
      omittedAttempts: 0,
    },
  });
  assert.equal(clean.pairs[0].pixels, undefined);
  assert.equal(clean.pairs[0].descriptors, undefined);
  assert.equal(clean.focalSearch.attempts[0].pixels, undefined);
  assert.equal(clean.focalSearch.attempts[0].bestPair.pixels, undefined);
  const failure = {
    message: error.message,
    operation: 'initializing 3D map',
    kernel: '',
    reason: '',
    framesProcessed: 5,
    initialization,
  };
  const record = JSON.parse(JSON.stringify(sampleRecord({ failure })));
  assert.equal(runRecordProblem(record), undefined);
  const kv = fakeKv();
  const response = await handle(post(record), { RECORDS: kv });
  assert.equal(response.status, 201);
  const { id } = await response.json();
  const stored = JSON.parse(kv.store.get('record/' + id).value);
  assert.deepEqual(stored.failure.initialization, initialization);
  for (const bad of [
    { ...initialization, pairs: Array(33).fill(pair) },
    { ...initialization, omittedPairs: -1 },
    { ...initialization, pairs: [{ ...pair, inliers: 81 }] },
    { ...initialization, pairs: [{ ...pair, positiveDepthCount: 81 }] },
    { ...initialization, focalSearch: { attempts: Array(65).fill(attempt), omittedAttempts: 0 } },
    {
      ...initialization,
      focalSearch: { attempts: [{ ...attempt, ratio: null }], omittedAttempts: 0 },
    },
    {
      ...initialization,
      focalSearch: { attempts: [{ ...attempt, eligiblePairs: 2 }], omittedAttempts: 0 },
    },
    {
      ...initialization,
      focalSearch: {
        attempts: [{ ...attempt, reasons: [{ reason: 'x', pairs: 2 }] }],
        omittedAttempts: 0,
      },
    },
    {
      ...initialization,
      focalSearch: { attempts: [{ ...attempt, failure: 'x'.repeat(257) }], omittedAttempts: 0 },
    },
  ]) {
    const malformed = sampleRecord({ failure: { ...failure, initialization: bad } });
    assert.equal(runRecordProblem(malformed), 'failure');
    assert.equal((await handle(post(malformed), { RECORDS: fakeKv() })).status, 400);
  }
});

/** An in-memory R2 bucket: `head`, and `get` with an `{ offset, length }` range. */
function rangeBucket(objects) {
  const meta = (key) => ({
    size: objects[key].byteLength,
    httpEtag: '"e1"',
    writeHttpMetadata(headers) {
      headers.set('content-type', 'video/mp4');
    },
  });
  return {
    async head(key) {
      return key in objects ? meta(key) : null;
    },
    async get(key, options) {
      if (!(key in objects)) return null;
      const bytes = objects[key];
      const { offset, length } = options?.range ?? { offset: 0, length: bytes.byteLength };
      return { ...meta(key), body: bytes.slice(offset, offset + length) };
    },
  };
}

const VIDEO = new Uint8Array(Array.from({ length: 100 }, (_, i) => i));
const rangeEnv = () => ({
  SAMPLES: rangeBucket({ 'sample/canyon/source.mp4': VIDEO }),
  ASSETS: { fetch: async () => new Response('asset', { status: 200 }) },
});
const sampleGet = (range, method = 'GET') =>
  new Request('https://cv.example/sample/canyon/source.mp4', {
    method,
    headers: range ? { range } : {},
  });

test('a sample answers a byte range with 206 and Content-Range', async () => {
  const response = await handle(sampleGet('bytes=10-19'), rangeEnv());
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('content-range'), 'bytes 10-19/100');
  assert.equal(response.headers.get('content-length'), '10');
  assert.equal(response.headers.get('accept-ranges'), 'bytes');
  assert.equal(response.headers.get('content-type'), 'video/mp4');
  assert.deepEqual(
    [...new Uint8Array(await response.arrayBuffer())],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  );
});

test('an open-ended range, a suffix range and a range past the end are cut to the object', async () => {
  const open = await handle(sampleGet('bytes=95-'), rangeEnv());
  assert.equal(open.status, 206);
  assert.equal(open.headers.get('content-range'), 'bytes 95-99/100');
  assert.equal(open.headers.get('content-length'), '5');
  const suffix = await handle(sampleGet('bytes=-4'), rangeEnv());
  assert.equal(suffix.headers.get('content-range'), 'bytes 96-99/100');
  assert.deepEqual([...new Uint8Array(await suffix.arrayBuffer())], [96, 97, 98, 99]);
  const wide = await handle(sampleGet('bytes=90-5000'), rangeEnv());
  assert.equal(wide.headers.get('content-range'), 'bytes 90-99/100');
  const bigSuffix = await handle(sampleGet('bytes=-5000'), rangeEnv());
  assert.equal(bigSuffix.headers.get('content-range'), 'bytes 0-99/100');
});

test('a range outside the object answers 416 with the size', async () => {
  for (const range of ['bytes=100-', 'bytes=100-200', 'bytes=-0']) {
    const response = await handle(sampleGet(range), rangeEnv());
    assert.equal(response.status, 416, range);
    assert.equal(response.headers.get('content-range'), 'bytes */100');
    assert.equal(response.headers.get('accept-ranges'), 'bytes');
  }
});

test('a request without a usable Range gets the whole object with Accept-Ranges', async () => {
  const whole = await handle(sampleGet(), rangeEnv());
  assert.equal(whole.status, 200);
  assert.equal(whole.headers.get('accept-ranges'), 'bytes');
  assert.equal((await whole.arrayBuffer()).byteLength, 100);
  // Several ranges, an unknown unit and a reversed range are ignored, not refused.
  for (const range of ['bytes=0-1,5-6', 'items=0-1', 'bytes=9-3', 'bytes=-', 'junk']) {
    const response = await handle(sampleGet(range), rangeEnv());
    assert.equal(response.status, 200, range);
    assert.equal((await response.arrayBuffer()).byteLength, 100);
  }
});

test('HEAD answers the size and Accept-Ranges, and a missing key falls through to the assets', async () => {
  const head = await handle(sampleGet(undefined, 'HEAD'), rangeEnv());
  assert.equal(head.status, 200);
  assert.equal(head.headers.get('content-length'), '100');
  assert.equal(head.headers.get('accept-ranges'), 'bytes');
  assert.equal((await head.arrayBuffer()).byteLength, 0);
  const missing = new Request('https://cv.example/sample/none/source.mp4', {
    headers: { range: 'bytes=0-9' },
  });
  assert.equal(await (await handle(missing, rangeEnv())).text(), 'asset');
});

test('parseRange reads one range and refuses what it cannot satisfy', () => {
  assert.deepEqual(parseRange('bytes=0-0', 1), { offset: 0, length: 1 });
  assert.equal(parseRange('bytes=0-', 0), 'unsatisfiable');
  assert.equal(parseRange(null, 10), undefined);
});

test('a v6 record carries the GPU time per kernel, the features and the compute limits', async () => {
  const record = sampleRecord();
  assert.equal(runRecordProblem(record), undefined);
  // v5 stays valid without any of the v6 fields.
  assert.equal(runRecordProblem(olderRecord('typecv-run-record-v5')), undefined);
  // A v5 record with a v6 field is neither.
  assert.equal(runRecordProblem(olderRecord('typecv-run-record-v5', { gpu: record.gpu })), 'gpu');
  assert.equal(
    runRecordProblem(
      olderRecord('typecv-run-record-v5', {
        timing: { ...olderRecord('typecv-run-record-v5').timing, gpu: [] },
      }),
    ),
    'timing',
  );
  // A v6 record has all of them. A device without timestamp-query has rows and no time.
  const { gpu: _rows, ...noRows } = record.timing;
  assert.equal(runRecordProblem(sampleRecord({ timing: noRows })), 'timing');
  const untimed = {
    ...record.timing,
    gpu: [{ name: 'pairModels', groups: 3, dispatches: 3, timedGroups: 0, gpuMs: null }],
    gpuTimestamps: false,
  };
  assert.equal(runRecordProblem(sampleRecord({ timing: untimed })), undefined);
  const row = record.timing.gpu[0];
  const bad = (change) =>
    runRecordProblem(sampleRecord({ timing: { ...record.timing, gpu: [{ ...row, ...change }] } }));
  assert.equal(bad({ gpuMs: -1 }), 'timing', 'a negative gpuMs');
  assert.equal(bad({ gpuMs: Number.NaN }), 'timing');
  assert.equal(bad({ gpuMs: null }), 'timing', 'a time of null with timed groups');
  assert.equal(bad({ timedGroups: row.groups + 1 }), 'timing');
  assert.equal(bad({ name: '' }), 'timing');
  assert.equal(bad({ groups: 1.5 }), 'timing');
  assert.equal(
    runRecordProblem(
      sampleRecord({
        timing: {
          ...record.timing,
          gpu: Array.from({ length: RUN_RECORD_MAX_GPU_ROWS + 1 }, (_, i) => ({
            ...row,
            name: 'k' + i,
          })),
        },
      }),
    ),
    'timing',
    'more rows than the collector keeps',
  );
  const stage = record.timing.stages.pairVerification;
  assert.equal(
    runRecordProblem(
      sampleRecord({
        timing: { ...record.timing, stages: { pairVerification: { ...stage, gpuMs: -0.5 } } },
      }),
    ),
    'timing',
    'a negative stage gpuMs',
  );
  const noFeatures = { ...record.gpu, features: undefined };
  assert.equal(runRecordProblem(sampleRecord({ gpu: noFeatures })), 'gpu');
  assert.equal(
    runRecordProblem(
      sampleRecord({
        gpu: {
          ...record.gpu,
          limits: { ...record.gpu.limits, maxComputeWorkgroupStorageSize: -1 },
        },
      }),
    ),
    'gpu',
  );
});

test('gpuRecord and reportGpuMs turn the worker rows into record rows', () => {
  assert.equal(reportGpuMs({ groups: [{ calls: ['a'] }, { calls: ['b'] }] }), undefined);
  assert.equal(reportGpuMs({ groups: [{ gpuTime: 1500000 }, {}, { gpuTime: 500000 }] }), 2);
  assert.equal(reportGpuMs(undefined), undefined);
  const timed = gpuRecord(
    [
      { name: 'a+b', groups: 4, dispatches: 4, timedGroups: 4, gpuNs: 2_500_000 },
      { name: 'c', groups: 2, dispatches: 3, timedGroups: 0, gpuNs: 0 },
      { name: '', groups: 1, dispatches: 1, timedGroups: 0, gpuNs: 0 },
    ],
    5,
  );
  assert.deepEqual(timed, {
    gpu: [
      { name: 'a+b', groups: 4, dispatches: 4, timedGroups: 4, gpuMs: 2.5 },
      { name: 'c', groups: 2, dispatches: 3, timedGroups: 0, gpuMs: null },
    ],
    gpuTimestamps: true,
    gpuDropped: 5,
  });
  const untimed = gpuRecord([{ name: 'c', groups: 2, dispatches: 3, timedGroups: 0, gpuNs: 0 }]);
  assert.equal(untimed.gpuTimestamps, false);
  assert.equal(untimed.gpu[0].gpuMs, null);
  // Names cut to the record's limit that then collide are summed, and the table stays bounded.
  const long = 'k'.repeat(RUN_RECORD_MAX_GPU_NAME + 10);
  const merged = gpuRecord([
    { name: long + 'x', groups: 1, dispatches: 1, timedGroups: 1, gpuNs: 1000 },
    { name: long + 'y', groups: 1, dispatches: 1, timedGroups: 1, gpuNs: 2000 },
  ]);
  assert.equal(merged.gpu.length, 1);
  assert.equal(merged.gpu[0].groups, 2);
  const many = gpuRecord(
    Array.from({ length: RUN_RECORD_MAX_GPU_ROWS + 3 }, (_, i) => ({
      name: 'k' + i,
      groups: 1,
      dispatches: 1,
      timedGroups: 0,
      gpuNs: 0,
    })),
  );
  assert.equal(many.gpu.length, RUN_RECORD_MAX_GPU_ROWS);
  assert.equal(many.gpuDropped, 3);
  assert.equal(
    runRecordProblem(sampleRecord({ timing: { ...sampleRecord().timing, ...timed } })),
    undefined,
  );
});
