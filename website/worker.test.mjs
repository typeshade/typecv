import test from 'node:test';
import assert from 'node:assert/strict';
import { handle } from './worker.mjs';
import { RUN_RECORD_FORMAT, runRecordProblem, summarizeStages } from './src/lib/run-record.ts';

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
  assert.deepEqual(
    summarizeStages([
      { operation: 'runTotal', elapsedMs: 10 },
      { operation: 'pnp', elapsedMs: 2, dispatches: 3, submissions: 1 },
      { operation: 'pnp', elapsedMs: 4, dispatches: 5, submissions: 2 },
    ]),
    { pnp: { calls: 2, elapsedMs: 6, dispatches: 8, submissions: 3 } },
  );
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
