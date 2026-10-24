import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import vm from 'node:vm';
import { build } from 'esbuild';

// Execute the actual request queue and device-loss guards. Only the numerical runtime and the
// browser messaging surface are fake; no tests call prepareDenseDevice or finish directly.
const workerPath = new URL('./src/workers/sfm.worker.ts', import.meta.url).pathname;
const source = await readFile(workerPath, 'utf8');
const bundle = await build({
  stdin: {
    contents:
      source +
      `
export function __setRuntime(api: any) {
  apiPromise = Promise.resolve(api);
  adapter = {features:['timestamp-query'],limits:{maxBufferSize:128*1024**2}};
}
export async function __send(message: any) {
  self.onmessage!({data:message} as MessageEvent<Request>);
  await queue;
  await Promise.resolve();
}
export function __state() {
  return {device,deviceUnavailable,deviceTimestampRequest,jobId:job?.id};
}
export function __finalFields() {
  return job?.dense ? denseContext(job).finalFields!() : undefined;
}
export async function __close() {
  await releaseFeatures();
  job = undefined;
  const previous = device;
  device = undefined;
  await previous?.dispose();
}
`,
    loader: 'ts',
    sourcefile: workerPath,
    resolveDir: dirname(workerPath),
  },
  bundle: true,
  format: 'iife',
  platform: 'browser',
  globalName: 'WorkerLifecycleTest',
  write: false,
});

function harness({
  actualFeatures,
  failOpen,
  diagnosticsByAttempt,
  captureFinalFields = false,
} = {}) {
  const messages = [];
  const events = [];
  const devices = [];
  const opens = [];
  const repeatedFinalFields = [];
  let numericalCalls = 0;
  let api;
  const listeners = new Map();
  const worker = {
    location: { origin: 'https://typecv-worker.test' },
    postMessage(message) {
      messages.push(structuredClone(message));
      events.push({ type: 'post', message: message.type, id: message.id });
      if (captureFinalFields && message.type === 'denseResult')
        for (let read = 0; read < 2; read++)
          repeatedFinalFields.push(structuredClone(api.__finalFields()));
    },
    addEventListener(type, listener) {
      listeners.set(type, [...(listeners.get(type) ?? []), listener]);
    },
  };
  const context = vm.createContext({
    self: worker,
    navigator: {},
    performance,
    AbortController,
    setTimeout,
    clearTimeout,
    TextEncoder,
    TextDecoder,
    URL,
    Blob,
    ReadableStream,
    CompressionStream,
    structuredClone,
    console,
  });
  vm.runInContext(bundle.outputFiles[0].text, context, { filename: workerPath });
  const runtime = {
    ShadeDevice: {
      async open(options) {
        const attempt = opens.length;
        const request = options.gpuTimestamps ?? true;
        opens.push(options);
        events.push({ type: 'open', attempt, request });
        if (failOpen?.(attempt, options)) throw new Error('synthetic Dense GPU open failure');
        let lose;
        const features = actualFeatures ?? (request ? ['timestamp-query'] : []);
        const diagnostics = diagnosticsByAttempt?.(attempt) ?? { rows: [], dropped: 0, chunks: {} };
        const statistics = {
          liveBytes: 0,
          pooledBytes: 0,
          applicationUploadBytes: 0,
          applicationReadbackBytes: 0,
        };
        const device = {
          ...options,
          attempt,
          backend: 'webgpu',
          boundedSubmissions: options.boundedSubmissions === true,
          fp64Flavor: options.fp64Flavor === 'auto' ? 'integer' : options.fp64Flavor,
          timeBudgetMs: options.timeBudgetMs ?? 250,
          info: { webgpu: { features: [...features] } },
          statistics,
          get kernelStatistics() {
            if (this.disposeCount) throw new Error('disposed device kernel statistics');
            return { rows: diagnostics.rows, dropped: diagnostics.dropped };
          },
          get chunkRates() {
            if (this.disposeCount) throw new Error('disposed device chunk rates');
            return diagnostics.chunks;
          },
          disposeCount: 0,
          resetKernelStatistics() {},
          lost: new Promise((resolve) => {
            lose = resolve;
          }),
          lose() {
            lose({ reason: 'destroyed', message: 'late replaced-device loss' });
          },
          async dispose() {
            this.disposeCount++;
            events.push({ type: 'dispose', attempt });
          },
        };
        devices.push(device);
        return device;
      },
    },
    IncrementalSfmSession: class {
      constructor(device) {
        this.device = device;
      }
      async finish() {
        events.push({ type: 'finishSparse', attempt: this.device.attempt });
        return { reconstruction: { views: [], landmarks: [] }, diagnostics: {}, unregistered: [] };
      }
    },
    async reconstructDenseGpu() {
      numericalCalls++;
      throw new Error('Empty reconstruction must skip numerical Dense execution');
    },
  };
  api = context.WorkerLifecycleTest;
  api.__setRuntime(runtime);
  return {
    messages,
    events,
    devices,
    opens,
    repeatedFinalFields,
    numericalCalls: () => numericalCalls,
    send: (message) => api.__send(message),
    state: () => api.__state(),
    close: () => api.__close(),
    async lose(device) {
      device.lose();
      await Promise.resolve();
      await Promise.resolve();
    },
  };
}

const start = (id, extra = {}) => ({
  type: 'start',
  id,
  focalRatio: 0.8,
  analysisEdge: 640,
  firstOctave: 0,
  dense: true,
  ...extra,
});
const result = (worker, id, type) =>
  worker.messages.find((message) => message.id === id && message.type === type);
const kernelTotals = (message) => new Map(message.gpuKernels.map((row) => [row.name, row]));
const diagnosticProfiles = () => [
  {
    rows: [
      { name: 'sparseOnly', groups: 2, dispatches: 4, timedGroups: 2, gpuNs: 20 },
      { name: 'shared', groups: 3, dispatches: 6, timedGroups: 3, gpuNs: 30 },
    ],
    dropped: 4,
    chunks: { sparseOnly: { msPerUnit: 1 }, shared: { msPerUnit: 2 } },
  },
  {
    rows: [
      { name: 'shared', groups: 7, dispatches: 14, timedGroups: 0, gpuNs: 0 },
      { name: 'denseOnly', groups: 5, dispatches: 10, timedGroups: 0, gpuNs: 0 },
    ],
    dropped: 6,
    chunks: { denseOnly: { msPerUnit: 3 }, shared: { msPerUnit: 9 } },
  },
  {
    rows: [{ name: 'nextSparseOnly', groups: 11, dispatches: 22, timedGroups: 11, gpuNs: 110 }],
    dropped: 1,
    chunks: { nextSparseOnly: { msPerUnit: 4 } },
  },
];

test('worker saves Sparse before unbounded Dense replacement and restores Sparse query defaults', async () => {
  const worker = harness();
  try {
    await worker.send(start(1));
    assert.ok(result(worker, 1, 'ready'));
    assert.equal(worker.opens.length, 1);
    assert.equal(
      worker.opens[0].gpuTimestamps,
      undefined,
      'Sparse keeps the runtime default request',
    );
    await worker.send({ type: 'finish', id: 1 });
    const sparse = result(worker, 1, 'result');
    const dense = result(worker, 1, 'denseResult');
    assert.ok(sparse);
    assert.ok(dense);
    assert.equal(dense.failure, null);
    assert.equal(dense.gpuTimestamps, false);
    assert.equal(worker.opens.length, 2);
    assert.equal(worker.opens[1].gpuTimestamps, false);
    for (const field of ['workingBytes', 'poolBytes', 'boundedSubmissions'])
      assert.equal(worker.opens[1][field], worker.opens[0][field], field);
    assert.equal(worker.opens[1].fp64Flavor, worker.devices[0].fp64Flavor);
    assert.equal(worker.opens[1].timeBudgetMs, worker.devices[0].timeBudgetMs);
    const saved = worker.events.findIndex(
      (event) => event.type === 'post' && event.message === 'result',
    );
    const disposed = worker.events.findIndex((event) => event.type === 'dispose');
    const denseOpen = worker.events.findIndex(
      (event) => event.type === 'open' && event.attempt === 1,
    );
    assert.ok(saved >= 0 && saved < disposed && disposed < denseOpen);
    assert.deepEqual(
      worker.devices.map((device) => device.disposeCount),
      [1, 0],
    );
    assert.equal(worker.numericalCalls(), 0);

    await worker.send(start(2, { dense: false }));
    assert.equal(worker.opens.length, 3, 'a performance-mode Dense device is not reused by Sparse');
    assert.equal(worker.opens[2].gpuTimestamps, undefined);
    assert.equal(worker.state().deviceTimestampRequest, true);
    assert.deepEqual(
      worker.devices.map((device) => device.disposeCount),
      [1, 1, 0],
    );
    await worker.lose(worker.devices[0]);
    await worker.lose(worker.devices[1]);
    assert.equal(worker.state().device, worker.devices[2]);
    assert.equal(worker.state().jobId, 2);
    assert.equal(worker.state().deviceUnavailable, false);
    assert.equal(worker.devices[2].disposeCount, 0);
    assert.deepEqual(
      worker.messages.filter((message) => message.type === 'error'),
      [],
    );
    await worker.send({ type: 'finish', id: 2 });
    assert.ok(result(worker, 2, 'result'));
  } finally {
    await worker.close();
  }
});

test('worker retains timestamp profiling for bounded Dense and an explicit profiling override', async () => {
  for (const extra of [{ bounded: true }, { denseGpuTimestamps: true }]) {
    const worker = harness();
    try {
      await worker.send(start(1, extra));
      await worker.send({ type: 'finish', id: 1 });
      const dense = result(worker, 1, 'denseResult');
      assert.equal(dense.gpuTimestamps, true);
      assert.equal(worker.opens.length, 1, 'the compatible Sparse device is retained');
      assert.equal(worker.devices[0].disposeCount, 0);
      assert.equal(worker.state().deviceTimestampRequest, true);
      assert.equal(worker.numericalCalls(), 0);
    } finally {
      await worker.close();
    }
  }
});

test('worker reports enabled native features rather than the requested profiling mode', async () => {
  for (const extra of [{}, { bounded: true }, { denseGpuTimestamps: true }]) {
    const worker = harness({ actualFeatures: [] });
    try {
      await worker.send(start(1, extra));
      await worker.send({ type: 'finish', id: 1 });
      const dense = result(worker, 1, 'denseResult');
      assert.equal(
        dense.gpuTimestamps,
        false,
        'known disabled feature must not be reported as requested',
      );
      assert.equal(worker.opens.length, 1, 'an already query-free device needs no replacement');
      assert.equal(worker.events.find((event) => event.type === 'open').request, true);
      assert.equal(worker.devices[0].disposeCount, 0);
      assert.deepEqual(
        worker.messages.filter((message) => message.type === 'error'),
        [],
      );
    } finally {
      await worker.close();
    }
  }
});

test('Dense device-open failure preserves Sparse and omits an unknown GPU timestamp mode', async () => {
  const profiles = diagnosticProfiles();
  const worker = harness({
    failOpen: (attempt) => attempt === 1,
    diagnosticsByAttempt: (attempt) => profiles[attempt],
    captureFinalFields: true,
  });
  try {
    await worker.send(start(1));
    await worker.send({ type: 'finish', id: 1 });
    const sparse = result(worker, 1, 'result');
    const dense = result(worker, 1, 'denseResult');
    assert.ok(sparse, 'Sparse result was posted before opening the Dense device');
    assert.ok(dense);
    assert.match(
      dense.failure.message,
      /Could not open the dense GPU device:.*synthetic Dense GPU open failure/s,
    );
    assert.equal(Object.hasOwn(dense, 'gpuTimestamps'), false);
    assert.deepEqual(kernelTotals(dense), kernelTotals(sparse));
    assert.equal(dense.gpuDropped, profiles[0].dropped);
    assert.deepEqual(dense.chunks, profiles[0].chunks);
    assert.equal(worker.repeatedFinalFields.length, 2);
    for (const fields of worker.repeatedFinalFields) {
      assert.deepEqual(kernelTotals(fields), kernelTotals(sparse));
      assert.equal(fields.gpuDropped, profiles[0].dropped);
      assert.deepEqual(fields.chunks, profiles[0].chunks);
      assert.equal(Object.hasOwn(fields, 'gpuTimestamps'), false);
    }
    assert.equal(
      worker.messages.filter((message) => message.id === 1 && message.type === 'result').length,
      1,
    );
    assert.deepEqual(
      worker.messages.filter((message) => message.type === 'error'),
      [],
    );
    assert.equal(worker.devices[0].disposeCount, 1);
    assert.equal(worker.state().device, undefined);
    assert.equal(worker.state().jobId, undefined);
    await worker.send(start(2, { dense: false }));
    assert.equal(worker.opens[2].gpuTimestamps, undefined);
    assert.equal(worker.state().deviceTimestampRequest, true);
    await worker.lose(worker.devices[0]);
    assert.equal(worker.state().jobId, 2);
    assert.equal(worker.state().deviceUnavailable, false);
    assert.deepEqual(
      worker.messages.filter((message) => message.type === 'error'),
      [],
    );
    await worker.send({ type: 'finish', id: 2 });
    const next = result(worker, 2, 'result');
    assert.deepEqual(next.gpuKernels, profiles[2].rows);
    assert.equal(next.gpuDropped, profiles[2].dropped);
    assert.deepEqual(next.chunks, profiles[2].chunks);
  } finally {
    await worker.close();
  }
});

test('worker preserves Sparse GPU totals through Dense replacement without cumulative double counting', async () => {
  const profiles = diagnosticProfiles();
  const before = structuredClone(profiles);
  const worker = harness({
    diagnosticsByAttempt: (attempt) => profiles[attempt],
    captureFinalFields: true,
  });
  try {
    await worker.send(start(1));
    await worker.send({ type: 'finish', id: 1 });
    const sparse = result(worker, 1, 'result');
    const dense = result(worker, 1, 'denseResult');
    assert.deepEqual(sparse.gpuKernels, profiles[0].rows);
    assert.equal(sparse.gpuDropped, profiles[0].dropped);
    assert.deepEqual(sparse.chunks, profiles[0].chunks);
    const expectedRows = new Map([
      ['sparseOnly', profiles[0].rows[0]],
      ['shared', { name: 'shared', groups: 10, dispatches: 20, timedGroups: 3, gpuNs: 30 }],
      ['denseOnly', profiles[1].rows[1]],
    ]);
    const expectedChunks = { ...profiles[0].chunks, ...profiles[1].chunks };
    for (const fields of [dense, ...worker.repeatedFinalFields]) {
      assert.deepEqual(kernelTotals(fields), expectedRows);
      assert.equal(fields.gpuDropped, 10);
      assert.deepEqual(
        fields.chunks,
        expectedChunks,
        'Dense overrides only overlapping chunk keys',
      );
    }
    assert.equal(worker.repeatedFinalFields.length, 2);
    assert.deepEqual(profiles, before, 'merging must not mutate either device diagnostic snapshot');

    await worker.send(start(2, { dense: false }));
    await worker.send({ type: 'finish', id: 2 });
    const next = result(worker, 2, 'result');
    assert.deepEqual(next.gpuKernels, profiles[2].rows);
    assert.equal(next.gpuDropped, profiles[2].dropped);
    assert.deepEqual(next.chunks, profiles[2].chunks);
    assert.equal(
      next.gpuKernels.some((row) => row.name === 'sparseOnly' || row.name === 'denseOnly'),
      false,
    );
  } finally {
    await worker.close();
  }
});
