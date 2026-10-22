import test from 'node:test';
import assert from 'node:assert/strict';
import { DeviceDiagnosticTracker } from '../src/gpu/device-diagnostics.ts';
import { GpuRequestContext, gpuLossContextValid } from './src/lib/gpu-loss-context.ts';
import { denseRecord, runRecordProblem } from './src/lib/run-record.ts';
import { createDenseJob, denseFailure } from './src/lib/dense-run.ts';
import { sampleRecord, olderRecord } from './record-fixture.mjs';

test('a loss between frames retains history without blaming the last completed request', () => {
  let now = 10;
  const context = new GpuRequestContext(() => now);
  context.begin('frame', 10);
  now = 20;
  context.progress('scatterFloatMatches', 'start');
  now = 30;
  context.progress('scatterFloatMatches', 'complete');
  now = 40;
  context.end();
  now = 100;
  assert.deepEqual(context.location('guided matching 1 → 10'), {
    operation: 'waiting for next request',
    kernel: '',
  });
  const idle = context.snapshot();
  assert.equal(idle.state, 'idle');
  assert.equal(idle.request, null);
  assert.deepEqual(idle.lastCompletedRequest, { type: 'frame', frameId: 10, ageMs: 60 });
  assert.deepEqual(idle.lastKernel, {
    name: 'scatterFloatMatches',
    phase: 'submitted',
    ageMs: 70,
    inCurrentRequest: false,
  });
  assert.ok(gpuLossContextValid(idle));
  context.begin('frame', 13);
  assert.deepEqual(context.location('processing frame 13'), {
    operation: 'processing frame 13',
    kernel: '',
  });
  context.progress('preparePairModels', 'start');
  assert.equal(context.location('pair verification').kernel, 'preparePairModels');
  assert.equal(context.snapshot().lastKernel.inCurrentRequest, true);
  assert.equal(context.snapshot().lastKernel.phase, 'started');
  assert.equal(idle.lastKernel.name, 'scatterFloatMatches', 'old snapshots stay immutable');
});

test('device context preserves resident peaks, bounded native errors and destroy ownership', () => {
  const gpu = new EventTarget();
  const bytes = { liveBytes: 8192, pooledBytes: 4096 };
  const tracker = new DeviceDiagnosticTracker(
    () => bytes,
    16384,
    () => 2,
    gpu,
  );
  tracker.observe();
  bytes.pooledBytes = 0;
  tracker.observe();
  bytes.liveBytes = 1024;
  bytes.pooledBytes = 1024;
  for (let i = 0; i < 6; i++) {
    const event = new Event('uncapturederror', { cancelable: true });
    Object.defineProperty(event, 'error', {
      value: { name: 'GPUValidationError', message: String(i) + 'x'.repeat(500) },
    });
    gpu.dispatchEvent(event);
    assert.equal(event.defaultPrevented, false);
  }
  tracker.disposalRequested = true;
  const beforeDestroy = tracker.snapshot();
  assert.equal(
    beforeDestroy.destroyRequested,
    false,
    'pending disposal is not an observed destroy',
  );
  assert.equal(beforeDestroy.recentErrors.length, 4);
  assert.equal(beforeDestroy.recentErrors[0].message[0], '2');
  assert.equal(beforeDestroy.recentErrors[0].message.length, 400);
  assert.deepEqual(beforeDestroy.memory, {
    liveBytes: 1024,
    pooledBytes: 1024,
    peakLiveBytes: 8192,
    peakActiveBytes: 8192,
    workingBytes: 16384,
    pendingReads: 2,
  });
  tracker.destroyRequested = true;
  assert.equal(beforeDestroy.destroyRequested, false);
  assert.equal(tracker.snapshot().destroyRequested, true);
  tracker.detach();
  gpu.dispatchEvent(new Event('uncapturederror'));
  assert.deepEqual(tracker.snapshot().recentErrors, beforeDestroy.recentErrors);
  const opaque = new DeviceDiagnosticTracker(
    () => bytes,
    16384,
    () => 0,
    undefined,
    true,
  );
  assert.equal(opaque.snapshot().destroyRequested, null);
  assert.equal(opaque.snapshot().errorEventsAvailable, false);
});

test('collector accepts optional sparse and dense context, rejects malformed fields and keeps older records', () => {
  const request = new GpuRequestContext(() => 100);
  request.begin('frame', 10);
  const tracker = new DeviceDiagnosticTracker(
    () => ({ liveBytes: 2048, pooledBytes: 1024 }),
    16384,
    () => 0,
  );
  const context = request.snapshot(tracker.snapshot());
  const failure = {
    message: 'WebGPU device lost',
    operation: 'processing frame 10',
    kernel: '',
    reason: 'destroyed',
    framesProcessed: 3,
    context,
  };
  assert.equal(runRecordProblem(sampleRecord({ failure })), undefined);
  for (const format of ['typecv-run-record-v3', 'typecv-run-record-v4', 'typecv-run-record-v5']) {
    const record = olderRecord(format);
    if (format === 'typecv-run-record-v3') {
      delete record.timing.chunks;
      delete record.timing.budgetMs;
      delete record.timing.mainThread;
    }
    assert.equal(runRecordProblem(record), undefined);
  }
  const malformed = [
    null,
    {},
    { ...context, state: 'idle' },
    { ...context, state: ['active'] },
    { ...context, request: { type: ['frame'], ageMs: 0 } },
    { ...context, request: { type: 'frame', ageMs: -1 } },
    { ...context, request: { type: 'frame', frameId: 1.5, ageMs: 0 } },
    {
      ...context,
      lastKernel: { name: 'x'.repeat(65), phase: 'started', ageMs: 0, inCurrentRequest: true },
    },
    {
      ...context,
      device: { ...context.device, recentErrors: Array(5).fill({ name: '', message: '' }) },
    },
    {
      ...context,
      device: { ...context.device, memory: { ...context.device.memory, liveBytes: Infinity } },
    },
  ];
  for (const invalid of malformed) {
    assert.equal(gpuLossContextValid(invalid), false);
    assert.equal(
      runRecordProblem(sampleRecord({ failure: { ...failure, context: invalid } })),
      'failure',
    );
  }
  const dense = denseRecord({
    failure: denseFailure(
      createDenseJob(false),
      'WebGPU device lost',
      'mvsPropagate',
      'destroyed',
      context,
    ),
  });
  assert.deepEqual(dense.failure.context, context);
  assert.equal(runRecordProblem(sampleRecord({ dense })), undefined);
  const invalidDense = structuredClone(dense);
  invalidDense.failure.context.device.memory.pendingReads = -1;
  assert.equal(runRecordProblem(sampleRecord({ dense: invalidDense })), 'dense');
});
