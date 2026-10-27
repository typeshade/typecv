// docs/gpu-budget.md, section 2: a worker's working budget comes from the adapter's limits.
import test from 'node:test';
import assert from 'node:assert/strict';
import { workingBytesFromAdapter } from './src/lib/working-budget.ts';
import {
  ADAPTER,
  WorkerHarness,
  bundleWorker,
  pageMultiStart,
  pageStart,
} from './lab-worker-harness.mjs';

const MiB = 1024 ** 2;

test('the working budget is half of the adapter maxBufferSize when that is larger', () => {
  // The adapter of run record d40f8950: maxBufferSize 2 GiB, 640 px frames (64 MiB scale).
  assert.equal(workingBytesFromAdapter(64 * MiB, 2048 * MiB), 1024 * MiB);
  assert.equal(workingBytesFromAdapter(256 * MiB, 2048 * MiB), 1024 * MiB);
});

test('a small adapter, or none, keeps the scale of the analysis frame', () => {
  assert.equal(workingBytesFromAdapter(64 * MiB, 256 * MiB), 128 * MiB);
  assert.equal(workingBytesFromAdapter(64 * MiB, 64 * MiB), 64 * MiB);
  assert.equal(workingBytesFromAdapter(64 * MiB, undefined), 64 * MiB);
});

// Until phase 5a this test read the source of both workers. Each had to call
// `workingBytesFromAdapter(` and set `adapterLimits: true`, and neither could set `workingBytes` to
// the bare scale budget `Math.round(64 * 1024 ** 2 * scale)`. Phase 5a moved the budgets into
// `deviceBudgets` of `src/lib/working-budget.ts`, so the test now runs both workers in the
// transcript harness and reads the options of the device that each opens. It checks the same three
// things (`docs/library-structure.md`, section 7, item 2).
test('both workers open the device with the adapter limits and the derived budget', async () => {
  const code = await bundleWorker();
  // The harness's adapter reports 256 MiB. Half of it is above the 64 MiB scale of a 640 px frame.
  const maxBufferSize = ADAPTER.limits.maxBufferSize;
  assert.equal(maxBufferSize, 256 * MiB);
  for (const [worker, message] of [
    ['src/workers/sfm.worker.ts', pageStart(1)],
    ['src/workers/multi-video.ts', pageMultiStart(1)],
  ]) {
    assert.equal(message.analysisEdge, 640, worker);
    assert.equal(message.firstOctave, 0, worker);
    const h = new WorkerHarness(code);
    await h.send(message);
    const opens = h.events.filter((event) => event.t === 'call' && event.fn === 'ShadeDevice.open');
    assert.equal(opens.length, 1, worker);
    const [options] = opens[0].args;
    assert.equal(options.adapterLimits, true, worker);
    assert.equal(options.workingBytes, workingBytesFromAdapter(64 * MiB, maxBufferSize), worker);
    assert.equal(options.workingBytes, 128 * MiB, worker);
    assert.notEqual(options.workingBytes, Math.round(64 * 1024 ** 2 * 1), worker);
  }
});
