import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { compactInput, assertCompact } from './klt-compact-fixture.mjs';
import { runCompactCases } from './klt-compact-cases.mjs';

test('compact KLT copies exact checked records/NCC and reduces CPU-tier snapshots without dropping validation', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = compactInput(),
      result = await runCompactCases(api, device, input);
    assertCompact(result, input);
    console.log(
      JSON.stringify(
        result.rows.map(({ name, legacy, compact }) => ({
          name,
          legacy: legacy.counts,
          compact: compact.counts,
        })),
      ),
    );
  } finally {
    await device.dispose();
  }
});

test('compact KLT checked-read failure waits for a pending backward read before recycling buffers', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    originalRead = device.read.bind(device),
    originalRecycle = device.recycle.bind(device),
    input = compactInput(),
    failure = new Error('compact checked read failed'),
    recycled = [];
  let calls = 0,
    releasePending,
    signalPending,
    settled = false;
  const pendingStarted = new Promise((resolve) => {
    signalPending = resolve;
  });
  device.recycle = (buffer) => {
    recycled.push(buffer);
    return originalRecycle(buffer);
  };
  device.read = (buffer, length) => {
    calls++;
    if (calls === 1) return Promise.reject(failure);
    if (calls === 2) {
      signalPending();
      return new Promise((resolve, reject) => {
        releasePending = () => originalRead(buffer, length).then(resolve, reject);
      });
    }
    return originalRead(buffer, length);
  };
  const result = api
    .trackKltGpu(
      device,
      Float32Array.from(input.first),
      Float32Array.from(input.shifted),
      input.width,
      input.height,
      input.points,
      api.kltOptionsFor(input.width, { levels: 2, halfWindow: 4 }),
      { compactOutput: true },
    )
    .then(
      () => {
        settled = true;
        return null;
      },
      (error) => {
        settled = true;
        return error;
      },
    );
  try {
    assert.equal(
      await Promise.race([pendingStarted.then(() => true), result.then(() => false)]),
      true,
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls, 2, 'only checked+NCC and backward snapshots were requested');
    assert.equal(settled, false);
    assert.equal(recycled.length, 0);
    const release = releasePending;
    releasePending = undefined;
    release();
    assert.equal(await result, failure);
    assert.equal(recycled.length, 8);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    releasePending?.();
    await result;
    device.read = originalRead;
    device.recycle = originalRecycle;
    await device.dispose();
  }
});
