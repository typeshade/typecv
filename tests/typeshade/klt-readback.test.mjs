import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { runKltReadbackCases } from './klt-readback-cases.mjs';
import { assertKltReadbacks, kltReadbackInput } from './klt-readback-fixture.mjs';

test('KLT independent reads overlap without changing outputs, pyramids or snapshot bytes on CPU', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await runKltReadbackCases(api, device, kltReadbackInput());
    assertKltReadbacks(result);
    console.log(
      JSON.stringify(
        result.rows.map(({ name, serial, concurrent }) => ({
          name,
          serial: serial.counts,
          concurrent: concurrent.counts,
        })),
      ),
    );
  } finally {
    await device.dispose();
  }
});

test('a rejected KLT read waits for pending siblings before recycling any resident buffer', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
    originalRead = device.read.bind(device),
    originalRecycle = device.recycle.bind(device),
    input = kltReadbackInput(),
    failure = new Error('injected first read failure'),
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
      input.first,
      input.second,
      input.width,
      input.height,
      input.points,
      api.kltOptionsFor(input.width, { levels: 2, halfWindow: 4 }),
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
      'tracking starts the sibling read before it can settle',
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls, 3, 'all independent result reads were started');
    assert.equal(settled, false);
    assert.equal(recycled.length, 0, 'pending read still owns its buffers');
    const release = releasePending;
    releasePending = undefined;
    release();
    assert.equal(await result, failure);
    assert.equal(recycled.length, 8, 'point buffers and image pyramids released after settlement');
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    releasePending?.();
    await result;
    device.read = originalRead;
    device.recycle = originalRecycle;
    await device.dispose();
  }
});
