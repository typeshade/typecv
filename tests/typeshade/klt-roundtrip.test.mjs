import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import {
  roundtripFixture,
  roundtripReferences,
  assertRoundtrips,
} from './klt-roundtrip-fixture.mjs';
import { runRoundtrips } from './klt-roundtrip-cases.mjs';

test('opt-in grouped KLT roundtrips preserve f32 tracking and the separate-pass default', async () => {
  const input = roundtripFixture(),
    reference = roundtripReferences(api, input),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertRoundtrips(await runRoundtrips(api, device, input), input, reference);
  } finally {
    await device.dispose();
  }
});

test('adaptive grouped KLT chunks reserve time for a measured per-track cost increase', () => {
  const budget = 250;
  // The loaded WebGPU observation: about .619 ms/point followed by .937 ms/point.
  // This cost model reproduces the missed prediction without relying on wall-clock speed.
  const measured = 0.619453125;
  const observed = 299.9 / 320;
  const requested = Math.floor((budget * 0.8) / measured);
  assert.ok(requested * observed > budget, 'the ordinary proposal misses this cost increase');
  const chunks = [
    ...api.kltChunkSteps(
      {
        backend: 'webgpu',
        chunkSize: (_key, remaining, min) => Math.min(remaining, Math.max(min, requested)),
      },
      { count: 1024 + 37, key: 'kltRoundtrip', min: 128 },
      () => {},
    ),
  ];
  assert.equal(
    chunks.reduce((sum, step) => sum + step[3].units, 0),
    1024 + 37,
  );
  for (const step of chunks) {
    assert.equal(step[3].key, 'kltRoundtrip');
    assert.ok(step[3].units * observed <= budget, 'the reserved proposal fits this cost increase');
  }
  for (const step of chunks.slice(0, -1)) assert.equal(step[3].units % 64, 0);
  const sized = (device, plan) =>
    [...api.kltChunkSteps(device, plan, () => {})].map((step) => step[3]?.units);
  // A fast rate has enough reserved capacity for the complete final range. Do not split it
  // merely because the device first capped its proposal to that range.
  const fast = { backend: 'webgpu', chunkSize: (_key, max) => Math.min(max, 4096) };
  assert.deepEqual(sized(fast, { count: 256, key: 'kltRoundtrip', min: 128 }), [256]);
  assert.deepEqual(sized(fast, { count: 257, key: 'kltRoundtrip', min: 128 }), [257]);
  const minimum = { backend: 'webgpu', chunkSize: (_key, _remaining, min) => min };
  assert.deepEqual(sized(minimum, { count: 129, key: 'kltRoundtrip', min: 128 }), [128, 1]);
  assert.deepEqual(
    sized(minimum, { count: 325, fixed: 320, key: 'kltRoundtrip', min: 128 }),
    [320, 5],
  );
  assert.deepEqual(
    sized(
      { backend: 'cpu', chunkSize: (_key, n) => n },
      { count: 1061, key: 'kltRoundtrip', min: 128 },
    ),
    [1061],
  );
  assert.deepEqual(
    sized(
      { backend: 'webgpu', chunkSize: (_key, n) => n },
      { count: 1061, key: 'kltTrack', min: 64 },
    ),
    [1061],
  );
});
