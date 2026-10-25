import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { assertKltRateCache } from './klt-rate-cache.mjs';
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

test('KLT rate reuse accepts a fixed variable-cost schedule and catches an in-call reset', () => {
  // CI 40216127198: all eight timed chunks fit 250ms, but the rising cost made warm use
  // 32 logical chunks after 30 cold chunks. Reuse does not imply monotonic chunk counts.
  const timedChunks = [
    [128, 83.6],
    [192, 134.1],
    [192, 161.6],
    [192, 189.2],
    [128, 93.4],
    [128, 139.8],
    [128, 140.2],
    [128, 132.3],
  ].map(([units, elapsedMs]) => ({ units, elapsedMs }));
  const rate = { msPerUnit: 1.02725830078125, chunks: 8, longestMs: 189.2, lastUnits: 128 };
  const coldRanges = [...timedChunks.map((chunk) => chunk.units), ...Array(6).fill(128), 64];
  const warmRanges = Array(16).fill(128);
  const cold = {
    rates: { kltRoundtrip: { ...rate } },
    rateBefore: null,
    firstSizingRate: null,
    sizingCalls: coldRanges.length,
    timedBefore: 0,
    timedAfter: 8,
    trackChunks: [...coldRanges, ...coldRanges],
  };
  const warm = {
    rates: { kltRoundtrip: { ...rate } },
    rateBefore: { ...rate },
    firstSizingRate: { ...rate },
    sizingCalls: warmRanges.length,
    timedBefore: 8,
    timedAfter: 8,
    trackChunks: [...warmRanges, ...warmRanges],
  };
  assert.equal(cold.trackChunks.length, 30);
  assert.equal(warm.trackChunks.length, 32);
  for (const pass of [cold, warm])
    assert.equal(
      pass.trackChunks.reduce((s, n) => s + n, 0),
      4096,
    );
  assert.ok(timedChunks.every((chunk) => chunk.elapsedMs < 250));
  assertKltRateCache([cold, warm], timedChunks, 8);
  const resetAtFirstSizing = structuredClone(warm);
  // The outside snapshot still looks correct: only the actual first sizing detects this reset.
  resetAtFirstSizing.firstSizingRate = null;
  assert.throws(
    () => assertKltRateCache([cold, resetAtFirstSizing], timedChunks, 8),
    /first warm sizing reuses the cold rate/,
  );
  const resetAtEntry = structuredClone(warm);
  resetAtEntry.rateBefore = null;
  assert.throws(
    () => assertKltRateCache([cold, resetAtEntry], timedChunks, 8),
    /warm entry reuses the cold rate/,
  );
});

test('KLT warm tracking continues the cold timing history until the sample limit', () => {
  const timedChunks = [
    [128, 83.6],
    [192, 134.1],
    [192, 161.6],
    [192, 189.2],
    [128, 93.4],
    [128, 139.8],
    [128, 140.2],
    [128, 132.3],
  ].map(([units, elapsedMs]) => ({ units, elapsedMs }));
  const coldRate = { msPerUnit: 0.946533203125, chunks: 6, longestMs: 189.2, lastUnits: 128 };
  const warmRate = { msPerUnit: 1.02725830078125, chunks: 8, longestMs: 189.2, lastUnits: 128 };
  const coldRanges = timedChunks.slice(0, 6).map((chunk) => chunk.units);
  const warmRanges = [...Array(7).fill(128), 64];
  const cold = {
    rates: { kltRoundtrip: { ...coldRate } },
    rateBefore: null,
    firstSizingRate: null,
    sizingCalls: coldRanges.length,
    timedBefore: 0,
    timedAfter: 6,
    trackChunks: [...coldRanges, ...coldRanges],
  };
  const warm = {
    rates: { kltRoundtrip: { ...warmRate } },
    rateBefore: { ...coldRate },
    firstSizingRate: { ...coldRate },
    sizingCalls: warmRanges.length,
    timedBefore: 6,
    timedAfter: 8,
    trackChunks: [...warmRanges, ...warmRanges],
  };
  for (const pass of [cold, warm])
    assert.equal(
      pass.trackChunks.reduce((s, n) => s + n, 0),
      1920,
    );
  assertKltRateCache([cold, warm], timedChunks, 8);
  const stoppedLearning = structuredClone(warm);
  stoppedLearning.rates.kltRoundtrip.chunks = 6;
  assert.throws(
    () => assertKltRateCache([cold, stoppedLearning], timedChunks, 8),
    /warm tracking keeps learning/,
  );
});
