// docs/gpu-throughput.md, item 2 (PR 2): `chunkSize` honors an occupancy floor (`minUnits`), learns
// the rate of a chunk from its GPU time when the device measured one, and halves the floor after
// one dispatch over the 100 ms ceiling. The test drives `ShadeDevice` with a stand-in for the
// TypeShade device, so that the planner arithmetic runs without a GPU.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/gpu.js';

const FLOOR = 16 * 1024,
  MIN = 16;

/** A device that plans like a WebGPU one, over a CPU device and a stand-in TypeShade device. */
async function fixture({ chunkTimingSamples = 8, rates = {}, gpuMs, occupancyFloors = true } = {}) {
  const device = await ShadeDevice.open({
    backend: 'cpu',
    poolBytes: 0,
    chunkTimingSamples,
    occupancyFloors,
  });
  const chunkRates = { ...rates };
  const fuses = [];
  const stand = {
    timeBudgetMs: 250,
    chunkRates,
    // The arithmetic of TypeShade's `fitChunk` for the fence-based fallback.
    chunkSize(key, max, min = 1) {
      const rate = chunkRates[key];
      if (rate === undefined) return Math.max(1, Math.min(max, min));
      const fit = Math.floor((250 * 0.8) / rate.msPerUnit);
      return Math.min(max, Math.max(1, min, Math.min(fit, rate.lastUnits * 4)));
    },
    async fuse(body, options) {
      body();
      fuses.push(options?.chunk);
      const units = options?.chunk?.units ?? 1;
      if (options?.chunk) {
        const previous = chunkRates[options.chunk.key];
        chunkRates[options.chunk.key] = {
          msPerUnit: 3 / units,
          chunks: (previous?.chunks ?? 0) + 1,
          longestMs: 3,
          lastUnits: units,
        };
      }
      return {
        tier: 'webgpu',
        dispatches: 1,
        submissions: 1,
        groups: [
          {
            calls: ['kernel'],
            fused: false,
            ...(gpuMs === undefined ? {} : { gpuTime: gpuMs(units) * 1e6 }),
          },
        ],
      };
    },
  };
  Object.defineProperty(device, 'shade', { value: stand });
  Object.defineProperty(device, 'backend', { value: 'webgpu' });
  /** Run the chunks of a job of `total` units; return the unit counts of its chunks. */
  async function run(key, total, minUnits = FLOOR) {
    const sizes = [];
    for (let first = 0; first < total;) {
      const n = Math.min(total - first, device.chunkSize(key, total - first, MIN, minUnits));
      sizes.push(n);
      await device.planSteps([['kernel', () => {}, undefined, { key, units: n, minUnits }]]);
      first += n;
    }
    return sizes;
  }
  return { device, chunkRates, fuses, run };
}

test('a fresh key reaches its floor in steps of at most four times the last chunk', async () => {
  const { run } = await fixture();
  const sizes = await run('k', 40000);
  assert.deepEqual(sizes.slice(0, 5), [16, 64, 256, 1024, 4096]);
  assert.equal(
    sizes.reduce((a, b) => a + b, 0),
    40000,
  );
});

test('a settled key plans ceil(N / floor) chunks, whatever the fence rate says', async () => {
  // A fence rate of 3 ms for one unit would size a chunk at 66 units; the floor wins.
  const rates = { k: { msPerUnit: 3, chunks: 8, longestMs: 3, lastUnits: 1 } };
  for (const total of [1, 1000, FLOOR, FLOOR + 1, 3 * FLOOR + 5]) {
    const { run } = await fixture({ rates });
    const sizes = await run('k', total);
    assert.equal(sizes.length, Math.ceil(total / FLOOR), `N ${total}: ${sizes}`);
  }
});

test('a key with no floor keeps the old arithmetic', async () => {
  const rates = { k: { msPerUnit: 3, chunks: 8, longestMs: 3, lastUnits: 1 } };
  const { device } = await fixture({ rates });
  assert.equal(device.chunkSize('k', 100000, MIN), 66);
  assert.equal(device.chunkSize('fresh', 100000, MIN), MIN);
});

test('the rate is learned from the GPU time of the chunk', async () => {
  // 0.001 ms a unit on the GPU, 3 ms a chunk on the fence: the fence rate would give 66 units.
  const { device, run } = await fixture({ chunkTimingSamples: 1, gpuMs: (units) => units * 0.001 });
  await run('k', MIN);
  assert.equal(device.chunkSize('k', 1e9, MIN, MIN), 80_000);
});

test('the rate falls back to the fence when a group has no GPU time', async () => {
  const { device, run } = await fixture({ chunkTimingSamples: 1 });
  await run('k', MIN);
  assert.equal(device.chunkSize('k', 1e9, MIN, MIN), Math.floor((250 * 0.8) / (3 / MIN)));
});

test('one dispatch over 100 GPU ms halves the floor, and a chunk fits the ceiling', async () => {
  const { device, run } = await fixture({ chunkTimingSamples: 1, gpuMs: () => 150 });
  await run('k', MIN);
  assert.equal(device.chunkSize('k', 1e9, MIN, FLOOR), FLOOR / 2);
  // A second overrun halves it again.
  await device.planSteps([
    ['kernel', () => {}, undefined, { key: 'k', units: 64, minUnits: FLOOR }],
  ]);
  assert.ok(device.chunkSize('k', 1e9, MIN, FLOOR) <= FLOOR / 2);
  // The size also holds to the ceiling at the measured rate: 80 ms at 150 / 64 ms a unit.
  const { device: other, run: runOther } = await fixture({
    chunkTimingSamples: 1,
    gpuMs: (units) => units * 0.5,
  });
  await runOther('k', MIN);
  assert.equal(other.chunkSize('k', 1e9, MIN, 1), 160);
});

test('a settled key with a floor is timed again every 16 chunks', async () => {
  const { fuses, device } = await fixture({
    rates: { k: { msPerUnit: 0.001, chunks: 8, longestMs: 1, lastUnits: 100 } },
  });
  for (let i = 0; i < 32; i++)
    await device.planSteps([
      ['kernel', () => {}, undefined, { key: 'k', units: 100, minUnits: FLOOR }],
    ]);
  assert.equal(fuses.filter((chunk) => chunk !== undefined).length, 2);
  const { fuses: plain, device: other } = await fixture({
    rates: { k: { msPerUnit: 0.001, chunks: 8, longestMs: 1, lastUnits: 100 } },
  });
  for (let i = 0; i < 32; i++)
    await other.planSteps([['kernel', () => {}, undefined, { key: 'k', units: 100 }]]);
  assert.equal(plain.filter((chunk) => chunk !== undefined).length, 0);
});

test('a device without occupancy floors keeps the old arithmetic and learns nothing from GPU time', async () => {
  const rates = { k: { msPerUnit: 3, chunks: 8, longestMs: 3, lastUnits: 1 } };
  const { device, run } = await fixture({ rates, occupancyFloors: false, gpuMs: () => 150 });
  assert.equal(device.chunkSize('k', 100000, MIN, FLOOR), 66);
  const sizes = await run('k', 1000);
  assert.ok(
    sizes.every((n) => n <= 66),
    `${sizes}`,
  );
});
