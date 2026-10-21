// Verifies: docs/dense-mvs.md Amendment 3 in section 8 (the dense stage runs within any device
// budget): more views than the all-resident layout of 4.2 held, on a budget it could not hold,
// and the streamed depth passes and the streamed fusion against window mode, bit for bit.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, planMvsViews, reconstructDense } from '../../dist/internal/gpu.js';
import { MVS_HEIGHT, MVS_WIDTH, mvsScene, mvsWidePlacements } from './mvs-cases.mjs';

const P = MVS_WIDTH * MVS_HEIGHT;
// A small configuration, since the paths are compared bit for bit: a 5 × 5 window sampled at step
// 2 and one iteration of each pass.
const SMALL = { windowRadius: 2, windowStep: 2, photometricIterations: 1, geometricIterations: 1 };

async function dense(scene, options, deviceOptions = {}) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, ...deviceOptions });
  const active = () => device.statistics.liveBytes - device.statistics.pooledBytes;
  let peak = 0;
  const sample = () => (peak = Math.max(peak, active()));
  try {
    const started = performance.now();
    const result = await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      ...options,
      onView: sample,
      onProgress: sample,
    });
    const ms = performance.now() - started;
    assert.equal(device.statistics.liveBytes, 0);
    return { result, ms, peak };
  } finally {
    await device.dispose();
  }
}

const modes = (residency) => ({
  photometric: [residency.photometric.mode, residency.photometric.slots],
  geometric: [residency.geometric.mode, residency.geometric.slots],
  fusion: [residency.fusion.mode, residency.fusion.slots],
});

test('reconstructDense: 9 views on a budget below the all-resident layout of 4.2', async () => {
  const scene = mvsScene({ placements: mvsWidePlacements(9), reference: 4, landmarks: 60 });
  const plan = planMvsViews(scene);
  const N = plan.packed.length;
  assert.equal(N, 9);
  // The all-resident depth stage needs (16 N + 4) P bytes. This budget held at most 7 views
  // (`denseViewLimit` before Amendment 3: ⌊(workingBytes / P − 4) / 16⌋).
  const workingBytes = 1.5 * 1024 ** 2;
  assert.ok(Math.floor((workingBytes / P - 4) / 16) < N);
  assert.ok((16 * N + 4) * P > workingBytes);
  const { result, ms, peak } = await dense(scene, {}, { workingBytes });
  console.log(
    JSON.stringify({
      views: N,
      ms: Math.round(ms),
      points: result.points.count,
      peak,
      allResident: (16 * N + 4) * P,
      workingBytes,
      residency: modes(result.residency),
      peakBytes: result.residency.peakBytes,
    }),
  );
  assert.deepEqual(
    result.views.map((view) => view.status),
    Array(N).fill('done'),
  );
  assert.ok(result.points.count > 0);
  assert.ok(result.residency.peakBytes <= workingBytes);
  assert.ok(peak <= workingBytes);
});

test('streamed depth passes and the streamed fusion give the window mode results bit for bit', async () => {
  const scene = mvsScene();
  const runs = {
    // The window of every stage fits.
    window: await dense(scene, {}),
    // Every stage streams two sources or fusion views at a time, forced, over the whole image.
    groups: await dense(scene, { streamGroup: 2 }),
    // Less than one window of any stage: one source and one fusion view at a time, in chunks of
    // rows.
    tiny: await dense(scene, { residentBytes: 470_000 }),
  };
  const report = Object.fromEntries(
    Object.entries(runs).map(([name, { result, ms, peak }]) => [
      name,
      {
        ms: Math.round(ms),
        peak,
        points: result.points.count,
        residency: {
          ...modes(result.residency),
          elements: result.residency.geometric.elements,
          chunk: result.residency.fusion.chunk,
          peakBytes: result.residency.peakBytes,
        },
      },
    ]),
  );
  console.log(JSON.stringify(report));
  const { window, groups, tiny } = runs;
  assert.deepEqual(modes(window.result.residency), {
    photometric: ['window', 5],
    geometric: ['window', 5],
    fusion: ['window', 5],
  });
  assert.deepEqual(modes(groups.result.residency), {
    photometric: ['streamed', 3],
    geometric: ['streamed', 3],
    fusion: ['streamed', 3],
  });
  assert.equal(groups.result.residency.geometric.elements, (MVS_WIDTH / 2) * MVS_HEIGHT);
  assert.equal(groups.result.residency.fusion.chunk, P);
  assert.deepEqual(modes(tiny.result.residency), {
    photometric: ['streamed', 2],
    geometric: ['streamed', 2],
    fusion: ['streamed', 2],
  });
  assert.ok(tiny.result.residency.geometric.elements < (MVS_WIDTH / 2) * MVS_HEIGHT);
  assert.ok(tiny.result.residency.fusion.chunk < P);
  // The last chunk of pixels is shorter than the others: the scan must not count the flags or
  // the block sums of the chunk before it.
  assert.notEqual(P % tiny.result.residency.fusion.chunk, 0);
  assert.notEqual(((MVS_WIDTH / 2) * MVS_HEIGHT) % tiny.result.residency.geometric.elements, 0);
  for (const [name, budget] of [
    ['groups', 64 * 1024 ** 2],
    ['tiny', 470_000],
  ]) {
    const run = runs[name];
    assert.ok(run.result.residency.peakBytes <= budget, `${name} plan`);
    assert.ok(run.peak <= budget, `${name} peak ${run.peak}`);
    assert.ok(window.result.points.count > 0);
    assert.deepEqual(run.result.views, window.result.views, `${name} views`);
    assert.deepEqual(run.result.points, window.result.points, `${name} points`);
  }
});

test('a budget below the floor of streamed mode is refused with the bytes it needs', async () => {
  const scene = mvsScene();
  await assert.rejects(dense(scene, { residentBytes: 100_000 }), {
    name: 'RangeError',
    message: /needs at least \d+ B for its photometric stage at 128 × 96 .* budget is 100000 B/,
  });
});
