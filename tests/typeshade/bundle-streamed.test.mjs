// docs/gpu-budget.md, section 3: the bundle adjustment streams its observations and points in
// windows when the whole problem does not fit the working budget. The CPU tier gives the results
// of the all-resident run, bit for bit, and a budget that used to throw now completes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, bundleAdjustGpu } from '../../dist/gpu.js';
import { bundleFixture } from './geometry-cases.mjs';

const counts = (model) => ({
  C: model.views.length,
  P: model.landmarks.length,
  M: model.landmarks.reduce((n, p) => n + p.observations.length, 0),
});
// The bytes of the arrays that stay resident when the BA streams (section 3.1 of the record), the
// CPU tier with the camera scratch of the streamed route, and the window of `m` observations.
function floorBytes(model, shared) {
  const { C, P, M } = counts(model);
  const doubles =
    18 * C +
    3 * P +
    2 * M +
    (12 * C + 3 * P) +
    (30 * C + 3 * P) +
    27 +
    3 * Math.ceil(M / 64) +
    (12 * C + 3 * P + 26) +
    78 * C +
    36 * C;
  return 8 * doubles + 4 * (3 * M + P + 2 * C + 2) + 8 + (shared ? 8 * (6 * C + 6) : 0);
}
function residentBytes(model, shared) {
  const { C, P, M } = counts(model);
  return (
    floorBytes(model, shared) -
    8 * 36 * C +
    8 * ((shared ? 45 : 42) * M + 13 * P) +
    (shared ? 16 * M : 0)
  );
}
function windowBytes(model, m, shared) {
  const { P, M } = counts(model);
  return m * (shared ? 376 : 336) + Math.max(1, Math.ceil((m * P) / M)) * 104;
}
const key = (r) =>
  JSON.stringify([
    r.reconstruction.views.map((v) => [...v.pose.R, ...v.pose.t]),
    r.reconstruction.landmarks.map((l) => [...l.xyz]),
    r.diagnostics,
    r.cgIterations,
    r.linearSolverFailures,
  ]);
async function run(model, workingBytes, options) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes });
  try {
    const result = await bundleAdjustGpu(device, model, options);
    assert.equal(device.statistics.liveBytes, 0);
    return result;
  } finally {
    await device.dispose();
  }
}

for (const shared of [false]) {
  const name = shared ? 'shared focal' : 'fixed focal';
  test(`${name}: windows of 64, 150 and 500 observations give the resident run, bit for bit`, async () => {
    const model = bundleFixture({ count: 120, views: 8 });
    const options = { iterations: 5, huberPx: 1, refineSharedFocal: shared };
    const expected = await run(model, 2 * residentBytes(model, shared), options);
    assert.ok(expected.diagnostics.acceptedSteps > 0);
    for (const m of [64, 150, 500]) {
      const budget = floorBytes(model, shared) + windowBytes(model, m, shared);
      assert.ok(budget < residentBytes(model, shared));
      const streamed = await run(model, budget, options);
      assert.equal(key(streamed), key(expected), `window of ${m} observations`);
    }
  });

  test(`${name}: a local window and a shifted gauge stream bit for bit`, async () => {
    const model = bundleFixture({ count: 90, views: 6, shift: [7, -3, 2] });
    const options = {
      iterations: 4,
      huberPx: 1,
      refineSharedFocal: shared,
      anchorViewId: 20,
      scaleViewId: 40,
    };
    const expected = await run(model, 2 * residentBytes(model, shared), options);
    const budget = floorBytes(model, shared) + windowBytes(model, 128, shared);
    assert.equal(key(await run(model, budget, options)), key(expected));
    const local = { ...options, variableViewIds: [30, 50], scaleViewId: 40 };
    const localExpected = await run(model, 2 * residentBytes(model, shared), local);
    assert.equal(key(await run(model, budget, local)), key(localExpected));
  });

  test(`${name}: an outlier and a failed solve stream bit for bit`, async () => {
    const model = bundleFixture({ count: 100, views: 6, outlier: true });
    const options = { iterations: 6, huberPx: 1, refineSharedFocal: shared };
    const expected = await run(model, 2 * residentBytes(model, shared), options);
    const budget = floorBytes(model, shared) + windowBytes(model, 64, shared);
    assert.equal(key(await run(model, budget, options)), key(expected));
  });
}

test('a budget that used to throw now completes, with the least window', async () => {
  const model = bundleFixture({ count: 200, views: 10 });
  const options = { iterations: 3, huberPx: 1 };
  const expected = await run(model, 2 * residentBytes(model, false), options);
  // The floor and one window of 64 observations: a seventh of the resident bytes.
  const budget = floorBytes(model, false) + windowBytes(model, 64, false);
  assert.ok(budget * 2 < residentBytes(model, false));
  const result = await run(model, budget, options);
  assert.equal(key(result), key(expected));
});

test('a budget below the arrays that stay resident throws, naming the floor', async () => {
  const model = bundleFixture({ count: 60, views: 6 });
  await assert.rejects(
    run(model, Math.max(4096, floorBytes(model, false) - 1000), { iterations: 1 }),
    /arrays that stay resident/,
  );
});
