// Verifies: docs/dense-mvs.md step 4 (sections 3.8, 3.9 and 7.1, the geometric pass and the
// filter).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  decodeMvsNormal,
  mvsDepthMapsGpu,
  mvsPhotometricGpu,
  planMvsViews,
} from '../../dist/internal/gpu.js';
import { depthMetrics, mvsScene } from './mvs-cases.mjs';

const REFERENCE_ID = 2;
const scene = mvsScene();
const plan = planMvsViews(scene);
const reference = plan.views[REFERENCE_ID].index;
const range = plan.views[REFERENCE_ID].depthRange;
const metrics = (map, keep) =>
  depthMetrics(
    map,
    scene.truth[REFERENCE_ID],
    plan.normalization.scale,
    range,
    decodeMvsNormal,
    keep,
  );
const rounded = (record) =>
  Object.fromEntries(Object.entries(record).map(([k, v]) => [k, +v.toFixed(4)]));

async function withDevice(body) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const started = performance.now();
    const result = await body(device);
    const ms = performance.now() - started;
    assert.equal(device.statistics.liveBytes, 0);
    return { result, ms };
  } finally {
    await device.dispose();
  }
}
const depthMaps = (options) =>
  withDevice((device) => mvsDepthMapsGpu(device, scene, plan, scene.images, options));

// The accuracy run. Every view runs the photometric pass, because the geometric pass of the
// central view reads the depth maps of all its sources. Only the central view runs the geometric
// pass and the filter, since the order and the set of the geometric references change no result
// (the test below). The CI run uses 4 photometric iterations to stay near 40 s on the CPU tier.
// TYPECV_MVS_DEFAULTS=1 runs the defaults (8 iterations), the numbers of the step 4 record.
const DEFAULTS = process.env.TYPECV_MVS_DEFAULTS === '1';
test('mvsPropagate geometric and mvsFilter on the CPU tier: the error tail shrinks', async () => {
  const { result, ms } = await depthMaps({
    readStages: true,
    geometricOrder: [reference],
    ...(DEFAULTS ? {} : { photometricIterations: 4 }),
  });
  const map = result.views[reference];
  const photometric = metrics(map.photometric),
    geometric = metrics(map.geometric),
    filtered = metrics(map, (i) => map.depth[i] > 0);
  console.log(
    JSON.stringify({
      ms: Math.round(ms),
      photometricIterations: DEFAULTS ? 8 : 4,
      photometric: rounded(photometric),
      geometric: rounded(geometric),
      filtered: rounded(filtered),
    }),
  );
  // Section 7.1: the median is not larger than after the photometric pass, and the 95th
  // percentile is smaller. The step 4 record also holds the 90th percentile.
  assert.ok(filtered.median <= photometric.median, `median ${filtered.median}`);
  assert.ok(filtered.p95 < photometric.p95 / 2, `p95 ${filtered.p95} ${photometric.p95}`);
  assert.ok(filtered.p90 < photometric.p90 / 1.5, `p90 ${filtered.p90} ${photometric.p90}`);
  assert.ok(filtered.median <= 0.01, `median relative depth error ${filtered.median}`);
  // The geometric pass alone, before the filter, already shortens the tail.
  assert.ok(geometric.p95 < photometric.p95, `geometric p95 ${geometric.p95}`);
  // Regression bounds of the step 4 record: the kept fraction (76.8 % at the defaults, 75.7 % at
  // 4 iterations) and the median normal error (10.6° at the defaults, 14.6° at 4 iterations).
  // Section 7.1's 5° is open.
  assert.ok(filtered.kept >= 0.7, `kept ${filtered.kept}`);
  assert.ok(filtered.medianNormalDeg < photometric.medianNormalDeg, 'the normal error grows');
  const normalBound = DEFAULTS ? 12 : 16;
  assert.ok(
    filtered.medianNormalDeg <= normalBound,
    `median normal error ${filtered.medianNormalDeg}°`,
  );
  // A rejected pixel has depth 0. A view outside `geometricOrder` keeps no depth.
  for (let v = 0; v < result.views.length; v++)
    if (v !== reference) assert.ok(result.views[v].depth.every((d) => d === 0));
});

// A small configuration, since only bit equality matters here: a 3 × 3 window and one iteration
// of each pass. All five views run the geometric pass and the filter.
const SMALL = { windowRadius: 2, windowStep: 2, photometricIterations: 1, geometricIterations: 1 };
const bytes = (array) => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
function assertSameMaps(actual, expected, label) {
  assert.equal(actual.views.length, expected.views.length);
  for (let v = 0; v < expected.views.length; v++)
    for (const stage of [null, 'photometric', 'geometric']) {
      const a = stage ? actual.views[v][stage] : actual.views[v],
        b = stage ? expected.views[v][stage] : expected.views[v];
      const where = `${label}: view ${v} ${stage ?? 'filtered'}`;
      assert.deepEqual(bytes(a.depth), bytes(b.depth), `${where} depth`);
      assert.deepEqual(a.normal, b.normal, `${where} normal`);
      assert.deepEqual(bytes(a.cost), bytes(b.cost), `${where} cost`);
    }
}

test('mvsDepthMapsGpu: the order of the references and chunks change no result', async () => {
  const views = plan.packed.map((_, v) => v);
  const runs = [];
  // The second run reverses both orders and forces chunks of 1000 elements, which divide no row
  // (64 color indices or 128 pixels), so chunks cross rows. Chunks of 48 are step 2's test.
  for (const options of [
    {},
    { order: [...views].reverse(), geometricOrder: [3, 1, 4, 0, 2], chunkPixels: 1000 },
  ])
    runs.push(await depthMaps({ ...SMALL, readStages: true, ...options }));
  console.log(JSON.stringify({ smallMs: runs.map((run) => Math.round(run.ms)) }));
  const [forward, reordered] = runs.map((run) => run.result);
  const calls = (report, name) =>
    report.groups.flatMap((group) => group.calls).filter((call) => call === name).length;
  assert.equal(calls(forward.report, 'mvsFilter'), 5);
  assert.equal(calls(reordered.report, 'mvsFilter'), 5 * Math.ceil((128 * 96) / 1000));
  // The filter does work: it rejects pixels and keeps pixels.
  const kept = forward.views[reference].depth.filter((d) => d > 0).length;
  assert.ok(kept > 0 && kept < forward.views[reference].depth.length, `kept ${kept}`);
  assertSameMaps(reordered, forward, 'other orders, chunks of 1000');
  // The photometric pass of mvsDepthMapsGpu is that of mvsPhotometricGpu, bit for bit.
  const { result: single } = await withDevice((device) =>
    mvsPhotometricGpu(device, scene, plan, scene.images, reference, SMALL),
  );
  const photometric = forward.views[reference].photometric;
  assert.deepEqual(bytes(photometric.depth), bytes(single.depth));
  assert.deepEqual(photometric.normal, single.normal);
  assert.deepEqual(bytes(photometric.cost), bytes(single.cost));
});
