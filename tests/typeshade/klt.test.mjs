// Verifies: docs/klt-tracking.md step 2 (sections 4.2, 4.3, 4.5 and 7.1 on the CPU tier).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  ShadeDevice,
  KLT_END,
  buildKltPyramid,
  kltOptionsFor,
  trackKlt,
  trackKltGpu,
} from '../../dist/internal/gpu.js';
import { flowErrors, gridPoints, renderScene } from '../klt-cases.ts';

const W = 320,
  H = 200,
  scene = { width: W, height: H },
  options = kltOptionsFor(W),
  first = renderScene(scene),
  points = gridPoints(W, H, 60, 50, 4, 3);

// The translation cases run at 640 x 360, the size of section 7.1: 30 px needs the pyramid there.
const BW = 640,
  BH = 360,
  bigScene = { width: BW, height: BH },
  bigOptions = kltOptionsFor(BW),
  bigFirst = renderScene(bigScene),
  bigPoints = gridPoints(BW, BH, 120, 100, 4, 3);

const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

async function withDevice(body) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await body(device);
    assert.equal(device.statistics.liveBytes, 0);
    return result;
  } finally {
    await device.dispose();
  }
}

/** The kernels and the reference on one frame pair: equal bit for bit. */
async function compare(device, second, pts, opts = options, extra = {}) {
  const started = performance.now(),
    gpu = await trackKltGpu(device, first, second, W, H, pts, opts, extra),
    ms = performance.now() - started,
    reference = trackKlt(
      buildKltPyramid(first, W, H, opts.levels),
      buildKltPyramid(second, W, H, opts.levels),
      pts,
      opts,
    );
  assert.ok(same(gpu.forward, reference.forward), 'forward records');
  assert.ok(same(gpu.back, reference.back), 'backward records');
  assert.ok(same(gpu.records, reference.records), 'checked records');
  return { gpu, reference, ms };
}

test('kltPyrDown and kltScharr on the CPU tier equal the reference bit for bit', async () => {
  await withDevice(async (device) => {
    const second = renderScene(scene, { dx: 2.5, dy: -1.5 }),
      { gpu } = await compare(device, second, points, options, { readPyramids: true }),
      a = buildKltPyramid(first, W, H, options.levels),
      b = buildKltPyramid(second, W, H, options.levels);
    assert.ok(same(gpu.pyramids.gray[0], a.gray));
    assert.ok(same(gpu.pyramids.gray[1], b.gray));
    assert.ok(same(gpu.pyramids.grad[0], a.grad));
    assert.ok(same(gpu.pyramids.grad[1], b.grad));
  });
});

test('kltTrack recovers the translations of section 7.1 and equals the reference', async () => {
  await withDevice(async (device) => {
    const rows = [];
    for (const [dx, dy, bound] of [
      [0.3, 0.2, 0.05],
      [3.7, -2.1, 0.05],
      [12.5, 9.0, 0.1],
      [30, -24, 0.1],
    ]) {
      const warp = { dx, dy },
        second = renderScene(bigScene, warp),
        started = performance.now(),
        gpu = await trackKltGpu(device, bigFirst, second, BW, BH, bigPoints, bigOptions),
        ms = performance.now() - started,
        reference = trackKlt(
          buildKltPyramid(bigFirst, BW, BH, bigOptions.levels),
          buildKltPyramid(second, BW, BH, bigOptions.levels),
          bigPoints,
          bigOptions,
        );
      assert.ok(same(gpu.records, reference.records));
      assert.ok(same(gpu.forward, reference.forward));
      const { error, alive } = flowErrors(gpu.records, bigPoints, warp, BW, BH),
        kept = Array.from(error).filter((_, k) => alive[k]);
      rows.push({
        dx,
        dy,
        ms: Math.round(ms),
        alive: kept.length,
        max: +Math.max(...kept).toFixed(4),
      });
      assert.equal(kept.length, bigPoints.length / 2, `shift (${dx}, ${dy}): every track is kept`);
      assert.ok(Math.max(...kept) <= bound, `shift (${dx}, ${dy}): max error ${Math.max(...kept)}`);
    }
    console.log(JSON.stringify(rows));
  });
});

test('kltTrack: one level cannot reach 30 px (proves the instrument)', async () => {
  await withDevice(async (device) => {
    const warp = { dx: 30, dy: -24 },
      one = kltOptionsFor(BW, { levels: 1 }),
      second = renderScene(bigScene, warp),
      gpu = await trackKltGpu(device, bigFirst, second, BW, BH, bigPoints, one),
      { error, alive } = flowErrors(gpu.records, bigPoints, warp, BW, BH),
      kept = Array.from(error).filter((_, k) => alive[k]);
    assert.ok(kept.length < bigPoints.length / 2 || Math.max(...kept) > 0.1);
  });
});

test('kltTrack on rotations, scale and brightness equals the reference', async () => {
  await withDevice(async (device) => {
    const rows = [];
    for (const warp of [{ degrees: 5 }, { scale: 1.05 }, { gain: 1.1, offset: 0.05 }]) {
      const { gpu } = await compare(device, renderScene(scene, warp), points),
        { error, alive } = flowErrors(gpu.records, points, warp, W, H),
        kept = Array.from(error).filter((_, k) => alive[k]);
      rows.push({ warp, alive: kept.length, max: +Math.max(...kept).toFixed(3) });
      assert.ok(kept.length >= 10, `${JSON.stringify(warp)}: ${kept.length} kept`);
    }
    console.log(JSON.stringify(rows));
    // The plain SSD form on the CPU tier is the same code path as the reference.
    const plain = kltOptionsFor(W, { photometric: 0 });
    await compare(device, renderScene(scene, { gain: 1.1, offset: 0.05 }), points, plain);
  });
});

test('kltTrack and kltCheck: failure cases end with the same reasons as the reference', async () => {
  await withDevice(async (device) => {
    const a = Float32Array.from(first),
      b = Float32Array.from(renderScene(scene, { dx: 2, dy: 1 }));
    for (const f of [a, b])
      for (let y = 60; y < 110; y++) for (let x = 40; x < 90; x++) f[y * W + x] = 0.5;
    for (const [f, shift] of [
      [a, 0],
      [b, 2],
    ]) {
      for (let y = 20; y < 80; y++) for (let x = 180; x < 260; x++) f[y * W + x] = 0.3;
      for (let y = 20; y < 80; y++) f[y * W + 220 + shift] = 0.9;
    }
    for (let y = 120; y < 150; y++)
      for (let x = 252; x < 280; x++)
        b[y * W + x] = 0.5 + 0.3 * Math.sin(x * 0.9 + y * 1.7) * Math.cos(y * 0.5);
    const pts = Float32Array.from([
        65,
        85,
        220,
        50,
        3,
        100,
        W - 4,
        100,
        100,
        3,
        250,
        135,
        150,
        150,
      ]),
      gpu = await trackKltGpu(device, a, b, W, H, pts, options),
      reference = trackKlt(
        buildKltPyramid(a, W, H, options.levels),
        buildKltPyramid(b, W, H, options.levels),
        pts,
        options,
      ),
      reason = Array.from(gpu.records)
        .filter((_, i) => i % 4 === 3)
        .map((s) => (s >> 1) & 15);
    console.log(JSON.stringify({ reason }));
    assert.ok(same(gpu.records, reference.records));
    assert.deepEqual(reason.slice(0, 5), [
      KLT_END.lowStructure,
      KLT_END.lowStructure,
      KLT_END.outOfFrame,
      KLT_END.outOfFrame,
      KLT_END.outOfFrame,
    ]);
    assert.ok(
      [KLT_END.noConvergence, KLT_END.forwardBackward, KLT_END.residual].includes(reason[5]),
      `occlusion: reason ${reason[5]}`,
    );
    assert.equal(reason[6], KLT_END.alive);
    assert.equal(gpu.records[4 * 6 + 3] & 1, 1);
  });
});

test('kltTrack and kltCheck: chunks of 3 tracks give the same bits as one chunk', async () => {
  await withDevice(async (device) => {
    const second = renderScene(scene, { dx: 3.1, dy: 1.2, degrees: 1 }),
      whole = await trackKltGpu(device, first, second, W, H, points, options),
      chunked = await trackKltGpu(device, first, second, W, H, points, options, {
        chunkTracks: 3,
      }),
      calls = (report, name) =>
        report.groups.flatMap((group) => group.calls).filter((call) => call === name).length,
      n = points.length / 2;
    assert.ok(same(chunked.forward, whole.forward));
    assert.ok(same(chunked.back, whole.back));
    assert.ok(same(chunked.records, whole.records));
    // 12 tracks: one dispatch of each kernel for the whole run, 4 chunks for the chunked run.
    assert.equal(calls(whole.report, 'kltTrack'), 2);
    assert.equal(calls(chunked.report, 'kltTrack'), 2 * Math.ceil(n / 3));
    assert.equal(calls(chunked.report, 'kltCheck'), Math.ceil(n / 3));
    // A chunk of 5 does not divide 12: the last chunk holds 2 tracks, and the lanes past it idle.
    const odd = await trackKltGpu(device, first, second, W, H, points, options, { chunkTracks: 5 });
    assert.ok(same(odd.records, whole.records));
  });
});

test('klt.shade.ts follows the rules of section 4.3', () => {
  const source = readFileSync(new URL('../../src/gpu/klt.shade.ts', import.meta.url), 'utf8')
    .split('\n')
    .filter((line) => !line.trim().startsWith('//') && !line.trim().startsWith('*'))
    .join('\n');
  // No `0u` literal, and no unsigned literal of any value.
  assert.doesNotMatch(source, /\b\d+u\b/);
  // An idle lane takes the last element and writes nothing; no kernel returns early.
  assert.doesNotMatch(source, /\breturn\s*;/);
  assert.doesNotMatch(source, /\bbreak\b/);
});

test('kltTrack rate on the CPU tier', async () => {
  await withDevice(async (device) => {
    const second = renderScene(scene, { dx: 3.7, dy: -2.1 }),
      started = performance.now(),
      result = await trackKltGpu(device, first, second, W, H, points, options),
      ms = performance.now() - started;
    console.log(
      JSON.stringify({
        tracks: points.length / 2,
        ms: Math.round(ms),
        msPerTrackDirection: +(ms / (points.length / 2 / 1) / 2).toFixed(1),
        dispatches: result.report.groups.reduce((n, group) => n + group.calls.length, 0),
      }),
    );
  });
});
