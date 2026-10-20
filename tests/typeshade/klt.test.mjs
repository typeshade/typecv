// Verifies: docs/klt-tracking.md step 2 (sections 4.2, 4.3, 4.5 and 7.1 on the CPU tier).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  ShadeDevice,
  KLT_END,
  buildKltPyramid,
  KLT_DEFAULT_CAPACITY,
  KLT_MAX_CAPACITY,
  createKltTracker,
  extractSiftGpuResident,
  kltCellGrid,
  kltEigenMapReference,
  kltLayout,
  kltOptionsFor,
  kltSelectReference,
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

// Step 4: selection and the tracker (sections 2.2, 3.4, 4.1, 4.2 and 5.1) on the CPU tier.

const D = options.minDistancePx;
const emptyCells = () => {
  const { cellsX, cellsY } = kltCellGrid(W, H, D);
  return new Float32Array(2 * cellsX * cellsY).fill(-1);
};
const toRgba = (gray) => {
  const rgba = new Uint8Array(gray.length * 4);
  for (let i = 0; i < gray.length; i++) {
    const v = Math.round(gray[i] * 255);
    rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = v;
    rgba[4 * i + 3] = 255;
  }
  return rgba;
};
/** `rgbaToGray` in f32, in the order of the kernel. */
const grayOfRgba = (rgba) => {
  const f = Math.fround,
    out = new Float32Array(rgba.length / 4);
  for (let i = 0; i < out.length; i++)
    out[i] = f(
      f(
        f(f(f(0.299) * rgba[4 * i]) + f(f(0.587) * rgba[4 * i + 1])) +
          f(f(0.114) * rgba[4 * i + 2]),
      ) / 255,
    );
  return out;
};
const closest = (a, b, same = false) => {
  let least = Infinity;
  for (let i = 0; i < a.x.length; i++)
    for (let j = 0; j < b.x.length; j++)
      if (!(same && i === j)) least = Math.min(least, Math.hypot(a.x[i] - b.x[j], a.y[i] - b.y[j]));
  return least;
};

test('the first push selects the points of the reference, kltEigenMap and kltSelect equal it', async () => {
  await withDevice(async (device) => {
    const tracker = await createKltTracker(device, W, H, { options: { capacity: 1000 } });
    assert.equal(device.statistics.liveBytes, tracker.residentBytes);
    const step = await tracker.push({ gray: first });
    const eig = await tracker.readEigenMap(),
      pyramid = buildKltPyramid(first, W, H, options.levels),
      expectedEig = kltEigenMapReference(pyramid.grad, W, H, options.halfWindow),
      expected = kltSelectReference(expectedEig, W, H, options, emptyCells()),
      tracks = tracker.tracks();
    assert.ok(same(eig, expectedEig), 'the lambda-min map');
    assert.equal(step.maxEigen, expected.max);
    assert.equal(step.selected, expected.picks.length);
    assert.equal(step.appended, expected.picks.length);
    assert.equal(tracks.count, expected.picks.length);
    assert.ok(tracks.count > 300, `${tracks.count} points`);
    for (let k = 0; k < tracks.count; k++) {
      assert.equal(tracks.x[k], expected.picks[k].x);
      assert.equal(tracks.y[k], expected.picks[k].y);
      assert.equal(tracks.seedKeypoint[k], 0xffffffff);
      assert.equal(tracks.age[k], 0);
    }
    // minDistancePx: no two points closer than the distance, and none within the margin.
    assert.ok(closest(tracks, tracks, true) >= D);
    assert.ok(tracks.x.every((x, k) => x >= options.halfWindow && x + options.halfWindow < W));
    assert.ok(tracks.y.every((y) => y >= options.halfWindow && y + options.halfWindow < H));
    assert.equal(step.eigenChunks.length, 1);
    // The memory formula of section 4.1 holds after the push, and the arrays go back on dispose.
    assert.equal(device.statistics.liveBytes, tracker.residentBytes);
    await tracker.dispose();
    assert.equal(device.statistics.liveBytes, 0);
  });
});

test('selection: chunks of kltEigenMap give the same bits, and the capacity keeps the strongest', async () => {
  await withDevice(async (device) => {
    const whole = await createKltTracker(device, W, H, { options: { capacity: 1000 } }),
      chunked = await createKltTracker(device, W, H, {
        options: { capacity: 1000 },
        chunkTracks: 5000,
      }),
      small = await createKltTracker(device, W, H, { options: { capacity: 100 } });
    const a = await whole.push({ gray: first }),
      b = await chunked.push({ gray: first }),
      c = await small.push({ gray: first });
    assert.equal(a.eigenChunks.length, 1);
    assert.equal(b.eigenChunks.length, Math.ceil((W * H) / 5000));
    assert.ok(same(await chunked.readEigenMap(), await whole.readEigenMap()));
    assert.ok(same(chunked.tracks().x, whole.tracks().x));
    assert.ok(same(chunked.tracks().y, whole.tracks().y));
    // A capacity below the picks keeps the strongest ones.
    assert.equal(c.selected, a.selected);
    assert.equal(c.appended, 100);
    const eig = await whole.readEigenMap(),
      kept = small.tracks(),
      scoreOf = (x, y) => eig[y * W + x],
      weakest = Math.min(...Array.from(kept.x, (x, k) => scoreOf(x, kept.y[k]))),
      all = whole.tracks(),
      stronger = Array.from(all.x, (x, k) => scoreOf(x, all.y[k])).filter((v) => v > weakest);
    assert.equal(stronger.length, 99);
    for (const t of [whole, chunked, small]) await t.dispose();
    assert.equal(device.statistics.liveBytes, 0);
  });
});

test('the tracker follows a sequence, ages tracks out and refreshes at a keyframe', async () => {
  await withDevice(async (device) => {
    const tracker = await createKltTracker(device, W, H, { options: { capacity: 40 } }),
      shift = { dx: 3.7, dy: -2.1 },
      frame = (i) => renderScene(scene, { dx: shift.dx * i, dy: shift.dy * i });
    const s0 = await tracker.push({ gray: frame(0) });
    assert.equal(s0.tracked, 0);
    assert.equal(s0.alive, 40);
    const seeds = tracker.tracks();
    const s1 = await tracker.push({ gray: frame(1) });
    const s2 = await tracker.push({ gray: frame(2) });
    assert.equal(s1.tracked, 40);
    assert.ok(s2.alive >= 38, `${s2.alive} of 40 alive`);
    assert.equal(s2.selected, 0, 'no selection without a keyframe');
    const at2 = tracker.tracks();
    for (let k = 0; k < at2.count; k++) {
      const origin = seeds.id.indexOf(at2.id[k]);
      assert.ok(origin >= 0);
      assert.ok(
        Math.hypot(
          at2.x[k] - (seeds.x[origin] + 2 * shift.dx),
          at2.y[k] - (seeds.y[origin] + 2 * shift.dy),
        ) < 0.1,
      );
      assert.equal(at2.age[k], 2);
      assert.equal(at2.seedX[k], seeds.x[origin]);
    }
    // A keyframe selects in the cells with no track, at least minDistancePx from the tracks.
    const s3 = await tracker.push({ gray: frame(3) }, { keyframe: true });
    assert.equal(s3.appended, 40 - (s3.tracked - s3.ended.reduce((a, b) => a + b, 0)));
    const at3 = tracker.tracks(),
      born = { x: [], y: [] },
      old = { x: [], y: [] };
    for (let k = 0; k < at3.count; k++) {
      const target = at3.seedFrame[k] === 3 ? born : old;
      target.x.push(at3.x[k]);
      target.y.push(at3.y[k]);
    }
    assert.equal(new Set(at3.id).size, at3.count);
    if (born.x.length && old.x.length) assert.ok(closest(born, old) >= D);
    await tracker.dispose();

    // The age bound: with maxAgeFrames 1 a track of age 2 ends with the reason `age`.
    const aging = await createKltTracker(device, W, H, {
      options: { capacity: 6, maxAgeFrames: 1 },
    });
    await aging.push({ gray: frame(0) });
    const a1 = await aging.push({ gray: frame(1) });
    assert.equal(a1.ended[KLT_END.age], 0);
    const a2 = await aging.push({ gray: frame(2) });
    assert.equal(a2.ended[KLT_END.age], a2.tracked);
    assert.equal(a2.alive, 0);
    await aging.dispose();
  });
});

test('the RGBA path converts on the device, and the SIFT gray serves as level 0', async () => {
  await withDevice(async (device) => {
    const gray = renderScene(scene),
      rgba = toRgba(gray),
      opts = { capacity: 12 };
    const viaGray = await createKltTracker(device, W, H, { options: opts }),
      viaRgba = await createKltTracker(device, W, H, { options: opts }),
      viaSift = await createKltTracker(device, W, H, { options: opts });
    await viaGray.push({ gray: grayOfRgba(rgba) });
    await viaRgba.push({ rgba });
    assert.ok(same(await viaRgba.readEigenMap(), await viaGray.readEigenMap()));
    assert.ok(same(viaRgba.tracks().x, viaGray.tracks().x));

    // SIFT keeps the gray before its own use. Its keypoints do not change.
    const plain = await extractSiftGpuResident(device, rgba, W, H, { maxFeatures: 256 }),
      kept = await extractSiftGpuResident(device, rgba, W, H, {
        maxFeatures: 256,
        grayForKlt: { levels: options.levels },
      });
    assert.equal(plain.gray, undefined);
    assert.equal(kept.gray.levels, options.levels);
    assert.equal(kept.count, plain.count);
    assert.ok(same(await kept.readDetails(), await plain.readDetails()));
    const before = device.statistics.liveBytes;
    await viaSift.push({ sift: kept });
    assert.ok(same(await viaSift.readEigenMap(), await viaGray.readEigenMap()));
    assert.ok(same(viaSift.tracks().x, viaGray.tracks().x));
    // The buffer moved to the tracker: the tracker's own level 0 buffer went back, and a second
    // take throws. The result's dispose leaves the gray, so the next push still reads it.
    assert.equal(device.statistics.liveBytes, before - 4 * kltLayout(W, H, options.levels).total);
    assert.throws(() => kept.gray.take(), /not available/);
    await kept.dispose();
    const next = await viaSift.push({ gray: renderScene(scene, { dx: 2.5, dy: 1.5 }) });
    assert.equal(next.tracked, 12);
    assert.ok(next.alive >= 11, `${next.alive} of 12 alive`);
    // A frame of another size, or a result without the gray, is refused.
    await assert.rejects(viaSift.push({ sift: plain }), /grayForKlt/);
    await Promise.all([plain.dispose(), viaGray.dispose(), viaRgba.dispose(), viaSift.dispose()]);
  });
});

test('klt tracker: an untaken SIFT gray is freed with the result', async () => {
  await withDevice(async (device) => {
    const rgba = toRgba(renderScene(scene)),
      kept = await extractSiftGpuResident(device, rgba, W, H, {
        maxFeatures: 64,
        grayForKlt: { levels: options.levels },
      });
    await kept.dispose();
  });
});

test('the tracker capacity defaults to 1024 and reaches 8192', async () => {
  await withDevice(async (device) => {
    assert.equal(KLT_DEFAULT_CAPACITY, 1024);
    assert.equal(KLT_MAX_CAPACITY, 8192);
    assert.equal(kltOptionsFor(640).capacity, 1024);
    assert.equal(kltOptionsFor(1280).capacity, 1024);
    const normal = await createKltTracker(device, W, H);
    assert.equal(normal.options.capacity, 1024);
    const full = await createKltTracker(device, W, H, { options: { capacity: 8192 } });
    assert.equal(full.options.capacity, 8192);
    // Four 16-byte record arrays of the capacity are the difference of the two trackers.
    assert.equal(full.residentBytes - normal.residentBytes, 4 * 16 * (8192 - 1024));
    await Promise.all([normal.dispose(), full.dispose()]);
    for (const capacity of [0, 8193, 1.5])
      await assert.rejects(createKltTracker(device, W, H, { options: { capacity } }), /capacity/);
  });
});
