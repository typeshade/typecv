// Verifies: docs/klt-tracking.md steps 3 and 4 (sections 4.1, 4.4, 5.1 and 7.2): the kernels of `trackKltGpu` on
// WebGPU agree with the CPU reference, the chunk key `kltRoundtrip` and its warm step run, and the
// SwiftShader rate of one point's forward/backward/check roundtrip is recorded.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  buildKltPyramid,
  kltCellGrid,
  kltEigenMapReference,
  kltOptionsFor,
  kltSelectReference,
  trackKlt,
  trackKltReference,
} from '../../dist/internal/gpu.js';
import { flowErrors, gridPoints, renderScene } from '../klt-cases.ts';
import { inBrowser } from './browser.mjs';
import { assertKltRateCache } from './klt-rate-cache.mjs';

/** Section 7.2: the share of the tracks within this many pixels of the CPU reference. */
const AGREE_PX = 1e-3;
const AGREE_SHARE = 0.99;
/**
 * The tracks of the rate case, a multiple of 64. The capacity of section 4.1 is 8192, and the
 * environment variable measures that size once for the record.
 */
const RATE_TRACKS = Number(process.env.TYPECV_KLT_RATE_TRACKS ?? 2048);
/** The tracks of the rate case that the CPU reference checks. */
const RATE_CHECKED = 128;
/** The default grouped roundtrip floor: two 64-lane workgroups. */
const ROUNDTRIP_MIN_TRACKS = 128;

const BW = 640,
  BH = 360,
  W = 320,
  H = 200;
const big = { width: BW, height: BH };
const small = { width: W, height: H };
// The 1280 px defaults of section 2.4 (6 levels, a window of 21 px) are inferred from the 640 px
// ones. This case checks them with twice the largest shift of 7.1.
const huge = { width: 1280, height: 720 };
const bigFirst = renderScene(big),
  rateSecond = renderScene(big, { dx: 3.7, dy: -2.1 }),
  ratePoints = gridPoints(BW, BH, 40, 40, 64, RATE_TRACKS / 64),
  smallFirst = renderScene(small),
  hugeFirst = renderScene(huge),
  hugePoints = gridPoints(1280, 720, 240, 200, 4, 3),
  bigPoints = gridPoints(BW, BH, 120, 100, 4, 3),
  smallPoints = gridPoints(W, H, 60, 50, 4, 3);

/** The pairs of 7.1 run on both. `chunkTracks` is set where the case also checks the chunks. */
const cases = [
  ...[
    [0.3, 0.2],
    [3.7, -2.1],
    [12.5, 9.0],
    [30, -24],
  ].map(([dx, dy]) => ({
    name: `shift ${dx},${dy}`,
    size: big,
    warp: { dx, dy },
    first: bigFirst,
    points: bigPoints,
    bound: Math.hypot(dx, dy) < 12 ? 0.05 : 0.1,
  })),
  {
    name: 'shift 60,-48 at 1280 px',
    size: huge,
    warp: { dx: 60, dy: -48 },
    first: hugeFirst,
    points: hugePoints,
    bound: 0.2,
  },
  ...[{ degrees: 5 }, { scale: 1.05 }, { gain: 1.1, offset: 0.05 }].map((warp) => ({
    name: `warp ${JSON.stringify(warp)}`,
    size: small,
    warp,
    first: smallFirst,
    points: smallPoints,
    bound: 0.5,
  })),
];

test('trackKltGpu on WebGPU agrees with the CPU reference and the chunk key kltRoundtrip runs', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'klt-webgpu-')),
    serve = {};
  try {
    const put = (name, frame) => {
      const file = join(dir, name);
      writeFileSync(file, Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength));
      serve[`/${name}`] = file;
    };
    put('big-first.bin', bigFirst);
    put('small-first.bin', smallFirst);
    put('huge-first.bin', hugeFirst);
    const seconds = cases.map((c, i) => {
      const second = renderScene(c.size, c.warp);
      put(`second-${i}.bin`, second);
      return second;
    });
    // The rate case: a grid of RATE_TRACKS points on the 640 x 360 frame, and a second frame.
    put('rate-second.bin', rateSecond);

    const { result, errors } = await inBrowser(
      async (input) => {
        const { ShadeDevice, kltOptionsFor, trackKltGpu } = await import('/dist/internal/gpu.js');
        const load = async (name) =>
          new Float32Array(await (await fetch(`/${name}`)).arrayBuffer());
        const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        const plain = (r) => ({
          records: Array.from(r.records),
          forward: Array.from(r.forward),
          back: Array.from(r.back),
        });
        const calls = (report) =>
          report.groups.flatMap((g) => g.calls).filter((c) => c === 'kltTrack').length;
        try {
          const firsts = {
            big: await load('big-first.bin'),
            small: await load('small-first.bin'),
            huge: await load('huge-first.bin'),
          };
          const out = { cases: [] };
          for (const [i, c] of input.cases.entries()) {
            const options = kltOptionsFor(c.width),
              second = await load(`second-${i}.bin`),
              started = performance.now(),
              r = await trackKltGpu(
                device,
                firsts[c.size],
                second,
                c.width,
                c.height,
                new Float32Array(c.points),
                options,
              );
            out.cases.push({
              ms: performance.now() - started,
              calls: calls(r.report),
              dispatches: r.report.dispatches,
              ...plain(r),
            });
          }
          // Chunk invariance on WebGPU: the first case in chunks of 5 tracks. 12 tracks in
          // chunks of 5 leave 2 in the last, so 62 of its 64 lanes idle and must write nothing.
          {
            const c = input.cases[1],
              second = await load('second-1.bin'),
              r = await trackKltGpu(
                device,
                firsts.big,
                second,
                c.width,
                c.height,
                new Float32Array(c.points),
                kltOptionsFor(c.width),
                { chunkTracks: 5 },
              );
            out.chunked = { calls: calls(r.report), ...plain(r) };
          }
          out.liveBytes = device.statistics.liveBytes;
          return out;
        } finally {
          await device.dispose();
        }
      },
      {
        cases: cases.map((c) => ({
          size: c.size === big ? 'big' : c.size === huge ? 'huge' : 'small',
          width: c.size.width,
          height: c.size.height,
          points: Array.from(c.points),
        })),
        ratePoints: Array.from(ratePoints),
      },
      { timeoutMs: 600000, serve },
    );
    // The opt-in rate case on a device of its own, so `kltRoundtrip` is learned from chunks
    // of the real size (the 12-track cases above would teach it a rate dominated by the fixed
    // cost). The second call reuses the learned rate and completes any remaining timing samples.
    const { result: rateResult, errors: rateErrors } = await inBrowser(
      async (input) => {
        const { ShadeDevice, kltOptionsFor, trackKltGpu } = await import('/dist/internal/gpu.js');
        const load = async (name) =>
          new Float32Array(await (await fetch(`/${name}`)).arrayBuffer());
        const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        const timedChunks = [];
        const snapshotRate = () => {
          const rate = device.chunkRates.kltRoundtrip;
          return rate ? { ...rate } : null;
        };
        let sizing = { calls: 0, firstRate: null };
        const chunkSize = device.chunkSize.bind(device);
        device.chunkSize = (...args) => {
          if (args[0] === 'kltRoundtrip') {
            if (sizing.calls === 0) sizing.firstRate = snapshotRate();
            sizing.calls++;
          }
          return chunkSize(...args);
        };
        const plan = device.plan.bind(device);
        // Observe the existing fenced timing report; do not add timers, reads or submissions.
        device.plan = async (...args) => {
          const report = await plan(...args);
          if (report.chunk?.key === 'kltRoundtrip')
            timedChunks.push({ units: report.chunk.units, elapsedMs: report.chunk.elapsedMs });
          return report;
        };
        try {
          const first = await load('big-first.bin'),
            second = await load('rate-second.bin'),
            passes = [];
          for (let pass = 0; pass < 2; pass++) {
            const rateBefore = snapshotRate();
            const timedBefore = timedChunks.length;
            sizing = { calls: 0, firstRate: null };
            const started = performance.now(),
              r = await trackKltGpu(
                device,
                first,
                second,
                640,
                360,
                new Float32Array(input.ratePoints),
                kltOptionsFor(640),
                { batchPasses: true },
              );
            passes.push({
              ms: performance.now() - started,
              dispatches: r.report.dispatches,
              submissions: r.report.submissions,
              trackChunks: r.trackChunks,
              rates: Object.fromEntries(
                Object.entries(device.chunkRates).map(([key, rate]) => [key, { ...rate }]),
              ),
              rateBefore,
              firstSizingRate: sizing.firstRate,
              sizingCalls: sizing.calls,
              timedBefore,
              timedAfter: timedChunks.length,
              forward: Array.from(r.forward),
              records: Array.from(r.records),
            });
          }
          return {
            passes,
            timedChunks,
            timeBudgetMs: device.timeBudgetMs,
            chunkTimingSamples: device.chunkTimingSamples,
            liveBytes: device.statistics.liveBytes,
          };
        } finally {
          await device.dispose();
        }
      },
      { ratePoints: Array.from(ratePoints) },
      { timeoutMs: 1800000, serve },
    );
    assert.deepEqual(rateErrors, []);
    assert.equal(rateResult.liveBytes, 0);
    assert.deepEqual(errors, []);
    assert.equal(result.liveBytes, 0);

    // Agreement with the CPU reference, case by case (section 7.2).
    const rows = [];
    const agreement = (gpu, reference, count) => {
      let within = 0,
        sameStatus = 0;
      for (let k = 0; k < count; k++) {
        const dx = gpu[4 * k] - reference[4 * k],
          dy = gpu[4 * k + 1] - reference[4 * k + 1];
        if (Math.hypot(dx, dy) <= AGREE_PX) within++;
        if (gpu[4 * k + 3] === reference[4 * k + 3]) sameStatus++;
      }
      return { within: within / count, sameStatus: sameStatus / count };
    };
    cases.forEach((c, i) => {
      const o = kltOptionsFor(c.size.width),
        reference = trackKlt(
          buildKltPyramid(c.first, c.size.width, c.size.height, o.levels),
          buildKltPyramid(seconds[i], c.size.width, c.size.height, o.levels),
          c.points,
          o,
        ),
        n = c.points.length / 2,
        gpu = result.cases[i],
        a = agreement(gpu.records, reference.records, n),
        { error, alive } = flowErrors(
          Float32Array.from(gpu.records),
          c.points,
          c.warp,
          c.size.width,
          c.size.height,
        ),
        kept = Array.from(error).filter((_, k) => alive[k]);
      rows.push({
        case: c.name,
        within: a.within,
        sameStatus: a.sameStatus,
        kept: kept.length,
        max: +Math.max(...kept).toFixed(4),
        ms: Math.round(gpu.ms),
      });
      assert.ok(a.within >= AGREE_SHARE, `${c.name}: ${a.within} of the tracks within ${AGREE_PX}`);
      assert.equal(a.sameStatus, 1, `${c.name}: the status words`);
      assert.ok(kept.length >= 10, `${c.name}: ${kept.length} kept`);
      assert.ok(Math.max(...kept) <= c.bound, `${c.name}: max error ${Math.max(...kept)}`);
    });
    // Chunks of 5: the same bits as one chunk, and 3 chunks of kltTrack in each direction.
    assert.deepEqual(result.chunked.records, result.cases[1].records);
    assert.deepEqual(result.chunked.forward, result.cases[1].forward);
    assert.deepEqual(result.chunked.back, result.cases[1].back);
    assert.ok(result.chunked.calls > result.cases[1].calls, 'the chunks split the dispatches');

    // The rate case: the first RATE_CHECKED tracks against the reference (forward pass).
    const o = kltOptionsFor(640),
      from = buildKltPyramid(bigFirst, BW, BH, o.levels),
      to = buildKltPyramid(rateSecond, BW, BH, o.levels),
      start = new Float32Array(4 * RATE_TRACKS);
    for (let k = 0; k < RATE_TRACKS; k++) {
      start[4 * k] = ratePoints[2 * k];
      start[4 * k + 1] = ratePoints[2 * k + 1];
      start[4 * k + 3] = 1;
    }
    const reference = trackKltReference(from, to, start, o, { first: 0, end: RATE_CHECKED }),
      [cold, warm] = rateResult.passes,
      rateAgreement = agreement(cold.forward, reference, RATE_CHECKED),
      alive = cold.records.filter((v, i) => i % 4 === 3 && v & 1).length;
    assert.ok(rateAgreement.within >= AGREE_SHARE, `rate case: ${rateAgreement.within} within`);
    assert.equal(rateAgreement.sameStatus, 1);
    // The two passes track the same points on the same device: the same bits, whatever the chunks.
    assert.deepEqual(warm.records, cold.records);
    assert.ok(alive >= 0.9 * RATE_TRACKS, `${alive} of ${RATE_TRACKS} tracks alive`);

    // Section 4.4: kltRoundtrip times one point's forward/backward/check chain. The first chunk
    // has the minimum size, and the logical report lists each point once in each direction.
    const rate = warm.rates.kltRoundtrip;
    assert.ok(
      rate && rate.chunks === rateResult.chunkTimingSamples,
      'kltRoundtrip has a settled rate',
    );
    assert.ok(rate.msPerUnit > 0);
    assert.equal(rateResult.timedChunks.length, rate.chunks);
    const timedDetail = JSON.stringify(rateResult.timedChunks);
    for (const pass of [cold, warm])
      assert.equal(
        pass.trackChunks.reduce((a, b) => a + b, 0),
        2 * RATE_TRACKS,
      );
    assert.equal(
      cold.trackChunks[0],
      ROUNDTRIP_MIN_TRACKS,
      'the first roundtrip chunk has the two-workgroup minimum size',
    );
    assert.ok(cold.trackChunks.length > 2, 'the cold pass ran as several chunks');
    // After the rate has settled, a chunk larger than the minimum must fit the time budget.
    // Point cost can rise during tracking, so a warm pass may correctly use more chunks.
    // Verify the retained rate at entry and its first actual sizing call instead.
    assertKltRateCache(rateResult.passes, rateResult.timedChunks, rateResult.chunkTimingSamples);
    for (const size of warm.trackChunks.slice(0, -1)) assert.equal(size % 64, 0);
    for (const size of warm.trackChunks.slice(1))
      assert.ok(
        size <= ROUNDTRIP_MIN_TRACKS || size * rate.msPerUnit <= rateResult.timeBudgetMs,
        `a chunk of ${size} point roundtrips is expected to take ${size * rate.msPerUnit} ms`,
      );
    // Chunk sizing controls the size of a chunk and not the speed of the runner. The minimum
    // chunk has 128 point roundtrips, so a busy CI runner can push it past the budget. A smaller
    // final remainder is also allowed.
    // So the longest timed chunk must stay within the budget whenever the device ran any chunk
    // larger than the minimum (the sizing then chose a size from the rate, and the rate must hold).
    // When every chunk had the minimum size, only a hang guard of 4 budgets applies.
    const largest = Math.max(...cold.trackChunks, ...warm.trackChunks);
    if (largest > ROUNDTRIP_MIN_TRACKS)
      assert.ok(
        rate.longestMs <= rateResult.timeBudgetMs,
        `the longest timed chunk took ${rate.longestMs} ms, the budget is ${rateResult.timeBudgetMs} ms, with chunks of up to ${largest} point roundtrips; timed chunks ${timedDetail}`,
      );
    else
      assert.ok(
        rate.longestMs <= 4 * rateResult.timeBudgetMs,
        `a chunk of the minimum size took ${rate.longestMs} ms, the budget is ${rateResult.timeBudgetMs} ms; timed chunks ${timedDetail}`,
      );
    console.log(
      JSON.stringify({
        agreement: rows,
        chunked: { calls: result.chunked.calls, single: result.cases[1].calls },
        rate: {
          tracks: RATE_TRACKS,
          alive,
          timeBudgetMs: rateResult.timeBudgetMs,
          chunkTimingSamples: rateResult.chunkTimingSamples,
          msPerPointRoundtrip: +rate.msPerUnit.toPrecision(4),
          pointRoundtripsPerSecond: Math.round(1000 / rate.msPerUnit),
          longestMs: Math.round(rate.longestMs),
          timedChunks: rateResult.timedChunks,
          warmRateBefore: warm.rateBefore,
          warmFirstSizingRate: warm.firstSizingRate,
          cold: {
            ms: Math.round(cold.ms),
            dispatches: cold.dispatches,
            submissions: cold.submissions,
            trackChunks: cold.trackChunks,
          },
          settled: {
            ms: Math.round(warm.ms),
            dispatches: warm.dispatches,
            submissions: warm.submissions,
            trackChunks: warm.trackChunks,
          },
          checked: { tracks: RATE_CHECKED, ...rateAgreement },
        },
      }),
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// Step 4: the tracker on WebGPU. Selection and tracking agree with the CPU reference, the memory
// formula of section 4.1 holds, and the SIFT gray serves as level 0. The environment variable
// `TYPECV_KLT_SWEEP=1` also measures the time of a push for several capacities (the track count
// of the lab, Step 4 record).

const TRACKER_CAPACITY = 512;
const SWEEP = process.env.TYPECV_KLT_SWEEP === '1';
const SWEEP_CAPACITIES = [256, 512, 1024, 2048];

test('the KLT tracker on WebGPU agrees with the CPU reference and holds the memory formula', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'klt-tracker-webgpu-')),
    serve = {},
    shift = { dx: 3.7, dy: -2.1 },
    frames = [0, 1, 2].map((i) => renderScene(big, { dx: shift.dx * i, dy: shift.dy * i })),
    siftGray = renderScene(small),
    siftRgba = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    siftRgba[4 * i] = siftRgba[4 * i + 1] = siftRgba[4 * i + 2] = Math.round(siftGray[i] * 255);
    siftRgba[4 * i + 3] = 255;
  }
  try {
    const put = (name, array) => {
      const file = join(dir, name);
      writeFileSync(file, Buffer.from(array.buffer, array.byteOffset, array.byteLength));
      serve[`/${name}`] = file;
    };
    frames.forEach((frame, i) => put(`tracker-${i}.bin`, frame));
    put('sift-rgba.bin', siftRgba);
    const { result, errors } = await inBrowser(
      async (input) => {
        const { ShadeDevice, createKltTracker, extractSiftGpuResident, kltResidentBytes } =
          await import('/dist/internal/gpu.js');
        const load = async (name, Type) => new Type(await (await fetch(`/${name}`)).arrayBuffer());
        const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 }),
          out = { steps: [], sweep: [] };
        try {
          const tracker = await createKltTracker(device, 640, 360, {
            options: { capacity: input.capacity },
          });
          out.residentBytes = tracker.residentBytes;
          out.formulaBytes = kltResidentBytes(640, 360, tracker.options);
          out.liveAtStart = device.statistics.liveBytes;
          for (let i = 0; i < 3; i++) {
            const frame = await load(`tracker-${i}.bin`, Float32Array),
              step = await tracker.push({ gray: frame }, { keyframe: i === 0 });
            out.steps.push(step);
            const t = tracker.tracks();
            out[`tracks${i}`] = {
              x: Array.from(t.x),
              y: Array.from(t.y),
              id: Array.from(t.id),
              seedFrame: Array.from(t.seedFrame),
            };
            if (i === 0) out.eig = Array.from(await tracker.readEigenMap());
          }
          out.liveAfterPushes = device.statistics.liveBytes;
          await tracker.dispose();
          out.liveAfterDispose = device.statistics.liveBytes;

          // The SIFT gray as level 0 on WebGPU.
          const rgba = await load('sift-rgba.bin', Uint8Array),
            plainTracker = await createKltTracker(device, 320, 200, {
              options: { capacity: 64 },
            }),
            siftTracker = await createKltTracker(device, 320, 200, { options: { capacity: 64 } }),
            sift = await extractSiftGpuResident(device, rgba, 320, 200, {
              maxFeatures: 128,
              grayForKlt: { levels: siftTracker.options.levels },
            });
          await plainTracker.push({ rgba });
          await siftTracker.push({ sift });
          await sift.dispose();
          out.sift = {
            count: sift.count,
            plain: Array.from(plainTracker.tracks().x),
            adopted: Array.from(siftTracker.tracks().x),
          };
          await plainTracker.dispose();
          await siftTracker.dispose();
          out.liveAfterSift = device.statistics.liveBytes;

          if (input.sweep) {
            const second = await load('tracker-1.bin', Float32Array),
              first = await load('tracker-0.bin', Float32Array);
            for (const minChunkTracks of [64, 256])
              for (const capacity of input.capacities) {
                const t = await createKltTracker(device, 640, 360, {
                  options: { capacity },
                  minChunkTracks,
                  batchPasses: true,
                });
                const a = await t.push({ gray: first }, { keyframe: true }),
                  b = await t.push({ gray: second });
                out.sweep.push({
                  minChunkTracks,
                  capacity,
                  selectMs: a.msSelect,
                  trackMs: b.msTrack,
                  tracked: b.tracked,
                  alive: b.alive,
                  chunks: b.trackChunks.length,
                  longest: Math.max(...b.trackChunks),
                  msPerPointRoundtrip: device.chunkRates.kltRoundtrip?.msPerUnit,
                  longestMs: device.chunkRates.kltRoundtrip?.longestMs,
                });
                await t.dispose();
              }
          }
          out.liveBytes = device.statistics.liveBytes;
          out.timeBudgetMs = device.timeBudgetMs;
          return out;
        } finally {
          await device.dispose();
        }
      },
      { capacity: TRACKER_CAPACITY, sweep: SWEEP, capacities: SWEEP_CAPACITIES },
      { timeoutMs: 3600000, serve },
    );
    assert.deepEqual(errors, []);
    // Section 4.1: the formula equals the live bytes of the device at start and after the pushes.
    assert.equal(result.residentBytes, result.formulaBytes);
    assert.equal(result.liveAtStart, result.residentBytes);
    assert.equal(result.liveAfterPushes, result.residentBytes);
    assert.equal(result.liveAfterDispose, 0);
    assert.equal(result.liveAfterSift, 0);
    assert.equal(result.liveBytes, 0);

    // Selection against the reference, on the lambda-min map of the device.
    const o = kltOptionsFor(BW),
      pyramid = buildKltPyramid(frames[0], BW, BH, o.levels),
      referenceEig = kltEigenMapReference(pyramid.grad, BW, BH, o.halfWindow),
      { cellsX, cellsY } = kltCellGrid(BW, BH, o.minDistancePx),
      expected = kltSelectReference(
        referenceEig,
        BW,
        BH,
        o,
        new Float32Array(2 * cellsX * cellsY).fill(-1),
      );
    let eigWithin = 0;
    for (let i = 0; i < referenceEig.length; i++)
      if (Math.abs(result.eig[i] - referenceEig[i]) <= 1e-6 * Math.max(1e-3, referenceEig[i]))
        eigWithin++;
    assert.ok(eigWithin / referenceEig.length >= 0.999, `lambda-min within: ${eigWithin}`);
    // The tracker keeps the strongest `capacity` of the picks; compare with the strongest of the
    // reference.
    const strongest = [...expected.picks]
        .sort((a, b) => b.score - a.score || 0)
        .slice(0, TRACKER_CAPACITY),
      key = (x, y) => `${x},${y}`,
      wanted = new Set(strongest.map((p) => key(p.x, p.y))),
      got = result.tracks0.x.map((x, k) => key(x, result.tracks0.y[k])),
      overlap = got.filter((k) => wanted.has(k)).length;
    assert.equal(result.steps[0].appended, TRACKER_CAPACITY);
    assert.equal(result.steps[0].selected, expected.picks.length);
    assert.ok(overlap / TRACKER_CAPACITY >= 0.99, `${overlap} of ${TRACKER_CAPACITY} picks agree`);
    assert.ok(Math.abs(result.steps[0].maxEigen - expected.max) <= 1e-6 * expected.max);

    // Tracking against the reference, from the device's own picks.
    const picks = Float32Array.from(result.tracks0.x.flatMap((x, k) => [x, result.tracks0.y[k]])),
      reference = trackKlt(pyramid, buildKltPyramid(frames[1], BW, BH, o.levels), picks, o),
      step1 = result.steps[1],
      byId = new Map(result.tracks1.id.map((id, k) => [id, k]));
    let tracked = 0,
      agree = 0,
      referenceAlive = 0;
    for (let k = 0; k < picks.length / 2; k++) {
      if (!(reference.records[4 * k + 3] & 1)) continue;
      referenceAlive++;
      const at = byId.get(result.tracks0.id[k]);
      if (at === undefined) continue;
      tracked++;
      if (
        Math.hypot(
          result.tracks1.x[at] - reference.records[4 * k],
          result.tracks1.y[at] - reference.records[4 * k + 1],
        ) <= AGREE_PX
      )
        agree++;
    }
    assert.equal(step1.tracked, TRACKER_CAPACITY);
    assert.ok(agree / referenceAlive >= AGREE_SHARE, `${agree} of ${referenceAlive} agree`);
    assert.ok(tracked / referenceAlive >= AGREE_SHARE, `${tracked} of ${referenceAlive} alive`);
    assert.ok(step1.alive >= 0.9 * TRACKER_CAPACITY, `${step1.alive} alive`);
    // The second push, a keyframe in no case, and the third, still tracking.
    assert.equal(result.steps[2].selected, 0);
    assert.ok(result.steps[2].alive >= 0.85 * TRACKER_CAPACITY);
    // The SIFT gray: the tracker that took it selects exactly what the RGBA path selects.
    assert.ok(result.sift.count > 0);
    assert.deepEqual(result.sift.adopted, result.sift.plain);

    console.log(
      JSON.stringify({
        tracker: {
          capacity: TRACKER_CAPACITY,
          residentBytes: result.residentBytes,
          eigWithin: eigWithin / referenceEig.length,
          picks: expected.picks.length,
          overlap,
          step0: {
            msSelect: Math.round(result.steps[0].msSelect),
            chunks: result.steps[0].eigenChunks,
          },
          step1: {
            msTrack: Math.round(step1.msTrack),
            alive: step1.alive,
            trackChunks: step1.trackChunks,
            agree: agree / referenceAlive,
          },
          step2: { msTrack: Math.round(result.steps[2].msTrack), alive: result.steps[2].alive },
        },
        sweep: result.sweep.map((r) => ({
          ...r,
          selectMs: Math.round(r.selectMs),
          trackMs: Math.round(r.trackMs),
          msPerPointRoundtrip: r.msPerPointRoundtrip && +r.msPerPointRoundtrip.toPrecision(3),
          pointRoundtripsPerSecond:
            r.msPerPointRoundtrip && Math.round(1000 / r.msPerPointRoundtrip),
          longestMs: r.longestMs && Math.round(r.longestMs),
        })),
      }),
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// Verifies: docs/klt-tracking.md step 5 (sections 3.3 and 5.1): the join on WebGPU.
test('the KLT join on WebGPU matches the tracks to the moved keypoints and seeds the rest', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'klt-join-webgpu-')),
    serve = {},
    shift = { dx: 3.7, dy: -2.1 },
    count = 24,
    frames = [0, 1, 2].map((i) => renderScene(small, { dx: shift.dx * i, dy: shift.dy * i })),
    o = kltOptionsFor(W),
    pyramid = buildKltPyramid(frames[0], W, H, o.levels),
    { cellsX, cellsY } = kltCellGrid(W, H, o.minDistancePx),
    picks = kltSelectReference(
      kltEigenMapReference(pyramid.grad, W, H, o.halfWindow),
      W,
      H,
      o,
      new Float32Array(2 * cellsX * cellsY).fill(-1),
    ).picks,
    base = [...picks]
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .flatMap((p) => [p.x, p.y]);
  try {
    frames.forEach((frame, i) => {
      const file = join(dir, `join-${i}.bin`);
      writeFileSync(file, Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength));
      serve[`/join-${i}.bin`] = file;
    });
    const { result, errors } = await inBrowser(
      async (input) => {
        const { ShadeDevice, createKltTracker } = await import('/dist/internal/gpu.js');
        const load = async (name) =>
          new Float32Array(await (await fetch(`/${name}`)).arrayBuffer());
        const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        try {
          const tracker = await createKltTracker(device, 320, 200, { options: { capacity: 96 } }),
            out = {};
          await tracker.push({ gray: await load('join-0.bin') }, { frameId: 7, select: false });
          out.first = await tracker.join({ frameId: 7, keypoints: input.base });
          await tracker.push({ gray: await load('join-1.bin') }, { frameId: 8, select: false });
          await tracker.push({ gray: await load('join-2.bin') }, { frameId: 9, select: false });
          // The keypoints of frame 9: the corners moved by two shifts, in reverse order.
          const moved = [];
          for (let k = input.base.length / 2 - 1; k >= 0; k--)
            moved.push(input.base[2 * k] + 2 * input.dx, input.base[2 * k + 1] + 2 * input.dy);
          const join = await tracker.join({ frameId: 9, keypoints: moved });
          out.second = {
            pairs: join.pairs.map((p) => ({
              first: p.first,
              second: p.second,
              matches: Array.from(p.matches),
            })),
            joined: join.joined,
            seeded: join.seeded,
          };
          out.stats = tracker.stats();
          await tracker.dispose();
          out.live = device.statistics.liveBytes;
          return out;
        } finally {
          await device.dispose();
        }
      },
      { base, dx: shift.dx, dy: shift.dy },
      { timeoutMs: 600000, serve },
    );
    assert.deepEqual(errors, []);
    assert.equal(result.live, 0);
    assert.equal(result.first.seeded, count);
    assert.equal(result.first.pairs.length, 0);
    const pair = result.second.pairs[0];
    assert.equal(result.second.pairs.length, 1);
    assert.equal(pair.first, 7);
    assert.equal(pair.second, 9);
    // Keypoint k of frame 9 is corner count - 1 - k: the match (i, k) needs i + k = count - 1.
    let right = 0;
    for (let r = 0; r < pair.matches.length; r += 2)
      if (pair.matches[r] + pair.matches[r + 1] === count - 1) right++;
    assert.equal(right, pair.matches.length / 2, 'every match is a correspondence');
    assert.ok(pair.matches.length / 2 >= 0.85 * count, `${pair.matches.length / 2} of ${count}`);
    assert.equal(result.second.joined, pair.matches.length / 2);
    assert.equal(result.stats.joins, 2);
    console.log(
      JSON.stringify({
        join: { matches: pair.matches.length / 2, of: count, seeded: result.second.seeded },
      }),
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('KLT initial targets on WebGPU refine image coordinates without a tail-lane race', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'klt-initial-webgpu-'));
  try {
    const w = 192,
      h = 144,
      first = renderScene({ width: w, height: h }),
      second = renderScene({ width: w, height: h }, { dx: 24, dy: -17 }),
      points = gridPoints(w, h, 45, 40, 5, 3),
      guesses = Float32Array.from(points, (v, i) => v + (i % 2 ? -16 : 23)),
      opts = kltOptionsFor(w, { levels: 1, halfWindow: 5 }),
      reference = trackKlt(
        buildKltPyramid(first, w, h, 1),
        buildKltPyramid(second, w, h, 1),
        points,
        opts,
        guesses,
      ),
      serve = {};
    for (const [name, frame] of [
      ['first', first],
      ['second', second],
    ]) {
      const file = join(dir, name + '.bin');
      writeFileSync(file, Buffer.from(frame.buffer));
      serve['/' + name + '.bin'] = file;
    }
    const { result, errors } = await inBrowser(
      async (input) => {
        const { ShadeDevice, trackKltGpu } = await import('/dist/internal/gpu.js');
        const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        try {
          const a = new Float32Array(await (await fetch('/first.bin')).arrayBuffer());
          const b = new Float32Array(await (await fetch('/second.bin')).arrayBuffer());
          const out = [];
          for (const chunkTracks of [5, 64, 7]) {
            const r = await trackKltGpu(device, a, b, input.w, input.h, input.points, input.opts, {
              initialTargets: input.guesses,
              chunkTracks,
            });
            out.push(Array.from(r.records));
          }
          const bad = await trackKltGpu(
            device,
            a,
            new Float32Array(a.length).fill(0.5),
            input.w,
            input.h,
            input.points,
            input.opts,
            { initialTargets: input.guesses },
          );
          return { out, bad: Array.from(bad.records), liveBytes: device.statistics.liveBytes };
        } finally {
          await device.dispose();
        }
      },
      { w, h, points: Array.from(points), guesses: Array.from(guesses), opts },
      { serve },
    );
    assert.deepEqual(errors, []);
    assert.equal(result.liveBytes, 0);
    for (const records of result.out) {
      const e = flowErrors(records, points, { dx: 24, dy: -17 }, w, h);
      assert.equal(
        e.alive.reduce((a, b) => a + b, 0),
        points.length / 2,
      );
      assert.ok(Math.max(...e.error) < 0.1);
      records.forEach((v, i) => {
        if (i % 4 < 2) assert.ok(Math.abs(v - reference.records[i]) < 1e-3);
      });
    }
    assert.deepEqual(result.out[0], result.out[1]);
    assert.deepEqual(result.out[0], result.out[2]);
    assert.ok(result.bad.every((v, i) => i % 4 !== 3 || !(v & 1)));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
