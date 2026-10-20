// Verifies: docs/klt-tracking.md step 1 (sections 2.1 to 2.4, 3.5 and 7.1: the CPU reference).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KLT_END,
  KLT_DEFAULTS_640,
  buildKltPyramid,
  checkKltReference,
  kltLayout,
  kltOptionsFor,
  kltSeedRecords,
  trackKlt,
  trackKltReference,
} from '../src/gpu/klt.ts';
import { flowErrors, gridPoints, percentile, renderScene, type Warp } from './klt-cases.ts';

const W = 640,
  H = 360,
  scene = { width: W, height: H },
  options = kltOptionsFor(W),
  first = renderScene(scene),
  firstPyramid = buildKltPyramid(first, W, H, options.levels),
  points = gridPoints(W, H, 100, 80, 8, 5),
  center = [(W - 1) / 2, (H - 1) / 2];

function run(warp: Warp, opts = options, pyramid = firstPyramid, pts = points) {
  const second = renderScene(scene, warp),
    result = trackKlt(pyramid, buildKltPyramid(second, W, H, opts.levels), pts, opts),
    { error, alive } = flowErrors(result.records, pts, warp, W, H);
  return { result, error, alive };
}
const aliveErrors = (e: { error: Float64Array; alive: Uint8Array }, keep = (_k: number) => true) =>
  Array.from(e.error).filter((_, k) => e.alive[k] && keep(k));
const radius = (k: number) =>
  Math.hypot(points[2 * k]! - center[0]!, points[2 * k + 1]! - center[1]!);

test('klt layout, pyramid and Scharr gradient on exact images', () => {
  const layout = kltLayout(37, 23, 4);
  assert.deepEqual([...layout.widths], [37, 19, 10, 5]);
  assert.deepEqual([...layout.heights], [23, 12, 6, 3]);
  assert.equal(layout.total, 37 * 23 + 19 * 12 + 10 * 6 + 5 * 3);
  assert.deepEqual([...layout.offsets], [0, 851, 1079, 1139]);
  assert.throws(() => kltLayout(37, 23, 0), RangeError);
  // A constant image stays constant on every level and has no gradient.
  const flat = buildKltPyramid(new Float32Array(37 * 23).fill(0.25), 37, 23, 4);
  assert.ok(flat.gray.every((v) => v === 0.25));
  assert.ok(flat.grad.every((v) => v === 0));
  // A ramp has the gradient of its slope away from the border (Scharr scaled by 1/32).
  const ramp = new Float32Array(37 * 23);
  for (let y = 0; y < 23; y++) for (let x = 0; x < 37; x++) ramp[y * 37 + x] = 0.01 * x + 0.02 * y;
  const p = buildKltPyramid(ramp, 37, 23, 3);
  for (let y = 1; y < 22; y++)
    for (let x = 1; x < 36; x++) {
      assert.ok(Math.abs(p.grad[2 * (y * 37 + x)]! - 0.01) < 1e-6);
      assert.ok(Math.abs(p.grad[2 * (y * 37 + x) + 1]! - 0.02) < 1e-6);
    }
  // Level 1 of the ramp: the binomial blur keeps a linear function away from the border, so the
  // sample at (x, y) of level 1 is the ramp at (2x, 2y).
  const offset1 = p.offsets[1]!,
    width1 = p.widths[1]!;
  for (let y = 2; y < 9; y++)
    for (let x = 2; x < 16; x++)
      assert.ok(Math.abs(p.gray[offset1 + y * width1 + x]! - (0.01 * 2 * x + 0.02 * 2 * y)) < 1e-5);
});

test('klt defaults follow the frame width', () => {
  assert.deepEqual(kltOptionsFor(640), KLT_DEFAULTS_640);
  assert.equal(kltOptionsFor(1280).halfWindow, 10);
  assert.equal(kltOptionsFor(1280).levels, 6);
  assert.equal(kltOptionsFor(640, { levels: 3 }).levels, 3);
});

test('translations of section 7.1 are recovered; the pyramid is needed at 30 px', () => {
  const rows = [];
  for (const [dx, dy, bound] of [
    [0.3, 0.2, 0.05],
    [3.7, -2.1, 0.05],
    [12.5, 9.0, 0.1],
    [30, -24, 0.1],
  ] as const) {
    const e = run({ dx, dy }),
      errors = aliveErrors(e);
    rows.push({ dx, dy, alive: errors.length, max: +Math.max(...errors).toFixed(4) });
    assert.equal(errors.length, points.length / 2, `shift (${dx}, ${dy}): every track is kept`);
    assert.ok(
      Math.max(...errors) <= bound,
      `shift (${dx}, ${dy}): max error ${Math.max(...errors)}`,
    );
  }
  console.log(JSON.stringify(rows));
  // Prove the instrument: one level cannot reach 30 px, so a check of the same form fails.
  const one = kltOptionsFor(W, { levels: 1 }),
    single = run({ dx: 30, dy: -24 }, one, buildKltPyramid(first, W, H, 1)),
    errors = aliveErrors(single);
  assert.ok(
    errors.length < points.length / 2 || Math.max(...errors) > 0.1,
    'one level must fail the check of 30 px',
  );
  // Deviation from section 2.4: four levels, the old default, lose tracks at this shift.
  const four = run(
    { dx: 30, dy: -24 },
    kltOptionsFor(W, { levels: 4 }),
    buildKltPyramid(first, W, H, 4),
  );
  assert.ok(aliveErrors(four).length < points.length / 2);
  assert.ok(Math.max(...aliveErrors(four)) <= 0.1, 'a lost track ends; a kept track is right');
});

test('rotations and scale: the error grows with the distance from the centre', () => {
  const rows = [];
  for (const [warp, slope] of [
    [{ degrees: 2 }, 0.0015],
    [{ degrees: 5 }, 0.0035],
    [{ scale: 1.05 }, 0.0006],
  ] as const) {
    const e = run(warp),
      kept = aliveErrors(e);
    rows.push({ warp, alive: kept.length, max: +Math.max(...kept).toFixed(3) });
    assert.ok(kept.length >= 38, `${JSON.stringify(warp)}: ${kept.length} of 40 kept`);
    for (let k = 0; k < e.error.length; k++)
      if (e.alive[k])
        assert.ok(
          e.error[k]! <= 0.05 + slope * radius(k),
          `${JSON.stringify(warp)}: point ${k} at ${radius(k).toFixed(0)} px has error ${e.error[k]}`,
        );
    // Within 60 px of the centre, the record's bound for 5 degrees: under 0.2 px.
    for (let k = 0; k < e.error.length; k++)
      if (e.alive[k] && radius(k) <= 70) assert.ok(e.error[k]! < 0.2);
  }
  // 10 degrees is past the translation model's reach for some points. The test records what
  // the checks keep; it asserts the median only.
  const ten = run({ degrees: 10 }),
    kept = aliveErrors(ten);
  rows.push({
    warp: { degrees: 10 },
    alive: kept.length,
    median: +percentile(kept, 0.5).toFixed(3),
    over1px: kept.filter((v) => v > 1).length,
  });
  console.log(JSON.stringify(rows));
  assert.ok(kept.length >= 34);
  assert.ok(percentile(kept, 0.5) < 0.4);
});

test('brightness change: the zero-mean, unit-variance residual keeps the track; plain SSD does not', () => {
  const warp = { gain: 1.1, offset: 0.05 },
    e = run(warp),
    kept = aliveErrors(e);
  console.log(
    JSON.stringify({
      gain: 1.1,
      offset: 0.05,
      alive: kept.length,
      max: +Math.max(...kept).toFixed(3),
    }),
  );
  assert.equal(kept.length, points.length / 2);
  assert.ok(Math.max(...kept) < 0.3, `max error ${Math.max(...kept)}`);
  // The known limit of the plain SSD form (section 9): the same change ends most tracks.
  const plainOptions = kltOptionsFor(W, { photometric: 0 }),
    plain = run(warp, plainOptions);
  assert.ok(aliveErrors(plain).length <= 10, `plain SSD kept ${aliveErrors(plain).length}`);
  // The zero-mean form without the variance scale is in between and keeps most tracks.
  const zero = run(warp, kltOptionsFor(W, { photometric: 1 }));
  assert.ok(aliveErrors(zero).length >= 38);
});

test('failure cases end with the reason of section 3.5', () => {
  const a = Float32Array.from(first),
    b = Float32Array.from(renderScene(scene, { dx: 2, dy: 1 }));
  // A textureless patch in both frames.
  for (const f of [a, b])
    for (let y = 120; y < 200; y++) for (let x = 170; x < 250; x++) f[y * W + x] = 0.5;
  // A one-pixel vertical line on a flat background: the aperture problem.
  for (const [f, shift] of [
    [a, 0],
    [b, 2],
  ] as const) {
    for (let y = 60; y < 140; y++) for (let x = 360; x < 460; x++) f[y * W + x] = 0.3;
    for (let y = 60; y < 140; y++) f[y * W + 400 + shift] = 0.9;
  }
  // An occlusion: a second texture covers the right part of two windows in the second frame.
  for (let y = 235; y < 270; y++)
    for (let x = 502; x < 530; x++)
      b[y * W + x] = 0.5 + 0.3 * Math.sin(x * 0.9 + y * 1.7) * Math.cos(y * 0.5);
  const pts = Float32Array.from([
    210,
    160,
    400,
    100,
    400,
    70,
    3,
    100,
    W - 4,
    100,
    100,
    3,
    500,
    250,
    520,
    250,
    480,
    250,
    300,
    300,
  ]);
  const result = trackKlt(
    buildKltPyramid(a, W, H, options.levels),
    buildKltPyramid(b, W, H, options.levels),
    pts,
    options,
  );
  const reason = Array.from(result.reason);
  console.log(JSON.stringify({ reason }));
  assert.equal(reason[0], KLT_END.lowStructure, 'textureless');
  assert.equal(reason[1], KLT_END.lowStructure, 'line, flat both sides');
  assert.equal(reason[2], KLT_END.lowStructure, 'line');
  assert.equal(reason[3], KLT_END.outOfFrame, 'left border');
  assert.equal(reason[4], KLT_END.outOfFrame, 'right border');
  assert.equal(reason[5], KLT_END.outOfFrame, 'top border');
  for (const k of [6, 7])
    assert.ok(
      ([KLT_END.noConvergence, KLT_END.forwardBackward, KLT_END.residual] as number[]).includes(
        reason[k]!,
      ),
      `occlusion ${k}: reason ${reason[k]}`,
    );
  // The two points with a clean window stay alive.
  assert.equal(reason[8], KLT_END.alive);
  assert.equal(reason[9], KLT_END.alive);
  assert.equal(result.alive[8], 1);
  // A track that ends keeps its status bits: alive 0 and the reason in bits 1 to 4.
  assert.equal(result.records[4 * 0 + 3], KLT_END.lowStructure << 1);
});

test('a range of tracks equals the same tracks in the whole run, bit for bit', () => {
  const second = buildKltPyramid(renderScene(scene, { dx: 3.1, dy: 1.2 }), W, H, options.levels),
    start = kltSeedRecords(points),
    whole = trackKltReference(firstPyramid, second, start, options),
    n = start.length >> 2,
    joined = new Float32Array(start.length);
  for (let first = 0; first < n; first += 3) {
    const part = trackKltReference(firstPyramid, second, start, options, {
      first,
      end: Math.min(first + 3, n),
    });
    for (let i = 4 * first; i < 4 * Math.min(first + 3, n); i++) joined[i] = part[i]!;
  }
  assert.deepEqual(new Uint8Array(joined.buffer), new Uint8Array(whole.buffer));
  const back = trackKltReference(second, firstPyramid, whole, options),
    checked = checkKltReference(start, whole, back, options),
    piece = checkKltReference(start, whole, back, options, { first: 2, end: 5 });
  for (let i = 8; i < 20; i++) assert.equal(piece[i], checked[i]);
  assert.equal(piece[0], 0);
});

test('forward-backward check drops the tracks that converged to a wrong place', () => {
  // A texture with a period of 8 px in x and y, shifted by 4 px. Each forward track may
  // converge to a place that is 4 px or 8 px away from the truth.
  const texture = (shift: number) => {
    const f = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        f[y * W + x] =
          0.5 + 0.3 * Math.sin((2 * Math.PI * (x - shift)) / 8) * Math.sin((2 * Math.PI * y) / 8);
    return f;
  };
  const pts = gridPoints(W, H, 100, 80, 12, 8),
    from = buildKltPyramid(texture(0), W, H, options.levels),
    to = buildKltPyramid(texture(4), W, H, options.levels),
    count = (opts: typeof options) => {
      const r = trackKlt(from, to, pts, opts);
      let wrong = 0,
        wrongKept = 0;
      for (let k = 0; k < pts.length / 2; k++) {
        if (!(r.forward[4 * k + 3]! % 2)) continue;
        const error = Math.hypot(
          r.forward[4 * k]! - (pts[2 * k]! + 4),
          r.forward[4 * k + 1]! - pts[2 * k + 1]!,
        );
        if (error > 1) {
          wrong++;
          wrongKept += r.records[4 * k + 3]! & 1;
        }
      }
      return { wrong, wrongKept };
    },
    checked = count(options),
    unchecked = count({ ...options, fbThresholdPx: 1e9, nccThreshold: -2 });
  console.log(JSON.stringify({ checked, unchecked }));
  assert.ok(checked.wrong >= 20, `${checked.wrong} wrong forward tracks`);
  assert.ok(checked.wrongKept <= 0.1 * checked.wrong);
  // Prove the instrument: with both checks off, the wrong tracks stay.
  assert.ok(unchecked.wrongKept >= 0.9 * unchecked.wrong);
});
