import test from 'node:test';
import assert from 'node:assert/strict';
import { PhaseGate, PHASE_GATE_OPTIONS, keyframeGateMode } from './src/lib/keyframe-gate.ts';
import { fft2d } from './src/lib/phase-correlation.ts';

// A deterministic band-limited texture, so a failure repeats.
function texture(size, seed) {
  let s = seed >>> 0;
  const random = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
  const re = new Float64Array(size * size),
    im = new Float64Array(size * size);
  for (let i = 0; i < size * size; i++) re[i] = random() - 0.5;
  fft2d(re, im, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const gain = 1 / (1 + Math.hypot(Math.min(x, size - x), Math.min(y, size - y))) ** 1.5;
      re[y * size + x] *= gain;
      im[y * size + x] *= gain;
    }
  fft2d(re, im, size, true);
  const lo = Math.min(...re),
    hi = Math.max(...re);
  return re.map((v) => (255 * (v - lo)) / (hi - lo));
}

const SIZE = 256,
  W = 128;
const scene = texture(SIZE, 5),
  other = texture(SIZE, 77);

/** An RGBA frame of 128 × 128 pixels: the scene seen through a window at column `x0`. */
function view(source, x0) {
  const rgba = new Uint8ClampedArray(W * W * 4);
  for (let y = 0; y < W; y++)
    for (let x = 0; x < W; x++) {
      const v = source[(y + 64) * SIZE + x0 + x];
      rgba.set([v, v, v, 255], (y * W + x) * 4);
    }
  return rgba;
}

/** Run a sequence of `[time, window column]` through the gate, as the SfM worker does. */
function run(sequence, options = {}, lastIndex = sequence.length - 1) {
  const gate = new PhaseGate(options);
  return sequence.map(([time, x0, source = scene], i) => {
    const decision = gate.decide(view(source, x0), W, W, time, i === lastIndex);
    if (decision.action === 'skip') gate.discard();
    else gate.accept();
    return decision;
  });
}

test('the gate keeps the first frame, skips a still camera and keeps a camera that moved', () => {
  // The window slides right by `x0 - 64` columns: 5.12 px is 4 % of 128.
  const sequence = [
    [0, 64],
    [0.25, 64],
    [0.5, 65],
    [0.75, 66],
    [0.9, 72], // 8 px: moved
    [1.25, 72],
    [1.5, 73],
  ];
  // The final frame is not the last of the video here: pass an out-of-range index.
  const out = run(sequence, {}, -1);
  assert.deepEqual(
    out.map((d) => d.action),
    ['keep', 'skip', 'skip', 'skip', 'keep', 'skip', 'skip'],
  );
  assert.equal(out[0].reason, 'first frame');
  assert.equal(out[4].reason, 'moved');
  // The estimate is the signed translation from the reference: the window moved right, so the
  // content moved left.
  assert.ok(Math.abs(out[4].shift.dx + 8) < 0.2, 'dx ' + out[4].shift.dx);
  assert.ok(Math.abs(out[1].shift.dx) < 0.1);
});

test('the reference is the last keyframe, not the last frame', () => {
  // Three steps of 2 px: no single step reaches 5.12 px, the third frame is 6 px from the first.
  const out = run(
    [
      [0, 64],
      [0.25, 66],
      [0.5, 68],
      [0.75, 70],
    ],
    {},
    -1,
  );
  assert.deepEqual(
    out.map((d) => d.action),
    ['keep', 'skip', 'skip', 'keep'],
  );
});

test('maximum gap, minimum gap and the last frame keep the guarantees of the legacy gate', () => {
  const still = (time) => [time, 64];
  const out = run([still(0), still(0.1), still(0.5), still(1.0), still(1.1), still(1.5)], {}, 5);
  assert.deepEqual(
    out.map((d) => [d.action, d.reason]),
    [
      ['keep', 'first frame'],
      ['skip', 'minimum gap'],
      ['skip', 'barely moved'],
      ['keep', 'maximum gap'],
      ['skip', 'minimum gap'],
      ['keep', 'last frame'],
    ],
  );
  assert.equal(PHASE_GATE_OPTIONS.maxGapSeconds, 1);
});

test('a weak correlation peak leaves the decision to the legacy rule', () => {
  const out = run(
    [
      [0, 64],
      [0.5, 64, other],
    ],
    {},
    -1,
  );
  assert.equal(out[1].action, 'fallback');
  assert.ok(out[1].shift.ratio < PHASE_GATE_OPTIONS.minRatio);
});

test('a rejected keyframe does not become the reference', () => {
  const gate = new PhaseGate();
  gate.decide(view(scene, 64), W, W, 0);
  gate.accept();
  // The worker did not register this frame (too few features): no accept().
  gate.decide(view(scene, 72), W, W, 0.5);
  const next = gate.decide(view(scene, 73), W, W, 0.75);
  assert.equal(next.action, 'keep');
  assert.ok(Math.abs(next.shift.dx + 9) < 0.2, 'measured from the first frame: ' + next.shift.dx);
});

test('?gate= chooses the gate and ignores other values', () => {
  assert.equal(keyframeGateMode('?gate=legacy'), 'legacy');
  assert.equal(keyframeGateMode('?gate=phase'), 'phase');
  assert.equal(keyframeGateMode(''), 'phase');
  assert.equal(keyframeGateMode('?gate=other'), 'phase');
  assert.equal(keyframeGateMode('?x=1', 'legacy'), 'legacy');
});
