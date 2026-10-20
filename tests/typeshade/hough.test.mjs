import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, canny, cannyGpu, houghLines, houghLinesGpu } from '../../dist/gpu.js';
import {
  ABSOLUTE,
  STROKES,
  accumulate,
  agreement,
  brokenBar,
  nearest,
  noise,
  photo,
  step,
  strokes,
} from './hough-cases.mjs';

const size = 256;

/** The eight borders of the four strokes. */
const expected = STROKES.flatMap(([degrees, rho]) => [
  [degrees, rho],
  [degrees, rho + 12],
]);

function assertLines(lines) {
  for (const [degrees, rho] of expected) {
    const { dTheta, dRho } = nearest(lines, degrees, rho);
    assert.ok(dTheta <= 1 && dRho <= 1, `line (${degrees} deg, ${rho}): ${dTheta} deg, ${dRho} px`);
  }
  assert.ok(lines.length <= 12, `${lines.length} lines for 8 borders`);
}

test('Hough: the CPU reference recovers lines at 0, 30, 90 and 135 degrees within 1 degree and 1 px', () => {
  const result = houghLines(strokes(size, size, STROKES), size, size, ABSOLUTE);
  assertLines(result.lines);
  assert.equal(result.accumulator.length, result.thetaBins * result.rhoCount);
  assert.ok(result.lines.every((line) => line.theta >= 0 && line.theta < Math.PI));
});

test('Hough: the defaults find the strokes of a textured, noisy photo-like image', async () => {
  const image = photo(size, size, STROKES),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const cpu = houghLines(image, size, size),
      gpu = await houghLinesGpu(device, image, size, size);
    for (const result of [cpu, gpu]) {
      // Quantile thresholds: a few thousand edge pixels in a textured 65536 px image.
      assert.ok(result.count > 1000 && result.count < 20000, `${result.count} edge pixels`);
      assert.ok(result.low > 0 && result.high >= result.low);
      for (const [degrees, rho] of expected) {
        const { dTheta, dRho } = nearest(result.lines, degrees, rho);
        assert.ok(
          dTheta <= 2 && dRho <= 2,
          `line (${degrees} deg, ${rho}): ${dTheta} deg, ${dRho} px`,
        );
      }
      assert.ok(result.lines.length <= 40, `${result.lines.length} lines for 8 borders`);
    }
    assert.ok(agreement(cpu.edges, gpu.edges) >= 0.99, `${agreement(cpu.edges, gpu.edges)}`);
    // The absolute thresholds 0.1 and 0.2 miss most of the strokes here (the old default).
    const absolute = houghLines(image, size, size, { thresholdMode: 'absolute' });
    assert.ok(absolute.lines.length < 8, `${absolute.lines.length} lines`);
  } finally {
    await device.dispose();
  }
});

test('Hough: noise gives edges but no line above the default threshold', () => {
  const image = noise(size, size, 127),
    result = houghLines(image, size, size, ABSOLUTE);
  assert.ok(result.count > 300, `only ${result.count} edge pixels: the test would prove nothing`);
  const most = result.accumulator.reduce((a, b) => Math.max(a, b), 0);
  assert.ok(most < 64, `a cell has ${most} votes`);
  assert.deepEqual(result.lines, []);
  assert.deepEqual(result.segments, []);
  // The same instrument finds a line once the threshold is below the noise maximum.
  assert.ok(
    houghLines(image, size, size, { ...ABSOLUTE, threshold: Math.max(1, most - 2) }).lines.length >
      0,
  );
});

test('Canny: a step edge is one pixel thin, at the step, on both tiers', async () => {
  const w = 64,
    h = 48,
    image = step(w, h, 32),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const out of [
      canny(image, w, h, ABSOLUTE),
      await cannyGpu(device, image, w, h, ABSOLUTE),
    ]) {
      assert.equal(out.count, h - 2);
      for (let y = 0; y < h; y++) {
        const row = Array.from(out.edges.slice(y * w, (y + 1) * w), (v, x) => (v ? x : -1)).filter(
          (x) => x >= 0,
        );
        if (y === 0 || y === h - 1) assert.deepEqual(row, [], `border row ${y}`);
        else assert.equal(row.length, 1, `row ${y}: ${row}`);
        if (row.length) assert.ok(row[0] === 31 || row[0] === 32, `row ${y}: x ${row[0]}`);
      }
    }
    const flat = canny(new Uint8Array(w * h * 4).fill(90), w, h, ABSOLUTE);
    assert.equal(flat.count, 0);
  } finally {
    await device.dispose();
  }
});

test('Canny: a weak edge joins a strong one only when connected, with both hysteresis modes', async () => {
  // A vertical edge whose contrast falls from strong at the top to weak at the bottom.
  const w = 64,
    h = 64,
    image = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const contrast = Math.round(255 - ((255 - 70) * y) / (h - 1)),
        v = x < 32 ? 0 : contrast;
      image.set([v, v, v, 255], (y * w + x) * 4);
    }
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const weakOnly = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++)
      weakOnly.set([i % w < 32 ? 0 : 70, i % w < 32 ? 0 : 70, i % w < 32 ? 0 : 70, 255], i * 4);
    assert.equal(canny(weakOnly, w, h, ABSOLUTE).count, 0, 'a weak edge alone is dropped');
    const cpu = canny(image, w, h, ABSOLUTE);
    assert.ok(cpu.count === h - 2, `the weak half joins the strong half: ${cpu.count}`);
    for (let y = 1; y < h - 1; y++)
      assert.equal(
        cpu.edges.slice(y * w, (y + 1) * w).reduce((a, b) => a + b),
        1,
        `weak row ${y}`,
      );
    for (const hysteresis of ['gpu', 'cpu']) {
      const out = await cannyGpu(device, image, w, h, { ...ABSOLUTE, hysteresis });
      assert.equal(out.count, cpu.count, hysteresis);
      assert.equal(agreement(out.edges, cpu.edges), 1);
      if (hysteresis === 'gpu') assert.ok(out.hysteresisPasses >= 8);
      const alone = await cannyGpu(device, weakOnly, w, h, { ...ABSOLUTE, hysteresis });
      assert.equal(alone.count, 0, hysteresis);
    }
  } finally {
    await device.dispose();
  }
});

test('Hough: the device equals the CPU reference (edges, accumulator, lines, segments)', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const image of [
      strokes(size, size, STROKES),
      brokenBar(size, size, [100, 20, 100, 130, 230]),
    ]) {
      const cpu = houghLines(image, size, size, ABSOLUTE),
        gpu = await houghLinesGpu(device, image, size, size, ABSOLUTE);
      assert.ok(
        agreement(cpu.edges, gpu.edges) >= 0.999,
        `edges ${agreement(cpu.edges, gpu.edges)}`,
      );
      // The vote is exact: the device accumulator is the accumulator of its own edge map.
      assert.deepEqual(
        gpu.accumulator,
        accumulate(gpu.edges, size, gpu.thetaBins, gpu.rhoCount, gpu.maxRho),
      );
      if (cpu.count === gpu.count && agreement(cpu.edges, gpu.edges) === 1) {
        assert.deepEqual(gpu.accumulator, cpu.accumulator);
        assert.deepEqual(gpu.lines, cpu.lines);
        assert.deepEqual(gpu.segments, cpu.segments);
      }
      assert.ok(gpu.lines.length >= 2);
    }
    assertLines(
      (await houghLinesGpu(device, strokes(size, size, STROKES), size, size, ABSOLUTE)).lines,
    );
  } finally {
    await device.dispose();
  }
});

test('Hough: segments follow the edge pixels and break at a gap longer than maxLineGap', async () => {
  const image = brokenBar(size, size, [100, 20, 100, 130, 230]),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const result of [
      houghLines(image, size, size, ABSOLUTE),
      await houghLinesGpu(device, image, size, size, ABSOLUTE),
    ]) {
      // Lines at 90 degrees: rho 100 and 112, the two long borders of the bars.
      const horizontal = result.lines
        .map((line, index) => ({ line, index }))
        .filter(({ line }) => Math.abs(line.theta - Math.PI / 2) < 0.02);
      assert.ok(horizontal.length >= 2);
      for (const { line, index } of horizontal) {
        const own = result.segments.filter((s) => s.line === index).sort((a, b) => a.x1 - b.x1);
        assert.equal(own.length, 2, `rho ${line.rho}: ${JSON.stringify(own)}`);
        assert.ok(
          Math.abs(own[0].x1 - 20) <= 3 && Math.abs(own[0].x2 - 99) <= 3,
          JSON.stringify(own[0]),
        );
        assert.ok(
          Math.abs(own[1].x1 - 130) <= 3 && Math.abs(own[1].x2 - 229) <= 3,
          JSON.stringify(own[1]),
        );
        for (const s of own)
          assert.ok(Math.abs(s.y1 - line.rho) <= 1 && Math.abs(s.y2 - line.rho) <= 1);
      }
    }
    // A gap bridge: maxLineGap 40 joins the bars; minLineLength 150 drops both halves.
    const joined = houghLines(image, size, size, { ...ABSOLUTE, maxLineGap: 40 });
    assert.ok(joined.segments.some((s) => s.x2 - s.x1 >= 200));
    assert.equal(
      houghLines(image, size, size, { ...ABSOLUTE, minLineLength: 150 }).segments.length,
      0,
    );
    assert.deepEqual(houghLines(image, size, size, { ...ABSOLUTE, segments: false }).segments, []);
  } finally {
    await device.dispose();
  }
});

test('Hough: invalid input is refused and a blank image has no lines', async () => {
  const rgba = new Uint8Array(16 * 16 * 4),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const options of [
      { sigma: 0 },
      { thetaBins: 1 },
      { threshold: 0 },
      { minAngle: 90 },
      { hysteresis: 'x' },
    ])
      assert.throws(() => houghLines(rgba, 16, 16, options), RangeError);
    assert.throws(() => houghLines(rgba, 15, 16), RangeError);
    const blank = await houghLinesGpu(device, rgba, 16, 16);
    assert.equal(blank.count, 0);
    assert.deepEqual(blank.lines, []);
    assert.equal(
      blank.accumulator.every((v) => v === 0),
      true,
    );
  } finally {
    await device.dispose();
  }
});
