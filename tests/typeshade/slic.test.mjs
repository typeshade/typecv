import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, slic, slicGpu } from '../../dist/gpu.js';

/** Colored rectangles on a mid-gray ground, with a little deterministic noise. */
export function rectangles(width, height) {
  const rgba = new Uint8Array(width * height * 4),
    truth = new Uint32Array(width * height),
    boxes = [
      [0, 0, 0.45, 0.5, [200, 40, 40]],
      [0.45, 0, 1, 0.3, [40, 90, 210]],
      [0.45, 0.3, 0.8, 0.75, [230, 200, 50]],
      [0.8, 0.3, 1, 1, [60, 170, 80]],
      [0, 0.5, 0.45, 1, [150, 60, 190]],
      [0.45, 0.75, 0.8, 1, [30, 30, 40]],
    ];
  let state = 7;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const id = boxes.findIndex(
        ([x0, y0, x1, y1]) =>
          x >= x0 * width && x < x1 * width && y >= y0 * height && y < y1 * height,
      );
      truth[y * width + x] = id;
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const noise = ((state >>> 8) % 5) - 2;
      for (let c = 0; c < 3; c++) rgba[(y * width + x) * 4 + c] = boxes[id][4][c] + noise;
      rgba[(y * width + x) * 4 + 3] = 255;
    }
  return { rgba, truth };
}

/** Pixels whose right or lower neighbour has another label. */
const boundary = (labels, width, height) => {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (
        (x + 1 < width && labels[i] !== labels[i + 1]) ||
        (y + 1 < height && labels[i] !== labels[i + width])
      )
        out[i] = 1;
    }
  return out;
};

/** Share of the pixels of `truth` that have a pixel of `found` within `radius` (Chebyshev). */
export function boundaryRecall(truth, found, width, height, radius) {
  let total = 0,
    hit = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (!truth[y * width + x]) continue;
      total++;
      search: for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++) {
          const u = x + dx,
            v = y + dy;
          if (u >= 0 && v >= 0 && u < width && v < height && found[v * width + u]) {
            hit++;
            break search;
          }
        }
    }
  return hit / total;
}

/** Number of 4-connected components of each label. */
export function componentsPerLabel(labels, width, height, count) {
  const seen = new Uint8Array(labels.length),
    components = new Array(count).fill(0);
  for (let s = 0; s < labels.length; s++) {
    if (seen[s]) continue;
    components[labels[s]]++;
    const stack = [s];
    seen[s] = 1;
    while (stack.length) {
      const p = stack.pop(),
        x = p % width,
        y = (p - x) / width;
      for (const q of [
        x > 0 && p - 1,
        x < width - 1 && p + 1,
        y > 0 && p - width,
        y < height - 1 && p + width,
      ])
        if (q !== false && !seen[q] && labels[q] === labels[p]) {
          seen[q] = 1;
          stack.push(q);
        }
    }
  }
  return components;
}

const width = 192,
  height = 144;

test('SLIC: the CPU reference recovers the rectangle boundaries with about the requested count', () => {
  const { rgba, truth } = rectangles(width, height),
    result = slic(rgba, width, height, { nSegments: 30 });
  assert.equal(result.labels.length, width * height);
  assert.ok(result.count >= 20 && result.count <= 45, `count ${result.count}`);
  const recall = boundaryRecall(
    boundary(truth, width, height),
    boundary(result.labels, width, height),
    width,
    height,
    2,
  );
  assert.ok(recall >= 0.95, `boundary recall at 2 px: ${recall}`);
  // Every label is one 4-connected component.
  const components = componentsPerLabel(result.labels, width, height, result.count);
  assert.ok(
    components.every((n) => n === 1),
    `components per label: ${components}`,
  );
  // The defaults give about 100 segments.
  const defaults = slic(rgba, width, height);
  assert.ok(defaults.count >= 70 && defaults.count <= 130, `default count ${defaults.count}`);
});

test('SLIC: the device agrees with the CPU reference and keeps the boundaries and connectivity', async () => {
  const { rgba, truth } = rectangles(width, height),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const options of [{ nSegments: 30 }, {}, { nSegments: 12, compactness: 25 }]) {
      const cpu = slic(rgba, width, height, options),
        gpu = await slicGpu(device, rgba, width, height, options);
      // Labels are numbered in scan order, so two runs that cut the same regions agree
      // exactly; a f32 tie broken the other way moves a few pixels at a segment edge.
      let same = 0;
      for (let i = 0; i < cpu.labels.length; i++) if (cpu.labels[i] === gpu.labels[i]) same++;
      const agreement = same / cpu.labels.length;
      assert.ok(agreement >= 0.99, `agreement ${agreement} for ${JSON.stringify(options)}`);
      const components = componentsPerLabel(gpu.labels, width, height, gpu.count);
      assert.ok(
        components.every((n) => n === 1),
        `components: ${components}`,
      );
      const recall = boundaryRecall(
        boundary(truth, width, height),
        boundary(gpu.labels, width, height),
        width,
        height,
        2,
      );
      assert.ok(recall >= 0.95, `gpu boundary recall ${recall} for ${JSON.stringify(options)}`);
    }
  } finally {
    await device.dispose();
  }
});

test('SLIC: a hand-checked image of two flat halves gives two labels, on both tiers of code', async () => {
  // Left half red, right half blue, two segments: a 2 x 1 grid, so the halves are labels 0 and 1.
  const w = 40,
    h = 20,
    rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const left = i % w < w / 2;
    rgba.set(left ? [220, 30, 30, 255] : [30, 30, 220, 255], i * 4);
  }
  const want = Uint32Array.from({ length: w * h }, (_, i) => (i % w < w / 2 ? 0 : 1)),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assert.deepEqual(slic(rgba, w, h, { nSegments: 2 }).labels, want);
    assert.deepEqual((await slicGpu(device, rgba, w, h, { nSegments: 2 })).labels, want);
  } finally {
    await device.dispose();
  }
});

test('SLIC: connectivity enforcement leaves one component per label at a low compactness', () => {
  const { rgba } = rectangles(width, height),
    options = { nSegments: 60, compactness: 1 },
    raw = slic(rgba, width, height, { ...options, enforceConnectivity: false }),
    clean = slic(rgba, width, height, options);
  assert.ok(raw.count >= clean.count - 5, `${raw.count} labels raw, ${clean.count} enforced`);
  assert.ok(componentsPerLabel(clean.labels, width, height, clean.count).every((n) => n === 1));
});

test('SLIC: a flat image and tiny images are labelled without error', async () => {
  const flat = new Uint8Array(40 * 30 * 4).fill(128),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const cpu = slic(flat, 40, 30, { nSegments: 6 });
    assert.ok(cpu.count >= 1);
    const tiny = await slicGpu(device, new Uint8Array(2 * 2 * 4).fill(200), 2, 2, { nSegments: 4 });
    assert.equal(tiny.labels.length, 4);
    const none = await slicGpu(device, flat, 40, 30, { nSegments: 6, maxIterations: 0 });
    assert.equal(none.labels.length, 1200);
  } finally {
    await device.dispose();
  }
});

test('SLIC: invalid input is refused', () => {
  const rgba = new Uint8Array(16 * 16 * 4);
  for (const options of [
    { nSegments: 0 },
    { nSegments: 1.5 },
    { compactness: 0 },
    { maxIterations: -1 },
  ])
    assert.throws(() => slic(rgba, 16, 16, options), RangeError);
  assert.throws(() => slic(rgba, 15, 16), RangeError);
});
