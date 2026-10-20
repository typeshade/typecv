import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  equalizeGrayCpu,
  equalizeGrayGpu,
  extractSiftGpuResident,
} from '../../dist/gpu.js';

/** A smooth, bright, low-contrast field with fine texture: values in about [0.78, 0.86]. */
function brightFace(width, height) {
  const gray = new Float32Array(width * height);
  let state = 99;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const noise = ((state >>> 8) % 1000) / 1000;
      gray[y * width + x] = 0.82 + 0.03 * Math.sin(x / 9) * Math.cos(y / 7) + 0.012 * (noise - 0.5);
    }
  return gray;
}
const toRgba = (gray) => {
  const rgba = new Uint8Array(gray.length * 4);
  for (let i = 0; i < gray.length; i++) {
    rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = Math.round(gray[i] * 255);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
};

test('CLAHE: a hand-checked 4 x 4 image with one tile and no clipping', async () => {
  // Half the pixels are black (bin 0), half white (bin 252). With one tile the map is the
  // cumulative histogram: floor(8 * 16383 / 16) / 16383 for every bin below 252, then 1.
  const gray = Float32Array.from({ length: 16 }, (_, i) => (i % 2 ? 1 : 0)),
    expected = Math.floor((8 * 16383) / 16) / 16383,
    options = { tiles: 1, clipLimit: 1 };
  const cpu = equalizeGrayCpu(gray, 4, 4, options);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const gpu = await equalizeGrayGpu(device, gray, 4, 4, options);
    for (let i = 0; i < 16; i++) {
      const want = i % 2 ? 1 : expected;
      assert.ok(Math.abs(cpu[i] - want) < 1e-6, `cpu pixel ${i}: ${cpu[i]} vs ${want}`);
      assert.ok(Math.abs(gpu[i] - want) < 1e-6, `gpu pixel ${i}: ${gpu[i]} vs ${want}`);
    }
  } finally {
    await device.dispose();
  }
});

test('CLAHE: clipping caps a tile bin and the clipped excess is redistributed', async () => {
  // 64 pixels in one tile: 48 at bin 0 and 16 spread over 16 other bins. Limit = floor(0.25 * 64)
  // = 16. Excess 32 over 256 bins: increment 0 for the uniform part, then one unit per bin at
  // the stride, so the map of bin 0 is at most 16 / 64 plus the redistributed units.
  const gray = new Float32Array(64);
  for (let i = 48; i < 64; i++) gray[i] = (i - 47) * 0.05;
  const cpu = equalizeGrayCpu(gray, 8, 8, { tiles: 1, clipLimit: 0.25 });
  const unclipped = equalizeGrayCpu(gray, 8, 8, { tiles: 1, clipLimit: 1 });
  assert.ok(cpu[0] < unclipped[0], `clipped ${cpu[0]} against unclipped ${unclipped[0]}`);
  assert.ok(cpu[0] <= 0.5 + 1e-6, `bin 0 holds at most 16 + redistribution of 64: ${cpu[0]}`);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const gpu = await equalizeGrayGpu(device, gray, 8, 8, { tiles: 1, clipLimit: 0.25 });
    for (let i = 0; i < 64; i++)
      assert.ok(Math.abs(gpu[i] - cpu[i]) < 1e-6, `pixel ${i}: ${gpu[i]} vs ${cpu[i]}`);
  } finally {
    await device.dispose();
  }
});

test('CLAHE: the device agrees with the CPU reference on a bright low-contrast image', async () => {
  const width = 131,
    height = 97,
    gray = brightFace(width, height),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const options of [true, { tiles: [5, 3], clipLimit: 0.02 }, { tiles: 32 }]) {
      const cpu = equalizeGrayCpu(gray, width, height, options),
        gpu = await equalizeGrayGpu(device, gray, width, height, options);
      let worst = 0;
      for (let i = 0; i < cpu.length; i++) worst = Math.max(worst, Math.abs(cpu[i] - gpu[i]));
      // One 14-bit level is 6.1e-5; the device sums in f32 and may land one level apart.
      assert.ok(worst < 2.5e-4, `max |gpu - cpu| = ${worst} for ${JSON.stringify(options)}`);
    }
    // The input spans about 0.07 of the range; clipping keeps the stretch finite but large.
    const out = equalizeGrayCpu(gray, width, height, true);
    const span = (v) => Math.max(...v) - Math.min(...v);
    assert.ok(span(gray) < 0.1 && span(out) > 3 * span(gray), `${span(gray)} -> ${span(out)}`);
  } finally {
    await device.dispose();
  }
});

test('CLAHE: invalid options are refused', () => {
  const gray = new Float32Array(64 * 64);
  for (const options of [
    { tiles: 0 },
    { tiles: 33 },
    { tiles: 1.5 },
    { clipLimit: 0 },
    { clipLimit: 2 },
  ])
    assert.throws(() => equalizeGrayCpu(gray, 64, 64, options), RangeError);
});

test('SIFT: clahe is off by default, leaves the result unchanged, and adds features on a bright low-contrast image', async () => {
  const width = 192,
    height = 160,
    rgba = toRgba(brightFace(width, height)),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const plain = await extractSiftGpuResident(device, rgba, width, height, { maxFeatures: 2048 });
    const off = await extractSiftGpuResident(device, rgba, width, height, {
      maxFeatures: 2048,
      clahe: false,
    });
    assert.equal(off.count, plain.count);
    assert.deepEqual(await off.readDetails(), await plain.readDetails());
    const on = await extractSiftGpuResident(device, rgba, width, height, {
      maxFeatures: 2048,
      clahe: true,
    });
    assert.ok(on.count > plain.count, `${on.count} with CLAHE against ${plain.count} without`);
    assert.throws(
      () => extractSiftGpuResident(device, rgba, width, height, { clahe: { tiles: 0 } }),
      RangeError,
    );
  } finally {
    await device.dispose();
  }
});
