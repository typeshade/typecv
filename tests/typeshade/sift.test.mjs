import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, extractSiftGpuResident, matchFloatDescriptorsGpu } from '../../dist/gpu.js';

/** A 256 × 192 texture of blobs and bars with smooth edges, so that scale space has structure. */
function texture(width, height) {
  const gray = new Float32Array(width * height);
  let state = 12345;
  const blobs = [];
  for (let i = 0; i < 70; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const x = 12 + (state % (width - 24));
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const y = 12 + (state % (height - 24));
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const r = 2 + (state % 7);
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const sign = state % 2 ? 1 : -1;
    blobs.push({ x, y, r, sign });
  }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let value = 0.5 + 0.08 * Math.sin(x / 23) * Math.cos(y / 17);
      for (const blob of blobs) {
        const d2 = (x - blob.x) ** 2 + (y - blob.y) ** 2;
        value += blob.sign * 0.4 * Math.exp(-d2 / (2 * blob.r * blob.r));
      }
      gray[y * width + x] = Math.max(0, Math.min(1, value));
    }
  return gray;
}
/** Sample the texture under a similarity (scale, rotation, translation) with bilinear interpolation. */
function warped(gray, width, height, outWidth, outHeight, scale, angle, tx, ty) {
  const rgba = new Uint8Array(outWidth * outHeight * 4);
  const c = Math.cos(angle),
    s = Math.sin(angle);
  for (let y = 0; y < outHeight; y++)
    for (let x = 0; x < outWidth; x++) {
      // Inverse map: source = R⁻¹ ((target - t) / scale).
      const ux = (x - tx) / scale,
        uy = (y - ty) / scale,
        sx = c * ux + s * uy,
        sy = -s * ux + c * uy;
      let value = 0.5;
      if (sx >= 0 && sy >= 0 && sx < width - 1 && sy < height - 1) {
        const x0 = Math.floor(sx),
          y0 = Math.floor(sy),
          fx = sx - x0,
          fy = sy - y0;
        value =
          gray[y0 * width + x0] * (1 - fx) * (1 - fy) +
          gray[y0 * width + x0 + 1] * fx * (1 - fy) +
          gray[(y0 + 1) * width + x0] * (1 - fx) * fy +
          gray[(y0 + 1) * width + x0 + 1] * fx * fy;
      }
      const byte = Math.round(value * 255),
        i = (y * outWidth + x) * 4;
      rgba[i] = rgba[i + 1] = rgba[i + 2] = byte;
      rgba[i + 3] = 255;
    }
  return rgba;
}
function similarityInliers(matches, source, target, scale, angle, tx, ty, tolerance) {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  let inliers = 0;
  for (let i = 0; i < matches.length; i += 2) {
    const a = matches[i] * 2,
      b = matches[i + 1] * 2;
    const px = scale * (c * source[a] - s * source[a + 1]) + tx,
      py = scale * (s * source[a] + c * source[a + 1]) + ty;
    if (Math.hypot(px - target[b], py - target[b + 1]) <= tolerance) inliers++;
  }
  return inliers;
}

test('SIFT on the CPU tier: keypoints repeat under scale and rotation and descriptors match', async () => {
  const width = 256,
    height = 192,
    gray = texture(width, height);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const base = warped(gray, width, height, width, height, 1, 0, 0, 0);
    const first = await extractSiftGpuResident(device, base, width, height, { maxFeatures: 1024 });
    assert.ok(first.count > 40, `only ${first.count} keypoints`);
    assert.ok(first.count <= first.capacity);
    assert.equal(first.descriptors.dimensions, 128);
    assert.equal(first.descriptors.elements, first.count * 128);
    const details = await first.readDetails();
    for (let i = 0; i < first.count; i++) {
      assert.ok(details[i * 8 + 2] > 1, 'sigma in image pixels');
      assert.ok(details[i * 8 + 3] >= 0 && details[i * 8 + 3] < 2 * Math.PI + 1e-6);
      assert.ok(Math.abs(details[i * 8 + 4]) >= 0.02 / 3);
    }
    const scale = 0.75,
      angle = Math.PI / 7,
      tx = 40,
      ty = 30;
    const moved = warped(gray, width, height, width, height, scale, angle, tx, ty);
    const second = await extractSiftGpuResident(device, moved, width, height, {
      maxFeatures: 1024,
    });
    assert.ok(second.count > 40, `only ${second.count} keypoints`);
    const matched = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
    });
    const pairs = matched.matches.length / 2;
    assert.ok(pairs >= 25, `only ${pairs} matches`);
    const inliers = similarityInliers(
      matched.matches,
      await first.readKeypoints(),
      await second.readKeypoints(),
      scale,
      angle,
      tx,
      ty,
      3,
    );
    assert.ok(inliers >= 0.7 * pairs, `${inliers} of ${pairs} matches follow the similarity`);
    // Descriptors are non-negative unit vectors, clipped at 0.2 before the final normalization.
    const vector = await device.enqueue(() => device.read(first.descriptors.descriptors, 128));
    let norm = 0;
    for (let j = 0; j < 128; j++) {
      assert.ok(vector[j] >= 0 && vector[j] <= 1);
      norm += vector[j] * vector[j];
    }
    assert.ok(Math.abs(norm - 1) < 1e-4, `norm ${norm}`);
    console.log(JSON.stringify({ first: first.count, second: second.count, pairs, inliers }));
    await first.dispose();
    await second.dispose();
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('SIFT rejects invalid frames and options and honours the capacity', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const width = 64,
      height = 64,
      rgba = warped(texture(width, height), width, height, width, height, 1, 0, 0, 0);
    assert.throws(() => extractSiftGpuResident(device, rgba, 16, 16), RangeError);
    assert.throws(
      () => extractSiftGpuResident(device, rgba, width, height, { maxFeatures: 0 }),
      RangeError,
    );
    assert.throws(
      () => extractSiftGpuResident(device, rgba, width, height, { maxOrientations: 3 }),
      RangeError,
    );
    const capped = await extractSiftGpuResident(device, rgba, width, height, { maxFeatures: 5 });
    assert.ok(capped.count <= 5);
    assert.ok(capped.requested >= capped.count);
    await capped.dispose();
  } finally {
    await device.dispose();
  }
});
