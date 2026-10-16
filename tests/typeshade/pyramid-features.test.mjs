import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  extractFeaturesGpuPyramidResident,
  extractFeaturesGpuResident,
  matchResidentFeaturesGpu,
} from '../../dist/gpu.js';

function baseTexture() {
  const gray = new Uint8Array(320 * 240);
  gray.fill(24);
  for (let cy = 10; cy < 230; cy += 18) {
    for (let cx = 10; cx < 310; cx += 18) {
      const seed = (cx * 73856093) ^ (cy * 19349663);
      const size = 4 + (Math.abs(seed) % 7);
      const bit = Math.abs(seed >>> 4) % 2 === 0;
      for (let y = cy; y < Math.min(cy + size, 240); y++)
        for (let x = cx; x < Math.min(cx + size, 320); x++) gray[y * 320 + x] = bit ? 232 : 8;
    }
  }
  return gray;
}

function texturedFrame(gray, width, height, scale = 1, tx = 0, ty = 0) {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = Math.floor((x - tx) / scale),
        sy = Math.floor((y - ty) / scale);
      const value = sx >= 0 && sy >= 0 && sx < 320 && sy < 240 ? gray[sy * 320 + sx] : 24;
      const i = (y * width + x) * 4;
      pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
      pixels[i + 3] = 255;
    }
  }
  return pixels;
}

function noisyTranslation(gray, width, height, tx, ty, seed) {
  const pixels = new Uint8Array(width * height * 4);
  let state = seed >>> 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const sx = x - tx,
        sy = y - ty,
        base = sx >= 0 && sy >= 0 && sx < 320 && sy < 240 ? gray[sy * 320 + sx] : 24,
        noise = (state % 49) - 24,
        value = Math.max(0, Math.min(255, base + noise)),
        i = (y * width + x) * 4;
      pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
      pixels[i + 3] = 255;
    }
  }
  return pixels;
}

function affineInlierCount(matches, source, target, scale, tx, ty, tolerance = 2.5) {
  let count = 0;
  for (let i = 0; i < matches.matches.length; i += 2) {
    const a = matches.matches[i],
      b = matches.matches[i + 1];
    const dx = target.keypoints[b * 2] - (source.keypoints[a * 2] * scale + tx);
    const dy = target.keypoints[b * 2 + 1] - (source.keypoints[a * 2 + 1] * scale + ty);
    if (Math.hypot(dx, dy) <= tolerance) count++;
  }
  return count;
}

test('two-scale resident features stay within budget and recover exact 2x scale-transform support', async (t) => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  const gray = baseTexture();
  const sourcePixels = texturedFrame(gray, 320, 240),
    targetPixels = texturedFrame(gray, 640, 480, 2, 8, 6);
  const resident = [];
  try {
    const [singleSource, singleTarget, pyramidSource, pyramidTarget] = await Promise.all([
      extractFeaturesGpuResident(device, sourcePixels, 320, 240, { maxFeatures: 400 }),
      extractFeaturesGpuResident(device, targetPixels, 640, 480, { maxFeatures: 400 }),
      extractFeaturesGpuPyramidResident(device, sourcePixels, 320, 240, { maxFeatures: 400 }),
      extractFeaturesGpuPyramidResident(device, targetPixels, 640, 480, { maxFeatures: 400 }),
    ]);
    resident.push(singleSource, singleTarget, pyramidSource, pyramidTarget);
    for (const features of resident) {
      assert.ok(features.count <= 400);
      assert.equal(features.capacity, 400);
    }
    const [singleSourceHost, singleTargetHost, pyramidSourceHost, pyramidTargetHost] =
      await Promise.all([
        singleSource.read(),
        singleTarget.read(),
        pyramidSource.read(),
        pyramidTarget.read(),
      ]);
    const [singleMatches, pyramidMatches] = await Promise.all([
      matchResidentFeaturesGpu(device, singleSource, singleTarget, {
        ratio: 0.9,
        maxDistance: 112,
      }),
      matchResidentFeaturesGpu(device, pyramidSource, pyramidTarget, {
        ratio: 0.9,
        maxDistance: 112,
      }),
    ]);
    const singleInliers = affineInlierCount(
        singleMatches,
        singleSourceHost,
        singleTargetHost,
        2,
        8.5,
        6.5,
      ),
      pyramidInliers = affineInlierCount(
        pyramidMatches,
        pyramidSourceHost,
        pyramidTargetHost,
        2,
        8.5,
        6.5,
      );
    t.diagnostic(
      `known 2x transform support: single=${singleInliers}, pyramid=${pyramidInliers}; ` +
        `mutual matches single=${singleMatches.matches.length / 2}, pyramid=${pyramidMatches.matches.length / 2}`,
    );
    assert.ok(pyramidInliers >= 8, `pyramid found ${pyramidInliers} known-transform inliers`);
    assert.ok(
      pyramidInliers > singleInliers,
      `pyramid ${pyramidInliers} vs single ${singleInliers} inliers`,
    );
  } finally {
    await Promise.all(resident.map((features) => features.dispose()));
    await device.dispose();
  }
});

test('pyramid extraction releases the point buffer if descriptor allocation fails', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const rgba = new Uint8Array(320 * 240 * 4);
  rgba.fill(255);
  const before = device.statistics,
    allocateU32 = device.allocateU32;
  let injected = false;
  device.allocateU32 = (length, values) => {
    if (!injected && length === 400 * 8) {
      injected = true;
      throw new Error('injected merged-descriptor allocation failure');
    }
    return allocateU32.call(device, length, values);
  };
  try {
    await assert.rejects(
      extractFeaturesGpuPyramidResident(device, rgba, 320, 240, { maxFeatures: 400 }),
      /injected merged-descriptor allocation failure/,
    );
    assert.equal(injected, true);
    assert.equal(device.statistics.liveBuffers, before.liveBuffers);
    assert.equal(device.statistics.liveBytes, before.liveBytes);
  } finally {
    device.allocateU32 = allocateU32;
    await device.dispose();
  }
});

test('smoothed BRIEF improves noisy translation precision while the default path stays identical', async (t) => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  const gray = new Uint8Array(320 * 240);
  let textureSeed = 0x1234abcd;
  for (let i = 0; i < gray.length; i++) {
    textureSeed ^= textureSeed << 13;
    textureSeed ^= textureSeed >>> 17;
    textureSeed ^= textureSeed << 5;
    gray[i] = textureSeed & 255;
  }
  const source = texturedFrame(gray, 320, 240),
    target = noisyTranslation(gray, 320, 240, 5, 3, 19),
    resident = [];
  try {
    const [defaultFeatures, explicitDefault, noisySource, noisyTarget, smoothSource, smoothTarget] =
      await Promise.all([
        extractFeaturesGpuResident(device, source, 320, 240, { maxFeatures: 700, threshold: 0.1 }),
        extractFeaturesGpuResident(device, source, 320, 240, {
          maxFeatures: 700,
          threshold: 0.1,
          smoothDescriptors: false,
        }),
        extractFeaturesGpuResident(device, source, 320, 240, {
          maxFeatures: 700,
          threshold: 0.1,
          smoothDescriptors: false,
        }),
        extractFeaturesGpuResident(device, target, 320, 240, {
          maxFeatures: 700,
          threshold: 0.1,
          smoothDescriptors: false,
        }),
        extractFeaturesGpuResident(device, source, 320, 240, {
          maxFeatures: 700,
          threshold: 0.1,
          smoothDescriptors: true,
        }),
        extractFeaturesGpuResident(device, target, 320, 240, {
          maxFeatures: 700,
          threshold: 0.1,
          smoothDescriptors: true,
        }),
      ]);
    resident.push(
      defaultFeatures,
      explicitDefault,
      noisySource,
      noisyTarget,
      smoothSource,
      smoothTarget,
    );
    const [
      defaultHost,
      explicitDefaultHost,
      rawSourceHost,
      rawTargetHost,
      smoothSourceHost,
      smoothTargetHost,
    ] = await Promise.all(resident.map((features) => features.read()));
    assert.deepEqual(defaultHost.keypoints, explicitDefaultHost.keypoints);
    assert.deepEqual(defaultHost.scores, explicitDefaultHost.scores);
    assert.deepEqual(defaultHost.descriptors, explicitDefaultHost.descriptors);
    const [rawMatches, smoothMatches] = await Promise.all([
      matchResidentFeaturesGpu(device, noisySource, noisyTarget, { ratio: 0.98, maxDistance: 160 }),
      matchResidentFeaturesGpu(device, smoothSource, smoothTarget, {
        ratio: 0.98,
        maxDistance: 160,
      }),
    ]);
    const rawInliers = affineInlierCount(rawMatches, rawSourceHost, rawTargetHost, 1, 5, 3),
      smoothInliers = affineInlierCount(smoothMatches, smoothSourceHost, smoothTargetHost, 1, 5, 3),
      rawPrecision = rawMatches.matches.length ? rawInliers / (rawMatches.matches.length / 2) : 0,
      smoothPrecision = smoothMatches.matches.length
        ? smoothInliers / (smoothMatches.matches.length / 2)
        : 0;
    t.diagnostic(
      `noisy translation: raw=${rawInliers}/${rawMatches.matches.length / 2} ` +
        `(${rawPrecision.toFixed(3)}), smooth=${smoothInliers}/${smoothMatches.matches.length / 2} ` +
        `(${smoothPrecision.toFixed(3)})`,
    );
    assert.ok(smoothInliers >= 8, `smoothed BRIEF recovered ${smoothInliers} inliers`);
    assert.ok(
      smoothPrecision >= rawPrecision,
      `precision smoothed=${smoothPrecision} (${smoothInliers} inliers) vs raw=${rawPrecision} (${rawInliers})`,
    );
  } finally {
    await Promise.all(resident.map((features) => features.dispose()));
    await device.dispose();
  }
});
