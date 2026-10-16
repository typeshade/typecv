import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  extractFeaturesGpu,
  extractFeaturesTiledGpu,
  featureTileRegions,
} from '../../dist/gpu.js';

function pixelsFor(width, height) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  let seed = 0x6d2b79f5;
  for (let i = 0; i < width * height; i++) {
    seed = Math.imul(seed ^ (seed >>> 15), seed | 1);
    seed ^= seed + Math.imul(seed ^ (seed >>> 7), seed | 61);
    const gray = ((seed ^ (seed >>> 14)) >>> 0) & 255;
    const at = i * 4;
    pixels[at] = gray;
    pixels[at + 1] = gray;
    pixels[at + 2] = gray;
    pixels[at + 3] = 255;
  }
  return pixels;
}

function regionSource(width, height, pixels, tileEdge = 96) {
  const regions = [];
  return {
    regions,
    source: {
      width,
      height,
      readRegion(region) {
        assert.ok(Number.isSafeInteger(region.x) && Number.isSafeInteger(region.y));
        assert.ok(Number.isSafeInteger(region.width) && Number.isSafeInteger(region.height));
        assert.ok(region.x >= 0 && region.y >= 0, `outside source: ${JSON.stringify(region)}`);
        assert.ok(region.width >= 48 && region.height >= 48);
        assert.ok(region.x + region.width <= width && region.y + region.height <= height);
        assert.ok(region.width <= tileEdge + 48 && region.height <= tileEdge + 48);
        regions.push({ ...region });
        const output = new Uint8ClampedArray(region.width * region.height * 4);
        for (let row = 0; row < region.height; row++) {
          const from = ((region.y + row) * width + region.x) * 4;
          output.set(pixels.subarray(from, from + region.width * 4), row * region.width * 4);
        }
        return output;
      },
    },
  };
}

function rows(features) {
  const values = [];
  for (let i = 0; i < features.scores.length; i++) {
    values.push({
      x: features.keypoints[2 * i],
      y: features.keypoints[2 * i + 1],
      score: features.scores[i],
      descriptor: Array.from(features.descriptors.subarray(i * 8, i * 8 + 8)),
    });
  }
  return values.sort((a, b) => a.y - b.y || a.x - b.x);
}

async function compareFullAndTiled(device, width, height) {
  const pixels = pixelsFor(width, height),
    provider = regionSource(width, height, pixels),
    full = await extractFeaturesGpu(device, pixels, width, height, { maxFeatures: 2000 }),
    tiled = await extractFeaturesTiledGpu(device, provider.source, {
      tileEdge: 96,
      maxFeatures: 2000,
    });
  assert.deepEqual(rows(tiled), rows(full));
  assert.equal(tiled.tiling.sourceWidth, width);
  assert.equal(tiled.tiling.sourceHeight, height);
  assert.equal(tiled.tiling.tiles, provider.regions.length);
  assert.ok(provider.regions.length > 1);
  return { pixels, provider, full, tiled };
}

test('tiled FAST/BRIEF matches full-frame output across seams and odd source edges', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    await compareFullAndTiled(device, 192, 144);
    await compareFullAndTiled(device, 193, 145);
  } finally {
    await device.dispose();
  }
});

test('tiled maxFeatures keeps the deterministic global top scores with stable ties', async () => {
  const width = 193,
    height = 145,
    pixels = pixelsFor(width, height),
    provider = regionSource(width, height, pixels),
    device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const all = await extractFeaturesTiledGpu(device, provider.source, {
        tileEdge: 96,
        maxFeatures: 2000,
      }),
      capped = await extractFeaturesTiledGpu(device, provider.source, {
        tileEdge: 96,
        maxFeatures: 25,
      }),
      repeated = await extractFeaturesTiledGpu(device, provider.source, {
        tileEdge: 96,
        maxFeatures: 25,
      });
    const expected = rows(all)
      .sort((a, b) => b.score - a.score || a.y - b.y || a.x - b.x)
      .slice(0, 25)
      .sort((a, b) => a.y - b.y || a.x - b.x);
    assert.equal(capped.scores.length, expected.length);
    assert.deepEqual(rows(capped), expected);
    assert.deepEqual(rows(repeated), rows(capped));
  } finally {
    await device.dispose();
  }
});

test('lazy tile planner covers a 32K by 16K source with bounded, in-bounds halo reads', () => {
  const width = 32768,
    height = 16384,
    edge = 480,
    nx = Math.ceil(width / edge),
    ny = Math.ceil(height / edge);
  let count = 0,
    covered = 0;
  for (const { core, read } of featureTileRegions(width, height, edge)) {
    const column = count % nx,
      row = Math.floor(count / nx);
    assert.deepEqual(core, {
      x: column * edge,
      y: row * edge,
      width: Math.min(edge, width - column * edge),
      height: Math.min(edge, height - row * edge),
    });
    assert.ok(read.x >= 0 && read.y >= 0);
    assert.ok(read.x + read.width <= width && read.y + read.height <= height);
    assert.ok(read.width >= 48 && read.height >= 48);
    assert.ok(read.width <= edge + 48 && read.height <= edge + 48);
    assert.ok(read.x <= core.x && read.y <= core.y);
    assert.ok(read.x + read.width >= core.x + core.width);
    assert.ok(read.y + read.height >= core.y + core.height);
    covered += core.width * core.height;
    count++;
  }
  assert.equal(count, nx * ny);
  assert.equal(covered, width * height);
});
