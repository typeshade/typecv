import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU tiled FAST/BRIEF matches full-frame output and streams beyond 16K width', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, extractFeaturesGpu, extractFeaturesTiledGpu } =
        await import('/dist/gpu.js');
      const pixelsFor = (width, height) => {
        const pixels = new Uint8ClampedArray(width * height * 4);
        let seed = 0x6d2b79f5;
        for (let i = 0; i < width * height; i++) {
          seed = Math.imul(seed ^ (seed >>> 15), seed | 1);
          seed ^= seed + Math.imul(seed ^ (seed >>> 7), seed | 61);
          const gray = ((seed ^ (seed >>> 14)) >>> 0) & 255;
          pixels.set([gray, gray, gray, 255], i * 4);
        }
        return pixels;
      };
      const rows = (features) => {
        const values = [];
        for (let i = 0; i < features.scores.length; i++)
          values.push({
            x: features.keypoints[2 * i],
            y: features.keypoints[2 * i + 1],
            score: features.scores[i],
            descriptor: Array.from(features.descriptors.subarray(i * 8, i * 8 + 8)),
          });
        return values.sort((a, b) => a.y - b.y || a.x - b.x);
      };
      const streamProcedural = async (device, width, height, edge) => {
        const regions = [];
        const features = await extractFeaturesTiledGpu(
          device,
          {
            width,
            height,
            readRegion({ x, y, width: regionWidth, height: regionHeight }) {
              if (
                x < 0 ||
                y < 0 ||
                x + regionWidth > width ||
                y + regionHeight > height ||
                regionWidth < 48 ||
                regionHeight < 48 ||
                regionWidth > edge + 48 ||
                regionHeight > edge + 48
              )
                throw new RangeError('Tiled reader requested an unbounded or out-of-bounds region');
              regions.push({ x, y, width: regionWidth, height: regionHeight });
              const region = new Uint8ClampedArray(regionWidth * regionHeight * 4);
              for (let row = 0; row < regionHeight; row++) {
                for (let column = 0; column < regionWidth; column++) {
                  const gx = x + column,
                    gy = y + row,
                    value = Math.imul(gx ^ (gy * 0x45d9f3b), 0x27d4eb2d) >>> 0,
                    gray = (value ^ (value >>> 13)) & 255,
                    at = (row * regionWidth + column) * 4;
                  region[at] = gray;
                  region[at + 1] = gray;
                  region[at + 2] = gray;
                  region[at + 3] = 255;
                }
              }
              return region;
            },
          },
          { tileEdge: edge, maxFeatures: 2000 },
        );
        return {
          sourceWidth: features.tiling.sourceWidth,
          sourceHeight: features.tiling.sourceHeight,
          tiles: features.tiling.tiles,
          expectedTiles: Math.ceil(width / edge) * Math.ceil(height / edge),
          peakRegionBytes: features.tiling.peakRegionBytes,
          maxRegionBytes: (edge + 48) * (edge + 48) * 4,
          reads: regions.length,
          featureCount: features.scores.length,
          maximumSourceX: Math.max(...Array.from(features.keypoints).filter((_, i) => i % 2 === 0)),
        };
      };
      const device = await ShadeDevice.open({ backend: 'webgpu' });
      const differential = [];
      try {
        for (const [width, height] of [
          [192, 144],
          [193, 145],
        ]) {
          const pixels = pixelsFor(width, height);
          const full = await extractFeaturesGpu(device, pixels, width, height, {
            maxFeatures: 2000,
          });
          const tiled = await extractFeaturesTiledGpu(
            device,
            {
              width,
              height,
              readRegion({ x, y, width: regionWidth, height: regionHeight }) {
                const region = new Uint8ClampedArray(regionWidth * regionHeight * 4);
                for (let row = 0; row < regionHeight; row++) {
                  const from = ((y + row) * width + x) * 4;
                  region.set(pixels.subarray(from, from + regionWidth * 4), row * regionWidth * 4);
                }
                return region;
              },
            },
            { tileEdge: 96, maxFeatures: 2000 },
          );
          differential.push({ full: rows(full), tiled: rows(tiled) });
        }

        return {
          differential,
          large2D: await streamProcedural(device, 8192, 4096, 480),
          wideThin: await streamProcedural(device, 32768, 48, 480),
        };
      } finally {
        await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  for (const { full, tiled } of result.differential) assert.deepEqual(tiled, full);
  for (const [resultKey, width, height] of [
    ['large2D', 8192, 4096],
    ['wideThin', 32768, 48],
  ]) {
    const streamed = result[resultKey];
    assert.equal(streamed.sourceWidth, width);
    assert.equal(streamed.sourceHeight, height);
    assert.equal(streamed.tiles, Math.ceil(width / 480) * Math.ceil(height / 480));
    assert.equal(streamed.expectedTiles, streamed.tiles);
    assert.equal(streamed.reads, streamed.tiles);
    assert.ok(streamed.featureCount <= 2000);
    assert.equal(streamed.maxRegionBytes, 528 * 528 * 4);
    assert.ok(streamed.peakRegionBytes <= streamed.maxRegionBytes);
    assert.ok(streamed.maximumSourceX > width / 2);
  }
  console.log(
    JSON.stringify({ tiledWebGpu: { large2D: result.large2D, wideThin: result.wideThin } }),
  );
});
