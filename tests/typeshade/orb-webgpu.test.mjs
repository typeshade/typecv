import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { inBrowser } from './browser.mjs';
import { extractOrbCpu } from '../../dist/gpu.js';
import { framePair, grayToRgba, hamming, mapPoint } from '../orb-images.ts';

const fixture = JSON.parse(
  readFileSync(new URL('../fixtures/orb-opencv.json', import.meta.url), 'utf8'),
);
const fixtureRgba = grayToRgba(
  new Uint8Array(inflateSync(Buffer.from(fixture.gray, 'base64'))),
  fixture.width,
  fixture.height,
);

test('WebGPU ORB agrees with the CPU reference and matches a known homography', async () => {
  const width = 320,
    height = 240,
    homography = [0.96, 0.18, 14, -0.14, 0.99, 9, 0.00035, -0.00025, 1],
    pair = framePair(5, width, height, homography),
    { result, errors } = await inBrowser(
      async ({ fixtureFrame, first, second, fixtureWidth, fixtureHeight, width, height }) => {
        const { ShadeDevice, extractOrbGpuResident, matchResidentFeaturesGpu } =
          await import('/dist/gpu.js');
        const adapter = await navigator.gpu.requestAdapter(),
          gpu = await adapter.requestDevice(),
          device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 });
        try {
          const started = performance.now(),
            orb = await extractOrbGpuResident(
              device,
              Uint8Array.from(fixtureFrame),
              fixtureWidth,
              fixtureHeight,
              {
                maxFeatures: 300,
              },
            ),
            host = await orb.read(),
            extractionMs = performance.now() - started,
            fixtureOut = {
              count: orb.count,
              requested: orb.requested,
              keypoints: Array.from(host.keypoints),
              descriptors: Array.from(host.descriptors),
              extractionMs,
              tier: orb.report.tier,
            };
          await orb.dispose();
          const a = await extractOrbGpuResident(device, Uint8Array.from(first), width, height, {
              maxFeatures: 600,
            }),
            b = await extractOrbGpuResident(device, Uint8Array.from(second), width, height, {
              maxFeatures: 600,
            }),
            matches = await matchResidentFeaturesGpu(device, a, b, { ratio: 0.8, maxDistance: 80 }),
            pairOut = {
              first: Array.from((await a.read()).keypoints),
              second: Array.from((await b.read()).keypoints),
              matches: Array.from(matches.matches),
            };
          await a.dispose();
          await b.dispose();
          return { fixtureOut, pairOut };
        } finally {
          await device.dispose();
        }
      },
      {
        fixtureFrame: Array.from(fixtureRgba),
        first: Array.from(pair.first),
        second: Array.from(pair.second),
        fixtureWidth: fixture.width,
        fixtureHeight: fixture.height,
        width,
        height,
      },
      { timeoutMs: 300000 },
    );
  assert.deepEqual(errors, []);
  assert.equal(result.fixtureOut.tier, 'webgpu');
  // The same frame through the reference: the keypoints of the two tiers coincide, and so do
  // almost all descriptors (the GPU rounds float operations in its own way at a few boundaries).
  const reference = extractOrbCpu(fixtureRgba, fixture.width, fixture.height, { maxFeatures: 300 }),
    gpu = result.fixtureOut,
    index = new Map();
  for (let i = 0; i < reference.count; i++)
    index.set(
      `${reference.keypoints[i * 2].toFixed(2)}:${reference.keypoints[i * 2 + 1].toFixed(2)}`,
      i,
    );
  let found = 0,
    worst = 0,
    identical = 0;
  const descriptors = Uint32Array.from(gpu.descriptors);
  for (let i = 0; i < gpu.count; i++) {
    const j = index.get(
      `${gpu.keypoints[i * 2].toFixed(2)}:${gpu.keypoints[i * 2 + 1].toFixed(2)}`,
    );
    if (j === undefined) continue;
    found++;
    const distance = hamming(descriptors, i, reference.descriptors, j);
    worst = Math.max(worst, distance);
    if (distance === 0) identical++;
  }
  assert.ok(
    Math.abs(gpu.count - reference.count) <= 3,
    `${gpu.count} keypoints against ${reference.count}`,
  );
  assert.ok(found >= 0.97 * reference.count, `${found} of ${reference.count} keypoints coincide`);
  assert.ok(
    identical >= 0.9 * found && worst <= 8,
    `${identical} of ${found} descriptors identical, worst ${worst} bits`,
  );
  // The matches on the moved frame obey the homography.
  const { first, second, matches } = result.pairOut;
  let inliers = 0;
  for (let m = 0; m < matches.length; m += 2) {
    const [x, y] = mapPoint(homography, first[matches[m] * 2], first[matches[m] * 2 + 1]);
    if (Math.hypot(second[matches[m + 1] * 2] - x, second[matches[m + 1] * 2 + 1] - y) <= 3)
      inliers++;
  }
  assert.ok(matches.length / 2 >= 40, `${matches.length / 2} matches`);
  assert.ok(
    inliers / (matches.length / 2) >= 0.85,
    `inlier ratio ${inliers}/${matches.length / 2}`,
  );
  process.stderr.write(
    `# WebGPU ORB (SwiftShader): ${gpu.count} keypoints in ${gpu.extractionMs.toFixed(0)} ms, ${identical}/${found} descriptors identical, ${inliers}/${matches.length / 2} inliers\n`,
  );
});
