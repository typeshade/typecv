import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

// Exercises the Apple fp64 flavor explicitly on every adapter. Prepared JPEGs cannot reproduce
// Safari's native driver, 4K video decoder or GPU-process memory pressure.
// Full integer essential compilation exceeds this runner's software-GPU resources. Keep the
// guide source explicit, and retain an opt-in native probe instead of silently falling back.
const gpuGuide = process.env.TYPECV_INTEGER_GUIDE_GPU === '1';
test(`integer-selected WebGPU device completes pyramid SIFT and guided matching (${gpuGuide ? 'GPU' : 'CPU-reference'} guide)`, async () => {
  const serve = Object.fromEntries(
    [0, 1, 2].map((i) => [
      `/sample/${i}.jpg`,
      new URL(`../fixtures/lab-images/pyramid/frame-0${i}.jpg`, import.meta.url).pathname,
    ]),
  );
  const { result, errors } = await inBrowser(
    async ({ gpuGuide }) => {
      const api = await import('/dist/gpu.js');
      let phase = 'opening',
        lastKernel = '';
      const device = await api.ShadeDevice.open({
        adapterLimits: true,
        fp64Flavor: 'integer',
        workingBytes: 256 * 1024 ** 2,
        poolBytes: 128 * 1024 ** 2,
        solverCheckpoints: true,
        onProgress: (event) => {
          lastKernel = event.kernel;
        },
      });
      const reference = gpuGuide ? undefined : await api.ShadeDevice.open({ backend: 'cpu' });
      const width = 640,
        height = 360,
        camera = { fx: 512, fy: 512, cx: 320, cy: 180, k1: 0, k2: 0 },
        features = [],
        pairs = [];
      try {
        for (let i = 0; i < 3; i++) {
          phase = 'SIFT frame ' + i;
          const response = await fetch(`/sample/${i}.jpg`);
          if (!response.ok) throw new Error('Missing pyramid frame');
          const image = await createImageBitmap(await response.blob());
          const canvas = new OffscreenCanvas(width, height);
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          ctx.drawImage(image, 0, 0, width, height);
          image.close();
          const feature = await api.extractSiftGpuResident(
            device,
            ctx.getImageData(0, 0, width, height).data,
            width,
            height,
            { firstOctave: -1, maxFeatures: 16384 },
          );
          const points = await feature.readKeypoints();
          console.debug('integer: extracted frame', i, feature.count);
          for (const other of features) {
            phase = 'plain matching frame ' + i;
            const plain = await api.matchFloatDescriptorsGpu(
              device,
              other.feature.descriptors,
              feature.descriptors,
              {
                ratio: 0.8,
                mutual: true,
              },
            );
            const pixels = new Float64Array(plain.matches.length * 2);
            for (let j = 0; j < plain.matches.length; j += 2) {
              const a = plain.matches[j] * 2,
                b = plain.matches[j + 1] * 2;
              pixels.set([other.points[a], other.points[a + 1], points[b], points[b + 1]], j * 2);
            }
            phase = 'essential guide frame ' + i;
            console.debug('integer: estimating guide', i, plain.matches.length / 2);
            const guide = await api.estimateEssentialRansacGpu(
              reference ?? device,
              pixels,
              camera,
              camera,
              {
                iterations: 50,
                minIterations: 50,
                threshold: 3 / 512,
                seed: 101 + i,
              },
            );
            const normalize = (source) =>
              Float64Array.from(source, (v, k) => (v - (k % 2 ? 180 : 320)) / 512);
            phase = 'guided matching frame ' + i;
            const guided = await api.matchFloatDescriptorsGpu(
              device,
              other.feature.descriptors,
              feature.descriptors,
              {
                ratio: 0.8,
                mutual: true,
                epipolar: {
                  fundamental: guide.E,
                  firstPoints: normalize(other.points),
                  secondPoints: normalize(points),
                  maxLinePx: 4 / 512,
                },
              },
            );
            pairs.push({
              plain: plain.matches.length / 2,
              guided: guided.matches.length / 2,
              inliers: guide.inlierCount,
              finite: [...guide.E, ...guided.distances].every(Number.isFinite),
              tiers: [plain.report.tier, guide.execution.report.tier, guided.report.tier],
            });
          }
          features.push({ feature, points });
        }
        return {
          flavor: device.fp64Flavor,
          counts: features.map(({ feature }) => feature.count),
          pairs,
          diagnostics: device.diagnostics,
          kernels: device.kernelStatistics.rows.map((r) => r.name),
        };
      } catch (error) {
        throw new Error(
          'Integer check failed during ' +
            phase +
            ', last kernel ' +
            lastKernel +
            ': ' +
            error +
            '; diagnostics ' +
            JSON.stringify(device.diagnostics),
          { cause: error },
        );
      } finally {
        for (const { feature } of features) await feature.dispose();
        await reference?.dispose();
        await device.dispose();
      }
    },
    { gpuGuide },
    { serve, timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.flavor, 'integer');
  assert.equal(result.counts.length, 3);
  assert.ok(result.counts.every((n) => n > 80 && n <= 16384));
  assert.equal(result.pairs.length, 3);
  for (const pair of result.pairs) {
    assert.ok(pair.plain > 20 && pair.guided > 20 && pair.inliers > 20, JSON.stringify(pair));
    assert.equal(pair.finite, true);
    assert.deepEqual(pair.tiers, ['webgpu', gpuGuide ? 'webgpu' : 'cpu', 'webgpu']);
  }
  assert.equal(result.diagnostics.disposalRequested, false);
  assert.deepEqual(result.diagnostics.recentErrors, []);
  if (gpuGuide) assert.ok(result.kernels.some((name) => name.includes('appendPairModelRows')));
  assert.ok(result.kernels.some((name) => name.includes('scatterFloatMatches')));
});
