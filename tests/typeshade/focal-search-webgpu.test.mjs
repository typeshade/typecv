import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental } from '../../dist/index.js';
import { orbitFixture } from './focal-search-cases.mjs';
import { inBrowser } from './browser.mjs';

test('adaptive focal search on WebGPU agrees with the CPU reference and the known focal', async () => {
  const fixture = orbitFixture(),
    options = {
      ransacIterations: 128,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      focalSearch: { views: 4 },
    },
    reference = reconstructIncremental(fixture.images, fixture.pairs, options);
  const { result, errors } = await inBrowser(
    async ({ images, pairs, options, hardware }) => {
      const { ShadeDevice, reconstructIncrementalGpu } = await import('/dist/gpu.js');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const description = [
        adapter.info?.vendor,
        adapter.info?.architecture,
        adapter.info?.description,
      ].join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter)
      )
        throw new Error(`Hardware requested, software adapter selected: ${description}`);
      const gpu = await adapter.requestDevice(),
        validation = [];
      gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
      const device = await ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 });
      try {
        const run = await reconstructIncrementalGpu(
          device,
          images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
          pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
          options,
        );
        return {
          search: run.diagnostics.focalSearch,
          views: run.reconstruction.views.length,
          focals: run.reconstruction.views.map((v) => v.camera.fx),
          landmarks: run.reconstruction.landmarks.length,
          validation,
          live: device.statistics.liveBuffers,
        };
      } finally {
        await device.dispose();
        gpu.destroy();
      }
    },
    {
      images: fixture.images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
      pairs: fixture.pairs.map((p) => ({ ...p, matches: [...p.matches] })),
      options,
      hardware: process.env.TYPECV_HARDWARE_GPU === '1',
    },
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.validation, []);
  assert.equal(result.live, 0);
  assert.equal(result.views, fixture.images.length);
  assert.ok(result.landmarks > 0);
  assert.deepEqual(result.search.views, [0, 1, 2, 3]);
  assert.ok(Math.abs(result.search.ratio / reference.diagnostics.focalSearch.ratio - 1) < 1e-3);
  for (const focal of result.focals)
    assert.ok(Math.abs(focal / fixture.width / fixture.ratio - 1) < 0.02);
});
