import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental } from '../../dist/index.js';
import { orbitFixture } from './focal-search-cases.mjs';
import { inBrowser } from './browser.mjs';
import { focalComponentsFixture } from '../fixtures/focal-components.ts';

for (const { label, fixture, subset, registered } of [
  { label: 'connected', fixture: orbitFixture(), subset: [0, 1, 2, 3], registered: 6 },
  {
    label: 'disconnected prefix',
    fixture: focalComponentsFixture(),
    subset: [10, 11, 12, 13, 14],
    registered: 6,
  },
]) {
  test(`adaptive focal search on WebGPU agrees with CPU and known focal (${label})`, async () => {
    const options = {
        ransacIterations: label === 'connected' ? 128 : 64,
        bundleIterations: 4,
        finalBundleIterations: 8,
        refineSharedFocal: true,
        focalSearch: {
          views: subset.length,
          ...(label === 'connected' ? {} : { refineIterations: 12 }),
        },
      },
      reference = reconstructIncremental(fixture.images, fixture.pairs, options);
    const { result, errors } = await inBrowser(
      async ({ images, pairs, options, hardware, streaming }) => {
        const { ShadeDevice, reconstructIncrementalGpu, IncrementalSfmSession } =
          await import('/dist/gpu.js');
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
          const inputs = images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) }));
          const edges = pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) }));
          let session;
          let run;
          if (streaming) {
            session = new IncrementalSfmSession(device, options);
            const known = new Set();
            for (const image of inputs) {
              const added = await session.add(
                image,
                edges.filter(
                  (p) =>
                    (p.first === image.id && known.has(p.second)) ||
                    (p.second === image.id && known.has(p.first)),
                ),
              );
              known.add(image.id);
              if (image.id < 14 && (added || session.focalSearch))
                throw new Error('Focal calibrated before five connected views');
            }
            run = await session.finish();
          } else run = await reconstructIncrementalGpu(device, inputs, edges, options);
          return {
            search: session?.focalSearch ?? run.diagnostics.focalSearch,
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
        streaming: label === 'disconnected prefix',
      },
      { timeoutMs: 180000 },
    );
    assert.deepEqual(errors, []);
    assert.deepEqual(result.validation, []);
    assert.equal(result.live, 0);
    assert.equal(result.views, registered);
    assert.ok(result.landmarks > 0);
    assert.deepEqual(result.search.views, subset);
    assert.ok(Math.abs(result.search.ratio / reference.diagnostics.focalSearch.ratio - 1) < 1e-3);
    for (const focal of result.focals)
      assert.ok(Math.abs(focal / fixture.width / fixture.ratio - 1) < 0.02);
  });
}
