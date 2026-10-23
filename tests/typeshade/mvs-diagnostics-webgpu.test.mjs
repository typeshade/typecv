import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU stage diagnostics retain native depth/confidence and leave fused points unchanged', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, reconstructDenseGpu } = await import('/dist/gpu.js');
      const { mvsScene } = await import('/tests/typeshade/mvs-cases.mjs');
      const scene = mvsScene({ width: 40, height: 30 });
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const options = {
        fusion: 'tsdf',
        planarDepth: true,
        photometricIterations: 2,
        geometricIterations: 1,
        windowRadius: 3,
        windowStep: 1,
      };
      try {
        const baseline = await reconstructDenseGpu(device, scene, scene.images, options);
        const maps = [];
        const captured = await reconstructDenseGpu(device, scene, scene.images, {
          ...options,
          onDepthDiagnostic: async (e) => {
            const compressed = await new Response(
              new Blob([e.depth]).stream().pipeThrough(new CompressionStream('gzip')),
            ).arrayBuffer();
            const decoded = new Float32Array(
              await new Response(
                new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip')),
              ).arrayBuffer(),
            );
            maps.push({
              stage: e.stage,
              viewId: e.viewId,
              width: e.width,
              height: e.height,
              lossless: decoded.every((v, i) => Object.is(v, e.depth[i])),
              confidence: e.confidence?.length,
            });
          },
        });
        const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
        return {
          maps,
          points: captured.points.count,
          same: ['positions', 'colors', 'normals'].every((k) =>
            same(captured.points[k], baseline.points[k]),
          ),
          liveBytes: device.statistics.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.ok(result.points > 0);
  assert.equal(result.same, true);
  assert.equal(result.liveBytes, 0);
  assert.equal(result.maps.length, 20);
  assert.ok(result.maps.every((m) => m.width === 40 && m.height === 30 && m.lossless));
  assert.ok(
    result.maps
      .filter((m) => ['geometric-filtered', 'planar'].includes(m.stage))
      .every((m) => m.confidence === 1200),
  );
});
