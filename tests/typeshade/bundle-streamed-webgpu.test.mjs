// docs/gpu-budget.md, sections 3 and 4, on WebGPU (SwiftShader without a physical adapter): the
// bundle adjustment streamed in windows and the triangulation in chunks give the results of the
// all-resident run, bit for bit, and a budget below the whole problem completes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU BA streamed in windows equals the resident run, fixed and shared focal', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const { bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const model = bundleFixture({ count: 60, views: 6 });
      const C = model.views.length,
        P = model.landmarks.length,
        M = model.landmarks.reduce((n, p) => n + p.observations.length, 0);
      const out = [];
      for (const shared of [false, true]) {
        const options = { iterations: 4, huberPx: 1, refineSharedFocal: shared };
        const doubles =
          18 * C +
          3 * P +
          2 * M +
          (12 * C + 3 * P) +
          (30 * C + 3 * P + (shared ? 5 : 0)) +
          26 +
          3 * Math.ceil(M / 64) +
          (12 * C + 3 * P + (shared ? 27 : 26)) +
          78 * C +
          36 * C +
          (shared ? 6 * C + 6 : 0);
        const floor = 8 * doubles + 4 * (3 * M + P + 2 * C + 2) + 8;
        const budgets = [512 * 1024 ** 2, floor + 64 * (shared ? 376 : 336) + 64 * 104];
        const runs = [];
        for (const workingBytes of budgets) {
          const device = await ShadeDevice.open({ workingBytes });
          try {
            const r = await bundleAdjustGpu(device, model, options);
            runs.push({
              tier: r.execution.report.tier,
              streamed: r.execution.report.groups.some((g) =>
                g.calls.includes('checkBundlePoints'),
              ),
              text: JSON.stringify([
                r.reconstruction.views.map((v) => [...v.pose.R, ...v.pose.t, v.camera.fx]),
                r.reconstruction.landmarks.map((l) => [...l.xyz]),
                r.diagnostics,
                r.cgIterations,
              ]),
              accepted: r.diagnostics.acceptedSteps,
            });
          } finally {
            await device.dispose();
          }
        }
        out.push({ shared, runs, resident: floor });
      }
      return out;
    },
    undefined,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  for (const { shared, runs } of result) {
    assert.equal(runs[0].tier, 'webgpu');
    assert.equal(runs[0].streamed, false, `resident ${shared}`);
    assert.equal(runs[1].streamed, true, `streamed ${shared}`);
    assert.ok(runs[0].accepted > 0);
    assert.equal(runs[1].text, runs[0].text, `shared ${shared}`);
  }
});

test('WebGPU triangulation in chunks of tracks equals one chunk', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, triangulateTracksGpu } = await import('/dist/gpu.js');
      const { triangulationFixture, triangulationDigest } =
        await import('/tests/typeshade/geometry-cases.mjs');
      const { tracks } = triangulationFixture(120);
      const digests = [];
      for (const workingBytes of [512 * 1024 ** 2, 20_000]) {
        const device = await ShadeDevice.open({ workingBytes });
        try {
          const r = await triangulateTracksGpu(device, tracks);
          digests.push({
            digest: await triangulationDigest(r),
            tier: r.execution.report.tier,
            dispatches: r.execution.report.dispatches,
          });
        } finally {
          await device.dispose();
        }
      }
      return digests;
    },
    undefined,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result[0].tier, 'webgpu');
  assert.equal(result[1].digest, result[0].digest);
  assert.ok(result[1].dispatches > result[0].dispatches, 'the small budget ran more chunks');
});
