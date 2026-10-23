import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';
import { scene } from '../scene.ts';
import { inBrowser } from './browser.mjs';

test('WebGPU refines disconnected carried-map components with fixed gauges and no invented observations', async () => {
  const { reconstruction } = scene({ count: 48, views: 7, noise: 0.15 });
  const model = {
    ...reconstruction,
    landmarks: reconstruction.landmarks.map((p, i) => ({
      ...p,
      xyz: p.xyz.map((x) => x + 0.015),
      observations: p.observations.filter((o) =>
        (i < 24 ? [10, 20, 30] : [40, 50, 60]).includes(o.viewId),
      ),
    })),
  };
  const directory = await mkdtemp(join(tmpdir(), 'typecv-bundle-partitions-'));
  try {
    const control = join(directory, 'control.mjs');
    await build({
      entryPoints: [resolve('src/reconstruction/local-bundle.ts')],
      outfile: control,
      bundle: true,
      platform: 'browser',
      format: 'esm',
    });
    const { result, errors } = await inBrowser(
      async (raw) => {
        const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
        const { planConnectedBundles, mergeBundleRefinements } = await import('/control.mjs');
        const model = {
          views: raw.views.map((v) => ({
            ...v,
            pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
          })),
          landmarks: raw.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
        };
        const device = await ShadeDevice.open({ poolBytes: 0 });
        const rows = [];
        try {
          for (const input of [
            model,
            {
              views: model.views.filter((v) => [10, 20, 70].includes(v.id)),
              landmarks: model.landmarks.slice(0, 24).map((p) => ({
                ...p,
                observations: p.observations.filter((o) => o.viewId !== 30),
              })),
            },
          ]) {
            const snapshot = JSON.stringify(input);
            let refused = false;
            try {
              await bundleAdjustGpu(device, input, { iterations: 4 });
            } catch (error) {
              refused = error.message.includes('one connected observation graph');
            }
            const plan = planConnectedBundles(input, {
              iterations: 4,
              refineSharedFocal: true,
            });
            const refined = [],
              costs = [];
            for (const part of plan.problems) {
              const solved = await bundleAdjustGpu(device, part.reconstruction, part.options);
              costs.push([solved.diagnostics.initialCost, solved.diagnostics.finalCost]);
              refined.push(solved.reconstruction);
            }
            const merged = mergeBundleRefinements(input, refined);
            rows.push({
              refused,
              partition: plan.partition,
              costs,
              unchangedInput: snapshot === JSON.stringify(input),
              sameObservations:
                JSON.stringify(input.landmarks.map((p) => p.observations)) ===
                JSON.stringify(merged.landmarks.map((p) => p.observations)),
              sameCameras:
                JSON.stringify(input.views.map((v) => v.camera)) ===
                JSON.stringify(merged.views.map((v) => v.camera)),
              fixedPoses: plan.partition.fixedViewIds.every(
                (id) =>
                  JSON.stringify(input.views.find((v) => v.id === id).pose) ===
                  JSON.stringify(merged.views.find((v) => v.id === id).pose),
              ),
              isolatedPose:
                JSON.stringify(input.views.find((v) => v.id === 70).pose) ===
                JSON.stringify(merged.views.find((v) => v.id === 70).pose),
              liveBytes: device.statistics.liveBytes,
            });
          }
        } finally {
          await device.dispose();
        }
        return rows;
      },
      JSON.parse(JSON.stringify(model, (_key, v) => (ArrayBuffer.isView(v) ? [...v] : v))),
      { serve: { '/control.mjs': control }, timeoutMs: 120000 },
    );
    assert.deepEqual(errors, []);
    assert.equal(result.length, 2);
    for (const row of result) {
      assert.equal(row.refused, true, 'the raw disconnected solve still rejects');
      assert.deepEqual(row.partition.deferredViewIds, [70]);
      assert.equal(row.partition.sharedFocalDeferred, true);
      assert.ok(row.costs.every(([before, after]) => after < before));
      assert.ok(row.unchangedInput && row.sameObservations && row.sameCameras);
      assert.ok(row.fixedPoses && row.isolatedPose);
      assert.equal(row.liveBytes, 0);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
