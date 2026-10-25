import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  reconstructDense,
  planMvsViews,
  mvsPhotometricGpu,
  decodeMvsNormal,
} from '../../dist/internal/gpu.js';
import { mvsScene } from './mvs-cases.mjs';

for (const fusion of ['pixel', 'tsdf'])
  test(`full diagnostic capture leaves ${fusion} reconstruction unchanged`, async () => {
    const scene = mvsScene({ width: 40, height: 30 });
    const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    const options = {
      fusion,
      planarDepth: fusion === 'tsdf',
      photometricIterations: 2,
      geometricIterations: 1,
      windowRadius: 3,
      windowStep: 1,
    };
    try {
      const baseline = await reconstructDense(device, scene, scene.images, options);
      const maps = [];
      const result = await reconstructDense(device, scene, scene.images, {
        ...options,
        onDepthDiagnostic: async (e) => {
          await Promise.resolve();
          maps.push({
            ...e,
            depth: e.depth.slice(),
            confidence: e.confidence?.slice(),
            normal: e.normal?.slice(),
            cost: e.cost?.slice(),
          });
        },
      });
      assert.ok(baseline.points.count > 0);
      assert.deepEqual(result.points, baseline.points);
      assert.deepEqual(result.views, baseline.views);
      const plan = planMvsViews(scene);
      for (const v of result.views.filter((v) => v.status === 'done')) {
        const stages = maps.filter((e) => e.viewId === v.id);
        assert.deepEqual(
          stages.map((e) => e.stage),
          [
            'photometric',
            'geometric-unfiltered',
            'geometric-filtered',
            ...(fusion === 'tsdf' ? ['planar'] : []),
          ],
        );
        for (const e of stages) {
          assert.equal(e.width, 40);
          assert.equal(e.height, 30);
          assert.equal(e.depth.length, 1200);
          assert.equal(e.normal.length, 1200);
          for (let i = 0; i < e.depth.length; i++)
            if (e.depth[i] > 0)
              assert.ok(Math.abs(Math.hypot(...decodeMvsNormal(e.normal[i])) - 1) < 1e-6);
          if (['photometric', 'geometric-unfiltered'].includes(e.stage)) {
            assert.equal(e.cost.length, 1200);
            assert.equal(
              e.costMeaning,
              e.stage === 'photometric' ? 'photometric' : 'photometric-geometric',
            );
            assert.ok(e.cost.every((v) => Number.isFinite(v) && v >= 0 && v <= 2.9 + 1e-6));
            assert.equal(e.confidence, undefined);
          } else {
            // The filter reuses the GPU cost buffer for confidence. It must not be archived
            // under the raw-cost meaning, even if both arrays happen to be in [0,1].
            assert.equal(e.cost, undefined);
            assert.equal(e.costMeaning, undefined);
          }
          assert.equal(e.depthScaleToModel, plan.normalization.scale);
          assert.deepEqual(e.pose, scene.views.find((v) => v.id === e.viewId).pose);
          assert.deepEqual(e.sourceViewIds, v.sources);
        }
        const filtered = stages.find((e) => e.stage === 'geometric-filtered');
        const unfiltered = stages.find((e) => e.stage === 'geometric-unfiltered');
        assert.ok(filtered.confidence);
        assert.deepEqual(filtered.normal, unfiltered.normal);
        for (let i = 0; i < filtered.depth.length; i++) {
          assert.ok(filtered.confidence[i] >= 0 && filtered.confidence[i] <= 1);
          if (filtered.depth[i] > 0) assert.equal(filtered.depth[i], unfiltered.depth[i]);
          else assert.equal(filtered.confidence[i], 0);
        }
        const final = stages.at(-1);
        assert.equal(
          final.depth.reduce((n, z) => n + (z > 0), 0),
          v.depthPixels,
        );
      }
      // The diagnostic must expose the actual optimized normal and cost, not a normal
      // reconstructed from the depth gradient or a post-filter confidence map.
      const reference = plan.views.find((v) => v.status === 'ready').index;
      const direct = await mvsPhotometricGpu(device, scene, plan, scene.images, reference, options);
      const captured = maps.find(
        (e) => e.stage === 'photometric' && e.viewId === scene.views[plan.packed[reference]].id,
      );
      assert.deepEqual(captured.depth, direct.depth);
      assert.deepEqual(captured.normal, direct.normal);
      assert.deepEqual(captured.cost, direct.cost);
      assert.equal(device.statistics.liveBytes, 0);
    } finally {
      await device.dispose();
    }
  });
