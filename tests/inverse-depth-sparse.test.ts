import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjustInverseDepth } from '../src/reconstruction/inverse-depth-bundle.ts';
import { scene } from './scene.ts';
import { projectPoint, identityPose } from '../src/geometry/pose.ts';

test('sparse and dense joint depth/focal solves agree on objective and all point marginals', () => {
  const { reconstruction } = scene({ count: 40, views: 6, noise: 0.15 });
  for (const refineSharedFocal of [false, true]) {
    const dense = bundleAdjustInverseDepth(reconstruction, [], {
        solver: 'dense',
        iterations: 2,
        refineSharedFocal,
      }),
      sparse = bundleAdjustInverseDepth(reconstruction, [], {
        solver: 'sparse',
        iterations: 2,
        refineSharedFocal,
      });
    assert.ok(dense.uncertainty.observable && sparse.uncertainty.observable);
    assert.ok(Math.abs(sparse.diagnostics.finalCost - dense.diagnostics.finalCost) < 1e-7);
    for (let i = 0; i < dense.estimates.length; i++)
      for (let k = 0; k < 3; k++) {
        const a = dense.estimates[i]!.covariance![4 * k]!,
          b = sparse.estimates[i]!.covariance![4 * k]!;
        assert.ok(Math.abs(b / a - 1) < 1e-6, `${refineSharedFocal}/${i}/${k}: ${b / a}`);
      }
    assert.equal(sparse.diagnostics.linearSolver, 'sparse');
    assert.ok(
      sparse.diagnostics.cameraCovarianceEntries < dense.diagnostics.cameraCovarianceEntries,
    );
  }
});

test('one zero-parallax finite seed is deferred without removing other point uncertainty', () => {
  const { reconstruction } = scene({ count: 30, views: 4, noise: 0 }),
    base = reconstruction.views[0]!,
    repeated = { ...base, id: 100 },
    good = reconstruction.landmarks.map((p) => ({
      ...p,
      observations: [
        ...p.observations,
        {
          viewId: repeated.id,
          featureId: p.trackId,
          pixel: projectPoint(base.camera, base.pose, p.xyz)!,
        },
      ],
    })),
    xyz = Float64Array.of(0.2, 0.1, 8),
    pixel = projectPoint(base.camera, base.pose, xyz)!,
    bad = {
      trackId: 1000,
      xyz,
      observations: [
        { viewId: base.id, featureId: 1000, pixel },
        { viewId: repeated.id, featureId: 1000, pixel },
      ],
    };
  for (const solver of ['dense', 'sparse'] as const) {
    const clean = bundleAdjustInverseDepth(
      { views: [...reconstruction.views, repeated], landmarks: good },
      [],
      { solver, iterations: 1 },
    );
    const result = bundleAdjustInverseDepth(
      { views: [...reconstruction.views, repeated], landmarks: [...good, bad] },
      [],
      { solver, iterations: 1 },
    );
    assert.ok(result.uncertainty.observable);
    assert.deepEqual(result.diagnostics.deferredTrackIds, [1000]);
    assert.ok(result.estimates.slice(0, 30).every((p) => p.covariance));
    for (let i = 0; i < 30; i++)
      assert.ok(
        Math.abs(result.estimates[i]!.covariance![8]! / clean.estimates[i]!.covariance![8]! - 1) <
          1e-10,
      );
    assert.ok(!result.estimates[30]!.covariance);
    assert.equal(result.estimates[30]!.informationDeferred, true);
    assert.equal(result.reconstruction.landmarks.length, 30);
    assert.equal(result.diagnostics.inputObservations - result.diagnostics.observations, 2);
    const third = reconstruction.views[1]!,
      recovered = bundleAdjustInverseDepth(
        {
          views: [...reconstruction.views, repeated],
          landmarks: [
            ...good,
            {
              ...bad,
              observations: [
                ...bad.observations,
                {
                  viewId: third.id,
                  featureId: 1000,
                  pixel: projectPoint(third.camera, third.pose, xyz)!,
                },
              ],
            },
          ],
        },
        [],
        { solver, iterations: 1 },
      );
    assert.deepEqual(recovered.diagnostics.deferredTrackIds, []);
    assert.ok(recovered.estimates[30]!.covariance);
    assert.equal(recovered.reconstruction.landmarks.length, 31);
  }
});

/** Camera graph with local overlapping point tracks: the constraints remain global and joint. */
function longScene(count: number) {
  const f = scene({ count: 12, views: 4, noise: 0 }),
    camera = f.registered[0]!.camera,
    views = Array.from({ length: count }, (_, id) => ({
      id,
      camera,
      pose: {
        ...identityPose(),
        t: Float64Array.of(-0.03 * id, -0.04 * Math.sin(id), -0.02 * Math.sin(id * 0.2)),
      },
    }));
  return {
    views,
    landmarks: Array.from({ length: count - 3 }, (_, start) =>
      f.points.map((xyz, j) => ({
        trackId: start * 12 + j,
        xyz,
        observations: views.slice(start, start + 4).map((v) => ({
          viewId: v.id,
          featureId: start * 12 + j,
          pixel: projectPoint(v.camera, v.pose, xyz)!,
        })),
      })),
    ).flat(),
  };
}
test('large globally variable camera graph exceeds the old cap with compact cross blocks and selected inverse', () => {
  const input = longScene(40),
    result = bundleAdjustInverseDepth(input, [], { iterations: 0 });
  assert.equal(result.diagnostics.cameraParameters, 233);
  assert.equal(result.diagnostics.linearSolver, 'sparse');
  assert.ok(result.uncertainty.observable);
  assert.ok(result.estimates.every((p) => p.covariance));
  assert.ok(
    result.diagnostics.pointCrossEntries < result.diagnostics.denseEquivalentCrossEntries / 5,
  );
  assert.ok(
    result.diagnostics.cameraCovarianceEntries < result.diagnostics.cameraParameters ** 2 / 3,
  );
  assert.throws(
    () => bundleAdjustInverseDepth(input, [], { iterations: 0, maxSchurEntries: 10 }),
    /budget/,
  );
  assert.throws(
    () =>
      bundleAdjustInverseDepth(input, [], {
        iterations: 0,
        solver: 'sparse',
        maxFactorEntries: 10,
      }),
    /budget/,
  );
});
