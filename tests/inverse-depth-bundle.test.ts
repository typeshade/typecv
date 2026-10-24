import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjustInverseDepth } from '../src/reconstruction/inverse-depth-bundle.ts';
import { fitInverseDepth } from '../src/geometry/inverse-depth.ts';
import { cameraCenter, updatePose } from '../src/geometry/pose.ts';
import { scene } from './scene.ts';
import { seededRandom } from '../src/geometry/numerics.ts';

test('anchored joint BA improves poses/points, keeps gauge, and marginalizes camera uncertainty', () => {
  const { reconstruction, rng } = scene({ count: 30, views: 4, noise: 0.1 }),
    input = {
      views: reconstruction.views.map((v, i) =>
        i === 0
          ? v
          : {
              ...v,
              pose: updatePose(
                v.pose,
                Float64Array.from(
                  [0.002, 0.002, 0.002, 0.01, 0.01, 0.01],
                  (s) => (rng() - 0.5) * s,
                ),
              ),
            },
      ),
      landmarks: reconstruction.landmarks.map((p) => ({
        ...p,
        xyz: p.xyz.map((x) => x + (rng() - 0.5) * 0.03),
      })),
    },
    before = structuredClone(input),
    result = bundleAdjustInverseDepth(input, [], { iterations: 12 });
  assert.ok(
    result.diagnostics.finalCost < result.diagnostics.initialCost * 0.1,
    JSON.stringify(result.diagnostics),
  );
  assert.equal(result.uncertainty.observable, true);
  assert.equal(result.uncertainty.poses, 'joint');
  assert.deepEqual(input, before);
  assert.deepEqual(result.reconstruction.views[0]!.pose, input.views[0]!.pose);
  const baseline = (m: typeof result.reconstruction) =>
    Math.hypot(
      ...cameraCenter(m.views[0]!.pose).map((x, i) => x - cameraCenter(m.views[1]!.pose)[i]!),
    );
  assert.ok(Math.abs(baseline(input) - baseline(result.reconstruction)) < 1e-10);
  const p = result.estimates[0]!,
    byId = new Map(result.reconstruction.views.map((v) => [v.id, v])),
    fixed = fitInverseDepth(
      p.observations.map((o) => ({ ...byId.get(o.viewId)!, pixel: o.pixel })),
    );
  assert.ok(p.covariance);
  assert.ok(fixed.covariance);
  assert.ok(
    p.covariance![8]! > fixed.covariance![8]! * 1.01,
    `${p.covariance![8]} vs ${fixed.covariance![8]}`,
  );
});

test('joint local-Gaussian inverse-depth uncertainty agrees with repeated image-noise trials', () => {
  const { reconstruction } = scene({ count: 20, views: 4, noise: 0 }),
    rng = seededRandom(3862),
    sigma = 0.2,
    gaussian = () =>
      Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) * Math.cos(2 * Math.PI * rng()),
    z: number[] = [];
  for (let i = 0; i < 80; i++) {
    const noisy = {
        ...reconstruction,
        landmarks: reconstruction.landmarks.map((p) => ({
          ...p,
          observations: p.observations.map((o) => ({
            ...o,
            pixel: [o.pixel[0] + sigma * gaussian(), o.pixel[1] + sigma * gaussian()] as const,
          })),
        })),
      },
      r = bundleAdjustInverseDepth(noisy, [], { iterations: 8, pixelNoiseStdDevPx: sigma }),
      p = r.estimates[0]!;
    assert.ok(p.covariance);
    const truth = fitInverseDepth(
      reconstruction.landmarks[0]!.observations.map((o) => ({
        ...reconstruction.views.find((v) => v.id === o.viewId)!,
        pixel: o.pixel,
      })),
    ).point.parameters[2]!;
    z.push((p.point.parameters[2]! - truth) / Math.sqrt(p.covariance![8]!));
  }
  const within2 = z.filter((x) => Math.abs(x) <= 2).length / z.length,
    rms = Math.sqrt(z.reduce((s, x) => s + x * x, 0) / z.length);
  assert.ok(within2 >= 0.85 && within2 <= 1, String(within2));
  assert.ok(rms > 0.65 && rms < 1.35, String(rms));
  console.log(
    JSON.stringify({ jointNoiseTrials: z.length, within2Stddev: within2, standardizedRms: rms }),
  );
});

test('uncertainty excludes LM damping, scales with noise, and changes with nuisance focal parameters', () => {
  const { reconstruction } = scene({ count: 25, views: 4, noise: 0 }),
    a = bundleAdjustInverseDepth(reconstruction, [], { iterations: 0 }),
    b = bundleAdjustInverseDepth(reconstruction, [], { iterations: 0, pixelNoiseStdDevPx: 2 }),
    focal = bundleAdjustInverseDepth(reconstruction, [], {
      iterations: 0,
      refineSharedFocal: true,
    });
  assert.ok(a.uncertainty.observable);
  assert.ok(focal.uncertainty.observable);
  assert.ok(Math.abs(b.estimates[0]!.covariance![8]! / a.estimates[0]!.covariance![8]! - 4) < 1e-8);
  assert.ok(focal.estimates[0]!.covariance![8]! > a.estimates[0]!.covariance![8]!);
  const degenerate = {
    views: reconstruction.views.slice(0, 2),
    landmarks: reconstruction.landmarks.slice(0, 2).map((p) => ({
      ...p,
      observations: p.observations.filter((o) =>
        reconstruction.views.slice(0, 2).some((v) => v.id === o.viewId),
      ),
    })),
  };
  const unknown = bundleAdjustInverseDepth(degenerate, [], { iterations: 0 });
  assert.equal(unknown.uncertainty.observable, false);
  assert.ok(unknown.estimates.every((p) => !p.covariance));
});

test('joint inverse-depth solve recovers a shared focal error with finite geometry', () => {
  const { reconstruction } = scene({ count: 30, views: 5, noise: 0 }),
    input = {
      ...reconstruction,
      views: reconstruction.views.map((v) => ({
        ...v,
        camera: { ...v.camera, fx: v.camera.fx * 0.9, fy: v.camera.fy * 0.9 },
      })),
    },
    result = bundleAdjustInverseDepth(input, [], { iterations: 30, refineSharedFocal: true });
  assert.ok(result.diagnostics.finalCost < result.diagnostics.initialCost * 1e-8);
  assert.ok(
    Math.abs(result.reconstruction.views[0]!.camera.fx / reconstruction.views[0]!.camera.fx - 1) <
      1e-6,
  );
  assert.equal(result.uncertainty.intrinsics, 'joint-focal');
});
