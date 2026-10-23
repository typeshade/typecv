import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, IncrementalSfmSession } from '../../dist/gpu.js';
import { mappingFixture } from './geometry-cases.mjs';
import { scene } from '../scene.ts';

test('finishing a noisy local-BA session filters history and reports its actual mean without final BA', async () => {
  const { inputs, pairs } = scene({ count: 80, views: 10, noise: 2.5, seed: 731 });
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const session = new IncrementalSfmSession(device, {
      ransacIterations: 128,
      seed: 42,
      maxErrorPx: 3,
      bundleIterations: 2,
      finalBundleIterations: 0,
      initialPair: [10, 20],
      globalBundleViewRatio: Infinity,
      globalBundlePointRatio: Infinity,
    });
    const known = new Set();
    for (const image of inputs) {
      await session.add(
        image,
        pairs.filter(
          (p) =>
            (p.first === image.id && known.has(p.second)) ||
            (p.second === image.id && known.has(p.first)),
        ),
      );
      known.add(image.id);
    }
    const result = await session.finish();
    const views = new Map(result.reconstruction.views.map((v) => [v.id, v]));
    const errors = result.reconstruction.landmarks.flatMap((point) =>
      point.observations.map((o) => {
        const { R, t } = views.get(o.viewId).pose,
          k = views.get(o.viewId).camera,
          X = point.xyz;
        const q = [0, 1, 2].map(
          (i) => R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + t[i],
        );
        assert.ok(q[2] > 0);
        const x = q[0] / q[2],
          y = q[1] / q[2],
          r2 = x * x + y * y;
        const d = 1 + (k.k1 ?? 0) * r2 + (k.k2 ?? 0) * r2 * r2;
        const error = Math.hypot(
          k.fx * x * d + k.cx - o.pixel[0],
          k.fy * y * d + k.cy - o.pixel[1],
        );
        assert.ok(error <= 3);
        return error;
      }),
    );
    assert.equal(views.size, 10);
    assert.equal(result.reconstruction.landmarks.length, 80);
    assert.ok(
      Math.abs(
        errors.reduce((a, b) => a + b, 0) / errors.length -
          result.diagnostics.meanReprojectionErrorPx,
      ) < 1e-12,
    );
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('keyframe session preserves its gauge, caches verified edges and publishes independent growing maps', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const { images, pairs } = mappingFixture();
    const snapshots = [];
    const session = new IncrementalSfmSession(device, {
      ransacIterations: 128,
      bundleIterations: 1,
      finalBundleIterations: 2,
      onReconstruction(model) {
        snapshots.push(model.views.length);
        // An observer cannot alter the live map used by the next registration.
        model.views[0].pose.R.fill(999);
        model.landmarks[0]?.xyz.fill(999);
      },
    });
    assert.equal(await session.add(images[0], []), undefined);
    const known = new Set([images[0].id]);
    let verified = 0;
    let anchor;
    for (const image of images.slice(1)) {
      const edges = pairs.filter(
        (p) =>
          (p.first === image.id && known.has(p.second)) ||
          (p.second === image.id && known.has(p.first)),
      );
      const pending = session.add(image, edges);
      await assert.rejects(session.finish(), /Await the previous/);
      const result = await pending;
      known.add(image.id);
      assert.equal(result.reconstruction.views.length, known.size);
      assert.equal(result.reconstruction.landmarks.length, 40);
      assert.ok(result.diagnostics.meanReprojectionErrorPx < 1e-6);
      verified += result.execution.stages.filter((s) => s.operation === 'pairVerification').length;
      anchor ??= result.diagnostics.initialPair.slice();
      assert.deepEqual(result.diagnostics.initialPair, anchor);
      assert.equal(result.reconstruction.views[0].pose.R[0], 1);
      // Neither returned numerical state nor diagnostics alias the private session state.
      result.reconstruction.views[0].pose.R.fill(123);
      result.diagnostics.initialPair.fill(-1);
    }
    const final = await session.finish();
    assert.equal(verified, pairs.length);
    assert.equal(
      final.execution.stages.filter((s) => s.operation === 'pairVerification').length,
      0,
    );
    assert.deepEqual(final.diagnostics.initialPair, anchor);
    assert.equal(final.reconstruction.views.length, 4);
    assert.ok(final.diagnostics.meanReprojectionErrorPx < 1e-6);
    assert.ok([2, 3, 4].every((count) => snapshots.includes(count)));
    await assert.rejects(session.add(images[0], []), /Duplicate keyframe/);
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});

test('session waits for connected input and recovers after rejected input', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const { images, pairs } = mappingFixture();
    const executed = [];
    const session = new IncrementalSfmSession(device, {
      ransacIterations: 128,
      bundleIterations: 0,
      finalBundleIterations: 0,
      onStage: (stage) => executed.push(stage.operation),
    });
    assert.equal(await session.add(images[0], []), undefined);
    assert.equal(await session.add(images[1], []), undefined);
    await assert.rejects(
      session.add(images[2], [{ first: 999, second: images[2].id, matches: new Uint32Array() }]),
      /Invalid/,
    );
    const known = new Set(images.slice(0, 3).map((image) => image.id));
    const result = await session.add(
      images[2],
      pairs.filter(
        (pair) =>
          known.has(pair.first) &&
          known.has(pair.second) &&
          (pair.first === images[2].id || pair.second === images[2].id),
      ),
    );
    // Supply only edges to frames that have arrived, retaining the earlier disconnected frames.
    assert.equal(result.reconstruction.views.length, 3);
    assert.ok(result.diagnostics.meanReprojectionErrorPx < 1e-6);
    assert.ok(executed.includes('pairVerification'));
  } finally {
    await device.dispose();
  }
});
