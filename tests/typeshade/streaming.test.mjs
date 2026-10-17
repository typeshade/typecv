import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, IncrementalSfmSession } from '../../dist/gpu.js';
import { mappingFixture } from './geometry-cases.mjs';

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
