import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, IncrementalSfmSession } from '../../dist/gpu.js';
import { mappingFixture } from './geometry-cases.mjs';
import { scene } from '../scene.ts';

test('finishing an evicted window solves all archived observations and synchronizes later updates', async () => {
  const { inputs, pairs } = scene({ count: 28, views: 11, noise: 0.15, seed: 41 }),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const session = new IncrementalSfmSession(device, {
      windowSize: 8,
      ransacIterations: 64,
      minInitialPoints: 8,
      minPnPInliers: 6,
      initialPair: [10, 20],
      bundleIterations: 1,
      finalBundleIterations: 3,
      historyBundleIterations: 4,
      seed: 42,
    });
    const add = async (image) => {
      const active = new Set(session.activeViewIds);
      return session.add(
        image,
        pairs.filter(
          (p) =>
            (p.first === image.id && active.has(p.second)) ||
            (p.second === image.id && active.has(p.first)),
        ),
      );
    };
    for (const image of inputs.slice(0, 10)) await add(image);
    assert.ok(session.activeViewIds.length < 10, 'fixture must actually evict old cameras');
    const finished = await session.finish(),
      history = finished.diagnostics.historyRefinement;
    assert.equal(finished.reconstruction.views.length, 10);
    assert.equal(history.views, 10);
    assert.ok(history.finalCost <= history.initialCost);
    const observations = finished.reconstruction.landmarks.reduce(
      (n, p) => n + p.observations.length,
      0,
    );
    assert.equal(history.observations, observations);
    const repeated = await session.finish();
    assert.equal(repeated.reconstruction.views.length, 10);
    assert.ok(repeated.diagnostics.historyRefinement.finalCost <= history.finalCost + 1e-8);
    await add(inputs[10]);
    const resumed = await session.finish();
    assert.equal(resumed.reconstruction.views.length, 11);
    assert.ok(Number.isFinite(resumed.diagnostics.meanReprojectionErrorPx));
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('an auto-focal archive carries accepted calibration into repeated finish and newly added images', async () => {
  const { inputs, pairs } = scene({ count: 28, views: 11, noise: 0.1, seed: 61 }),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const session = new IncrementalSfmSession(device, {
      windowSize: 8,
      ransacIterations: 64,
      minInitialPoints: 8,
      minPnPInliers: 6,
      focalSearch: { views: 3, minRatio: 0.65, maxRatio: 1.2, step: 1.5, refineIterations: 2 },
      refineSharedFocal: true,
      bundleIterations: 1,
      finalBundleIterations: 1,
      historyBundleIterations: 3,
      seed: 42,
    });
    const add = async (image) => {
      const active = new Set(session.activeViewIds);
      return session.add(
        image,
        pairs.filter(
          (p) =>
            (p.first === image.id && active.has(p.second)) ||
            (p.second === image.id && active.has(p.first)),
        ),
      );
    };
    for (const image of inputs.slice(0, 10)) await add(image);
    const first = await session.finish(),
      repeated = await session.finish(),
      active = new Set(session.activeViewIds),
      camera = repeated.reconstruction.views.find((v) => active.has(v.id)).camera,
      ratio = session.focalSearch.ratio;
    assert.ok(first.diagnostics.historyRefinement);
    assert.ok(Math.abs(camera.fx / (2 * camera.cx + 1) - ratio) < 1e-12);
    const next = await add(inputs[10]),
      newest = next.reconstruction.views.find((v) => v.id === inputs[10].id);
    assert.ok(Math.abs(newest.camera.fx / (2 * newest.camera.cx + 1) - ratio) < 1e-12);
    const final = await session.finish();
    assert.equal(final.reconstruction.views.length, 11);
    assert.ok(Number.isFinite(final.diagnostics.meanReprojectionErrorPx));
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

for (const finalIterations of [0, 1])
  test(`late boundary calibration without eviction respects final iterations ${finalIterations}`, async () => {
    const { inputs, pairs } = scene({ count: 28, views: 10, noise: 0.1, seed: 61 }),
      device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    try {
      const session = new IncrementalSfmSession(device, {
        windowSize: 32,
        ransacIterations: 64,
        minInitialPoints: 8,
        minPnPInliers: 6,
        focalSearch: { views: 3, minRatio: 0.65, maxRatio: 1.2, step: 1.5, refineIterations: 2 },
        refineSharedFocal: true,
        bundleIterations: 1,
        finalBundleIterations: finalIterations,
        historyBundleIterations: 3,
        seed: 42,
      });
      const add = async (image) => {
        const active = new Set(session.activeViewIds);
        return session.add(
          image,
          pairs.filter(
            (p) =>
              (p.first === image.id && active.has(p.second)) ||
              (p.second === image.id && active.has(p.first)),
          ),
        );
      };
      for (const image of inputs.slice(0, 3)) await add(image);
      const initial = await session.finish();
      assert.equal(
        initial.diagnostics.historyRefinement,
        undefined,
        'the same calibration subset adds no evidence',
      );
      for (const image of inputs.slice(3)) await add(image);
      assert.equal(session.activeViewIds.length, inputs.length);
      const result = await session.finish();
      assert.equal(result.reconstruction.views.length, 10);
      if (finalIterations === 0) assert.equal(result.execution.gpuBundleAdjustments, 0);
      assert.equal(result.diagnostics.historyRefinement.views, 10);
      assert.ok(
        result.diagnostics.historyRefinement.finalCost <=
          result.diagnostics.historyRefinement.initialCost,
      );
      assert.equal(
        result.diagnostics.historyRefinement.observations,
        result.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
      );
    } finally {
      await device.dispose();
    }
    assert.equal(device.statistics.liveBuffers, 0);
  });

test('archived camera groups retain their intrinsics when GPU shared calibration is unsupported', async () => {
  const { inputs, pairs } = scene({ count: 28, views: 10, noise: 0.1, seed: 57 }),
    device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const focalGroups = new Map(inputs.map((v, i) => [v.id, i < 3 ? 0 : 1])),
      session = new IncrementalSfmSession(device, {
        windowSize: 8,
        ransacIterations: 64,
        minInitialPoints: 8,
        minPnPInliers: 6,
        initialPair: [10, 20],
        bundleIterations: 1,
        finalBundleIterations: 2,
        refineSharedFocal: true,
        focalGroups,
        seed: 42,
      });
    for (const image of inputs) {
      const active = new Set(session.activeViewIds);
      await session.add(
        image,
        pairs.filter(
          (p) =>
            (p.first === image.id && active.has(p.second)) ||
            (p.second === image.id && active.has(p.first)),
        ),
      );
    }
    assert.ok(session.activeViewIds.every((id) => focalGroups.get(id) === 1));
    const result = await session.finish();
    assert.equal(result.reconstruction.views.length, 10);
    assert.equal(result.diagnostics.historyRefinement.deferred, true);
    assert.ok(Number.isFinite(result.diagnostics.meanReprojectionErrorPx));
    assert.equal(
      result.diagnostics.historyRefinement.observations,
      result.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
    );
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

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
