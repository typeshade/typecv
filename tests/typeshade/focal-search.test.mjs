import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental } from '../../dist/index.js';
import { ShadeDevice, IncrementalSfmSession } from '../../dist/gpu.js';
import { orbitFixture } from './focal-search-cases.mjs';
import { focalComponentsFixture } from '../fixtures/focal-components.ts';

// docs/auto-focal.md: the session runs the focal search through the device's numerical requests
// and lands on the CPU reference's estimate. The orbit's true focal is 0.7 × width; the cameras
// start at 1.2 × width, which the final refinement alone leaves at 1.04.
const options = {
  ransacIterations: 128,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
  focalSearch: { views: 4 },
};

test('the session finds the focal of the CPU reference on the CPU tier', async () => {
  const { images, pairs, width, ratio } = orbitFixture({ start: 1.2 });
  const reference = reconstructIncremental(images, pairs, options);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const session = new IncrementalSfmSession(device, options);
    const known = new Set();
    for (const image of images) {
      const edges = pairs.filter(
        (p) =>
          (p.first === image.id && known.has(p.second)) ||
          (p.second === image.id && known.has(p.first)),
      );
      const result = await session.add(image, edges);
      known.add(image.id);
      // The first map waits for the search's four views.
      if (known.size < 4) assert.equal(result, undefined);
      else assert.ok(result.reconstruction.views.length >= 2);
    }
    const found = session.focalSearch;
    assert.ok(found, 'the search ran');
    assert.deepEqual(found.views, [0, 1, 2, 3]);
    assert.ok(
      Math.abs(found.ratio / reference.diagnostics.focalSearch.ratio - 1) < 1e-3,
      `${found.ratio} against ${reference.diagnostics.focalSearch.ratio}`,
    );
    const final = await session.finish();
    assert.equal(
      session.tracking,
      undefined,
      'automatic focal sessions do not enable calibrated tracking on resume',
    );
    assert.ok(!('tracking' in final));
    assert.ok(!final.execution.cpuStages.some((s) => s.includes('rotation hypotheses (f64 CPU')));
    assert.equal(final.reconstruction.views.length, images.length);
    for (const view of final.reconstruction.views)
      assert.ok(Math.abs(view.camera.fx / width / ratio - 1) < 0.02, `${view.camera.fx / width}`);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('a streaming session defers calibration through a disconnected prefix until five usable views arrive', async () => {
  const fixture = focalComponentsFixture();
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const session = new IncrementalSfmSession(device, {
    ...options,
    ransacIterations: 64,
    focalSearch: { views: 5, refineIterations: 12 },
  });
  const known = new Set();
  try {
    for (const image of fixture.images) {
      const pairs = fixture.pairs.filter(
        (p) =>
          (p.first === image.id && known.has(p.second)) ||
          (p.second === image.id && known.has(p.first)),
      );
      const result = await session.add(image, pairs);
      known.add(image.id);
      if (image.id < 14) {
        assert.equal(result, undefined);
        assert.equal(session.focalSearch, undefined);
      } else assert.ok(result.reconstruction.views.length >= 5);
    }
    assert.deepEqual(session.focalSearch.views, [10, 11, 12, 13, 14]);
    assert.ok(Math.abs(session.focalSearch.ratio / fixture.ratio - 1) < 0.02);
    const final = await session.finish();
    assert.deepEqual(
      final.reconstruction.views.map((v) => v.id).sort((a, b) => a - b),
      [10, 11, 12, 13, 14, 15],
    );
    assert.equal(final.reconstruction.landmarks.length, 48);
    for (const view of final.reconstruction.views)
      assert.ok(Math.abs(view.camera.fx / fixture.width / fixture.ratio - 1) < 0.02);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});

test('finish calibrates a short connected clip that was deferred during streaming', async () => {
  const fixture = orbitFixture({ views: 3, points: 24 });
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const session = new IncrementalSfmSession(device, {
      ...options,
      ransacIterations: 32,
      bundleIterations: 2,
      finalBundleIterations: 2,
      focalSearch: { minRatio: 0.7, maxRatio: 0.7, refineIterations: 0 },
    });
    for (const image of fixture.images)
      assert.equal(
        await session.add(
          image,
          fixture.pairs.filter((p) => p.second === image.id),
        ),
        undefined,
      );
    const result = await session.finish();
    assert.equal(result.reconstruction.views.length, 3);
    assert.deepEqual(session.focalSearch.views, [0, 1, 2]);
  } finally {
    await device.dispose();
  }
  assert.equal(device.statistics.liveBuffers, 0);
});
