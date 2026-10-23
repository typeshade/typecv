import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, reconstructIncrementalFloatGpu } from '../../dist/gpu.js';
import { mappingFixture } from './geometry-cases.mjs';

test('float adapter matches resident rows and retains unchanged geometric mapper validation', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const owned = [];
  try {
    const fixture = mappingFixture();
    const images = fixture.images.map((image) => {
      const values = new Float32Array(40 * 40);
      for (let row = 0; row < 40; row++) values[row * 40 + row] = 1;
      const buffer = device.allocate(values.length, values);
      owned.push(buffer);
      return {
        ...image,
        camera: { ...image.camera },
        descriptors: { descriptors: buffer, count: 40, dimensions: 40, elements: values.length },
      };
    });
    const events = [];
    const matchingOptions = { ratio: 0.8, maxDistance: 0.001, mutual: true };
    const pending = reconstructIncrementalFloatGpu(device, images, fixture.pairs, {
      matching: matchingOptions,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 4,
      onMatchProgress: (event) => events.push(event),
    });
    // Host metadata is snapshotted before the first matching await; residents stay borrowed.
    images[0].keypoints.fill(0);
    images[0].camera.fx = 1;
    images[0].descriptors.count = 0;
    fixture.pairs[0].first = 999;
    // Invalid replacements would fail later pairs if the adapter retained this options object.
    matchingOptions.ratio = 0;
    matchingOptions.maxDistance = -1;
    matchingOptions.mutual = null;
    const result = await pending;
    assert.equal(result.reconstruction.views.length, 4);
    assert.equal(result.reconstruction.landmarks.length, 40);
    assert.deepEqual(result.unregistered, []);
    assert.ok(result.diagnostics.meanReprojectionErrorPx < 1e-6);
    assert.equal(
      result.execution.stages.filter((s) => s.operation === 'pairVerification').length,
      6,
    );
    assert.equal(result.matching.pairs.length, 6);
    assert.ok(result.matching.pairs.every((pair) => pair.matches === 40));
    assert.equal(events.length, 12);
    assert.equal(events[0].phase, 'start');
    assert.equal(events[11].phase, 'complete');
    assert.ok(result.tracking, 'the float adapter inherits the calibrated default');
    const baseline = mappingFixture();
    const disabled = await reconstructIncrementalFloatGpu(
      device,
      baseline.images.map((image, i) => ({
        ...image,
        descriptors: {
          descriptors: owned[i],
          count: 40,
          dimensions: 40,
          elements: 1600,
        },
      })),
      baseline.pairs,
      {
        rotationTracking: false,
        matching: { ratio: 0.8, maxDistance: 0.001, mutual: true },
        ransacIterations: 128,
        bundleIterations: 2,
        finalBundleIterations: 4,
      },
    );
    assert.equal(disabled.reconstruction.views.length, 4);
    assert.ok(!('tracking' in disabled), 'explicit false survives both asynchronous adapters');
    // The adapter did not dispose caller-owned descriptors.
    assert.equal((await owned[0].read())[0], 1);
    assert.equal(device.statistics.liveBuffers, owned.length);
  } finally {
    for (const buffer of owned) device.drop(buffer);
    await device.dispose();
  }
});

test('float adapter rejects malformed graphs and row metadata before any device dispatch', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const fixture = mappingFixture();
    const images = fixture.images.map((image) => ({
      ...image,
      descriptors: {
        descriptors: { read() {}, write() {} },
        count: 40,
        dimensions: 40,
        elements: 1600,
      },
    }));
    let calls = 0;
    device.plan = async () => {
      calls++;
      throw new Error('unexpected device dispatch');
    };
    const pair = { first: 10, second: 20 };
    await assert.rejects(
      reconstructIncrementalFloatGpu(device, images, [pair, { first: 20, second: 10 }]),
      /Duplicate/,
    );
    await assert.rejects(
      reconstructIncrementalFloatGpu(device, images, [{ first: 10, second: 999 }]),
      /Invalid float-feature image pair/,
    );
    images[0].descriptors.count = 39;
    await assert.rejects(reconstructIncrementalFloatGpu(device, images, [pair]), /rows/);
    images[0].descriptors.count = 40;
    // Even views absent from the supplied edge list must provide valid public handles.
    images[3].descriptors.descriptors = {};
    await assert.rejects(reconstructIncrementalFloatGpu(device, images, [pair]), /rows/);
    assert.equal(calls, 0);
  } finally {
    await device.dispose();
  }
});

test('float adapter propagates device failures while retaining caller descriptor ownership', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const buffers = [
    device.allocate(1600, new Float32Array(1600)),
    device.allocate(1600, new Float32Array(1600)),
  ];
  try {
    const fixture = mappingFixture();
    const images = fixture.images.map((image, index) => ({
      ...image,
      descriptors: {
        descriptors: buffers[index % 2],
        count: 40,
        dimensions: 40,
        elements: 1600,
      },
    }));
    device.plan = async () => {
      throw new Error('injected float matching device failure');
    };
    await assert.rejects(
      reconstructIncrementalFloatGpu(device, images, [{ first: 10, second: 20 }]),
      /injected float matching device failure/,
    );
    assert.equal(device.statistics.liveBuffers, 2);
    assert.equal((await buffers[0].read()).length, 1600);
    assert.equal((await buffers[1].read()).length, 1600);
  } finally {
    for (const buffer of buffers) device.drop(buffer);
    await device.dispose();
  }
});
