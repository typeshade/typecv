import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, matchFloatDescriptorsGpu } from '../../dist/gpu.js';
import { checkFloatMatches, floatDescriptorFixture } from './float-matching-cases.mjs';

test('float matcher exhaustively matches independent L2 oracle across query/candidate/channel tails', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const [n, m, dimensions] of [
      [1, 1, 1],
      [2, 3, 3],
      [7, 9, 15],
      [9, 7, 17],
      [17, 19, 33],
      [65, 67, 128],
    ]) {
      const first = floatDescriptorFixture(n, dimensions),
        second = floatDescriptorFixture(m, dimensions, 29);
      // Real near correspondences plus unmatched random descriptors, both directions.
      for (let i = 0; i < Math.min(n, m); i++)
        for (let d = 0; d < dimensions; d++)
          second[i * dimensions + d] = first[i * dimensions + d] + 0.001;
      for (const mutual of [false, true])
        await checkFloatMatches(device, matchFloatDescriptorsGpu, first, second, dimensions, {
          mutual,
        });
    }
  } finally {
    await device.dispose();
  }
});

test('float matcher handles ties, singleton, zero distance, inclusive distance and strict ratio', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const [a, b, options] of [
      [[0, 10], [0, 0, 10], {}], // Duplicate best fails strict ratio.
      [[0, 4], [1, 9], { maxDistance: 1, mutual: false }], // Inclusive distance.
      [[0, 4], [1, 9], { maxDistance: 0.99, mutual: false }],
      [[0, 10], [1, 2], { ratio: 0.5, mutual: false }], // Strict ratio equality rejects.
      [[0], [0, 4], { mutual: false }],
      [[0], [0, 4], { mutual: true }], // Reverse has no second neighbour.
      [[0, 4], [0, 4], { maxDistance: 0 }],
      [[], [0, 4], {}],
      [[0, 4], [], {}],
    ])
      await checkFloatMatches(
        device,
        matchFloatDescriptorsGpu,
        Float32Array.from(a),
        Float32Array.from(b),
        1,
        options,
      );
  } finally {
    await device.dispose();
  }
});

test('float matcher rejects metadata/options/budgets and nonfinite or overflowing arithmetic', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', workingBytes: 4096 });
  const a = device.allocate(2, Float32Array.of(0, 1));
  const b = device.allocate(2, Float32Array.of(0, 1));
  const shape = { descriptors: a, count: 2, dimensions: 1, elements: 2 };
  const other = { ...shape, descriptors: b };
  try {
    assert.throws(() => matchFloatDescriptorsGpu(device, shape, shape), /distinct/);
    for (const changed of [
      { count: -1 },
      { elements: 3 },
      { dimensions: 0 },
      { descriptors: null },
    ])
      assert.throws(
        () => matchFloatDescriptorsGpu(device, { ...shape, ...changed }, shape),
        RangeError,
      );
    for (const options of [
      { ratio: 0 },
      { ratio: 1 },
      { ratio: NaN },
      { maxDistance: -1 },
      { maxDistance: Infinity },
      { mutual: 1 },
    ])
      assert.throws(() => matchFloatDescriptorsGpu(device, shape, shape, options), RangeError);
    assert.throws(
      () => matchFloatDescriptorsGpu(device, { ...shape, dimensions: 2, count: 1 }, shape),
      RangeError,
    );
    assert.throws(
      () =>
        matchFloatDescriptorsGpu(
          device,
          { ...shape, count: 1000, elements: 1000 },
          { ...shape, count: 1000, elements: 1000 },
        ),
      RangeError,
    );
    for (const value of [NaN, Infinity, -Infinity, 3e38]) {
      a.write(Float32Array.of(value, 0));
      b.write(Float32Array.of(value, 0));
      await assert.rejects(matchFloatDescriptorsGpu(device, shape, other), /Nonfinite|overflow/);
    }
    a.write(Float32Array.of(0, 1));
    b.write(Float32Array.of(0, 1));
    const result = await matchFloatDescriptorsGpu(device, shape, other);
    assert.deepEqual([...result.matches], [0, 0, 1, 1]); // Failed jobs do not poison subsequent work.
  } finally {
    device.drop(a);
    device.drop(b);
    await device.dispose();
  }
});
