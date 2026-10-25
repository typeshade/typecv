import test from 'node:test';
import assert from 'node:assert/strict';
import { sortOwnedTsdfPoints, sortTsdfPoints } from '../src/reconstruction/tsdf.ts';

type Channels = Parameters<typeof sortTsdfPoints>;
const bytes = (array: ArrayBufferView) =>
  new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
const copy = (channels: Channels): Channels => [
  channels[0].slice(),
  channels[1].slice(),
  channels[2].slice(),
  channels[3].slice(),
  channels[4].slice(),
  channels[5].slice(),
  channels[6]?.slice(),
];
function compare(channels: Channels) {
  const before = copy(channels);
  const reference = sortTsdfPoints(...channels);
  for (let c = 0; c < channels.length; c++)
    if (channels[c]) assert.deepEqual(bytes(channels[c]!), bytes(before[c]!));
  const owned = sortOwnedTsdfPoints(...channels);
  for (const [key, source] of [
    ['positions', channels[0]],
    ['normals', channels[1]],
    ['colors', channels[2]],
    ['weights', channels[3]],
    ['free', channels[4]],
    ['tallies', channels[5]],
    ...(channels[6] ? [['freeWeights', channels[6]]] : []),
  ] as const) {
    const name = key as keyof Omit<typeof owned, 'count'>;
    assert.equal(owned[name], source);
    assert.notEqual(reference[name], source);
    assert.deepEqual(bytes(owned[name]!), bytes(reference[name]!), String(name));
  }
  assert.equal(owned.count, reference.count);
}

test('owned TSDF sorting matches the immutable oracle across stable ties, cycles and metadata', () => {
  let random = 0x17d83a5;
  const next = () => (random = (Math.imul(random, 1664525) + 1013904223) >>> 0);
  for (const count of [0, 1, 2, 3, 31, 257, 4097]) {
    for (const weighted of [false, true]) {
      const positions = new Float32Array(3 * count);
      const channels = Array.from({ length: 5 }, () => new Uint32Array(count));
      for (let i = 0; i < count; i++) {
        positions.set([(next() % 9) - 4, (next() % 7) - 3, (next() % 5) - 2], 3 * i);
        for (const channel of channels) channel[i] = next() % 4;
      }
      const freeWeights = weighted
        ? Float32Array.from({ length: count }, () => (next() % 5) / 3)
        : undefined;
      compare([
        positions,
        channels[0]!,
        channels[1]!,
        channels[2]!,
        channels[3]!,
        channels[4]!,
        freeWeights,
      ]);
    }
  }
  // Signed zero distinguishes equal keys by raw bits, proving stable ties retain input order.
  compare([
    new Float32Array([-0, 0, 0, 0, -0, 0, -0, -0, 0, 0, 0, -0]),
    new Uint32Array(4),
    new Uint32Array(4),
    new Uint32Array(4),
    new Uint32Array(4),
    new Uint32Array(4),
    new Float32Array([0, -0, 0, -0]),
  ]);
});

test('nonfinite TSDF keys preserve the old Array.sort ordering and position payloads', () => {
  for (const weighted of [false, true]) {
    const positions = new Float32Array([
      Infinity,
      1,
      0,
      2,
      NaN,
      0,
      -Infinity,
      0,
      0,
      0,
      1,
      NaN,
      2,
      1,
      0,
    ]);
    new Uint32Array(positions.buffer)[4] = 0x7fa12345;
    const freeWeights = weighted ? new Float32Array([NaN, Infinity, -Infinity, -0, 0]) : undefined;
    if (freeWeights) new Uint32Array(freeWeights.buffer)[0] = 0x7fa23456;
    compare([
      positions,
      new Uint32Array([5, 2, 1, 7, 2]),
      new Uint32Array([1, 4, 6, 2, 4]),
      new Uint32Array([1, 2, 3, 4, 5]),
      new Uint32Array([5, 4, 3, 2, 1]),
      new Uint32Array([0, 2, 4, 8, 9]),
      freeWeights,
    ]);
  }
});

test('owned TSDF sorting respects subarray offsets and rejects mismatched channels', () => {
  const positions = new Float32Array([99, 99, 99, 3, 2, 1, 0, 1, 2, 99, 99, 99]);
  const metadata = Array.from({ length: 5 }, () => new Uint32Array([99, 2, 1, 99]));
  compare([
    positions.subarray(3, 9),
    metadata[0]!.subarray(1, 3),
    metadata[1]!.subarray(1, 3),
    metadata[2]!.subarray(1, 3),
    metadata[3]!.subarray(1, 3),
    metadata[4]!.subarray(1, 3),
  ]);
  assert.deepEqual([...positions.slice(0, 3), ...positions.slice(9)], [99, 99, 99, 99, 99, 99]);
  for (const channel of metadata) assert.deepEqual([channel[0], channel[3]], [99, 99]);
  assert.throws(
    () =>
      sortOwnedTsdfPoints(
        new Float32Array(0),
        ...(metadata as [Uint32Array, Uint32Array, Uint32Array, Uint32Array, Uint32Array]),
      ),
    /Invalid owned TSDF/,
  );
});
