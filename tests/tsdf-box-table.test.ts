import test from 'node:test';
import assert from 'node:assert/strict';
import { tsdfBoxTable, tsdfKeySlot } from '../src/gpu/tsdf-box-table.ts';
import {
  TSDF_HEADER,
  TSDF_WHOLE,
  tsdfBlockKey,
  tsdfInBox,
  tsdfKeyBlock,
  type TsdfVolume,
} from '../src/reconstruction/tsdf.ts';

function volume(): TsdfVolume {
  const grid = new Uint32Array(24);
  const signed = new Int32Array(grid.buffer);
  let keys = 0;
  for (let level = 0; level < 3; level++) {
    const nx = 32 >> level,
      ny = 24 >> level,
      nz = 16 >> level;
    signed.set([-16 >> level, -12 >> level, -8 >> level], 8 * level);
    grid.set([nx, ny, nz, keys], 8 * level + 3);
    keys += nx * ny * nz;
  }
  return {
    origin: [0, 0, 0],
    voxel: 0.01,
    inverse: 100,
    levels: 3,
    truncation: 3,
    grid,
    keys,
    footprints: new Float32Array(),
    samples: 0,
  };
}

function lookup(words: Uint32Array, slots: number, key: number) {
  let slot = tsdfKeySlot(key, slots - 1);
  for (let probe = 0; probe < slots; probe++) {
    const found = words[TSDF_HEADER + slot];
    if (found === key) return slot;
    if (found === 0xffffffff) return -1;
    slot = (slot + 1) & (slots - 1);
  }
  throw new Error('unterminated probe');
}

test('TSDF box tables contain exactly the complete discovery subset including negative-coordinate halos', () => {
  const v = volume();
  const keys = Uint32Array.from({ length: v.keys }, (_, key) => key);
  for (const box of [
    TSDF_WHOLE,
    { lo: [-2, -2, -1], hi: [1, 1, 1] },
    { lo: [-4, -3, -2], hi: [-3, -2, -1] },
    { lo: [100, 100, 100], hi: [101, 101, 101] },
  ] as const) {
    const staged = tsdfBoxTable(v, keys, box);
    const expected = Array.from(keys).filter((key) => {
      const [level, ...b] = tsdfKeyBlock(v, key);
      return tsdfInBox(v, box, level, b, true);
    });
    const actual = Array.from(staged.words.slice(TSDF_HEADER, TSDF_HEADER + staged.slots))
      .filter((key) => key !== 0xffffffff)
      .sort((a, b) => a - b);
    assert.deepEqual(actual, expected);
    assert.equal(staged.blocks, expected.length);
    assert.equal(staged.words[0], expected.length);
    assert.ok(2 * staged.blocks <= staged.slots);
    assert.equal(staged.slots & (staged.slots - 1), 0);
    for (const key of expected) assert.ok(lookup(staged.words, staged.slots, key) >= 0);
    assert.deepEqual(staged.words.slice(1, TSDF_HEADER), new Uint32Array(TSDF_HEADER - 1));
    assert.deepEqual(
      staged.words.slice(TSDF_HEADER + staged.slots),
      new Uint32Array(2 * staged.slots),
    );
  }
  const box = { lo: [-1, -1, -1], hi: [0, 0, 0] } as const;
  const boundary = tsdfBlockKey(v, 0, 0, 0, 0);
  const beyond = tsdfBlockKey(v, 0, 1, 0, 0);
  const staged = tsdfBoxTable(v, keys, box);
  assert.ok(
    lookup(staged.words, staged.slots, boundary) >= 0,
    'positive neighbour retained for interpolation',
  );
  assert.equal(
    lookup(staged.words, staged.slots, beyond),
    -1,
    'only the first positive neighbour is a halo',
  );
});

test('TSDF box hash matches the shader u32 arithmetic and resolves long collisions', () => {
  const oracle = (key: number, mask: number) => {
    const u32 = (value: bigint) => value & 0xffffffffn;
    let h = u32(BigInt(key) * 0x9e3779b1n);
    h ^= h >> 16n;
    h = u32(h * 0x7feb352dn);
    h ^= h >> 15n;
    h = u32(h * 0x846ca68bn);
    return Number((h ^ (h >> 16n)) & BigInt(mask));
  };
  for (const key of [0, 1, 0x7fffffff, 0x80000000, 0xfffffffe])
    for (const mask of [63, 255, 65535, 0x7fffffff, 0xffffffff])
      assert.equal(tsdfKeySlot(key, mask), oracle(key, mask));
  const v = volume();
  const collision: number[] = [];
  for (let key = 0; key < v.keys && collision.length < 24; key++)
    if (tsdfKeySlot(key, 63) === 63) collision.push(key);
  assert.equal(collision.length, 24);
  const staged = tsdfBoxTable(v, collision, TSDF_WHOLE);
  assert.equal(staged.slots, 64);
  assert.equal(lookup(staged.words, 64, collision[0]!), 63);
  assert.equal(lookup(staged.words, 64, collision[1]!), 0, 'probe wraps to slot zero');
  for (const key of collision) assert.ok(lookup(staged.words, 64, key) >= 0);
});
