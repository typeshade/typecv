import test from 'node:test';
import assert from 'node:assert/strict';
import { readOnlyRgbaWords } from '../src/gpu/rgba-read-view.ts';

const littleEndian = new Uint8Array(Uint32Array.of(0x01020304).buffer)[0] === 4;
const packed = (rgba: Uint8Array | Uint8ClampedArray, pixels: number) =>
  Uint32Array.from(
    { length: pixels },
    (_, i) =>
      (rgba[4 * i]! |
        (rgba[4 * i + 1]! << 8) |
        (rgba[4 * i + 2]! << 16) |
        (rgba[4 * i + 3]! << 24)) >>>
      0,
  );

test('read-only RGBA words borrow only exact aligned image bytes and preserve their contents', () => {
  for (const ArrayType of [Uint8Array, Uint8ClampedArray]) {
    for (const offset of [0, 4, 12]) {
      const bytes = new ArrayType(44);
      for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 73 + 255) & 255;
      const image = bytes.subarray(offset, offset + 20);
      const before = bytes.slice();
      const words = readOnlyRgbaWords(image, 5);
      assert.deepEqual(words, packed(image, 5));
      assert.deepEqual(bytes, before);
      assert.equal(words.buffer === bytes.buffer, littleEndian);
      if (littleEndian) assert.equal(words.byteOffset, image.byteOffset);
    }
  }
});

test('unaligned and nonexact RGBA views use owning copies with the same packed values', () => {
  for (const ArrayType of [Uint8Array, Uint8ClampedArray]) {
    const bytes = new ArrayType(40);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 61 + 127) & 255;
    for (const [start, end, pixels] of [
      [1, 21, 5],
      [0, 24, 5],
      [4, 24, 3],
    ]) {
      const image = bytes.subarray(start, end);
      const before = bytes.slice();
      const words = readOnlyRgbaWords(image, pixels);
      assert.deepEqual(words, packed(image, pixels));
      assert.notEqual(words.buffer, bytes.buffer);
      words.fill(0);
      assert.deepEqual(bytes, before);
    }
  }
});
