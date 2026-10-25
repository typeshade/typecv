import test from 'node:test';
import assert from 'node:assert/strict';
import { TsdfViewChannels } from '../src/gpu/tsdf-view-channels.ts';
import type { TsdfView } from '../src/reconstruction/tsdf.ts';

function view(
  depth: Float32Array,
  normal: Uint32Array,
  rgba: Uint32Array,
  confidence?: Float32Array,
): TsdfView {
  return {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    center: [0, 0, 0],
    fx: 1,
    fy: 1,
    cx: 0,
    cy: 0,
    depth,
    normal,
    rgba,
    confidence,
  };
}

/** The removed interleaved host packer as a local, independent oracle. */
function packed(source: TsdfView, pixels: number) {
  const words = new Uint32Array(4 * pixels);
  const floats = new Float32Array(words.buffer);
  for (let i = 0; i < pixels; i++) {
    floats[4 * i] = source.depth[i]!;
    words[4 * i + 1] = source.normal[i]!;
    floats[4 * i + 2] = source.confidence ? source.confidence[i]! : 1;
    words[4 * i + 3] = source.rgba[i]!;
  }
  return words;
}

function compare(source: TsdfView, channels: TsdfViewChannels, pixels: number) {
  const expected = packed(source, pixels);
  const actual = Object.values(channels.of(source));
  for (let c = 0; c < 4; c++)
    assert.deepEqual(
      actual[c],
      Uint32Array.from({ length: pixels }, (_, i) => expected[4 * i + c]!),
    );
  return actual;
}

test('TSDF input channels borrow exact nonzero-offset pixel ranges and match the old packer', () => {
  const pixels = 7;
  const depth = new Float32Array([99, -0, 0, 1.25, -1, Infinity, -Infinity, NaN, 99]).subarray(
    1,
    8,
  );
  const confidence = new Float32Array([99, 1, 0.1, 0, -0, NaN, 0.25, 1, 99]).subarray(1, 8);
  const normal = new Uint32Array([99, 0, 1, 0xffffffff, 0x80000000, 7, 8, 9, 99]).subarray(1, 8);
  // An extra trailing word must not change the resident buffer length.
  const rgba = new Uint32Array([99, 0, 0xffffffff, 0x01234567, 0x89abcdef, 1, 2, 3, 99]).subarray(
    1,
  );
  const source = view(depth, normal, rgba, confidence);
  const before = [depth, normal, confidence, rgba].map((a) => new Uint8Array(a.buffer).slice());
  const channels = new TsdfViewChannels(pixels);
  const actual = compare(source, channels, pixels);
  assert.equal(channels.hostBytes, 0);
  for (const [c, array] of [depth, normal, confidence, rgba].entries()) {
    assert.equal(actual[c]!.buffer, array.buffer);
    assert.equal(actual[c]!.byteOffset, array.byteOffset);
    assert.deepEqual(new Uint8Array(array.buffer), before[c]);
  }
});

test('TSDF missing confidence uses a single bounded default channel across every view', () => {
  const pixels = 11;
  const channels = new TsdfViewChannels(pixels);
  let first: Uint32Array | undefined;
  for (let v = 0; v < 240; v++) {
    const source = view(
      new Float32Array(pixels).fill(v),
      new Uint32Array(pixels),
      new Uint32Array(pixels),
    );
    const actual = compare(source, channels, pixels);
    first ??= actual[2];
    assert.equal(actual[2], first);
    assert.equal(channels.hostBytes, 4 * pixels);
  }
});

test('short TSDF channels preserve undefined padding with reusable bounded scratch', () => {
  const pixels = 9;
  const channels = new TsdfViewChannels(pixels);
  for (const length of [0, 4, 1, 8, 2, 9, 3]) {
    const source = view(
      new Float32Array(length).fill(2),
      new Uint32Array(length).fill(7),
      new Uint32Array(length).fill(0xffabcdef),
      new Float32Array(length).fill(0.75),
    );
    compare(source, channels, pixels);
    assert.ok(channels.hostBytes <= 16 * pixels);
  }
  compare(view(new Float32Array(1), new Uint32Array(1), new Uint32Array(1)), channels, pixels);
  assert.ok(channels.hostBytes <= 20 * pixels);
});
