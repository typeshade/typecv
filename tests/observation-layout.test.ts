import test from 'node:test';
import assert from 'node:assert/strict';
import { observationLayout } from '../src/gpu/observation-layout.ts';

test('indexed observation storage removes repeated camera and point records at aggregate recorded run counts', () => {
  const cameras = 333,
    points = 69_245,
    observations = 494_842;
  const packed = observationLayout(cameras, points, observations, true);
  assert.equal(packed.inputBytes, 144 * cameras + 24 * points + 24 * observations);
  assert.equal(packed.inputBytes, 13_586_040);
  assert.ok(packed.inputBytes < (184 * observations) / 6);
  const bytes = Object.entries(packed.lengths).reduce(
    (sum, [name, length]) =>
      sum + length * (['indices', 'flags', 'offsets'].includes(name) ? 4 : 8),
    0,
  );
  assert.equal(packed.bytes, bytes);
  assert.equal(packed.largestBindingBytes, 16 * observations);
  assert.equal(packed.lengths.flags, observations + points);
});

test('observation layout accounts for every minimum allocation and optional depth flag', () => {
  const empty = observationLayout(0, 0, 0, false);
  assert.equal(empty.bytes, 76);
  assert.equal(empty.inputBytes, 32);
  assert.equal(empty.largestBindingBytes, 16);
  const plain = observationLayout(2, 3, 7, false),
    guarded = observationLayout(2, 3, 7, true);
  assert.equal(guarded.bytes - plain.bytes, 4 * 3 + 4 * (3 + 1));
  assert.equal(guarded.inputBytes, plain.inputBytes);
  assert.equal(guarded.lengths.partials, 2);
  assert.equal(observationLayout(2, 32, 65, true).lengths.partials, 4);
});
