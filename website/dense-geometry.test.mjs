import test from 'node:test';
import assert from 'node:assert/strict';
import { denseGeometry } from './src/lib/dense-geometry.ts';

// The owner's south-building run drew most dense points black: the color attribute held a copy
// of the array the viewer appends into, so only the prefix copied at each growth had colors.
test('the dense attributes alias the arrays the viewer writes into', () => {
  const positions = new Float32Array(3 * 4),
    colors = new Uint8Array(3 * 4);
  const geometry = denseGeometry(positions, colors);
  const color = geometry.getAttribute('color');
  assert.equal(geometry.getAttribute('position').array, positions);
  assert.equal(color.array, colors);
  assert.equal(color.normalized, true);
  assert.equal(color.itemSize, 3);
  colors.set([230, 200, 150], 3);
  assert.deepEqual(
    [color.getX(1), color.getY(1), color.getZ(1)].map((v) => Math.round(v * 255)),
    [230, 200, 150],
  );
});
