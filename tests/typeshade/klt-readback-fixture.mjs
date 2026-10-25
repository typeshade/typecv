import assert from 'node:assert/strict';
import { gridPoints, renderScene } from '../klt-cases.ts';

export function kltReadbackInput() {
  const width = 128,
    height = 96;
  return {
    width,
    height,
    first: [...renderScene({ width, height })],
    second: [...renderScene({ width, height }, { dx: 2, dy: -1 })],
    points: [...gridPoints(width, height, 30, 25, 3, 3)],
  };
}
export function assertKltReadbacks(result) {
  assert.equal(result.liveBytes, 0);
  for (const [i, row] of result.rows.entries()) {
    assert.deepEqual(row.concurrent.output, row.serial.output, row.name);
    for (const field of ['reads', 'bytes', 'snapshots'])
      assert.equal(row.concurrent.counts[field], row.serial.counts[field], row.name + ' ' + field);
    assert.equal(row.serial.counts.reads, [3, 2, 7][i]);
    assert.equal(row.serial.counts.waves, row.serial.counts.reads);
    assert.equal(row.concurrent.counts.waves, 1, row.name);
  }
  assert.equal(result.rows[1].concurrent.output.back.length, 0);
  assert.ok(result.rows[1].concurrent.counts.snapshots > result.rows[1].concurrent.counts.bytes);
  assert.equal(result.rows[2].concurrent.output.pyramids.gray.length, 2);
  assert.equal(result.rows[2].concurrent.output.pyramids.grad.length, 2);
}
