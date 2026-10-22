import test from 'node:test';
import assert from 'node:assert/strict';
import { tsdfVisibilityGroups } from '../src/reconstruction/tsdf.ts';

const view = (x: number) => ({
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  center: [x, 0, 0],
  fx: 32,
  fy: 32,
  cx: 7.5,
  cy: 7.5,
  depth: new Float32Array(256).fill(2),
  normal: new Uint32Array(256),
  rgba: new Uint32Array(256),
});
test('visibility groups cap nearby and rotated cameras without dropping their observations', () => {
  const views = [view(0), view(0.01), view(0.2), view(0.2)];
  views[3]!.rotation = [0, -1, 0, 1, 0, 0, 0, 0, 1];
  const input = { width: 16, height: 16, views };
  assert.deepEqual(tsdfVisibilityGroups(input), [
    [0, 1],
    [2, 3],
  ]);
  assert.deepEqual(tsdfVisibilityGroups(input, 1), [[2, 3]]);
  assert.deepEqual(tsdfVisibilityGroups(input, 3), [[0, 1]]);
  assert.deepEqual(
    views.map((v) => v.center[0]),
    [0, 0.01, 0.2, 0.2],
  );
});
