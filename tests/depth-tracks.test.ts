import test from 'node:test';
import assert from 'node:assert/strict';
import { DepthTrackHistory } from '../src/reconstruction/depth-tracks.ts';

const images = [0, 1, 2, 3].map((id) => ({
  id,
  camera: { fx: 10, fy: 10, cx: 0, cy: 0 },
  keypoints: Float64Array.of(1, 2, 3, 4),
}));
const track = (id: number, pairs: number[][]) => ({
  id,
  observations: pairs.map(([viewId, featureId]) => ({ viewId: viewId!, featureId: featureId! })),
});

test('depth identity persists across local ids/windows and repeated data adds no information', () => {
  const store = new DepthTrackHistory(),
    a = store.update(
      [
        track(17, [
          [0, 0],
          [1, 0],
        ]),
      ],
      images,
    );
  const b = store.update(
    [
      track(99, [
        [1, 0],
        [2, 0],
      ]),
    ],
    images,
  );
  assert.equal(b[0]!.id, a[0]!.id);
  assert.equal(b[0]!.observations.length, 3);
  assert.deepEqual(
    store.update(
      [
        track(99, [
          [1, 0],
          [2, 0],
        ]),
      ],
      images,
    ),
    b,
  );
  (b[0]!.observations[0]!.pixel as unknown as number[])[0] = 900;
  assert.equal(store.snapshot()[0]!.observations[0]!.pixel[0], 1);
});

test('merges use each observation once; splits rebuild separate lineages; exclusions revoke history', () => {
  const store = new DepthTrackHistory();
  store.update(
    [
      track(0, [
        [0, 0],
        [1, 0],
      ]),
      track(1, [
        [2, 0],
        [3, 0],
      ]),
    ],
    images,
  );
  const merged = store.update(
    [
      track(8, [
        [1, 0],
        [2, 0],
      ]),
    ],
    images,
  );
  assert.equal(merged.length, 1);
  assert.equal(merged[0]!.observations.length, 4);
  const split = store.update(
    [
      track(0, [
        [0, 0],
        [1, 0],
      ]),
      track(1, [
        [2, 0],
        [3, 0],
      ]),
    ],
    images,
  );
  assert.equal(split.length, 2);
  assert.ok(split.every((t) => t.id !== merged[0]!.id));
  const excluded = store.update([], images, new Set(['0:0', '1:0']));
  assert.equal(excluded.length, 1);
  assert.equal(excluded[0]!.observations.length, 2);
});
