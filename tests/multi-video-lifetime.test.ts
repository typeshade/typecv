import assert from 'node:assert/strict';
import test from 'node:test';
import {
  OwnedMultiVideoFeatures,
  backingBytes,
  graphMatchBytes,
} from '../src/gpu/multi-video-lifetime.ts';

const camera = { fx: 4, fy: 4, cx: 2, cy: 2 };
const frame = (i: number) => ({
  videoId: 'video',
  frameKey: String(i),
  time: i,
  camera,
  keypoints: new Float64Array([i, i + 1]),
  descriptors: new Uint8Array(128).fill(i),
  global: new Float32Array([1, i]),
});

test('owned features retire descriptors, then only the completed component positions', () => {
  const input = [frame(1), frame(2), frame(3)];
  const sources = input.map((row) => ({
    points: row.keypoints,
    descriptors: row.descriptors,
    global: row.global,
  }));
  const snapshots = sources.map((row) => Object.values(row).map((array) => array.slice()));
  const owned = new OwnedMultiVideoFeatures(input, [20, 4, 8]);
  assert.equal(input.length, 0, 'the input container has no hidden feature roots');
  assert.equal(owned.bytes, 3 * (16 + 128 + 8));
  assert.equal(owned.keyframes[0]!.descriptors[0], 1 / 512);
  owned.releaseDescriptors();
  assert.equal(owned.bytes, 3 * 16);
  assert.throws(() => owned.keyframes[0]!.descriptors, /Released/);
  owned.releaseKeypoints([4, 8]);
  assert.equal(owned.bytes, 16);
  assert.equal(owned.keyframes[0]!.keypoints, sources[0]!.points);
  owned.clear();
  assert.equal(owned.bytes, 0);
  owned.clear();
  for (let i = 0; i < sources.length; i++)
    Object.values(sources[i]!).forEach((array, channel) =>
      assert.deepEqual(array, snapshots[i]![channel]),
    );
});

test('backing allocation budgets deduplicate overlapping views and shared channels', () => {
  const root = new ArrayBuffer(512);
  assert.equal(backingBytes([new Uint8Array(root, 8, 32), new Float64Array(root, 64, 3)]), 512);
  const input = [frame(1), frame(2)];
  input[0]!.keypoints = new Float64Array(root, 64, 2);
  input[1]!.keypoints = new Float64Array(root, 80, 2);
  input[0]!.descriptors = new Uint8Array(root, 128, 128);
  const owned = new OwnedMultiVideoFeatures(input, [0, 1]);
  assert.equal(owned.bytes, 512 + 128 + 8 + 8);
  owned.releaseDescriptors();
  assert.equal(
    owned.bytes,
    512,
    'a shared descriptor/keypoint allocation remains while any reader exists',
  );
  owned.releaseKeypoints([0]);
  assert.equal(owned.bytes, 512);
  owned.releaseKeypoints([1]);
  assert.equal(owned.bytes, 0);
  const matches = new Uint32Array(root, 0, 8);
  const distances = new Float64Array(root, 64, 4);
  assert.equal(
    graphMatchBytes([
      { matches, distances },
      { matches, distances },
    ] as any),
    512,
  );
});

test('empty owned input and cleanup preserve metadata with zero feature roots', () => {
  const empty = new OwnedMultiVideoFeatures([], []);
  empty.clear();
  assert.equal(empty.bytes, 0);
  const owned = new OwnedMultiVideoFeatures([frame(2)], [9]);
  owned.clear();
  assert.deepEqual(
    owned.keyframes.map(({ viewId, videoId, frameKey, time, camera }) => ({
      viewId,
      videoId,
      frameKey,
      time,
      camera,
    })),
    [{ viewId: 9, videoId: 'video', frameKey: '2', time: 2, camera }],
  );
});
