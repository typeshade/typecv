import test from 'node:test';
import assert from 'node:assert/strict';
import { MapHistory } from '../src/reconstruction/map-history.ts';
import { validateReconstruction } from '../src/reconstruction/bundle-adjustment.ts';
import { scene } from './scene.ts';

test('history replaces active observations and features and preserves independent snapshots', () => {
  const { reconstruction } = scene({ count: 1, views: 3 });
  const history = new MapHistory();
  const before = history.update(reconstruction);
  const point = reconstruction.landmarks[0]!;
  const current = {
    ...reconstruction,
    landmarks: [{ ...point, observations: point.observations.slice(0, 2) }],
  };
  const after = history.update(current);
  assert.deepEqual(after.landmarks[0]!.observations, current.landmarks[0]!.observations);
  assert.equal(before.landmarks[0]!.observations.length, 3);
  after.landmarks[0]!.xyz.fill(999);
  assert.deepEqual(history.snapshot().landmarks[0]!.xyz, point.xyz);
  assert.equal(history.update(current).landmarks[0]!.observations.length, 2);
  const replacement = {
    ...current,
    landmarks: [
      {
        ...point,
        observations: current.landmarks[0]!.observations.map((o, i) => ({
          ...o,
          featureId: i === 1 ? 123 : o.featureId,
        })),
      },
    ],
  };
  const replaced = history.update(replacement);
  assert.deepEqual(replaced.landmarks[0]!.observations, replacement.landmarks[0]!.observations);
  validateReconstruction(replaced);
});

test('history retains archived observations through local track renumbering and pose updates', () => {
  const { reconstruction } = scene({ count: 1, views: 4 });
  const history = new MapHistory();
  const initial = history.update(reconstruction);
  const point = reconstruction.landmarks[0]!;
  const current = {
    views: reconstruction.views.slice(1).map((view) => ({
      ...view,
      pose: { ...view.pose, t: view.pose.t.map((x) => x + 0.01) },
    })),
    landmarks: [
      {
        ...point,
        trackId: 999,
        xyz: point.xyz.map((x) => x + 0.01),
        observations: point.observations.slice(1, 3),
      },
    ],
  };
  const after = history.update(current);
  assert.equal(after.landmarks[0]!.trackId, initial.landmarks[0]!.trackId);
  assert.deepEqual(after.landmarks[0]!.observations, point.observations.slice(0, 3));
  assert.equal(after.views.length, 4);
  assert.deepEqual(after.views[0], reconstruction.views[0]);
  assert.deepEqual(after.views.slice(1), current.views);
  assert.deepEqual(after.landmarks[0]!.xyz, current.landmarks[0]!.xyz);
  validateReconstruction(after);
});

test('history merges archived tracks without restoring their rejected active observations', () => {
  const { reconstruction } = scene({ count: 1, views: 5 });
  const point = reconstruction.landmarks[0]!;
  const history = new MapHistory();
  history.update({
    views: reconstruction.views,
    landmarks: [
      { ...point, trackId: 10, observations: point.observations.slice(0, 2) },
      { ...point, trackId: 20, observations: point.observations.slice(2, 4) },
    ],
  });
  const current = {
    views: reconstruction.views.slice(1),
    landmarks: [
      { ...point, trackId: 30, observations: [point.observations[1]!, point.observations[3]!] },
    ],
  };
  const after = history.update(current);
  assert.equal(after.landmarks.length, 1);
  assert.deepEqual(
    after.landmarks[0]!.observations.map((o) => o.viewId),
    [10, 20, 40],
  );
  validateReconstruction(after);
});

test('a split local track keeps both points instead of aliasing the old identity', () => {
  const { reconstruction } = scene({ count: 1, views: 4 });
  const point = reconstruction.landmarks[0]!;
  const history = new MapHistory();
  history.update(reconstruction);
  const after = history.update({
    ...reconstruction,
    landmarks: [
      { ...point, trackId: 10, observations: point.observations.slice(0, 2) },
      { ...point, trackId: 20, observations: point.observations.slice(2) },
    ],
  });
  assert.equal(after.landmarks.length, 2);
  assert.equal(new Set(after.landmarks.map((p) => p.trackId)).size, 2);
  assert.deepEqual(
    after.landmarks.map((p) => p.observations.length),
    [2, 2],
  );
  validateReconstruction(after);
});

test('a rejected active feature cannot reconnect to a retained historical point', () => {
  const { reconstruction } = scene({ count: 2, views: 4 });
  const point = reconstruction.landmarks[0]!,
    other = reconstruction.landmarks[1]!;
  const history = new MapHistory();
  const initial = history.update({ ...reconstruction, landmarks: [point] });
  const current = {
    views: reconstruction.views.slice(1),
    landmarks: [{ ...point, observations: point.observations.slice(1, 3) }],
  };
  history.update(current);
  const after = history.update({
    ...current,
    landmarks: [
      ...current.landmarks,
      { ...other, observations: [other.observations[1]!, point.observations[3]!] },
    ],
  });
  assert.equal(after.landmarks.length, 2);
  assert.equal(after.landmarks[0]!.trackId, initial.landmarks[0]!.trackId);
  assert.equal(after.landmarks[1]!.observations.length, 2);
  validateReconstruction(after);
});

test('rejected local points disappear while points outside local triangulation remain archived', () => {
  const { reconstruction } = scene({ count: 2, views: 4 });
  const history = new MapHistory();
  history.update({
    ...reconstruction,
    landmarks: [
      reconstruction.landmarks[0]!,
      {
        ...reconstruction.landmarks[1]!,
        observations: reconstruction.landmarks[1]!.observations.slice(0, 2),
      },
    ],
  });
  const after = history.update({ views: reconstruction.views.slice(1), landmarks: [] });
  assert.equal(after.landmarks.length, 1);
  assert.deepEqual(after.landmarks[0]!.xyz, reconstruction.landmarks[1]!.xyz);
  validateReconstruction(after);
});
