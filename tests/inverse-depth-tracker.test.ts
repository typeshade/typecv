import test from 'node:test';
import assert from 'node:assert/strict';
import { InverseDepthTracker } from '../src/reconstruction/inverse-depth-tracker.ts';
import { projectPoint } from '../src/geometry/pose.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';

function fixture() {
  const f = generalVideoFixture('translation'),
    points = f.truth.points.map((p, i) => (i < 40 ? p : p.map((x, j) => (j === 2 ? x + 1000 : x)))),
    views = f.truth.views.map((v, i) =>
      i === 4 ? { ...v, pose: { ...v.pose, t: Float64Array.of(-200, 0, 0) } } : v,
    ),
    images = views.map((v) => ({
      id: v.id,
      camera: v.camera,
      keypoints: Float64Array.from(points.flatMap((p) => [...projectPoint(v.camera, v.pose, p)!])),
    })),
    tracks = points.map((_, id) => ({
      id,
      observations: images.map((v) => ({ viewId: v.id, featureId: id })),
    }));
  const model = (n: number) => ({
    views: views.slice(0, n),
    landmarks: points.slice(0, 40).map((xyz, trackId) => ({
      trackId,
      xyz,
      observations: tracks[trackId]!.observations.slice(0, n).map((o) => ({
        ...o,
        pixel: [
          images[o.viewId]!.keypoints[2 * trackId]!,
          images[o.viewId]!.keypoints[2 * trackId + 1]!,
        ] as const,
      })),
    })),
  });
  return { images, tracks, model };
}

test('pending identities survive translation-free updates, then gain finite depth only with useful baseline', () => {
  const f = fixture(),
    tracker = new InverseDepthTracker({ iterations: 3, pixelNoiseStdDevPx: 0.25 }),
    first = tracker.update({
      tracks: f.tracks.map((t) => ({ ...t, observations: t.observations.slice(0, 3) })),
      images: f.images.slice(0, 3),
    });
  assert.equal(first.snapshot.diagnostics.bearingOnly, 60);
  assert.ok(first.snapshot.tracks.every((t) => !t.estimate));
  const weak = tracker.update({
    tracks: f.tracks.map((t) => ({
      ...t,
      id: t.id + 100,
      observations: t.observations.slice(0, 4),
    })),
    images: f.images.slice(0, 4),
    reconstruction: f.model(4),
  });
  assert.equal(weak.snapshot.diagnostics.finite, 40);
  assert.ok(weak.snapshot.tracks.filter((t) => t.id >= 40).every((t) => t.status !== 'finite'));
  const strong = tracker.update({ tracks: f.tracks, images: f.images, reconstruction: f.model(5) });
  assert.equal(strong.snapshot.diagnostics.finite, 60);
  assert.equal(strong.reconstruction!.landmarks.length, 60);
  const repeated = tracker.update({
    tracks: f.tracks,
    images: f.images,
    reconstruction: f.model(5),
  });
  for (let i = 0; i < strong.snapshot.tracks.length; i++) {
    const a = repeated.snapshot.tracks[i]!,
      b = strong.snapshot.tracks[i]!;
    assert.equal(a.id, b.id);
    assert.equal(a.status, b.status);
    assert.ok(
      Math.abs(a.covariance![8]! / b.covariance![8]! - 1) < 1e-5,
      `refitting the same data must not add precision: ${i} ratio ${a.covariance![8]! / b.covariance![8]!}`,
    );
  }
  assert.equal(repeated.snapshot.diagnostics.uniqueObservations, 300);
  repeated.snapshot.tracks[0]!.covariance![0] = 900;
  assert.notEqual(tracker.snapshot()!.tracks[0]!.covariance![0], 900);
  const inconsistent = f.images.map((v) => ({ ...v, keypoints: v.keypoints.slice() }));
  inconsistent[3]!.keypoints[0] += 80;
  const revoked = tracker.update({
    tracks: f.tracks,
    images: inconsistent,
    reconstruction: f.model(5),
  });
  assert.equal(revoked.snapshot.tracks[0]!.status, 'estimating');
  assert.equal(revoked.snapshot.tracks[0]!.reason, 'inconsistent-observations');
  assert.ok(!revoked.reconstruction!.landmarks.some((p) => p.trackId === 0));
});

test('evicted observations retain baseline information and map gauge changes transform depth uncertainty', () => {
  const f = fixture(),
    tracker = new InverseDepthTracker({ pixelNoiseStdDevPx: 0.25 });
  tracker.update({
    images: f.images.slice(0, 4),
    tracks: f.tracks.map((t) => ({ ...t, observations: t.observations.slice(0, 4) })),
    reconstruction: f.model(4),
  });
  const current = f.model(5),
    window = {
      views: current.views.slice(2),
      landmarks: current.landmarks.map((p) => ({
        ...p,
        observations: p.observations.filter((o) => o.viewId >= 2),
      })),
    },
    input = {
      images: f.images.slice(2),
      tracks: f.tracks.map((t) => ({ ...t, observations: t.observations.slice(2) })),
      reconstruction: window,
    },
    continued = tracker.update(input);
  assert.equal(continued.snapshot.diagnostics.finite, 60);
  assert.equal(continued.snapshot.diagnostics.uniqueObservations, 300);
  assert.equal(continued.reconstruction!.views.length, 5);
  const scale = 100,
    shift = [7, -5, 2],
    rescaled = {
      views: window.views.map((v) => ({
        ...v,
        pose: {
          ...v.pose,
          t: v.pose.t.map(
            (t, i) =>
              scale * t -
              v.pose.R[3 * i]! * shift[0]! -
              v.pose.R[3 * i + 1]! * shift[1]! -
              v.pose.R[3 * i + 2]! * shift[2]!,
          ),
        },
      })),
      landmarks: window.landmarks.map((p) => ({
        ...p,
        xyz: p.xyz.map((x, i) => scale * x + shift[i]!),
      })),
    },
    changed = tracker.update({ ...input, reconstruction: rescaled });
  assert.equal(changed.snapshot.diagnostics.finite, 60);
  assert.equal(changed.snapshot.diagnostics.uniqueObservations, 300);
  for (let i = 0; i < 60; i++) {
    const a = continued.snapshot.tracks[i]!,
      b = changed.snapshot.tracks[i]!;
    assert.ok(Math.abs((b.inverseRange!.mean * scale) / a.inverseRange!.mean - 1) < 1e-7);
    assert.ok(Math.abs((b.covariance![8]! * scale * scale) / a.covariance![8]! - 1) < 1e-6);
  }
});

test('late optical exclusions revoke historical observations instead of promoting a precise virtual surface', () => {
  const f = fixture(),
    tracker = new InverseDepthTracker();
  tracker.update({ tracks: f.tracks, images: f.images, reconstruction: f.model(5) });
  const cameraObservationMasks = f.images.map((v) => ({
    viewId: v.id,
    source: 'manual' as const,
    values: Uint8Array.from({ length: 60 }, (_, i) => (i < 40 ? 1 : 0)),
  }));
  const result = tracker.update({
    tracks: f.tracks,
    images: f.images,
    reconstruction: f.model(5),
    cameraObservationMasks,
  });
  assert.equal(result.snapshot.tracks.length, 40);
  assert.equal(result.reconstruction!.landmarks.length, 40);
  assert.ok(result.snapshot.tracks.every((t) => t.observations.every((o) => o.featureId < 40)));
  const again = tracker.update({ tracks: f.tracks, images: f.images, reconstruction: f.model(5) });
  assert.equal(again.snapshot.tracks.length, 40, 'optical exclusions survive subsequent updates');
});
