// Verifies: docs/dense-mvs.md section 10 (the dense cloud scores: accuracy, completeness, F-score).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decimateByVoxel,
  distancesToCloud,
  landmarkCoverage,
  readPly,
  scoreCloud,
  visibleGroundTruth,
  type ScoreCamera,
} from '../src/reconstruction/dense-score.ts';

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cloud(n: number, seed: number, scale = 1): Float32Array {
  const r = random(seed);
  const out = new Float32Array(3 * n);
  for (let i = 0; i < out.length; i++) out[i] = (r() - 0.5) * scale;
  return out;
}

test('distancesToCloud equals brute force within the radius and Infinity past it', () => {
  const a = cloud(300, 1),
    b = cloud(500, 2),
    max = 0.15;
  const got = distancesToCloud(a, b, max);
  for (let i = 0; i < 300; i++) {
    let best = Infinity;
    for (let j = 0; j < 500; j++)
      best = Math.min(
        best,
        Math.hypot(
          a[3 * i]! - b[3 * j]!,
          a[3 * i + 1]! - b[3 * j + 1]!,
          a[3 * i + 2]! - b[3 * j + 2]!,
        ),
      );
    if (best <= max) assert.ok(Math.abs(got[i]! - best) < 1e-6, `point ${i}`);
    else assert.equal(got[i], Infinity);
  }
});

test('scoreCloud: a perfect cloud scores 1, a half cloud halves the completeness', () => {
  const truth = cloud(2000, 3);
  const perfect = scoreCloud(truth, truth, { taus: [0.01, 0.05], maskRadius: 0.5 });
  for (const row of perfect.rows) {
    assert.equal(row.accuracy, 1);
    assert.equal(row.completeness, 1);
    assert.equal(row.fscore, 1);
  }
  // The first half of a cloud sorted along x covers half of the ground truth.
  const order = Array.from({ length: 2000 }, (_, i) => i).sort(
    (i, j) => truth[3 * i]! - truth[3 * j]!,
  );
  const half = new Float32Array(3 * 1000);
  for (let i = 0; i < 1000; i++)
    for (let a = 0; a < 3; a++) half[3 * i + a] = truth[3 * order[i]! + a]!;
  const score = scoreCloud(half, truth, { taus: [1e-6], maskRadius: 0.5 });
  assert.equal(score.rows[0]!.accuracy, 1);
  assert.ok(Math.abs(score.rows[0]!.completeness - 0.5) < 1e-9);
  assert.ok(Math.abs(score.rows[0]!.fscore - 2 / 3) < 1e-9);
});

test('scoreCloud: a point beyond the mask is counted apart, a point inside it hurts the accuracy', () => {
  const truth = cloud(500, 4);
  const extra = new Float32Array(truth.length + 6);
  extra.set(truth);
  extra.set([10, 10, 10], truth.length); // far from the scan: not judged
  extra.set([truth[0]! + 0.2, truth[1]!, truth[2]!], truth.length + 3); // near, but off the surface
  const score = scoreCloud(extra, truth, { taus: [0.01], maskRadius: 0.5 });
  assert.equal(score.outsideMask, 1);
  assert.equal(score.judged, 501);
  assert.ok(Math.abs(score.rows[0]!.accuracy - 500 / 501) < 1e-9);
});

test('scoreCloud: an empty cloud scores 0', () => {
  const score = scoreCloud(new Float32Array(0), cloud(10, 5), { taus: [0.1], maskRadius: 1 });
  assert.equal(score.rows[0]!.fscore, 0);
  assert.equal(score.rows[0]!.completeness, 0);
});

test('visibleGroundTruth drops the points behind a nearer surface and outside the image', () => {
  const camera: ScoreCamera = {
    R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    t: [0, 0, 0],
    fx: 100,
    fy: 100,
    cx: 50,
    cy: 50,
    width: 100,
    height: 100,
  };
  // A wall at z = 2 and a point behind it on the same ray, one outside the image, one behind the camera.
  const pts: number[] = [];
  for (let y = -0.4; y <= 0.4; y += 0.02) for (let x = -0.4; x <= 0.4; x += 0.02) pts.push(x, y, 2);
  const wall = pts.length / 3;
  pts.push(0, 0, 3, 5, 0, 2, 0, 0, -1);
  const gt = Float32Array.from(pts);
  const kept = visibleGroundTruth(gt, [camera, camera]);
  assert.ok(kept.length >= wall - 2 && kept.length <= wall);
  assert.ok(!Array.from(kept).some((i) => i >= wall));
  assert.equal(visibleGroundTruth(gt, [camera], { minViews: 2 }).length, 0);
});

test('landmarkCoverage uses the radius of each landmark', () => {
  const dense = Float32Array.from([0, 0, 0, 1, 0, 0]);
  const near = { xyz: [0.05, 0, 0], radius: 0.1 };
  const far = { xyz: [1.05, 0, 0], radius: 0.01 };
  const wide = { xyz: [1.05, 0, 0], radius: 0.1 };
  const result = landmarkCoverage(dense, [near, far, wide]);
  assert.equal(result.missing, 1);
  assert.ok(Math.abs(result.missingFraction - 1 / 3) < 1e-12);
  assert.equal(landmarkCoverage(new Float32Array(0), [near]).missingFraction, 1);
});

test('readPly reads ASCII and binary little-endian vertices with colors', () => {
  const ascii = new TextEncoder().encode(
    'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n1 2 3 4 5 6\n7 8 9 10 11 12\n',
  );
  const a = readPly(ascii);
  assert.deepEqual(Array.from(a.positions), [1, 2, 3, 7, 8, 9]);
  assert.deepEqual(Array.from(a.colors!), [4, 5, 6, 10, 11, 12]);
  const header = new TextEncoder().encode(
    'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty double x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n',
  );
  const body = new DataView(new ArrayBuffer(8 + 4 + 4 + 3));
  body.setFloat64(0, 1.5, true);
  body.setFloat32(8, 2.5, true);
  body.setFloat32(12, 3.5, true);
  body.setUint8(16, 9);
  body.setUint8(17, 8);
  body.setUint8(18, 7);
  const file = new Uint8Array(header.length + 19);
  file.set(header);
  file.set(new Uint8Array(body.buffer), header.length);
  const b = readPly(file);
  assert.deepEqual(Array.from(b.positions), [1.5, 2.5, 3.5]);
  assert.deepEqual(Array.from(b.colors!), [9, 8, 7]);
});

test('decimateByVoxel keeps one point per cube', () => {
  const pts = Float32Array.from([0, 0, 0, 0.001, 0, 0, 1, 0, 0]);
  assert.equal(decimateByVoxel(pts, 0.1).length / 3, 2);
});
