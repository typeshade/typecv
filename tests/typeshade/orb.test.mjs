import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import {
  ShadeDevice,
  extractOrbCpu,
  extractOrbGpuResident,
  matchResidentFeaturesGpu,
} from '../../dist/gpu.js';
import { framePair, grayToRgba, hamming, mapPoint, similarity } from '../orb-images.ts';

const fixture = JSON.parse(
  readFileSync(new URL('../fixtures/orb-opencv.json', import.meta.url), 'utf8'),
);
const fixtureGray = new Uint8Array(inflateSync(Buffer.from(fixture.gray, 'base64')));

test('ORB kernels on the CPU tier equal the CPU reference bit for bit', async () => {
  const { width, height } = fixture,
    rgba = grayToRgba(fixtureGray, width, height),
    reference = extractOrbCpu(rgba, width, height, { maxFeatures: 300 }),
    device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const orb = await extractOrbGpuResident(device, rgba, width, height, { maxFeatures: 300 }),
      host = await orb.read(),
      details = await orb.readDetails();
    assert.equal(orb.count, reference.count);
    assert.equal(orb.requested, reference.requested);
    assert.deepEqual(host.descriptors, reference.descriptors);
    assert.deepEqual(details, reference.details);
    assert.deepEqual(host.keypoints, reference.keypoints);
    await orb.dispose();
  } finally {
    await device.dispose();
  }
});

/** Pairs of keypoints that land within `tolerance` pixels of each other under `motion`. */
function correspondences(motion, first, second, tolerance = 2) {
  const pairs = [];
  for (let i = 0; i < first.length / 2; i++) {
    const [x, y] = mapPoint(motion, first[i * 2], first[i * 2 + 1]);
    let best = -1,
      bestDistance = tolerance;
    for (let j = 0; j < second.length / 2; j++) {
      const d = Math.hypot(second[j * 2] - x, second[j * 2 + 1] - y);
      if (d < bestDistance) {
        bestDistance = d;
        best = j;
      }
    }
    if (best >= 0) pairs.push([i, best]);
  }
  return pairs;
}

test('ORB descriptors are invariant under a 30 degree rotation and matches obey a known homography', async () => {
  const width = 320,
    height = 240,
    device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    // Rotation: the Hamming distance of true correspondences against that of unrelated pairs.
    const rotation = similarity((30 * Math.PI) / 180, width / 2, height / 2),
      rotated = framePair(5, width, height, rotation),
      a = await extractOrbGpuResident(device, rotated.first, width, height, { maxFeatures: 600 }),
      b = await extractOrbGpuResident(device, rotated.second, width, height, { maxFeatures: 600 }),
      ha = await a.read(),
      hb = await b.read(),
      pairs = correspondences(rotation, ha.keypoints, hb.keypoints),
      distances = pairs
        .map(([i, j]) => hamming(ha.descriptors, i, hb.descriptors, j))
        .sort((x, y) => x - y),
      unrelated = [];
    for (let i = 0; i < 200; i++)
      unrelated.push(hamming(ha.descriptors, i % a.count, hb.descriptors, (i * 7 + 3) % b.count));
    unrelated.sort((x, y) => x - y);
    const median = distances[distances.length >> 1];
    assert.ok(pairs.length >= 40, `${pairs.length} repeated keypoints under rotation`);
    assert.ok(
      median <= 64,
      `median Hamming distance of ${pairs.length} rotated correspondences is ${median}`,
    );
    assert.ok(
      unrelated[unrelated.length >> 1] >= 100,
      'unrelated descriptors sit near 128 bits apart',
    );
    // The matcher on the same pair: most matches are true correspondences.
    const rotMatches = await matchResidentFeaturesGpu(device, a, b, {
        ratio: 0.8,
        maxDistance: 80,
      }),
      rotInliers = inlierCount(rotMatches.matches, ha.keypoints, hb.keypoints, rotation, 3);
    assert.ok(
      rotMatches.matches.length / 2 >= 30 && rotInliers / (rotMatches.matches.length / 2) >= 0.85,
    );
    await a.dispose();
    await b.dispose();

    // A projective motion: the inlier ratio of the ratio-tested mutual matches.
    const homography = [0.96, 0.18, 14, -0.14, 0.99, 9, 0.00035, -0.00025, 1],
      moved = framePair(5, width, height, homography),
      c = await extractOrbGpuResident(device, moved.first, width, height, { maxFeatures: 600 }),
      d = await extractOrbGpuResident(device, moved.second, width, height, { maxFeatures: 600 }),
      matches = await matchResidentFeaturesGpu(device, c, d, { ratio: 0.8, maxDistance: 80 }),
      inliers = inlierCount(
        matches.matches,
        (await c.read()).keypoints,
        (await d.read()).keypoints,
        homography,
        3,
      );
    assert.ok(matches.matches.length / 2 >= 40, `${matches.matches.length / 2} matches`);
    assert.ok(
      inliers / (matches.matches.length / 2) >= 0.85,
      `inlier ratio ${inliers}/${matches.matches.length / 2}`,
    );
    await c.dispose();
    await d.dispose();
  } finally {
    await device.dispose();
  }
});

function inlierCount(matches, first, second, motion, tolerance) {
  let inliers = 0;
  for (let m = 0; m < matches.length; m += 2) {
    const [x, y] = mapPoint(motion, first[matches[m] * 2], first[matches[m] * 2 + 1]);
    if (Math.hypot(second[matches[m + 1] * 2] - x, second[matches[m + 1] * 2 + 1] - y) <= tolerance)
      inliers++;
  }
  return inliers;
}

test('ORB options and frames are validated', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const frame = new Uint8Array(64 * 64 * 4);
    assert.throws(() => extractOrbGpuResident(device, frame, 63, 64), RangeError);
    for (const bad of [
      { maxFeatures: 0 },
      { scaleFactor: 1 },
      { fastThreshold: 300 },
      { levels: 13 },
    ])
      assert.throws(() => extractOrbGpuResident(device, frame, 64, 64, bad), RangeError);
    // A flat frame has no corner: an empty, valid result.
    const flat = await extractOrbGpuResident(device, frame, 64, 64);
    assert.equal(flat.count, 0);
    await flat.dispose();
  } finally {
    await device.dispose();
  }
});
