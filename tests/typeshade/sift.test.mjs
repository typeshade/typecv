import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, extractSiftGpuResident, matchFloatDescriptorsGpu } from '../../dist/gpu.js';
import { texture, warped } from './sift-cases.mjs';

function similarityInliers(matches, source, target, scale, angle, tx, ty, tolerance) {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  let inliers = 0;
  for (let i = 0; i < matches.length; i += 2) {
    const a = matches[i] * 2,
      b = matches[i + 1] * 2;
    const px = scale * (c * source[a] - s * source[a + 1]) + tx,
      py = scale * (s * source[a] + c * source[a + 1]) + ty;
    if (Math.hypot(px - target[b], py - target[b + 1]) <= tolerance) inliers++;
  }
  return inliers;
}

test('SIFT on the CPU tier: keypoints repeat under scale and rotation and descriptors match', async () => {
  const width = 256,
    height = 192,
    gray = texture(width, height);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const base = warped(gray, width, height, width, height, 1, 0, 0, 0);
    const first = await extractSiftGpuResident(device, base, width, height, { maxFeatures: 1024 });
    assert.ok(first.count > 40, `only ${first.count} keypoints`);
    assert.ok(first.count <= first.capacity);
    assert.equal(first.descriptors.dimensions, 128);
    assert.equal(first.descriptors.elements, first.count * 128);
    const details = await first.readDetails();
    for (let i = 0; i < first.count; i++) {
      assert.ok(details[i * 8 + 2] > 1, 'sigma in image pixels');
      assert.ok(details[i * 8 + 3] >= 0 && details[i * 8 + 3] < 2 * Math.PI + 1e-6);
      assert.ok(Math.abs(details[i * 8 + 4]) >= 0.02 / 3);
    }
    const scale = 0.75,
      angle = Math.PI / 7,
      tx = 40,
      ty = 30;
    const moved = warped(gray, width, height, width, height, scale, angle, tx, ty);
    const second = await extractSiftGpuResident(device, moved, width, height, {
      maxFeatures: 1024,
    });
    assert.ok(second.count > 40, `only ${second.count} keypoints`);
    const matched = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
    });
    const pairs = matched.matches.length / 2;
    assert.ok(pairs >= 25, `only ${pairs} matches`);
    const inliers = similarityInliers(
      matched.matches,
      await first.readKeypoints(),
      await second.readKeypoints(),
      scale,
      angle,
      tx,
      ty,
      3,
    );
    assert.ok(inliers >= 0.7 * pairs, `${inliers} of ${pairs} matches follow the similarity`);
    // Descriptors are non-negative unit vectors, clipped at 0.2 before the final normalization.
    const vector = await device.enqueue(() => device.read(first.descriptors.descriptors, 128));
    let norm = 0;
    for (let j = 0; j < 128; j++) {
      assert.ok(vector[j] >= 0 && vector[j] <= 1);
      norm += vector[j] * vector[j];
    }
    assert.ok(Math.abs(norm - 1) < 1e-4, `norm ${norm}`);
    // The default is RootSIFT: the square root of the L1-normalized reference descriptor.
    const plain = await extractSiftGpuResident(device, base, width, height, {
      maxFeatures: 1024,
      rootSift: false,
    });
    assert.equal(plain.count, first.count);
    const reference = await device.enqueue(() => device.read(plain.descriptors.descriptors, 128));
    let l1 = 0;
    for (let j = 0; j < 128; j++) l1 += reference[j];
    for (let j = 0; j < 128; j++)
      assert.ok(
        Math.abs(vector[j] - Math.sqrt(reference[j] / l1)) < 1e-4,
        `RootSIFT element ${j}: ${vector[j]} vs ${Math.sqrt(reference[j] / l1)}`,
      );
    await plain.dispose();
    console.log(JSON.stringify({ first: first.count, second: second.count, pairs, inliers }));
    await first.dispose();
    await second.dispose();
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('Guided matching keeps candidates on the epipolar line and finds at least the plain matches', async () => {
  const width = 256,
    height = 192,
    gray = texture(width, height);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    // A pure horizontal shift: the epipolar lines are the image rows, F = [e2]x with e2 = (1, 0, 0).
    const shift = 14;
    const first = await extractSiftGpuResident(
      device,
      warped(gray, width, height, width, height, 1, 0, 0, 0),
      width,
      height,
      { maxFeatures: 1024 },
    );
    const second = await extractSiftGpuResident(
      device,
      warped(gray, width, height, width, height, 1, 0, shift, 0),
      width,
      height,
      { maxFeatures: 1024 },
    );
    const a = await first.readKeypoints(),
      b = await second.readKeypoints();
    const plain = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
    });
    const epipolar = {
      fundamental: [0, 0, 0, 0, 0, -1, 0, 1, 0],
      firstPoints: a,
      secondPoints: b,
      maxLinePx: 2,
    };
    const guided = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
      epipolar,
    });
    for (let i = 0; i < guided.matches.length; i += 2) {
      const p = guided.matches[i] * 2,
        q = guided.matches[i + 1] * 2;
      assert.ok(Math.abs(a[p + 1] - b[q + 1]) <= 2 + 1e-3, 'match off its epipolar line');
    }
    const onShift = (m) => {
      let n = 0;
      for (let i = 0; i < m.length; i += 2) {
        const p = m[i] * 2,
          q = m[i + 1] * 2;
        if (Math.hypot(b[q] - a[p] - shift, b[q + 1] - a[p + 1]) <= 2) n++;
      }
      return n;
    };
    assert.ok(onShift(guided.matches) >= onShift(plain.matches), 'guided lost correct matches');
    // A band wider than the image admits every candidate: guided matching is plain matching.
    const unbounded = await matchFloatDescriptorsGpu(
      device,
      first.descriptors,
      second.descriptors,
      {
        ratio: 0.8,
        epipolar: { ...epipolar, maxLinePx: 1e4 },
      },
    );
    assert.deepEqual([...unbounded.matches], [...plain.matches]);
    assert.deepEqual([...unbounded.distances], [...plain.distances]);
    // Lines across the shift (vertical) leave almost nothing: the guide constrains the search.
    const wrong = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
      epipolar: { ...epipolar, fundamental: [0, 0, -1, 0, 0, 0, 1, 0, 0] },
    });
    assert.ok(onShift(wrong.matches) < onShift(plain.matches) / 4, 'a wrong guide still matched');
    console.log(
      JSON.stringify({
        plain: plain.matches.length / 2,
        plainCorrect: onShift(plain.matches),
        guided: guided.matches.length / 2,
        guidedCorrect: onShift(guided.matches),
        wrong: wrong.matches.length / 2,
      }),
    );
    assert.throws(
      () =>
        matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
          epipolar: { ...epipolar, fundamental: [1, 2, 3] },
        }),
      RangeError,
    );
    await first.dispose();
    await second.dispose();
  } finally {
    await device.dispose();
  }
});

test('SIFT first octave -1: more keypoints, still in image pixels and matching the similarity', async () => {
  const width = 256,
    height = 192,
    gray = texture(width, height);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 256 << 20 });
  try {
    const base = warped(gray, width, height, width, height, 1, 0, 0, 0);
    const scale = 0.75,
      angle = Math.PI / 7,
      tx = 40,
      ty = 30;
    const moved = warped(gray, width, height, width, height, scale, angle, tx, ty);
    const options = { maxFeatures: 4096, firstOctave: -1 };
    const reference = await extractSiftGpuResident(device, base, width, height, {
      maxFeatures: 4096,
    });
    const first = await extractSiftGpuResident(device, base, width, height, options);
    const second = await extractSiftGpuResident(device, moved, width, height, options);
    // Octave -1 adds the finest scale on top of every octave the default searches.
    assert.ok(first.count > reference.count, `${first.count} vs ${reference.count} keypoints`);
    assert.equal(first.octaves, reference.octaves + 1);
    const points = await first.readKeypoints();
    for (let i = 0; i < points.length; i += 2) {
      assert.ok(points[i] >= 0 && points[i] <= width - 1, `x ${points[i]}`);
      assert.ok(points[i + 1] >= 0 && points[i + 1] <= height - 1, `y ${points[i + 1]}`);
    }
    const matched = await matchFloatDescriptorsGpu(device, first.descriptors, second.descriptors, {
      ratio: 0.8,
    });
    const pairs = matched.matches.length / 2;
    const inliers = similarityInliers(
      matched.matches,
      points,
      await second.readKeypoints(),
      scale,
      angle,
      tx,
      ty,
      3,
    );
    assert.ok(inliers >= 0.7 * pairs, `${inliers} of ${pairs} matches follow the similarity`);
    console.log(JSON.stringify({ reference: reference.count, first: first.count, pairs, inliers }));
    await reference.dispose();
    await first.dispose();
    await second.dispose();
    assert.throws(
      () => extractSiftGpuResident(device, base, width, height, { firstOctave: 1 }),
      RangeError,
    );
  } finally {
    await device.dispose();
  }
});

test('SIFT rejects invalid frames and options and honours the capacity', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const width = 64,
      height = 64,
      rgba = warped(texture(width, height), width, height, width, height, 1, 0, 0, 0);
    assert.throws(() => extractSiftGpuResident(device, rgba, 16, 16), RangeError);
    assert.throws(
      () => extractSiftGpuResident(device, rgba, width, height, { maxFeatures: 0 }),
      RangeError,
    );
    assert.throws(
      () => extractSiftGpuResident(device, rgba, width, height, { maxOrientations: 3 }),
      RangeError,
    );
    const capped = await extractSiftGpuResident(device, rgba, width, height, { maxFeatures: 5 });
    assert.ok(capped.count <= 5);
    assert.ok(capped.requested >= capped.count);
    await capped.dispose();
  } finally {
    await device.dispose();
  }
});
