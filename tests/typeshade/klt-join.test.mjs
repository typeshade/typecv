// Verifies: docs/klt-tracking.md step 5 (sections 3.2, 3.3, 3.5 and 5.1 on the CPU tier).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  KLT_END,
  createKltTracker,
  kltOptionsFor,
  kltReject,
} from '../../dist/internal/gpu.js';
import { renderScene } from '../klt-cases.ts';

const W = 320,
  H = 200,
  scene = { width: W, height: H },
  shift = { dx: 3.7, dy: -2.1 },
  frame = (i) => renderScene(scene, { dx: shift.dx * i, dy: shift.dy * i });

async function withDevice(body) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await body(device);
    assert.equal(device.statistics.liveBytes, 0);
    return result;
  } finally {
    await device.dispose();
  }
}

/** The corners of the first frame as a tracker selects them: x, y pairs. */
async function corners(device, count) {
  const tracker = await createKltTracker(device, W, H, { options: { capacity: count } });
  await tracker.push({ gray: frame(0) });
  const t = tracker.tracks(),
    out = new Float64Array(2 * t.count);
  for (let k = 0; k < t.count; k++) {
    out[2 * k] = t.x[k];
    out[2 * k + 1] = t.y[k];
  }
  await tracker.dispose();
  return out;
}

/** The keypoints of a later frame: the points moved by the shift of `i` frames, in another order. */
function moved(points, i, jitter) {
  const n = points.length / 2,
    order = Array.from({ length: n }, (_, k) => (k * 7 + 3) % n),
    out = new Float64Array(points.length);
  order.forEach((source, k) => {
    out[2 * k] = points[2 * source] + i * shift.dx + jitter * Math.sin(k);
    out[2 * k + 1] = points[2 * source + 1] + i * shift.dy + jitter * Math.cos(k);
  });
  return { keypoints: out, order };
}

test('join: SIFT seeds, the match of a track at a later keyframe, the age from the last join', async () => {
  await withDevice(async (device) => {
    const base = await corners(device, 30),
      tracker = await createKltTracker(device, W, H, { options: { capacity: 200 } });
    // Frame 10 is the first keyframe. The keypoints are the corners: all of them seed.
    await tracker.push({ gray: frame(0) }, { frameId: 10, select: false });
    assert.equal(tracker.tracks().count, 0, 'select: false leaves selection to join');
    const j0 = await tracker.join({ frameId: 10, keypoints: base });
    assert.equal(j0.pairs.length, 0);
    assert.equal(j0.joined, 0);
    assert.equal(j0.seeded, 30);
    assert.ok(j0.filledIn > 0, 'fill-in points around the seeds');
    const seeds = tracker.tracks();
    assert.equal(
      seeds.seedKeypoint.slice(0, 30).every((k, i) => k === i),
      true,
    );
    assert.equal(seeds.seedFrame[0], 10);

    // Frames 11 and 12 are not keyframes. Frame 13 is the second one, with shifted keypoints in
    // another order and a little noise, inside the join radius.
    await tracker.push({ gray: frame(1) }, { frameId: 11, select: false });
    await tracker.push({ gray: frame(2) }, { frameId: 12, select: false });
    const s3 = await tracker.push({ gray: frame(3) }, { frameId: 13, select: false });
    assert.equal(s3.selected, 0);
    const { keypoints, order } = moved(base, 3, 0.3),
      j1 = await tracker.join({ frameId: 13, keypoints });
    assert.equal(j1.pairs.length, 1);
    const pair = j1.pairs[0];
    assert.equal(pair.first, 10);
    assert.equal(pair.second, 13);
    assert.ok(pair.matches.length / 2 >= 26, `${pair.matches.length / 2} of 30 joined`);
    assert.equal(pair.tracks.length, pair.matches.length / 2);
    // Every match is the right correspondence: first index i, second index k with order[k] = i.
    for (let r = 0; r < pair.tracks.length; r++)
      assert.equal(order[pair.matches[2 * r + 1]], pair.matches[2 * r]);
    const seconds = Array.from(pair.matches.filter((_, i) => i % 2 === 1));
    assert.deepEqual(
      seconds,
      [...seconds].sort((a, b) => a - b),
      'in the order of the second index',
    );
    assert.equal(new Set(seconds).size, seconds.length, 'a keypoint belongs to one track');
    // The joined tracks are anchored again: age 0, seed frame 13, position on the keypoint.
    const after = tracker.tracks();
    let anchored = 0;
    for (let k = 0; k < after.count; k++)
      if (after.seedFrame[k] === 13) {
        anchored++;
        assert.equal(after.age[k], 0);
      }
    assert.ok(anchored >= pair.tracks.length);
    assert.equal(j1.joined, pair.tracks.length);

    // The third keyframe, frame 16, gives matches against both earlier keyframes, and against
    // frame 13 only when the pairs are limited to it.
    for (const [i, id] of [
      [4, 14],
      [5, 15],
      [6, 16],
    ])
      await tracker.push({ gray: frame(i) }, { frameId: id, select: false });
    const later = moved(base, 6, 0.3),
      j2 = await tracker.join({ frameId: 16, keypoints: later.keypoints });
    assert.deepEqual(
      j2.pairs.map((p) => p.first),
      [10, 13],
    );
    const direct = j2.pairs[0],
      chained = j2.pairs[1];
    assert.ok(direct.matches.length / 2 >= 20, `${direct.matches.length / 2} matches to frame 10`);
    assert.ok(chained.matches.length / 2 >= 20);
    for (let r = 0; r < direct.tracks.length; r++)
      assert.equal(later.order[direct.matches[2 * r + 1]], direct.matches[2 * r]);
    const st = tracker.stats();
    assert.equal(st.joins, 3 - 1 + 1);
    assert.equal(
      st.matches,
      pair.matches.length / 2 + direct.matches.length / 2 + chained.matches.length / 2,
    );
    await tracker.dispose();
  });
});

test('join: views limit the pairs, a keypoint is claimed once, and nothing joins past the radius', async () => {
  await withDevice(async (device) => {
    const base = await corners(device, 12),
      tracker = await createKltTracker(device, W, H, { options: { capacity: 100 } });
    await tracker.push({ gray: frame(0) }, { frameId: 1, select: false });
    await tracker.join({ frameId: 1, keypoints: base });
    await tracker.push({ gray: frame(1) }, { frameId: 2, select: false });
    // The keypoints are the tracks' positions moved by 1.5 px, past the radius of 2 px for the first
    // three, and a pair of keypoints at the position of one track.
    const t = tracker.tracks(),
      keypoints = [];
    for (let k = 0; k < 12; k++) {
      const id = Array.from(t.seedKeypoint.slice(0, t.count)).indexOf(k);
      assert.ok(id >= 0);
      const far = k < 3 ? 3 : 0;
      keypoints.push(t.x[id] + far, t.y[id]);
    }
    // Keypoint 12 sits 0.5 px from the position of keypoint 5, keypoint 13 sits 1 px from it.
    const five = Array.from(t.seedKeypoint.slice(0, t.count)).indexOf(5);
    keypoints.push(t.x[five] + 0.5, t.y[five], t.x[five] - 1, t.y[five]);
    const join = await tracker.join({ frameId: 2, keypoints, views: [] });
    assert.equal(join.pairs.length, 0, 'no earlier frame can take a pair');
    assert.equal(join.joined, 9, 'three tracks are 3 px away, nine join');
    assert.equal(tracker.stats().matches, 0);
    // The nearest keypoint of track five is 12, and 5 itself (0 px) is nearer: it keeps keypoint 5.
    const after = tracker.tracks();
    assert.equal(after.seedKeypoint[Array.from(after.id).indexOf(t.id[five])], 5);
    await tracker.dispose();
  });
});

test('join: only the frame of the last push can be joined', async () => {
  await withDevice(async (device) => {
    const tracker = await createKltTracker(device, W, H, { options: { capacity: 10 } });
    await assert.rejects(tracker.join({ frameId: 0, keypoints: [] }), /last push/);
    await tracker.push({ gray: frame(0) }, { frameId: 4 });
    await assert.rejects(tracker.join({ frameId: 5, keypoints: [] }), /last push/);
    const none = await tracker.join({ frameId: 4, keypoints: [] });
    assert.equal(none.pairs.length, 0);
    assert.equal(none.seeded, 0);
    await tracker.dispose();
  });
});

test('reject ends tracks as epipolar outliers and the statistics count them', async () => {
  await withDevice(async (device) => {
    const tracker = await createKltTracker(device, W, H, { options: { capacity: 20 } });
    await tracker.push({ gray: frame(0) });
    await tracker.push({ gray: frame(1) });
    const before = tracker.tracks(),
      ids = [before.id[0], before.id[3], 99999];
    const removed = await tracker.reject(ids);
    assert.equal(removed, 2);
    const after = tracker.tracks();
    assert.equal(after.count, before.count - 2);
    assert.ok(!Array.from(after.id).includes(before.id[0]));
    assert.ok(Array.from(after.id).includes(before.id[1]));
    const st = tracker.stats();
    assert.equal(st.rejected, 2);
    assert.equal(st.ended[KLT_END.epipolar], 2);
    assert.equal(st.alive, after.count);
    assert.equal(
      st.lengths.reduce((a, b) => a + b, 0),
      st.alive + 2 + st.ended.reduce((a, b, i) => (i === KLT_END.epipolar ? a : a + b), 0),
    );
    assert.equal(await tracker.reject([]), 0);
    // The verdict of a pair: dropped and never kept ends the track.
    assert.deepEqual(
      Array.from(
        kltReject(
          new Map([
            [1, { inlier: 0, outlier: 1 }],
            [2, { inlier: 1, outlier: 1 }],
            [3, { inlier: 1, outlier: 0 }],
            [4, { inlier: 0, outlier: 0 }],
          ]),
        ),
      ),
      [1],
    );
    await tracker.dispose();
  });
});

test('selection around SIFT seeds keeps minDistancePx from every alive track', async () => {
  await withDevice(async (device) => {
    const D = kltOptionsFor(W).minDistancePx,
      base = await corners(device, 60),
      // A second keypoint 4 px from each corner: two tracks in one cell or in neighbouring cells.
      crowded = new Float64Array(4 * 60);
    for (let k = 0; k < 60; k++) {
      crowded[4 * k] = base[2 * k];
      crowded[4 * k + 1] = base[2 * k + 1];
      crowded[4 * k + 2] = base[2 * k] + 4;
      crowded[4 * k + 3] = base[2 * k + 1];
    }
    const tracker = await createKltTracker(device, W, H, { options: { capacity: 1000 } });
    await tracker.push({ gray: frame(0) }, { frameId: 0, select: false });
    const join = await tracker.join({ frameId: 0, keypoints: crowded }),
      t = tracker.tracks();
    assert.ok(join.seeded > 60, `${join.seeded} seeds`);
    assert.ok(join.filledIn > 0);
    // The occupancy holds one track for each cell, so some picks lie within D of a second track.
    // The host check drops them: this is what proves the check is needed.
    assert.ok(join.filtered > 0, 'the host check dropped a pick');
    // Every fill-in track is D from every other track. Seeds may be closer to each other.
    let least = Infinity;
    for (let a = 0; a < t.count; a++) {
      if (t.seedKeypoint[a] !== 0xffffffff) continue;
      for (let b = 0; b < t.count; b++)
        if (a !== b) least = Math.min(least, Math.hypot(t.x[a] - t.x[b], t.y[a] - t.y[b]));
    }
    assert.ok(least >= D, `closest fill-in pair ${least.toFixed(2)} px, minDistancePx ${D}`);
    console.log(
      `seeds ${join.seeded}, fill-in ${join.filledIn}, picks dropped by the host check ${join.filtered}`,
    );
    await tracker.dispose();
  });
});
