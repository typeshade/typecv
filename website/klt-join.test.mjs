// Verifies: docs/klt-tracking.md step 5 (sections 3.3 and 3.5: the merge into the pairs and the verdict).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KLT_COVER_MATCHES,
  KLT_GATE_MIN_TRACKS,
  MIN_PAIR_MATCHES,
  medianTrackDisplacement,
  trackAnchor,
  matchKey,
  mergeTrackerPairs,
  trackerVerdicts,
} from './src/lib/klt-join.ts';

const pair = (first, second, rows, distance = 0.5) => ({
  first,
  second,
  matches: Uint32Array.from(rows.flat()),
  distances: Float64Array.from(rows.map(() => distance)),
});
const tracker = (first, second, rows) => ({
  first,
  second,
  matches: Uint32Array.from(rows.map(([a, b]) => [a, b]).flat()),
  tracks: Uint32Array.from(rows.map(([, , id]) => id)),
});
const range = (n, offset = 0) => Array.from({ length: n }, (_, i) => [i + offset, i + offset]);

test('a match both sources found is one match and keeps the descriptor distance', () => {
  const merged = mergeTrackerPairs(
    [pair(1, 5, range(20))],
    [tracker(1, 5, [...range(20).map(([a, b]) => [a, b, 100 + a]), [30, 31, 900]])],
  );
  assert.equal(merged.pairs.length, 1);
  const p = merged.pairs[0];
  assert.equal(p.matches.length / 2, 21);
  assert.equal(merged.duplicate, 20);
  assert.equal(merged.added, 1);
  assert.equal(p.source, 'merged');
  assert.equal(p.kltMatches, 1);
  assert.deepEqual(Array.from(p.distances.slice(0, 20)), new Array(20).fill(0.5));
  assert.equal(p.distances[20], 0, 'a tracker-only match has distance 0');
  // All 21 tracker matches are tracked: the 20 duplicates and the new one.
  assert.equal(merged.tracked.size, 21);
  assert.equal(merged.tracked.get(matchKey(1, 5, 30, 31)), 900);
});

test('a tracker match that disagrees with a descriptor match on either feature is left out', () => {
  const merged = mergeTrackerPairs(
    [pair(1, 5, range(20))],
    [
      tracker(1, 5, [
        [3, 7, 1], // first feature 3 matched to 3 by descriptor
        [40, 4, 2], // second feature 4 matched from 4 by descriptor
        [41, 42, 3], // new
      ]),
    ],
  );
  assert.equal(merged.conflicting, 2);
  assert.equal(merged.added, 1);
  assert.equal(merged.pairs[0].matches.length / 2, 21);
  assert.deepEqual([...merged.tracked.values()], [3]);
});

test('a pair only the tracker found goes to the mapper with enough matches', () => {
  const enough = range(MIN_PAIR_MATCHES).map(([a, b]) => [a, b, a]),
    few = range(MIN_PAIR_MATCHES - 1).map(([a, b]) => [a, b, a]);
  const merged = mergeTrackerPairs(
    [pair(2, 9, range(20))],
    [tracker(1, 9, enough), tracker(3, 9, few)],
  );
  assert.deepEqual(
    merged.pairs.map((p) => [p.first, p.source]),
    [
      [1, 'klt'],
      [2, undefined],
    ],
  );
  assert.equal(merged.pairs[0].matches.length / 2, MIN_PAIR_MATCHES);
  assert.equal(merged.pairs[0].kltMatches, MIN_PAIR_MATCHES);
  assert.ok(merged.pairs[0].distances.every((d) => d === 0));
  // The pair of 15 matches does not go, so its tracks get no verdict.
  assert.equal(
    [...merged.tracked.keys()].some((key) => key.startsWith('3:9:')),
    false,
  );
  // A descriptor pair with no tracker match passes unchanged.
  assert.equal(merged.pairs[1].source, undefined);
});

test('the verdict counts, for each track, the matches a verified pair kept and dropped', () => {
  const tracked = new Map([
    [matchKey(1, 5, 0, 0), 10],
    [matchKey(1, 5, 1, 1), 11],
    [matchKey(2, 5, 1, 1), 11],
    [matchKey(3, 5, 4, 4), 12],
    [matchKey(1, 6, 9, 9), 13],
  ]);
  const verdicts = trackerVerdicts(
    tracked,
    [
      { first: 1, second: 5, matches: [0, 0] },
      { first: 2, second: 5, matches: [7, 7] },
      // Pair (3, 5) failed verification: absent. Pair (1, 6) is of another frame.
      { first: 1, second: 6, matches: [] },
    ],
    5,
  );
  assert.deepEqual(verdicts.get(10), { inlier: 1, outlier: 0 });
  assert.deepEqual(verdicts.get(11), { inlier: 0, outlier: 2 });
  assert.equal(verdicts.has(12), false, 'a pair that failed verification gives no verdict');
  assert.equal(verdicts.has(13), false, 'a pair of another frame gives none');
  assert.equal(trackerVerdicts(tracked, undefined, 5).size, 0);
});

const alive = (rows) => ({
  count: rows.length,
  id: Uint32Array.from(rows.map(([id]) => id)),
  x: Float32Array.from(rows.map(([, x]) => x)),
  y: Float32Array.from(rows.map(([, , y]) => y)),
});

test('the track gate: the median displacement of the tracks alive at both keyframes (3.6)', () => {
  assert.equal(KLT_GATE_MIN_TRACKS, 64);
  assert.equal(KLT_COVER_MATCHES, 64);
  const before = alive(Array.from({ length: 100 }, (_, i) => [i, 10 * i, 5])),
    anchor = trackAnchor(before);
  assert.equal(anchor.size, 100);
  // 70 tracks moved 3 px along x, 10 moved 40 px, 20 ended, 5 are new: the median is 3.
  const now = alive([
    ...Array.from({ length: 70 }, (_, i) => [i, 10 * i + 3, 5]),
    ...Array.from({ length: 10 }, (_, i) => [70 + i, 10 * (70 + i) + 40, 5]),
    ...Array.from({ length: 5 }, (_, i) => [1000 + i, 0, 0]),
  ]);
  assert.deepEqual(medianTrackDisplacement(anchor, now), { median: 3, tracks: 80 });
  // Fewer than 64 common tracks decide nothing.
  const few = alive(Array.from({ length: 63 }, (_, i) => [i, 10 * i + 50, 5]));
  assert.deepEqual(medianTrackDisplacement(anchor, few), { median: Infinity, tracks: 63 });
  assert.equal(medianTrackDisplacement(anchor, few, 60).median, 50);
});

test('the lab gate reads two run records and holds them to the table of 7.4', async () => {
  const { kltGatePasses, kltGateRows, kltGateRun } = await import('./src/lib/klt-gate.ts');
  const record = (landmarks, observations, error, stagesMs, kltChunk, extra = {}) => ({
    result: {
      registeredViews: 12,
      landmarks,
      observations,
      meanReprojectionErrorPx: error,
      ...extra,
    },
    timing: {
      totalMs: 1,
      budgetMs: 250,
      stages: {
        features: { elapsedMs: stagesMs[0] },
        matches: { elapsedMs: stagesMs[1] },
        ...(stagesMs[2] ? { kltPush: { elapsedMs: stagesMs[2] } } : {}),
      },
      chunks: { kltTrack: { longestMs: kltChunk }, other: { longestMs: 400 } },
    },
  });
  const off = kltGateRun(record(1000, 4800, 0.25, [10000, 90000], 0)),
    klt = { alive: 10, ended: [0, 1, 2, 3, 4, 5, 6, 7], meanLived: 9 };
  assert.equal(off.frontMs, 100000);
  const good = kltGateRun(record(1000, 5000, 0.26, [9000, 60000, 5000], 200, { klt }));
  const rows = kltGateRows(off, good);
  assert.equal(rows.length, 7);
  assert.equal(kltGatePasses(rows), true);
  assert.equal(rows.at(-1).pass, null, 'the last row is recorded and has no gate');
  assert.equal(rows[5].pass, true, 'a chunk of another key above the budget does not count');
  // Each gated row fails alone.
  const fail = (run) => kltGateRows(off, kltGateRun(run)).filter((r) => r.pass === false).length;
  assert.equal(fail(record(1000, 4790, 0.26, [9000, 60000, 5000], 200)), 2, 'track length rows');
  assert.equal(
    fail(record(800, 4000, 0.26, [9000, 60000, 5000], 200)),
    1,
    'fewer landmarks, though more observations for each',
  );
  assert.equal(fail(record(1000, 5000, 0.31, [9000, 60000, 5000], 200)), 1, 'error');
  assert.equal(fail(record(1000, 5000, 0.26, [9000, 90000, 5000], 200)), 1, 'stage times');
  assert.equal(fail(record(1000, 5000, 0.26, [9000, 60000, 5000], 251)), 1, 'chunk');
});
