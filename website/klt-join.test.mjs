// Verifies: docs/klt-tracking.md step 5 (sections 3.3 and 3.5: the merge into the pairs and the verdict).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MIN_PAIR_MATCHES,
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
