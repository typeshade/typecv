// Phase 3 of the refactoring programme (`docs/library-structure.md`, section 5): the table tests of
// the lab's named rules. Each rule was inline in `src/workers/sfm.worker.ts` (the feature minimum
// also in `src/lib/multi-video-job.ts`) on `main` at 87724f9, the parent of phase 3. Each table
// holds the rule's value and unit, and rows on both sides of each branch. `inline` is the
// expression of the branch as it stood there, copied with its literals, and every row holds the
// named rule to it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MIN_FEATURES, tooFewFeatures } from './src/lib/lab-features.ts';
import {
  ADJACENT_KEYFRAMES,
  MAX_REFERENCES,
  PAIR_MIN_MATCH_ENTRIES,
  PRECOMPUTED_MIN_MATCH_ENTRIES,
  REFERENCE_SPACING_SECONDS,
  admitsPair,
  admitsPrecomputedPair,
  retainedFeatureIds,
  retainedReferences,
  tracksCoverPair,
} from './src/lib/lab-matching.ts';
import {
  LEGACY_GATE_MIN_MATCH_ENTRIES,
  MOTION_REFERENCE_WIDTH,
  legacyGateSkips,
  scaledMotionThresholdPx,
  trackGateDecision,
} from './src/lib/keyframe-gate.ts';
import { GUIDED_MIN_MATCH_ENTRIES, guidedMatchingRuns } from './src/lib/guided-matching.ts';
import { KLT_COVER_MATCHES, MIN_PAIR_MATCHES, mergeTrackerPairs } from './src/lib/klt-join.ts';

/** `LAB_OPTIONS.keyframeMinMotionPx` (px at 640 px) and `keyframeMaxGapSeconds` (s). */
const LAB = { keyframeMinMotionPx: 8, keyframeMaxGapSeconds: 1 };

test('the feature minimum: 8 detected features (processFrame, MultiVideoJob.frame)', () => {
  assert.equal(MIN_FEATURES, 8);
  const inline = (count) => count < 8;
  for (const [count, rejected] of [
    [0, true],
    [7, true],
    [8, false],
    [9, false],
    [4096, false],
  ]) {
    assert.equal(tooFewFeatures(count), rejected, `${count} features`);
    assert.equal(inline(count), rejected, `${count} features, inline`);
  }
});

test('the pair minimums: 16 entries, 32 entries and 16 tracker matches', () => {
  assert.equal(PAIR_MIN_MATCH_ENTRIES, 16);
  assert.equal(PRECOMPUTED_MIN_MATCH_ENTRIES, 32);
  assert.equal(MIN_PAIR_MATCHES, 16);
  // processFrame, the pair admission: `matched.matches.length >= 16`.
  const descriptor = (entries) => entries >= 16;
  // processPrecomputedFrame: a pair is skipped when `pair.matches.length < 32`.
  const precomputed = (entries) => !(entries < 32);
  for (const [entries, pair, precomputedPair] of [
    [0, false, false],
    [14, false, false],
    [15, false, false],
    [16, true, false],
    [30, true, false],
    [31, true, false],
    [32, true, true],
    [64, true, true],
  ]) {
    assert.equal(admitsPair(entries), pair, `${entries} entries, descriptor pair`);
    assert.equal(descriptor(entries), pair, `${entries} entries, descriptor pair, inline`);
    assert.equal(
      admitsPrecomputedPair(entries),
      precomputedPair,
      `${entries} entries, precomputed`,
    );
    assert.equal(precomputed(entries), precomputedPair, `${entries} entries, precomputed, inline`);
  }
  // mergeTrackerPairs (klt-join.ts): a tracker-only pair goes to the mapper with
  // `rows.length / 2 >= MIN_PAIR_MATCHES` matches.
  const trackerOnly = (matches) =>
    mergeTrackerPairs(
      [],
      [
        {
          first: 1,
          second: 2,
          matches: Uint32Array.from({ length: 2 * matches }, (_, i) => i >> 1),
          tracks: Uint32Array.from({ length: matches }, (_, i) => i),
        },
      ],
    ).pairs.length === 1;
  for (const [matches, pair] of [
    [8, false],
    [15, false],
    [16, true],
    [17, true],
  ])
    assert.equal(trackerOnly(matches), pair, `${matches} tracker matches`);
});

test('the retention rule: 0.5 s, 3 references, 3 newest frames (registerFrame)', () => {
  assert.equal(REFERENCE_SPACING_SECONDS, 0.5);
  assert.equal(MAX_REFERENCES, 3);
  assert.equal(ADJACENT_KEYFRAMES, 3);
  // registerFrame, with `active` written out.
  const inline = (references, features, frame) => {
    references = references.filter((id) => features.has(id));
    const reference = features.get(references.at(-1) ?? -1);
    if (!reference || frame.time - reference.frame.time >= 0.5) references.push(frame.id);
    if (references.length > 3) references.splice(1, references.length - 3);
    const keepFeatures = new Set([...references, ...[...features.keys()].slice(-3)]);
    return { references, keep: [...keepFeatures] };
  };
  const held = (times) =>
    new Map(Object.entries(times).map(([id, time]) => [Number(id), { frame: { time } }]));
  for (const [what, references, times, frame, expected] of [
    ['the first keyframe', [], { 0: 0 }, { id: 0, time: 0 }, { references: [0], keep: [0] }],
    [
      '0.49 s after the last reference',
      [0],
      { 0: 0, 1: 0.49 },
      { id: 1, time: 0.49 },
      { references: [0], keep: [0, 1] },
    ],
    [
      '0.5 s after the last reference',
      [0],
      { 0: 0, 1: 0.5 },
      { id: 1, time: 0.5 },
      { references: [0, 1], keep: [0, 1] },
    ],
    [
      'a fourth reference: the first and the two most recent stay',
      [0, 1, 2],
      { 0: 0, 1: 1, 2: 2, 3: 3 },
      { id: 3, time: 3 },
      { references: [0, 2, 3], keep: [0, 2, 3, 1] },
    ],
    [
      'a reference whose features went',
      [0, 5],
      { 0: 0, 6: 3.2, 7: 3.4 },
      { id: 7, time: 3.4 },
      { references: [0, 7], keep: [0, 7, 6] },
    ],
    [
      'no reference held',
      [4],
      { 8: 9, 9: 9.1 },
      { id: 9, time: 9.1 },
      { references: [9], keep: [9, 8] },
    ],
    [
      'the three newest frames besides the references',
      [0],
      { 0: 0, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.4 },
      { id: 6, time: 1.4 },
      { references: [0, 6], keep: [0, 6, 4, 5] },
    ],
  ]) {
    const features = held(times);
    const kept = retainedReferences(references, features, frame);
    const actual = { references: kept, keep: [...retainedFeatureIds(kept, features.keys())] };
    assert.deepEqual(actual, expected, what);
    assert.deepEqual(inline([...references], features, frame), expected, what + ', inline');
  }
});

test('the legacy rule: 16 entries, 8 px at 640 px and under 1 s (processFrame)', () => {
  assert.equal(LEGACY_GATE_MIN_MATCH_ENTRIES, 16);
  assert.equal(MOTION_REFERENCE_WIDTH, 640);
  assert.equal(scaledMotionThresholdPx(8, 640), 8);
  assert.equal(scaledMotionThresholdPx(8, 1280), 16);
  assert.equal(scaledMotionThresholdPx(8, 320), 4);
  const inline = (entries, motion, gap, width) => {
    const scaledMinimum = (LAB.keyframeMinMotionPx * width) / 640;
    return entries >= 16 && motion < scaledMinimum && gap < LAB.keyframeMaxGapSeconds;
  };
  for (const [entries, motion, gap, width, skip] of [
    [16, 0, 0, 640, true],
    [16, 7.99, 0.99, 640, true],
    [15, 0, 0, 640, false],
    [16, 8, 0.5, 640, false],
    [16, 1, 1, 640, false],
    [16, Infinity, 0.1, 640, false],
    [16, 15.9, 0.5, 1280, true],
    [16, 16, 0.5, 1280, false],
    [32, 3.9, 0.2, 320, true],
    [32, 4, 0.2, 320, false],
    [16, 7.5, 0.1, 601, true],
    [16, 7.52, 0.1, 601, false],
  ]) {
    const row = `${entries} entries, ${motion} px, ${gap} s, ${width} px wide`;
    assert.equal(legacyGateSkips(entries, motion, gap, width, LAB), skip, row);
    assert.equal(inline(entries, motion, gap, width), skip, row + ', inline');
  }
});

test('the track rule: the median track displacement against 8 px at 640 px (processFrame)', () => {
  const inline = (median, gap, width) => {
    const scaledMinimum = (LAB.keyframeMinMotionPx * width) / 640;
    if (!Number.isFinite(median)) return 'undecided';
    return median < scaledMinimum && gap < LAB.keyframeMaxGapSeconds ? 'skip' : 'keep';
  };
  for (const [median, gap, width, decision] of [
    [Infinity, 0.1, 640, 'undecided'],
    [NaN, 0.1, 640, 'undecided'],
    [0, 0.1, 640, 'skip'],
    [7.9, 0.9, 640, 'skip'],
    [8, 0.1, 640, 'keep'],
    [0, 1, 640, 'keep'],
    [15, 0.5, 1280, 'skip'],
    [16, 0.5, 1280, 'keep'],
  ]) {
    const row = `${median} px, ${gap} s, ${width} px wide`;
    assert.equal(trackGateDecision(median, gap, width, LAB), decision, row);
    assert.equal(inline(median, gap, width), decision, row + ', inline');
  }
});

test('the guided trigger: SIFT, guided matching on and 16 plain entries (processFrame)', () => {
  assert.equal(GUIDED_MIN_MATCH_ENTRIES, 16);
  const inline = (sift, guided, entries) => sift && guided && entries >= 16;
  for (const [sift, guided, entries, runs] of [
    [true, true, 16, true],
    [true, true, 200, true],
    [true, true, 15, false],
    [true, false, 200, false],
    [false, true, 200, false],
    [false, false, 0, false],
  ]) {
    const row = `sift ${sift}, guided ${guided}, ${entries} entries`;
    assert.equal(guidedMatchingRuns(sift, guided, entries), runs, row);
    assert.equal(inline(sift, guided, entries), runs, row + ', inline');
  }
});

test('KLT cover: 64 tracker matches skip the descriptor matching (processFrame)', () => {
  assert.equal(KLT_COVER_MATCHES, 64);
  const inline = (entries) => entries / 2 >= KLT_COVER_MATCHES;
  for (const [entries, covered] of [
    [0, false],
    [126, false],
    [127, false],
    [128, true],
    [129, true],
    [512, true],
  ]) {
    assert.equal(tracksCoverPair(entries), covered, `${entries} entries`);
    assert.equal(inline(entries), covered, `${entries} entries, inline`);
  }
});
