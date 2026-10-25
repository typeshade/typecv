import test from 'node:test';
import assert from 'node:assert/strict';
import { historyFocalSeeds, type FocalSearchResult } from '../src/reconstruction/focal-search.ts';

function search(winner: number): FocalSearchResult {
  return {
    ratio: winner,
    gridRatio: winner,
    views: [0, 1, 2, 3, 4],
    candidates: [0.3, 0.6, 1.2, 2.4].map((ratio) => ({
      ratio,
      views: 5,
      rmsePx: 0.2,
      observations: 400,
      score: ratio === winner ? 0.04 : 0.05,
    })),
  };
}

test('late joint calibration explores the original range when early evidence chose an edge', () => {
  assert.deepEqual(historyFocalSeeds(search(0.3), 0.32), [0.32, 0.6, 2.4]);
  assert.deepEqual(historyFocalSeeds(search(2.4), 2.3), [2.3, 0.6, 0.3]);
  // A nearby local refinement must not erase the fact that the grid minimum was unbracketed.
  const refined = { ...search(0.3), ratio: 0.35 };
  assert.deepEqual(historyFocalSeeds(refined, 0.36), [0.36, 0.6, 2.4]);
});

test('a bracketed calibration adds no restarts and every schedule is bounded and unique', () => {
  assert.deepEqual(historyFocalSeeds(search(0.6), 0.65), [0.65]);
  assert.deepEqual(historyFocalSeeds(search(1.2), 1.1), [1.1]);
  assert.deepEqual(historyFocalSeeds(search(0.3), 0.6), [0.6, 2.4]);
  const one = search(0.3);
  assert.deepEqual(
    historyFocalSeeds({ ...one, candidates: one.candidates.slice(0, 1) }, 0.31),
    [0.31],
  );
  for (const current of [0, -1, NaN, Infinity])
    assert.throws(() => historyFocalSeeds(one, current), RangeError);
});
