// Verifies the lab's pair selection (`website/src/lib/photo-pairs.ts`): a photo set is matched
// in all pairs while it is small, and a video keeps the window of the first frame and three
// recent frames.
import test from 'node:test';
import assert from 'node:assert/strict';
import { partnerIds, retainedFeatureIds } from './src/lib/photo-pairs.ts';

/** The worker's loop on ids only: match against the partners, register, prune the features. */
function simulate(profile, count) {
  let retained = [];
  let references = [];
  let pairs = 0;
  for (let id = 0; id < count; id++) {
    pairs += partnerIds(
      profile,
      retained.map((r) => ({ id: r })),
    ).length;
    retained.push(id);
    // One second apart: every frame becomes a reference, as in `registerFrame`.
    references.push(id);
    if (references.length > 3) references.splice(1, references.length - 3);
    const keep = retainedFeatureIds(profile, references, retained);
    retained = retained.filter((r) => keep.has(r));
  }
  return pairs;
}

test('the photo profile matches all 66 pairs of 12 photos', () => {
  assert.equal(simulate('photo', 12), 66);
});

test('the video profile keeps its window: 38 pairs of 12 frames', () => {
  assert.equal(simulate('video', 12), 38);
});

test('a photo set larger than the session window uses retrieval candidates', () => {
  const unit = (i) => Float32Array.from({ length: 4 }, (_, k) => (k === i % 4 ? 1 : 0));
  const candidates = Array.from({ length: 32 }, (_, id) => ({ id, global: unit(id) }));
  const ids = partnerIds('photo', candidates, 40, unit(2));
  // First view, three latest views and the ten best by similarity (ids 2, 6, ... 38 -> in the set: 2..30 step 4).
  assert.ok(ids.includes(0) && ids.includes(29) && ids.includes(31));
  for (const id of [2, 6, 10, 14, 18, 22, 26, 30]) assert.ok(ids.includes(id), 'retrieved ' + id);
  assert.ok(ids.length <= 1 + 3 + 10);
  // Without global descriptors: the first view and the order neighbours.
  assert.deepEqual(
    partnerIds(
      'photo',
      candidates.map((c) => ({ id: c.id })),
      40,
    ),
    [0, 29, 30, 31],
  );
});

test('a video frame is always matched against all its retained views', () => {
  const candidates = Array.from({ length: 4 }, (_, id) => ({ id }));
  assert.deepEqual(partnerIds('video', candidates, 100), [0, 1, 2, 3]);
});
