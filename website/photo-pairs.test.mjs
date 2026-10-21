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
