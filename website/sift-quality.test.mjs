import test from 'node:test';
import assert from 'node:assert/strict';
import { siftQuality, UPSAMPLED_EDGE_LIMIT } from './src/lib/sift-quality.ts';

test('the lab upsamples SIFT at 640 px and below, guides matching by default, and obeys the query', () => {
  assert.equal(UPSAMPLED_EDGE_LIMIT, 640);
  assert.deepEqual(siftQuality(640), { firstOctave: -1, guided: true });
  assert.deepEqual(siftQuality(480), { firstOctave: -1, guided: true });
  assert.deepEqual(siftQuality(1280), { firstOctave: 0, guided: true });
  assert.deepEqual(siftQuality(640, '?octave=0&guided=0'), { firstOctave: 0, guided: false });
  assert.deepEqual(siftQuality(1280, '?octave=-1'), { firstOctave: -1, guided: true });
  assert.deepEqual(siftQuality(640, '?guided=1&octave=7'), { firstOctave: -1, guided: true });
});
