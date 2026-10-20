import test from 'node:test';
import assert from 'node:assert/strict';
import { featureFrontEnd, siftQuality, UPSAMPLED_EDGE_LIMIT } from './src/lib/sift-quality.ts';

test('the lab upsamples SIFT at 640 px and below, guides matching by default, and obeys the query', () => {
  assert.equal(UPSAMPLED_EDGE_LIMIT, 640);
  assert.deepEqual(siftQuality(640), { firstOctave: -1, guided: true, clahe: false });
  assert.deepEqual(siftQuality(480), { firstOctave: -1, guided: true, clahe: false });
  assert.deepEqual(siftQuality(1280), { firstOctave: 0, guided: true, clahe: false });
  assert.deepEqual(siftQuality(640, '?octave=0&guided=0'), {
    firstOctave: 0,
    guided: false,
    clahe: false,
  });
  assert.deepEqual(siftQuality(1280, '?octave=-1'), {
    firstOctave: -1,
    guided: true,
    clahe: false,
  });
  assert.deepEqual(siftQuality(640, '?guided=1&octave=7'), {
    firstOctave: -1,
    guided: true,
    clahe: false,
  });
});

test('?clahe=1 turns CLAHE on and nothing else does', () => {
  assert.equal(siftQuality(640).clahe, false);
  assert.equal(siftQuality(640, '?clahe=1').clahe, true);
  assert.equal(siftQuality(640, '?clahe=0').clahe, false);
  assert.equal(siftQuality(640, '?clahe=true').clahe, false);
});

test('?features=orb selects the ORB front end and nothing else does', () => {
  assert.equal(featureFrontEnd(), 'sift');
  assert.equal(featureFrontEnd('?features=orb'), 'orb');
  assert.equal(featureFrontEnd('?features=sift'), 'sift');
  assert.equal(featureFrontEnd('?features=ORB'), 'sift');
  assert.equal(featureFrontEnd('?gate=phase&features=orb'), 'orb');
  // It leaves the SIFT settings alone.
  assert.deepEqual(siftQuality(640, '?features=orb'), siftQuality(640));
});
