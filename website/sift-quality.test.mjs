import test from 'node:test';
import assert from 'node:assert/strict';
import {
  featureFrontEnd,
  kltSetup,
  guidedKernel,
  kltTracking,
  siftQuality,
  UPSAMPLED_EDGE_LIMIT,
} from './src/lib/sift-quality.ts';

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

test('?klt=1 turns KLT tracking on and nothing else does', () => {
  assert.equal(kltTracking(), false);
  assert.equal(kltTracking('?klt=0'), false);
  assert.equal(kltTracking('?klt=1'), true);
  assert.equal(kltTracking('?gate=phase&klt=1'), true);
  assert.equal(kltTracking('?klt=true'), false);
  // It leaves the SIFT settings alone.
  assert.deepEqual(siftQuality(640, '?klt=1'), siftQuality(640));
});

test('the step 6 KLT switches parse and ignore what they do not name', () => {
  assert.equal(kltSetup(), undefined);
  assert.equal(kltSetup('?klt=1'), undefined);
  assert.deepEqual(kltSetup('?klt=1&kltCapacity=512'), { capacity: 512 });
  assert.deepEqual(kltSetup('?kltFill=1&kltMatching=tracks&kltGate=1&kltCapacity=2048'), {
    capacity: 2048,
    fillIn: true,
    matching: 'tracks',
    gate: true,
  });
  for (const bad of ['?kltCapacity=0', '?kltCapacity=9000', '?kltCapacity=1.5', '?kltCapacity=x'])
    assert.equal(kltSetup(bad), undefined, bad);
  for (const other of ['?kltFill=0', '?kltMatching=both', '?kltGate=0'])
    assert.equal(kltSetup(other), undefined, other);
});

test('?guided=tiled picks the tiled guided kernel, keeps guidance on, and nothing else does', () => {
  assert.equal(guidedKernel(), 'scalar');
  assert.equal(guidedKernel('?guided=1'), 'scalar');
  assert.equal(guidedKernel('?guided=0'), 'scalar');
  assert.equal(guidedKernel('?guided=scalar'), 'scalar');
  assert.equal(guidedKernel('?guided=tiled'), 'tiled');
  assert.equal(siftQuality(640, '?guided=tiled').guided, true);
});
