import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reconstructIncremental } from '../src/index.ts';
import type { ImageFeatures, PairMatches } from '../src/index.ts';
import { alignPoses } from '../website/src/lib/pose-alignment.ts';

// An owner run on 2026-10-21 of the lab with the COLMAP graham-hall sample (12 photos, 640 × 427,
// SIFT with first octave -1, guided matching) registered 9 views. The lab matches each new frame
// against the first frame and the three before it, so photos 8 to 11 reached photos 0 to 7 through
// a few matches and matched each other strongly. The fixture holds the keypoints and the 38 pair
// matches of that front end (only the keypoints a match uses), the input of the mapper.
const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/graham-hall-lab-matches.json', import.meta.url), 'utf8'),
) as {
  frames: { id: number; camera: ImageFeatures['camera']; keypoints: number[] }[];
  pairs: { first: number; second: number; matches: number[] }[];
};
const sample = JSON.parse(
  readFileSync(
    new URL('../website/public/sample/graham-hall/sample.json', import.meta.url),
    'utf8',
  ),
) as { frames: { id: number; reference: { R: number[]; t: number[] } }[] };

const images: ImageFeatures[] = fixture.frames.map((f) => ({
  id: f.id,
  camera: f.camera,
  keypoints: Float64Array.from(f.keypoints),
}));
const pairs: PairMatches[] = fixture.pairs.map((p) => ({
  first: p.first,
  second: p.second,
  matches: Uint32Array.from(p.matches),
}));
const lab = { ransacIterations: 500, maxErrorPx: 3, bundleIterations: 4, finalBundleIterations: 8 };

test('graham-hall with the lab pair set registers all 12 photos near the COLMAP reference', () => {
  const result = reconstructIncremental(images, pairs, lab);
  assert.deepEqual(
    result.reconstruction.views.map((v) => v.id).sort((a, b) => a - b),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    JSON.stringify(result.unregistered),
  );
  assert.ok(result.diagnostics.meanReprojectionErrorPx < 0.5);
  const alignment = alignPoses(
    result.reconstruction.views.map((v) => ({ id: v.id, R: v.pose.R, t: v.pose.t })),
    sample.frames.map((f) => ({ id: f.id, R: f.reference.R, t: f.reference.t })),
  );
  assert.equal(alignment.status, 'aligned');
  assert.equal(alignment.commonViews, 12);
  assert.ok(alignment.rotationDeg.median < 0.5, `median ${alignment.rotationDeg.median}`);
  assert.ok(alignment.rotationDeg.max < 1, `max ${alignment.rotationDeg.max}`);
  assert.ok(alignment.centerOfRadius.max < 0.8, `center ${alignment.centerOfRadius.max}`);
});
