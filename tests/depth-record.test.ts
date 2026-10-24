import test from 'node:test';
import assert from 'node:assert/strict';
import { inverseDepthRecord } from '../src/reconstruction/depth-record.ts';
import { InverseDepthTracker } from '../src/reconstruction/inverse-depth-tracker.ts';

test('portable depth records preserve unknown states, units and assumed noise without inventing geometry', () => {
  const tracker = new InverseDepthTracker({ pixelNoiseStdDevPx: 0.25 }),
    images = [0, 1].map((id) => ({
      id,
      camera: { fx: 100, fy: 100, cx: 20, cy: 20 },
      keypoints: Float64Array.of(10, 10),
    }));
  const { snapshot } = tracker.update({
    images,
    tracks: [
      {
        id: 7,
        observations: [
          { viewId: 0, featureId: 0 },
          { viewId: 1, featureId: 0 },
        ],
      },
    ],
  });
  const record = JSON.parse(JSON.stringify(inverseDepthRecord(snapshot)));
  assert.equal(record.format, 'typecv-inverse-depth-v1');
  assert.equal(record.units, 'current-monocular-map-gauge');
  assert.equal(record.assumptions.pixelNoiseStdDevPx, 0.25);
  assert.equal(record.tracks[0].status, 'bearing-only');
  assert.ok(Array.isArray(record.tracks[0].bearing));
  assert.ok(!('estimate' in record.tracks[0]) && !('covariance' in record.tracks[0]));
});
