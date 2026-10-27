// The gate of phase 2 of docs/library-structure.md (section 5): the lab's options module holds the
// values that the worker, the page and the run record used before phase 2 (section 4, rule 6: no
// change of a lab preset value while it is centralized). Each copy below is the value on main at
// 4022861, with the order of its keys: the run record and the transcripts write the options in it.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DENSE_LAB_FUSION,
  LAB_KLT_CAPACITY,
  LAB_OPTIONS,
  LAB_PAGE_ANALYSIS_EDGE,
  LAB_WORKER_DEFAULTS,
  MULTI_VIDEO_RECORD_OPTIONS,
  ORB_LAB_FEATURES,
  ORB_LAB_RATIO,
  ORB_MAX_DISTANCE,
  PHOTO_PROFILE,
  denseCleanDefault,
} from './src/lib/lab-options.ts';
import { SNAPSHOT_INTERVAL_MS } from './src/lib/multi-video-job.ts';

/** `LAB_OPTIONS` of `src/workers/sfm.worker.ts` on 4022861. */
const LAB_OPTIONS_TODAY = {
  windowSize: 32,
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
  frontEnd: 'sift',
  threshold: 0.045,
  maxFeatures: 4096,
  smoothDescriptors: true,
  matchRatio: 0.8,
  maxDistance: 1e19,
  keyframeMinMotionPx: 8,
  keyframeMaxGapSeconds: 1,
  videoSampleFps: 4,
  snapshotIntervalMs: 500,
  detailedStageLimit: 400,
  denseCleanOptions: { cluster: { minFraction: 0.1 } },
  dense: {
    maxSources: 6,
    bestSources: 3,
    windowRadius: 5,
    windowStep: 2,
    photometricIterations: 8,
    geometricIterations: 4,
    minConsistentViews: 3,
    tsdf: { visibility: 'view-weighted' },
    planarDepth: true,
  },
};

const same = (actual, expected, name) =>
  assert.equal(JSON.stringify(actual), JSON.stringify(expected), name);

test('LAB_OPTIONS and the photograph profile hold their values on 4022861', () => {
  same(LAB_OPTIONS, LAB_OPTIONS_TODAY, 'LAB_OPTIONS');
  same(PHOTO_PROFILE, { refineSharedFocal: false }, 'PHOTO_PROFILE');
});

test('the worker, page and record defaults hold their values on 4022861', () => {
  same([ORB_LAB_FEATURES, ORB_LAB_RATIO, ORB_MAX_DISTANCE], [4000, 0.95, 256], 'ORB');
  assert.equal(LAB_KLT_CAPACITY, 512);
  same(
    LAB_WORKER_DEFAULTS,
    {
      analysisEdge: 640,
      firstOctave: 0,
      guided: false,
      dense: false,
      gate: 'legacy',
      focalSearch: false,
    },
    'LAB_WORKER_DEFAULTS',
  );
  assert.equal(LAB_PAGE_ANALYSIS_EDGE, 1280);
  same(
    MULTI_VIDEO_RECORD_OPTIONS,
    { windowSize: 0, clahe: false, keyframeGate: 'phase' },
    'MULTI_VIDEO_RECORD_OPTIONS',
  );
  assert.equal(SNAPSHOT_INTERVAL_MS, 500);
  assert.equal(DENSE_LAB_FUSION, 'tsdf');
  same([denseCleanDefault('pixel'), denseCleanDefault('tsdf')], [true, false], 'denseCleanDefault');
});
