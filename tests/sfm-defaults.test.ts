import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_SEED,
  HOMOGRAPHY_DEFAULTS,
  PNP_DEFAULTS,
  TRIANGULATION_DEFAULTS,
  imagePairHomographyIterations,
  pnpMinInliers,
} from '../src/geometry/operator-defaults.ts';
import {
  BUNDLE_DEFAULTS,
  BUNDLE_SOLVER_STALL_TOLERANCE,
  DEFAULT_BUNDLE_STALL_TOLERANCE,
  DEFAULT_GLOBAL_BUNDLE_RATIO,
  MAPPER_DEFAULTS,
  SESSION_DEFAULTS,
  bundleHuberPx,
  maxErrorOfHuberPx,
} from '../src/sfm/config/defaults.ts';
import { bundleStalled } from '../src/sfm/bundle/bundle-shared.ts';
import { observationKey, pairKey } from '../src/contracts/keys.ts';
import { TriangulationStatus } from '../src/contracts/triangulation-status.ts';
import {
  CHEIRALITY_FAILURE_PREFIX,
  RECOVERY_FAILURES,
  TwoViewFailure,
} from '../src/contracts/two-view-failures.ts';

// Phase 2 of docs/library-structure.md (section 5) writes each default, code and message once.
// Section 4, rule 1: no number, default or message text changes. Each value below is the literal
// that its sites held on main at 4022861. Each of those sites now reads the definition.

const same = (actual: unknown, expected: unknown) =>
  assert.equal(JSON.stringify(actual), JSON.stringify(expected));

test('the operator defaults are the literals of both tiers on 4022861', () => {
  assert.equal(DEFAULT_SEED, 17);
  same(PNP_DEFAULTS, {
    iterations: 1000,
    thresholdPx: 3,
    confidence: 0.999,
    seed: 17,
    refinementIterations: 20,
  });
  assert.deepEqual([pnpMinInliers(4), pnpMinInliers(6), pnpMinInliers(40)], [4, 6, 6]);
  same(TRIANGULATION_DEFAULTS, { maxErrorPx: 3, minAngleDeg: 0.5, maxPairHypotheses: 32 });
  same(HOMOGRAPHY_DEFAULTS, { thresholdPx: 2, iterations: 500, confidence: 0.999, seed: 17 });
  assert.deepEqual(
    [imagePairHomographyIterations(750), imagePairHomographyIterations(64)],
    [500, 64],
  );
});

test('the mapper, session and solver defaults are the literals on 4022861', () => {
  same(MAPPER_DEFAULTS, {
    seed: 17,
    ransacIterations: 750,
    maxErrorPx: 3,
    minInitialAngleDeg: 1,
    minTriangulationAngleDeg: 0.5,
    minPnPInliers: 12,
    minInitialPoints: 12,
    bundleIterations: 8,
    finalBundleIterations: 25,
    localBundleSize: 6,
    localBundleSubset: true,
    globalBundleFrequency: 0,
    pairRefinementIterations: 0,
    refineSharedFocal: false,
  });
  same(SESSION_DEFAULTS, { windowSize: 32, finalBundleIterations: 8 });
  same(BUNDLE_DEFAULTS, { iterations: 20, huberPx: 2, cgRelativeTolerance: 1e-6 });
  assert.equal(DEFAULT_GLOBAL_BUNDLE_RATIO, 1.1);
  assert.equal(DEFAULT_BUNDLE_STALL_TOLERANCE, 1e-4);
  assert.equal(BUNDLE_SOLVER_STALL_TOLERANCE, 1e-9);
  assert.deepEqual([bundleHuberPx(3), maxErrorOfHuberPx(1.5)], [1.5, 3]);
  // The relative stall test: a change below the tolerance times the cost, at least 1.
  assert.deepEqual(
    [bundleStalled(0.5e-4, 1, 1e-4), bundleStalled(2e-4, 1, 1e-4), bundleStalled(5e-4, 10, 1e-4)],
    [true, false, true],
  );
});

test('the keys, the triangulation status and the two-view reasons keep their spellings', () => {
  assert.equal(observationKey(3, 41), '3:41');
  assert.deepEqual([pairKey(7, 2), pairKey(2, 7), pairKey(0, 0)], ['2:7', '2:7', '0:0']);
  same(TriangulationStatus, {
    accepted: 0,
    insufficientObservations: 1,
    noConsensus: 3,
    lowParallax: 4,
    depthSensitivity: 5,
  });
  assert.equal(CHEIRALITY_FAILURE_PREFIX, 'Cheirality failed');
  same(TwoViewFailure, {
    noGeometry: 'Could not find a valid two-view geometry',
    cheirality: 'Cheirality failed: low parallax, planar scene, or invalid matching',
    lowParallax: 'Low parallax: insufficient baseline for stable reconstruction',
    fewReliablePoints: 'Insufficient reliable points after parallax and reprojection filtering',
    gpuRefinement: 'GPU two-view numerical refinement or median selection failed',
    gpuRecovery: 'GPU two-view recovery failed',
    fewInvertible: 'Insufficient invertible correspondences',
  });
  same(RECOVERY_FAILURES, [
    '',
    'Could not find a valid two-view geometry',
    'Cheirality failed: low parallax, planar scene, or invalid matching',
    'Low parallax: insufficient baseline for stable reconstruction',
    'Insufficient reliable points after parallax and reprojection filtering',
    'GPU two-view numerical refinement or median selection failed',
  ]);
});
