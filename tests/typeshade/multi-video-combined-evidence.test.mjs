import test from 'node:test';
import assert from 'node:assert/strict';
import {
  combinedDigest,
  combinedEvidence,
  combinedGraphSummary,
  combinedInput,
  summarizeCombinedReceipt,
} from './multi-video-combined-cases.mjs';

test('combined evidence hashes complete numeric bits, typed ranges and stable object keys', async () => {
  assert.equal(await combinedDigest({ a: 1, b: 2 }), await combinedDigest({ b: 2, a: 1 }));
  assert.notEqual(await combinedDigest(0), await combinedDigest(-0));
  assert.notEqual(await combinedDigest(NaN), await combinedDigest(Infinity));
  const backing = new Float64Array([99, -0, 2, 77]);
  assert.deepEqual(
    combinedEvidence(backing.subarray(1, 3)),
    combinedEvidence(new Float64Array([-0, 2])),
  );
  assert.notDeepEqual(
    combinedEvidence(backing.subarray(1, 3)),
    combinedEvidence(new Float64Array([0, 2])),
  );
  assert.notEqual(
    await combinedDigest(new Uint8Array([1, 2])),
    await combinedDigest(new Uint32Array([1, 2])),
  );
});

test('combined public receipt keeps work and hashes without raw paths or geometry', () => {
  const counters = {
    applicationUploadBytes: 10,
    applicationReadbackBytes: 2,
    liveBytes: 0,
    kernels: [{ dispatches: 3, groups: 4, timedGroups: 1 }],
  };
  const row = {
    variant: 'candidate',
    pair: 0,
    counts: { registered: 30 },
    digests: { components: 'sha256', inputBefore: 'same', inputAfter: 'same' },
    mappingMs: 20,
    observationMs: 1,
    stageMs: { pairMatchingMs: 2 },
    mapperStages: [
      {
        operation: 'bundle',
        dispatches: 3,
        submissions: 1,
        applicationUploadBytes: 10,
        applicationReadbackBytes: 2,
      },
    ],
    counters,
    observationCounters: counters,
    sourceContainerLength: 30,
    rawSourcePath: '/private/source',
    positions: new Float64Array([1, 2, 3]),
    descriptorCache: { hits: 5, misses: 2, privatePath: '/private/cache' },
  };
  const summary = summarizeCombinedReceipt({
    scope: 'generated tracks',
    excludes: ['RSS'],
    fixture: { videos: 15 },
    modules: { candidate: 'sha256' },
    backend: 'webgpu',
    pairs: 1,
    timeoutSeconds: 120,
    exactEvidenceEqual: true,
    rows: [row],
    owned: { ...row, sourceContainerLength: 0, hostRetention: { featuresAfterMappingBytes: 0 } },
    privatePath: '/private/result',
  });
  assert.equal(summary.exactEvidenceEqual, true);
  assert.equal(summary.protocol.wallSpeedClaim, false);
  assert.equal(summary.rows[0].counters.kernelDispatches, 3);
  assert.equal(summary.rows[0].mapperWork.bundle.calls, 1);
  assert.equal(summary.rows[0].digests.components, 'sha256');
  assert.equal(Object.keys(summary.rows[0].descriptorCache).length, 12);
  assert.equal(summary.owned.hostRetention.featuresAfterMappingBytes, 0);
  assert.doesNotMatch(JSON.stringify(summary), /\/private|positions/);
});

test('combined fixture reconstruction owns its containers and graph summary retains scalar verdicts', () => {
  const source = [{ camera: { fx: 1 }, keypoints: [2, 3], descriptors: [4], global: [5] }];
  const input = combinedInput(source);
  input[0].camera.fx = 7;
  input[0].keypoints[0] = 8;
  assert.equal(source[0].camera.fx, 1);
  assert.equal(source[0].keypoints[0], 2);
  assert.ok(input[0].keypoints instanceof Float64Array);
  assert.ok(input[0].descriptors instanceof Uint8Array);
  assert.ok(input[0].global instanceof Float32Array);
  const graph = {
    components: [],
    videos: [],
    stats: { edges: 1 },
    edges: [
      {
        first: 0,
        second: 1,
        crossVideo: true,
        inliers: 32,
        homographyInliers: 10,
        medianAngleDeg: 3,
        seedEligible: true,
        source: 'retrieval',
        matches: new Uint32Array([1, 2]),
      },
    ],
  };
  const summary = combinedGraphSummary(graph);
  assert.equal(summary.edges[0].seedEligible, true);
  assert.equal(summary.edges[0].inliers, 32);
  assert.equal(summary.edges[0].matches, undefined);
});
