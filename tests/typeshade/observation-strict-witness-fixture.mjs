import assert from 'node:assert/strict';
import { renderScene } from '../klt-cases.ts';

export function strictWitnessFixture() {
  const width = 192,
    height = 144,
    camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
    unowned = Array.from({ length: 8 }, (_, i) => [50.1 + i * 0.1, 50.1]).flat(),
    pair = (first, second, a = 0, b = 0) => ({
      first,
      second,
      matches: [a, b],
      distances: [0.1],
    }),
    required = (keypoints, pairs, extra = {}) => ({
      keypoints,
      pairs,
      mode: 'required',
      verified: true,
      ...extra,
    });
  return {
    frames: Array.from({ length: 4 }, (_, id) => ({
      id,
      width,
      height,
      camera: { ...camera },
      // Rendered once in Node and serialized exactly for both execution backends.
      gray: Array.from(renderScene({ width, height }, { dx: 2 * id, dy: -id })),
    })),
    cases: {
      plain: { keypoints: [[50, 50], []], pairs: [] },
      legacy: { keypoints: [[50, 50], []], pairs: [], mode: true },
      unavailable: required([[50, 50], []], []),
      untrusted: required([[50, 50], [], [54, 48]], [pair(0, 2)], { verified: false }),
      verified: required([[50, 50], [], [54, 48]], [pair(0, 2)]),
      joined: required(
        [
          [50, 50],
          [52.2, 49],
          [54, 48],
        ],
        [pair(0, 2)],
      ),
      joinedPrimary: required([[50, 50], [52, 49], [54, 48], []], [pair(0, 2)], {
        radius: 2,
      }),
      badJoin: required(
        [
          [50, 50],
          [52.6, 49],
          [54, 48],
        ],
        [pair(0, 2)],
      ),
      descendants: required([[50, 50], [52, 49], [], []], [pair(0, 1)]),
      reverse: required([[], [52, 49], [54, 48]], [pair(1, 2)]),
      budget: required([[...unowned, 50, 50], [], [54, 48]], [pair(0, 2, 8)], {
        maxSeedsPerFrame: 1,
      }),
    },
  };
}

const edgeKey = (sourceViewId, sourceFeatureId, viewId, featureId) =>
  sourceViewId < viewId
    ? `${sourceViewId}:${sourceFeatureId}:${viewId}:${featureId}`
    : `${viewId}:${featureId}:${sourceViewId}:${sourceFeatureId}`;

function assertCaseEvidence(result, input, entry, name) {
  assert.equal(result.inputsUnchanged, true, `${name}: inputs were not mutated`);
  assert.equal(result.liveBytes, 0, `${name}: device allocations were released`);
  assert.equal(result.newEdges.length, result.diagnostics.addedEdges, name);
  assert.equal(result.newObservations.length, result.diagnostics.newCoordinates, name);
  const originalCounts = new Map(
      entry.keypoints.map((keypoints, i) => [input.frames[i].id, keypoints.length / 2]),
    ),
    originalEdges = new Set(),
    admittedEdges = new Set();
  for (const [i, keypoints] of entry.keypoints.entries()) {
    const frame = result.images.find((image) => image.id === input.frames[i].id);
    assert.ok(frame, `${name}: original frame identity`);
    assert.deepEqual(frame.keypoints.slice(0, keypoints.length), keypoints, name);
  }
  for (const original of entry.pairs) {
    const pair = result.pairs.find(
      (candidate) => candidate.first === original.first && candidate.second === original.second,
    );
    assert.ok(pair, `${name}: raw pair retained`);
    assert.deepEqual(pair.matches.slice(0, original.matches.length), original.matches, name);
    assert.deepEqual(pair.distances.slice(0, original.distances.length), original.distances, name);
    for (let i = 0; i < original.matches.length; i += 2)
      originalEdges.add(
        edgeKey(original.first, original.matches[i], original.second, original.matches[i + 1]),
      );
  }
  for (const edge of result.newEdges) {
    const sourceCount = originalCounts.get(edge.sourceViewId),
      target = result.images.find((frame) => frame.id === edge.viewId),
      key = edgeKey(edge.sourceViewId, edge.sourceFeatureId, edge.viewId, edge.featureId);
    assert.ok(sourceCount !== undefined && target, `${name}: source/target identities`);
    assert.ok(
      Number.isSafeInteger(edge.sourceFeatureId) &&
        edge.sourceFeatureId >= 0 &&
        edge.sourceFeatureId < sourceCount,
      `${name}: primary source is an original feature`,
    );
    assert.ok(
      Number.isSafeInteger(edge.featureId) &&
        edge.featureId >= 0 &&
        2 * edge.featureId + 1 < target.keypoints.length,
      `${name}: valid target feature identity`,
    );
    assert.equal(edge.geometry, 'candidate', name);
    assert.ok(!originalEdges.has(key), `${name}: original pair row is not new provenance`);
    assert.ok(!admittedEdges.has(key), `${name}: one provenance row per new pair edge`);
    admittedEdges.add(key);
    assert.deepEqual(
      edge.targetPixel,
      target.keypoints.slice(2 * edge.featureId, 2 * edge.featureId + 2),
      `${name}: actual preserved/appended target coordinate`,
    );
    const pair = result.pairs.find(
      (candidate) =>
        candidate.first === Math.min(edge.sourceViewId, edge.viewId) &&
        candidate.second === Math.max(edge.sourceViewId, edge.viewId),
    );
    assert.ok(
      pair &&
        Array.from({ length: pair.matches.length / 2 }, (_, i) =>
          edgeKey(pair.first, pair.matches[2 * i], pair.second, pair.matches[2 * i + 1]),
        ).includes(key),
      `${name}: provenance refers to its appended pair row`,
    );
    if (entry.mode === 'required') {
      assert.equal(edge.corroboration?.status, 'verified', name);
      const witness = edge.corroboration,
        anchorCount = originalCounts.get(witness.viewId);
      assert.notEqual(witness.viewId, edge.sourceViewId, name);
      assert.notEqual(witness.viewId, edge.viewId, name);
      assert.ok(
        anchorCount !== undefined &&
          Number.isSafeInteger(witness.featureId) &&
          witness.featureId >= 0 &&
          witness.featureId < anchorCount,
        `${name}: witness is an original feature`,
      );
      assert.ok(witness.disagreementPx <= 0.5, name);
      assert.ok(
        Math.hypot(
          edge.targetPixel[0] - edge.measuredPixel[0],
          edge.targetPixel[1] - edge.measuredPixel[1],
        ) <= 0.5 &&
          Math.hypot(
            edge.targetPixel[0] - witness.measuredPixel[0],
            edge.targetPixel[1] - witness.measuredPixel[1],
          ) <= 0.5,
        `${name}: canonical coordinate agrees with both image measurements`,
      );
    }
    if (edge.createdFeature) {
      assert.ok(edge.featureId >= originalCounts.get(edge.viewId), name);
      assert.ok(
        result.newObservations.some(
          (observation) =>
            edgeKey(
              observation.sourceViewId,
              observation.sourceFeatureId,
              observation.viewId,
              observation.featureId,
            ) === key,
        ),
        `${name}: created coordinate retains observation provenance`,
      );
    }
  }
  for (const observation of result.newObservations) {
    assert.ok(observation.featureId >= originalCounts.get(observation.viewId), name);
    assert.ok(
      result.newEdges.some(
        (edge) =>
          edge.createdFeature &&
          edge.sourceViewId === observation.sourceViewId &&
          edge.sourceFeatureId === observation.sourceFeatureId &&
          edge.viewId === observation.viewId &&
          edge.featureId === observation.featureId,
      ),
      `${name}: appended coordinate has a newly added edge`,
    );
  }
}

export function assertStrictWitnessCases(result, input) {
  assert.equal(result.inputsUnchanged, true);
  assert.equal(result.liveBytes, 0);
  assert.deepEqual(Object.keys(result.cases), Object.keys(input.cases));
  for (const [name, entry] of Object.entries(input.cases))
    assertCaseEvidence(result.cases[name], input, entry, name);
  const cases = result.cases;
  assert.equal(cases.plain.newObservations.length, 1);
  assert.deepEqual(cases.legacy.images, cases.plain.images);
  assert.deepEqual(cases.legacy.pairs, cases.plain.pairs);
  assert.equal(cases.legacy.newEdges[0].corroboration.status, 'unavailable');
  for (const name of ['unavailable', 'untrusted']) {
    assert.equal(cases[name].newEdges.length, 0, name);
    assert.equal(cases[name].newObservations.length, 0, name);
    assert.equal(cases[name].diagnostics.attempts, 0, name);
    assert.ok(cases[name].diagnostics.rejected.multiViewUnavailable > 0, name);
  }
  assert.equal(cases.unavailable.pairs.length, 0);
  assert.equal(cases.verified.newEdges.length, 1);
  assert.equal(cases.verified.newObservations.length, 1);
  assert.equal(cases.verified.newEdges[0].createdFeature, true);
  assert.equal(cases.verified.newEdges[0].sourceViewId, 0);
  assert.equal(cases.verified.newEdges[0].viewId, 1);
  assert.equal(cases.verified.newEdges[0].corroboration.viewId, 2);
  assert.equal(cases.joined.newEdges.length, 1);
  assert.equal(cases.joined.newObservations.length, 0);
  const join = cases.joined.newEdges[0];
  assert.equal(join.createdFeature, false);
  assert.deepEqual(join.targetPixel, [52.2, 49]);
  assert.ok(Math.hypot(join.measuredPixel[0] - 52, join.measuredPixel[1] - 49) < 0.1);
  assert.ok(Math.abs(join.measuredPixel[0] - join.targetPixel[0]) > 0.1);
  assert.equal(cases.joinedPrimary.newEdges.length, 2);
  assert.equal(cases.joinedPrimary.newObservations.length, 1);
  const [originalJoin, joinedPrimary] = cases.joinedPrimary.newEdges;
  assert.equal(originalJoin.sourceViewId, 0);
  assert.equal(originalJoin.viewId, 1);
  assert.equal(originalJoin.featureId, 0);
  assert.equal(originalJoin.createdFeature, false);
  assert.deepEqual(originalJoin.targetPixel, [52, 49]);
  assert.equal(joinedPrimary.sourceViewId, 1);
  assert.equal(joinedPrimary.sourceFeatureId, 0);
  assert.equal(joinedPrimary.viewId, 3);
  assert.equal(joinedPrimary.createdFeature, true);
  assert.equal(joinedPrimary.corroboration.viewId, 2);
  assert.equal(joinedPrimary.corroboration.featureId, 0);
  assert.equal(cases.badJoin.newEdges.length, 0);
  assert.equal(cases.badJoin.newObservations.length, 0);
  assert.ok(cases.badJoin.diagnostics.rejected.multiViewJoin > 0);
  assert.equal(cases.descendants.newEdges.length, 2);
  assert.deepEqual(
    cases.descendants.newEdges.map((edge) => edge.viewId),
    [2, 3],
  );
  assert.equal(
    cases.descendants.newEdges.find((edge) => edge.viewId === 3).corroboration.viewId,
    1,
  );
  assert.equal(cases.reverse.newEdges.length, 1);
  assert.equal(cases.reverse.newEdges[0].sourceViewId, 1);
  assert.equal(cases.reverse.newEdges[0].viewId, 0);
  assert.equal(cases.reverse.newEdges[0].corroboration.viewId, 2);
  assert.equal(cases.budget.newEdges.length, 1);
  assert.equal(cases.budget.newEdges[0].sourceFeatureId, 8);
  assert.ok(cases.budget.diagnostics.rejected.multiViewUnavailable >= 8);
}
