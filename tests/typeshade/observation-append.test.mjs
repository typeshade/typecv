import test from 'node:test';
import assert from 'node:assert/strict';
import { identityPose, projectPoint } from '../../src/geometry/pose.ts';
import { appendExpandedObservations } from '../../src/gpu/observation-append.ts';
function fixture() {
  const camera = { fx: 100, fy: 100, cx: 50, cy: 50 },
    translations = [
      [0, 0, 0],
      [-1, 0, 0],
      [-2, 0, 0],
      [0, -1, 0],
      [0, 0, -6],
    ],
    views = translations.map((t, id) => ({
      id,
      camera: { ...camera },
      pose: { ...identityPose(), t: Float64Array.from(t) },
    })),
    xyz = [Float64Array.of(0, 0, 5), Float64Array.of(0.02, 0, 5), Float64Array.of(3, 1, 5)],
    pixel = (viewId, point) => projectPoint(views[viewId].camera, views[viewId].pose, xyz[point]),
    model = {
      views,
      landmarks: xyz.map((point, i) => ({
        trackId: [11, 22, 33][i],
        xyz: point,
        observations: [0, 1].map((viewId) => ({ viewId, featureId: i, pixel: pixel(viewId, i) })),
      })),
    };
  model.landmarks[2].observations.push({
    viewId: 2,
    featureId: 1,
    pixel: pixel(2, 2),
  });
  const coordinates = [
      [...pixel(0, 0), ...pixel(0, 1), ...pixel(0, 2), 125, 115],
      [...pixel(1, 0), ...pixel(1, 1), ...pixel(1, 2)],
      [10.2, 50, ...pixel(2, 2)],
      [50, 30],
      [],
      [],
    ],
    originalImages = coordinates.map((keypoints, id) => ({
      id,
      camera: { ...camera },
      keypoints: Float64Array.from(keypoints),
    })),
    images = originalImages.map((image) => ({
      ...image,
      camera: { ...image.camera },
      keypoints: image.keypoints.slice(),
    }));
  images[2].keypoints = Float64Array.from([...images[2].keypoints, 10, 50, 10.1, 50]);
  images[3].keypoints = Float64Array.from([...images[3].keypoints, 50.1, 30]);
  images[4].keypoints = Float64Array.of(50, 50);
  images[5].keypoints = Float64Array.of(50, 50);
  return { model, originalImages, expansion: { images, newEdges: [] } };
}
function edge(f, targetView = 2, targetFeature = 2, point = 0, extra = {}) {
  const target = f.expansion.images.find((image) => image.id === targetView),
    original = f.originalImages.find((image) => image.id === targetView),
    targetPixel = [target.keypoints[2 * targetFeature], target.keypoints[2 * targetFeature + 1]];
  return {
    viewId: targetView,
    featureId: targetFeature,
    sourceViewId: 0,
    sourceFeatureId: point,
    measuredPixel: [...targetPixel],
    targetPixel,
    createdFeature: targetFeature >= original.keypoints.length / 2,
    ncc: 0.99,
    forwardBackwardErrorPx: 0.05,
    uniquenessMargin: 0.2,
    prediction: 'none',
    geometry: 'candidate',
    corroboration: {
      status: 'verified',
      viewId: 1,
      featureId: point,
      measuredPixel: [...targetPixel],
      disagreementPx: 0,
      ncc: 0.98,
      forwardBackwardErrorPx: 0.04,
    },
    ...extra,
  };
}
const apply = (f, maxErrorPx = 1.5) =>
  appendExpandedObservations(f.model, f.originalImages, f.expansion, { maxErrorPx });
function verifiedWitness(e) {
  const witness = e.corroboration;
  if (witness?.status !== 'verified') throw new Error('Fixture requires a verified witness');
  return witness;
}
const nonzero = (result) =>
  Object.fromEntries(Object.entries(result.diagnostics.rejected).filter(([, count]) => count > 0));
function assertPreserved(f, result) {
  assert.deepEqual(result.reconstruction.views, f.model.views);
  assert.equal(result.reconstruction.landmarks.length, f.model.landmarks.length);
  for (const [i, point] of f.model.landmarks.entries()) {
    const next = result.reconstruction.landmarks[i];
    assert.equal(next.trackId, point.trackId);
    assert.deepEqual(next.xyz, point.xyz);
    assert.deepEqual(next.observations.slice(0, point.observations.length), point.observations);
  }
  assert.equal(result.diagnostics.candidates, f.expansion.newEdges.length);
  assert.equal(result.diagnostics.appended, result.admitted.length);
  assert.equal(
    result.diagnostics.appended +
      Object.values(result.diagnostics.rejected).reduce((a, b) => a + b, 0),
    result.diagnostics.candidates,
  );
}
test('append adds canonical new/joined measurements while preserving every original value and ownership', () => {
  const f = fixture();
  f.expansion.newEdges = [edge(f), edge(f, 3, 0, 0, { measuredPixel: [50.25, 30] })];
  const before = structuredClone(f),
    result = apply(f);
  assert.deepEqual(result.admitted, [
    { edgeIndex: 0, trackId: 11, errorPx: 0 },
    { edgeIndex: 1, trackId: 11, errorPx: 0 },
  ]);
  assert.deepEqual(nonzero(result), {});
  assertPreserved(f, result);
  assert.deepEqual(result.reconstruction.landmarks[0].observations.slice(2), [
    { viewId: 2, featureId: 2, pixel: [10, 50] },
    { viewId: 3, featureId: 0, pixel: [50, 30] },
  ]);
  assert.deepEqual(f, before);
  result.reconstruction.views[0].pose.R[0] = 9;
  result.reconstruction.landmarks[0].xyz[0] = 9;
  result.reconstruction.landmarks[0].observations[0].pixel[0] = 9;
  result.reconstruction.landmarks[0].observations[2].pixel[0] = 9;
  assert.deepEqual(f, before, 'returned original/new arrays do not alias any input');
});
test('a canonical join fails the target-pixel gate even when its measured KLT endpoint fits', () => {
  const f = fixture();
  f.expansion.newEdges = [edge(f, 2, 0, 0, { measuredPixel: [10, 50] })];
  const result = apply(f, 0.1);
  assert.equal(result.diagnostics.appended, 0);
  assert.deepEqual(nonzero(result), { reprojection: 1 });
  assert.deepEqual(result.reconstruction, f.model);
  assertPreserved(f, result);
});
test('append projects through the frozen refined camera without requiring model K to equal detector K', () => {
  const f = fixture();
  f.model.views[3].camera.fy = 200;
  f.expansion.images[3].keypoints[2] = 50;
  f.expansion.images[3].keypoints[3] = 10;
  f.expansion.newEdges = [edge(f, 3, 1)];
  const result = apply(f, 0.01);
  assert.equal(result.diagnostics.appended, 1);
  assert.equal(result.admitted[0].errorPx, 0);
  assertPreserved(f, result);
});
test('well-formed unqualified evidence is deferred with distinct ownership and fixed-projection reasons', () => {
  const f = fixture();
  f.model.landmarks[0].observations.push({
    viewId: 3,
    featureId: 0,
    pixel: [50, 30],
  });
  f.expansion.images[1].keypoints = Float64Array.from([
    ...f.expansion.images[1].keypoints,
    30.1,
    50,
  ]);
  f.expansion.images[2].keypoints[6] = 14;
  const anchor3 = {
    ...verifiedWitness(edge(f)),
    status: 'verified',
    viewId: 3,
    featureId: 0,
  };
  f.expansion.newEdges = [
    edge(f, 2, 2, 0, { corroboration: { status: 'unavailable' } }),
    edge(f, 2, 2, 0, { corroboration: { ...anchor3, viewId: 0 } }),
    edge(f, 2, 2, 0, { sourceFeatureId: 3 }),
    edge(f, 2, 2, 0, { corroboration: { ...anchor3, viewId: 1, featureId: 1 } }),
    edge(f, 5, 0),
    edge(f, 2, 1),
    edge(f, 1, 0, 0, { corroboration: anchor3 }),
    edge(f, 1, 3, 0, { corroboration: anchor3 }),
    edge(f, 4, 0),
    edge(f, 2, 3),
  ];
  const result = apply(f);
  assert.deepEqual(nonzero(result), {
    unavailableWitness: 1,
    nonIndependentWitness: 1,
    sourceNotOriginal: 1,
    witnessNotOriginalPoint: 1,
    unregisteredTarget: 1,
    targetOwned: 1,
    alreadyObserved: 1,
    targetViewConflict: 1,
    nonpositiveProjection: 1,
    reprojection: 1,
  });
  assert.equal(result.diagnostics.appended, 0);
  assert.deepEqual(result.reconstruction, f.model);
  assertPreserved(f, result);
});
for (const role of ['source', 'witness'])
  for (const condition of ['behind', 'stale'])
    test(`append defers a ${condition} original ${role} even when the target fits`, () => {
      const f = fixture(),
        originalViewId = role === 'source' ? 0 : 1,
        view = f.model.views[originalViewId];
      if (condition === 'behind') view.pose.t[2] = -6;
      else view.pose.t[0] += 0.1;
      f.expansion.newEdges = [edge(f)];
      const point = f.model.landmarks[0],
        originalObservation = point.observations.find(
          (observation) => observation.viewId === originalViewId,
        ),
        originalProjection = projectPoint(view.camera, view.pose, point.xyz),
        target = f.model.views[2];
      if (condition === 'behind') assert.equal(originalProjection, undefined);
      else
        assert.ok(
          Math.hypot(
            originalProjection[0] - originalObservation.pixel[0],
            originalProjection[1] - originalObservation.pixel[1],
          ) > 1.5,
        );
      assert.deepEqual(
        projectPoint(target.camera, target.pose, point.xyz),
        f.expansion.newEdges[0].targetPixel,
      );
      const before = structuredClone(f),
        result = apply(f);
      assert.deepEqual(nonzero(result), { [`${role}Geometry`]: 1 });
      assert.equal(result.diagnostics.appended, 0);
      assert.deepEqual(result.admitted, []);
      assert.deepEqual(result.reconstruction, f.model);
      assert.deepEqual(
        f,
        before,
        'rejection preserves supplied poses, XYZ, observations and image evidence',
      );
      assertPreserved(f, result);
    });

test('an original-but-newly-owned join cannot become source or witness authority, in either order', () => {
  for (const reverse of [false, true]) {
    const f = fixture(),
      originalJoin = edge(f, 3, 0),
      sourceChain = edge(f, 2, 2, 0, { sourceViewId: 3, sourceFeatureId: 0 }),
      witnessChain = edge(f, 2, 2, 0, {
        corroboration: {
          ...verifiedWitness(originalJoin),
          status: 'verified',
          viewId: 3,
          featureId: 0,
        },
      });
    f.expansion.newEdges = reverse
      ? [witnessChain, sourceChain, originalJoin]
      : [originalJoin, sourceChain, witnessChain];
    const result = apply(f);
    assert.equal(result.diagnostics.appended, 1);
    assert.deepEqual(nonzero(result), { sourceNotOriginal: 1, witnessNotOriginalPoint: 1 });
    assert.equal(result.admitted[0].edgeIndex, reverse ? 2 : 0);
    assert.equal(result.reconstruction.landmarks[0].observations.at(-1).viewId, 3);
    assertPreserved(f, result);
  }
});
test('all eligible target-feature and target-view competitors defer independently of input order', () => {
  for (const mode of ['feature', 'view', 'graph'])
    for (const reverse of [false, true]) {
      const f = fixture(),
        claims =
          mode === 'feature'
            ? [edge(f, 2, 0), edge(f, 2, 0, 1)]
            : mode === 'view'
              ? [edge(f, 2, 2), edge(f, 2, 3)]
              : [edge(f, 2, 0), edge(f, 2, 2), edge(f, 2, 2, 1)];
      f.expansion.newEdges = reverse ? claims.reverse() : claims;
      const result = apply(f);
      assert.equal(result.diagnostics.appended, 0, mode);
      assert.deepEqual(
        nonzero(result),
        mode === 'feature'
          ? { conflictingTargetFeature: 2 }
          : mode === 'view'
            ? { conflictingTargetView: 2 }
            : { conflictingTargetView: 2, conflictingTargetFeature: 1 },
      );
      assert.deepEqual(result.reconstruction, f.model);
      assertPreserved(f, result);
    }
});
test('ineligible competitors cannot veto an eligible claim', () => {
  for (const reason of ['ownership', 'projection'])
    for (const reverse of [false, true]) {
      const f = fixture();
      f.originalImages[2].keypoints[0] = 10;
      f.expansion.images[2].keypoints[0] = 10;
      const good = edge(f, 2, 0),
        bad = edge(
          f,
          2,
          0,
          1,
          reason === 'ownership'
            ? {
                corroboration: {
                  ...verifiedWitness(good),
                  status: 'verified',
                  viewId: 1,
                  featureId: 0,
                },
              }
            : {},
        );
      f.expansion.newEdges = reverse ? [bad, good] : [good, bad];
      const result = apply(f, 0.1);
      assert.equal(result.diagnostics.appended, 1);
      assert.equal(result.admitted[0].trackId, 11);
      assert.deepEqual(
        nonzero(result),
        reason === 'ownership' ? { witnessNotOriginalPoint: 1 } : { reprojection: 1 },
      );
      assertPreserved(f, result);
    }
});
test('duplicate claims retain first provenance once and cannot resurrect a conflicting claim', () => {
  for (const competing of [false, true]) {
    const f = fixture(),
      first = edge(f),
      duplicate = edge(f, 2, 2, 0, {
        sourceViewId: 1,
        sourceFeatureId: 0,
        corroboration: { ...verifiedWitness(first), status: 'verified', viewId: 0, featureId: 0 },
      });
    f.expansion.newEdges = [first, duplicate, ...(competing ? [edge(f, 2, 2, 1)] : [])];
    const result = apply(f);
    assert.deepEqual(
      nonzero(result),
      competing ? { conflictingTargetFeature: 2, redundant: 1 } : { redundant: 1 },
    );
    assert.deepEqual(result.admitted, competing ? [] : [{ edgeIndex: 0, trackId: 11, errorPx: 0 }]);
    assertPreserved(f, result);
  }
});
test('append validates the complete original prefix and image identities, including unowned features', () => {
  const damages = [
    (f) => {
      f.expansion.images[0].keypoints[6] += 0.01;
    },
    (f) => {
      f.expansion.images[2].camera.fx += 1;
    },
    (f) => {
      f.expansion.images[0].keypoints = new Float64Array();
    },
    (f) => {
      f.expansion.images.pop();
    },
    (f) => {
      f.expansion.images[5].id = 4;
    },
    (f) => {
      f.originalImages[0].keypoints[0] += 1;
      f.expansion.images[0].keypoints[0] += 1;
    },
  ];
  for (const damage of damages) {
    const f = fixture();
    f.expansion.newEdges = [edge(f)];
    damage(f);
    const before = structuredClone(f);
    assert.throws(() => apply(f), RangeError);
    assert.deepEqual(f, before);
  }
});
test('malformed late rows throw before any append and preserve all supplied inputs', () => {
  const damages = [
    (e) => ({ ...e, featureId: -2 }),
    (e) => ({ ...e, featureId: 0.5 }),
    (e) => ({ ...e, featureId: 10000 }),
    (e) => ({ ...e, viewId: 99 }),
    (e) => ({ ...e, sourceFeatureId: -1 }),
    (e) => ({ ...e, sourceViewId: 99 }),
    (e) => ({ ...e, targetPixel: [10.1, 50] }),
    (e) => ({ ...e, targetPixel: [NaN, 50] }),
    (e) => ({ ...e, measuredPixel: [10, Infinity] }),
    (e) => ({ ...e, ncc: NaN }),
    (e) => ({ ...e, forwardBackwardErrorPx: -1 }),
    (e) => ({ ...e, uniquenessMargin: -1 }),
    (e) => ({ ...e, createdFeature: false }),
    (e) => ({ ...e, prediction: 'fabricated' }),
    (e) => ({ ...e, patchWarp: [1, 0, 0] }),
    (e) => ({ ...e, patchWarp: [1, 0, 0, NaN] }),
    (e) => ({ ...e, surfaceSupport: { points: 0, maxRelativeDepthResidual: 0 } }),
    (e) => ({ ...e, surfaceSupport: { points: 1.5, maxRelativeDepthResidual: 0 } }),
    (e) => ({ ...e, surfaceSupport: { points: 3, maxRelativeDepthResidual: -1 } }),
    (e) => ({
      ...e,
      corroboration: { ...verifiedWitness(e), status: 'verified', viewId: 1, featureId: 99 },
    }),
    (e) => ({
      ...e,
      corroboration: {
        ...verifiedWitness(e),
        status: 'verified',
        viewId: 1,
        featureId: 0,
        disagreementPx: NaN,
      },
    }),
    (e) => ({
      ...e,
      corroboration: {
        ...verifiedWitness(e),
        status: 'verified',
        viewId: 1,
        featureId: 0,
        measuredPixel: [10, NaN],
      },
    }),
  ];
  for (const damage of damages) {
    const f = fixture();
    f.expansion.newEdges = [edge(f, 3, 0), damage(edge(f))];
    const before = structuredClone(f);
    assert.throws(() => apply(f), RangeError);
    assert.deepEqual(f, before);
  }
  for (const authority of ['source', 'witness']) {
    const f = fixture(),
      first = edge(f);
    f.expansion.newEdges = [
      first,
      authority === 'source'
        ? edge(f, 3, 1, 0, { sourceViewId: 2, sourceFeatureId: 2 })
        : edge(f, 3, 1, 0, {
            corroboration: {
              ...verifiedWitness(first),
              status: 'verified',
              viewId: 2,
              featureId: 2,
            },
          }),
    ];
    assert.throws(() => apply(f), /invalid feature identity/);
  }
  for (const threshold of [0, -1, NaN, Infinity])
    assert.throws(() => apply(fixture(), threshold), RangeError);
});
