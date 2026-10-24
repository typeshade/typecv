import test from 'node:test';
import assert from 'node:assert/strict';
// @ts-expect-error Node-only benchmark helper is a native ES module with no declaration output.
import * as evaluation from '../scripts/observation-expansion-evaluation.mjs';
const { evaluateExpansionCohorts, landmarkCohorts } = evaluation;

const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const view = (id: number, x: number) => ({
  id,
  camera: { fx: 10, fy: 10, cx: 1.5, cy: 1.5 },
  pose: { R: [...identity], t: [-x, 0, 0] },
});
const observation = (viewId: number, featureId: number) => ({ viewId, featureId, pixel: [1, 1] });
const point = (id: number, z: number) => ({
  trackId: id,
  xyz: [0, 0, z],
  observations: [observation(0, id), observation(1, id)],
});
const report = (model: unknown, initialPair = [0, 1]) => ({ model, initialPair });

test('depth cohorts compare retained geometry at the same original frame/pixel despite new weak points', () => {
  const a = point(0, 12),
    lost = point(2, 15),
    b = point(0, 11),
    fresh = point(9, 30);
  // Array order changes cannot select a different truth depth for the retained cohort.
  a.observations.reverse();
  b.observations.reverse();
  const before = { views: [view(0, 0), view(1, 1)], landmarks: [a, lost] },
    after = { views: [view(0, 0), view(1, 1), view(2, 7)], landmarks: [b, fresh] },
    truth = {
      views: [view(0, 0), view(1, 1), view(2, 2)],
      rendered: [
        { id: 0, width: 4, height: 4, depthZ: new Float32Array(16).fill(10) },
        { id: 1, width: 4, height: 4, depthZ: new Float32Array(16).fill(100) },
      ],
    },
    result = evaluateExpansionCohorts(report(before), report(after), truth);
  assert.deepEqual(result.ownership, {
    baseline: 2,
    expanded: 2,
    retainedOneToOne: 1,
    lostBaseline: 1,
    newWithoutBaselineObservation: 1,
    splitBaseline: 0,
    mergedExpanded: 0,
    baselineWithAmbiguousOwnership: 0,
  });
  assert.equal(result.depth.retained.before.median, 0.2);
  assert.equal(result.depth.retained.after.median, 0.1);
  assert.equal(result.depth.retained.better, 1);
  assert.equal(result.depth.newLandmarks.median, 2);
  assert.equal(result.depth.lostBaselineLandmarks.median, 0.5);
  assert.deepEqual(result.pose.sharedViewIds, [0, 1]);
  assert.deepEqual(result.pose.addedViewIds, [2]);
  assert.equal(
    result.pose.after.centerError.rmse,
    0,
    'newly registered erroneous camera does not alter the shared-frame population',
  );
});

test('cohort identity reports split/merged ownership rather than pairing by point proximity', () => {
  const before = { landmarks: [point(0, 10), point(1, 10), point(2, 10)] },
    after = {
      landmarks: [
        { ...point(10, 10), observations: [observation(0, 0), observation(2, 7)] },
        { ...point(11, 10), observations: [observation(1, 0), observation(3, 7)] },
        { ...point(12, 10), observations: [observation(0, 1), observation(1, 2)] },
      ],
    },
    result = landmarkCohorts(before, after);
  assert.deepEqual(result.counts, {
    baseline: 3,
    expanded: 3,
    retainedOneToOne: 0,
    lostBaseline: 0,
    newWithoutBaselineObservation: 0,
    splitBaseline: 1,
    mergedExpanded: 1,
    baselineWithAmbiguousOwnership: 3,
  });
  assert.deepEqual(result.retained, []);
});

test('reference pose comparison freezes one baseline whole-map alignment across a carried gauge', () => {
  const before = { views: [view(0, 0), view(1, 1), view(2, 2)], landmarks: [] },
    after = { views: [view(0, 0), view(1, 1), view(2, 3)], landmarks: [] },
    references = [0, 1, 2].map((i) => ({ id: i, ...view(i, i).pose })),
    result = evaluateExpansionCohorts(report(before), report(after), { references });
  assert.equal(result.pose.alignment, 'one baseline whole-map Sim(3), frozen for both models');
  assert.deepEqual(result.pose.alignmentFittedViewIds, [0, 1, 2]);
  assert.ok(result.pose.before.centerError.max < 1e-10);
  assert.ok(Math.abs(result.pose.after.centerError.max - 1) < 1e-10);
  assert.ok(Math.abs(result.pose.after.centerError.rmse - Math.sqrt(1 / 3)) < 1e-10);
});

test('rebuild with a different gauge is labelled separately, never as a frozen alignment', () => {
  const model = { views: [view(0, 0), view(1, 1), view(2, 2)], landmarks: [] },
    references = [0, 1, 2].map((i) => ({ id: i, ...view(i, i).pose })),
    result = evaluateExpansionCohorts(report(model), report(model, [1, 2]), { references });
  assert.match(result.pose.alignment, /^control with a different gauge/);
});

test('deferred models and unlabelled videos do not fabricate depth or pose accuracy', () => {
  assert.equal(evaluateExpansionCohorts({}, {}, {}).status, 'unavailable');
  const model = { views: [view(0, 0), view(1, 1)], landmarks: [point(0, 10)] },
    result = evaluateExpansionCohorts(report(model), report(model), { video: true });
  assert.equal(result.ownership.retainedOneToOne, 1);
  assert.ok(!('depth' in result));
  assert.equal(result.pose.before.status, 'unavailable');
  assert.equal(result.pose.after.status, 'unavailable');
});

test('frozen whole-map reference alignment handles a rotated, translated and scaled input gauge', () => {
  const Q = [0, -1, 0, 1, 0, 0, 0, 0, 1],
    scale = 2,
    translation = [3, -2, 1];
  const sourcePose = (id: number, c: number[]) => {
    const p = c.map((x, i) => (x - translation[i]) / scale),
      C = [p[1], -p[0], p[2]], // Q transpose applied to (C_reference - t) / s.
      t = [-C[1], C[0], C[2]].map((x) => -x);
    return { ...view(id, 0), pose: { R: [...Q], t } };
  };
  const positions = [
      [0, 0, 0],
      [1, 0, 0],
      [2, 1, 0],
    ],
    references = positions.map((c, id) => ({ id, R: identity, t: c.map((x) => -x) })),
    before = { views: positions.map((c, id) => sourcePose(id, c)), landmarks: [] },
    after = { views: [...before.views.slice(0, 2), sourcePose(2, [2.4, 0.8, 0])], landmarks: [] },
    result = evaluateExpansionCohorts(report(before), report(after), { references });
  assert.ok(result.pose.before.centerError.max < 1e-9);
  assert.ok(result.pose.before.rotationErrorDeg.max < 1e-5);
  assert.ok(Math.abs(result.pose.after.centerError.max - Math.hypot(0.4, -0.2)) < 1e-9);
});

test('second seed pose can rotate and move on its unit baseline sphere without changing the gauge', () => {
  const theta = 0.4,
    cosine = Math.cos(theta),
    sine = Math.sin(theta),
    R = [cosine, -sine, 0, sine, cosine, 0, 0, 0, 1],
    C = [cosine, sine, 0],
    t = [-(R[0] * C[0] + R[1] * C[1]), -(R[3] * C[0] + R[4] * C[1]), 0],
    before = { views: [view(0, 0), view(1, 1), view(2, 2)], landmarks: [] },
    after = {
      views: [before.views[0], { ...view(1, 0), pose: { R, t } }, before.views[2]],
      landmarks: [],
    },
    references = before.views.map((v) => ({ id: v.id, ...v.pose })),
    result = evaluateExpansionCohorts(report(before), report(after), { references });
  assert.equal(result.pose.alignment, 'one baseline whole-map Sim(3), frozen for both models');
  const changed = result.pose.after.perView.find((v: { id: number }) => v.id === 1);
  assert.ok(Math.abs(changed.centerError - Math.hypot(cosine - 1, sine)) < 1e-10);
  assert.ok(Math.abs(changed.rotationErrorDeg - (theta * 180) / Math.PI) < 1e-8);
});

test('an actual baseline scale change is a different gauge despite unchanged first pose and seed IDs', () => {
  const before = { views: [view(0, 0), view(1, 1), view(2, 2)], landmarks: [] },
    after = { views: [view(0, 0), view(1, 2), view(2, 4)], landmarks: [] },
    references = before.views.map((v) => ({ id: v.id, ...v.pose })),
    result = evaluateExpansionCohorts(report(before), report(after), { references });
  assert.match(result.pose.alignment, /^control with a different gauge/);
});
