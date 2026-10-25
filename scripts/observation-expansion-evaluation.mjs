// Evaluation only. This module runs in Node after reconstruction; labels never enter tracking.
import { multiply, svd3, transpose } from '../src/sfm/linear.ts';

const observationKey = (o) => `${o.viewId}:${o.featureId}`;
const center = (pose) =>
  [0, 1, 2].map(
    (i) => -pose.R[i] * pose.t[0] - pose.R[3 + i] * pose.t[1] - pose.R[6 + i] * pose.t[2],
  );
const apply = (R, p) =>
  [0, 1, 2].map((i) => R[3 * i] * p[0] + R[3 * i + 1] * p[1] + R[3 * i + 2] * p[2]);
const orderedObservations = (point) =>
  [...point.observations].sort((a, b) => a.viewId - b.viewId || a.featureId - b.featureId);
const angle = (a, b) =>
  (Math.acos(Math.max(-1, Math.min(1, (a.reduce((n, x, i) => n + x * b[i], 0) - 1) / 2))) * 180) /
  Math.PI;
const summary = (values) => {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  return {
    count: sorted.length,
    invalidCount: values.length - sorted.length,
    median: sorted[Math.floor(sorted.length * 0.5)] ?? null,
    p90: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))] ?? null,
    rmse: sorted.length ? Math.sqrt(sorted.reduce((n, x) => n + x * x, 0) / sorted.length) : null,
    max: sorted.at(-1) ?? null,
  };
};
const depthSummary = (values) => {
  const stats = summary(values);
  return {
    ...stats,
    fractionAbove10Percent: stats.count
      ? values.filter((x) => Number.isFinite(x) && x > 0.1).length / stats.count
      : null,
  };
};

/** One correspondence between observations is sufficient for ownership, but ambiguous splits/merges are withheld. */
export function landmarkCohorts(before, after) {
  const owners = new Map();
  before.landmarks.forEach((p, i) =>
    p.observations.forEach((o) => {
      const key = observationKey(o);
      if (owners.has(key)) throw new Error('Baseline evaluation has multiply owned observations');
      owners.set(key, i);
    }),
  );
  const beforeLinks = before.landmarks.map(() => new Set());
  const afterLinks = after.landmarks.map((p, j) => {
    const linked = new Set();
    for (const o of p.observations) {
      const i = owners.get(observationKey(o));
      if (i !== undefined) {
        linked.add(i);
        beforeLinks[i].add(j);
      }
    }
    return linked;
  });
  const retained = [],
    lost = [],
    fresh = [];
  beforeLinks.forEach((linked, i) => {
    if (!linked.size) lost.push(i);
    else if (linked.size === 1) {
      const j = [...linked][0];
      if (afterLinks[j].size === 1) retained.push([i, j]);
    }
  });
  afterLinks.forEach((linked, i) => {
    if (!linked.size) fresh.push(i);
  });
  return {
    retained,
    lost,
    fresh,
    counts: {
      baseline: before.landmarks.length,
      expanded: after.landmarks.length,
      retainedOneToOne: retained.length,
      lostBaseline: lost.length,
      newWithoutBaselineObservation: fresh.length,
      splitBaseline: beforeLinks.filter((s) => s.size > 1).length,
      mergedExpanded: afterLinks.filter((s) => s.size > 1).length,
      baselineWithAmbiguousOwnership: before.landmarks.length - retained.length - lost.length,
    },
  };
}

function seedGauge(views, pair) {
  const first = views.find((v) => v.id === pair?.[0]),
    second = views.find((v) => v.id === pair?.[1]);
  if (!first || !second) return undefined;
  const C = center(first.pose),
    B = center(second.pose),
    scale = Math.hypot(...B.map((x, i) => x - C[i]));
  if (!(scale > 1e-12)) return undefined;
  const rotation = transpose(first.pose.R, 3);
  return {
    rotation,
    scale,
    translation: C,
    world: (point) => apply(rotation, point).map((x, i) => C[i] + scale * x),
  };
}
function depthError(truth, world, point, observation) {
  const view = truth.views?.find((v) => v.id === observation.viewId),
    frame = truth.rendered?.find((v) => v.id === observation.viewId);
  if (!view || !frame || !world) return undefined;
  const x = Math.round(observation.pixel[0]),
    y = Math.round(observation.pixel[1]);
  if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) return undefined;
  const expected = frame.depthZ[y * frame.width + x];
  if (!(expected > 0 && Number.isFinite(expected))) return undefined;
  const p = world(point.xyz),
    R = view.pose.R,
    t = view.pose.t,
    z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
  return Math.abs(z - expected) / expected;
}
function depthCohorts(before, after, truth, cohorts, firstGauge, secondGauge) {
  const oldErrors = [],
    newErrors = [],
    freshErrors = [],
    lostErrors = [];
  let missingReference = 0,
    better = 0,
    worse = 0,
    equal = 0;
  for (const [i, j] of cohorts.retained) {
    const a = before.landmarks[i],
      b = after.landmarks[j],
      original = orderedObservations(a)[0],
      oldError = depthError(truth, firstGauge?.world, a, original),
      newError = depthError(truth, secondGauge?.world, b, original);
    if (oldError === undefined || newError === undefined) {
      missingReference++;
      continue;
    }
    oldErrors.push(oldError);
    newErrors.push(newError);
    if (Math.abs(oldError - newError) <= 1e-12) equal++;
    else if (newError < oldError) better++;
    else worse++;
  }
  for (const i of cohorts.fresh) {
    const point = after.landmarks[i],
      error = depthError(truth, secondGauge?.world, point, orderedObservations(point)[0]);
    if (error !== undefined) freshErrors.push(error);
  }
  for (const i of cohorts.lost) {
    const point = before.landmarks[i],
      error = depthError(truth, firstGauge?.world, point, orderedObservations(point)[0]);
    if (error !== undefined) lostErrors.push(error);
  }
  return {
    method:
      'camera +Z relative error; both retained estimates use the same first baseline observation ordered by viewId/featureId; one fixed seed gauge per map',
    reference: 'nearest rendered pixel depth; quantization and boundaries included',
    retained: {
      before: depthSummary(oldErrors),
      after: depthSummary(newErrors),
      better,
      worse,
      equal,
      unavailableReference: missingReference,
    },
    newLandmarks: {
      ...depthSummary(freshErrors),
      unavailableReference: cohorts.fresh.length - freshErrors.length,
    },
    lostBaselineLandmarks: {
      ...depthSummary(lostErrors),
      unavailableReference: cohorts.lost.length - lostErrors.length,
    },
  };
}

/** Fit the baseline once to all of its reference views, then freeze that Sim(3) for a carried map. */
function referenceGauge(model, references) {
  const refs = new Map(references.map((v) => [v.id, v])),
    views = model.views.filter((v) => refs.has(v.id));
  if (views.length < 3) return undefined;
  const sum = new Float64Array(9);
  for (const v of views) {
    const term = multiply(transpose(refs.get(v.id).R, 3), v.pose.R, 3);
    for (let i = 0; i < 9; i++) sum[i] += term[i];
  }
  let rotation;
  try {
    const { U, Vt } = svd3(sum);
    rotation = multiply(U, Vt, 3);
  } catch {
    return undefined;
  }
  const xs = views.map((v) => apply(rotation, center(v.pose))),
    ys = views.map((v) => center(refs.get(v.id))),
    mean = (rows) => [0, 1, 2].map((i) => rows.reduce((n, p) => n + p[i], 0) / rows.length),
    mx = mean(xs),
    my = mean(ys);
  let variance = 0,
    covariance = 0;
  for (let j = 0; j < views.length; j++)
    for (let i = 0; i < 3; i++) {
      variance += (xs[j][i] - mx[i]) ** 2;
      covariance += (xs[j][i] - mx[i]) * (ys[j][i] - my[i]);
    }
  const scale = covariance / variance,
    translation = my.map((x, i) => x - scale * mx[i]),
    radius = Math.max(...ys.map((p) => Math.hypot(...p.map((x, i) => x - my[i]))));
  if (!(scale > 0 && Number.isFinite(scale) && radius > 1e-12)) return undefined;
  return {
    rotation,
    scale,
    translation,
    radius,
    fittedViewIds: views.map((v) => v.id),
    world: (p) => apply(rotation, p).map((x, i) => scale * x + translation[i]),
  };
}
function sameGauge(before, after, firstPair, secondPair) {
  if (firstPair?.length !== 2 || !firstPair.every((id, i) => id === secondPair?.[i])) return false;
  const firstBefore = before.views.find((v) => v.id === firstPair[0])?.pose,
    secondBefore = before.views.find((v) => v.id === firstPair[1])?.pose,
    firstAfter = after.views.find((v) => v.id === firstPair[0])?.pose,
    secondAfter = after.views.find((v) => v.id === firstPair[1])?.pose;
  if (!firstBefore || !secondBefore || !firstAfter || !secondAfter) return false;
  if (
    !['R', 't'].every((field) =>
      firstBefore[field].every((x, k) => Math.abs(x - firstAfter[field][k]) <= 1e-8),
    )
  )
    return false;
  // BA holds the first pose and baseline magnitude, not the second pose or
  // baseline direction. Those observable quantities may legitimately improve.
  const baseline = (first, second) => {
      const a = center(first),
        b = center(second);
      return Math.hypot(...a.map((x, i) => x - b[i]));
    },
    oldLength = baseline(firstBefore, secondBefore),
    newLength = baseline(firstAfter, secondAfter);
  return (
    oldLength > 0 &&
    newLength > 0 &&
    Number.isFinite(oldLength) &&
    Number.isFinite(newLength) &&
    Math.abs(oldLength - newLength) <= 1e-8 * Math.max(oldLength, newLength)
  );
}
function poseCohort(before, after, truth, firstPair, secondPair, firstSeed, secondSeed) {
  const aIds = new Set(before.views.map((v) => v.id)),
    bIds = new Set(after.views.map((v) => v.id)),
    sharedViewIds = [...aIds].filter((id) => bIds.has(id)).sort((a, b) => a - b),
    addedViewIds = [...bIds].filter((id) => !aIds.has(id)).sort((a, b) => a - b),
    lostViewIds = [...aIds].filter((id) => !bIds.has(id)).sort((a, b) => a - b);
  let first = firstSeed,
    second = secondSeed,
    reference;
  let alignment = 'one fixed truth seed gauge per map; no per-frame alignment';
  if (truth.references) {
    reference = new Map(truth.references.map((v) => [v.id, v]));
    first = referenceGauge(before, truth.references);
    const frozen = sameGauge(before, after, firstPair, secondPair);
    second = frozen ? first : referenceGauge(after, truth.references);
    alignment = frozen
      ? 'one baseline whole-map Sim(3), frozen for both models'
      : 'control with a different gauge: one whole-map Sim(3) fit per model before restricting to shared frames';
  } else if (truth.views) reference = new Map(truth.views.map((v) => [v.id, v.pose]));
  const evaluate = (model, gauge) => {
    if (!gauge || !reference) return { status: 'unavailable', reason: 'No valid reference gauge' };
    const errors = model.views
      .filter((v) => sharedViewIds.includes(v.id) && reference.has(v.id))
      .map((v) => {
        const expected = reference.get(v.id),
          predictedR = multiply(v.pose.R, transpose(gauge.rotation, 3), 3),
          target = center(expected),
          predicted = gauge.world(center(v.pose)),
          distance = Math.hypot(...predicted.map((x, i) => x - target[i]));
        return {
          id: v.id,
          rotationErrorDeg: angle(predictedR, expected.R),
          centerError: distance,
          ...(gauge.radius ? { centerErrorOfReferenceRadius: distance / gauge.radius } : {}),
        };
      });
    return {
      status: 'evaluated',
      rotationErrorDeg: summary(errors.map((e) => e.rotationErrorDeg)),
      centerError: summary(errors.map((e) => e.centerError)),
      ...(gauge.radius
        ? {
            centerErrorOfReferenceRadius: summary(
              errors.map((e) => e.centerErrorOfReferenceRadius),
            ),
          }
        : {}),
      perView: errors,
    };
  };
  return {
    sharedViewIds,
    addedViewIds,
    lostViewIds,
    alignment,
    before: evaluate(before, first),
    after: evaluate(after, second),
    ...(first?.fittedViewIds ? { alignmentFittedViewIds: first.fittedViewIds } : {}),
  };
}

/** Compare ownership cohorts before deleting private model geometry from the benchmark report. */
export function evaluateExpansionCohorts(beforeReport, afterReport, truth) {
  const before = beforeReport.model,
    after = afterReport.model;
  if (!before || !after) return { status: 'unavailable', reason: 'No finite model on both sides' };
  const cohorts = landmarkCohorts(before, after),
    first = truth.views && seedGauge(truth.views, beforeReport.initialPair),
    second = truth.views && seedGauge(truth.views, afterReport.initialPair);
  return {
    status: 'evaluated',
    ownership: cohorts.counts,
    ...(truth.rendered
      ? { depth: depthCohorts(before, after, truth, cohorts, first, second) }
      : {}),
    pose: poseCohort(
      before,
      after,
      truth,
      beforeReport.initialPair,
      afterReport.initialPair,
      first,
      second,
    ),
    limitation:
      'observation ownership establishes evaluation correspondence, not global physical identity; no truth positions enter reconstruction',
  };
}
