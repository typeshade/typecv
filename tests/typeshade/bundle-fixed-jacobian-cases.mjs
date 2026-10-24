import { bundleFixture } from './geometry-cases.mjs';

export function fixedJacobianCases({ count = 24, views = 6, iterations = 3 } = {}) {
  return ['all-fixed', 'local', 'all-fixed-shared', 'local-shared', 'global'].map((name) => {
    const local = name.startsWith('local');
    const model = bundleFixture({ count, views, outlier: local });
    const shared = name.endsWith('shared');
    if (shared)
      for (const view of model.views) {
        view.camera.fx *= 0.97;
        view.camera.fy *= 0.97;
      }
    const variableViewIds =
      name === 'global' ? undefined : local ? model.views.slice(-2).map((view) => view.id) : [];
    const variable = new Set(variableViewIds ?? model.views.slice(1).map((view) => view.id));
    const fixedObservations = model.landmarks.reduce(
      (sum, point) => sum + point.observations.filter((o) => !variable.has(o.viewId)).length,
      0,
    );
    return {
      name,
      model,
      variable,
      fixedObservations,
      options: {
        iterations,
        huberPx: 2,
        refineSharedFocal: shared,
        ...(variableViewIds === undefined ? {} : { variableViewIds }),
      },
    };
  });
}

function numbers(model) {
  return Float64Array.from([
    ...model.views.flatMap((view) => [
      view.id,
      ...view.pose.R,
      ...view.pose.t,
      view.camera.fx,
      view.camera.fy,
      view.camera.cx,
      view.camera.cy,
      view.camera.k1 ?? 0,
      view.camera.k2 ?? 0,
    ]),
    ...model.landmarks.flatMap((point) => [
      point.trackId,
      ...point.xyz,
      ...point.observations.flatMap((o) => [o.viewId, o.featureId, ...o.pixel]),
    ]),
  ]);
}

export async function fixedJacobianDigest(model, result) {
  const arrays = [numbers(model)];
  if (result)
    arrays.push(
      Float64Array.from([
        ...Object.values(result.diagnostics),
        result.cgIterations,
        result.linearSolverFailures,
      ]),
    );
  const bytes = new Uint8Array(arrays.reduce((sum, array) => sum + array.byteLength, 0));
  let at = 0;
  for (const array of arrays) {
    bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), at);
    at += array.byteLength;
  }
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join(
    '',
  );
}

export async function checkFixedJacobianCases(api, backend = 'cpu') {
  const device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  const rows = [];
  try {
    for (const fixture of fixedJacobianCases()) {
      const before = await fixedJacobianDigest(fixture.model);
      const result = await api.bundleAdjustGpu(device, fixture.model, fixture.options);
      const fixed = fixture.model.views.filter((view) => !fixture.variable.has(view.id));
      const posesPreserved = fixed.every((view) => {
        const moved = result.reconstruction.views.find((candidate) => candidate.id === view.id);
        return [...view.pose.R, ...view.pose.t].every((value, index) =>
          Object.is(value, [...moved.pose.R, ...moved.pose.t][index]),
        );
      });
      rows.push({
        name: fixture.name,
        before,
        after: await fixedJacobianDigest(fixture.model),
        output: await fixedJacobianDigest(result.reconstruction, result),
        fixedCameras: fixed.length,
        fixedObservations: fixture.fixedObservations,
        // A structural count per active linearization, not a sampled GPU performance counter.
        unusedPoseEntriesPerLinearization: 12 * fixture.fixedObservations,
        unusedPoseRowsPerLinearization: 2 * fixture.fixedObservations,
        posesPreserved,
        acceptedSteps: result.diagnostics.acceptedSteps,
        initialCost: result.diagnostics.initialCost,
        finalCost: result.diagnostics.finalCost,
        focalRatio: result.reconstruction.views[0].camera.fx / fixture.model.views[0].camera.fx,
        liveBytes: device.statistics.liveBytes,
      });
    }
    return rows;
  } finally {
    await device.dispose();
  }
}

export function assertFixedJacobianCases(assert, rows) {
  assert.equal(rows.length, 5);
  for (const row of rows) {
    assert.equal(row.before, row.after, row.name);
    assert.equal(row.posesPreserved, true, row.name);
    assert.equal(row.liveBytes, 0, row.name);
    assert.ok(row.acceptedSteps > 0, row.name);
    assert.ok(row.finalCost < row.initialCost, row.name);
    assert.ok(row.fixedObservations > 0, row.name);
    if (row.name.endsWith('shared')) assert.ok(Math.abs(row.focalRatio - 1) > 1e-6);
    else assert.equal(row.focalRatio, 1, row.name);
  }
}
