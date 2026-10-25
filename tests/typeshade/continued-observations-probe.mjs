import { ShadeDevice } from '../../src/gpu/device.ts';
import {
  evaluateObservationsGpu,
  evaluateContinuedObservationsGpu,
} from '../../src/gpu/observations.ts';
import { observationModel } from './indexed-observations-cases.mjs';

function measured(result) {
  return {
    masks: result.masks.map((mask) => [...mask]),
    depthRejected: result.depthRejected ? [...result.depthRejected] : null,
    meanBits: [...new Uint32Array(Float64Array.of(result.meanErrorPx).buffer)],
    inlierCount: result.inlierCount,
  };
}
export async function checkContinuedObservations(backend = 'cpu') {
  const device = await ShadeDevice.open({ backend, fp64Flavor: 'float', poolBytes: 0 });
  try {
    const model = observationModel({ points: 32, views: 12 });
    for (const [i, point] of model.landmarks.entries())
      if (i % 7 === 0) point.observations.reverse();
    model.landmarks[0].observations.at(-1).pixel[0] += 40;
    const before = JSON.stringify(model);
    const all = await evaluateObservationsGpu(device, model);
    const continued = await evaluateContinuedObservationsGpu(device, model);
    const guardedAll = await evaluateObservationsGpu(device, model, 3, {
      maxRelativeDepthStdDev: 0.1,
    });
    const guarded = await evaluateContinuedObservationsGpu(device, model, 3, {
      maxRelativeDepthStdDev: 0.1,
    });
    const weak = observationModel({ points: 3, views: 3, depth: 2000, baseline: 0.0001 });
    const weakAll = await evaluateObservationsGpu(device, weak, 3, {
      maxRelativeDepthStdDev: 0.01,
    });
    const weakContinued = await evaluateContinuedObservationsGpu(device, weak, 3, {
      maxRelativeDepthStdDev: 0.01,
    });
    const invalid = [];
    for (const kind of [
      'old-pixel',
      'old-pixel-hole',
      'old-camera',
      'old-pose',
      'duplicate-old-feature',
    ]) {
      const input = observationModel({ points: 2, views: 4 });
      if (kind === 'old-pixel') input.landmarks[0].observations[0].pixel[0] = 1e-35;
      if (kind === 'old-pixel-hole') input.landmarks[0].observations[0].pixel = Array(2);
      if (kind === 'old-camera') input.views[0].camera.fx = 1e8;
      if (kind === 'old-pose') input.views[0].pose.R[1] = 1e-35;
      if (kind === 'duplicate-old-feature') input.landmarks[1].observations[0].featureId = 0;
      const failures = [];
      for (const evaluate of [evaluateObservationsGpu, evaluateContinuedObservationsGpu]) {
        try {
          await evaluate(device, input);
          failures.push('unexpected success');
        } catch (error) {
          failures.push(error.name + ': ' + error.message);
        }
      }
      invalid.push({ kind, failures });
    }
    const snapshot = observationModel({ points: 3, views: 4 });
    const reference = measured(await evaluateContinuedObservationsGpu(device, snapshot));
    const pending = evaluateContinuedObservationsGpu(device, snapshot);
    snapshot.views.forEach((v) => {
      v.pose.R.fill(NaN);
      v.pose.t.fill(NaN);
      v.camera.fx = 1e20;
    });
    snapshot.landmarks.forEach((p) => {
      p.xyz.fill(NaN);
      p.observations.forEach((o) => o.pixel.fill(NaN));
    });
    return {
      all: measured(all),
      continued: measured(continued),
      guardedAll: measured(guardedAll),
      guarded: measured(guarded),
      weakAll: measured(weakAll),
      weakContinued: measured(weakContinued),
      measurements: [
        all.masks.reduce((sum, mask) => sum + mask.length, 0),
        continued.masks.reduce((sum, mask) => sum + mask.length, 0),
      ],
      uploads: [all.execution.applicationUploadBytes, continued.execution.applicationUploadBytes],
      readbacks: [
        all.execution.applicationReadbackBytes,
        continued.execution.applicationReadbackBytes,
      ],
      guardedUploads: [
        guardedAll.execution.applicationUploadBytes,
        guarded.execution.applicationUploadBytes,
      ],
      inputUnchanged: before === JSON.stringify(model),
      invalid,
      reference,
      snapshotted: measured(await pending),
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
