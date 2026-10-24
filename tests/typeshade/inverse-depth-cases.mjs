export function inverseDepthScene(count = 4) {
  const camera = { fx: 800, fy: 760, cx: 320, cy: 240, k1: -0.03 },
    views = Array.from({ length: count }, (_, i) => {
      const y = i * 0.07,
        c = Math.cos(y),
        s = Math.sin(y),
        R = Float64Array.of(c, 0, s, 0, 1, 0, -s, 0, c),
        C = [i * 0.32, Math.sin(i) * 0.04, 0.02 * i * i];
      return {
        id: i,
        camera,
        pose: {
          R,
          t: Float64Array.from(
            [0, 1, 2],
            (r) => -R[3 * r] * C[0] - R[3 * r + 1] * C[1] - R[3 * r + 2] * C[2],
          ),
        },
      };
    }),
    points = Array.from({ length: 25 }, (_, i) => [
      Math.sin(i * 1.3),
      Math.cos(i * 1.7) * 0.8,
      5 + (i % 7) * 0.4,
    ]);
  const project = (v, X) => {
    const q = [0, 1, 2].map(
        (r) =>
          v.pose.R[3 * r] * X[0] +
          v.pose.R[3 * r + 1] * X[1] +
          v.pose.R[3 * r + 2] * X[2] +
          v.pose.t[r],
      ),
      x = q[0] / q[2],
      y = q[1] / q[2],
      r2 = x * x + y * y,
      d = 1 + camera.k1 * r2;
    return [camera.fx * x * d + camera.cx, camera.fy * y * d + camera.cy];
  };
  return {
    views,
    landmarks: points.map((xyz, trackId) => ({
      trackId,
      xyz: Float64Array.from(xyz),
      observations: views.map((v) => ({
        viewId: v.id,
        featureId: trackId,
        pixel: project(v, xyz),
      })),
    })),
  };
}

export async function inverseDepthGpuCases(api, backend) {
  const model = inverseDepthScene(),
    device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const point = api.initializeInverseDepth(
        { ...model.views[0], pixel: model.landmarks[0].observations[0].pixel },
        model.landmarks[0].xyz,
      ),
      measurements = [];
    for (const rho of [0, 1e-9, 0.2, -0.01])
      for (const view of model.views)
        measurements.push({
          view,
          point: {
            ...point,
            parameters: Float64Array.of(point.parameters[0], point.parameters[1], rho),
          },
        });
    const rows = await api.linearizeInverseDepthGpu(device, { measurements });
    let maxRelativeError = 0;
    measurements.forEach((o, i) => {
      const cpu = api.inverseDepthJacobian(o.view.camera, o.view.pose, o.point),
        gpu = rows[i];
      if (!!cpu !== !!gpu) throw new Error('Inverse-depth validity disagreement');
      if (!cpu) return;
      for (const key of ['pixel', 'point', 'pose', 'focal'])
        for (let j = 0; j < cpu[key].length; j++) {
          const e = Math.abs(cpu[key][j] - gpu[key][j]) / Math.max(1, Math.abs(cpu[key][j]));
          maxRelativeError = Math.max(e, maxRelativeError);
          if (e > 1e-8) throw new Error('Inverse-depth device Jacobian mismatch: ' + key + ' ' + e);
        }
    });
    const cpu = api.bundleAdjustInverseDepth(model, [], { iterations: 1 }),
      gpu = await api.bundleAdjustInverseDepthGpu(device, model, [], { iterations: 1 });
    if (!cpu.uncertainty.observable || !gpu.uncertainty.observable)
      throw new Error('Joint device information is unobservable');
    let covarianceRelativeError = 0;
    cpu.estimates.forEach((p, i) => {
      const e = Math.abs(p.covariance[8] - gpu.estimates[i].covariance[8]) / p.covariance[8];
      covarianceRelativeError = Math.max(covarianceRelativeError, e);
      if (e > 1e-6) throw new Error('Joint covariance differs across backends: ' + e);
    });
    if (device.statistics.liveBytes !== 0) throw new Error('Inverse-depth GPU arrays leaked');
    return {
      backend: device.backend,
      measurements: rows.length,
      maxRelativeError,
      covarianceRelativeError,
      liveBytes: device.statistics.liveBytes,
      execution: gpu.diagnostics.backend,
    };
  } finally {
    await device.dispose();
  }
}

export async function inverseDepthSessionCase(
  api,
  backend,
  Tracker = api.InverseDepthTracker,
  count = 4,
) {
  const model = inverseDepthScene(count),
    tracker = new Tracker({ pixelNoiseStdDevPx: 0.25, iterations: 2 }),
    images = model.views.map((v) => ({
      id: v.id,
      camera: v.camera,
      keypoints: Float64Array.from(
        model.landmarks.flatMap((p) => p.observations.find((o) => o.viewId === v.id).pixel),
      ),
    })),
    device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const session = new api.IncrementalSfmSession(device, {
      depthTracking: tracker,
      windowSize: 8,
      ransacIterations: 64,
      bundleIterations: 1,
      finalBundleIterations: 1,
    });
    let output;
    for (const image of images) {
      const pairs = images
        .filter((v) => session.activeViewIds.includes(v.id))
        .map((v) => ({
          first: v.id,
          second: image.id,
          matches: Uint32Array.from(model.landmarks.flatMap((_, i) => [i, i])),
        }));
      output = await session.add(image, pairs);
    }
    output = await session.finish();
    if (!output.inverseDepth || output.inverseDepth.diagnostics.finite < 15)
      throw new Error('Session did not publish stable depth');
    if (output.inverseDepth.diagnostics.uniqueObservations !== 25 * count)
      throw new Error('Session duplicated raw observations');
    if (!output.execution.cpuStages.some((s) => s.includes('host f64 Schur')))
      throw new Error('Hybrid execution was not disclosed');
    if (device.statistics.liveBytes) throw new Error('Depth session arrays leaked');
    return {
      registered: output.reconstruction.views.length,
      finite: output.reconstruction.landmarks.length,
      rawObservations: output.inverseDepth.diagnostics.uniqueObservations,
      backend: output.inverseDepth.diagnostics.backend,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
