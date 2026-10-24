/** Shared browser/CPU-tier exercise; fixtures explicitly supply bearing hypotheses to BA. */
export async function runBundleDirectionCases(api, device, fixtures) {
  const rows = [],
    revive = (f) => ({
      ...f,
      input: {
        views: f.input.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: f.input.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
      },
      directions: f.directions.map((d) => ({ ...d, direction: Float64Array.from(d.direction) })),
    });
  for (const [i, f] of fixtures.entries()) {
    const { input, directions } = revive(f),
      before = JSON.stringify({ input, directions }),
      result = await api.bundleAdjustGpu(device, input, {
        directionalConstraints: directions,
        refineSharedFocal: i === 1,
        iterations: 40,
        maxCgIterations: 256,
        cgRelativeTolerance: 1e-8,
      });
    rows.push({
      diagnostics: result.diagnostics,
      finiteIds: result.reconstruction.landmarks.map((p) => p.trackId),
      directionalIds: result.directionalConstraints.map((d) => d.trackId),
      norms: result.directionalConstraints.map((d) => Math.hypot(...d.direction)),
      focals: result.reconstruction.views.map((v) => v.camera.fx),
      pointRmse: Math.sqrt(
        result.reconstruction.landmarks.reduce(
          (sum, p, j) =>
            sum + p.xyz.reduce((s, x, k) => s + (x - f.truth.landmarks[j].xyz[k]) ** 2, 0),
          0,
        ) / 60,
      ),
      rotationMaxDeg: Math.max(
        ...result.reconstruction.views.map((v, j) => {
          const trace = v.pose.R.reduce((sum, x, k) => sum + x * f.truth.views[j].pose.R[k], 0);
          return (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
        }),
      ),
      cpuStages: result.execution.cpuStages,
      execution: result.execution,
      unchanged: before === JSON.stringify({ input, directions }),
      liveBytes: device.statistics.liveBytes,
      liveBuffers: device.statistics.liveBuffers,
    });
  }
  return { rows, statistics: device.statistics, device: device.info, backend: device.backend };
}
