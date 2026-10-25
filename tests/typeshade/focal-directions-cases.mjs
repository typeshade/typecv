/** Geometry-only input: keypoints, putative identities and an incorrect focal guess; no truth. */
export async function runFocalDirectionCase(api, device, input) {
  const images = input.images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    pairs = input.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
    before = device.statistics,
    started = performance.now(),
    r = await api.reconstructIncrementalGpu(device, images, pairs, {
      separateDirectionalObservations: true,
      focalSearch: true,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
    }),
    after = device.statistics;
  return {
    elapsedMs: performance.now() - started,
    initialPair: r.diagnostics.initialPair,
    model: r.reconstruction,
    pending: r.tracking.untriangulatedTrackIds.length,
    focalSearch: r.diagnostics.focalSearch,
    cpuStages: r.execution.cpuStages,
    stageTimes: r.execution.stages.reduce((out, s) => {
      out[s.operation] = (out[s.operation] ?? 0) + s.elapsedMs;
      return out;
    }, {}),
    uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
    readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
    readbacks: after.reads - before.reads,
    readWaves: after.readWaves - before.readWaves,
    liveBuffers: after.liveBuffers,
    liveBytes: after.liveBytes,
    backend: device.backend,
    device: device.info,
  };
}
