// Geometry-only validation of the existing mapper resume path, independent of RGB measurements.
export async function runRefinementCase(api, core, device, input) {
  const images = input.images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    limited = input.limited.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    pairs = input.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
    oldPairs = input.oldPairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
    initial = core.reconstructIncremental(limited, oldPairs, input.controls),
    snapshot = JSON.stringify(initial.reconstruction),
    result = await api.refineIncrementalGpu(device, images, pairs, initial, input.controls);
  const count = (m) => ({
    views: m.views.length,
    points: m.landmarks.length,
    observations: m.landmarks.reduce((n, p) => n + p.observations.length, 0),
  });
  return {
    before: count(initial.reconstruction),
    after: count(result.reconstruction),
    unchanged: JSON.stringify(initial.reconstruction) === snapshot,
    originalSeed: initial.diagnostics.initialPair,
    seed: result.diagnostics.initialPair,
    error: result.diagnostics.meanReprojectionErrorPx,
    execution: result.execution,
    liveBytes: device.statistics.liveBytes,
  };
}
