// The browser and the TypeShade CPU interpreter exercise the same resident kernels.
export async function runDirectionCases(api, device, fixtures) {
  const unpack = (f) => ({
    images: f.images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    pairs: f.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  });
  const pairs = [];
  for (const f of fixtures.pairCases) {
    const r = await api.reconstructTwoViewGpu(
      device,
      Float64Array.from(f.pixels),
      f.first,
      f.second,
      f.options,
    );
    pairs.push({
      R: [...r.R],
      t: [...r.t],
      points: r.positiveDepthCount,
      directional: r.diagnostics.directionCompatibleCount,
      medianAngleDeg: r.diagnostics.medianParallaxDeg,
      falseFinite: r.validPoints.slice(60).reduce((n, x) => n + x, 0),
      directionConsensus: r.inliers.slice(60).reduce((n, x) => n + x, 0),
      unresolvedAreNaN: r.points.slice(60 * 3).every(Number.isNaN),
      cpuStages: r.execution.cpuStages,
      uploadBytes: r.execution.applicationUploadBytes,
      readbackBytes: r.execution.applicationReadbackBytes,
    });
  }
  const controls = {
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
      separateDirectionalObservations: true,
    },
    mixed = unpack(fixtures.mixed),
    map = await api.reconstructIncrementalGpu(device, mixed.images, mixed.pairs, controls),
    pure = unpack(fixtures.pure);
  let failure;
  try {
    await api.reconstructIncrementalGpu(device, pure.images, pure.pairs, controls);
  } catch (e) {
    failure = {
      name: e.name,
      pending: e.tracking?.untriangulatedTrackIds.length,
      fabricatedMap: 'reconstruction' in e,
    };
  }
  const far = unpack(fixtures.far),
    options = { ...controls, initialPair: [0, 1] },
    prefix = await api.reconstructIncrementalGpu(
      device,
      far.images.slice(0, 2),
      far.pairs.filter((p) => p.second < 2),
      options,
    ),
    full = await api.reconstructIncrementalGpu(device, far.images, far.pairs, options);
  return {
    pairs,
    mixed: {
      views: map.reconstruction.views.length,
      points: map.reconstruction.landmarks.length,
      pending: map.tracking.untriangulatedTrackIds.length,
      falseFinite: map.reconstruction.landmarks.filter((p) =>
        p.observations.some((o) => o.featureId >= 60),
      ).length,
    },
    pure: failure,
    far: {
      prefixPoints: prefix.reconstruction.landmarks.length,
      prefixPending: prefix.tracking.untriangulatedTrackIds.length,
      points: full.reconstruction.landmarks.length,
      pending: full.tracking.untriangulatedTrackIds.length,
    },
    liveBuffers: device.statistics.liveBuffers,
    statistics: device.statistics,
  };
}
