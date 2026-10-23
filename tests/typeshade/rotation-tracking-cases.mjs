// Shared CPU/WebGPU exercise. Only calibrated keypoints and supplied matches enter the mapper.
export async function runRotationTrackingCases(api, device, fixtures) {
  const unpack = (f) => ({
    images: f.images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    pairs: f.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  });
  const describe = (t) =>
    t && {
      rotations: t.rotations.map((p) => ({
        first: p.first,
        second: p.second,
        R: [...p.rotation.R],
        translation: p.rotation.translation,
        depth: p.rotation.depth,
      })),
      tracks: t.tracks.length,
      pending: t.untriangulatedTrackIds.length,
    };
  const controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 1 };
  const prefix = unpack(fixtures.prefix),
    rotationOptions = { minInlierRatio: 0.6 };
  const session = new api.IncrementalSfmSession(device, {
    ...controls,
    rotationTracking: rotationOptions,
  });
  rotationOptions.minInlierRatio = 0; // The session owns its options snapshot.
  const checkpoints = [];
  let isolated,
    waitingFinish,
    pairVerifications = 0,
    last;
  for (const image of prefix.images) {
    last = await session.add(
      image,
      prefix.pairs.filter((p) => p.second === image.id),
    );
    if (last)
      pairVerifications += last.execution.stages.filter(
        (s) => s.operation === 'pairVerification',
      ).length;
    checkpoints.push({
      id: image.id,
      registered: last?.reconstruction.views.length ?? 0,
      tracking: describe(session.tracking),
    });
    if (image.id === 2) {
      const before = JSON.stringify(session.tracking),
        snapshot = session.tracking;
      snapshot.rotations[0].rotation.R.fill(999);
      snapshot.rotations[0].firstCamera.fx = 999;
      snapshot.pairs[0].matches.fill(999);
      snapshot.tracks[0].observations[0].featureId = 999;
      snapshot.untriangulatedTrackIds.push(999);
      isolated = JSON.stringify(session.tracking) === before;
      try {
        await session.finish();
      } catch (error) {
        waitingFinish = {
          name: error.name,
          tracking: describe(error.tracking),
          fabricatedMap: 'reconstruction' in error,
        };
      }
    }
  }
  const before = JSON.stringify(session.tracking);
  last.tracking.rotations[0].rotation.R.fill(123);
  last.tracking.tracks[0].observations[0].featureId = 123;
  const resultIsolated = JSON.stringify(session.tracking) === before;
  const final = await session.finish();
  const transfer = [];
  let defaultFailureTracking;
  for (const rotationTracking of [false, true, undefined]) {
    const before = device.statistics;
    let failure;
    try {
      await api.reconstructIncrementalGpu(
        device,
        prefix.images.slice(0, 3),
        prefix.pairs.filter((p) => p.second < 3),
        { ...controls, rotationTracking },
      );
    } catch (error) {
      failure = error.name;
      if (rotationTracking === undefined) defaultFailureTracking = describe(error.tracking);
    }
    const after = device.statistics;
    transfer.push({
      failure,
      upload: after.applicationUploadBytes - before.applicationUploadBytes,
      readback: after.applicationReadbackBytes - before.applicationReadbackBytes,
    });
  }
  const defaults = [];
  for (const rotationTracking of [undefined, false]) {
    const session = new api.IncrementalSfmSession(device, { ...controls, rotationTracking });
    for (const image of prefix.images)
      await session.add(
        image,
        prefix.pairs.filter((p) => p.second === image.id),
      );
    const final = await session.finish();
    defaults.push({
      registered: final.reconstruction.views.length,
      points: final.reconstruction.landmarks.length,
      tracking: describe(session.tracking),
      hasResultTracking: 'tracking' in final,
      cpuStages: final.execution.cpuStages,
    });
  }
  const window = new api.IncrementalSfmSession(device, {
    ...controls,
    windowSize: 8,
    rotationTracking: true,
  });
  let windowSafe = true;
  for (let id = 0; id < 10; id++) {
    const image = { ...prefix.images[Math.min(id, 2)], id };
    const result = await window.add(
      image,
      id ? [{ first: id - 1, second: id, matches: prefix.pairs[0].matches }] : [],
    );
    const active = new Set(window.activeViewIds);
    windowSafe &&=
      !result &&
      active.size <= 8 &&
      window.tracking.rotations.every((p) => active.has(p.first) && active.has(p.second)) &&
      window.tracking.tracks.every((t) => t.observations.every((o) => active.has(o.viewId)));
  }
  const cases = [];
  for (const fixture of [fixtures.unseen, fixtures.reseen]) {
    const f = unpack(fixture);
    const result = await api.reconstructIncrementalGpu(device, f.images, f.pairs, {
      ...controls,
      initialPair: [0, 1],
      rotationTracking: true,
    });
    cases.push({
      registered: result.reconstruction.views.length,
      points: result.reconstruction.landmarks.length,
      tracking: describe(result.tracking),
      pendingObservations: result.tracking.untriangulatedTrackIds.map(
        (id) => result.tracking.tracks[id].observations,
      ),
      cpuStages: result.execution.cpuStages,
    });
  }
  return {
    checkpoints,
    isolated,
    resultIsolated,
    waitingFinish,
    pairVerifications,
    final: {
      registered: final.reconstruction.views.length,
      points: final.reconstruction.landmarks.length,
      tracking: describe(final.tracking),
      cpuStages: final.execution.cpuStages,
    },
    finishPairVerifications: final.execution.stages.filter(
      (s) => s.operation === 'pairVerification',
    ).length,
    cases,
    transfer,
    defaultFailureTracking,
    defaults,
    windowSafe,
    liveBuffers: device.statistics.liveBuffers,
  };
}
