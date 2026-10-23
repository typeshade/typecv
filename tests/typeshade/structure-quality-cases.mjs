export async function runStructureQualityCases(api, device, fixtures) {
  const camera = { fx: 800, fy: 800, cx: 320, cy: 240 },
    xyz = Float64Array.of(0.1, -0.15, 10);
  const observations = [0, 0.1, 0.4].map((x) => ({
    camera,
    pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(-x, 0, 0) },
    pixel: [(800 * (0.1 - x)) / 10 + 320, (800 * -0.15) / 10 + 240],
  }));
  const triangulated = await api.triangulateTracksGpu(
    device,
    [observations.slice(0, 2), [observations[0], observations[2]]],
    { maxRelativeDepthStdDev: 0.1 },
  );
  const model = {
    views: observations.map((o, id) => ({ id, camera, pose: o.pose })),
    landmarks: [
      [0, 1],
      [0, 2],
    ].map((ids, trackId) => ({
      trackId,
      xyz,
      observations: ids.map((viewId) => ({
        viewId,
        featureId: trackId,
        pixel: observations[viewId].pixel,
      })),
    })),
  };
  const before = device.statistics;
  const unguarded = await api.evaluateObservationsGpu(device, model, 3);
  const after = device.statistics;
  const guarded = await api.evaluateObservationsGpu(device, model, 3, {
    maxRelativeDepthStdDev: 0.1,
  });
  const last = device.statistics;
  const unpack = (f) => ({
    images: f.images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
    pairs: f.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  });
  const controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 2 };
  const t = unpack(fixtures.translation),
    session = new api.IncrementalSfmSession(device, { ...controls, maxRelativeDepthStdDev: 0.03 });
  let sessionPending;
  for (const image of t.images) {
    await session.add(
      image,
      t.pairs.filter((p) => p.second === image.id),
    );
    if (image.id === 1)
      sessionPending = {
        depth: session.tracking.depthDeferredTrackIds.length,
        tracks: session.tracking.untriangulatedTrackIds.length,
      };
  }
  const final = await session.finish();
  const f = unpack(fixtures.foreground),
    masks = fixtures.foreground.masks.map((m) => ({ ...m, values: Uint8Array.from(m.values) }));
  const pending = api.reconstructIncrementalGpu(device, f.images, f.pairs, {
    ...controls,
    cameraObservationMasks: masks,
  });
  masks.forEach((m) => m.values.fill(0));
  const mapped = await pending;
  const maskedSession = new api.IncrementalSfmSession(device, controls);
  let selectionIsolated = true;
  for (const image of f.images) {
    const mask = {
      ...fixtures.foreground.masks.find((m) => m.viewId === image.id),
      values: Uint8Array.from(fixtures.foreground.masks.find((m) => m.viewId === image.id).values),
    };
    const update = maskedSession.add(
      image,
      f.pairs.filter((p) => p.second === image.id),
      mask,
    );
    mask.values.fill(0);
    await update;
    const before = JSON.stringify(maskedSession.observationSelection),
      snapshot = maskedSession.observationSelection;
    snapshot.masks[0].values.fill(99);
    selectionIsolated &&= JSON.stringify(maskedSession.observationSelection) === before;
  }
  const maskedFinal = await maskedSession.finish();
  const n = unpack(fixtures.noisy),
    noisy = await api.reconstructIncrementalGpu(device, n.images, n.pairs, {
      ...controls,
      pairRefinementIterations: 10,
      maxRelativeDepthStdDev: 0.1,
    });
  return {
    status: [...triangulated.status],
    rejectedHasGeometry: !!triangulated.tracks[0],
    postMasks: guarded.masks.map((m) => [...m]),
    depthRejected: [...guarded.depthRejected],
    extraReadbackBytes:
      guarded.execution.applicationReadbackBytes - unguarded.execution.applicationReadbackBytes,
    readCalls: [after.reads - before.reads, last.reads - after.reads],
    sessionPending,
    sessionFinal: {
      registered: final.reconstruction.views.length,
      points: final.reconstruction.landmarks.length,
      depth: session.tracking.depthDeferredTrackIds,
    },
    foreground: {
      sessionPoints: maskedFinal.reconstruction.landmarks.length,
      selectionIsolated,
      registered: mapped.reconstruction.views.length,
      points: mapped.reconstruction.landmarks.length,
      backgroundOnly: mapped.reconstruction.landmarks.every((p) =>
        p.observations.every((o) => o.featureId < 60),
      ),
      separated: mapped.observationSelection.separatedPairs.reduce(
        (n, p) => n + p.matches.length / 2,
        0,
      ),
      inputRows: mapped.observationSelection.inputMatchCount,
      source: mapped.observationSelection.masks[0].source,
    },
    noisy: {
      registered: noisy.reconstruction.views.length,
      initialPair: noisy.diagnostics.initialPair,
      model: {
        views: noisy.reconstruction.views.map((v) => ({
          ...v,
          pose: { R: [...v.pose.R], t: [...v.pose.t] },
        })),
        landmarks: noisy.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
      },
    },
    liveBuffers: device.statistics.liveBuffers,
  };
}
