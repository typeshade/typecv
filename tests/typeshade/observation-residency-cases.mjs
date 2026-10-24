const typedPairs = (pairs) =>
  pairs.map((pair) => ({ ...pair, matches: Uint32Array.from(pair.matches) }));

const quality = (result) => ({
  images: result.images.map((image) => ({
    id: image.id,
    camera: image.camera,
    keypoints: Array.from(image.keypoints),
  })),
  pairs: result.candidatePairs.map((pair) => ({
    first: pair.first,
    second: pair.second,
    matches: Array.from(pair.matches),
    distances: pair.distances ? Array.from(pair.distances) : null,
  })),
  observations: result.newObservations,
});

export async function runResidencyCases(api, input, settings = {}) {
  const rows = [],
    levels = 3,
    maxSeedsPerFrame = 32,
    originalSentinel = Array.from({ length: 17 }, (_, i) => i / 32 - 3);
  for (const entry of input.cases) {
    const modes = [];
    for (const name of ['off', 'auto', 'three']) {
      const frames = entry.frames.map((frame) => ({
          ...frame,
          gray: Float32Array.from(frame.gray),
          keypoints: Float64Array.from(frame.keypoints),
        })),
        { width, height } = frames[0],
        pyramidBytes = api.kltLayout(width, height, levels).total * 12,
        sentinelBytes = originalSentinel.length * 4,
        workingBytes = settings.tightBudget
          ? 2 * pyramidBytes + 96 * maxSeedsPerFrame + sentinelBytes
          : 64 * 1024 ** 2,
        device = await api.ShadeDevice.open({
          backend: settings.backend ?? 'cpu',
          ...(settings.gpuDevice ? { gpuDevice: settings.gpuDevice } : {}),
          poolBytes: settings.poolBytes ?? 0,
          workingBytes,
        }),
        sentinel = device.allocate(originalSentinel.length, Float32Array.from(originalSentinel)),
        pairs = typedPairs(entry.pairs),
        options = {
          ...(name === 'off'
            ? { pyramidCache: false }
            : name === 'three'
              ? { pyramidCache: true }
              : {}),
          radius: entry.radius,
          maxSeedsPerFrame,
          geometryPatchWarp: true,
          multiViewConsistency: entry.multiViewConsistency ?? false,
          ...(entry.verifiedPairs ? { verifiedPairs: typedPairs(entry.verifiedPairs) } : {}),
          ...(entry.model
            ? {
                initialModel: {
                  views: entry.model.views.map((view) => ({
                    ...view,
                    pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
                  })),
                  landmarks: entry.model.landmarks.map((point) => ({
                    ...point,
                    xyz: Float64Array.from(point.xyz),
                  })),
                },
              }
            : {}),
          klt: { levels, halfWindow: 5, maxIterations: 20, epsilonPx: 0.005 },
        };
      let released = false;
      try {
        const first = await api.expandObservationsGpu(device, frames, pairs, options),
          activeAfter = device.statistics.liveBytes - device.statistics.pooledBytes,
          second = await api.expandObservationsGpu(device, frames, pairs, {
            ...options,
            ...(name === 'auto' ? { pyramidCache: undefined } : {}),
          }),
          activeAfterRepeat = device.statistics.liveBytes - device.statistics.pooledBytes,
          values = Array.from(await device.read(sentinel, originalSentinel.length)),
          invalidCacheOptions = [];
        for (const value of [0, 1, null, 'auto', {}, []]) {
          try {
            await api.expandObservationsGpu(device, frames, pairs, {
              ...options,
              pyramidCache: value,
            });
            invalidCacheOptions.push('accepted');
          } catch (error) {
            invalidCacheOptions.push(error.name);
          }
        }
        const peakActiveBytes = device.diagnostics.memory.peakActiveBytes,
          originalGrayPreserved = frames.every((frame, j) =>
            frame.gray.every((value, i) => value === entry.frames[j].gray[i]),
          );
        device.recycle(sentinel);
        released = true;
        const activeAfterRelease = device.statistics.liveBytes - device.statistics.pooledBytes;
        await device.dispose();
        modes.push({
          name,
          quality: quality(first),
          repeatedQuality: quality(second),
          cache: first.diagnostics.cache ?? null,
          repeatedCache: second.diagnostics.cache ?? null,
          activeAfter,
          activeAfterRepeat,
          activeAfterRelease,
          liveAfterDispose: device.statistics.liveBytes,
          peakActiveBytes,
          workingBytes,
          sentinelBytes,
          sentinel: values,
          originalSentinel,
          pendingReads: device.diagnostics.memory.pendingReads,
          originalGrayPreserved,
          originalKeypoints: frames.map((frame) => Array.from(frame.keypoints)),
          invalidCacheOptions,
        });
      } finally {
        if (!released) device.recycle(sentinel);
        await device.dispose();
      }
    }
    rows.push({ name: entry.name, modes });
  }
  return { rows };
}
