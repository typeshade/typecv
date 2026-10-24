export async function runCompactCases(api, device, input) {
  const first = Float32Array.from(input.first),
    options = api.kltOptionsFor(input.width, {
      levels: 2,
      halfWindow: 4,
      maxIterations: 30,
      epsilonPx: 0.005,
    }),
    affine = {
      affineWarps: Float32Array.from(input.affineWarps),
      initialTargets: Float32Array.from(input.initialTargets),
    },
    modes = [
      { name: 'standalone backward output', second: input.shifted, gpu: {} },
      { name: 'affine batch', second: input.warped, gpu: { ...affine, readBack: false } },
      {
        name: 'affine pyramid snapshots',
        second: input.warped,
        gpu: { ...affine, readPyramids: true },
      },
      {
        name: 'ended forward tracks',
        second: Array(input.width * input.height).fill(0.5),
        gpu: { readBack: false },
      },
      {
        name: 'cached affine batch',
        second: input.warped,
        gpu: { ...affine, readBack: false },
        cache: true,
      },
      { name: 'zero tracks', second: input.shifted, gpu: { readBack: false }, empty: true },
    ],
    snapshot = () => {
      const s = device.statistics;
      return {
        reads: s.reads,
        waves: s.readWaves,
        bytes: s.applicationReadbackBytes,
        snapshots: s.residentSnapshotBytes,
      };
    },
    rows = [];
  for (const mode of modes) {
    const second = Float32Array.from(mode.second),
      points = mode.empty ? new Float32Array() : Float32Array.from(input.points),
      frames = [first, second].map((gray, id) => ({
        id,
        width: input.width,
        height: input.height,
        gray,
      })),
      cache = mode.cache ? api.createKltPairCache(device, { capacity: 2 }) : undefined,
      run = async (compactOutput) => {
        const before = snapshot(),
          gpu = { ...mode.gpu, chunkTracks: 5, compactOutput },
          result = cache
            ? await cache.track(frames[0], frames[1], points, options, gpu)
            : await api.trackKltGpu(
                device,
                first,
                second,
                input.width,
                input.height,
                points,
                options,
                gpu,
              ),
          after = snapshot();
        return {
          output: {
            records: Array.from(result.records),
            forward: Array.from(result.forward),
            back: Array.from(result.back),
            ncc: result.ncc ? Array.from(result.ncc) : null,
            pyramids: result.pyramids
              ? {
                  gray: result.pyramids.gray.map((x) => Array.from(x)),
                  grad: result.pyramids.grad.map((x) => Array.from(x)),
                }
              : null,
          },
          dispatches: result.report.dispatches,
          sharedSnapshot: result.ncc?.buffer === result.records.buffer,
          counts: Object.fromEntries(
            Object.keys(before).map((key) => [key, after[key] - before[key]]),
          ),
        };
      };
    try {
      rows.push({
        name: mode.name,
        count: points.length / 2,
        affine: !!mode.gpu.affineWarps,
        legacy: await run(false),
        compact: await run(true),
      });
    } finally {
      await cache?.dispose();
    }
  }
  return { rows, liveBytes: device.statistics.liveBytes };
}
