export async function runPatches(api, device, input) {
  const opts = api.kltOptionsFor(input.width, {
      levels: 2,
      halfWindow: 5,
      epsilonPx: 0.005,
      maxIterations: 30,
    }),
    out = [];
  for (const c of input.cases) {
    const plain = await api.trackKltGpu(
      device,
      input.first,
      c.second,
      input.width,
      input.height,
      input.points,
      opts,
      { initialTargets: c.guesses },
    );
    const r = await api.trackKltGpu(
      device,
      input.first,
      c.second,
      input.width,
      input.height,
      input.points,
      opts,
      { initialTargets: c.guesses, affineWarps: c.affine, chunkTracks: 5 },
    );
    let plainMax = 0;
    for (let k = 0; k < input.points.length / 2; k++)
      if (plain.records[4 * k + 3] & 1)
        plainMax = Math.max(
          plainMax,
          Math.hypot(
            plain.records[4 * k] - c.truth[2 * k],
            plain.records[4 * k + 1] - c.truth[2 * k + 1],
          ),
        );
    out.push({
      records: Array.from(r.records),
      forward: Array.from(r.forward),
      back: Array.from(r.back),
      plainMax,
    });
  }
  return { cases: out, liveBytes: device.statistics.liveBytes };
}
