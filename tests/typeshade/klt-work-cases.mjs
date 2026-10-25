export async function runWork(api, device, input) {
  const rows = [];
  for (const c of input.cases) {
    const r = await api.trackKltGpu(
      device,
      c.first ?? input.first,
      c.second,
      input.width,
      input.height,
      c.points,
      api.kltOptionsFor(input.width, c.options),
      c.gpu,
    );
    rows.push({
      name: c.name,
      records: Array.from(r.records),
      forward: Array.from(r.forward),
      back: Array.from(r.back),
      trackChunks: r.trackChunks,
    });
  }
  return { rows, liveBytes: device.statistics.liveBytes };
}
