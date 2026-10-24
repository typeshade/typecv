export async function runRoundtrips(api, device, input) {
  const run = async (targetDevice, group, c, batchPasses) => {
    const originalPlan = targetDevice.plan,
      plan = originalPlan.bind(targetDevice);
    let planCalls = 0,
      result;
    // Observe the real execution path; every call still delegates to the device.
    targetDevice.plan = (...args) => {
      planCalls++;
      return plan(...args);
    };
    try {
      result = await api.trackKltGpu(
        targetDevice,
        c.first ?? group.first,
        c.second,
        group.width,
        group.height,
        c.points,
        api.kltOptionsFor(group.width, c.options),
        { ...c.gpu, ...(batchPasses === undefined ? {} : { batchPasses }) },
      );
    } finally {
      targetDevice.plan = originalPlan;
    }
    return {
      name: c.name,
      forward: Array.from(result.forward),
      back: Array.from(result.back),
      records: Array.from(result.records),
      planCalls,
      entryDispatches: result.report.dispatches,
      pairEntries: result.report.groups.reduce(
        (count, group) => count + group.calls.filter((name) => name === 'kltTrackPair').length,
        0,
      ),
      directionalEntries: result.report.groups.reduce(
        (count, group) => count + group.calls.filter((name) => name === 'kltTrack').length,
        0,
      ),
      checkEntries: result.report.groups.reduce(
        (count, group) => count + group.calls.filter((name) => name === 'kltCheck').length,
        0,
      ),
    };
  };
  // Warm both public scheduling modes before comparing real work. Their zero-work
  // pipeline warmups intentionally record different entry lists on the GPU tier.
  const firstGroup = input.groups[0],
    firstCase = firstGroup.cases[0];
  await run(device, firstGroup, firstCase, true);
  await run(device, firstGroup, firstCase, false);
  const defaultRow = await run(device, firstGroup, firstCase);
  const rows = [],
    separateRows = [];
  for (const group of input.groups)
    for (const c of group.cases) {
      rows.push(await run(device, group, c, true));
      separateRows.push(await run(device, group, c, false));
    }
  const group = input.groups[0],
    c = group.cases[0],
    layout = api.kltLayout(group.width, group.height, c.options.levels),
    count = c.points.length / 2,
    // This non-affine case admits its two gray/gradient pyramids and four point
    // arrays exactly. Grouping may not require any extra working storage.
    workingBytes = 24 * layout.total + 64 * count,
    tightDevice = await api.ShadeDevice.open({
      backend: device.backend,
      poolBytes: 0,
      workingBytes,
    });
  let tight;
  try {
    await run(tightDevice, group, c, true);
    await run(tightDevice, group, c, false);
    tight = {
      grouped: await run(tightDevice, group, c, true),
      separate: await run(tightDevice, group, c, false),
      workingBytes,
      liveBytes: tightDevice.statistics.liveBytes,
      peakActiveBytes: tightDevice.diagnostics.memory.peakActiveBytes,
    };
  } finally {
    await tightDevice.dispose();
  }
  return {
    rows,
    separateRows,
    defaultRow,
    tight,
    liveBytes: device.statistics.liveBytes,
    peakActiveBytes: device.diagnostics.memory.peakActiveBytes,
  };
}
