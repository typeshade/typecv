// The caller supplies the same original gray images and seeds to CPU and WebGPU.
export async function runKltReadbackCases(api, device, input) {
  const first = Float32Array.from(input.first),
    second = Float32Array.from(input.second),
    points = Float32Array.from(input.points),
    options = api.kltOptionsFor(input.width, { levels: 2, halfWindow: 4 }),
    originalRead = device.read.bind(device);
  let serial = false,
    pending = Promise.resolve();
  device.read = (...args) =>
    serial ? (pending = pending.then(() => originalRead(...args))) : originalRead(...args);
  const snapshot = () => {
    const s = device.statistics;
    return {
      reads: s.reads,
      waves: s.readWaves,
      bytes: s.applicationReadbackBytes,
      snapshots: s.residentSnapshotBytes,
    };
  };
  const run = async (gpu, isSerial) => {
    serial = isSerial;
    const before = snapshot(),
      result = await api.trackKltGpu(
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
        records: [...result.records],
        forward: [...result.forward],
        back: [...result.back],
        ...(result.pyramids
          ? {
              pyramids: {
                gray: result.pyramids.gray.map((array) => [...array]),
                grad: result.pyramids.grad.map((array) => [...array]),
              },
            }
          : {}),
      },
      counts: Object.fromEntries(Object.keys(before).map((key) => [key, after[key] - before[key]])),
    };
  };
  const modes = [
    ['standalone outputs', {}],
    [
      'affine batch outputs',
      {
        readBack: false,
        affineWarps: Float32Array.from({ length: points.length * 2 }, (_, i) =>
          i % 4 === 0 || i % 4 === 3 ? 1 : 0,
        ),
      },
    ],
    ['outputs and image pyramids', { readPyramids: true }],
  ];
  try {
    const rows = [];
    for (const [name, gpu] of modes)
      rows.push({ name, serial: await run(gpu, true), concurrent: await run(gpu, false) });
    return { rows, liveBytes: device.statistics.liveBytes };
  } finally {
    device.read = originalRead;
  }
}
