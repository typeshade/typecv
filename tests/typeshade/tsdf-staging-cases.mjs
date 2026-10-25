/** Synthetic views for the TSDF channel staging and discovery-key reuse checks. */
export function tsdfStagingInput({ width = 16, height = 16, views = 8 } = {}) {
  const pixels = width * height;
  return {
    width,
    height,
    views: Array.from({ length: views }, (_, v) => {
      // Nonzero offsets catch a borrowed channel accidentally exposing its entire backing array.
      const depth = new Float32Array(new ArrayBuffer((pixels + 6) * 4), 12, pixels);
      const normal = new Uint32Array(new ArrayBuffer((pixels + 4) * 4), 8, pixels);
      const rgba = new Uint32Array(new ArrayBuffer((pixels + 8) * 4), 16, pixels);
      const confidence = new Float32Array(new ArrayBuffer((pixels + 2) * 4), 4, pixels);
      for (let i = 0; i < pixels; i++) {
        depth[i] = i % 59 === 0 ? 0 : 2;
        if (
          v === 0 &&
          i % width > width * 0.35 &&
          i % width < width * 0.55 &&
          Math.floor(i / width) > height * 0.35 &&
          Math.floor(i / width) < height * 0.55
        )
          depth[i] = 1.65;
        normal[i] = 0x7fff7fff;
        confidence[i] = i % 37 === 0 ? 0 : 0.5 + (i % 17) / 34;
        rgba[i] =
          (0xff000000 |
            ((i * 13 + v * 7) % 256) |
            (((i * 3 + v * 11) % 256) << 8) |
            (((i * 17 + v * 19) % 256) << 16)) >>>
          0;
      }
      return {
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        center: [(v - (views - 1) * 0.5) * 0.02, ((v % 3) - 1) * 0.018, 0],
        fx: width * 2,
        fy: width * 2,
        cx: (width - 1) * 0.5,
        cy: (height - 1) * 0.5,
        k1: v % 3 === 0 ? 0.003 : 0,
        k2: v % 3 === 0 ? -0.0001 : 0,
        depth,
        normal,
        rgba,
        ...(v % 4 === 0 ? {} : { confidence }),
      };
    }),
  };
}

export const TSDF_STAGING_OPTIONS = {
  levels: 2,
  voxelScale: 1,
  truncation: 3,
  visibility: 'view-weighted',
};
export const TSDF_POINT_CHANNELS = [
  'positions',
  'normals',
  'colors',
  'weights',
  'free',
  'freeWeights',
  'tallies',
];

export function inputChannels(input) {
  return input.views.flatMap((view) => [
    view.depth,
    view.normal,
    view.rgba,
    ...(view.confidence ? [view.confidence] : []),
  ]);
}

/** Bitwise digest, including f32 payloads and only each typed view's own byte range. */
export async function digestArrays(arrays) {
  const bytes = new Uint8Array(arrays.reduce((n, array) => n + array.byteLength, 0));
  let offset = 0;
  for (const array of arrays) {
    bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), offset);
    offset += array.byteLength;
  }
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (value) =>
    value.toString(16).padStart(2, '0'),
  ).join('');
}

export async function pointDigests(points) {
  return Object.fromEntries(
    await Promise.all(
      TSDF_POINT_CHANNELS.filter((name) => points[name]).map(async (name) => [
        name,
        await digestArrays([points[name]]),
      ]),
    ),
  );
}

/** Run both budget paths and failure cleanup using the same caller-owned arrays. */
export async function stagingCase({ ShadeDevice, fuseTsdfGpu }, backend) {
  const input = tsdfStagingInput();
  const before = await digestArrays(inputChannels(input));
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const whole = await fuseTsdfGpu(device, input, TSDF_STAGING_OPTIONS);
    const boxes = await fuseTsdfGpu(device, input, { ...TSDF_STAGING_OPTIONS, maxBlocks: 12 });
    const explicitInput = {
      ...input,
      views: input.views.map((view) => ({
        ...view,
        confidence: view.confidence ?? new Float32Array(input.width * input.height).fill(1),
      })),
    };
    const explicit = await fuseTsdfGpu(device, explicitInput, TSDF_STAGING_OPTIONS);
    const failures = [];
    for (const kind of ['abort', 'callback']) {
      const controller = new AbortController();
      let reached = false;
      try {
        await fuseTsdfGpu(device, input, {
          ...TSDF_STAGING_OPTIONS,
          maxBlocks: 12,
          signal: controller.signal,
          onProgress(event) {
            if (kind === 'abort' && event.phase === 'integrate') {
              reached = true;
              controller.abort();
            }
          },
          onPart() {
            if (kind === 'callback') {
              reached = true;
              throw new Error('synthetic callback failure');
            }
          },
        });
        failures.push({ kind, reached, error: null, liveBytes: device.statistics.liveBytes });
      } catch (error) {
        failures.push({
          kind,
          reached,
          error: kind === 'abort' ? error.name : error.message,
          liveBytes: device.statistics.liveBytes,
        });
      }
    }
    // Admit three view channels, then reject the fourth before a kernel can be submitted.
    // 512 pixels keep the requested budget above ShadeDevice's 4 KiB minimum option.
    const budgetInput = tsdfStagingInput({ width: 32, height: 16, views: 2 });
    const budgetDevice = await ShadeDevice.open({
      backend,
      poolBytes: 0,
      workingBytes: 3 * budgetInput.width * budgetInput.height * 4,
    });
    try {
      try {
        await fuseTsdfGpu(budgetDevice, budgetInput, TSDF_STAGING_OPTIONS);
        failures.push({ kind: 'budget', reached: false, error: null });
      } catch (error) {
        failures.push({
          kind: 'budget',
          reached: true,
          error: error.name,
          message: error.message,
          liveBytes: budgetDevice.statistics.liveBytes,
        });
      }
    } finally {
      await budgetDevice.dispose();
    }
    const plain = async (points) => ({
      count: points.count,
      channels: await pointDigests(points),
      tsdf: { ...points.tsdf, report: undefined },
    });
    return {
      whole: await plain(whole),
      boxes: await plain(boxes),
      explicit: await plain(explicit),
      before,
      after: await digestArrays(inputChannels(input)),
      failures,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
