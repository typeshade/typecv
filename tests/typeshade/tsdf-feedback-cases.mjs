/** Small, exact planes so feedback and transfer ownership are tested without a long MVS run. */
export function feedbackInput() {
  const width = 8,
    height = 8,
    n = width * height;
  return {
    width,
    height,
    views: [0, 1, 2].map((i) => ({
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      center: [i * 0.1, 0, 0],
      fx: 8,
      fy: 8,
      cx: 3.5,
      cy: 3.5,
      depth: new Float32Array(n).fill(2),
      normal: new Uint32Array(n).fill(0x7fff7fff),
      rgba: new Uint32Array(n).fill(0xff3366aa),
    })),
  };
}

export async function feedbackCase({ ShadeDevice, fuseTsdfGpu }, backend) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  const options = { levels: 1, voxelScale: 1, truncation: 2 };
  const plain = (result) => ({
    count: result.count,
    positions: [...result.positions],
    colors: [...result.colors],
    normals: [...result.normals],
    weights: [...result.weights],
    free: [...result.free],
    tallies: [...result.tallies],
    dispatches: result.tsdf.report.dispatches,
    submissions: result.tsdf.report.submissions,
  });
  try {
    const input = feedbackInput();
    // Warm shader entries before comparing submission counts.
    await fuseTsdfGpu(device, input, options);
    const baseline = await fuseTsdfGpu(device, input, options);
    const progress = [],
      parts = [];
    const observed = await fuseTsdfGpu(device, input, {
      ...options,
      onProgress: (event) => progress.push(event),
      onPart: (part) =>
        parts.push(structuredClone(part, { transfer: [part.positions.buffer, part.data.buffer] })),
    });
    const abort = new AbortController();
    let cancelled = false;
    try {
      await fuseTsdfGpu(device, input, {
        ...options,
        signal: abort.signal,
        onProgress: (event) => {
          if (event.phase === 'integrate' && event.done > 0) abort.abort();
        },
      });
    } catch (error) {
      cancelled = error.name === 'AbortError';
    }
    return {
      baseline: plain(baseline),
      observed: plain(observed),
      progress,
      partCounts: parts.map((part) => part.positions.length / 3),
      cancelled,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}

/** The public MVS callback maps packed indices back to scene ids and previews to world units. */
export function mvsFeedbackInput(planMvsViews) {
  const input = feedbackInput();
  const views = input.views.map((v, i) => ({
    id: 10 + i,
    width: 8,
    height: 8,
    pose: { R: v.rotation, t: [-v.center[0], 0, 0] },
    camera: { fx: 8, fy: 8, cx: 3.5, cy: 3.5 },
  }));
  const landmarks = Array.from({ length: 16 }, (_, i) => {
    const xyz = [((i % 4) - 1.5) / 4, (Math.floor(i / 4) - 1.5) / 4, 2];
    return {
      trackId: i,
      xyz,
      observations: views.map((v) => ({
        viewId: v.id,
        featureId: i,
        pixel: [(8 * (xyz[0] + v.pose.t[0])) / 2 + 3.5, (8 * xyz[1]) / 2 + 3.5],
      })),
    };
  });
  const scene = { views, landmarks },
    plan = planMvsViews(scene);
  const images = new Map(
    views.map((v) => [v.id, { width: 8, height: 8, rgba: new Uint8ClampedArray(256).fill(180) }]),
  );
  const depthMaps = {
    depth: input.views.map((v) => Float32Array.from(v.depth, (x) => x / plan.normalization.scale)),
    normal: input.views.map((v) => v.normal),
    confidence: input.views.map(() => new Float32Array(64).fill(1)),
  };
  return { scene, plan, images, depthMaps };
}

export async function mvsFeedbackCase({ ShadeDevice, reconstructDense, planMvsViews }, backend) {
  const { scene, images, depthMaps } = mvsFeedbackInput(planMvsViews);
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const options = {
      fusion: 'tsdf',
      tsdf: { levels: 1, voxelScale: 1, truncation: 2 },
      depthMaps,
    };
    const baseline = await reconstructDense(device, scene, images, options);
    const events = [],
      previews = [],
      finalEvents = [];
    const observed = await reconstructDense(device, scene, images, {
      ...options,
      onFusionProgress: (e) => events.push(e),
      onFusionPreview: (p) =>
        previews.push(
          structuredClone(p, { transfer: [p.positions.buffer, p.colors.buffer, p.normals.buffer] }),
        ),
      onView: (e) => finalEvents.push(e.type),
    });
    const ordered = (positions) =>
      Array.from({ length: positions.length / 3 }, (_, i) =>
        Array.from(positions.slice(i * 3, i * 3 + 3)),
      ).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return {
      baseline: { count: baseline.points.count, positions: [...baseline.points.positions] },
      observed: { count: observed.points.count, positions: [...observed.points.positions] },
      previewPoints: ordered(Float32Array.from(previews.flatMap((p) => [...p.positions]))),
      finalPoints: ordered(observed.points.positions),
      finalEvents,
      viewIds: [...new Set(events.filter((e) => e.viewId !== undefined).map((e) => e.viewId))],
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
