function packedNormal() {
  const length = Math.hypot(0.15, 0.08, -1);
  const n = [0.15 / length, 0.08 / length, -1 / length];
  const sum = n.reduce((a, v) => a + Math.abs(v), 0);
  const x = n[0] / sum,
    y = n[1] / sum;
  const snorm = (v) => Math.round(v * 32767) & 0xffff;
  return (snorm(1 - Math.abs(y)) | (snorm(1 - Math.abs(x)) << 16)) >>> 0;
}
export function planarInput(mode = 'plane') {
  const width = 16,
    height = 16,
    P = width * height;
  const truth = [];
  const centers =
    mode === 'limited-overlap'
      ? [-0.2, 0, 0.2]
      : mode === 'repeated-source'
        ? [-0.1, 0, 0]
        : [-0.1, 0, 0.1];
  const views = centers.map((cx, index) => {
    const depth = new Float32Array(P);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        depth[y * width + x] =
          (2 + 0.15 * cx) / (1 - (0.15 * (x - 7.5)) / 32 - (0.08 * (y - 7.5)) / 32);
    truth.push(depth.slice());
    const rgba = new Uint32Array(P).fill(0xff808080);
    if (index === 0)
      for (let y = 5; y < 11; y++)
        for (let x = 5; x < 11; x++) {
          depth[y * width + x] = 0;
          if (mode === 'color-edge') rgba[y * width + x] = 0xffffffff;
        }
    if (mode === 'occluded' && index > 0) depth.fill(1);
    if (mode === 'one-source' && index === 2) depth.fill(4);
    if (mode === 'depth-edge' && index === 0)
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++)
          if (depth[y * width + x] > 0) depth[y * width + x] = x < 8 ? 1 : 3;
    return {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      center: [cx, 0, 0],
      fx: 32,
      fy: 32,
      cx: 7.5,
      cy: 7.5,
      depth,
      confidence: Float32Array.from(depth, (d) =>
        d > 0 ? (mode === 'weak-support' ? 0.05 : 1) : 0,
      ),
      normal: new Uint32Array(P).fill(packedNormal()),
      rgba,
    };
  });
  if (mode === 'rotated')
    for (const v of views) {
      v.center = [0.3, -0.2 + v.center[0], 0.7];
      v.rotation = [0, 1, 0, -1, 0, 0, 0, 0, 1];
    }
  return { input: { width, height, views }, truth };
}

export async function planarDepthCase(
  { ShadeDevice, completePlanarDepths, decodeMvsNormal },
  backend,
) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    const results = [];
    for (const mode of [
      'plane',
      'rotated',
      'weak-support',
      'limited-overlap',
      'occluded',
      'one-source',
      'repeated-source',
      'color-edge',
      'depth-edge',
    ]) {
      const { input, truth } = planarInput(mode);
      const before = input.views.map((v) => ({
        depth: v.depth.slice(),
        normal: v.normal.slice(),
        confidence: v.confidence.slice(),
      }));
      const completed = await completePlanarDepths(device, input);
      let worstRelativeError = 0,
        worstNormalError = 0,
        changedExisting = 0,
        changedInput = 0;
      for (let v = 0; v < input.views.length; v++)
        for (let i = 0; i < input.width * input.height; i++) {
          const output = completed.input.views[v];
          for (const key of ['depth', 'normal', 'confidence']) {
            if (before[v][key][i] !== input.views[v][key][i]) changedInput++;
            if (before[v].depth[i] > 0 && before[v][key][i] !== output[key][i]) changedExisting++;
          }
          if (!(before[v].depth[i] > 0) && output.depth[i] > 0) {
            worstRelativeError = Math.max(
              worstRelativeError,
              Math.abs(output.depth[i] / truth[v][i] - 1),
            );
            const n = decodeMvsNormal(output.normal[i]);
            const cosine = (0.15 * n[0] + 0.08 * n[1] - n[2]) / Math.hypot(0.15, 0.08, -1);
            worstNormalError = Math.max(worstNormalError, 1 - cosine);
            if (output.confidence[i] > 0.250001)
              throw new Error('Plane confidence exceeded its bound');
          }
        }
      results.push({
        mode,
        ...completed.report,
        worstRelativeError,
        worstNormalError,
        changedExisting,
        changedInput,
        depth: [...completed.input.views[0].depth],
        confidence: [...completed.input.views[0].confidence],
      });
    }
    let budgetRejected = false,
      cancelled = false;
    try {
      await completePlanarDepths(device, planarInput().input, { residentBytes: 1 });
    } catch (error) {
      budgetRejected = error instanceof RangeError;
    }
    const abort = new AbortController();
    abort.abort();
    try {
      await completePlanarDepths(device, planarInput().input, { signal: abort.signal });
    } catch (error) {
      cancelled = error.name === 'AbortError';
    }
    const selected = await completePlanarDepths(device, planarInput().input, {
      sources: [[1], [], []],
    });
    const limit = device.maxKernelElements;
    let boundedFilled;
    Object.defineProperty(device, 'maxKernelElements', { value: 64, configurable: true });
    try {
      boundedFilled = (await completePlanarDepths(device, planarInput().input)).report.filled;
    } finally {
      Object.defineProperty(device, 'maxKernelElements', { value: limit, configurable: true });
    }
    return {
      results,
      budgetRejected,
      cancelled,
      selectedFilled: selected.report.filled,
      boundedFilled,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}

/** Public reconstruction wiring, bypassing unrelated stereo search with frozen filtered maps. */
export async function planarMvsCase({ ShadeDevice, reconstructDense, planMvsViews }, backend) {
  const { input, truth } = planarInput();
  const views = input.views.map((v, id) => ({
    id,
    width: 16,
    height: 16,
    pose: { R: v.rotation, t: v.center.map((x) => -x) },
    camera: { fx: v.fx, fy: v.fy, cx: v.cx, cy: v.cy },
  }));
  const landmarks = [];
  for (const y of [3, 6, 9, 12])
    for (const x of [5, 7, 9, 11]) {
      const z = truth[0][y * 16 + x];
      const xyz = [input.views[0].center[0] + ((x - 7.5) / 32) * z, ((y - 7.5) / 32) * z, z];
      landmarks.push({
        xyz,
        observations: views.map((v) => ({
          viewId: v.id,
          pixel: [(32 * (xyz[0] + v.pose.t[0])) / z + 7.5, (32 * xyz[1]) / z + 7.5],
        })),
      });
    }
  const scene = { views, landmarks },
    plan = planMvsViews(scene);
  const images = new Map(
    input.views.map((v, id) => [
      id,
      {
        width: 16,
        height: 16,
        rgba: new Uint8ClampedArray(v.rgba.buffer.slice(0)),
      },
    ]),
  );
  const depthMaps = {
    depth: plan.packed.map((id) =>
      Float32Array.from(input.views[id].depth, (z) => z / plan.normalization.scale),
    ),
    normal: plan.packed.map((id) => input.views[id].normal),
    confidence: plan.packed.map((id) => input.views[id].confidence),
  };
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  try {
    let maps;
    const result = await reconstructDense(device, scene, images, {
      fusion: 'tsdf',
      planarDepth: true,
      depthMaps,
      tsdf: { levels: 1, voxelScale: 1, truncation: 3, visibility: 'view-weighted' },
      onDepthMaps: (m) => {
        maps = m;
      },
    });
    let maxSurfaceError = 0;
    for (let k = 0; k < result.points.count; k++) {
      const p = result.points.positions.subarray(3 * k, 3 * k + 3);
      maxSurfaceError = Math.max(
        maxSurfaceError,
        Math.abs(p[2] - 0.15 * p[0] - 0.08 * p[1] - 2) / Math.hypot(0.15, 0.08, -1),
      );
    }
    return {
      filled: result.planar?.filled,
      points: result.points.count,
      maxSurfaceError,
      remainingHoles: maps.depth[0].filter((d) => !(d > 0)).length,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
