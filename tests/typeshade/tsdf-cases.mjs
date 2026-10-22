/**
 * The TSDF fusion's cases (`docs/dense-mvs.md`, section 10.4): the exact depth and normal maps of
 * the synthetic scene of 7.1 (`mvs-cases.mjs`) as the input of `fuseTsdf` and `fuseTsdfGpu`, in
 * the world frame, with a floater: a patch of one view whose depth is 30 % short, a surface in
 * front of the true one that only that view sees and the other views see through.
 */
import { MVS_HEIGHT, MVS_WIDTH, mvsScene, nearestSurface } from './mvs-cases.mjs';

/** The octahedral map of a unit vector as `pack2x16snorm` packs it. */
function encodeNormal(n) {
  const s = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  let px = n[0] / s,
    py = n[1] / s;
  if (n[2] < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1),
      oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  const snorm = (v) => Math.floor(Math.min(Math.max(v, -1), 1) * 32767 + 0.5) & 0xffff;
  return (snorm(px) | (snorm(py) << 16)) >>> 0;
}

/**
 * The TSDF options of the synthetic tests. The scene is 128 × 96 pixels, so they take the voxel of
 * one footprint and a truncation of 4 voxels. The defaults (voxel of 2 footprints, truncation 3)
 * are the measurement of `pipes` at 778 × 518 (the step 3 record).
 */
export const TSDF_SYNTHETIC = { voxelScale: 1, truncation: 4 };

/** The floater's patch: 16 × 16 pixels of view 0, and the factor of its depth. */
export const FLOATER = { view: 0, x0: 40, y0: 30, size: 16, factor: 0.7 };

/**
 * A plane at z = 2 seen twice. Two additional views can have an erroneous, farther patch
 * whose integration weight is zero: zero confidence, grazing normal, or rounding to zero.
 * Those observations must not erase the central part of the trusted plane.
 */
export function untrustedCarvingInput(kind = 'none') {
  const width = 16,
    height = 16,
    pixels = width * height;
  const view = (untrusted) => {
    const depth = new Float32Array(pixels).fill(untrusted ? 0 : 2);
    const confidence = new Float32Array(pixels).fill(
      untrusted ? (kind === 'grazing' ? 1 : kind === 'rounded-zero' ? 0.000001 : 0) : 1,
    );
    if (untrusted)
      for (let y = 4; y < 12; y++) for (let x = 4; x < 12; x++) depth[y * width + x] = 4;
    return {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      center: [0, 0, 0],
      fx: 32,
      fy: 32,
      cx: 7.5,
      cy: 7.5,
      depth,
      confidence,
      // The trusted normal faces the camera (-z); +x grazes every ray of the bad patch.
      normal: new Uint32Array(pixels).fill(
        untrusted && kind === 'grazing' ? 0x00007fff : 0x7fff7fff,
      ),
      rgba: new Uint32Array(pixels).fill(0xffffffff),
    };
  };
  return {
    width,
    height,
    views: [view(false), view(false), ...(kind === 'none' ? [] : [view(true), view(true)])],
  };
}

/** Small visibility regression shared by the CPU tier and WebGPU. */
export async function weightedVisibilityCase({ ShadeDevice, fuseTsdfGpu }, backend) {
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  const options = { levels: 1, voxelScale: 1, truncation: 3, visibility: 'view-weighted' };
  const plain = (r) => ({
    count: r.count,
    positions: [...r.positions],
    normals: [...r.normals],
    colors: [...r.colors],
    weights: [...r.weights],
    free: [...r.free],
    freeWeights: [...r.freeWeights],
    tallies: [...r.tallies],
  });
  try {
    const baseline = plain(await fuseTsdfGpu(device, untrustedCarvingInput(), options));
    const variants = [];
    for (const confidence of [0, 0.01, 0.1, 0.3]) {
      const input = untrustedCarvingInput('zero-confidence');
      input.views[2].center = [-0.15, 0, 0];
      input.views[3].center = [0.15, 0, 0];
      for (const view of input.views.slice(2)) view.confidence.fill(confidence);
      const whole = await fuseTsdfGpu(device, input, options);
      const uncarved = await fuseTsdfGpu(device, input, { ...options, carveRatio: 1e9 });
      const boxes = await fuseTsdfGpu(device, input, {
        ...options,
        maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
      });
      variants.push({
        confidence,
        whole: plain(whole),
        uncarved: plain(uncarved),
        boxes: plain(boxes),
        passes: boxes.tsdf.passes,
      });
    }
    // A single contradictory camera can cast many rays, but must not erase the trusted plane.
    const input = untrustedCarvingInput('zero-confidence');
    input.views.length = 3;
    input.views[2].confidence.fill(1);
    const single = plain(await fuseTsdfGpu(device, input, options));
    const repeated = untrustedCarvingInput('zero-confidence');
    repeated.views.length = 3;
    repeated.views[2].center = [0.15, 0, 0];
    repeated.views[2].confidence.fill(1);
    for (let k = 0; k < 10; k++) repeated.views.push({ ...repeated.views[2] });
    const correlated = plain(await fuseTsdfGpu(device, repeated, options));
    return { baseline, variants, single, correlated, liveBytes: device.statistics.liveBytes };
  } finally {
    await device.dispose();
  }
}

/**
 * The input of the fusion: every view of the scene with its exact depth and normal, a confidence
 * that falls from 1 at the image center to 0.5 at its corners, and its RGBA. `floater` adds the
 * patch of `FLOATER`.
 */
export function tsdfInput({ floater = true, scene = mvsScene() } = {}) {
  const W = MVS_WIDTH,
    H = MVS_HEIGHT,
    P = W * H;
  const views = scene.views.map((view, v) => {
    const { R, t } = view.pose;
    const center = [0, 1, 2].map((i) => -(R[i] * t[0] + R[i + 3] * t[1] + R[i + 6] * t[2]));
    const truth = scene.truth[v];
    const depth = new Float32Array(P),
      normal = new Uint32Array(P),
      confidence = new Float32Array(P);
    for (let i = 0; i < P; i++) {
      if (!(truth.depth[i] > 0)) continue;
      depth[i] = truth.depth[i];
      normal[i] = encodeNormal([
        truth.normal[3 * i],
        truth.normal[3 * i + 1],
        truth.normal[3 * i + 2],
      ]);
      const dx = (i % W) / (W - 1) - 0.5,
        dy = Math.floor(i / W) / (H - 1) - 0.5;
      confidence[i] = 1 - Math.hypot(dx, dy) / Math.SQRT2;
    }
    if (floater && v === FLOATER.view)
      for (let y = FLOATER.y0; y < FLOATER.y0 + FLOATER.size; y++)
        for (let x = FLOATER.x0; x < FLOATER.x0 + FLOATER.size; x++)
          depth[y * W + x] *= FLOATER.factor;
    const rgba = scene.images.get(view.id).rgba;
    const packed = new Uint32Array(P);
    for (let i = 0; i < P; i++)
      packed[i] =
        (rgba[4 * i] |
          (rgba[4 * i + 1] << 8) |
          (rgba[4 * i + 2] << 16) |
          (rgba[4 * i + 3] << 24)) >>>
        0;
    return {
      rotation: R,
      center,
      fx: view.camera.fx,
      fy: view.camera.fy,
      cx: view.camera.cx,
      cy: view.camera.cy,
      k1: view.camera.k1 ?? 0,
      k2: view.camera.k2 ?? 0,
      depth,
      normal,
      confidence,
      rgba: packed,
    };
  });
  return { width: W, height: H, views };
}

/** The share of points within `bound` of the true surface, and the points farther than `far`. */
export function surfaceMetrics(points, bound = 0.01, far = 0.1) {
  let within = 0,
    farther = 0;
  for (let k = 0; k < points.count; k++) {
    const p = [points.positions[3 * k], points.positions[3 * k + 1], points.positions[3 * k + 2]];
    const { distance } = nearestSurface(p);
    if (distance <= bound) within++;
    if (distance > far) farther++;
  }
  return { within: within / Math.max(points.count, 1), farther };
}

/**
 * Two point sets agree: the share of the points of `a` with a point of `b` within `tolerance`, by
 * a grid of cells of that size.
 */
export function matchedShare(a, b, tolerance) {
  const cells = new Map();
  const key = (x, y, z) => `${x},${y},${z}`;
  for (let k = 0; k < b.count; k++) {
    const c = [0, 1, 2].map((i) => Math.floor(b.positions[3 * k + i] / tolerance));
    const id = key(...c);
    if (!cells.has(id)) cells.set(id, []);
    cells.get(id).push(k);
  }
  let matched = 0;
  for (let k = 0; k < a.count; k++) {
    const p = [0, 1, 2].map((i) => a.positions[3 * k + i]);
    const c = p.map((v) => Math.floor(v / tolerance));
    let found = false;
    for (let dx = -1; dx <= 1 && !found; dx++)
      for (let dy = -1; dy <= 1 && !found; dy++)
        for (let dz = -1; dz <= 1 && !found; dz++)
          for (const j of cells.get(key(c[0] + dx, c[1] + dy, c[2] + dz)) ?? [])
            if (
              Math.hypot(
                p[0] - b.positions[3 * j],
                p[1] - b.positions[3 * j + 1],
                p[2] - b.positions[3 * j + 2],
              ) <= tolerance
            ) {
              found = true;
              break;
            }
    if (found) matched++;
  }
  return matched / Math.max(a.count, 1);
}

/**
 * The pairs (index in `a`, index in `b`) of points within `tolerance` of each other: each point of
 * `a` with its nearest point of `b` in the cells around it.
 */
export function pairedPoints(a, b, tolerance) {
  const cells = new Map();
  const key = (x, y, z) => `${x},${y},${z}`;
  for (let k = 0; k < b.count; k++) {
    const id = key(...[0, 1, 2].map((i) => Math.floor(b.positions[3 * k + i] / tolerance)));
    if (!cells.has(id)) cells.set(id, []);
    cells.get(id).push(k);
  }
  const pairs = [];
  for (let k = 0; k < a.count; k++) {
    const p = [0, 1, 2].map((i) => a.positions[3 * k + i]);
    const c = p.map((v) => Math.floor(v / tolerance));
    let best = -1,
      bestDistance = tolerance;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++)
          for (const j of cells.get(key(c[0] + dx, c[1] + dy, c[2] + dz)) ?? []) {
            const d = Math.hypot(
              p[0] - b.positions[3 * j],
              p[1] - b.positions[3 * j + 1],
              p[2] - b.positions[3 * j + 2],
            );
            if (d <= bestDistance) {
              best = j;
              bestDistance = d;
            }
          }
    if (best >= 0) pairs.push([k, best]);
  }
  return pairs;
}
