import type { ShadeDevice } from '../../device/gpu/device.ts';
import { chunkedPass } from '../../mvs/gpu/cloud-clean.ts';
import { packRgba } from '../../features/gpu/sift.ts';
import { slicAssign, slicSeeds, slicToLab, slicUpdate } from './slic.shade.ts';

/** Options of `slic` and `slicGpu`, with the defaults of scikit-image's `segmentation.slic`. */
export interface SlicOptions {
  /** Approximate number of superpixels; the grid has `round(W / S) x round(H / S)` cells. Default 100. */
  readonly nSegments?: number;
  /** Weight of the pixel distance against the color distance in CIELAB. Default 10. */
  readonly compactness?: number;
  /** Number of k-means iterations. Default 10. */
  readonly maxIterations?: number;
  /** Merge the segments smaller than `minSizeFactor` of the mean size and keep one component per label. Default true. */
  readonly enforceConnectivity?: boolean;
  /** Smallest segment kept by connectivity enforcement, as a fraction of the mean size. Default 0.5. */
  readonly minSizeFactor?: number;
}

export interface SlicResult {
  readonly width: number;
  readonly height: number;
  /** Label of each pixel, row-major, in `0 .. count - 1`. */
  readonly labels: Uint32Array;
  /** Number of labels. */
  readonly count: number;
}

/** The resolved grid of one SLIC run. */
export interface SlicPlan {
  readonly width: number;
  readonly height: number;
  readonly gridX: number;
  readonly gridY: number;
  readonly clusters: number;
  /** `(compactness / S)^2`, with `S = sqrt(pixels / clusters)`. */
  readonly ratio: number;
  readonly maxIterations: number;
  readonly enforceConnectivity: boolean;
  readonly minSize: number;
}

export function planSlic(options: SlicOptions, width: number, height: number): SlicPlan {
  const nSegments = options.nSegments ?? 100,
    compactness = options.compactness ?? 10,
    maxIterations = options.maxIterations ?? 10,
    minSizeFactor = options.minSizeFactor ?? 0.5;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    !Number.isSafeInteger(nSegments) ||
    nSegments < 1 ||
    !(compactness > 0 && Number.isFinite(compactness)) ||
    !Number.isSafeInteger(maxIterations) ||
    maxIterations < 0 ||
    maxIterations > 1000 ||
    !(minSizeFactor >= 0 && minSizeFactor <= 1) ||
    (options.enforceConnectivity !== undefined && typeof options.enforceConnectivity !== 'boolean')
  )
    throw new RangeError('Invalid SLIC options');
  const step = Math.sqrt((width * height) / nSegments),
    gridX = Math.min(width, Math.max(1, Math.round(width / step))),
    gridY = Math.min(height, Math.max(1, Math.round(height / step))),
    clusters = gridX * gridY,
    actualStep = Math.sqrt((width * height) / clusters);
  return {
    width,
    height,
    gridX,
    gridY,
    clusters,
    ratio: (compactness / actualStep) ** 2,
    maxIterations,
    enforceConnectivity: options.enforceConnectivity ?? true,
    minSize: Math.floor((minSizeFactor * width * height) / clusters),
  };
}

const validRgba = (rgba: ArrayLike<number>, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 1 &&
  height >= 1 &&
  rgba.length === width * height * 4;

/** SLIC on the device. `rgba` is 8-bit sRGB; the result is read back and connectivity is enforced on the CPU. */
export function slicGpu(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: SlicOptions = {},
): Promise<SlicResult> {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for SLIC');
  const plan = planSlic(options, width, height),
    pixels = width * height;
  return device.enqueue(async () => {
    const slic = {
      width,
      height,
      gridX: plan.gridX,
      gridY: plan.gridY,
      ratio: plan.ratio,
    };
    const input = device.allocateU32(pixels, packRgba(rgba, pixels)),
      lab = device.allocate(3 * pixels),
      centers = device.allocate(5 * plan.clusters),
      labels = device.allocateU32(pixels);
    try {
      function* steps() {
        yield* chunkedPass(device, 'slicLab', 'slicToLab', pixels, (slicRange, groups) =>
          slicToLab({ slicRange, slicRgba: input, slicLab: lab }, groups),
        );
        yield* chunkedPass(device, 'slicSeeds', 'slicSeeds', plan.clusters, (slicRange, groups) =>
          slicSeeds({ slic, slicRange, slicLabRead: lab, slicCenters: centers }, groups),
        );
        for (let iteration = 0; iteration < plan.maxIterations; iteration++) {
          yield* chunkedPass(device, 'slicAssign', 'slicAssign', pixels, (slicRange, groups) =>
            slicAssign(
              {
                slic,
                slicRange,
                slicLabRead: lab,
                slicCentersRead: centers,
                slicLabels: labels,
              },
              groups,
            ),
          );
          yield* chunkedPass(
            device,
            'slicUpdate',
            'slicUpdate',
            plan.clusters,
            (slicRange, groups) =>
              slicUpdate(
                {
                  slic,
                  slicRange,
                  slicLabRead: lab,
                  slicLabelsRead: labels,
                  slicCenters: centers,
                },
                groups,
              ),
          );
        }
      }
      await device.planSteps(steps());
      // With no iteration the labels are never assigned; one assignment gives the seed cells.
      if (plan.maxIterations === 0)
        await device.planSteps(
          chunkedPass(device, 'slicAssign', 'slicAssign', pixels, (slicRange, groups) =>
            slicAssign(
              {
                slic,
                slicRange,
                slicLabRead: lab,
                slicCentersRead: centers,
                slicLabels: labels,
              },
              groups,
            ),
          ),
        );
      const raw = await device.read(labels, pixels);
      return finish(plan, raw);
    } finally {
      for (const buffer of [input, labels, lab, centers]) device.recycle(buffer);
    }
  });
}

/**
 * The CPU reference of `slicGpu`: the same grid, seeds, 3 x 3-cell assignment and gathered update
 * in double precision, then the same connectivity enforcement.
 */
export function slic(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  options: SlicOptions = {},
): SlicResult {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for SLIC');
  const plan = planSlic(options, width, height),
    { gridX, gridY, clusters, ratio } = plan,
    pixels = width * height,
    lab = new Float64Array(3 * pixels);
  for (let i = 0; i < pixels; i++)
    rgbToLab(rgba[4 * i]!, rgba[4 * i + 1]!, rgba[4 * i + 2]!, lab, 3 * i);
  const centers = new Float64Array(5 * clusters),
    labels = new Uint32Array(pixels),
    gradient = (x: number, y: number) => {
      let g = 0;
      for (let c = 0; c < 3; c++) {
        const dx = lab[3 * (y * width + x + 1) + c]! - lab[3 * (y * width + x - 1) + c]!,
          dy = lab[3 * ((y + 1) * width + x) + c]! - lab[3 * ((y - 1) * width + x) + c]!;
        g += dx * dx + dy * dy;
      }
      return g;
    };
  for (let k = 0; k < clusters; k++) {
    const cx = Math.floor(((2 * (k % gridX) + 1) * width) / (2 * gridX)),
      cy = Math.floor(((2 * Math.floor(k / gridX) + 1) * height) / (2 * gridY));
    let bx = cx,
      by = cy;
    if (width >= 3 && height >= 3) {
      let best = Infinity;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const x = cx + dx,
            y = cy + dy;
          if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
            const g = gradient(x, y);
            if (g < best) {
              best = g;
              bx = x;
              by = y;
            }
          }
        }
    }
    const at = 3 * (by * width + bx);
    centers.set([lab[at]!, lab[at + 1]!, lab[at + 2]!, bx, by], 5 * k);
  }
  const assign = () => {
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const i = y * width + x,
          cellX = Math.floor((x * gridX) / width),
          cellY = Math.floor((y * gridY) / height);
        let best = Infinity,
          label = cellY * gridX + cellX;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const gx = cellX + dx,
              gy = cellY + dy;
            if (gx < 0 || gy < 0 || gx >= gridX || gy >= gridY) continue;
            const k = gy * gridX + gx,
              dl = lab[3 * i]! - centers[5 * k]!,
              da = lab[3 * i + 1]! - centers[5 * k + 1]!,
              db = lab[3 * i + 2]! - centers[5 * k + 2]!,
              ex = x - centers[5 * k + 3]!,
              ey = y - centers[5 * k + 4]!,
              d = dl * dl + da * da + db * db + ratio * (ex * ex + ey * ey);
            if (d < best) {
              best = d;
              label = k;
            }
          }
        labels[i] = label;
      }
  };
  for (let iteration = 0; iteration < plan.maxIterations; iteration++) {
    assign();
    for (let k = 0; k < clusters; k++) {
      const gx = k % gridX,
        gy = Math.floor(k / gridX),
        x0 = gx >= 1 ? Math.ceil(((gx - 1) * width) / gridX) : 0,
        x1 = Math.min(Math.ceil(((gx + 2) * width) / gridX), width),
        y0 = gy >= 1 ? Math.ceil(((gy - 1) * height) / gridY) : 0,
        y1 = Math.min(Math.ceil(((gy + 2) * height) / gridY), height);
      let sl = 0,
        sa = 0,
        sb = 0,
        sx = 0,
        sy = 0,
        count = 0;
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++) {
          const i = y * width + x;
          if (labels[i] !== k) continue;
          sl += lab[3 * i]!;
          sa += lab[3 * i + 1]!;
          sb += lab[3 * i + 2]!;
          sx += x;
          sy += y;
          count++;
        }
      if (count > 0)
        centers.set([sl / count, sa / count, sb / count, sx / count, sy / count], 5 * k);
    }
  }
  if (plan.maxIterations === 0) assign();
  return finish(plan, labels);
}

/** sRGB bytes to CIELAB (D65, L in [0, 100]), written at `out[at .. at + 2]`. */
function rgbToLab(r8: number, g8: number, b8: number, out: Float64Array, at: number): void {
  const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4),
    f = (t: number) => (t > 0.008856451679035631 ? Math.cbrt(t) : 7.787037037037037 * t + 16 / 116),
    r = linear(r8 / 255),
    g = linear(g8 / 255),
    b = linear(b8 / 255),
    fx = f((0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.95047),
    fy = f(0.212671 * r + 0.71516 * g + 0.072169 * b),
    fz = f((0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883);
  out[at] = 116 * fy - 16;
  out[at + 1] = 500 * (fx - fy);
  out[at + 2] = 200 * (fy - fz);
}

function finish(plan: SlicPlan, raw: Uint32Array): SlicResult {
  const { width, height } = plan;
  return plan.enforceConnectivity
    ? enforceConnectivity(raw, width, height, plan.minSize)
    : compact(raw, width, height);
}

/** Relabel to `0 .. count - 1` in order of first appearance. */
function compact(raw: Uint32Array, width: number, height: number): SlicResult {
  const map = new Map<number, number>(),
    labels = new Uint32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    let id = map.get(raw[i]!);
    if (id === undefined) map.set(raw[i]!, (id = map.size));
    labels[i] = id;
  }
  return { width, height, labels, count: map.size };
}

/**
 * Connectivity enforcement after scikit-image's `_enforce_label_connectivity_cython`: split every
 * label into its 4-connected components, merge a component smaller than `minSize` into the
 * component left of or above its first pixel (in scan order) or, for the first pixel of the image,
 * right of or below it, and relabel in scan order. A merge joins two adjacent components, so
 * every final label is one 4-connected component.
 */
function enforceConnectivity(
  raw: Uint32Array,
  width: number,
  height: number,
  minSize: number,
): SlicResult {
  const pixels = raw.length,
    component = new Int32Array(pixels).fill(-1),
    sizes: number[] = [],
    firsts: number[] = [],
    stack = new Int32Array(pixels);
  for (let start = 0; start < pixels; start++) {
    if (component[start]! >= 0) continue;
    const id = sizes.length;
    let top = 0,
      size = 0;
    stack[top++] = start;
    component[start] = id;
    while (top > 0) {
      const p = stack[--top]!,
        x = p % width,
        y = (p - x) / width;
      size++;
      const visit = (q: number) => {
        if (component[q]! < 0 && raw[q] === raw[p]) {
          component[q] = id;
          stack[top++] = q;
        }
      };
      if (x > 0) visit(p - 1);
      if (x < width - 1) visit(p + 1);
      if (y > 0) visit(p - width);
      if (y < height - 1) visit(p + width);
    }
    sizes.push(size);
    firsts.push(start);
  }
  const parent = Array.from(sizes, (_, i) => i),
    find = (i: number): number => {
      while (parent[i] !== i) i = parent[i] = parent[parent[i]!]!;
      return i;
    };
  for (let id = 0; id < sizes.length; id++) {
    if (sizes[id]! >= minSize) continue;
    const p = firsts[id]!,
      x = p % width,
      y = (p - x) / width,
      neighbour =
        x > 0 ? p - 1 : y > 0 ? p - width : x < width - 1 ? p + 1 : y < height - 1 ? p + width : -1;
    if (neighbour < 0) continue;
    const into = find(component[neighbour]!),
      from = find(id);
    if (into === from) continue;
    parent[from] = into;
  }
  const finalId = new Map<number, number>(),
    labels = new Uint32Array(pixels);
  for (let i = 0; i < pixels; i++) {
    const root = find(component[i]!);
    let id = finalId.get(root);
    if (id === undefined) finalId.set(root, (id = finalId.size));
    labels[i] = id;
  }
  return { width, height, labels, count: finalId.size };
}
