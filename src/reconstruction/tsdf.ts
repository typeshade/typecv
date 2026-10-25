/**
 * The TSDF fusion of filtered depth maps (`docs/dense-mvs.md`, section 10.4): the plan of the
 * volume that both tiers share, the boxes that cut a volume to a block budget, and the CPU
 * reference of the kernels of `src/gpu/tsdf.shade.ts` (`fuseTsdf`).
 *
 * The volume is a sparse set of blocks of 8 × 8 × 8 voxels at `levels` levels. Level L has voxel
 * size v_L = v_0 · 2^L. A depth sample at depth d of a view with focal f has the footprint d / f.
 * It goes to the least level with v_L ≥ `voxelScale` · d / f (the last level takes the rest), so a
 * far region is integrated at a coarser level and is not lost. v_0 is `voxelScale` times the
 * `percentile` of the footprints of all samples. The truncation is μ_L = `truncation` · v_L.
 *
 * Details that section 10.4 leaves open, with the choice made here:
 * - Every level's voxel grid has one origin, the least corner of all levels' boxes, so that a block
 *   of level L lies in exactly one block of each coarser level. Each level has its own box of
 *   blocks, the samples of that level with a margin of 32 voxels, and a block outside it is not
 *   stored. The key of a block is its index in the boxes of all levels, below 2^32 − 1. When the
 *   boxes need more keys, v_0 doubles until they do not.
 * - The signed distance of a voxel is the distance of its center to the sample's tangent plane
 *   (the plane through the sample's point with the sample's normal), positive in front. The band
 *   [d − μ, d + μ] is the ray's segment where that distance is within ±μ along the ray, with the
 *   cosine between the normal and the ray clamped to at least 0.15.
 * - The weight of a sample is its confidence (10.3) times that cosine, 0 below a cosine of 0.15 (the
 *   confidence's view-normal term is 0 there). A voxel's sums are fixed point: w · 4096 rounded,
 *   w · sdf / μ · 4096 rounded, and (w · 4096 rounded) · channel / 16 for each color channel.
 *   A sample whose rounded weight is 0 neither integrates nor carves existing surfaces.
 * - A voxel is kept when its weight is at least `minWeight`, its tally counts at least `minViews`
 *   views, and its free-space count is at most `carveRatio` times its weight (the deletion of `tsdfCarve`). A carving ray stops μ' before the
 *   sample along the ray, μ' the larger truncation of the sample's level and the voxel's level.
 * - The extraction emits at a sign change toward +x, +y and +z between kept voxels whose mean
 *   sdf / μ differ by at most 1. A larger jump is the edge of two truncated sides, not a surface.
 * - A coarse voxel emits nothing when the finer level keeps the voxel at its center.
 * - The points are sorted by x, then y, then z, so both tiers give one order.
 */

export interface TsdfView {
  /** World to camera rotation, row-major. */
  readonly rotation: ArrayLike<number>;
  /** The camera center in the world frame. */
  readonly center: ArrayLike<number>;
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
  /** Radial distortion of the RGBA, for the color sample (3.11). Default 0. */
  readonly k1?: number;
  readonly k2?: number;
  /** The squared radius past which the radial model folds. Default 1e30. */
  readonly radialLimit?: number;
  /** Optical-axis depth per pixel, 0 for "no depth". */
  readonly depth: Float32Array;
  /** Octahedral unit normal per pixel in the camera frame (`pack2x16snorm`), facing the camera. */
  readonly normal: Uint32Array;
  /** The confidence of each depth (section 10.3). Default 1 for every pixel. */
  readonly confidence?: Float32Array;
  /** Packed RGBA per pixel of the distorted image. */
  readonly rgba: Uint32Array;
}

export interface TsdfInput {
  readonly width: number;
  readonly height: number;
  readonly views: readonly TsdfView[];
}

/** Cameras within one pixel of translational parallax share visibility evidence, even if rotated. */
export function tsdfVisibilityGroups(input: TsdfInput, exclude?: number): number[][] {
  const P = input.width * input.height;
  const stats = input.views.map((v) => {
    const depths: number[] = [];
    for (let i = 0; i < P; i += Math.max(1, Math.ceil(P / 256))) {
      if (!(v.depth[i]! > 0)) continue;
      depths.push(v.depth[i]!);
    }
    depths.sort((a, b) => a - b);
    return (depths[(depths.length - 1) >> 1] ?? 0) / Math.max(v.fx, v.fy);
  });
  const groups: number[][] = [];
  for (const [i, v] of input.views.entries()) {
    const same = groups.find(([j]) => {
      const other = input.views[j]!;
      const distance = Math.hypot(...[0, 1, 2].map((a) => v.center[a]! - other.center[a]!));
      return distance <= Math.min(stats[i]!, stats[j]!);
    });
    if (!same) groups.push([i]);
    else same.push(i);
  }
  return groups.filter((g) => exclude === undefined || !g.includes(exclude));
}

export interface TsdfOptions {
  /** Legacy ray counts, or confidence-weighted votes capped per spatial camera group and voxel. */
  readonly visibility?: 'ray-count' | 'view-weighted';
  /** v_L is at least this times a sample's footprint d / f. Default 2 (the step 3 record). */
  readonly voxelScale?: number;
  /** The levels, 1 to 8. Default 4. */
  readonly levels?: number;
  /** μ_L in voxels of the level. Default 3 (the step 3 record). */
  readonly truncation?: number;
  /** The least weight of a kept voxel (one sample of confidence 1 facing the camera weighs 1). Default 1. */
  readonly minWeight?: number;
  /** A voxel whose free-space count is over this times its weight is deleted. Default 1. */
  readonly carveRatio?: number;
  /** The percentile of the footprints that sets v_0, in (0, 1). Default 0.05. */
  readonly percentile?: number;
  /**
   * The least count of views that integrated a kept voxel (the tally, section 10.5), 1 to 255.
   * Default 1: every voxel with weight.
   */
  readonly minViews?: number;
  /**
   * A carving ray walks level L only when the sample's level − L ≤ `carveSpan` (section 10.5).
   * Default 7: every level.
   */
  readonly carveSpan?: number;
}

export const TSDF_DEFAULTS = {
  visibility: 'ray-count',
  voxelScale: 2,
  levels: 4,
  truncation: 3,
  minWeight: 1,
  carveRatio: 1,
  percentile: 0.05,
  minViews: 1,
  carveSpan: 7,
} as const;

/** Fixed point of a weight: 4096 for 1 (`WEIGHT_SCALE` of the kernels). */
export const TSDF_WEIGHT_SCALE = 4096;
/** Packed view-weighted evidence: low 20 bits are weights in units of 1/256, high 12 bits views. */
export const TSDF_CARVE_SCALE = 256;
const CARVE_MASK = 0xfffff;
/** The least cosine between a sample's normal and its ray (`MIN_COSINE` of the kernels). */
export const TSDF_MIN_COSINE = 0.15;
/** u32 words of one voxel, and voxels of one block. */
export const TSDF_VOXEL_WORDS = 7;
/** u32 words of one extracted point in `tsdfPointData`: normal, color, weight, free, tally. */
export const TSDF_POINT_WORDS = 5;
export const TSDF_BLOCK_VOXELS = 512;
/** The words of a block's voxels: 3584, 14 KiB. */
export const TSDF_BLOCK_WORDS = TSDF_VOXEL_WORDS * TSDF_BLOCK_VOXELS;
/** The table's header words (`HEADER` of the kernels). */
export const TSDF_HEADER = 8;
const EMPTY = 0xffffffff;
const MAX_STEPS = 16384;
const FAR = 3.0e38;
const f32 = Math.fround;

/** The plan of the volume: what both tiers bind. */
export interface TsdfVolume {
  /** The origin of every level's grid, f32. */
  readonly origin: readonly [number, number, number];
  /** v_0 and its f32 inverse. */
  readonly voxel: number;
  readonly inverse: number;
  readonly levels: number;
  readonly truncation: number;
  /** 8 words per level: lo x, y, z (i32 bits), n x, y, z, the first key, 0. */
  readonly grid: Uint32Array;
  /** The keys of all levels. */
  readonly keys: number;
  /** Per view: `voxelScale` / ((fx + fy) / 2) in f32. A depth d goes to the least level with
   *  d · footprint ≤ v_L. */
  readonly footprints: Float32Array;
  /** The samples with a depth over all views. */
  readonly samples: number;
}

export interface TsdfPoints {
  readonly count: number;
  /** x, y, z per point, in the frame of the input. */
  readonly positions: Float32Array;
  /** The octahedral unit normal per point (`pack2x16snorm`), in the frame of the input. */
  readonly normals: Uint32Array;
  /** Packed RGBA per point. */
  readonly colors: Uint32Array;
  /** The weight (fixed point, `TSDF_WEIGHT_SCALE` for 1) of the voxel that emitted each point. */
  readonly weights: Uint32Array;
  /** Contradictory rays (legacy) or spatial camera groups (view-weighted) at that voxel. */
  readonly free: Uint32Array;
  /** Confidence-weighted contradictory camera groups, when visibility is 'view-weighted'. */
  readonly freeWeights?: Float32Array;
  /** The tally of that voxel: `tsdfSamples`, `tsdfViews` read it (section 10.5). */
  readonly tallies: Uint32Array;
}

/** The samples a tally counts (saturated at 4095). */
export function tsdfSamples(tally: number): number {
  return tally & 4095;
}
/** The views a tally counts (saturated at 255). */
export function tsdfViews(tally: number): number {
  return (tally >>> 12) & 255;
}
/** The tally after one more sample of view `view` (`tally` of the kernels). */
export function tsdfTally(tally: number, view: number): number {
  const mark = (view % 4095) + 1;
  const samples = Math.min((tally & 4095) + 1, 4095);
  let views = (tally >>> 12) & 255;
  if (tally >>> 20 !== mark) views = Math.min(views + 1, 255);
  return (samples | (views << 12) | (mark << 20)) >>> 0;
}

export function checkedTsdfOptions(options: TsdfOptions = {}): Required<TsdfOptions> {
  const o = { ...TSDF_DEFAULTS, ...options };
  if (o.visibility !== 'ray-count' && o.visibility !== 'view-weighted')
    throw new RangeError('visibility must be ray-count or view-weighted');
  if (!(o.voxelScale > 0 && Number.isFinite(o.voxelScale)))
    throw new RangeError('voxelScale must be positive');
  if (!Number.isInteger(o.levels) || o.levels < 1 || o.levels > 8)
    throw new RangeError('levels must be an integer in [1, 8]');
  if (!(o.truncation >= 1 && o.truncation <= 16))
    throw new RangeError('truncation must be in [1, 16]');
  if (!(o.minWeight >= 0 && Number.isFinite(o.minWeight)))
    throw new RangeError('minWeight must be at least 0');
  if (!(o.carveRatio > 0)) throw new RangeError('carveRatio must be positive');
  if (!(o.percentile > 0 && o.percentile < 1)) throw new RangeError('percentile must be in (0, 1)');
  if (!Number.isInteger(o.minViews) || o.minViews < 1 || o.minViews > 255)
    throw new RangeError('minViews must be an integer in [1, 255]');
  if (!Number.isInteger(o.carveSpan) || o.carveSpan < 0 || o.carveSpan > 7)
    throw new RangeError('carveSpan must be an integer in [0, 7]');
  return o;
}

/** The world ray of pixel (x, y) of a view, with optical-axis component 1. */
function pixelRay(view: TsdfView, x: number, y: number): [number, number, number] {
  const R = view.rotation;
  const a = f32(f32(x - view.cx) / view.fx),
    b = f32(f32(y - view.cy) / view.fy);
  return [
    R[0]! * a + R[3]! * b + R[6]!,
    R[1]! * a + R[4]! * b + R[7]!,
    R[2]! * a + R[5]! * b + R[8]!,
  ];
}

/** The level of a depth: the least L with fround(d · footprint) ≤ v_0 · 2^L, as the kernels do. */
function levelOf(depth: number, footprint: number, voxel: number, levels: number): number {
  const scaled = f32(depth * footprint);
  let size = voxel,
    level = 0;
  for (let k = 1; k < levels; k++) {
    if (!(scaled > size)) break;
    size *= 2;
    level = k;
  }
  return level;
}

/** The view's f32 record of `voxelScale` / focal. */
function footprintOf(view: TsdfView, voxelScale: number): number {
  return f32(voxelScale / ((view.fx + view.fy) / 2));
}

/**
 * The plan of the volume (section 10.4): v_0 from the percentile of the samples' scaled
 * footprints, and each level's box of blocks around the points of its samples. Undefined when no
 * view has a depth.
 */
export function planTsdfVolume(
  input: TsdfInput,
  options: TsdfOptions = {},
): TsdfVolume | undefined {
  const o = checkedTsdfOptions(options);
  const { width: W, height: H } = input;
  const P = W * H;
  const footprints = Float32Array.from(input.views, (v) => footprintOf(v, o.voxelScale));
  let samples = 0;
  for (const view of input.views) for (let i = 0; i < P; i++) if (view.depth[i]! > 0) samples++;
  if (!samples) return undefined;
  // The percentile of the scaled footprints, from at most 2^20 samples taken at a fixed stride.
  const stride = Math.max(1, Math.ceil(samples / 2 ** 20));
  const values: number[] = [];
  let seen = 0;
  input.views.forEach((view, v) => {
    for (let i = 0; i < P; i++) {
      const d = view.depth[i]!;
      if (!(d > 0)) continue;
      if (seen++ % stride === 0) values.push(f32(d * footprints[v]!));
    }
  });
  const sorted = Float64Array.from(values).sort();
  let voxel = f32(Math.max(sorted[Math.floor(o.percentile * (sorted.length - 1))]!, 1e-12));
  for (;;) {
    const plan = levelBoxes(input, o, footprints, voxel);
    if (plan) return { ...plan, footprints, samples };
    voxel = f32(voxel * 2);
  }
}

/** The boxes of the levels at v_0 = `voxel`, or undefined when their keys pass 2^32 − 2. */
function levelBoxes(
  input: TsdfInput,
  o: Required<TsdfOptions>,
  footprints: Float32Array,
  voxel: number,
) {
  const { width: W, height: H } = input;
  const L = o.levels;
  const min = new Float64Array(3 * L).fill(Infinity),
    max = new Float64Array(3 * L).fill(-Infinity);
  input.views.forEach((view, v) => {
    const C = view.center;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const d = view.depth[y * W + x]!;
        if (!(d > 0)) continue;
        const level = levelOf(d, footprints[v]!, voxel, L);
        const D = pixelRay(view, x, y);
        for (let a = 0; a < 3; a++) {
          const p = C[a]! + D[a]! * d;
          if (p < min[3 * level + a]!) min[3 * level + a] = p;
          if (p > max[3 * level + a]!) max[3 * level + a] = p;
        }
      }
  });
  const origin: [number, number, number] = [Infinity, Infinity, Infinity];
  for (let level = 0; level < L; level++) {
    const margin = 32 * voxel * 2 ** level;
    for (let a = 0; a < 3; a++) {
      if (min[3 * level + a]! > max[3 * level + a]!) continue;
      min[3 * level + a]! -= margin;
      max[3 * level + a]! += margin;
      origin[a] = Math.min(origin[a], min[3 * level + a]!);
    }
  }
  for (let a = 0; a < 3; a++) origin[a] = f32(origin[a]);
  const grid = new Uint32Array(8 * L);
  const words = new Int32Array(grid.buffer);
  let keys = 0;
  for (let level = 0; level < L; level++) {
    const block = 8 * voxel * 2 ** level;
    let n = 1;
    for (let a = 0; a < 3; a++) {
      if (min[3 * level + a]! > max[3 * level + a]!) {
        // A level without samples: one block at the origin.
        words[8 * level + a] = 0;
        grid[8 * level + 3 + a] = 1;
        continue;
      }
      const lo = Math.floor((min[3 * level + a]! - origin[a]) / block),
        hi = Math.floor((max[3 * level + a]! - origin[a]) / block);
      words[8 * level + a] = lo;
      grid[8 * level + 3 + a] = hi - lo + 1;
      n *= hi - lo + 1;
    }
    grid[8 * level + 6] = keys;
    keys += n;
    if (keys >= EMPTY) return undefined;
  }
  return {
    origin,
    voxel,
    inverse: f32(1 / voxel),
    levels: L,
    truncation: o.truncation,
    grid,
    keys,
  };
}

/** The key of a block, or EMPTY outside its level's box (`blockKey` of the kernels). */
export function tsdfBlockKey(
  volume: TsdfVolume,
  level: number,
  bx: number,
  by: number,
  bz: number,
) {
  const g = volume.grid,
    w = new Int32Array(g.buffer, g.byteOffset, g.length);
  const x = bx - w[8 * level]!,
    y = by - w[8 * level + 1]!,
    z = bz - w[8 * level + 2]!;
  const nx = g[8 * level + 3]!,
    ny = g[8 * level + 4]!,
    nz = g[8 * level + 5]!;
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return EMPTY;
  return g[8 * level + 6]! + x + nx * (y + ny * z);
}

/** The level and the block coordinates of a key. */
export function tsdfKeyBlock(volume: TsdfVolume, key: number): [number, number, number, number] {
  const g = volume.grid,
    w = new Int32Array(g.buffer, g.byteOffset, g.length);
  let level = 0;
  for (let k = 1; k < volume.levels; k++) if (key >= g[8 * k + 6]!) level = k;
  const nx = g[8 * level + 3]!,
    ny = g[8 * level + 4]!;
  const index = key - g[8 * level + 6]!;
  return [
    level,
    w[8 * level]! + (index % nx),
    w[8 * level + 1]! + (Math.floor(index / nx) % ny),
    w[8 * level + 2]! + Math.floor(index / (nx * ny)),
  ];
}

/** A box of a pass in cells of the coarsest block, [lo, hi) on each axis. */
export interface TsdfBox {
  readonly lo: readonly [number, number, number];
  readonly hi: readonly [number, number, number];
}
/** A box that holds every block. */
export const TSDF_WHOLE: TsdfBox = {
  lo: [-(2 ** 20), -(2 ** 20), -(2 ** 20)],
  hi: [2 ** 20, 2 ** 20, 2 ** 20],
};

/** Whether a block is in a box, or with `margin` in its margin (`blockInBox` of the kernels). */
export function tsdfInBox(
  volume: TsdfVolume,
  box: TsdfBox,
  level: number,
  b: readonly number[],
  margin: boolean,
): boolean {
  const shift = volume.levels - 1 - level;
  for (let a = 0; a < 3; a++) {
    const c = b[a]! >> shift;
    if (c >= box.lo[a]! && c < box.hi[a]!) continue;
    if (margin && b[a] === box.hi[a]! << shift) continue;
    return false;
  }
  return true;
}

/**
 * The boxes that cut the blocks of `keys` into passes of at most `maxBlocks` blocks each, margins
 * included (section 10.4, the budget rule of #160). The volume is cut along its longest axis into
 * slabs of whole coarse cells. A slab of one cell that is still too large is cut along the next
 * axis, then the third. A single coarse cell is the least unit, so a pass may hold more than
 * `maxBlocks` blocks only when one cell does.
 */
export function planTsdfBoxes(
  volume: TsdfVolume,
  keys: ArrayLike<number>,
  maxBlocks: number,
): TsdfBox[] {
  const blocks = Array.from(keys, (key) => tsdfKeyBlock(volume, key));
  if (blocks.length <= maxBlocks) return [TSDF_WHOLE];
  const lo = [Infinity, Infinity, Infinity],
    hi = [-Infinity, -Infinity, -Infinity];
  for (const [level, ...b] of blocks) {
    const shift = volume.levels - 1 - level;
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a]!, b[a]! >> shift);
      hi[a] = Math.max(hi[a]!, (b[a]! >> shift) + 1);
    }
  }
  const count = (box: TsdfBox) => {
    let n = 0;
    for (const [level, ...b] of blocks) if (tsdfInBox(volume, box, level, b, true)) n++;
    return n;
  };
  const axes = [0, 1, 2].sort((a, b) => hi[b]! - lo[b]! - (hi[a]! - lo[a]!));
  const boxes: TsdfBox[] = [];
  const split = (box: TsdfBox, depth: number) => {
    const a = axes[depth]!;
    const with_ = (from: number, to: number): TsdfBox => {
      const l = [...box.lo] as [number, number, number],
        h = [...box.hi] as [number, number, number];
      l[a] = from;
      h[a] = to;
      return { lo: l, hi: h };
    };
    let from = box.lo[a]!;
    while (from < box.hi[a]!) {
      // The largest slab from `from` that fits, found by doubling and bisection.
      let good = from,
        step = 1;
      while (good + step <= box.hi[a]! && count(with_(from, good + step)) <= maxBlocks) {
        good += step;
        step *= 2;
      }
      for (step = Math.max(1, step >> 1); step >= 1; step >>= 1)
        if (good + step <= box.hi[a]! && count(with_(from, good + step)) <= maxBlocks) good += step;
      if (good === from) {
        if (depth < 2) split(with_(from, from + 1), depth + 1);
        else boxes.push(with_(from, from + 1));
        from += 1;
      } else {
        boxes.push(with_(from, good));
        from = good;
      }
    }
  };
  split({ lo: lo as [number, number, number], hi: hi as [number, number, number] }, 0);
  return boxes.filter((box) => count(box) > 0);
}

// ---- The CPU reference --------------------------------------------------------------------------

/** `unpack2x16snorm` and the octahedral decode of the kernels. */
export function decodeTsdfNormal(word: number): [number, number, number] {
  const px = Math.max(((word << 16) >> 16) / 32767, -1),
    py = Math.max((word >> 16) / 32767, -1);
  const z = 1 - Math.abs(px) - Math.abs(py);
  const t = Math.max(-z, 0);
  const x = px + (px >= 0 ? -t : t),
    y = py + (py >= 0 ? -t : t);
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n];
}
/** The octahedral map of a unit vector into two snorm16 values, as `pack2x16snorm` rounds. */
export function encodeTsdfNormal(n: readonly number[]): number {
  const s = Math.abs(n[0]!) + Math.abs(n[1]!) + Math.abs(n[2]!);
  let px = n[0]! / s,
    py = n[1]! / s;
  if (n[2]! < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1),
      oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  const snorm = (v: number) => Math.floor(Math.min(Math.max(v, -1), 1) * 32767 + 0.5) & 0xffff;
  return (snorm(px) | (snorm(py) << 16)) >>> 0;
}

function channel(word: number, k: number): number {
  return (word >>> (8 * k)) & 255;
}

/** The color of pixel i (`pixelColor` of the kernels). */
function pixelColor(view: TsdfView, W: number, H: number, i: number): number {
  const k1 = view.k1 ?? 0,
    k2 = view.k2 ?? 0;
  if (k1 === 0 && k2 === 0) return view.rgba[i]!;
  const x = ((i % W) - view.cx) / view.fx,
    y = (Math.floor(i / W) - view.cy) / view.fy;
  const r2 = x * x + y * y;
  if (!(r2 < (view.radialLimit ?? 1e30))) return 0;
  const d = 1 + k1 * r2 + k2 * r2 * r2;
  const sx = view.fx * x * d + view.cx,
    sy = view.fy * y * d + view.cy;
  if (!(sx >= 0 && sy >= 0 && sx <= W - 1 && sy <= H - 1)) return 0;
  const fx = Math.floor(sx),
    fy = Math.floor(sy);
  const x1 = Math.min(fx + 1, W - 1),
    y1 = Math.min(fy + 1, H - 1);
  const ax = sx - fx,
    ay = sy - fy;
  const c00 = view.rgba[fy * W + fx]!,
    c10 = view.rgba[fy * W + x1]!,
    c01 = view.rgba[y1 * W + fx]!,
    c11 = view.rgba[y1 * W + x1]!;
  let color = 0;
  for (let k = 0; k < 4; k++) {
    const top = channel(c00, k) + (channel(c10, k) - channel(c00, k)) * ax,
      bottom = channel(c01, k) + (channel(c11, k) - channel(c01, k)) * ax;
    color |= Math.min(Math.floor(top + (bottom - top) * ay + 0.5), 255) << (8 * k);
  }
  return color >>> 0;
}

interface Sample {
  depth: number;
  ray: number[];
  point: number[];
  normal: number[];
  cosine: number;
  level: number;
  weight: number;
  color: number;
}

function sampleOf(
  view: TsdfView,
  footprint: number,
  volume: TsdfVolume,
  W: number,
  H: number,
  i: number,
): Sample | undefined {
  const depth = view.depth[i]!;
  if (!(depth > 0)) return undefined;
  const ray = pixelRay(view, i % W, Math.floor(i / W));
  const R = view.rotation;
  const c = decodeTsdfNormal(view.normal[i]!);
  const normal = [
    R[0]! * c[0] + R[3]! * c[1] + R[6]! * c[2],
    R[1]! * c[0] + R[4]! * c[1] + R[7]! * c[2],
    R[2]! * c[0] + R[5]! * c[1] + R[8]! * c[2],
  ];
  const facing = -(normal[0]! * ray[0] + normal[1]! * ray[1] + normal[2]! * ray[2]);
  const rawCosine = facing / Math.hypot(...ray);
  const confidence = view.confidence ? view.confidence[i]! : 1;
  const weight =
    rawCosine >= TSDF_MIN_COSINE && confidence > 0
      ? Math.floor(confidence * rawCosine * TSDF_WEIGHT_SCALE + 0.5)
      : 0;
  const C = view.center;
  return {
    depth,
    ray,
    point: [C[0]! + ray[0] * depth, C[1]! + ray[1] * depth, C[2]! + ray[2] * depth],
    normal,
    cosine: Math.max(rawCosine, TSDF_MIN_COSINE),
    level: levelOf(depth, footprint, volume.voxel, volume.levels),
    weight,
    color: pixelColor(view, W, H, i),
  };
}

/** Conservative free-space evidence from one view. All four neighbours must see past the voxel. */
function viewContradiction(
  view: TsdfView,
  volume: TsdfVolume,
  W: number,
  H: number,
  level: number,
  center: readonly number[],
  carveSpan: number,
  footprint: number,
): number {
  const R = view.rotation;
  const p = center.map((v, a) => v - view.center[a]!);
  const X = [0, 1, 2].map((a) => R[3 * a]! * p[0]! + R[3 * a + 1]! * p[1]! + R[3 * a + 2]! * p[2]!);
  if (!(X[2]! > 0)) return 0;
  const x = (view.fx * X[0]!) / X[2]! + view.cx,
    y = (view.fy * X[1]!) / X[2]! + view.cy;
  if (!(x >= 0 && y >= 0 && x < W - 1 && y < H - 1)) return 0;
  const x0 = Math.floor(x),
    y0 = Math.floor(y);
  let safeDepth = Infinity,
    vote = TSDF_CARVE_SCALE;
  for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++) {
      const i = (y0 + dy) * W + x0 + dx,
        d = view.depth[i]!;
      const confidence = Math.min(1, Math.max(0, view.confidence?.[i] ?? 1));
      if (!(d > 0 && confidence > 0)) return 0;
      const ray = [(x0 + dx - view.cx) / view.fx, (y0 + dy - view.cy) / view.fy, 1];
      const n = decodeTsdfNormal(view.normal[i]!);
      const facing = -(n[0]! * ray[0]! + n[1]! * ray[1]! + n[2]!);
      const cosine = facing / Math.hypot(...ray);
      if (!(cosine >= TSDF_MIN_COSINE)) return 0;
      const sampleLevel = levelOf(d, footprint, volume.voxel, volume.levels);
      if (level + carveSpan < sampleLevel) return 0;
      const mu = volume.truncation * volume.voxel * 2 ** Math.max(level, sampleLevel);
      safeDepth = Math.min(safeDepth, d - (mu * (2 - confidence)) / facing);
      vote = Math.min(vote, Math.floor(confidence * cosine * TSDF_CARVE_SCALE + 0.5));
    }
  const halfZ =
    0.5 * volume.voxel * 2 ** level * (Math.abs(R[6]!) + Math.abs(R[7]!) + Math.abs(R[8]!));
  return X[2]! + halfZ < safeDepth ? vote : 0;
}

function cellExit(o: ArrayLike<number>, D: readonly number[], lo: readonly number[], size: number) {
  let t = FAR;
  for (let a = 0; a < 3; a++) {
    if (D[a]! > 0) t = Math.min(t, (lo[a]! + size - o[a]!) / D[a]!);
    if (D[a]! < 0) t = Math.min(t, (lo[a]! - o[a]!) / D[a]!);
  }
  return t;
}

/**
 * The walk of the kernels: the cells of size `size` (blocks with `blocks`) that the ray o + D t
 * passes through for t in [from, to). `visit` returns false for a cell whose block is absent, and
 * the walk then jumps to the block's exit.
 */
function walk(
  volume: TsdfVolume,
  level: number,
  o: ArrayLike<number>,
  D: readonly number[],
  from: number,
  to: number,
  cellSize: number,
  visit: (x: number, y: number, z: number) => 'block' | 'skip' | 'next',
) {
  const origin = volume.origin;
  const eps = 0.01 * volume.voxel * 2 ** level;
  const inverse = 1 / cellSize;
  let t = from;
  for (let step = 0; step < MAX_STEPS && t < to; step++) {
    const c = [0, 1, 2].map((a) => Math.floor((o[a]! + D[a]! * t - origin[a]) * inverse));
    const result = visit(c[0]!, c[1]!, c[2]!);
    if (result === 'block') {
      const b = c.map((v) => v >> 3);
      const lo = b.map((v, a) => origin[a] + v * cellSize * 8);
      t = Math.max(cellExit(o, D, lo, cellSize * 8), t) + eps;
    } else {
      const lo = c.map((v, a) => origin[a] + v * cellSize);
      t = Math.max(cellExit(o, D, lo, cellSize), t) + eps;
    }
  }
}

/** The CPU state of a volume: the blocks by key and their voxels. */
class CpuVolume {
  readonly blocks = new Map<number, Int32Array>();
  readonly volume: TsdfVolume;
  constructor(volume: TsdfVolume) {
    this.volume = volume;
  }
  at(level: number, x: number, y: number, z: number): [Int32Array, number] | undefined {
    const block = this.blocks.get(tsdfBlockKey(this.volume, level, x >> 3, y >> 3, z >> 3));
    if (!block) return undefined;
    return [block, ((x & 7) + 8 * (y & 7) + 64 * (z & 7)) * TSDF_VOXEL_WORDS];
  }
}

/**
 * The CPU reference of the TSDF fusion (section 10.4) with the arithmetic of the kernels of
 * `src/gpu/tsdf.shade.ts` in double precision: the claim of every view, then the integration
 * and the free-space counts of every view, then the extraction. The points are in the frame of the
 * input, sorted by x, y, z.
 */
export function fuseTsdf(
  input: TsdfInput,
  options: TsdfOptions = {},
): TsdfPoints & {
  readonly volume?: TsdfVolume;
  readonly blocks: number;
} {
  const o = checkedTsdfOptions(options);
  const volume = planTsdfVolume(input, o);
  const empty = {
    count: 0,
    positions: new Float32Array(0),
    normals: new Uint32Array(0),
    colors: new Uint32Array(0),
    weights: new Uint32Array(0),
    free: new Uint32Array(0),
    tallies: new Uint32Array(0),
    ...(o.visibility === 'view-weighted' ? { freeWeights: new Float32Array(0) } : {}),
    blocks: 0,
  };
  if (!volume) return empty;
  const { width: W, height: H } = input;
  const P = W * H;
  const state = new CpuVolume(volume);
  const size = (level: number) => volume.voxel * 2 ** level;
  const samples = (v: number, each: (s: Sample) => void) => {
    const view = input.views[v]!;
    for (let i = 0; i < P; i++) {
      const s = sampleOf(view, volume.footprints[v]!, volume, W, H, i);
      if (s) each(s);
    }
  };
  // tsdfAllocate
  input.views.forEach((view, v) =>
    samples(v, (s) => {
      if (s.weight === 0) return;
      const half = (volume.truncation * size(s.level)) / (s.cosine * Math.hypot(...s.ray));
      walk(
        volume,
        s.level,
        view.center,
        s.ray,
        s.depth - half,
        s.depth + half,
        size(s.level) * 8,
        (x, y, z) => {
          const key = tsdfBlockKey(volume, s.level, x, y, z);
          if (key !== EMPTY && !state.blocks.has(key))
            state.blocks.set(key, new Int32Array(TSDF_BLOCK_WORDS));
          return 'next';
        },
      );
    }),
  );
  // tsdfIntegrate and tsdfCarve
  input.views.forEach((view, v) =>
    samples(v, (s) => {
      if (s.weight === 0) return;
      const o3 = view.center;
      if (s.weight > 0) {
        const v0 = size(s.level),
          mu = volume.truncation * v0,
          half = mu / (s.cosine * Math.hypot(...s.ray));
        const red = (s.weight * (s.color & 255)) >>> 4,
          green = (s.weight * ((s.color >>> 8) & 255)) >>> 4,
          blue = (s.weight * ((s.color >>> 16) & 255)) >>> 4;
        let last = '';
        walk(volume, s.level, o3, s.ray, s.depth - half, s.depth + half, v0, (x, y, z) => {
          const at = state.at(s.level, x, y, z);
          if (!at) return 'block';
          const id = `${x},${y},${z}`;
          if (id !== last) {
            const c = [x, y, z].map((q, a) => volume.origin[a] + (q + 0.5) * v0);
            const sdf =
              s.normal[0]! * (c[0]! - s.point[0]!) +
              s.normal[1]! * (c[1]! - s.point[1]!) +
              s.normal[2]! * (c[2]! - s.point[2]!);
            if (sdf >= -mu) {
              const [block, k] = at;
              block[k] = (block[k]! + Math.floor(s.weight * Math.min(sdf / mu, 1) + 0.5)) | 0;
              block[k + 1] = (block[k + 1]! + s.weight) | 0;
              block[k + 2] = (block[k + 2]! + red) | 0;
              block[k + 3] = (block[k + 3]! + green) | 0;
              block[k + 4] = (block[k + 4]! + blue) | 0;
              block[k + 6] = tsdfTally(block[k + 6]! >>> 0, v) | 0;
            }
          }
          last = id;
          return 'next';
        });
      }
      if (o.visibility === 'view-weighted') return;
      for (let level = 0; level < volume.levels; level++) {
        if (level + o.carveSpan < s.level) continue;
        const v0 = size(level),
          mu = volume.truncation * size(Math.max(level, s.level));
        const g = volume.grid,
          w = new Int32Array(g.buffer, g.byteOffset, g.length);
        let enter = 0,
          leave = s.depth - mu / (s.cosine * Math.hypot(...s.ray));
        for (let a = 0; a < 3; a++) {
          if (s.ray[a] === 0) continue;
          const lo = volume.origin[a] + w[8 * level + a]! * v0 * 8,
            hi = lo + g[8 * level + 3 + a]! * v0 * 8;
          const p = (lo - o3[a]!) / s.ray[a]!,
            q = (hi - o3[a]!) / s.ray[a]!;
          enter = Math.max(enter, Math.min(p, q));
          leave = Math.min(leave, Math.max(p, q));
        }
        let last = '';
        walk(volume, level, o3, s.ray, enter, leave, v0, (x, y, z) => {
          const at = state.at(level, x, y, z);
          if (!at) return 'block';
          const id = `${x},${y},${z}`;
          if (id !== last) at[0][at[1] + 5]!++;
          last = id;
          return 'next';
        });
      }
    }),
  );
  if (o.visibility === 'view-weighted') {
    for (const group of tsdfVisibilityGroups(input))
      for (const [key, block] of state.blocks) {
        const [level, bx, by, bz] = tsdfKeyBlock(volume, key),
          v0 = size(level);
        for (let local = 0; local < TSDF_BLOCK_VOXELS; local++) {
          const k = local * TSDF_VOXEL_WORDS;
          if (block[k + 1] === 0) continue;
          const center = [
            bx * 8 + (local % 8),
            by * 8 + (Math.floor(local / 8) % 8),
            bz * 8 + Math.floor(local / 64),
          ].map((q, a) => volume.origin[a] + (q + 0.5) * v0);
          let vote = 0;
          for (const v of group)
            vote = Math.max(
              vote,
              viewContradiction(
                input.views[v]!,
                volume,
                W,
                H,
                level,
                center,
                o.carveSpan,
                volume.footprints[v]!,
              ),
            );
          if (!vote) continue;
          const old = block[k + 5]! >>> 0;
          block[k + 5] =
            (Math.min(4095, (old >>> 20) + 1) << 20) |
            Math.min(CARVE_MASK, (old & CARVE_MASK) + vote) |
            0;
        }
      }
  }
  // tsdfExtract
  const minWeight = Math.floor(o.minWeight * TSDF_WEIGHT_SCALE + 0.5);
  const voxelAt = (level: number, x: number, y: number, z: number) => {
    const at = state.at(level, x, y, z);
    if (!at) return undefined;
    const [block, k] = at;
    const weight = block[k + 1]! >>> 0,
      free = block[k + 5]! >>> 0;
    const tally = block[k + 6]! >>> 0;
    const carved =
      o.visibility === 'view-weighted'
        ? free >>> 20 >= 2 &&
          (free & CARVE_MASK) * TSDF_WEIGHT_SCALE * Math.max(1, tsdfSamples(tally)) >
            o.carveRatio * weight * TSDF_CARVE_SCALE * Math.max(1, tsdfViews(tally))
        : free * TSDF_WEIGHT_SCALE > o.carveRatio * weight;
    if (weight < minWeight || weight === 0 || tsdfViews(block[k + 6]! >>> 0) < o.minViews || carved)
      return undefined;
    return { sdf: block[k]! / weight, block, k };
  };
  const positions: number[] = [],
    normals: number[] = [],
    colors: number[] = [],
    weights: number[] = [],
    frees: number[] = [],
    tallies: number[] = [],
    freeWeights: number[] = [];
  for (const key of state.blocks.keys()) {
    const [level, bx, by, bz] = tsdfKeyBlock(volume, key);
    const v0 = size(level);
    for (let local = 0; local < TSDF_BLOCK_VOXELS; local++) {
      const x = bx * 8 + (local % 8),
        y = by * 8 + (Math.floor(local / 8) % 8),
        z = bz * 8 + Math.floor(local / 64);
      const self = voxelAt(level, x, y, z);
      if (!self) continue;
      let finer = false;
      for (let fine = 0; fine < level && !finer; fine++) {
        const shift = level - fine,
          half = 1 << (shift - 1);
        if (voxelAt(fine, (x << shift) + half, (y << shift) + half, (z << shift) + half))
          finer = true;
      }
      if (finer) continue;
      const plus = [
        voxelAt(level, x + 1, y, z),
        voxelAt(level, x, y + 1, z),
        voxelAt(level, x, y, z + 1),
      ];
      const minus = [
        (x & 7) !== 0 ? voxelAt(level, x - 1, y, z) : undefined,
        (y & 7) !== 0 ? voxelAt(level, x, y - 1, z) : undefined,
        (z & 7) !== 0 ? voxelAt(level, x, y, z - 1) : undefined,
      ];
      const gradient = [0, 1, 2].map((a) =>
        plus[a] ? plus[a].sdf - self.sdf : minus[a] ? self.sdf - minus[a].sdf : 0,
      );
      const g = Math.hypot(...gradient);
      const normal = gradient.map((c) => c / g);
      const weight = self.block[self.k + 1]! >>> 0;
      const rgb = [2, 3, 4].map((c) =>
        Math.min(Math.floor(((self.block[self.k + c]! >>> 0) * 16) / weight + 0.5), 255),
      );
      const color = (rgb[0]! | (rgb[1]! << 8) | (rgb[2]! << 16) | (255 << 24)) >>> 0;
      const center = [x, y, z].map((q, a) => volume.origin[a] + (q + 0.5) * v0);
      for (let a = 0; a < 3; a++) {
        const other = plus[a];
        if (!other) continue;
        const sa = self.sdf,
          sb = other.sdf;
        if (sa >= 0 === sb >= 0 || Math.abs(sa - sb) > 1) continue;
        const p = center.slice();
        p[a]! += v0 * (sa / (sa - sb));
        positions.push(...p);
        normals.push(encodeTsdfNormal(normal));
        colors.push(color);
        weights.push(weight);
        const free = self.block[self.k + 5]! >>> 0;
        frees.push(o.visibility === 'view-weighted' ? free >>> 20 : free);
        if (o.visibility === 'view-weighted')
          freeWeights.push((free & CARVE_MASK) / TSDF_CARVE_SCALE);
        tallies.push(self.block[self.k + 6]! >>> 0);
      }
    }
  }
  return {
    ...sortTsdfPoints(
      Float32Array.from(positions),
      Uint32Array.from(normals),
      Uint32Array.from(colors),
      Uint32Array.from(weights),
      Uint32Array.from(frees),
      Uint32Array.from(tallies),
      o.visibility === 'view-weighted' ? Float32Array.from(freeWeights) : undefined,
    ),
    volume,
    blocks: state.blocks.size,
  };
}

/** The points sorted by x, then y, then z (then normal and color), so both tiers give one order. */
export function sortTsdfPoints(
  positions: Float32Array,
  normals: Uint32Array,
  colors: Uint32Array,
  weights: Uint32Array,
  free: Uint32Array,
  tallies: Uint32Array,
  freeWeights?: Float32Array,
): TsdfPoints {
  const n = normals.length;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) =>
      positions[3 * a]! - positions[3 * b]! ||
      positions[3 * a + 1]! - positions[3 * b + 1]! ||
      positions[3 * a + 2]! - positions[3 * b + 2]! ||
      normals[a]! - normals[b]! ||
      colors[a]! - colors[b]! ||
      weights[a]! - weights[b]! ||
      free[a]! - free[b]! ||
      (freeWeights?.[a] ?? 0) - (freeWeights?.[b] ?? 0) ||
      tallies[a]! - tallies[b]!,
  );
  const out = {
    count: n,
    positions: new Float32Array(3 * n),
    normals: new Uint32Array(n),
    colors: new Uint32Array(n),
    weights: new Uint32Array(n),
    free: new Uint32Array(n),
    tallies: new Uint32Array(n),
  };
  order.forEach((i, k) => {
    out.positions.set(positions.subarray(3 * i, 3 * i + 3), 3 * k);
    out.normals[k] = normals[i]!;
    out.colors[k] = colors[i]!;
    out.weights[k] = weights[i]!;
    out.free[k] = free[i]!;
    out.tallies[k] = tallies[i]!;
  });
  return freeWeights
    ? { ...out, freeWeights: Float32Array.from(order, (i) => freeWeights[i]!) }
    : out;
}

/**
 * @internal The same order as `sortTsdfPoints`, consuming exclusively owned, disjoint arrays.
 * GPU extraction has just concatenated its parts, so it can permute these channels in place
 * instead of keeping a second complete cloud. The public immutable sorter stays unchanged.
 */
export function sortOwnedTsdfPoints(
  positions: Float32Array,
  normals: Uint32Array,
  colors: Uint32Array,
  weights: Uint32Array,
  free: Uint32Array,
  tallies: Uint32Array,
  freeWeights?: Float32Array,
): TsdfPoints {
  const n = normals.length;
  if (
    positions.length !== 3 * n ||
    [colors, weights, free, tallies].some((channel) => channel.length !== n) ||
    (freeWeights && freeWeights.length !== n)
  )
    throw new RangeError('Invalid owned TSDF point channels');
  const compare = (a: number, b: number) =>
    positions[3 * a]! - positions[3 * b]! ||
    positions[3 * a + 1]! - positions[3 * b + 1]! ||
    positions[3 * a + 2]! - positions[3 * b + 2]! ||
    normals[a]! - normals[b]! ||
    colors[a]! - colors[b]! ||
    weights[a]! - weights[b]! ||
    free[a]! - free[b]! ||
    (freeWeights?.[a] ?? 0) - (freeWeights?.[b] ?? 0) ||
    tallies[a]! - tallies[b]!;
  // Finite keys form a total order: explicit input-index ties retain the old stable order.
  // For nonfinite keys the old comparator can be nontransitive; use its exact Array.sort path.
  const typedOrder =
    n <= 0xffffffff &&
    positions.every(Number.isFinite) &&
    (freeWeights === undefined || freeWeights.every(Number.isFinite));
  const order: Uint32Array | number[] = typedOrder
    ? Uint32Array.from({ length: n }, (_, i) => i)
    : Array.from({ length: n }, (_, i) => i);
  order.sort(typedOrder ? (a, b) => compare(a, b) || a - b : compare);
  const visited = typedOrder ? 0xffffffff : -1;
  // Raw position words preserve signed zero and NaN payloads, as the immutable sorter's set().
  const words = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  for (let start = 0; start < n; start++) {
    if (order[start] === visited) continue;
    const x = words[3 * start]!,
      y = words[3 * start + 1]!,
      z = words[3 * start + 2]!,
      normal = normals[start]!,
      color = colors[start]!,
      weight = weights[start]!,
      empty = free[start]!,
      tally = tallies[start]!,
      freeWeight = freeWeights?.[start];
    let target = start;
    for (;;) {
      const source = order[target]!;
      order[target] = visited;
      if (source === start) {
        words[3 * target] = x;
        words[3 * target + 1] = y;
        words[3 * target + 2] = z;
        normals[target] = normal;
        colors[target] = color;
        weights[target] = weight;
        free[target] = empty;
        tallies[target] = tally;
        if (freeWeights) freeWeights[target] = freeWeight!;
        break;
      }
      words[3 * target] = words[3 * source]!;
      words[3 * target + 1] = words[3 * source + 1]!;
      words[3 * target + 2] = words[3 * source + 2]!;
      normals[target] = normals[source]!;
      colors[target] = colors[source]!;
      weights[target] = weights[source]!;
      free[target] = free[source]!;
      tallies[target] = tallies[source]!;
      if (freeWeights) freeWeights[target] = freeWeights[source]!;
      target = source;
    }
  }
  return {
    count: n,
    positions,
    normals,
    colors,
    weights,
    free,
    tallies,
    ...(freeWeights ? { freeWeights } : {}),
  };
}
