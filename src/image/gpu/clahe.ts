import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from '../../device/gpu/device.ts';
import { claheApply, claheHistogram, claheMapTiles } from './clahe.shade.ts';

/** Gray levels of the quantization that scikit-image's `equalize_adapthist` works on (14 bit). */
const LEVELS = 16384;
const BINS = 256;
const BIN_SIZE = 1 + LEVELS / BINS;

export interface ClaheOptions {
  /** Tiles across and down, one count for both or `[across, down]`; defaults to 8. At most 32. */
  readonly tiles?: number | readonly [number, number];
  /**
   * Clipping limit as scikit-image's `clip_limit`: the largest count of a histogram bin, as a
   * fraction of the pixels of a tile. Defaults to 0.01, in (0, 1].
   */
  readonly clipLimit?: number;
}

/** The resolved plan of one CLAHE pass over an image of a given size. */
export interface ClahePlan {
  readonly width: number;
  readonly height: number;
  readonly tilesX: number;
  readonly tilesY: number;
  readonly tileWidth: number;
  readonly tileHeight: number;
  readonly bins: number;
  readonly binSize: number;
  readonly limit: number;
}

/** Resolve `clahe: true | { tiles, clipLimit }` for an image; throws `RangeError` when invalid. */
export function planClahe(
  options: boolean | ClaheOptions,
  width: number,
  height: number,
): ClahePlan {
  const given = typeof options === 'object' ? options : {};
  const tiles = given.tiles ?? 8,
    [requestedX, requestedY] = typeof tiles === 'number' ? [tiles, tiles] : tiles,
    clipLimit = given.clipLimit ?? 0.01;
  if (
    !Number.isSafeInteger(requestedX) ||
    !Number.isSafeInteger(requestedY) ||
    requestedX < 1 ||
    requestedY < 1 ||
    requestedX > 32 ||
    requestedY > 32 ||
    requestedX > width ||
    requestedY > height ||
    !(clipLimit > 0 && clipLimit <= 1)
  )
    throw new RangeError('Invalid CLAHE options');
  const tileWidth = Math.ceil(width / requestedX),
    tileHeight = Math.ceil(height / requestedY),
    // Ceil-sized tiles can leave fewer than requested; none is empty.
    tilesX = Math.ceil(width / tileWidth),
    tilesY = Math.ceil(height / tileHeight);
  return {
    width,
    height,
    tilesX,
    tilesY,
    tileWidth,
    tileHeight,
    bins: BINS,
    binSize: BIN_SIZE,
    limit: Math.max(1, Math.floor(clipLimit * tileWidth * tileHeight)),
  };
}

/** Bytes of the two scratch buffers `recordClahe` needs: `tiles × bins` words each. */
export const claheScratchLength = (plan: ClahePlan) => plan.tilesX * plan.tilesY * plan.bins;

/**
 * Record the three CLAHE kernels into an open `device.plan` body. `input` is a gray image in
 * [0, 1]; `output` receives the equalized image; `hist` (u32) and `map` (f32) are scratch of
 * `claheScratchLength(plan)` elements.
 */
export function recordClahe(
  plan: ClahePlan,
  input: Resident<Float32Array>,
  output: Resident<Float32Array>,
  hist: Resident<Uint32Array>,
  map: Resident<Float32Array>,
): void {
  const clahe = {
      width: plan.width,
      height: plan.height,
      tilesX: plan.tilesX,
      tilesY: plan.tilesY,
      tileWidth: plan.tileWidth,
      tileHeight: plan.tileHeight,
      bins: plan.bins,
      binSize: plan.binSize,
      limit: plan.limit,
    },
    tileGroups = Math.ceil((plan.tilesX * plan.tilesY) / 64);
  claheHistogram({ claheIn: input, claheHist: hist, clahe }, tileGroups);
  claheMapTiles({ claheHist: hist, claheMap: map, clahe }, tileGroups);
  claheApply(
    { claheIn: input, claheMapIn: map, claheOut: output, clahe },
    Math.ceil((plan.width * plan.height) / 64),
  );
}

/** CLAHE of a gray image in [0, 1] on the device; the result is read back. */
export function equalizeGrayGpu(
  device: ShadeDevice,
  gray: Float32Array,
  width: number,
  height: number,
  options: boolean | ClaheOptions = true,
): Promise<Float32Array> {
  if (!(width >= 1) || !(height >= 1) || gray.length !== width * height)
    throw new RangeError('Invalid gray image for CLAHE');
  const plan = planClahe(options, width, height);
  return device.enqueue(async () => {
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    try {
      const input = device.allocate(width * height, gray),
        output = device.allocate(width * height),
        hist = device.allocateU32(claheScratchLength(plan)),
        map = device.allocate(claheScratchLength(plan));
      owned.push(input, output, hist, map);
      await device.plan(() => recordClahe(plan, input, output, hist, map));
      return await device.read(output, width * height);
    } finally {
      for (const buffer of owned) device.recycle(buffer);
    }
  });
}

/**
 * The CPU reference of `equalizeGrayGpu`: scikit-image's `equalize_adapthist` in double precision
 * on the same tiles, bins and interpolation, written straight from its algorithm.
 */
export function equalizeGrayCpu(
  gray: ArrayLike<number>,
  width: number,
  height: number,
  options: boolean | ClaheOptions = true,
): Float32Array {
  if (gray.length !== width * height) throw new RangeError('Invalid gray image for CLAHE');
  const plan = planClahe(options, width, height),
    { tilesX, tilesY, tileWidth, tileHeight, bins, binSize, limit } = plan,
    binOf = (value: number) =>
      Math.floor(Math.round(Math.min(Math.max(value, 0), 1) * (LEVELS - 1)) / binSize),
    maps = new Float64Array(tilesX * tilesY * bins);
  for (let ty = 0; ty < tilesY; ty++)
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Array<number>(bins).fill(0),
        x1 = Math.min((tx + 1) * tileWidth, width),
        y1 = Math.min((ty + 1) * tileHeight, height);
      let pixels = 0;
      for (let y = ty * tileHeight; y < y1; y++)
        for (let x = tx * tileWidth; x < x1; x++, pixels++) hist[binOf(gray[y * width + x]!)]!++;
      clipHistogram(hist, limit);
      let cumulative = 0;
      for (let b = 0; b < bins; b++) {
        cumulative += hist[b]!;
        // `map_histogram`: truncation to an integer level.
        maps[(ty * tilesX + tx) * bins + b] =
          Math.min(Math.floor((cumulative * (LEVELS - 1)) / pixels), LEVELS - 1) / (LEVELS - 1);
      }
    }
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const fy = Math.min(Math.max((y + 0.5) / tileHeight - 0.5, 0), tilesY - 1),
      y0 = Math.min(Math.floor(fy), tilesY - 1),
      y1 = Math.min(y0 + 1, tilesY - 1),
      ay = fy - y0;
    for (let x = 0; x < width; x++) {
      const fx = Math.min(Math.max((x + 0.5) / tileWidth - 0.5, 0), tilesX - 1),
        x0 = Math.min(Math.floor(fx), tilesX - 1),
        x1 = Math.min(x0 + 1, tilesX - 1),
        ax = fx - x0,
        bin = binOf(gray[y * width + x]!),
        at = (tx: number, ty: number) => maps[(ty * tilesX + tx) * bins + bin]!;
      out[y * width + x] =
        (at(x0, y0) * (1 - ax) + at(x1, y0) * ax) * (1 - ay) +
        (at(x0, y1) * (1 - ax) + at(x1, y1) * ax) * ay;
    }
  }
  return out;
}

/** scikit-image's `clip_histogram`, in place. */
function clipHistogram(hist: number[], limit: number): void {
  const size = hist.length;
  let excess = 0;
  for (const count of hist) if (count > limit) excess += count - limit;
  const increment = Math.floor(excess / size),
    upper = limit - increment;
  for (let b = 0; b < size; b++) if (hist[b]! > limit) hist[b] = limit;
  for (let b = 0; b < size; b++)
    if (hist[b]! < upper) {
      hist[b]! += increment;
      excess -= increment;
    }
  for (let b = 0; b < size; b++)
    if (hist[b]! >= upper && hist[b]! < limit) {
      excess += hist[b]! - limit;
      hist[b] = limit;
    }
  while (excess > 0) {
    const before = excess;
    for (let index = 0; index < size; index++) {
      const under = hist.filter((count) => count < limit).length,
        step = Math.max(Math.floor(under / excess), 1);
      for (let b = index; b < size; b += step)
        if (hist[b]! < limit) {
          hist[b]!++;
          excess--;
        }
      if (excess <= 0) break;
    }
    if (excess === before) break;
  }
}
