'use typeshade';

// Contrast-limited adaptive histogram equalization (CLAHE) on a resident gray image in [0, 1],
// after scikit-image's `exposure.equalize_adapthist`: per-tile histograms, clipping with
// redistribution, a per-tile cumulative map, and bilinear interpolation between the maps of the
// four nearest tile centres. The histogram and map kernels run one invocation per tile, with no
// workgroup memory, so the CPU tier runs the same code as the GPU tier.
//
// Levels: a gray value is quantized to a 14-bit level (`round(g * 16383)`) and a level falls in
// bin `level / binSize` with `binSize = 1 + 16384 / bins`, as scikit-image does. The map is
// `floor(cumulative * 16383 / pixels) / 16383`.

class ClaheControl {
  width: u32;
  height: u32;
  tilesX: u32;
  tilesY: u32;
  // Tile size in pixels; the last tile of a row or column may be smaller.
  tileWidth: u32;
  tileHeight: u32;
  bins: u32;
  binSize: u32;
  // Largest count a bin may hold before clipping (at least 1); used by `claheMapTiles`.
  limit: u32;
}
declare const clahe: uniform<ClaheControl>;
declare const claheIn: storage<array<f32>>;
declare const claheHist: storage<array<u32>, 'read_write'>;
declare const claheMap: storage<array<f32>, 'read_write'>;

function claheBin(value: f32, binSize: u32): u32 {
  const level: u32 = u32(clamp(value, 0.0, 1.0) * 16383.0 + 0.5);
  return level / binSize;
}

/** One invocation per tile: zero the tile's histogram, then count its pixels. */
@compute([64])
export function claheHistogram(@builtin('global_invocation_id') gid: vec3u): void {
  const t: u32 = gid.x;
  if (t >= clahe.tilesX * clahe.tilesY) return;
  const tx: u32 = t % clahe.tilesX;
  const ty: u32 = t / clahe.tilesX;
  const base: u32 = t * clahe.bins;
  for (let b: u32 = 0; b < clahe.bins; b++) claheHist[base + b] = 0;
  const x0: u32 = tx * clahe.tileWidth;
  const y0: u32 = ty * clahe.tileHeight;
  const x1: u32 = min(x0 + clahe.tileWidth, clahe.width);
  const y1: u32 = min(y0 + clahe.tileHeight, clahe.height);
  for (let y: u32 = y0; y < y1; y++)
    for (let x: u32 = x0; x < x1; x++) {
      const bin: u32 = claheBin(claheIn[y * clahe.width + x], clahe.binSize);
      claheHist[base + bin] += 1;
    }
}

/**
 * One invocation per tile: clip the histogram at `limit` and redistribute the excess exactly as
 * scikit-image's `clip_histogram`, then write the map `floor(cumulative * 16383 / pixels) / 16383`.
 */
@compute([64])
export function claheMapTiles(@builtin('global_invocation_id') gid: vec3u): void {
  const t: u32 = gid.x;
  if (t >= clahe.tilesX * clahe.tilesY) return;
  const tx: u32 = t % clahe.tilesX;
  const ty: u32 = t / clahe.tilesX;
  const bins: u32 = clahe.bins;
  const limit: u32 = clahe.limit;
  const base: u32 = t * bins;
  const x0: u32 = tx * clahe.tileWidth;
  const y0: u32 = ty * clahe.tileHeight;
  const w: u32 = min(x0 + clahe.tileWidth, clahe.width) - x0;
  const h: u32 = min(y0 + clahe.tileHeight, clahe.height) - y0;
  const pixels: u32 = w * h;
  // Total excess over the limit.
  let excess: i32 = 0;
  for (let b: u32 = 0; b < bins; b++) {
    const count: u32 = claheHist[base + b];
    if (count > limit) excess += i32(count - limit);
  }
  const increment: u32 = u32(excess) / bins;
  const upper: i32 = i32(limit) - i32(increment);
  for (let b: u32 = 0; b < bins; b++) {
    const count: u32 = claheHist[base + b];
    if (count > limit) claheHist[base + b] = limit;
  }
  for (let b: u32 = 0; b < bins; b++) {
    const count: u32 = claheHist[base + b];
    if (i32(count) < upper) {
      claheHist[base + b] = count + increment;
      excess -= i32(increment);
    }
  }
  for (let b: u32 = 0; b < bins; b++) {
    const count: u32 = claheHist[base + b];
    if (i32(count) >= upper && count < limit) {
      excess += i32(count) - i32(limit);
      claheHist[base + b] = limit;
    }
  }
  // Remaining excess, one unit per eligible bin, at a stride that spreads it over the bins.
  for (let pass: u32 = 0; pass < bins; pass++) {
    if (excess <= 0) break;
    const before: i32 = excess;
    for (let index: u32 = 0; index < bins; index++) {
      if (excess <= 0) break;
      let under: u32 = 0;
      for (let b: u32 = 0; b < bins; b++) if (claheHist[base + b] < limit) under++;
      const step: u32 = max(under / u32(excess), 1);
      for (let n: u32 = 0; n < bins; n++) {
        const b: u32 = index + n * step;
        if (b >= bins) break;
        if (claheHist[base + b] < limit) {
          claheHist[base + b] += 1;
          excess -= 1;
        }
      }
    }
    if (excess === before) break;
  }
  let cumulative: u32 = 0;
  for (let b: u32 = 0; b < bins; b++) {
    cumulative += claheHist[base + b];
    const level: f32 = min(floor((f32(cumulative) * 16383.0) / f32(max(pixels, 1))), 16383.0);
    claheMap[base + b] = level / 16383.0;
  }
}

/** The index of `bin` in the map of tile (`tx`, `ty`). */
function mapIndex(tx: u32, ty: u32, bin: u32, tilesX: u32, bins: u32): u32 {
  return (ty * tilesX + tx) * bins + bin;
}

/** Each pixel: interpolate bilinearly between the maps of the four nearest tile centres. */
export function claheApply(
  width: u32,
  height: u32,
  tilesX: u32,
  tilesY: u32,
  tileWidth: u32,
  tileHeight: u32,
  bins: u32,
  binSize: u32,
  input: array<f32>,
  maps: array<f32>,
  out: array<f32>,
): void {
  for (let i: u32 = 0; i < width * height; i++) {
    const x: u32 = i % width;
    const y: u32 = i / width;
    const bin: u32 = claheBin(input[i], binSize);
    const fx: f32 = clamp((f32(x) + 0.5) / f32(tileWidth) - 0.5, 0.0, f32(tilesX - 1));
    const fy: f32 = clamp((f32(y) + 0.5) / f32(tileHeight) - 0.5, 0.0, f32(tilesY - 1));
    const x0: u32 = min(u32(floor(fx)), tilesX - 1);
    const y0: u32 = min(u32(floor(fy)), tilesY - 1);
    const x1: u32 = min(x0 + 1, tilesX - 1);
    const y1: u32 = min(y0 + 1, tilesY - 1);
    const ax: f32 = fx - f32(x0);
    const ay: f32 = fy - f32(y0);
    const top: f32 =
      maps[mapIndex(x0, y0, bin, tilesX, bins)] * (1.0 - ax) +
      maps[mapIndex(x1, y0, bin, tilesX, bins)] * ax;
    const bottom: f32 =
      maps[mapIndex(x0, y1, bin, tilesX, bins)] * (1.0 - ax) +
      maps[mapIndex(x1, y1, bin, tilesX, bins)] * ax;
    out[i] = top * (1.0 - ay) + bottom * ay;
  }
}
