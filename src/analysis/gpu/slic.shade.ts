'use typeshade';

// SLIC superpixels (Achanta et al. 2012) on a resident RGBA image, after scikit-image's
// `segmentation.slic`: sRGB to CIELAB, a grid of cluster centers, and k-means iterations with the
// distance `dLab^2 + (compactness / S)^2 * dXY^2`.
//
// The GPU-friendly form (gSLICr): the grid has `gridX x gridY` cells and cluster `k` starts at the
// center of cell `k`. A pixel searches the clusters of its own cell and the eight cells around it.
// So a cluster only owns pixels of the 3 x 3 cells around its grid cell. The center update is
// then a gather over those cells, one invocation per cluster, with no atomics: the sums run in a
// fixed order, so a run is repeatable and the CPU tier runs the same code as WebGPU.
//
// Arrays: `slicLab` has three f32 per pixel (L, a, b); `slicCenters` has five f32 per cluster
// (L, a, b, x, y); `slicLabels` has one u32 per pixel.

class SlicControl {
  width: u32;
  height: u32;
  gridX: u32;
  gridY: u32;
  // The weight of the squared pixel distance: (compactness / S)^2.
  ratio: f32;
}
class SlicRange {
  first: u32;
  end: u32;
}
declare const slic: uniform<SlicControl>;
declare const slicRange: uniform<SlicRange>;
declare const slicRgba: storage<array<u32>>;
declare const slicLab: storage<array<f32>, 'read_write'>;
declare const slicLabRead: storage<array<f32>>;
declare const slicCenters: storage<array<f32>, 'read_write'>;
declare const slicCentersRead: storage<array<f32>>;
declare const slicLabels: storage<array<u32>, 'read_write'>;
declare const slicLabelsRead: storage<array<u32>>;

function srgbToLinear(c: f32): f32 {
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

function labF(t: f32): f32 {
  return select(7.787037037037037 * t + 0.13793103448275862, pow(t, 0.3333333333333333), t > 0.008856451679035631);
}

/** One invocation per pixel: sRGB bytes to CIELAB (D65, L in [0, 100]). */
@compute([64])
export function slicToLab(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = slicRange.first + gid.x;
  const active: bool = i < slicRange.end;
  if (!active) i = max(slicRange.end, 1) - 1;
  const pixel: u32 = slicRgba[i];
  const r: f32 = srgbToLinear(f32(pixel & 255) / 255.0);
  const g: f32 = srgbToLinear(f32((pixel >> 8) & 255) / 255.0);
  const b: f32 = srgbToLinear(f32((pixel >> 16) & 255) / 255.0);
  const x: f32 = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.95047;
  const y: f32 = 0.212671 * r + 0.71516 * g + 0.072169 * b;
  const z: f32 = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883;
  const fy: f32 = labF(y);
  if (active) {
    slicLab[3 * i] = 116.0 * fy - 16.0;
    slicLab[3 * i + 1] = 500.0 * (labF(x) - fy);
    slicLab[3 * i + 2] = 200.0 * (fy - labF(z));
  }
}

function gradientAt(x: u32, y: u32): f32 {
  const w: u32 = slic.width;
  const left: u32 = 3 * (y * w + x - 1);
  const right: u32 = 3 * (y * w + x + 1);
  const up: u32 = 3 * ((y - 1) * w + x);
  const down: u32 = 3 * ((y + 1) * w + x);
  let g: f32 = 0;
  for (let c: u32 = 0; c < 3; c++) {
    const dx: f32 = slicLabRead[right + c] - slicLabRead[left + c];
    const dy: f32 = slicLabRead[down + c] - slicLabRead[up + c];
    g += dx * dx + dy * dy;
  }
  return g;
}

/**
 * One invocation per cluster: the center of grid cell `k`, moved to the pixel of lowest Lab
 * gradient in the 3 x 3 pixels around it (the first in scan order on a tie; the image border is
 * never chosen). The cluster color is that pixel's.
 */
@compute([64])
export function slicSeeds(@builtin('global_invocation_id') gid: vec3u): void {
  let k: u32 = slicRange.first + gid.x;
  const active: bool = k < slicRange.end;
  if (!active) k = max(slicRange.end, 1) - 1;
  const w: u32 = slic.width;
  const h: u32 = slic.height;
  const gx: u32 = k % slic.gridX;
  const gy: u32 = k / slic.gridX;
  // The center of cell gx: (gx + 0.5) * w / gridX, rounded down.
  const cx: u32 = ((2 * gx + 1) * w) / (2 * slic.gridX);
  const cy: u32 = ((2 * gy + 1) * h) / (2 * slic.gridY);
  let bx: u32 = cx;
  let by: u32 = cy;
  if (w >= 3 && h >= 3) {
    let best: f32 = 3.4e38;
    for (let dy: i32 = -1; dy <= 1; dy++)
      for (let dx: i32 = -1; dx <= 1; dx++) {
        const x: i32 = i32(cx) + dx;
        const y: i32 = i32(cy) + dy;
        if (x >= 1 && y >= 1 && x < i32(w) - 1 && y < i32(h) - 1) {
          const g: f32 = gradientAt(u32(x), u32(y));
          if (g < best) {
            best = g;
            bx = u32(x);
            by = u32(y);
          }
        }
      }
  }
  const at: u32 = 3 * (by * w + bx);
  if (active) {
    slicCenters[5 * k] = slicLabRead[at];
    slicCenters[5 * k + 1] = slicLabRead[at + 1];
    slicCenters[5 * k + 2] = slicLabRead[at + 2];
    slicCenters[5 * k + 3] = f32(bx);
    slicCenters[5 * k + 4] = f32(by);
  }
}

/** One invocation per pixel: the nearest center among the clusters of the 3 x 3 cells around its cell. */
@compute([64])
export function slicAssign(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = slicRange.first + gid.x;
  const active: bool = i < slicRange.end;
  if (!active) i = max(slicRange.end, 1) - 1;
  const w: u32 = slic.width;
  const x: u32 = i % w;
  const y: u32 = i / w;
  const cellX: i32 = i32((x * slic.gridX) / w);
  const cellY: i32 = i32((y * slic.gridY) / slic.height);
  const l: f32 = slicLabRead[3 * i];
  const a: f32 = slicLabRead[3 * i + 1];
  const b: f32 = slicLabRead[3 * i + 2];
  let best: f32 = 3.4e38;
  let label: u32 = u32(cellY) * slic.gridX + u32(cellX);
  for (let dy: i32 = -1; dy <= 1; dy++)
    for (let dx: i32 = -1; dx <= 1; dx++) {
      const gx: i32 = cellX + dx;
      const gy: i32 = cellY + dy;
      if (gx >= 0 && gy >= 0 && gx < i32(slic.gridX) && gy < i32(slic.gridY)) {
        const k: u32 = u32(gy) * slic.gridX + u32(gx);
        const dl: f32 = l - slicCentersRead[5 * k];
        const da: f32 = a - slicCentersRead[5 * k + 1];
        const db: f32 = b - slicCentersRead[5 * k + 2];
        const ex: f32 = f32(x) - slicCentersRead[5 * k + 3];
        const ey: f32 = f32(y) - slicCentersRead[5 * k + 4];
        const d: f32 = dl * dl + da * da + db * db + slic.ratio * (ex * ex + ey * ey);
        if (d < best) {
          best = d;
          label = k;
        }
      }
    }
  if (active) slicLabels[i] = label;
}

/**
 * One invocation per cluster: the new center is the mean of the pixels labelled `k`. They all lie
 * in the 3 x 3 cells around its grid cell. A cluster with no pixel keeps its center.
 */
@compute([64])
export function slicUpdate(@builtin('global_invocation_id') gid: vec3u): void {
  let k: u32 = slicRange.first + gid.x;
  const active: bool = k < slicRange.end;
  if (!active) k = max(slicRange.end, 1) - 1;
  const w: u32 = slic.width;
  const h: u32 = slic.height;
  const gx: u32 = k % slic.gridX;
  const gy: u32 = k / slic.gridX;
  // The pixels of cells gx - 1 .. gx + 1: x with floor(x * gridX / w) in that range.
  const x0: u32 = select(0, (((gx - 1) * w) + slic.gridX - 1) / slic.gridX, gx >= 1);
  const x1: u32 = min(((gx + 2) * w + slic.gridX - 1) / slic.gridX, w);
  const y0: u32 = select(0, (((gy - 1) * h) + slic.gridY - 1) / slic.gridY, gy >= 1);
  const y1: u32 = min(((gy + 2) * h + slic.gridY - 1) / slic.gridY, h);
  // Sums of offsets from the window origin keep the coordinate sums exact in f32.
  let sl: f32 = 0;
  let sa: f32 = 0;
  let sb: f32 = 0;
  let sx: f32 = 0;
  let sy: f32 = 0;
  let count: f32 = 0;
  for (let y: u32 = y0; y < y1; y++)
    for (let x: u32 = x0; x < x1; x++) {
      const i: u32 = y * w + x;
      if (slicLabelsRead[i] === k) {
        sl += slicLabRead[3 * i];
        sa += slicLabRead[3 * i + 1];
        sb += slicLabRead[3 * i + 2];
        sx += f32(x - x0);
        sy += f32(y - y0);
        count += 1.0;
      }
    }
  if (active && count > 0.0) {
    slicCenters[5 * k] = sl / count;
    slicCenters[5 * k + 1] = sa / count;
    slicCenters[5 * k + 2] = sb / count;
    slicCenters[5 * k + 3] = f32(x0) + sx / count;
    slicCenters[5 * k + 4] = f32(y0) + sy / count;
  }
}
