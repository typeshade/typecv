'use typeshade';

// The TSDF fusion of the filtered depth maps (`docs/dense-mvs.md`, section 10.4): a hash table of
// blocks of 8 × 8 × 8 voxels at several levels, the voxel size of level L being v_0 · 2^L. A depth
// sample goes to the least level whose voxel is at least its footprint (scaled), so a far region is
// integrated at a coarser level. Every kernel is one invocation per element without workgroup
// memory, so the CPU tier runs the same code as the GPU tier. `src/reconstruction/tsdf.ts` is the
// CPU reference of the same arithmetic, and plans the volume for both tiers.
//
// Layouts. P = W × H, pixel index i = y × W + x.
// - `tsdfDepth`, `tsdfNormal`, `tsdfConfidence`, `tsdfColor`: separate u32 channels of one view:
//   the bits of the f32 filtered depth (optical-axis z, normalized units, 0 for "no depth"), the
//   octahedral normal in the camera frame, the f32 confidence bits, and the packed RGBA. Separate
//   channels can be uploaded directly from the input without a host copy of every packed view.
// - `tsdfGrid`: u32 × 8 per level: the least block coordinate on x, y, z (i32 bits), the block
//   counts on x, y, z, the first key of the level, 0. The key of block (bx, by, bz) of level L is
//   first + (bx − lo_x) + n_x · ((by − lo_y) + n_y · (bz − lo_z)), unique below 0xffffffff.
// - `tsdfTable`: atomic u32: a header of 8 words (0 the claimed slots, 1 the assigned blocks, 2 an
//   overflow flag, 3 the counted points, 4 the written points), then T keys (0xffffffff empty),
//   then T block indices, one per slot, then T slots, one per block. T is a power of two.
// - `tsdfVoxels`: atomic u32 × 7 per voxel, 512 voxels per block (x fastest): the sum of
//   w · sdf / μ (i32 bits, fixed point), the sum of w, three weighted color sums, the free-space
//   count (or packed weighted group evidence) and the tally: bits 0..11 the samples integrated (saturated at 4095),
//   bits 12..19 the views that integrated it (saturated at 255), bits 20..31 the mark of the last
//   such view (its index mod 4095, plus 1). w is fixed point with 4096 for 1.
// - `tsdfPoints`: f32 × 3 per point, the position in the normalized world frame.
// - `tsdfPointData`: u32 × 5 per point, the octahedral world normal, the packed RGBA, and the
//   weight, the free-space count and the tally of the voxel that emitted it (section 10.5).
// View-weighted mode packs word 5 as a 12-bit group count and 20-bit weight sum (256 per unit).
// tsdfVotes has one u32 per voxel: the current group's maximum vote, reset between groups.
//
// A lane past its chunk takes the chunk's last element, runs the same control flow and writes
// nothing (the clamp-and-mask rule of section 3.6).

class TsdfControl {
  width: u32;
  height: u32;
  // The view: world-to-camera rotation by rows, the camera center, the intrinsics.
  r0: f32;
  r1: f32;
  r2: f32;
  r3: f32;
  r4: f32;
  r5: f32;
  r6: f32;
  r7: f32;
  r8: f32;
  centerX: f32;
  centerY: f32;
  centerZ: f32;
  fx: f32;
  fy: f32;
  cx: f32;
  cy: f32;
  // The view's distortion, for the color sample through the map of `mvsUndistort` (3.11).
  distorted: u32;
  k1: f32;
  k2: f32;
  radialLimit: f32;
  // A depth d goes to the least level L < levels with d · footprint ≤ v_L.
  footprint: f32;
  // The volume: the origin of every level's voxel grid, v_0 and its f32 inverse, the levels.
  originX: f32;
  originY: f32;
  originZ: f32;
  voxel: f32;
  inverse: f32;
  levels: u32;
  // The truncation μ_L in voxels of the level.
  truncation: f32;
  // T − 1, T the slots of the table.
  mask: u32;
  // The box of this pass in cells of the coarsest block, [lo, hi) on each axis. A pass claims the
  // blocks of the box and, on each axis, the first block of each level past hi (the margin), and
  // extracts the blocks of the box alone.
  boxLoX: i32;
  boxLoY: i32;
  boxLoZ: i32;
  boxHiX: i32;
  boxHiY: i32;
  boxHiZ: i32;
  // The index of the view in the input, for the tally of `tsdfIntegrate`.
  view: u32;
  // `tsdfCarve` walks level L only when sample level − L ≤ carveSpan (all levels coarser than
  // the sample's, and carveSpan finer ones).
  carveSpan: u32;
  visibility: u32;
  // The extraction: the least weight (fixed point), the least views of the tally, the carve
  // ratio, and 0 to count, 1 to write.
  minWeight: u32;
  minViews: u32;
  carveRatio: f32;
  write: u32;
  // The points `tsdfPoints` holds.
  capacity: u32;
}
// The elements of one chunk: pixels, slots or voxels `first..end`.
class TsdfRange {
  first: u32;
  end: u32;
}
declare const tsdfControl: uniform<TsdfControl>;
declare const tsdfRange: uniform<TsdfRange>;
declare const tsdfDepth: storage<array<u32>>;
declare const tsdfNormal: storage<array<u32>>;
declare const tsdfConfidence: storage<array<u32>>;
declare const tsdfColor: storage<array<u32>>;
declare const tsdfGrid: storage<array<u32>>;
declare const tsdfTable: storage<array<atomic<u32>>, 'read_write'>;
declare const tsdfVoxels: storage<array<atomic<u32>>, 'read_write'>;
declare const tsdfPoints: storage<array<f32>, 'read_write'>;
declare const tsdfPointData: storage<array<u32>, 'read_write'>;
declare const tsdfVotes: storage<array<u32>, 'read_write'>;

const EMPTY: u32 = 0xffffffff;
const HEADER: u32 = 8;
const VOXEL_WORDS: u32 = 7;
const BLOCK_VOXELS: u32 = 512;
const WEIGHT_SCALE: f32 = 4096.;
// A sample whose surface faces the ray at a cosine below this adds no weight (the view-normal term
// of the confidence is 0 there, section 10.3), and the walks clamp the cosine to it.
const MIN_COSINE: f32 = 0.15;
const MAX_STEPS: u32 = 16384;
const FAR: f32 = 3.0e38;

/** World to camera rotation applied to a world vector. */
function toCameraVector(p: vec3): vec3 {
  const c: TsdfControl = tsdfControl;
  return vec3(
    c.r0 * p.x + c.r1 * p.y + c.r2 * p.z,
    c.r3 * p.x + c.r4 * p.y + c.r5 * p.z,
    c.r6 * p.x + c.r7 * p.y + c.r8 * p.z,
  );
}
/** Rᵀ v: a camera-frame vector in the world frame. */
function toWorldVector(v: vec3): vec3 {
  const c: TsdfControl = tsdfControl;
  return vec3(
    c.r0 * v.x + c.r3 * v.y + c.r6 * v.z,
    c.r1 * v.x + c.r4 * v.y + c.r7 * v.z,
    c.r2 * v.x + c.r5 * v.y + c.r8 * v.z,
  );
}
function cameraCenter(): vec3 {
  return vec3(tsdfControl.centerX, tsdfControl.centerY, tsdfControl.centerZ);
}
function volumeOrigin(): vec3 {
  return vec3(tsdfControl.originX, tsdfControl.originY, tsdfControl.originZ);
}

/** The octahedral map of a unit vector, two snorm16 components (section 4.1). */
function encodeNormal(n: vec3): u32 {
  const s: f32 = abs(n.x) + abs(n.y) + abs(n.z);
  let px: f32 = n.x / s;
  let py: f32 = n.y / s;
  if (n.z < 0.) {
    const ox: f32 = (1. - abs(py)) * (px >= 0. ? 1. : -1.);
    const oy: f32 = (1. - abs(px)) * (py >= 0. ? 1. : -1.);
    px = ox;
    py = oy;
  }
  return pack2x16snorm(vec2(px, py));
}
function decodeNormal(word: u32): vec3 {
  const p: vec2 = unpack2x16snorm(word);
  const z: f32 = 1. - abs(p.x) - abs(p.y);
  const t: f32 = max(-z, 0.);
  const x: f32 = p.x + (p.x >= 0. ? -t : t);
  const y: f32 = p.y + (p.y >= 0. ? -t : t);
  return normalize(vec3(x, y, z));
}

/** Word `k` of level `level`'s record in `tsdfGrid`. */
function gridWord(level: u32, k: u32): u32 {
  return tsdfGrid[level * 8 + k];
}
/** The key of a block, or EMPTY when it lies outside its level's grid. */
function blockKey(level: u32, bx: i32, by: i32, bz: i32): u32 {
  const x: i32 = bx - i32(gridWord(level, 0));
  const y: i32 = by - i32(gridWord(level, 1));
  const z: i32 = bz - i32(gridWord(level, 2));
  const nx: u32 = gridWord(level, 3);
  const ny: u32 = gridWord(level, 4);
  const nz: u32 = gridWord(level, 5);
  if (x < 0 || y < 0 || z < 0) return EMPTY;
  if (u32(x) >= nx || u32(y) >= ny || u32(z) >= nz) return EMPTY;
  return gridWord(level, 6) + u32(x) + nx * (u32(y) + ny * u32(z));
}
/** The first slot of a key's probe sequence. */
function keySlot(key: u32): u32 {
  let h: u32 = key * 0x9e3779b1;
  h = h ^ (h >> 16);
  h = h * 0x7feb352d;
  h = h ^ (h >> 15);
  h = h * 0x846ca68b;
  return (h ^ (h >> 16)) & tsdfControl.mask;
}
/** The block index of a key, or EMPTY when the table has no such key. */
function findBlock(key: u32): u32 {
  if (key === EMPTY) return EMPTY;
  let slot: u32 = keySlot(key);
  let block: u32 = EMPTY;
  for (let probe: u32 = 0; probe <= tsdfControl.mask; probe++) {
    const k: u32 = atomicLoad(tsdfTable[HEADER + slot]);
    if (k === key) {
      block = atomicLoad(tsdfTable[HEADER + tsdfControl.mask + 1 + slot]);
      break;
    }
    if (k === EMPTY) break;
    slot = (slot + 1) & tsdfControl.mask;
  }
  return block;
}

/** The coarse cell of block coordinate `b` of level `level` on one axis. */
function coarseCell(level: u32, b: i32): i32 {
  return b >> (tsdfControl.levels - 1 - level);
}
/** Whether a block coordinate is in the box on one axis, or with `margin` its first block past hi. */
function inBox(level: u32, b: i32, lo: i32, hi: i32, margin: bool): bool {
  const c: i32 = coarseCell(level, b);
  if (c >= lo && c < hi) return true;
  return margin && b === hi << (tsdfControl.levels - 1 - level);
}
function blockInBox(level: u32, bx: i32, by: i32, bz: i32, margin: bool): bool {
  const c: TsdfControl = tsdfControl;
  return (
    inBox(level, bx, c.boxLoX, c.boxHiX, margin) &&
    inBox(level, by, c.boxLoY, c.boxHiY, margin) &&
    inBox(level, bz, c.boxLoZ, c.boxHiZ, margin)
  );
}

/** The pass's box in coarse cells: its least corner, or with `upper` its greatest corner plus one
 *  cell (the margin). Clamped to ±2^20 cells. */
function boxCorner(upper: bool): vec3 {
  const c: TsdfControl = tsdfControl;
  if (upper)
    return vec3(
      f32(min(c.boxHiX, 1048576) + 1),
      f32(min(c.boxHiY, 1048576) + 1),
      f32(min(c.boxHiZ, 1048576) + 1),
    );
  return vec3(f32(max(c.boxLoX, -1048576)), f32(max(c.boxLoY, -1048576)), f32(max(c.boxLoZ, -1048576)));
}

/** The voxel size of a level: v_0 · 2^L, exact. */
function levelSize(level: u32): f32 {
  return tsdfControl.voxel * f32(u32(1) << level);
}
/** The f32 inverse of a level's voxel size: the inverse of v_0 over 2^L, exact. */
function levelInverse(level: u32): f32 {
  return tsdfControl.inverse / f32(u32(1) << level);
}

/** The plane of voxel index `c` of voxel size `size` on one axis: origin + c · size. */
function plane(origin: f32, c: i32, size: f32): f32 {
  return origin + f32(c) * size;
}

/**
 * The parameter t at which the ray o + D t leaves the cube of voxels [c, c + n) on each axis of
 * voxel size `size` (n = 1 a voxel, n = 8 a block), FAR when the direction is 0 on every axis. A
 * plane is `plane` of its voxel index, so a voxel and its block give one t on their shared face (a
 * jump over a block lands where the walk of its voxels lands, step 4).
 */
function cellExit(o: vec3, D: vec3, cx: i32, cy: i32, cz: i32, n: i32, size: f32): f32 {
  const origin: vec3 = volumeOrigin();
  let t: f32 = FAR;
  if (D.x > 0.) t = min(t, (plane(origin.x, cx + n, size) - o.x) / D.x);
  if (D.x < 0.) t = min(t, (plane(origin.x, cx, size) - o.x) / D.x);
  if (D.y > 0.) t = min(t, (plane(origin.y, cy + n, size) - o.y) / D.y);
  if (D.y < 0.) t = min(t, (plane(origin.y, cy, size) - o.y) / D.y);
  if (D.z > 0.) t = min(t, (plane(origin.z, cz + n, size) - o.z) / D.z);
  if (D.z < 0.) t = min(t, (plane(origin.z, cz, size) - o.z) / D.z);
  return t;
}

/** One pixel's sample: its ray, point, world normal, level, weight and color. */
class TsdfSample {
  valid: u32;
  depth: f32;
  // The world ray of the pixel, scaled so that its optical-axis component is 1.
  ray: vec3;
  point: vec3;
  normal: vec3;
  // −n · D, and the cosine −n · D / |D| clamped to the least cosine.
  facing: f32;
  cosine: f32;
  level: u32;
  // The weight in fixed point (0: the sample adds nothing).
  weight: u32;
  color: u32;
}

/** Byte `k` of a packed RGBA word, as f32. */
function channel(word: u32, k: u32): f32 {
  return f32((word >> (8 * k)) & 255);
}
/** The position in the distorted image of undistorted pixel (u, v), as `mvsUndistort` maps it. */
function distortedPosition(u: f32, v: f32): vec3 {
  const c: TsdfControl = tsdfControl;
  const x: f32 = (u - c.cx) / c.fx;
  const y: f32 = (v - c.cy) / c.fy;
  const r2: f32 = x * x + y * y;
  let inside: f32 = 0.;
  let sx: f32 = 0.;
  let sy: f32 = 0.;
  if (r2 < c.radialLimit) {
    const d: f32 = 1. + c.k1 * r2 + c.k2 * r2 * r2;
    sx = c.fx * x * d + c.cx;
    sy = c.fy * y * d + c.cy;
    if (sx >= 0. && sy >= 0. && sx <= f32(c.width - 1) && sy <= f32(c.height - 1)) inside = 1.;
  }
  return vec3(sx, sy, inside);
}
/** The color of pixel `i` (section 3.11): the RGBA at its distorted position, bilinear, rounded. */
function pixelColor(i: u32): u32 {
  const W: u32 = tsdfControl.width;
  const H: u32 = tsdfControl.height;
  let color: u32 = tsdfColor[i];
  if (tsdfControl.distorted !== 0) {
    const s: vec3 = distortedPosition(f32(i % W), f32(i / W));
    color = 0;
    if (s.z > 0.) {
      const fx: f32 = floor(s.x);
      const fy: f32 = floor(s.y);
      const x0: u32 = u32(fx);
      const y0: u32 = u32(fy);
      const x1: u32 = min(x0 + 1, W - 1);
      const y1: u32 = min(y0 + 1, H - 1);
      const ax: f32 = s.x - fx;
      const ay: f32 = s.y - fy;
      const c00: u32 = tsdfColor[y0 * W + x0];
      const c10: u32 = tsdfColor[y0 * W + x1];
      const c01: u32 = tsdfColor[y1 * W + x0];
      const c11: u32 = tsdfColor[y1 * W + x1];
      for (let k: u32 = 0; k < 4; k++) {
        const value: f32 = mix(
          mix(channel(c00, k), channel(c10, k), ax),
          mix(channel(c01, k), channel(c11, k), ax),
          ay,
        );
        color = color | (min(u32(floor(value + 0.5)), 255) << (8 * k));
      }
    }
  }
  return color;
}

/** The sample of pixel `i` of the view. */
function pixelSample(i: u32): TsdfSample {
  const W: u32 = tsdfControl.width;
  const depth: f32 = bitcast<f32>(tsdfDepth[i]);
  const local: vec3 = vec3(
    (f32(i % W) - tsdfControl.cx) / tsdfControl.fx,
    (f32(i / W) - tsdfControl.cy) / tsdfControl.fy,
    1.,
  );
  const ray: vec3 = toWorldVector(local);
  const normal: vec3 = toWorldVector(decodeNormal(tsdfNormal[i]));
  const facing: f32 = -dot(normal, ray);
  const rawCosine: f32 = facing / length(ray);
  const confidence: f32 = bitcast<f32>(tsdfConfidence[i]);
  let level: u32 = 0;
  let size: f32 = tsdfControl.voxel;
  const footprint: f32 = depth * tsdfControl.footprint;
  for (let k: u32 = 1; k < tsdfControl.levels; k++) {
    if (!(footprint > size)) break;
    size = size * 2.;
    level = k;
  }
  let weight: u32 = 0;
  if (depth > 0. && rawCosine >= MIN_COSINE && confidence > 0.)
    weight = u32(floor(confidence * rawCosine * WEIGHT_SCALE + 0.5));
  let valid: u32 = 0;
  if (depth > 0.) valid = 1;
  return {
    valid: valid,
    depth: depth,
    ray: ray,
    point: cameraCenter() + ray * depth,
    normal: normal,
    facing: facing,
    cosine: max(rawCosine, MIN_COSINE),
    level: level,
    weight: weight,
    color: pixelColor(i),
  };
}

/** The half-length in t of a sample's band, μ / (cos · |D|): the plane distance is μ at its ends. */
function bandHalf(sample: TsdfSample, mu: f32): f32 {
  return mu / (sample.cosine * length(sample.ray));
}

/**
 * Claims the blocks of the band [d − μ, d + μ] of pixel `i`'s sample along its ray (section 10.4,
 * `tsdfAllocate`), within the pass's box and its margin. The walk visits the blocks the ray passes
 * through, one cube of 8 voxels of the sample's level at a time.
 */
@compute([64])
export function tsdfAllocate(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = tsdfRange.first + gid.x;
  const active: bool = i < tsdfRange.end;
  if (!active) i = max(tsdfRange.end, 1) - 1;
  const sample: TsdfSample = pixelSample(i);
  if (!active || sample.weight === 0) return;
  const level: u32 = sample.level;
  const size: f32 = levelSize(level);
  const blockSize: f32 = size * 8.;
  const inverse: f32 = levelInverse(level) * 0.125;
  const half: f32 = bandHalf(sample, tsdfControl.truncation * size);
  const o: vec3 = cameraCenter();
  const origin: vec3 = volumeOrigin();
  const end: f32 = sample.depth + half;
  const eps: f32 = 0.01 * size;
  let t: f32 = sample.depth - half;
  for (let step: u32 = 0; step < MAX_STEPS; step++) {
    if (!(t < end)) break;
    const p: vec3 = o + sample.ray * t;
    const bx: i32 = i32(floor((p.x - origin.x) * inverse));
    const by: i32 = i32(floor((p.y - origin.y) * inverse));
    const bz: i32 = i32(floor((p.z - origin.z) * inverse));
    const key: u32 = blockKey(level, bx, by, bz);
    if (key !== EMPTY && blockInBox(level, bx, by, bz, true)) {
      let slot: u32 = keySlot(key);
      let done: bool = false;
      for (let probe: u32 = 0; probe <= tsdfControl.mask + 64; probe++) {
        if (done) break;
        const claim = atomicCompareExchangeWeak(tsdfTable[HEADER + slot], EMPTY, key);
        if (claim.exchanged) {
          atomicAdd(tsdfTable[0], 1);
          done = true;
        } else if (claim.old_value === key) {
          done = true;
        } else if (claim.old_value !== EMPTY) {
          slot = (slot + 1) & tsdfControl.mask;
        }
      }
      if (!done) atomicStore(tsdfTable[2], 1);
    }
    t = max(cellExit(o, sample.ray, bx * 8, by * 8, bz * 8, 8, size), t) + eps;
  }
}

/** Gives each claimed slot `first..end` a block index in `tsdfVoxels`, and the block its slot. */
@compute([64])
export function tsdfAssign(@builtin('global_invocation_id') gid: vec3u): void {
  let s: u32 = tsdfRange.first + gid.x;
  const active: bool = s < tsdfRange.end;
  if (!active) s = max(tsdfRange.end, 1) - 1;
  const key: u32 = atomicLoad(tsdfTable[HEADER + s]);
  if (active && key !== EMPTY) {
    const block: u32 = atomicAdd(tsdfTable[1], 1);
    atomicStore(tsdfTable[HEADER + tsdfControl.mask + 1 + s], block);
    atomicStore(tsdfTable[HEADER + 2 * (tsdfControl.mask + 1) + block], s);
  }
}

/** The word index of word `k` of voxel (x, y, z) mod 8 of a block. */
function voxelWord(block: u32, x: i32, y: i32, z: i32, k: u32): u32 {
  const v: u32 = u32(x & 7) + 8 * u32(y & 7) + 64 * u32(z & 7);
  return (block * BLOCK_VOXELS + v) * VOXEL_WORDS + k;
}

/**
 * Adds one sample of the view `tsdfControl.view` to the tally at word `index` (the layout above):
 * the samples by one, and the views by one when the mark is not the view's. The views of one
 * dispatch share one mark and the views run in order, so the result does not depend on the order
 * of the lanes.
 */
function tally(index: u32): void {
  const mark: u32 = (tsdfControl.view % 4095) + 1;
  let old: u32 = atomicLoad(tsdfVoxels[index]);
  for (let k: u32 = 0; k < 4096; k++) {
    const samples: u32 = min((old & 4095) + 1, 4095);
    let views: u32 = (old >> 12) & 255;
    if (old >> 20 !== mark) views = min(views + 1, 255);
    const claim = atomicCompareExchangeWeak(tsdfVoxels[index], old, samples | (views << 12) | (mark << 20));
    if (claim.exchanged) break;
    old = claim.old_value;
  }
}

/**
 * Integrates pixel `i`'s sample (`tsdfIntegrate`): each voxel of its level that the ray passes
 * through in the band [d − μ, d + μ], in a claimed block, gains the weight w (confidence times the
 * cosine, fixed point), w times the signed distance of its center to the sample's tangent plane
 * over μ (at most 1, and nothing when it is below −1), and w times the color.
 */
@compute([64])
export function tsdfIntegrate(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = tsdfRange.first + gid.x;
  const active: bool = i < tsdfRange.end;
  if (!active) i = max(tsdfRange.end, 1) - 1;
  const sample: TsdfSample = pixelSample(i);
  if (!active || sample.weight === 0) return;
  const level: u32 = sample.level;
  const size: f32 = levelSize(level);
  const inverse: f32 = levelInverse(level);
  const mu: f32 = tsdfControl.truncation * size;
  const half: f32 = bandHalf(sample, mu);
  const o: vec3 = cameraCenter();
  const origin: vec3 = volumeOrigin();
  const end: f32 = sample.depth + half;
  const eps: f32 = 0.01 * size;
  const w: f32 = f32(sample.weight);
  const red: u32 = (sample.weight * (sample.color & 255)) >> 4;
  const green: u32 = (sample.weight * ((sample.color >> 8) & 255)) >> 4;
  const blue: u32 = (sample.weight * ((sample.color >> 16) & 255)) >> 4;
  let t: f32 = sample.depth - half;
  let cachedKey: u32 = EMPTY;
  let block: u32 = EMPTY;
  let lastX: i32 = 0;
  let lastY: i32 = 0;
  let lastZ: i32 = 0;
  let visited: bool = false;
  for (let step: u32 = 0; step < MAX_STEPS; step++) {
    if (!(t < end)) break;
    const p: vec3 = o + sample.ray * t;
    const vx: i32 = i32(floor((p.x - origin.x) * inverse));
    const vy: i32 = i32(floor((p.y - origin.y) * inverse));
    const vz: i32 = i32(floor((p.z - origin.z) * inverse));
    const key: u32 = blockKey(level, vx >> 3, vy >> 3, vz >> 3);
    if (key !== cachedKey) {
      cachedKey = key;
      block = findBlock(key);
    }
    if (block === EMPTY) {
      t = max(cellExit(o, sample.ray, vx & -8, vy & -8, vz & -8, 8, size), t) + eps;
      continue;
    }
    if (!(visited && vx === lastX && vy === lastY && vz === lastZ)) {
      const center: vec3 = origin + (vec3(f32(vx), f32(vy), f32(vz)) + 0.5) * size;
      const sdf: f32 = dot(sample.normal, center - sample.point);
      if (sdf >= -mu) {
        const s: f32 = min(sdf / mu, 1.);
        const fixed: i32 = i32(floor(w * s + 0.5));
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 0)], u32(fixed));
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 1)], sample.weight);
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 2)], red);
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 3)], green);
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 4)], blue);
        tally(voxelWord(block, vx, vy, vz, 6));
      }
    }
    visited = true;
    lastX = vx;
    lastY = vy;
    lastZ = vz;
    t = max(cellExit(o, sample.ray, vx, vy, vz, 1, size), t) + eps;
  }
}

/**
 * The free-space counts of pixel `i`'s ray (`tsdfCarve`): on each level, each voxel of a claimed
 * block that the ray passes through between the level's grid entry and μ' in front of the sample
 * (μ' the larger truncation of the sample's level and this level, along the ray as the band) gains
 * one count. It claims nothing. A sample whose integration weight is 0 cannot carve. A level more
 * than `carveSpan` levels finer than the sample's is not walked (section 10.5).
 */
@compute([64])
export function tsdfCarve(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = tsdfRange.first + gid.x;
  const active: bool = i < tsdfRange.end;
  if (!active) i = max(tsdfRange.end, 1) - 1;
  const sample: TsdfSample = pixelSample(i);
  if (!active || sample.valid === 0 || sample.weight === 0) return;
  const o: vec3 = cameraCenter();
  const origin: vec3 = volumeOrigin();
  const D: vec3 = sample.ray;
  const coarse: f32 = levelSize(tsdfControl.levels - 1) * 8.;
  for (let level: u32 = 0; level < tsdfControl.levels; level++) {
    if (level + tsdfControl.carveSpan < sample.level) continue;
    const size: f32 = levelSize(level);
    const inverse: f32 = levelInverse(level);
    const blockSize: f32 = size * 8.;
    const mu: f32 = tsdfControl.truncation * levelSize(max(level, sample.level));
    // The ray's entry to the level's grid of blocks, and its exit from the grid within the pass's
    // box and its margin (no block of the pass lies past them). The walk starts at the grid's entry
    // in every pass, so its voxels, and the free counts of a box, do not depend on the boxes (the
    // jumps over the blocks a pass does not hold land where one pass's walk lands).
    const gridLo: vec3 =
      origin +
      vec3(
        f32(i32(gridWord(level, 0))),
        f32(i32(gridWord(level, 1))),
        f32(i32(gridWord(level, 2))),
      ) *
        blockSize;
    const gridHi: vec3 =
      gridLo +
      vec3(f32(gridWord(level, 3)), f32(gridWord(level, 4)), f32(gridWord(level, 5))) * blockSize;
    const lo: vec3 = max(gridLo, origin + boxCorner(false) * coarse);
    const hi: vec3 = min(gridHi, origin + boxCorner(true) * coarse);
    let enter: f32 = 0.;
    let boxEnter: f32 = 0.;
    let leave: f32 = sample.depth - bandHalf(sample, mu);
    if (D.x !== 0.) {
      enter = max(enter, min((gridLo.x - o.x) / D.x, (gridHi.x - o.x) / D.x));
      boxEnter = max(boxEnter, min((lo.x - o.x) / D.x, (hi.x - o.x) / D.x));
      leave = min(leave, max((lo.x - o.x) / D.x, (hi.x - o.x) / D.x));
    }
    if (D.y !== 0.) {
      enter = max(enter, min((gridLo.y - o.y) / D.y, (gridHi.y - o.y) / D.y));
      boxEnter = max(boxEnter, min((lo.y - o.y) / D.y, (hi.y - o.y) / D.y));
      leave = min(leave, max((lo.y - o.y) / D.y, (hi.y - o.y) / D.y));
    }
    if (D.z !== 0.) {
      enter = max(enter, min((gridLo.z - o.z) / D.z, (gridHi.z - o.z) / D.z));
      boxEnter = max(boxEnter, min((lo.z - o.z) / D.z, (hi.z - o.z) / D.z));
      leave = min(leave, max((lo.z - o.z) / D.z, (hi.z - o.z) / D.z));
    }
    // A ray that misses the pass's region carves nothing in this pass.
    if (!(boxEnter < leave)) continue;
    const eps: f32 = 0.01 * size;
    let t: f32 = enter;
    let cachedKey: u32 = EMPTY;
    let block: u32 = EMPTY;
    let lastX: i32 = 0;
    let lastY: i32 = 0;
    let lastZ: i32 = 0;
    let visited: bool = false;
    for (let step: u32 = 0; step < MAX_STEPS; step++) {
      if (!(t < leave)) break;
      const p: vec3 = o + D * t;
      const vx: i32 = i32(floor((p.x - origin.x) * inverse));
      const vy: i32 = i32(floor((p.y - origin.y) * inverse));
      const vz: i32 = i32(floor((p.z - origin.z) * inverse));
      const key: u32 = blockKey(level, vx >> 3, vy >> 3, vz >> 3);
      if (key !== cachedKey) {
        cachedKey = key;
        block = findBlock(key);
      }
      if (block === EMPTY) {
        t = max(cellExit(o, D, vx & -8, vy & -8, vz & -8, 8, size), t) + eps;
        continue;
      }
      if (!(visited && vx === lastX && vy === lastY && vz === lastZ))
        atomicAdd(tsdfVoxels[voxelWord(block, vx, vy, vz, 5)], 1);
      visited = true;
      lastX = vx;
      lastY = vy;
      lastZ = vz;
      t = max(cellExit(o, D, vx, vy, vz, 1, size), t) + eps;
    }
  }
}

/** Scratch votes are reset between spatial camera groups. */
@compute([64])
export function tsdfResetVotes(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = tsdfRange.first + gid.x;
  if (i < tsdfRange.end) tsdfVotes[i] = 0;
}

@compute([64])
export function tsdfViewCarve(@builtin('global_invocation_id') gid: vec3u): void {
  const e: u32 = tsdfRange.first + gid.x;
  if (e >= tsdfRange.end) return;
  const block: u32 = e / BLOCK_VOXELS;
  const base: u32 = e * VOXEL_WORDS;
  if (atomicLoad(tsdfVoxels[base + 1]) === 0) return;
  const slot: u32 = atomicLoad(tsdfTable[HEADER + 2 * (tsdfControl.mask + 1) + block]);
  const key: u32 = atomicLoad(tsdfTable[HEADER + slot]);
  if (key === EMPTY) return;
  let level: u32 = 0;
  for (let k: u32 = 1; k < tsdfControl.levels; k++) if (key >= gridWord(k, 6)) level = k;
  const nx: u32 = gridWord(level, 3);
  const ny: u32 = gridWord(level, 4);
  const index: u32 = key - gridWord(level, 6);
  const bx: i32 = i32(gridWord(level, 0)) + i32(index % nx);
  const by: i32 = i32(gridWord(level, 1)) + i32((index / nx) % ny);
  const bz: i32 = i32(gridWord(level, 2)) + i32(index / (nx * ny));
  const local: u32 = e % BLOCK_VOXELS;
  const coord: vec3 = vec3(f32(bx * 8 + i32(local % 8)), f32(by * 8 + i32((local / 8) % 8)), f32(bz * 8 + i32(local / 64)));
  const X: vec3 = toCameraVector(volumeOrigin() + (coord + 0.5) * levelSize(level) - cameraCenter());
  if (!(X.z > 0.)) return;
  const q: vec2 = vec2(tsdfControl.fx * X.x / X.z + tsdfControl.cx, tsdfControl.fy * X.y / X.z + tsdfControl.cy);
  const W: u32 = tsdfControl.width;
  const H: u32 = tsdfControl.height;
  if (!(q.x >= 0. && q.y >= 0. && q.x < f32(W - 1) && q.y < f32(H - 1))) return;
  const x0: u32 = u32(floor(q.x));
  const y0: u32 = u32(floor(q.y));
  let safeDepth: f32 = FAR;
  let vote: u32 = 256;
  for (let dy: u32 = 0; dy < 2; dy++) for (let dx: u32 = 0; dx < 2; dx++) {
    const i: u32 = (y0 + dy) * W + x0 + dx;
    const d: f32 = bitcast<f32>(tsdfDepth[i]);
    const confidence: f32 = clamp(bitcast<f32>(tsdfConfidence[i]), 0., 1.);
    if (!(d > 0. && confidence > 0.)) return;
    const ray: vec3 = vec3((f32(x0 + dx) - tsdfControl.cx) / tsdfControl.fx, (f32(y0 + dy) - tsdfControl.cy) / tsdfControl.fy, 1.);
    const facing: f32 = -dot(decodeNormal(tsdfNormal[i]), ray);
    const cosine: f32 = facing / length(ray);
    if (!(cosine >= MIN_COSINE)) return;
    let sampleLevel: u32 = 0;
    let size: f32 = tsdfControl.voxel;
    for (let k: u32 = 1; k < tsdfControl.levels; k++) {
      if (!(d * tsdfControl.footprint > size)) break;
      size *= 2.;
      sampleLevel = k;
    }
    if (level + tsdfControl.carveSpan < sampleLevel) return;
    const mu: f32 = tsdfControl.truncation * levelSize(max(level, sampleLevel));
    safeDepth = min(safeDepth, d - mu * (2. - confidence) / facing);
    vote = min(vote, u32(floor(confidence * cosine * 256. + 0.5)));
  }
  const halfZ: f32 = 0.5 * levelSize(level) * (abs(tsdfControl.r6) + abs(tsdfControl.r7) + abs(tsdfControl.r8));
  if (vote === 0 || !(X.z + halfZ < safeDepth)) return;
  const previous: u32 = tsdfVotes[e];
  if (vote <= previous) return;
  tsdfVotes[e] = vote;
  const old: u32 = atomicLoad(tsdfVoxels[base + 5]);
  let added: u32 = 0;
  if (previous === 0) added = 1;
  atomicStore(tsdfVoxels[base + 5], (min(4095, (old >> 20) + added) << 20) | min(0xfffff, (old & 0xfffff) + vote - previous));
}

/** A voxel's state for the extraction. */
class TsdfVoxel {
  // 1 when the voxel has at least the least weight and is not carved.
  kept: u32;
  // The mean of sdf / μ.
  sdf: f32;
  block: u32;
}

/** Word `k` of a voxel of a known block. */
function voxelValue(block: u32, x: i32, y: i32, z: i32, k: u32): u32 {
  return atomicLoad(tsdfVoxels[voxelWord(block, x, y, z, k)]);
}
/** The state of voxel (x, y, z) of `level`: kept when its weight is at least the least weight, its
 *  views at least the least views, and its free count at most the carve ratio times its weight. */
function voxelAt(level: u32, x: i32, y: i32, z: i32): TsdfVoxel {
  const block: u32 = findBlock(blockKey(level, x >> 3, y >> 3, z >> 3));
  let kept: u32 = 0;
  let sdf: f32 = 0.;
  if (block !== EMPTY) {
    const weight: u32 = voxelValue(block, x, y, z, 1);
    const free: u32 = voxelValue(block, x, y, z, 5);
    const views: u32 = (voxelValue(block, x, y, z, 6) >> 12) & 255;
    const samples: u32 = voxelValue(block, x, y, z, 6) & 4095;
    let carved: bool = f32(free) * WEIGHT_SCALE > tsdfControl.carveRatio * f32(weight);
    if (tsdfControl.visibility !== 0)
      carved = (free >> 20) >= 2 && f32(free & 0xfffff) * WEIGHT_SCALE * f32(max(samples, 1)) >
        tsdfControl.carveRatio * f32(weight) * 256. * f32(max(views, 1));
    if (
      weight >= tsdfControl.minWeight &&
      weight > 0 &&
      views >= tsdfControl.minViews &&
      !carved
    ) {
      kept = 1;
      sdf = f32(i32(voxelValue(block, x, y, z, 0))) / f32(weight);
    }
  }
  return { kept: kept, sdf: sdf, block: block };
}

/**
 * The surface points of the voxels `first..end` (`tsdfExtract`): element e is voxel e mod 512 of
 * block e / 512. A kept voxel of a block of the pass's box emits a point at the zero
 * crossing toward each +x, +y, +z neighbour that is kept and whose mean signed distance has the
 * other sign, within 1 of its own (a larger jump is a truncation edge, not a surface). It emits
 * nothing when a finer level keeps the voxel of its center. The normal is the gradient of the
 * signed distance (a forward difference, a backward one within the block when the forward
 * neighbour is not kept, 0 otherwise), and the color the voxel's weighted mean. With `write` 0
 * the kernel counts the points, with 1 it writes them.
 */
@compute([64])
export function tsdfExtract(@builtin('global_invocation_id') gid: vec3u): void {
  let e: u32 = tsdfRange.first + gid.x;
  const active: bool = e < tsdfRange.end;
  if (!active) e = max(tsdfRange.end, 1) - 1;
  const slot: u32 = atomicLoad(tsdfTable[HEADER + 2 * (tsdfControl.mask + 1) + e / BLOCK_VOXELS]);
  const local: u32 = e % BLOCK_VOXELS;
  const key: u32 = atomicLoad(tsdfTable[HEADER + slot]);
  if (!active || key === EMPTY) return;
  // The level and the block coordinates of the key.
  let level: u32 = 0;
  for (let k: u32 = 1; k < tsdfControl.levels; k++) if (key >= gridWord(k, 6)) level = k;
  const nx: u32 = gridWord(level, 3);
  const ny: u32 = gridWord(level, 4);
  const index: u32 = key - gridWord(level, 6);
  const bx: i32 = i32(gridWord(level, 0)) + i32(index % nx);
  const by: i32 = i32(gridWord(level, 1)) + i32((index / nx) % ny);
  const bz: i32 = i32(gridWord(level, 2)) + i32(index / (nx * ny));
  if (!blockInBox(level, bx, by, bz, false)) return;
  const x: i32 = bx * 8 + i32(local % 8);
  const y: i32 = by * 8 + i32((local / 8) % 8);
  const z: i32 = bz * 8 + i32(local / 64);
  // The voxel's own block is e / 512: a voxel below the least weight leaves before any lookup.
  if (voxelValue(e / BLOCK_VOXELS, x, y, z, 1) < max(tsdfControl.minWeight, 1)) return;
  const self: TsdfVoxel = voxelAt(level, x, y, z);
  if (self.kept === 0) return;
  // A finer level that keeps the voxel at this voxel's center takes the surface there.
  for (let fine: u32 = 0; fine < level; fine++) {
    const shift: u32 = level - fine;
    const half: i32 = 1 << (shift - 1);
    if (voxelAt(fine, (x << shift) + half, (y << shift) + half, (z << shift) + half).kept !== 0)
      return;
  }
  const size: f32 = levelSize(level);
  const center: vec3 = volumeOrigin() + (vec3(f32(x), f32(y), f32(z)) + 0.5) * size;
  // The neighbours: +x, +y, +z, then −x, −y, −z when inside the block.
  const px: TsdfVoxel = voxelAt(level, x + 1, y, z);
  const py: TsdfVoxel = voxelAt(level, x, y + 1, z);
  const pz: TsdfVoxel = voxelAt(level, x, y, z + 1);
  let gx: f32 = 0.;
  let gy: f32 = 0.;
  let gz: f32 = 0.;
  if (px.kept !== 0) gx = px.sdf - self.sdf;
  else if ((x & 7) !== 0) {
    const m: TsdfVoxel = voxelAt(level, x - 1, y, z);
    if (m.kept !== 0) gx = self.sdf - m.sdf;
  }
  if (py.kept !== 0) gy = py.sdf - self.sdf;
  else if ((y & 7) !== 0) {
    const m: TsdfVoxel = voxelAt(level, x, y - 1, z);
    if (m.kept !== 0) gy = self.sdf - m.sdf;
  }
  if (pz.kept !== 0) gz = pz.sdf - self.sdf;
  else if ((z & 7) !== 0) {
    const m: TsdfVoxel = voxelAt(level, x, y, z - 1);
    if (m.kept !== 0) gz = self.sdf - m.sdf;
  }
  const weight: u32 = voxelValue(self.block, x, y, z, 1);
  const scale: f32 = 16. / f32(weight);
  const red: u32 = min(u32(floor(f32(voxelValue(self.block, x, y, z, 2)) * scale + 0.5)), 255);
  const green: u32 = min(u32(floor(f32(voxelValue(self.block, x, y, z, 3)) * scale + 0.5)), 255);
  const blue: u32 = min(u32(floor(f32(voxelValue(self.block, x, y, z, 4)) * scale + 0.5)), 255);
  const color: u32 = red | (green << 8) | (blue << 16) | (u32(255) << 24);
  for (let axis: u32 = 0; axis < 3; axis++) {
    let other: TsdfVoxel = px;
    let step: vec3 = vec3(1., 0., 0.);
    if (axis === 1) {
      other = py;
      step = vec3(0., 1., 0.);
    }
    if (axis === 2) {
      other = pz;
      step = vec3(0., 0., 1.);
    }
    if (other.kept === 0) continue;
    const a: f32 = self.sdf;
    const b: f32 = other.sdf;
    if ((a >= 0.) === (b >= 0.)) continue;
    if (abs(a - b) > 1.) continue;
    const point: vec3 = center + step * (size * (a / (a - b)));
    const normal: vec3 = normalize(vec3(gx, gy, gz));
    if (tsdfControl.write === 0) {
      atomicAdd(tsdfTable[3], 1);
    } else {
      const row: u32 = atomicAdd(tsdfTable[4], 1);
      if (row < tsdfControl.capacity) {
        tsdfPoints[row * 3] = point.x;
        tsdfPoints[row * 3 + 1] = point.y;
        tsdfPoints[row * 3 + 2] = point.z;
        tsdfPointData[row * 5] = encodeNormal(normal);
        tsdfPointData[row * 5 + 1] = color;
        tsdfPointData[row * 5 + 2] = weight;
        tsdfPointData[row * 5 + 3] = voxelValue(self.block, x, y, z, 5);
        tsdfPointData[row * 5 + 4] = voxelValue(self.block, x, y, z, 6);
      }
    }
  }
}
