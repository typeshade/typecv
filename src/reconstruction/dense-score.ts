/**
 * Scores a dense point cloud against a ground-truth point cloud (`docs/dense-mvs.md`, section 10,
 * step 0). The definitions are those of the ETH3D and Tanks and Temples benchmarks, at stated
 * thresholds τ:
 *
 * - accuracy(τ): the fraction of reconstructed points with a ground-truth point within τ.
 * - completeness(τ): the fraction of ground-truth points with a reconstructed point within τ.
 * - F-score(τ): 2 · accuracy · completeness / (accuracy + completeness).
 *
 * Two inputs make the scores fair to a laser scan that does not cover the whole scene: the
 * ground truth is limited to the points that at least two cameras see (`visibleGroundTruth`), and
 * the reconstructed points far from the ground truth (a background that the scan does not cover)
 * are counted apart (`maskRadius`) and left out of the accuracy. This is a local approximation of
 * the benchmarks' occlusion and free-space masks, not their evaluation program.
 *
 * `landmarkCoverage` is the regression metric of the run `dfc97665`: the fraction of sparse
 * landmarks that have no dense point within a radius.
 *
 * Nothing here runs in the product. The functions serve `scripts/eth3d-dense.mjs` and the tests.
 */
import { buildPointGrid, hashCell, type PointGrid } from './cloud-clean.ts';

// ---------------------------------------------------------------------------------------------
// Reading a PLY

/** The positions and the optional colors of a PLY file with x, y, z vertices. */
export interface PlyCloud {
  readonly count: number;
  readonly positions: Float32Array;
  /** r, g, b per point, or undefined when the file has no color. */
  readonly colors?: Uint8Array;
}

const PLY_SIZES: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

/** Reads the first element (`vertex`) of an ASCII or binary little-endian PLY. */
export function readPly(bytes: Uint8Array): PlyCloud {
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 8192)));
  const marker = head.indexOf('end_header');
  if (marker < 0) throw new Error('PLY: no end_header');
  const lineEnd = head.indexOf('\n', marker);
  const body = lineEnd + 1;
  const lines = head.slice(0, marker).split('\n');
  const format = /^format (\S+)/m.exec(head)?.[1];
  if (format !== 'ascii' && format !== 'binary_little_endian')
    throw new Error(`PLY: unsupported format ${format}`);
  let count = 0,
    inVertex = false;
  const props: { name: string; type: string }[] = [];
  for (const line of lines) {
    const w = line.trim().split(/\s+/);
    if (w[0] === 'element') {
      if (inVertex) break;
      inVertex = w[1] === 'vertex';
      if (inVertex) count = Number(w[2]);
      else if (props.length === 0) throw new Error('PLY: the vertex element must come first');
    } else if (w[0] === 'property' && inVertex) {
      if (w[1] === 'list') throw new Error('PLY: list property in vertex');
      props.push({ name: w[2]!, type: w[1]! });
    }
  }
  const index = (n: string) => props.findIndex((p) => p.name === n);
  const ix = index('x'),
    iy = index('y'),
    iz = index('z');
  if (ix < 0 || iy < 0 || iz < 0) throw new Error('PLY: no x, y, z');
  const ir = index('red') >= 0 ? index('red') : index('r');
  const ig = index('green') >= 0 ? index('green') : index('g');
  const ib = index('blue') >= 0 ? index('blue') : index('b');
  const hasColor = ir >= 0 && ig >= 0 && ib >= 0;
  const positions = new Float32Array(3 * count);
  const colors = hasColor ? new Uint8Array(3 * count) : undefined;
  if (format === 'ascii') {
    const rows = new TextDecoder().decode(bytes.subarray(body)).split('\n');
    for (let i = 0; i < count; i++) {
      const w = rows[i]!.trim().split(/\s+/);
      positions[3 * i] = Number(w[ix]);
      positions[3 * i + 1] = Number(w[iy]);
      positions[3 * i + 2] = Number(w[iz]);
      if (colors) {
        colors[3 * i] = Number(w[ir]);
        colors[3 * i + 1] = Number(w[ig]);
        colors[3 * i + 2] = Number(w[ib]);
      }
    }
  } else {
    const offsets: number[] = [];
    let stride = 0;
    for (const p of props) {
      offsets.push(stride);
      const size = PLY_SIZES[p.type];
      if (!size) throw new Error(`PLY: unknown type ${p.type}`);
      stride += size;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset + body, count * stride);
    const read = (at: number, type: string) =>
      PLY_SIZES[type] === 8
        ? view.getFloat64(at, true)
        : type.startsWith('float')
          ? view.getFloat32(at, true)
          : type === 'uchar' || type === 'uint8'
            ? view.getUint8(at)
            : type === 'char' || type === 'int8'
              ? view.getInt8(at)
              : PLY_SIZES[type] === 2
                ? type.startsWith('u')
                  ? view.getUint16(at, true)
                  : view.getInt16(at, true)
                : type.startsWith('u')
                  ? view.getUint32(at, true)
                  : view.getInt32(at, true);
    for (let i = 0; i < count; i++) {
      const at = i * stride;
      positions[3 * i] = read(at + offsets[ix]!, props[ix]!.type);
      positions[3 * i + 1] = read(at + offsets[iy]!, props[iy]!.type);
      positions[3 * i + 2] = read(at + offsets[iz]!, props[iz]!.type);
      if (colors) {
        colors[3 * i] = read(at + offsets[ir]!, props[ir]!.type);
        colors[3 * i + 1] = read(at + offsets[ig]!, props[ig]!.type);
        colors[3 * i + 2] = read(at + offsets[ib]!, props[ib]!.type);
      }
    }
  }
  return colors ? { count, positions, colors } : { count, positions };
}

// ---------------------------------------------------------------------------------------------
// Nearest neighbour within a radius

/** A grid whose cell is the query radius: a query visits 27 cells. */
export interface RadiusIndex {
  readonly grid: PointGrid;
  readonly radius: number;
}

export function buildRadiusIndex(positions: Float32Array, radius: number): RadiusIndex {
  if (!(radius > 0)) throw new RangeError('The radius must be positive');
  return { grid: buildPointGrid(positions, radius), radius };
}

/**
 * The squared distance from (x, y, z) to the nearest indexed point, or Infinity when none lies
 * within the radius of the index. Exact: a point within the radius lies in one of the 27 cells.
 */
export function nearestWithin(index: RadiusIndex, x: number, y: number, z: number): number {
  const { grid, radius } = index;
  const h = grid.cellSize;
  const cx = Math.floor((x - grid.origin[0]) / h),
    cy = Math.floor((y - grid.origin[1]) / h),
    cz = Math.floor((z - grid.origin[2]) / h);
  const p = grid.points;
  let best = radius * radius;
  let found = false;
  for (let dz = -1; dz <= 1; dz++)
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const ax = cx + dx,
          ay = cy + dy,
          az = cz + dz;
        if (ax < 1 || ay < 1 || az < 1) continue;
        const b = hashCell(ax, ay, az) & grid.mask;
        const end = grid.bucketStart[b + 1]!;
        for (let j = grid.bucketStart[b]!; j < end; j++) {
          if (
            grid.cells[3 * j] !== ax ||
            grid.cells[3 * j + 1] !== ay ||
            grid.cells[3 * j + 2] !== az
          )
            continue;
          const ex = p[3 * j]! - x,
            ey = p[3 * j + 1]! - y,
            ez = p[3 * j + 2]! - z;
          const d2 = ex * ex + ey * ey + ez * ez;
          if (d2 <= best) {
            best = d2;
            found = true;
          }
        }
      }
  return found ? best : Infinity;
}

/** The distance of each point of `query` to the nearest point of `reference`, Infinity past `max`. */
export function distancesToCloud(
  query: Float32Array,
  reference: Float32Array,
  max: number,
): Float32Array {
  const index = buildRadiusIndex(reference, max);
  const n = query.length / 3;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++)
    out[i] = Math.sqrt(nearestWithin(index, query[3 * i]!, query[3 * i + 1]!, query[3 * i + 2]!));
  return out;
}

// ---------------------------------------------------------------------------------------------
// The scores

export interface ScoreRow {
  /** The threshold τ, in the units of the clouds. */
  readonly tau: number;
  readonly accuracy: number;
  readonly completeness: number;
  readonly fscore: number;
}

export interface CloudScore {
  /** The reconstructed points, the ground-truth points and the points the accuracy counts. */
  readonly points: number;
  readonly groundTruth: number;
  readonly judged: number;
  /** The reconstructed points with no ground truth within `maskRadius`, left out of the accuracy. */
  readonly outsideMask: number;
  readonly maskRadius: number;
  readonly rows: readonly ScoreRow[];
  /** The mean distance to the ground truth of the judged points within the largest τ. */
  readonly meanAccuracyDistance: number;
}

export interface ScoreOptions {
  /** The thresholds τ. */
  readonly taus: readonly number[];
  /** A reconstructed point farther than this from every ground-truth point is not judged. */
  readonly maskRadius: number;
}

export function fscore(accuracy: number, completeness: number): number {
  return accuracy + completeness > 0
    ? (2 * accuracy * completeness) / (accuracy + completeness)
    : 0;
}

/**
 * Accuracy, completeness and F-score of `cloud` against `groundTruth` at each τ. `maskRadius` is
 * at least the largest τ. An empty cloud scores 0 everywhere.
 */
export function scoreCloud(
  cloud: Float32Array,
  groundTruth: Float32Array,
  options: ScoreOptions,
): CloudScore {
  const taus = [...options.taus].sort((a, b) => a - b);
  const top = taus[taus.length - 1]!;
  const maskRadius = Math.max(options.maskRadius, top);
  const n = cloud.length / 3,
    m = groundTruth.length / 3;
  // Accuracy: the distance of each reconstructed point to the ground truth, up to the mask.
  const toTruth = n && m ? distancesToCloud(cloud, groundTruth, maskRadius) : new Float32Array(n);
  let judged = 0;
  const accurate = new Float64Array(taus.length);
  let sum = 0,
    summed = 0;
  for (let i = 0; i < n; i++) {
    const d = n && m ? toTruth[i]! : Infinity;
    if (!Number.isFinite(d)) continue;
    judged++;
    if (d <= top) {
      sum += d;
      summed++;
    }
    for (let t = 0; t < taus.length; t++) if (d <= taus[t]!) accurate[t]!++;
  }
  // Completeness: the distance of each ground-truth point to the reconstruction, up to τ max.
  const toCloud = n && m ? distancesToCloud(groundTruth, cloud, top) : new Float32Array(m);
  const complete = new Float64Array(taus.length);
  for (let i = 0; i < m; i++) {
    const d = n && m ? toCloud[i]! : Infinity;
    for (let t = 0; t < taus.length; t++) if (d <= taus[t]!) complete[t]!++;
  }
  const rows = taus.map((tau, t) => {
    const accuracy = judged ? accurate[t]! / judged : 0;
    const completeness = m ? complete[t]! / m : 0;
    return { tau, accuracy, completeness, fscore: fscore(accuracy, completeness) };
  });
  return {
    points: n,
    groundTruth: m,
    judged,
    outsideMask: n - judged,
    maskRadius,
    rows,
    meanAccuracyDistance: summed ? sum / summed : NaN,
  };
}

// ---------------------------------------------------------------------------------------------
// The visible part of the ground truth

/** A pinhole camera at the size of `width` × `height`, world to camera x_c = R x_w + t. */
export interface ScoreCamera {
  readonly R: ArrayLike<number>;
  readonly t: ArrayLike<number>;
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
  readonly width: number;
  readonly height: number;
}

export interface VisibilityOptions {
  /** A ground-truth point counts as seen by a camera when it is within this fraction of the depth buffer. Default 0.02. */
  readonly depthTolerance?: number;
  /** The depth buffer has this many pixels per cell side. Default 4. */
  readonly cell?: number;
  /** The least number of cameras that see a point. Default 2. */
  readonly minViews?: number;
}

/**
 * The ground-truth points that at least `minViews` cameras see: the point projects inside the
 * image, in front of the camera, and no nearer ground-truth point covers it in a depth buffer of
 * `cell` × `cell` pixel cells (minimum depth per cell, then the nine cells around it).
 * Returns the indices into `groundTruth`.
 */
export function visibleGroundTruth(
  groundTruth: Float32Array,
  cameras: readonly ScoreCamera[],
  options: VisibilityOptions = {},
): Uint32Array {
  const tol = options.depthTolerance ?? 0.02,
    cell = options.cell ?? 4,
    minViews = options.minViews ?? 2;
  const m = groundTruth.length / 3;
  const seen = new Uint8Array(m);
  for (const c of cameras) {
    const gw = Math.ceil(c.width / cell),
      gh = Math.ceil(c.height / cell);
    const z = new Float32Array(m);
    const u = new Int32Array(m).fill(-1);
    const buffer = new Float32Array(gw * gh).fill(Infinity);
    for (let i = 0; i < m; i++) {
      const X = groundTruth[3 * i]!,
        Y = groundTruth[3 * i + 1]!,
        Z = groundTruth[3 * i + 2]!;
      const d = c.R[6]! * X + c.R[7]! * Y + c.R[8]! * Z + c.t[2]!;
      if (!(d > 0)) continue;
      const px = (c.fx * (c.R[0]! * X + c.R[1]! * Y + c.R[2]! * Z + c.t[0]!)) / d + c.cx,
        py = (c.fy * (c.R[3]! * X + c.R[4]! * Y + c.R[5]! * Z + c.t[1]!)) / d + c.cy;
      if (px < 0 || py < 0 || px >= c.width || py >= c.height) continue;
      const g = Math.floor(py / cell) * gw + Math.floor(px / cell);
      z[i] = d;
      u[i] = g;
      if (d < buffer[g]!) buffer[g] = d;
    }
    for (let i = 0; i < m; i++) {
      const g = u[i]!;
      if (g < 0) continue;
      const gx = g % gw,
        gy = (g - gx) / gw;
      let nearest = Infinity;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const x = gx + dx,
            y = gy + dy;
          if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
          nearest = Math.min(nearest, buffer[y * gw + x]!);
        }
      if (z[i]! <= nearest * (1 + tol)) seen[i]!++;
    }
  }
  const out: number[] = [];
  for (let i = 0; i < m; i++) if (seen[i]! >= minViews) out.push(i);
  return Uint32Array.from(out);
}

/** The positions of `groundTruth` at `indices`. */
export function takePositions(positions: Float32Array, indices: ArrayLike<number>): Float32Array {
  const out = new Float32Array(3 * indices.length);
  for (let i = 0; i < indices.length; i++)
    for (let a = 0; a < 3; a++) out[3 * i + a] = positions[3 * indices[i]! + a]!;
  return out;
}

/** One point per cube of side `size` (the first in input order), to bound a laser scan's size. */
export function decimateByVoxel(positions: Float32Array, size: number): Float32Array {
  const n = positions.length / 3;
  const seen = new Set<number>();
  const keep: number[] = [];
  const HALF = 65536;
  for (let i = 0; i < n; i++) {
    const ix = Math.floor(positions[3 * i]! / size) + HALF,
      iy = Math.floor(positions[3 * i + 1]! / size) + HALF,
      iz = Math.floor(positions[3 * i + 2]! / size) + HALF;
    if (!(ix >= 0 && iy >= 0 && iz >= 0 && ix < 2 * HALF && iy < 2 * HALF && iz < 2 * HALF))
      throw new RangeError('The cloud spans more than 131072 voxels on an axis');
    const key = (ix * 2 * HALF + iy) * 2 * HALF + iz;
    if (seen.has(key)) continue;
    seen.add(key);
    keep.push(i);
  }
  return takePositions(positions, keep);
}

// ---------------------------------------------------------------------------------------------
// The regression metric of the run dfc97665

export interface CoverageLandmark {
  readonly xyz: ArrayLike<number>;
  /** The radius of this landmark, in the units of the cloud. */
  readonly radius: number;
}

export interface CoverageResult {
  readonly landmarks: number;
  /** The landmarks with no dense point within their radius. */
  readonly missing: number;
  readonly missingFraction: number;
}

/**
 * The fraction of sparse landmarks with no dense point within their own radius. The radius is
 * k × the expected spacing at the landmark (`expectedSpacing`), so a far landmark, where the
 * dense points lie wider apart, is held to a wider radius than a near one.
 */
export function landmarkCoverage(
  cloud: Float32Array,
  landmarks: readonly CoverageLandmark[],
): CoverageResult {
  if (landmarks.length === 0) return { landmarks: 0, missing: 0, missingFraction: 0 };
  const widest = landmarks.reduce((a, l) => Math.max(a, l.radius), 0);
  let missing = 0;
  if (cloud.length === 0) missing = landmarks.length;
  else {
    const index = buildRadiusIndex(cloud, widest);
    for (const l of landmarks) {
      const d2 = nearestWithin(index, l.xyz[0]!, l.xyz[1]!, l.xyz[2]!);
      if (!(d2 <= l.radius * l.radius)) missing++;
    }
  }
  return { landmarks: landmarks.length, missing, missingFraction: missing / landmarks.length };
}

/**
 * The expected spacing of dense points at a landmark: the footprint of one pixel at its depth,
 * depth / focal, in the nearest view that observes it (`depth` in world units, `focal` in pixels
 * at the analysis size).
 */
export function expectedSpacing(depth: number, focal: number): number {
  return depth / focal;
}
