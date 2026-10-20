/**
 * A triangle mesh from the cleaned dense cloud (`docs/dense-mvs.md`, section 3.13): a signed
 * distance field on a sparse lattice from the oriented points, then marching cubes (Lorensen and
 * Cline, with the 256-case table of `marching-cubes-tables.ts`), welded into an indexed mesh with a
 * color per vertex. This file is the CPU reference. The GPU tier in `src/gpu/dense-mesh.ts` runs
 * the field, the cube classification and the cube emission as TypeShade kernels, and shares the
 * lattice, the weld and the assembly with this file, so the two tiers give the same triangles.
 *
 * The field. A lattice node x takes the points p within the truncation radius R of it. Each point
 * with unit normal n gives the signed distance d = n · (x − p) to its tangent plane, positive on
 * the side the normal points to, and the weight w = (1 − |x − p|² / R²)². The field is Σ w d / Σ w
 * and the node color is Σ w c / Σ w. A node with Σ w below `minWeight`, or with no point within R,
 * has no value, and a cube with such a corner makes no triangle. So the mesh stays inside the
 * truncation band of the points and leaves a hole where the cloud has none.
 *
 * The lattice. Nodes sit at origin + (i, j, k) · h. The lattice is sparse: the nodes are grouped
 * in bricks of 8 × 8 × 8, and only a brick with a node within R of a point is stored. A node's
 * address is its brick's slot (the rank of the brick among the stored ones, by brick index) times
 * 512 plus its place in the brick. A node in a brick that is not stored has no value.
 *
 * The cubes. A cube is inside where the field is below 0. A cube with all eight corner values
 * makes up to five triangles, with vertices on the crossed edges at the linear zero of the field,
 * and the color interpolated the same way. A vertex is named by its edge, the address of the
 * lower node of the edge times 3 plus its axis, so two cubes that share an edge share the vertex.
 */
import {
  boundingBox,
  buildPointGrid,
  decodeOctahedral,
  medianPointSpacing,
  type CleanCloud,
  type PointGrid,
} from './cloud-clean.ts';
import { MC_EDGE_CORNERS, MC_TRI_TABLE } from './marching-cubes-tables.ts';

export interface MeshOptions {
  /** The lattice spacing. Default `spacingFactor` times the median point spacing. */
  readonly voxelSize?: number;
  /** The default lattice spacing as a multiple of the median point spacing. Default 1.5. */
  readonly spacingFactor?: number;
  /** The truncation radius R in lattice spacings. Default 2.5. */
  readonly truncation?: number;
  /** The least Σ w for a node to have a value. Default 1. */
  readonly minWeight?: number;
}

export const MESH_DEFAULTS = { spacingFactor: 1.5, truncation: 2.5, minWeight: 1 } as const;

/** Nodes per brick edge, and per brick. */
export const BRICK = 8;
export const BRICK_NODES = BRICK * BRICK * BRICK;

/** The most nodes in the stored bricks and the most cells of the brick grid. The spacing grows to fit. */
export const MESH_MAX_NODES = 1 << 24;
export const MESH_MAX_BRICK_CELLS = 1 << 22;

/** The field value of a node without a value. */
export const MESH_NO_VALUE = 1e30;

/** The slot of a brick that is not stored. */
export const NO_BRICK = 0xffffffff;

/** An indexed triangle mesh with a color per vertex. */
export interface DenseMesh {
  readonly vertexCount: number;
  readonly faceCount: number;
  /** x, y, z per vertex. */
  readonly positions: Float32Array;
  /** r, g, b per vertex. */
  readonly colors: Uint8Array;
  /** Three vertex indices per face. The faces turn counter-clockwise seen from the positive side. */
  readonly indices: Uint32Array;
}

export interface MeshReport {
  readonly tier: 'cpu' | 'webgpu';
  readonly points: number;
  /** The lattice spacing and the truncation radius used. */
  readonly voxelSize: number;
  readonly radius: number;
  /** The lattice size in nodes. */
  readonly dims: readonly [number, number, number];
  /** The stored bricks, the nodes within R of a point, and the cubes that make triangles. */
  readonly bricks: number;
  readonly nodes: number;
  readonly cubes: number;
  readonly vertices: number;
  readonly faces: number;
  /** Milliseconds of each stage. */
  readonly ms: {
    readonly lattice: number;
    readonly field: number;
    readonly cubes: number;
    readonly vertices: number;
  };
}

export interface DenseMeshResult {
  readonly mesh: DenseMesh;
  readonly report: MeshReport;
}

/** The normals of a cloud as x, y, z floats, in input order. */
export function cloudNormals(cloud: CleanCloud): Float32Array {
  if (!cloud.normals) throw new RangeError('The mesh needs the normals of the cloud');
  const out = new Float32Array(3 * cloud.count),
    n: number[] = [0, 0, 0];
  for (let i = 0; i < cloud.count; i++) {
    decodeOctahedral(cloud.normals[2 * i]!, cloud.normals[2 * i + 1]!, n);
    out.set(n, 3 * i);
  }
  return out;
}

export function checkedMeshOptions(options: MeshOptions = {}) {
  const truncation = options.truncation ?? MESH_DEFAULTS.truncation,
    minWeight = options.minWeight ?? MESH_DEFAULTS.minWeight,
    spacingFactor = options.spacingFactor ?? MESH_DEFAULTS.spacingFactor;
  if (!(truncation >= 1.5)) throw new RangeError('truncation must be at least 1.5');
  if (!(minWeight >= 0)) throw new RangeError('minWeight must be at least 0');
  if (!(spacingFactor > 0)) throw new RangeError('spacingFactor must be positive');
  if (options.voxelSize !== undefined && !(options.voxelSize > 0))
    throw new RangeError('voxelSize must be positive');
  return { truncation, minWeight, spacingFactor, voxelSize: options.voxelSize };
}

/** The lattice and the nodes near a point. Every number that a kernel reads is a float32 value. */
export interface MeshLattice {
  readonly origin: readonly [number, number, number];
  readonly voxelSize: number;
  /** The nodes per axis, a multiple of `BRICK`. */
  readonly dims: readonly [number, number, number];
  /** The bricks per axis. */
  readonly bricks: readonly [number, number, number];
  readonly radius: number;
  readonly minWeight: number;
  /** The brick index (bi + bx (bj + by bk)) of each stored brick, ascending. */
  readonly brickIds: Uint32Array;
  /** The slot of each brick of the grid, `NO_BRICK` where it is not stored. */
  readonly brickSlot: Uint32Array;
  /** The address of each node within R of a point, ascending. */
  readonly active: Uint32Array;
  /** The points sorted into cells of side R. */
  readonly grid: PointGrid;
}

/** The address of node (i, j, k), or -1 when it is outside the lattice or in a brick not stored. */
export function nodeAddress(lattice: MeshLattice, i: number, j: number, k: number): number {
  const [nx, ny, nz] = lattice.dims,
    [bx, by] = lattice.bricks;
  if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return -1;
  const slot = lattice.brickSlot[(i >> 3) + bx * ((j >> 3) + by * (k >> 3))]!;
  if (slot === NO_BRICK) return -1;
  return slot * BRICK_NODES + (i & 7) + 8 * (j & 7) + 64 * (k & 7);
}

/** The coordinates of the node at an address. */
export function nodeCoordinates(
  lattice: MeshLattice,
  address: number,
  out: number[] = [0, 0, 0],
): number[] {
  const [bx, by] = lattice.bricks;
  const slot = Math.floor(address / BRICK_NODES),
    local = address % BRICK_NODES,
    brick = lattice.brickIds[slot]!;
  out[0] = (brick % bx) * BRICK + (local & 7);
  out[1] = (Math.floor(brick / bx) % by) * BRICK + ((local >> 3) & 7);
  out[2] = Math.floor(brick / (bx * by)) * BRICK + (local >> 6);
  return out;
}

const f32 = Math.fround;

/** Calls `visit(i, j, k, d2)` for each node within R of the point, in the order of i, j, k. */
function eachNodeNear(
  origin: readonly number[],
  h: number,
  r2: number,
  span: number,
  dims: readonly number[],
  p: readonly [number, number, number],
  visit: (i: number, j: number, k: number, dx: number, dy: number, dz: number, d2: number) => void,
): void {
  const cx = Math.round((p[0] - origin[0]!) / h),
    cy = Math.round((p[1] - origin[1]!) / h),
    cz = Math.round((p[2] - origin[2]!) / h);
  for (let k = Math.max(0, cz - span); k <= Math.min(dims[2]! - 1, cz + span); k++) {
    const dz = f32(origin[2]! + k * h) - p[2];
    if (dz * dz >= r2) continue;
    for (let j = Math.max(0, cy - span); j <= Math.min(dims[1]! - 1, cy + span); j++) {
      const dy = f32(origin[1]! + j * h) - p[1];
      if (dz * dz + dy * dy >= r2) continue;
      for (let i = Math.max(0, cx - span); i <= Math.min(dims[0]! - 1, cx + span); i++) {
        const dx = f32(origin[0]! + i * h) - p[0];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) visit(i, j, k, dx, dy, dz, d2);
      }
    }
  }
}

/** The lattice of a cloud: its spacing, its stored bricks, its nodes near a point and the cell grid. */
export function meshLattice(positions: Float32Array, options: MeshOptions = {}): MeshLattice {
  const { truncation, minWeight, spacingFactor, voxelSize } = checkedMeshOptions(options);
  const n = positions.length / 3;
  const { min, max } = boundingBox(positions);
  let h = voxelSize ?? spacingFactor * medianPointSpacing(positions);
  if (!(h > 0)) h = Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!) / 64 || 1;
  const brickCounts = (cell: number) =>
    [0, 1, 2].map((a) =>
      Math.ceil((Math.floor((max[a]! - min[a]! + 2 * (truncation + 1) * cell) / cell) + 2) / BRICK),
    );
  for (;;) {
    let counts = brickCounts(h);
    while (counts[0]! * counts[1]! * counts[2]! > MESH_MAX_BRICK_CELLS) {
      h *= Math.cbrt((counts[0]! * counts[1]! * counts[2]!) / MESH_MAX_BRICK_CELLS) * 1.001;
      counts = brickCounts(h);
    }
    const spacing = f32(h),
      radius = f32(truncation * spacing);
    const origin = [0, 1, 2].map((a) => f32(min[a]! - radius - spacing)) as [
      number,
      number,
      number,
    ];
    const [bx, by, bz] = counts as [number, number, number];
    const dims = [bx * BRICK, by * BRICK, bz * BRICK] as [number, number, number];
    const r2 = radius * radius,
      span = Math.ceil(radius / spacing) + 1;
    const brickMask = new Uint8Array(bx * by * bz);
    for (let q = 0; q < n; q++)
      eachNodeNear(
        origin,
        spacing,
        r2,
        span,
        dims,
        [positions[3 * q]!, positions[3 * q + 1]!, positions[3 * q + 2]!],
        (i, j, k) => {
          brickMask[(i >> 3) + bx * ((j >> 3) + by * (k >> 3))] = 1;
        },
      );
    let stored = 0;
    for (let b = 0; b < brickMask.length; b++) stored += brickMask[b]!;
    if (stored * BRICK_NODES > MESH_MAX_NODES) {
      h *= Math.cbrt((stored * BRICK_NODES) / MESH_MAX_NODES) * 1.05;
      continue;
    }
    const brickIds = new Uint32Array(stored),
      brickSlot = new Uint32Array(brickMask.length).fill(NO_BRICK);
    for (let b = 0, slot = 0; b < brickMask.length; b++)
      if (brickMask[b]) {
        brickIds[slot] = b;
        brickSlot[b] = slot++;
      }
    const nodeMask = new Uint8Array(stored * BRICK_NODES);
    for (let q = 0; q < n; q++)
      eachNodeNear(
        origin,
        spacing,
        r2,
        span,
        dims,
        [positions[3 * q]!, positions[3 * q + 1]!, positions[3 * q + 2]!],
        (i, j, k) => {
          const slot = brickSlot[(i >> 3) + bx * ((j >> 3) + by * (k >> 3))]!;
          nodeMask[slot * BRICK_NODES + (i & 7) + 8 * (j & 7) + 64 * (k & 7)] = 1;
        },
      );
    let count = 0;
    for (let a = 0; a < nodeMask.length; a++) count += nodeMask[a]!;
    const active = new Uint32Array(count);
    for (let a = 0, o = 0; a < nodeMask.length; a++) if (nodeMask[a]) active[o++] = a;
    return {
      origin,
      voxelSize: spacing,
      dims,
      bricks: [bx, by, bz],
      radius,
      minWeight,
      brickIds,
      brickSlot,
      active,
      grid: buildPointGrid(positions, radius),
    };
  }
}

/** The color bytes of a cloud packed as 0xBBGGRR per point (a gray when the cloud has none). */
export function cloudPackedColors(cloud: CleanCloud): Uint32Array {
  const out = new Uint32Array(cloud.count);
  for (let i = 0; i < cloud.count; i++) {
    const c = cloud.colors;
    out[i] = c ? c[3 * i]! | (c[3 * i + 1]! << 8) | (c[3 * i + 2]! << 16) : 0xc8c8c8;
  }
  return out;
}

/** The field and the colors at the nodes of the stored bricks, by address. */
export interface MeshField {
  /** `MESH_NO_VALUE` where a node has no value. */
  readonly sdf: Float32Array;
  /** The color of a node packed as 0xBBGGRR. */
  readonly rgb: Uint32Array;
}

/** The field of section 3.13 by scatter, summed in double: the reference of the kernel's gather. */
export function cpuMeshField(
  lattice: MeshLattice,
  cloud: CleanCloud,
  normals: Float32Array,
): MeshField {
  const { origin, voxelSize: h, dims, radius, minWeight, active, brickSlot } = lattice;
  const [bx, by] = lattice.bricks;
  const colors = cloudPackedColors(cloud);
  const size = lattice.brickIds.length * BRICK_NODES;
  const sdf = new Float32Array(size).fill(MESH_NO_VALUE),
    rgb = new Uint32Array(size);
  const rank = new Int32Array(size);
  for (let a = 0; a < active.length; a++) rank[active[a]!] = a;
  const sw = new Float64Array(active.length),
    sd = new Float64Array(active.length),
    sc = new Float64Array(3 * active.length);
  const r2 = radius * radius,
    span = Math.ceil(radius / h) + 1;
  const positions = cloud.positions;
  for (let q = 0; q < cloud.count; q++) {
    const mx = normals[3 * q]!,
      my = normals[3 * q + 1]!,
      mz = normals[3 * q + 2]!,
      color = colors[q]!;
    eachNodeNear(
      origin,
      h,
      r2,
      span,
      dims,
      [positions[3 * q]!, positions[3 * q + 1]!, positions[3 * q + 2]!],
      (i, j, k, dx, dy, dz, d2) => {
        const qq = 1 - d2 / r2,
          w = qq * qq;
        const slot = brickSlot[(i >> 3) + bx * ((j >> 3) + by * (k >> 3))]!;
        const id = rank[slot * BRICK_NODES + (i & 7) + 8 * (j & 7) + 64 * (k & 7)]!;
        sw[id]! += w;
        sd[id]! += w * (mx * dx + my * dy + mz * dz);
        sc[3 * id]! += w * (color & 255);
        sc[3 * id + 1]! += w * ((color >> 8) & 255);
        sc[3 * id + 2]! += w * ((color >> 16) & 255);
      },
    );
  }
  for (let a = 0; a < active.length; a++) {
    if (!(sw[a]! >= minWeight) || !(sw[a]! > 0)) continue;
    const node = active[a]!;
    sdf[node] = sd[a]! / sw[a]!;
    const byte = (v: number) => Math.min(255, Math.round(v / sw[a]!));
    rgb[node] = byte(sc[3 * a]!) | (byte(sc[3 * a + 1]!) << 8) | (byte(sc[3 * a + 2]!) << 16);
  }
  return { sdf, rgb };
}

// ---------------------------------------------------------------------------------------------
// The cubes

/** The offset of corner c of a cube from its first node, along x, y and z. */
export function cornerOffset(c: number): [number, number, number] {
  return [(c ^ (c >> 1)) & 1, (c >> 1) & 1, c >> 2];
}

/** Per edge: the corner at its lower end and its axis. */
export const MC_EDGE_LOWER: readonly (readonly [number, number])[] = MC_EDGE_CORNERS.map(
  ([a, b]) => {
    const oa = cornerOffset(a),
      ob = cornerOffset(b);
    const axis = oa[0] !== ob[0] ? 0 : oa[1] !== ob[1] ? 1 : 2;
    return [oa[axis]! < ob[axis]! ? a : b, axis] as const;
  },
);

/**
 * The triangles of a case as three edge indices each. Bourke's table winds a triangle clockwise
 * seen from the outside (the side with values at or above 0), so the second and third edge swap
 * to wind it counter-clockwise from the positive side, the side the normals point to.
 */
export function caseTriangles(caseIndex: number): number[][] {
  const out: number[][] = [];
  for (let k = 0; MC_TRI_TABLE[caseIndex * 16 + k]! >= 0; k += 3)
    out.push([
      MC_TRI_TABLE[caseIndex * 16 + k]!,
      MC_TRI_TABLE[caseIndex * 16 + k + 2]!,
      MC_TRI_TABLE[caseIndex * 16 + k + 1]!,
    ]);
  return out;
}

/**
 * The corners of the triangles of every cube whose first node is active, as vertex keys (the
 * address of the lower node of the edge times 3 plus its axis), three per triangle, in the order of
 * the nodes. A cube with a corner outside the stored bricks or without a value makes none.
 */
export function cpuMeshCubes(
  lattice: MeshLattice,
  sdf: Float32Array,
): { corners: number[]; cubes: number } {
  const keys: number[] = [],
    at = [0, 0, 0],
    address = new Array<number>(8);
  let cubes = 0;
  for (const node of lattice.active) {
    nodeCoordinates(lattice, node, at);
    let index = 0,
      valid = true;
    for (let c = 0; c < 8; c++) {
      const [dx, dy, dz] = cornerOffset(c);
      const a = nodeAddress(lattice, at[0]! + dx, at[1]! + dy, at[2]! + dz);
      const v = a < 0 ? MESH_NO_VALUE : sdf[a]!;
      if (v >= MESH_NO_VALUE * 0.5) {
        valid = false;
        break;
      }
      address[c] = a;
      if (v < 0) index |= 1 << c;
    }
    if (!valid || index === 0 || index === 255) continue;
    cubes++;
    for (const tri of caseTriangles(index))
      for (const e of tri) keys.push(address[MC_EDGE_LOWER[e]![0]]! * 3 + MC_EDGE_LOWER[e]![1]);
  }
  return { corners: keys, cubes };
}

// ---------------------------------------------------------------------------------------------
// Weld and assemble

/** The distinct keys in order of first appearance, and the index of each key's vertex. */
export function weldKeys(corners: ArrayLike<number>): { keys: Uint32Array; indices: Uint32Array } {
  const n = corners.length;
  let table = 16;
  while (table < 2 * n) table *= 2;
  const slotKey = new Uint32Array(table),
    slotIndex = new Int32Array(table).fill(-1);
  const keys: number[] = [],
    indices = new Uint32Array(n);
  const mask = table - 1;
  for (let c = 0; c < n; c++) {
    const key = corners[c]!;
    let slot = Math.imul(key, 0x9e3779b1) >>> 0;
    slot = (slot ^ (slot >>> 15)) & mask;
    for (;;) {
      const at = slotIndex[slot]!;
      if (at < 0) {
        slotKey[slot] = key;
        slotIndex[slot] = keys.length;
        indices[c] = keys.length;
        keys.push(key);
        break;
      }
      if (slotKey[slot] === key) {
        indices[c] = at;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  return { keys: Uint32Array.from(keys), indices };
}

/** The position and the color of the vertex of each key, from the field. */
export function cpuMeshVertices(
  lattice: MeshLattice,
  field: MeshField,
  keys: Uint32Array,
): { positions: Float32Array; colors: Uint8Array } {
  const { origin, voxelSize: h } = lattice;
  const positions = new Float32Array(3 * keys.length),
    colors = new Uint8Array(3 * keys.length);
  const at = [0, 0, 0];
  for (let v = 0; v < keys.length; v++) {
    const node = Math.floor(keys[v]! / 3),
      axis = keys[v]! % 3;
    nodeCoordinates(lattice, node, at);
    const next = at.slice();
    next[axis]! += 1;
    const other = nodeAddress(lattice, next[0]!, next[1]!, next[2]!);
    const s0 = field.sdf[node]!,
      s1 = field.sdf[other]!;
    const t = s0 === s1 ? 0.5 : Math.min(1, Math.max(0, s0 / (s0 - s1)));
    at[axis]! += t;
    for (let a = 0; a < 3; a++) positions[3 * v + a] = origin[a]! + at[a]! * h;
    const c0 = field.rgb[node]!,
      c1 = field.rgb[other]!;
    for (let a = 0; a < 3; a++) {
      const b0 = (c0 >> (8 * a)) & 255,
        b1 = (c1 >> (8 * a)) & 255;
      colors[3 * v + a] = Math.round(b0 + t * (b1 - b0));
    }
  }
  return { positions, colors };
}

export function emptyMesh(): DenseMesh {
  return {
    vertexCount: 0,
    faceCount: 0,
    positions: new Float32Array(0),
    colors: new Uint8Array(0),
    indices: new Uint32Array(0),
  };
}

/** The CPU reference: the field, the cubes and the weld. */
export function meshDenseCloud(cloud: CleanCloud, options: MeshOptions = {}): DenseMeshResult {
  const now = () => performance.now();
  const ms = { lattice: 0, field: 0, cubes: 0, vertices: 0 };
  const normals = cloudNormals(cloud);
  let t = now();
  const lattice = meshLattice(cloud.positions, options);
  ms.lattice = now() - t;
  t = now();
  const field = cpuMeshField(lattice, cloud, normals);
  ms.field = now() - t;
  t = now();
  const { corners, cubes } = cpuMeshCubes(lattice, field.sdf);
  ms.cubes = now() - t;
  t = now();
  const { keys, indices } = weldKeys(corners);
  const { positions, colors } = cpuMeshVertices(lattice, field, keys);
  ms.vertices = now() - t;
  return assembleMesh('cpu', cloud.count, lattice, cubes, keys, indices, positions, colors, ms);
}

/** The mesh and its report from the welded arrays; shared by both tiers. */
export function assembleMesh(
  tier: 'cpu' | 'webgpu',
  points: number,
  lattice: MeshLattice,
  cubes: number,
  keys: Uint32Array,
  indices: Uint32Array,
  positions: Float32Array,
  colors: Uint8Array,
  ms: MeshReport['ms'],
): DenseMeshResult {
  return {
    mesh: {
      vertexCount: keys.length,
      faceCount: indices.length / 3,
      positions,
      colors,
      indices,
    },
    report: {
      tier,
      points,
      voxelSize: lattice.voxelSize,
      radius: lattice.radius,
      dims: lattice.dims,
      bricks: lattice.brickIds.length,
      nodes: lattice.active.length,
      cubes,
      vertices: keys.length,
      faces: indices.length / 3,
      ms,
    },
  };
}
