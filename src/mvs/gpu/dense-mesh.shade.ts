'use typeshade';

// The mesh of a dense point cloud (`docs/dense-mvs.md`, section 3.13). Four kernels:
// - `meshSdf`: one invocation per active node. It gathers the points within R of the node from the
//   cell grid of the points and writes the field and the color of the node.
// - `meshFlags`: one invocation per active node, the first node of a cube. It writes 1 when the
//   cube makes triangles, for the shared flag scan (`planFlagScan` in `scan.ts`).
// - `meshEmit`: after the scan, a cube that makes triangles writes a row of 16 words to its slot:
//   the triangle count, then three vertex keys per triangle (the address of the lower node of the
//   edge times 3 plus its axis).
// - `meshVertices`: one invocation per distinct key, after the host welds the keys. It writes the
//   position and the packed color of the vertex.
//
// The lattice is sparse (`MeshLattice` in `src/reconstruction/dense-mesh.ts`): bricks of 8 × 8 × 8
// nodes, a node's address is its brick's slot × 512 plus its place in the brick. The arrays:
// - `meshStatic`: u32. The triangle table of 256 × 16 entries (255 for none), then 12 edge entries
//   (the lower corner of the edge, plus 8 × the axis), then the brick index of each stored brick.
// - `meshSlot`: u32 per brick of the grid, the slot of the brick or 0xffffffff.
// - `meshActive`: u32, the address of each active node.
// - `meshSdf`, `meshNodeRgb`: per node address, the field (1e30 for none) and the packed color.

class MeshControl {
  // The nodes per axis and the bricks along x and y.
  nx: u32;
  ny: u32;
  nz: u32;
  bx: u32;
  by: u32;
  // Where the brick indices start in `meshStatic`.
  brickBase: u32;
  // The bucket mask of the point grid.
  mask: u32;
  originX: f32;
  originY: f32;
  originZ: f32;
  spacing: f32;
  radius: f32;
  minWeight: f32;
  gridX: f32;
  gridY: f32;
  gridZ: f32;
  cellSize: f32;
}
class MeshRange {
  first: u32;
  end: u32;
}
declare const meshControl: uniform<MeshControl>;
declare const meshRange: uniform<MeshRange>;
declare const meshStatic: storage<array<u32>>;
declare const meshSlot: storage<array<u32>>;
declare const meshActive: storage<array<u32>>;
// Per sorted point: x, y, z, nx, ny, nz.
declare const gridPoints: storage<array<f32>>;
// Per sorted point: the cell x, y, z and the packed color.
declare const gridCells: storage<array<u32>>;
declare const bucketStart: storage<array<u32>>;
declare const meshSdf: storage<array<f32>, 'read_write'>;
declare const meshNodeRgb: storage<array<u32>, 'read_write'>;
declare const meshFlags: storage<array<u32>, 'read_write'>;
declare const meshBlockSums: storage<array<u32>>;
declare const meshKeys: storage<array<u32>, 'read_write'>;
declare const meshVertexKeys: storage<array<u32>>;
declare const meshPosition: storage<array<f32>, 'read_write'>;
declare const meshVertexRgb: storage<array<u32>, 'read_write'>;

const NONE: u32 = 0xffffffff;
const NO_VALUE: f32 = 1e30;
const TABLE_EDGES: u32 = 4096;

function cellBucket(x: u32, y: u32, z: u32): u32 {
  let h: u32 = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  h = h ^ (h >> 16);
  h = h * 0x7feb352d;
  h = h ^ (h >> 15);
  h = h * 0x846ca68b;
  return h ^ (h >> 16);
}

/** The address of node (i, j, k), or NONE outside the lattice or in a brick that is not stored. */
function nodeAt(i: i32, j: i32, k: i32): u32 {
  let address: u32 = NONE;
  if (
    i >= 0 &&
    j >= 0 &&
    k >= 0 &&
    u32(i) < meshControl.nx &&
    u32(j) < meshControl.ny &&
    u32(k) < meshControl.nz
  ) {
    const slot: u32 =
      meshSlot[(u32(i) >> 3) + meshControl.bx * ((u32(j) >> 3) + meshControl.by * (u32(k) >> 3))];
    if (slot !== NONE) {
      address = slot * 512 + (u32(i) & 7) + 8 * (u32(j) & 7) + 64 * (u32(k) & 7);
    }
  }
  return address;
}

/** The coordinates of the node at an address. */
function nodeCoordinates(address: u32): vec3i {
  const slot: u32 = address >> 9;
  const local: u32 = address & 511;
  const brick: u32 = meshStatic[meshControl.brickBase + slot];
  const bi: u32 = brick % meshControl.bx;
  const bj: u32 = (brick / meshControl.bx) % meshControl.by;
  const bk: u32 = brick / (meshControl.bx * meshControl.by);
  return vec3i(
    i32(bi * 8 + (local & 7)),
    i32(bj * 8 + ((local >> 3) & 7)),
    i32(bk * 8 + (local >> 6)),
  );
}

@compute([64])
export function meshNodeField(@builtin('global_invocation_id') gid: vec3u): void {
  let a: u32 = meshRange.first + gid.x;
  const active: bool = a < meshRange.end;
  if (!active) a = max(meshRange.end, 1) - 1;
  const at: vec3i = nodeCoordinates(meshActive[a]);
  const x: f32 = meshControl.originX + f32(at.x) * meshControl.spacing;
  const y: f32 = meshControl.originY + f32(at.y) * meshControl.spacing;
  const z: f32 = meshControl.originZ + f32(at.z) * meshControl.spacing;
  const cx: i32 = i32(floor((x - meshControl.gridX) / meshControl.cellSize));
  const cy: i32 = i32(floor((y - meshControl.gridY) / meshControl.cellSize));
  const cz: i32 = i32(floor((z - meshControl.gridZ) / meshControl.cellSize));
  const r2: f32 = meshControl.radius * meshControl.radius;
  let sw: f32 = 0.;
  let sd: f32 = 0.;
  let sr: f32 = 0.;
  let sg: f32 = 0.;
  let sb: f32 = 0.;
  for (let dz: i32 = -1; dz <= 1; dz++) {
    for (let dy: i32 = -1; dy <= 1; dy++) {
      for (let dx: i32 = -1; dx <= 1; dx++) {
        const qx: i32 = cx + dx;
        const qy: i32 = cy + dy;
        const qz: i32 = cz + dz;
        if (qx >= 1 && qy >= 1 && qz >= 1) {
          const bucket: u32 = cellBucket(u32(qx), u32(qy), u32(qz)) & meshControl.mask;
          const end: u32 = bucketStart[bucket + 1];
          for (let j: u32 = bucketStart[bucket]; j < end; j++) {
            if (
              gridCells[j * 4] === u32(qx) &&
              gridCells[j * 4 + 1] === u32(qy) &&
              gridCells[j * 4 + 2] === u32(qz)
            ) {
              const ex: f32 = x - gridPoints[j * 6];
              const ey: f32 = y - gridPoints[j * 6 + 1];
              const ez: f32 = z - gridPoints[j * 6 + 2];
              const d2: f32 = ex * ex + ey * ey + ez * ez;
              if (d2 < r2) {
                const q: f32 = 1. - d2 / r2;
                const w: f32 = q * q;
                const color: u32 = gridCells[j * 4 + 3];
                sw += w;
                sd +=
                  w *
                  (gridPoints[j * 6 + 3] * ex +
                    gridPoints[j * 6 + 4] * ey +
                    gridPoints[j * 6 + 5] * ez);
                sr += w * f32(color & 255);
                sg += w * f32((color >> 8) & 255);
                sb += w * f32((color >> 16) & 255);
              }
            }
          }
        }
      }
    }
  }
  if (active && sw >= meshControl.minWeight && sw > 0.) {
    const address: u32 = meshActive[a];
    meshSdf[address] = sd / sw;
    const r: u32 = min(u32(floor(sr / sw + 0.5)), 255);
    const g: u32 = min(u32(floor(sg / sw + 0.5)), 255);
    const b: u32 = min(u32(floor(sb / sw + 0.5)), 255);
    meshNodeRgb[address] = r | (g << 8) | (b << 16);
  }
}

/** The case index of the cube at node `at`, or 0 when a corner has no value; fills the corner addresses. */
function cubeCase(at: vec3i): u32 {
  let index: u32 = 0;
  let valid: bool = true;
  for (let c: u32 = 0; c < 8; c++) {
    const dx: i32 = i32((c ^ (c >> 1)) & 1);
    const dy: i32 = i32((c >> 1) & 1);
    const dz: i32 = i32(c >> 2);
    const address: u32 = nodeAt(at.x + dx, at.y + dy, at.z + dz);
    let v: f32 = NO_VALUE;
    if (address !== NONE) v = meshSdf[address];
    if (v >= 5e29) valid = false;
    if (v < 0.) index |= u32(1) << c;
  }
  if (!valid || index === 255) index = 0;
  return index;
}

@compute([64])
export function meshClassify(@builtin('global_invocation_id') gid: vec3u): void {
  let a: u32 = meshRange.first + gid.x;
  const active: bool = a < meshRange.end;
  if (!active) a = max(meshRange.end, 1) - 1;
  const index: u32 = cubeCase(nodeCoordinates(meshActive[a]));
  let flag: u32 = 0;
  if (index !== 0) flag = 1;
  if (active) meshFlags[a] = flag;
}

@compute([64])
export function meshEmit(@builtin('global_invocation_id') gid: vec3u): void {
  let a: u32 = meshRange.first + gid.x;
  const active: bool = a < meshRange.end;
  if (!active) a = max(meshRange.end, 1) - 1;
  const at: vec3i = nodeCoordinates(meshActive[a]);
  const index: u32 = cubeCase(at);
  let row: u32 = 0;
  if (index !== 0) row = meshBlockSums[(a / 64) * 2] + meshFlags[a] - 1;
  let triangles: u32 = 0;
  for (let t: u32 = 0; t < 5; t++) {
    if (meshStatic[index * 16 + t * 3] !== 255) triangles = t + 1;
  }
  if (active && index !== 0) {
    meshKeys[row * 16] = triangles;
    for (let t: u32 = 0; t < 5; t++) {
      if (t < triangles) {
        // The second and third edge swap: the faces wind counter-clockwise from the positive side.
        for (let m: u32 = 0; m < 3; m++) {
          let pick: u32 = m;
          if (m === 1) pick = 2;
          if (m === 2) pick = 1;
          const edge: u32 = meshStatic[index * 16 + t * 3 + pick];
          const info: u32 = meshStatic[TABLE_EDGES + edge];
          const corner: u32 = info & 7;
          const dx: i32 = i32((corner ^ (corner >> 1)) & 1);
          const dy: i32 = i32((corner >> 1) & 1);
          const dz: i32 = i32(corner >> 2);
          meshKeys[row * 16 + 1 + t * 3 + m] =
            nodeAt(at.x + dx, at.y + dy, at.z + dz) * 3 + (info >> 3);
        }
      }
    }
  }
}

@compute([64])
export function meshVertices(@builtin('global_invocation_id') gid: vec3u): void {
  let v: u32 = meshRange.first + gid.x;
  const active: bool = v < meshRange.end;
  if (!active) v = max(meshRange.end, 1) - 1;
  const key: u32 = meshVertexKeys[v];
  const address: u32 = key / 3;
  const axis: u32 = key % 3;
  const at: vec3i = nodeCoordinates(address);
  let next: vec3i = at;
  if (axis === 0) next.x = at.x + 1;
  if (axis === 1) next.y = at.y + 1;
  if (axis === 2) next.z = at.z + 1;
  const other: u32 = nodeAt(next.x, next.y, next.z);
  const s0: f32 = meshSdf[address];
  const s1: f32 = meshSdf[other];
  let t: f32 = 0.5;
  if (s0 !== s1) t = clamp(s0 / (s0 - s1), 0., 1.);
  let px: f32 = f32(at.x);
  let py: f32 = f32(at.y);
  let pz: f32 = f32(at.z);
  if (axis === 0) px += t;
  if (axis === 1) py += t;
  if (axis === 2) pz += t;
  const c0: u32 = meshNodeRgb[address];
  const c1: u32 = meshNodeRgb[other];
  let packed: u32 = 0;
  for (let k: u32 = 0; k < 3; k++) {
    const b0: f32 = f32((c0 >> (8 * k)) & 255);
    const b1: f32 = f32((c1 >> (8 * k)) & 255);
    packed |= u32(floor(b0 + t * (b1 - b0) + 0.5)) << (8 * k);
  }
  if (active) {
    meshPosition[v * 3] = meshControl.originX + px * meshControl.spacing;
    meshPosition[v * 3 + 1] = meshControl.originY + py * meshControl.spacing;
    meshPosition[v * 3 + 2] = meshControl.originZ + pz * meshControl.spacing;
    meshVertexRgb[v] = packed;
  }
}
