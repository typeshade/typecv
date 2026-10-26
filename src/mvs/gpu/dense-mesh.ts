/**
 * The mesh of a dense point cloud on the GPU (`docs/dense-mvs.md`, section 3.13). The host builds
 * the lattice and the cell grid of the points (`src/mvs/dense-mesh.ts`, the CPU
 * reference). Four TypeShade kernels (`dense-mesh.shade.ts`) compute the field, classify the
 * cubes, emit the triangle keys of the cubes that make triangles after the shared flag scan, and
 * compute the position and color of each welded vertex. The host welds the keys in between.
 */
import type { ShadeDevice } from '../../device/gpu/device.ts';
import { chunkedPass } from './cloud-clean.ts';
import { planFlagScan, scanBlockCount } from '../../device/gpu/scan.ts';
import { meshClassify, meshEmit, meshNodeField, meshVertices } from './dense-mesh.shade.ts';
import { buildPointGrid, type CleanCloud } from '../cloud-clean.ts';
import {
  MESH_NO_VALUE,
  assembleMesh,
  cloudNormals,
  cloudPackedColors,
  emptyMesh,
  meshDenseCloud,
  meshLattice,
  weldKeys,
  MC_EDGE_LOWER,
  type DenseMeshResult,
  type MeshOptions,
} from '../dense-mesh.ts';
import { MC_TRI_TABLE } from '../marching-cubes-tables.ts';

void buildPointGrid;

/** The triangle table (255 for none), the 12 edge entries and the brick indices, as one array. */
function staticTable(brickIds: Uint32Array): Uint32Array {
  const out = new Uint32Array(4096 + 12 + brickIds.length);
  for (let i = 0; i < 4096; i++) out[i] = MC_TRI_TABLE[i]! < 0 ? 255 : MC_TRI_TABLE[i]!;
  for (let e = 0; e < 12; e++) out[4096 + e] = MC_EDGE_LOWER[e]![0] | (MC_EDGE_LOWER[e]![1] << 3);
  out.set(brickIds, 4108);
  return out;
}

/** The mesh of a cloud with the field, the cube classification and the vertices on `device`. */
export async function meshDenseCloudGpu(
  device: ShadeDevice,
  cloud: CleanCloud,
  options: MeshOptions = {},
): Promise<DenseMeshResult> {
  const now = () => performance.now();
  const ms = { lattice: 0, field: 0, cubes: 0, vertices: 0 };
  const normals = cloudNormals(cloud);
  let t = now();
  const lattice = meshLattice(cloud.positions, options);
  ms.lattice = now() - t;
  const A = lattice.active.length,
    n = cloud.count,
    nodes = lattice.brickIds.length * 512;
  if (A === 0) {
    const empty = assembleMesh(
      'webgpu',
      n,
      lattice,
      0,
      new Uint32Array(0),
      new Uint32Array(0),
      new Float32Array(0),
      new Uint8Array(0),
      ms,
    );
    return { ...empty, mesh: emptyMesh() };
  }
  // A cloud whose arrays exceed the binding limit runs on the CPU tier instead.
  if (
    Math.max(nodes, 6 * n, lattice.brickSlot.length, 4108 + lattice.brickIds.length) * 4 >
    device.bindingBytes
  )
    return meshDenseCloud(cloud, options);
  const grid = lattice.grid,
    colors = cloudPackedColors(cloud);
  const points = new Float32Array(6 * n),
    cells = new Uint32Array(4 * n);
  for (let s = 0; s < n; s++) {
    const i = grid.order[s]!;
    points.set(grid.points.subarray(3 * s, 3 * s + 3), 6 * s);
    points.set(normals.subarray(3 * i, 3 * i + 3), 6 * s + 3);
    cells.set(grid.cells.subarray(3 * s, 3 * s + 3), 4 * s);
    cells[4 * s + 3] = colors[i]!;
  }
  const arrays: any[] = [];
  const own = <T>(buffer: T): T => (arrays.push(buffer), buffer);
  let successful = false;
  try {
    const meshStatic = own(
        device.allocateU32(4108 + lattice.brickIds.length, staticTable(lattice.brickIds)),
      ),
      meshSlot = own(device.allocateU32(lattice.brickSlot.length, lattice.brickSlot)),
      meshActive = own(device.allocateU32(A, lattice.active)),
      gridPoints = own(device.allocate(points.length, points)),
      gridCells = own(device.allocateU32(cells.length, cells)),
      bucketStart = own(device.allocateU32(grid.bucketStart.length, grid.bucketStart)),
      meshSdf = own(device.allocate(nodes, new Float32Array(nodes).fill(MESH_NO_VALUE))),
      meshNodeRgb = own(device.allocateU32(nodes, new Uint32Array(nodes))),
      meshFlags = own(device.allocateU32(A)),
      meshBlockSums = own(device.allocateU32(2 * scanBlockCount(A))),
      totals = own(device.allocateU32(2));
    const meshControl = {
      nx: lattice.dims[0],
      ny: lattice.dims[1],
      nz: lattice.dims[2],
      bx: lattice.bricks[0],
      by: lattice.bricks[1],
      brickBase: 4108,
      mask: grid.mask,
      originX: lattice.origin[0],
      originY: lattice.origin[1],
      originZ: lattice.origin[2],
      spacing: lattice.voxelSize,
      radius: lattice.radius,
      minWeight: lattice.minWeight,
      gridX: grid.origin[0],
      gridY: grid.origin[1],
      gridZ: grid.origin[2],
      cellSize: grid.cellSize,
    };
    t = now();
    await device.planSteps(
      chunkedPass(device, 'meshField', 'meshNodeField', A, (meshRange, groups) =>
        meshNodeField(
          {
            meshControl,
            meshRange,
            meshStatic,

            meshActive,
            gridPoints,
            gridCells,
            bucketStart,
            meshSdf,
            meshNodeRgb,
          },
          groups,
        ),
      ),
    );
    ms.field = now() - t;
    t = now();
    function* classify() {
      yield* chunkedPass(device, 'meshCubes', 'meshClassify', A, (meshRange, groups) =>
        meshClassify(
          { meshControl, meshRange, meshStatic, meshSlot, meshActive, meshSdf, meshFlags },
          groups,
        ),
      );
      yield [
        'meshScan',
        () => planFlagScan(device, { flags: meshFlags, blockSums: meshBlockSums, totals }, A),
      ] as const;
    }
    await device.planSteps(classify());
    const cubes = (await device.read(totals, 2))[0]!;
    if (cubes > A) throw new Error('The dense mesh returned an invalid cube count');
    let corners: number[] = [];
    if (cubes > 0) {
      const meshKeys = own(device.allocateU32(16 * cubes));
      await device.planSteps(
        chunkedPass(device, 'meshCubes', 'meshEmit', A, (meshRange, groups) =>
          meshEmit(
            {
              meshControl,
              meshRange,
              meshStatic,
              meshSlot,
              meshActive,
              meshSdf,
              meshFlags,
              meshBlockSums,
              meshKeys,
            },
            groups,
          ),
        ),
      );
      const rows = await device.read(meshKeys, 16 * cubes);
      ms.cubes = now() - t;
      t = now();
      for (let r = 0; r < cubes; r++) {
        const count = 3 * rows[16 * r]!;
        for (let k = 0; k < count; k++) corners.push(rows[16 * r + 1 + k]!);
      }
    } else ms.cubes = now() - t;
    const { keys, indices } = weldKeys(corners);
    corners = [];
    const V = keys.length;
    const positions = new Float32Array(3 * V),
      vertexColors = new Uint8Array(3 * V);
    if (V > 0) {
      const meshVertexKeys = own(device.allocateU32(V, keys)),
        meshPosition = own(device.allocate(3 * V)),
        meshVertexRgb = own(device.allocateU32(V));
      await device.planSteps(
        chunkedPass(device, 'meshVertices', 'meshVertices', V, (meshRange, groups) =>
          meshVertices(
            {
              meshControl,
              meshRange,
              meshStatic,
              meshSlot,
              meshSdf,
              meshNodeRgb,
              meshVertexKeys,
              meshPosition,
              meshVertexRgb,
            },
            groups,
          ),
        ),
      );
      positions.set((await device.read(meshPosition, 3 * V)).subarray(0, 3 * V));
      const rgb = await device.read(meshVertexRgb, V);
      for (let v = 0; v < V; v++)
        for (let c = 0; c < 3; c++) vertexColors[3 * v + c] = (rgb[v]! >>> (8 * c)) & 255;
    }
    ms.vertices = now() - t;
    successful = true;
    return assembleMesh(
      device.backend,
      n,
      lattice,
      cubes,
      keys,
      indices,
      positions,
      vertexColors,
      ms,
    );
  } finally {
    for (const a of arrays) successful ? device.recycle(a) : device.drop(a);
  }
}
