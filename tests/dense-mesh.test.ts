// Verifies: docs/dense-mvs.md section 3.13 (the CPU reference of the dense mesh).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BRICK_NODES,
  MESH_DEFAULTS,
  caseTriangles,
  meshDenseCloud,
  meshLattice,
  nodeAddress,
  nodeCoordinates,
  weldKeys,
  type DenseMesh,
} from '../src/reconstruction/dense-mesh.ts';
import {
  MC_CASE_TRIANGLES,
  MC_EDGE_CORNERS,
  MC_EDGE_TABLE,
} from '../src/reconstruction/marching-cubes-tables.ts';
import { planeCloud, sphereCloud } from './typeshade/dense-mesh-cases.mjs';

/** The number of faces on each undirected edge: 2 everywhere on a closed manifold mesh. */
function edgeUse(mesh: DenseMesh): Map<string, number> {
  const uses = new Map<string, number>();
  for (let f = 0; f < mesh.faceCount; f++)
    for (let k = 0; k < 3; k++) {
      const a = mesh.indices[3 * f + k]!,
        b = mesh.indices[3 * f + ((k + 1) % 3)]!;
      const key = Math.min(a, b) + ',' + Math.max(a, b);
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  return uses;
}

function signedVolume(mesh: DenseMesh): number {
  const p = mesh.positions;
  let volume = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    const [a, b, c] = [0, 1, 2].map((k) => mesh.indices[3 * f + k]! * 3) as [
      number,
      number,
      number,
    ];
    volume +=
      (p[a]! * (p[b + 1]! * p[c + 2]! - p[b + 2]! * p[c + 1]!) -
        p[a + 1]! * (p[b]! * p[c + 2]! - p[b + 2]! * p[c]!) +
        p[a + 2]! * (p[b]! * p[c + 1]! - p[b + 1]! * p[c]!)) /
      6;
  }
  return volume;
}

test('the case table has 256 cases and each case triangle uses only the edges it crosses', () => {
  assert.equal(MC_CASE_TRIANGLES.length, 256);
  assert.equal(MC_EDGE_TABLE.length, 256);
  for (let c = 0; c < 256; c++) {
    let crossed = 0;
    for (let e = 0; e < 12; e++) {
      const [a, b] = MC_EDGE_CORNERS[e]!;
      if (((c >> a) & 1) !== ((c >> b) & 1)) crossed |= 1 << e;
    }
    assert.equal(MC_EDGE_TABLE[c], crossed, `edge table of case ${c}`);
    let used = 0;
    for (const tri of caseTriangles(c)) for (const e of tri) used |= 1 << e;
    // Every crossed edge is a vertex of some triangle, and no other edge is.
    assert.equal(used, crossed, `edges of case ${c}`);
  }
  assert.equal(caseTriangles(0).length, 0);
  assert.equal(caseTriangles(255).length, 0);
});

test('a sphere of oriented points gives a closed mesh on the sphere', () => {
  const { mesh, report } = meshDenseCloud(sphereCloud(4000), { voxelSize: 0.08 });
  assert.equal(report.tier, 'cpu');
  assert.ok(mesh.faceCount > 3000);
  assert.equal(mesh.positions.length, 3 * mesh.vertexCount);
  assert.equal(mesh.colors.length, 3 * mesh.vertexCount);
  assert.equal(mesh.indices.length, 3 * mesh.faceCount);
  const half = report.voxelSize / 2;
  let worst = 0;
  for (let v = 0; v < mesh.vertexCount; v++)
    worst = Math.max(
      worst,
      Math.abs(
        Math.hypot(mesh.positions[3 * v]!, mesh.positions[3 * v + 1]!, mesh.positions[3 * v + 2]!) -
          1,
      ),
    );
  assert.ok(worst < half, `worst distance to the sphere ${worst} against half a voxel ${half}`);
  // Watertight: every edge is shared by exactly two faces.
  for (const [edge, count] of edgeUse(mesh)) assert.equal(count, 2, `edge ${edge}`);
  // Euler characteristic of a sphere: V − E + F = 2, with E = 3F / 2.
  assert.equal(mesh.vertexCount - (3 * mesh.faceCount) / 2 + mesh.faceCount, 2);
  // The faces wind counter-clockwise seen from outside: the volume is positive and about 4π/3.
  const volume = signedVolume(mesh);
  assert.ok(Math.abs(volume / ((4 * Math.PI) / 3) - 1) < 0.03, `volume ${volume}`);
  // The colors follow the points: red grows with x.
  let low = 0,
    high = 0;
  for (let v = 0; v < mesh.vertexCount; v++) {
    const x = mesh.positions[3 * v]!;
    if (x < -0.7) low = Math.max(low, mesh.colors[3 * v]!);
    if (x > 0.7) high = Math.max(high, mesh.colors[3 * v]!);
    assert.ok(
      Math.abs(mesh.colors[3 * v + 1]! - 80) <= 1 && Math.abs(mesh.colors[3 * v + 2]! - 200) <= 1,
    );
  }
  assert.ok(low < 100 && high > 200, `red ${low} and ${high}`);
});

test('the default options mesh a sphere closed', () => {
  const { mesh, report } = meshDenseCloud(sphereCloud(6000));
  assert.ok(
    report.voxelSize > 0 &&
      report.radius === Math.fround(MESH_DEFAULTS.truncation * report.voxelSize),
  );
  for (const count of edgeUse(mesh).values()) assert.equal(count, 2);
  assert.ok(signedVolume(mesh) > 0);
});

test('a plane of oriented points gives a flat mesh facing the normals', () => {
  const { mesh, report } = meshDenseCloud(planeCloud(40, 1), { voxelSize: 0.04 });
  assert.ok(mesh.faceCount > 500);
  for (let v = 0; v < mesh.vertexCount; v++) {
    assert.ok(Math.abs(mesh.positions[3 * v + 2]!) < 1e-5, `z ${mesh.positions[3 * v + 2]}`);
    const limit = report.radius + 1e-3;
    for (const a of [0, 1]) {
      const value = mesh.positions[3 * v + a]!;
      assert.ok(value > -limit && value < 1 + limit);
    }
  }
  let area = 0;
  for (let f = 0; f < mesh.faceCount; f++) {
    const p = (k: number) => mesh.indices[3 * f + k]! * 3;
    const [a, b, c] = [p(0), p(1), p(2)];
    const cross =
      (mesh.positions[b]! - mesh.positions[a]!) *
        (mesh.positions[c + 1]! - mesh.positions[a + 1]!) -
      (mesh.positions[b + 1]! - mesh.positions[a + 1]!) * (mesh.positions[c]! - mesh.positions[a]!);
    assert.ok(cross > 0, 'the face turns counter-clockwise seen from +z');
    area += cross / 2;
  }
  // The plane is the unit square, and the mesh may flare by less than the truncation radius.
  assert.ok(area > 0.9 && area < (1 + 2 * report.radius) ** 2, `area ${area}`);
  // The plane is open: its boundary edges are used once, and nothing is used more than twice.
  const uses = [...edgeUse(mesh).values()];
  assert.ok(uses.some((n) => n === 1) && uses.every((n) => n <= 2));
});

test('a node without enough weight makes no triangle', () => {
  const near = meshDenseCloud(planeCloud(40, 1), { voxelSize: 0.04, minWeight: 1 });
  const strict = meshDenseCloud(planeCloud(40, 1), { voxelSize: 0.04, minWeight: 6 });
  assert.ok(strict.mesh.faceCount < near.mesh.faceCount);
});

test('the mesh refuses a cloud without normals and an empty cloud gives an empty mesh', () => {
  const sphere = sphereCloud(200);
  assert.throws(() => meshDenseCloud({ ...sphere, normals: undefined }), /normals/);
  assert.throws(() => meshDenseCloud(sphere, { truncation: 1 }), /truncation/);
  const empty = meshDenseCloud({
    count: 0,
    positions: new Float32Array(0),
    normals: new Int16Array(0),
  });
  assert.equal(empty.mesh.faceCount, 0);
  assert.equal(empty.report.vertices, 0);
});

test('the lattice is sparse, addressed by brick, and keeps its spacing within the node limit', () => {
  const cloud = sphereCloud(2000);
  const lattice = meshLattice(cloud.positions, { voxelSize: 0.1 });
  assert.equal(lattice.dims[0] % 8, 0);
  assert.ok(
    lattice.brickIds.length * BRICK_NODES < lattice.dims[0] * lattice.dims[1] * lattice.dims[2],
  );
  const at = [0, 0, 0];
  for (const address of lattice.active) {
    nodeCoordinates(lattice, address, at);
    assert.equal(nodeAddress(lattice, at[0]!, at[1]!, at[2]!), address);
  }
  assert.equal(nodeAddress(lattice, -1, 0, 0), -1);
  // Two far points with a tiny voxel size would need a huge dense lattice: the spacing grows.
  const far = new Float32Array([0, 0, 0, 1000, 1000, 1000]);
  const grown = meshLattice(far, { voxelSize: 0.001 });
  assert.ok(grown.voxelSize > 0.001);
});

test('weldKeys numbers the keys by first appearance', () => {
  const { keys, indices } = weldKeys([9, 4, 9, 7, 4, 100]);
  assert.deepEqual(Array.from(keys), [9, 4, 7, 100]);
  assert.deepEqual(Array.from(indices), [0, 1, 0, 2, 1, 3]);
});
