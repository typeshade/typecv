// Verifies: docs/dense-mvs.md step 7 (section 6, the dense cloud's point budget and binary PLY).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DENSE_POINT_BUDGET,
  DENSE_TOUCH_POINT_BUDGET,
  decodeOctahedral,
  denseDrawnCount,
  denseMeshPly,
  densePly,
  densePointBudget,
  denseStride,
} from './src/lib/dense-cloud.ts';

/** The octahedral snorm16 pair of a unit vector, as `encodeNormal` in the kernels. */
function encode([x, y, z]) {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z);
  let px = x / l1,
    py = y / l1;
  if (z < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1),
      oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  return [Math.round(px * 32767), Math.round(py * 32767)];
}

/** A chunk of `n` points with known positions, colors and normals. */
function chunk(viewId, n, offset) {
  const positions = new Float32Array(3 * n),
    colors = new Uint8Array(3 * n),
    normals = new Int16Array(2 * n);
  for (let i = 0; i < n; i++) {
    const k = offset + i;
    positions.set([k, -k / 2, k * 0.25 + 0.125], 3 * i);
    colors.set([k % 256, (3 * k) % 256, 255 - (k % 256)], 3 * i);
    const angle = k * 0.7;
    normals.set(encode([Math.cos(angle) * 0.6, Math.sin(angle) * 0.6, k % 2 ? 0.8 : -0.8]), 2 * i);
  }
  return { viewId, positions, colors, normals };
}

/** A minimal reader of a binary little-endian PLY: the header's elements and properties. */
function readPly(bytes) {
  const text = new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, 4096)));
  const end = text.indexOf('end_header\n');
  assert.ok(end > 0, 'the header ends with end_header');
  const lines = text.slice(0, end).trim().split('\n');
  assert.equal(lines[0], 'ply');
  assert.equal(lines[1], 'format binary_little_endian 1.0');
  const vertex = lines.find((line) => line.startsWith('element '));
  assert.match(vertex, /^element vertex \d+$/);
  assert.equal(lines.filter((line) => line.startsWith('element ')).length, 1);
  const properties = lines
    .filter((line) => line.startsWith('property '))
    .map((line) => line.split(' ').slice(1));
  return {
    count: Number(vertex.split(' ')[2]),
    properties,
    body: end + 'end_header\n'.length,
  };
}

test('the point budget and the drawn subset follow section 6', () => {
  assert.equal(densePointBudget(0), DENSE_POINT_BUDGET);
  assert.equal(densePointBudget(5), DENSE_TOUCH_POINT_BUDGET);
  assert.equal(DENSE_POINT_BUDGET, 1_000_000);
  assert.equal(DENSE_TOUCH_POINT_BUDGET, 300_000);
  assert.equal(denseStride(0, 1000), 1);
  assert.equal(denseStride(1000, 1000), 1);
  assert.equal(denseStride(1001, 1000), 2);
  assert.equal(denseStride(2_400_000, DENSE_POINT_BUDGET), 3);
  // k is the smallest integer that fits: k − 1 would draw more than the budget.
  for (const count of [1, 999, 1000, 1001, 4321, 1_000_000, 1_000_001, 3_456_789]) {
    const k = denseStride(count, 1000);
    assert.ok(denseDrawnCount(count, k) <= 1000, String(count));
    if (k > 1) assert.ok(denseDrawnCount(count, k - 1) > 1000, String(count));
  }
  assert.throws(() => denseStride(10, 0), RangeError);
});

test('the octahedral decode returns the encoded unit normals', () => {
  for (const normal of [
    [0, 0, 1],
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
    [0.48, -0.6, 0.64],
    [-0.36, 0.48, -0.8],
  ]) {
    const [low, high] = encode(normal);
    const decoded = decodeOctahedral(low, high);
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(decoded[c] - normal[c]) < 1e-3, String(normal));
  }
});

test('the binary PLY has the header, the vertex count and the bytes of section 6', () => {
  const chunks = [chunk(3, 5, 0), chunk(7, 0, 5), chunk(9, 12, 5)];
  const bytes = densePly(chunks);
  const ply = readPly(bytes);
  assert.equal(ply.count, 17);
  assert.deepEqual(ply.properties, [
    ['float', 'x'],
    ['float', 'y'],
    ['float', 'z'],
    ['float', 'nx'],
    ['float', 'ny'],
    ['float', 'nz'],
    ['uchar', 'red'],
    ['uchar', 'green'],
    ['uchar', 'blue'],
  ]);
  // 27 B per point (section 6), and nothing after the last vertex.
  assert.equal(bytes.length, ply.body + 27 * 17);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let k = 0; k < 17; k++) {
    const at = ply.body + 27 * k;
    const source = k < 5 ? chunks[0] : chunks[2],
      i = k < 5 ? k : k - 5;
    for (let c = 0; c < 3; c++)
      assert.equal(view.getFloat32(at + 4 * c, true), source.positions[3 * i + c]);
    const normal = [0, 1, 2].map((c) => view.getFloat32(at + 12 + 4 * c, true));
    assert.ok(Math.abs(Math.hypot(...normal) - 1) < 1e-5);
    const expected = decodeOctahedral(source.normals[2 * i], source.normals[2 * i + 1]);
    for (let c = 0; c < 3; c++) assert.ok(Math.abs(normal[c] - expected[c]) < 1e-6);
    assert.deepEqual(
      [...bytes.subarray(at + 24, at + 27)],
      [...source.colors.subarray(3 * i, 3 * i + 3)],
    );
  }
});

test('a cloud without normals writes 15 B per point, an empty cloud a header only', () => {
  const { viewId, positions, colors } = chunk(1, 4, 0);
  const bytes = densePly([{ viewId, positions, colors }]);
  const ply = readPly(bytes);
  assert.equal(ply.count, 4);
  assert.equal(ply.properties.length, 6);
  assert.equal(bytes.length, ply.body + 15 * 4);
  const empty = densePly([]);
  assert.equal(readPly(empty).count, 0);
  assert.equal(empty.length, readPly(empty).body);
});

// Verifies: docs/dense-mvs.md 3.13 (the mesh as a binary PLY with faces).
test('the mesh PLY holds 15 B per vertex and 13 B per face, with a vertex_indices list', () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0.5]);
  const colors = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]);
  const indices = new Uint32Array([0, 1, 2, 2, 1, 3]);
  const bytes = denseMeshPly({ positions, colors, indices });
  const text = new TextDecoder().decode(bytes.subarray(0, 400));
  const end = text.indexOf('end_header\n') + 'end_header\n'.length;
  const header = text.slice(0, end).split('\n');
  assert.equal(header[0], 'ply');
  assert.equal(header[1], 'format binary_little_endian 1.0');
  assert.ok(header.includes('element vertex 4'));
  assert.ok(header.includes('element face 2'));
  assert.ok(header.includes('property list uchar uint vertex_indices'));
  assert.equal(bytes.length, end + 15 * 4 + 13 * 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let v = 0; v < 4; v++) {
    const at = end + 15 * v;
    for (let a = 0; a < 3; a++)
      assert.equal(view.getFloat32(at + 4 * a, true), positions[3 * v + a]);
    assert.deepEqual([...bytes.subarray(at + 12, at + 15)], [...colors.subarray(3 * v, 3 * v + 3)]);
  }
  for (let f = 0; f < 2; f++) {
    const at = end + 60 + 13 * f;
    assert.equal(bytes[at], 3);
    for (let a = 0; a < 3; a++)
      assert.equal(view.getUint32(at + 1 + 4 * a, true), indices[3 * f + a]);
  }
  const empty = denseMeshPly({
    positions: new Float32Array(0),
    colors: new Uint8Array(0),
    indices: new Uint32Array(0),
  });
  assert.ok(new TextDecoder().decode(empty).includes('element face 0'));
});
