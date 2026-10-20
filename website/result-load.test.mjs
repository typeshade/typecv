import test from 'node:test';
import assert from 'node:assert/strict';
import { densePly, denseMeshPly } from './src/lib/dense-cloud.ts';
import { parsePly, PLY_MAX_HEADER } from './src/lib/ply-parse.ts';
import { loadResultFiles, sceneFromJson } from './src/lib/result-load.ts';
import { reconstructionJson, reconstructionPly } from './src/lib/result-export.ts';
import { sampleRecord } from './record-fixture.mjs';

const bytes = (text) => new TextEncoder().encode(text);
const file = (name, data) => ({ name, data: typeof data === 'string' ? bytes(data) : data });

/** A scene as the worker returns it: typed arrays for the poses and the points. */
function scene() {
  const view = (id, x) => ({
    id,
    time: id * 0.5,
    sequence: id,
    width: 640,
    height: 360,
    pose: {
      R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      t: Float64Array.from([0 - x, 0.25, 0.125]),
    },
    camera: { fx: 512.5, fy: 512.5, cx: 320, cy: 180, k1: -0.01 },
  });
  return {
    views: [view(0, 0), view(1, 1.5), view(4, 3)],
    landmarks: [
      {
        trackId: 7,
        xyz: Float64Array.from([0.5, -1.25, 4]),
        color: [1, 0.5, 0],
        observations: [{ viewId: 0, featureId: 3, pixel: [100.5, 200] }],
      },
      { trackId: 9, xyz: Float64Array.from([1, 2, 3]), observations: [] },
    ],
  };
}

const exported = (extra = {}) =>
  reconstructionJson({
    source: { name: 'clip', width: 640, height: 360, duration: 12.5 },
    assumedFocalRatio: 0.8,
    diagnostics: { meanReprojectionError: 0.4 },
    unregistered: [{ viewId: 2, reason: 'few matches' }],
    analysisFrames: [{ id: 0, time: 0, width: 640, height: 360 }],
    scene: scene(),
    sourceCamera: (view) => view.camera,
    ...extra,
  });

test('the reconstruction JSON the lab exports loads back identically', () => {
  const loaded = loadResultFiles([file('typecv-reconstruction.json', exported())]);
  const expected = scene();
  assert.deepEqual(loaded.scene.views, expected.views);
  assert.equal(loaded.scene.landmarks.length, 2);
  loaded.scene.landmarks.forEach((point, i) => {
    assert.deepEqual(point.xyz, expected.landmarks[i].xyz);
    assert.ok(point.xyz instanceof Float64Array);
    assert.deepEqual(point.color, expected.landmarks[i].color);
    assert.deepEqual(point.observations, expected.landmarks[i].observations);
    assert.equal(point.trackId, expected.landmarks[i].trackId);
  });
  assert.deepEqual(loaded.source, { name: 'clip', width: 640, height: 360, duration: 12.5 });
  assert.deepEqual(loaded.unregistered, [{ viewId: 2, reason: 'few matches' }]);
  assert.deepEqual(loaded.diagnostics, { meanReprojectionError: 0.4 });
  assert.equal(loaded.runRecord, undefined);
  assert.deepEqual(loaded.notes, []);
});

test('the run record in the JSON comes back as the summary; a bad one is dropped with a note', () => {
  const record = sampleRecord();
  const loaded = loadResultFiles([file('r.json', exported({ runRecord: record }))]);
  assert.deepEqual(loaded.runRecord, record);
  const bad = loadResultFiles([file('r.json', exported({ runRecord: { format: 'x' } }))]);
  assert.equal(bad.runRecord, undefined);
  assert.match(bad.notes[0], /run record has a bad field: format/);
  // The run record the lab downloads on its own loads too.
  const alone = loadResultFiles([
    file('typecv-run-record.json', JSON.stringify(record)),
    file('typecv-reconstruction.json', exported()),
  ]);
  assert.deepEqual(alone.runRecord, record);
});

test('the JSON reader names the field that is wrong', () => {
  assert.throws(() => sceneFromJson({}), /not a reconstruction/);
  const json = JSON.parse(exported());
  json.views[1].pose.R = [1, 2, 3];
  assert.throws(() => sceneFromJson(json), /views\[1\]\.pose\.R must be 9 finite numbers/);
  const json2 = JSON.parse(exported());
  json2.landmarks[0].xyz = [0, null, 1];
  assert.throws(() => sceneFromJson(json2), /landmarks\[0\]\.xyz/);
  assert.throws(
    () => loadResultFiles([file('broken.json', '{"views":')]),
    /^RangeError: broken\.json:|broken\.json: /,
  );
});

test('the dense PLY the lab exports loads back: binary, with and without normals', () => {
  const positions = Float32Array.from([0, 1, 2, 3.5, -4, 5.25]);
  const colors = Uint8Array.from([10, 20, 30, 250, 251, 252]);
  const plain = parsePly(densePly([{ viewId: 0, positions, colors }]));
  assert.equal(plain.format, 'binary_little_endian');
  assert.deepEqual(plain.positions, positions);
  assert.deepEqual(plain.colors, colors);
  assert.equal(plain.normals, undefined);
  assert.equal(plain.indices, undefined);
  // Two chunks and octahedral normals (+z and +x).
  const withNormals = parsePly(
    densePly([
      {
        viewId: 0,
        positions: positions.subarray(0, 3),
        colors: colors.subarray(0, 3),
        normals: Int16Array.from([0, 0]),
      },
      {
        viewId: 1,
        positions: positions.subarray(3),
        colors: colors.subarray(3),
        normals: Int16Array.from([32767, 0]),
      },
    ]),
  );
  assert.deepEqual(withNormals.positions, positions);
  assert.deepEqual(withNormals.colors, colors);
  assert.equal(withNormals.normals.length, 6);
  assert.ok(Math.abs(Math.hypot(...withNormals.normals.subarray(3)) - 1) < 1e-5);
  const loaded = loadResultFiles([
    file('typecv-dense.ply', densePly([{ viewId: 0, positions, colors }])),
  ]);
  assert.equal(loaded.dense.length, 1);
  assert.deepEqual(loaded.dense[0].positions, positions);
  assert.deepEqual(loaded.dense[0].colors, colors);
  assert.equal(loaded.scene, undefined);
});

test('the mesh PLY the lab exports loads back with its faces', () => {
  const mesh = {
    positions: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    colors: Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    indices: Uint32Array.from([0, 1, 2, 2, 1, 3]),
  };
  const ply = parsePly(denseMeshPly(mesh));
  assert.deepEqual(ply.positions, mesh.positions);
  assert.deepEqual(ply.colors, mesh.colors);
  assert.deepEqual(ply.indices, mesh.indices);
  const loaded = loadResultFiles([file('typecv-mesh.ply', denseMeshPly(mesh))]);
  assert.deepEqual(loaded.mesh, mesh);
  assert.equal(loaded.dense, undefined);
});

test('several files load at once, in any order', () => {
  const dense = densePly([
    { viewId: 0, positions: Float32Array.of(1, 2, 3), colors: Uint8Array.of(9, 9, 9) },
  ]);
  const mesh = denseMeshPly({
    positions: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1, 0),
    colors: new Uint8Array(9),
    indices: Uint32Array.of(0, 1, 2),
  });
  const files = [
    file('typecv-mesh.ply', mesh),
    file('typecv-dense.ply', dense),
    file('typecv-reconstruction.json', exported({ runRecord: sampleRecord() })),
    file('typecv-reconstruction.ply', reconstructionPly(scene())),
  ];
  for (const order of [files, [...files].reverse()]) {
    const loaded = loadResultFiles(order);
    assert.equal(loaded.scene.views.length, 3);
    assert.equal(loaded.dense.length, 1);
    assert.equal(loaded.mesh.indices.length, 3);
    assert.ok(loaded.runRecord);
    assert.match(loaded.notes.join(), /sparse PLY repeats the JSON/);
  }
});

test('the sparse PLY the lab exports loads as points; the colors survive', () => {
  const text = reconstructionPly(scene());
  const ply = parsePly(bytes(text));
  assert.equal(ply.format, 'ascii');
  // Two landmarks and three camera centers.
  assert.equal(ply.positions.length / 3, 5);
  assert.deepEqual([...ply.colors.subarray(0, 6)], [255, 128, 0, 179, 179, 179]);
  assert.deepEqual([...ply.positions.subarray(0, 3)], [0.5, -1.25, 4]);
  // Camera centers: -R^T t with R = I.
  assert.deepEqual([...ply.positions.subarray(6, 9)], [0, -0.25, -0.125]);
  const loaded = loadResultFiles([file('typecv-reconstruction.ply', text)]);
  assert.equal(loaded.scene.landmarks.length, 5);
  assert.deepEqual(loaded.scene.landmarks[0].color, [1, 128 / 255, 0]);
  assert.equal(loaded.scene.views.length, 0);
});

test('the PLY reader takes ASCII polygons, big-endian files and extra properties', () => {
  const quad = parsePly(
    bytes(
      [
        'ply',
        'format ascii 1.0',
        'comment hand made',
        'element vertex 4',
        'property float x',
        'property float y',
        'property float z',
        'property float confidence',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'element face 1',
        'property list uchar int vertex_indices',
        'end_header',
        '0 0 0 0.9 255 0 0',
        '1 0 0 0.9 0 255 0',
        '1 1 0 0.9 0 0 255',
        '0 1 0 0.9 9 9 9',
        '4 0 1 2 3',
        '',
      ].join('\r\n'),
    ),
  );
  assert.deepEqual(quad.comments, ['hand made']);
  assert.deepEqual([...quad.indices], [0, 1, 2, 0, 2, 3]);
  assert.deepEqual([...quad.colors.subarray(3, 6)], [0, 255, 0]);

  const header = [
    'ply',
    'format binary_big_endian 1.0',
    'element vertex 1',
    'property double x',
    'property double y',
    'property double z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
    '',
  ].join('\n');
  const head = bytes(header);
  const data = new Uint8Array(head.length + 27);
  data.set(head);
  const view = new DataView(data.buffer);
  [1.5, -2, 0.25].forEach((v, i) => view.setFloat64(head.length + 8 * i, v, false));
  data.set([7, 8, 9], head.length + 24);
  const big = parsePly(data);
  assert.deepEqual([...big.positions], [1.5, -2, 0.25]);
  assert.deepEqual([...big.colors], [7, 8, 9]);
});

test('the PLY reader refuses what it cannot read, and says why', () => {
  assert.throws(() => parsePly(bytes('hello')), /not a PLY file/);
  assert.throws(
    () =>
      parsePly(bytes('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n')),
    /no "y" property/,
  );
  assert.throws(
    () => parsePly(bytes('ply\nformat ascii 1.0\nelement face 0\nend_header\n')),
    /no vertex element/,
  );
  // A binary body one byte short.
  const good = densePly([
    { viewId: 0, positions: Float32Array.of(1, 2, 3), colors: Uint8Array.of(1, 2, 3) },
  ]);
  assert.throws(() => parsePly(good.subarray(0, good.length - 1)), /file ends before/);
  // A face that names a vertex that is not there.
  const mesh = denseMeshPly({
    positions: new Float32Array(9),
    colors: new Uint8Array(9),
    indices: Uint32Array.of(0, 1, 5),
  });
  assert.throws(() => parsePly(mesh), /names vertex 5 of 3/);
  assert.throws(
    () =>
      parsePly(
        bytes(
          'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 nope\n',
        ),
      ),
    /bad number "nope"/,
  );
  assert.throws(() => loadResultFiles([]), /hold no reconstruction/);
  assert.ok(PLY_MAX_HEADER >= 1024);
});
