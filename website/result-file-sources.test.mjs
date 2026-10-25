import test from 'node:test';
import assert from 'node:assert/strict';
import { resultFileSources } from './src/lib/result-file-sources.ts';

function scene(id) {
  return {
    views: [
      {
        id,
        pose: { R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: new Float64Array(3) },
        camera: { fx: 1, fy: 1, cx: 0, cy: 0 },
      },
    ],
    landmarks: [{ trackId: id, xyz: new Float64Array([id, 2, 3]), observations: [] }],
  };
}

test('sharing snapshots every component and serializes only the requested export', () => {
  const first = scene(10),
    second = scene(40);
  const chunks = [
    { viewId: 40, positions: new Float32Array([4, 5, 6]), colors: new Uint8Array([1, 2, 3]) },
  ];
  const calls = [];
  const components = [
    { index: 1, scene: second, points: chunks },
    { index: 0, scene: first },
  ];
  const files = resultFileSources(components, [], (value) => {
    calls.push(value.views[0].id);
    return JSON.stringify({ view: value.views[0].id });
  });
  assert.deepEqual(calls, []);
  assert.deepEqual(
    files.map((file) => file.name),
    [
      'typecv-reconstruction-c000.json',
      'typecv-reconstruction-c000.ply',
      'typecv-reconstruction-c001.json',
      'typecv-reconstruction-c001.ply',
      'typecv-dense-c001.ply',
    ],
  );
  // Selecting/resetting the displayed scene must not change an already captured send.
  first.views = [];
  second.views = [];
  chunks.length = 0;
  components.length = 0;
  assert.deepEqual(JSON.parse(files[2].data()), { view: 40 });
  assert.deepEqual(JSON.parse(files[0].data()), { view: 10 });
  assert.deepEqual(calls, [40, 10]);
  const ply = files[4].data();
  const headerEnd =
    new TextDecoder().decode(ply.subarray(0, 512)).indexOf('end_header\n') + 'end_header\n'.length;
  assert.ok(headerEnd > 10);
  const data = new DataView(ply.buffer, ply.byteOffset + headerEnd, 15);
  assert.equal(data.getFloat32(0, true), 4);
  assert.equal(data.getFloat32(8, true), 6);
  assert.equal(data.getUint8(14), 3);
});

test('a single component keeps existing filenames and carries its mesh', () => {
  const mesh = {
    run: 1,
    positions: new Float32Array([1, 2, 3]),
    colors: new Uint8Array([4, 5, 6]),
    indices: new Uint32Array([0, 0, 0]),
  };
  const files = resultFileSources([{ index: 0, scene: scene(2), mesh }], [], () => '{}');
  assert.deepEqual(
    files.map((file) => file.name),
    ['typecv-reconstruction.json', 'typecv-reconstruction.ply', 'typecv-mesh.ply'],
  );
  assert.match(new TextDecoder().decode(files[2].data().subarray(0, 256)), /element face 1/);
});

test('capturing depth archives retains references without reading unreadable Blobs', () => {
  const blob = new Blob(['archive']);
  blob.arrayBuffer = () => {
    throw new Error('must not eagerly read');
  };
  const map = new Map([['typecv-depths-c000-0000.tar.gz', blob]]);
  const files = resultFileSources([], map, () => {
    throw new Error('must not serialize');
  });
  map.clear();
  assert.equal(files.length, 1);
  assert.equal(files[0].data, blob);
});

test('component identities cannot collide or create arbitrary upload paths', () => {
  for (const index of [-1, 1.5, 1000, NaN])
    assert.throws(
      () => resultFileSources([{ index, scene: scene(1) }], [], () => '{}'),
      /component index/,
    );
  assert.throws(
    () =>
      resultFileSources(
        [
          { index: 0, scene: scene(1) },
          { index: 0, scene: scene(2) },
        ],
        [],
        () => '{}',
      ),
    /component index/,
  );
});
