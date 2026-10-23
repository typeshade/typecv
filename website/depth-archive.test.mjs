import test from 'node:test';
import assert from 'node:assert/strict';
import { gunzipSync } from 'node:zlib';
import { DepthArchive } from './src/lib/depth-archive.ts';
import { planFiles } from './src/lib/send-record.ts';
import { runRecordProblem } from './src/lib/run-record.ts';
import { sampleRecord } from './record-fixture.mjs';

// Independent reader: checks tar header checksums, paths, sizes and payload bytes.
function untar(gzip) {
  const tar = gunzipSync(gzip),
    files = new Map();
  for (let at = 0; tar[at];) {
    const h = tar.subarray(at, at + 512);
    const string = (a, b) => h.subarray(a, b).toString().replace(/\0.*$/s, '');
    const checksum = Number.parseInt(string(148, 154), 8);
    const copy = Buffer.from(h);
    copy.fill(32, 148, 156);
    assert.equal(
      copy.reduce((sum, byte) => sum + byte, 0),
      checksum,
    );
    assert.equal(string(257, 263), 'ustar');
    const size = Number.parseInt(string(124, 136), 8);
    files.set(string(0, 100), tar.subarray(at + 512, at + 512 + size));
    at += 512 + Math.ceil(size / 512) * 512;
  }
  return files;
}
const event = (stage = 'photometric') => ({
  viewId: 7,
  stage,
  width: 2,
  height: 2,
  depth: new Float32Array([0, 1.25, 2.5, 0]),
  depthScaleToModel: 3,
  camera: { fx: 2, fy: 2, cx: 0.5, cy: 0.5 },
  pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] },
  sourceViewIds: [3, 9],
});

test('lossless split tar.gz diagnostics retain borrowed arrays, input pixels, confidence and units', async () => {
  const artifacts = [];
  const archive = new DepthArchive(12, (f) => artifacts.push(f), undefined, 1024);
  const images = { rgba: new Uint8ClampedArray([11, 22, 33, 255]), width: 1, height: 1 };
  const expected = Buffer.from(event().depth.buffer);
  for (const stage of ['photometric', 'geometric-unfiltered', 'geometric-filtered', 'planar']) {
    const e = event(stage);
    if (stage === 'geometric-filtered' || stage === 'planar')
      e.confidence = new Float32Array([0, 0.8, 0.5, 0]);
    await archive.capture(e, images);
    e.depth.fill(999); // Callback return permits immediate reuse by MVS.
  }
  await archive.finish('completed', [7], true);
  const manifest = JSON.parse(artifacts.at(-1).data);
  assert.equal(manifest.complete, true);
  assert.equal(manifest.error, null);
  assert.deepEqual(manifest.missing, []);
  assert.equal(manifest.maps.length, 4);
  assert.ok(manifest.archives.length > 1);
  const files = new Map();
  for (const f of artifacts.slice(0, -1))
    for (const [name, bytes] of untar(f.data)) files.set(name, bytes);
  for (const m of manifest.maps) {
    assert.deepEqual(Buffer.concat(m.depth.map((p) => files.get(p.path))), expected);
    assert.equal(m.validPixels, 2);
    assert.equal(m.depthScaleToModel, 3);
    assert.deepEqual(m.sourceViewIds, [3, 9]);
    if (m.confidence)
      assert.deepEqual(
        Buffer.concat(m.confidence.map((p) => files.get(p.path))),
        Buffer.from(new Float32Array([0, 0.8, 0.5, 0]).buffer),
      );
  }
  assert.equal([...files.keys()].filter((p) => /input\.json$/.test(p)).length, 1);
  assert.deepEqual(files.get('view-7/input.rgba8.0000'), Buffer.from(images.rgba));
  const plan = await planFiles(
    artifacts.map((f) => ({
      ...f,
      data: typeof f.data === 'string' ? f.data : new Blob([f.data]),
    })),
  );
  assert.equal(runRecordProblem(sampleRecord({ files: plan.entries })), undefined);
  assert.equal(plan.uploads.length, artifacts.length);
  assert.ok(plan.uploads[0].body instanceof Blob);
});

test('cancellation and compression failure produce explicit partial manifests', async () => {
  for (const fail of [false, true]) {
    const artifacts = [];
    const archive = new DepthArchive(
      0,
      (f) => artifacts.push(f),
      fail
        ? async () => {
            throw Error('compress failed');
          }
        : undefined,
    );
    await archive.capture(event());
    await archive.finish(fail ? 'completed' : 'cancelled', [7, 9], true);
    const m = JSON.parse(artifacts.at(-1).data);
    assert.equal(m.complete, false);
    assert.equal(m.outcome, fail ? 'completed' : 'cancelled');
    assert.equal(m.missing.length, 7);
    assert.equal(Boolean(m.error), fail);
  }
});
