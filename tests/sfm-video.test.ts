import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraInSourcePixels, parsePreparedVideo } from '../website/src/lib/sfm-input.ts';

const pixel =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7XcAAAAASUVORK5CYII=';
const bundle = () => ({
  format: 'typecv-sfm-video-v1',
  source: { name: 'future-32k.mkv', width: 32768, height: 32768, duration: 2, fps: 1 },
  frames: [0, 1].map((time) => ({ id: time, width: 1, height: 1, time, image: pixel })),
});

test('prepared video distinguishes source extent from bounded analysis images without a source cap', () => {
  const data = parsePreparedVideo(JSON.stringify(bundle()));
  assert.equal(data.source.width, 32768);
  assert.equal(
    data.frames.reduce((n, f) => n + f.width * f.height, 0),
    2,
  );
  for (const width of [4096, 8192, 16384, 32768, 65536]) {
    const input = bundle();
    input.source.width = input.source.height = width;
    assert.equal(parsePreparedVideo(JSON.stringify(input)).source.width, width);
  }
});

test('prepared video rejects inconsistent coordinates and unbounded frame memory before image decode', () => {
  const invalid = bundle();
  invalid.frames[1]!.time = 0;
  assert.throws(() => parsePreparedVideo(JSON.stringify(invalid)), /Invalid/);
  invalid.frames[1]!.time = 1;
  invalid.frames[1]!.id = 0;
  assert.throws(() => parsePreparedVideo(JSON.stringify(invalid)), /Invalid/);
  invalid.frames[1]!.id = 1;
  invalid.frames[0]!.width = 5000;
  assert.throws(() => parsePreparedVideo(JSON.stringify(invalid)), /unbounded/);
  invalid.frames[0]!.width = 512;
  assert.throws(() => parsePreparedVideo(JSON.stringify(invalid)), /aspect ratio/);
  invalid.frames = Array.from({ length: 5 }, (_, id) => ({
    id,
    time: id / 5,
    width: 4096,
    height: 4096,
    image: pixel,
  }));
  assert.throws(() => parsePreparedVideo(JSON.stringify(invalid)), /megapixel/);
});

test('frame PNG headers are bounded before a browser decoder sees the image', () => {
  const input = bundle();
  const png = Buffer.from(pixel.split(',')[1]!, 'base64');
  png.writeUInt32BE(32768, 16);
  png.writeUInt32BE(32768, 20);
  input.frames[0]!.image = 'data:image/png;base64,' + png.toString('base64');
  assert.throws(() => parsePreparedVideo(JSON.stringify(input)), /Encoded PNG dimensions/);
  png[0] = 0;
  input.frames[0]!.image = 'data:image/png;base64,' + png.toString('base64');
  assert.throws(() => parsePreparedVideo(JSON.stringify(input)), /Encoded PNG dimensions/);
});

test('32K source camera preserves normalized rays after pixel-center resize and rounding', () => {
  const source = { width: 32768, height: 17001 };
  const analysis = { width: 512, height: 266 };
  const original = { fx: 39000, fy: 39000, cx: 16000.25, cy: 8300.75, k1: -0.03 };
  const sx = analysis.width / source.width,
    sy = analysis.height / source.height;
  const scaled = {
    ...original,
    fx: original.fx * sx,
    fy: original.fy * sy,
    cx: (original.cx + 0.5) * sx - 0.5,
    cy: (original.cy + 0.5) * sy - 0.5,
  };
  const restored = cameraInSourcePixels(scaled, analysis, source);
  for (const key of ['fx', 'fy', 'cx', 'cy', 'k1'] as const)
    assert.ok(Math.abs(restored[key] - original[key]) < 1e-10, key);
  for (const [x, y] of [
    [0, 0],
    [32767, 17000],
    [5000, 8000],
  ]) {
    assert.ok(
      Math.abs(((x! + 0.5) * sx - 0.5 - scaled.cx) / scaled.fx - (x! - original.cx) / original.fx) <
        1e-15,
    );
    assert.ok(
      Math.abs(((y! + 0.5) * sy - 0.5 - scaled.cy) / scaled.fy - (y! - original.cy) / original.fy) <
        1e-15,
    );
  }
});
