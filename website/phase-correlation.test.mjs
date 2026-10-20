import test from 'node:test';
import assert from 'node:assert/strict';
import { fft2d, grayThumbnail, phaseCorrelate, spectrumOf } from './src/lib/phase-correlation.ts';

// A deterministic generator, so a failure repeats.
function rng(seed) {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
}

/** Band-limited random texture of `size × size` (smooth noise, 1/f spectrum), periodic. */
function texture(size, seed) {
  const random = rng(seed);
  const re = new Float64Array(size * size),
    im = new Float64Array(size * size);
  for (let i = 0; i < size * size; i++) re[i] = random() - 0.5;
  fft2d(re, im, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const fx = Math.min(x, size - x),
        fy = Math.min(y, size - y);
      const gain = 1 / (1 + Math.hypot(fx, fy)) ** 1.5;
      re[y * size + x] *= gain;
      im[y * size + x] *= gain;
    }
  fft2d(re, im, size, true);
  const out = new Float64Array(size * size);
  let lo = Infinity,
    hi = -Infinity;
  for (const v of re) ((lo = Math.min(lo, v)), (hi = Math.max(hi, v)));
  for (let i = 0; i < out.length; i++) out[i] = (255 * (re[i] - lo)) / (hi - lo);
  return out;
}

/** `image(x - dx, y - dy)` by an exact Fourier shift of a periodic image. */
function fourierShift(image, size, dx, dy) {
  const re = Float64Array.from(image),
    im = new Float64Array(size * size);
  fft2d(re, im, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const fx = x <= size / 2 ? x : x - size,
        fy = y <= size / 2 ? y : y - size;
      const angle = (-2 * Math.PI * (fx * dx + fy * dy)) / size;
      const c = Math.cos(angle),
        s = Math.sin(angle),
        i = y * size + x;
      [re[i], im[i]] = [re[i] * c - im[i] * s, re[i] * s + im[i] * c];
    }
  fft2d(re, im, size, true);
  return re;
}

const crop = (image, size, x0, y0, n) => {
  const out = new Float64Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) out[y * n + x] = image[(y0 + y) * size + x0 + x];
  return out;
};

test('an integer shift of a textured image is recovered', () => {
  const size = 256,
    big = texture(size, 7);
  for (const [dx, dy] of [
    [0, 0],
    [5, -3],
    [-17, 9],
    [31, 22],
  ]) {
    const shifted = fourierShift(big, size, dx, dy);
    const a = spectrumOf(crop(big, size, 64, 64, 128)),
      b = spectrumOf(crop(shifted, size, 64, 64, 128));
    const r = phaseCorrelate(a, b);
    assert.ok(
      Math.abs(r.dx - dx) < 0.1 && Math.abs(r.dy - dy) < 0.1,
      `(${dx},${dy}) -> (${r.dx},${r.dy})`,
    );
  }
});

test('a sub-pixel shift is recovered within 0.1 px', () => {
  const size = 256,
    big = texture(size, 11);
  let worst = 0;
  for (const [dx, dy] of [
    [0.5, 0],
    [2.25, -1.75],
    [-6.4, 3.1],
    [10.8, 0.3],
    [-0.2, 0.7],
  ]) {
    const shifted = fourierShift(big, size, dx, dy);
    const r = phaseCorrelate(
      spectrumOf(crop(big, size, 64, 64, 128)),
      spectrumOf(crop(shifted, size, 64, 64, 128)),
    );
    worst = Math.max(worst, Math.abs(r.dx - dx), Math.abs(r.dy - dy));
  }
  assert.ok(worst < 0.1, 'worst error ' + worst);
});

test('a pair with no motion has a zero shift and a pair of unrelated images has a low ratio', () => {
  const a = texture(128, 3);
  const same = phaseCorrelate(spectrumOf(a), spectrumOf(a));
  assert.ok(Math.abs(same.dx) < 1e-6 && Math.abs(same.dy) < 1e-6);
  assert.ok(same.peak > 0.99);
  const other = phaseCorrelate(spectrumOf(a), spectrumOf(texture(128, 99)));
  assert.ok(other.ratio < same.ratio / 4, `unrelated ${other.ratio} vs same ${same.ratio}`);
});

test('the thumbnail averages the luma of boxes', () => {
  const rgba = new Uint8ClampedArray(8 * 4 * 4).fill(255);
  const gray = grayThumbnail(rgba, 8, 4, 2);
  assert.ok(gray.every((v) => Math.abs(v - 255) < 1e-9));
});
