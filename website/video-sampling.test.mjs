import test from 'node:test';
import assert from 'node:assert/strict';
import { sampleTimestamps, videoSampleFps, VIDEO_SAMPLE_FPS } from './src/lib/video-sampling.ts';

test('the sampling schedule is a time grid that ends on the last frame', () => {
  assert.equal(VIDEO_SAMPLE_FPS, 4);
  // 23.6 s at 4 fps: 95 grid points (0..23.5). The end is 0.099 s away, under half an interval.
  const times = sampleTimestamps(0, 23.6, 4);
  assert.equal(times.length, 95);
  assert.equal(times[0], 0);
  assert.ok(times.every((t, i) => i === 0 || t > times[i - 1]));
  assert.ok(Math.abs(times.at(-1) - 23.5) < 1e-9);
  // The grid stops more than half an interval short of 23.7 s: the last frame is added.
  const longer = sampleTimestamps(0, 23.7, 4);
  assert.equal(longer.length, 96);
  assert.ok(longer.at(-1) < 23.7 && longer.at(-1) > 23.69);
  // The grid follows a first timestamp that is not zero.
  assert.deepEqual(sampleTimestamps(0.5, 1.5, 2), [0.5, 1, 1.499]);
});

test('degenerate durations still give a first frame', () => {
  assert.deepEqual(sampleTimestamps(0, 0, 4), [0]);
  assert.deepEqual(sampleTimestamps(0, 0.1, 4), [0]);
  assert.deepEqual(sampleTimestamps(0, 1, 0), [0]);
});

test('the fps query overrides the default only with a sane number', () => {
  assert.equal(videoSampleFps(''), 4);
  assert.equal(videoSampleFps('?fps=10'), 10);
  assert.equal(videoSampleFps('?fps=0.5'), 0.5);
  assert.equal(videoSampleFps('?fps=0'), 4);
  assert.equal(videoSampleFps('?fps=abc'), 4);
  assert.equal(videoSampleFps('?fps=1000'), 4);
});
