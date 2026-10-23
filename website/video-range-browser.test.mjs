import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';

test('real page, worker and fallback decoders restrict frame timestamps to the selected interval', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'typecv-video-range-'));
  try {
    const api = join(directory, 'api.js');
    const worker = join(directory, 'worker.js');
    await build({
      stdin: {
        contents: "export { decodeUploadedFrames } from './src/lib/sfm-video.ts';",
        resolveDir: import.meta.dirname,
      },
      outfile: api,
      bundle: true,
      format: 'esm',
      platform: 'browser',
    });
    await build({
      entryPoints: [new URL('./src/workers/decode.worker.ts', import.meta.url).pathname],
      outfile: worker,
      bundle: true,
      format: 'esm',
      platform: 'browser',
    });
    const { result, errors } = await inBrowser(
      async () => {
        const { decodeUploadedFrames } = await import('/decode-api.js');
        // Astro normally emits the worker asset URL. This standalone esbuild bundle keeps its
        // source URL, so point that URL at the separately bundled, correctly typed module asset.
        const BrowserWorker = window.Worker;
        window.Worker = class extends BrowserWorker {
          constructor(url, options) {
            super(String(url).endsWith('/decode.worker.ts') ? '/range-worker.js' : url, options);
          }
        };
        const file = new File([await (await fetch('/clip.mp4')).blob()], 'clip.mp4', {
          type: 'video/mp4',
        });
        const run = async (search, range, cancel = false) => {
          const controller = new AbortController();
          const info = { path: 'main-thread' };
          const frames = [];
          let error;
          try {
            for await (const frame of decodeUploadedFrames(
              file,
              controller.signal,
              640,
              4,
              info,
              search,
              range,
            )) {
              let hash = 2166136261;
              for (const byte of frame.pixels) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
              frames.push({ time: frame.time, hash, width: frame.width, height: frame.height });
              if (cancel) controller.abort();
            }
          } catch (caught) {
            error = { name: caught.name, message: caught.message };
          }
          return { path: info.path, frames, error };
        };
        const range = { start: 2, end: 4 };
        const full = await run('?decode=main');
        const main = await run('?decode=main', range);
        const worker = await run('', range);
        const fractional = await run('', { start: 2.1, end: 3.6 });
        const cancelled = await run('', range, true);
        const invalid = await run('?decode=main', { start: 5, end: 3 });
        const unavailable = await run('', { start: 20, end: 22 });
        const NativeWorker = window.Worker;
        window.Worker = class {
          constructor() {
            queueMicrotask(() => this.onerror?.({ message: 'Fixture worker unavailable' }));
          }
          postMessage() {}
          terminate() {}
        };
        let fallback;
        try {
          fallback = await run('', range);
        } finally {
          window.Worker = NativeWorker;
        }
        return { full, main, worker, fractional, cancelled, invalid, unavailable, fallback };
      },
      undefined,
      {
        serve: {
          '/decode-api.js': api,
          '/range-worker.js': worker,
          '/clip.mp4': new URL('../tests/fixtures/source-clip-vp9.mp4', import.meta.url).pathname,
        },
      },
    );
    assert.deepEqual(errors, []);
    assert.ok(result.full.frames.length > result.main.frames.length);
    assert.ok(result.main.frames.length >= 2);
    assert.equal(result.main.path, 'main-thread');
    assert.equal(result.worker.path, 'worker');
    assert.equal(result.fallback.path, 'main-thread');
    assert.equal(result.worker.error, undefined);
    assert.deepEqual(result.worker.frames, result.main.frames);
    assert.deepEqual(result.fallback.frames, result.main.frames);
    for (const frame of result.main.frames) {
      assert.ok(frame.time >= 2 && frame.time < 4, JSON.stringify(frame));
      const sameTime = result.full.frames.find((other) => other.time === frame.time);
      if (sameTime) assert.equal(frame.hash, sameTime.hash);
    }
    assert.ok(result.fractional.frames.length >= 2);
    assert.ok(result.fractional.frames.every((frame) => frame.time >= 2.1 && frame.time < 3.6));
    assert.equal(result.cancelled.frames.length, 1);
    assert.equal(result.cancelled.error.name, 'AbortError');
    assert.equal(result.invalid.error.name, 'RangeError');
    assert.equal(result.unavailable.error.name, 'RangeError');
    assert.equal(
      result.unavailable.frames.length,
      0,
      'an empty interval never falls back to whole-video analysis',
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
