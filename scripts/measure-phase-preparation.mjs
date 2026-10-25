// CPU frame-preparation timing; not SfM, worker transport or UI throughput.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const args = process.argv.slice(2),
  flag = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d),
  baseline = await import(pathToFileURL(resolve(flag('--baseline')))),
  optimized = await import('../website/src/lib/phase-correlation.ts'),
  folder = resolve(flag('--frames')),
  width = Number(flag('--width', '640')),
  height = Number(flag('--height', '360')),
  count = Number(flag('--count', '3')),
  repetitions = 9,
  passes = 20,
  hash = (x) => createHash('sha256').update(x).digest('hex');
const frames = await Promise.all(
  Array.from(
    { length: count },
    async (_, i) => new Uint8Array(await readFile(resolve(folder, `${i}.private.rgba`))),
  ),
);
for (const f of frames) assert.equal(f.length, width * height * 4);
for (const f of frames)
  assert.deepEqual(
    baseline.grayThumbnail(f, width, height),
    optimized.grayThumbnail(f, width, height),
  );
const rows = [];
for (const mode of ['thumbnail', 'thumbnail-spectrum']) {
  for (const api of [baseline, optimized])
    for (let w = 0; w < 4; w++)
      for (const f of frames) {
        const gray = api.grayThumbnail(f, width, height);
        if (mode === 'thumbnail-spectrum') api.spectrumOf(gray);
      }
  for (let round = 0; round < repetitions; round++)
    for (const version of round % 2 ? ['optimized', 'baseline'] : ['baseline', 'optimized']) {
      const api = version === 'baseline' ? baseline : optimized,
        started = performance.now();
      let guard = 0;
      for (let p = 0; p < passes; p++)
        for (const f of frames) {
          const gray = api.grayThumbnail(f, width, height);
          guard += mode === 'thumbnail-spectrum' ? api.spectrumOf(gray).re[0] : gray[0];
        }
      rows.push({
        mode,
        version,
        round,
        elapsedMs: performance.now() - started,
        frames: passes * frames.length,
        guard,
      });
    }
}
for (const mode of ['thumbnail', 'thumbnail-spectrum'])
  for (let round = 0; round < repetitions; round++) {
    const pair = rows.filter((r) => r.mode === mode && r.round === round);
    assert.equal(pair[0].guard, pair[1].guard);
  }
await writeFile(
  resolve(flag('--out')),
  JSON.stringify(
    {
      createdAt: new Date().toISOString(),
      baseline: '88295b4',
      inputHashes: frames.map(hash),
      width,
      height,
      rows,
      baselineHash: hash(await readFile(resolve(flag('--baseline')))),
      optimizedHash: hash(
        await readFile(new URL('../website/src/lib/phase-correlation.ts', import.meta.url)),
      ),
      scope:
        'Original-video-derived RGBA subset. Warm paired CPU preparation, same luma/FFT results. Includes no browser decode/JPEG/transfers/UI. Single most recent shape is cached; cold-path correctness tested separately.',
    },
    null,
    2,
  ) + '\n',
);
