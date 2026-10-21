/**
 * Measure how often two decodes of one video give different `frameKey`s
 * (docs/multi-video.md, 2.4 and step 2).
 *
 *   node scripts/measure-decoder-agreement.mjs video.webm [more videos] [--fps=4] [--edge=640] [--repeat=2]
 *
 * The lab's own decode (`decodeFrameStream` in `website/src/lib/sfm-video.ts`) runs in Chromium on
 * both of its paths: the page (an `HTMLCanvasElement`) and the decode worker (an `OffscreenCanvas`).
 * Each path decodes each video `repeat` times. The script hashes every analysis frame with
 * `frameKeyOf` and compares, frame by frame:
 *
 *   - repeat-page: two decodes on the page path.
 *   - repeat-worker: two decodes on the worker path.
 *   - page-worker: one decode on each path.
 *
 * `distinctKeys` is the number of different keys in the first page decode: a value near the frame
 * count shows that the key tells frames apart, so an agreement of 100 % is not a constant. It prints one JSON object. Needs Chromium (playwright) and a video that the browser decodes.
 * The Playwright Chromium build has no H.264 decoder: transcode first, for example
 * `ffmpeg -i in.mp4 -t 10 -vf scale=1280:-2 -c:v libvpx-vp9 -an out.webm`.
 * Two decodes on one machine and one browser build say nothing about two devices.
 */
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';

const root = resolve(import.meta.dirname, '..');
const flags = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith('--'))
    .map((a) => a.slice(2).split('=')),
);
const files = process.argv.slice(2).filter((a) => !a.startsWith('--'));
if (!files.length) throw new Error('Usage: node scripts/measure-decoder-agreement.mjs video...');
const fps = Number(flags.fps ?? 4),
  edge = Number(flags.edge ?? 640),
  repeat = Number(flags.repeat ?? 2);

const work = await mkdtemp(join(tmpdir(), 'decoder-agreement-'));
const lib = (name) => JSON.stringify(join(root, name));
const common = `
import { createPageReadback, createWorkerReadback, decodeFrameStream } from ${lib('website/src/lib/sfm-video.ts')};
import { frameKeyOf } from ${lib('src/reconstruction/identity.ts')};
export async function keys(file, path, edge, fps) {
  const out = [];
  const readback = path === 'page' ? createPageReadback() : createWorkerReadback();
  for await (const f of decodeFrameStream(file, new AbortController().signal, undefined, edge, fps, readback))
    out.push({ time: f.time, key: await frameKeyOf(f.pixels) });
  return out;
}`;
await writeFile(join(work, 'common.ts'), common);
await writeFile(
  join(work, 'page.ts'),
  `import { keys } from './common.ts';
(self as any).measure = async (url: string, path: string, edge: number, fps: number) => {
  const blob = await (await fetch(url)).blob();
  const file = new File([blob], 'video');
  if (path === 'page') return keys(file, 'page', edge, fps);
  const worker = new Worker('/worker.js', { type: 'module' });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => (worker.terminate(), e.data.error ? reject(new Error(e.data.error)) : resolve(e.data));
    worker.postMessage({ file, edge, fps });
  });
};`,
);
await writeFile(
  join(work, 'worker.ts'),
  `import { keys } from './common.ts';
self.onmessage = async (e: MessageEvent) => {
  try { postMessage(await keys(e.data.file, 'worker', e.data.edge, e.data.fps)); }
  catch (error) { postMessage({ error: String(error) }); }
};`,
);
for (const name of ['page', 'worker'])
  await build({
    entryPoints: [join(work, name + '.ts')],
    outfile: join(work, name + '.js'),
    bundle: true,
    format: 'esm',
    platform: 'browser',
    nodePaths: [join(root, 'website/node_modules')],
  });

const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  try {
    if (path === '/') {
      res.setHeader('content-type', 'text/html');
      res.end('<!doctype html><script type="module" src="/page.js"></script>');
    } else if (path === '/page.js' || path === '/worker.js') {
      res.setHeader('content-type', 'text/javascript');
      res.end(await readFile(join(work, path)));
    } else if (path.startsWith('/video/')) {
      res.end(await readFile(files[Number(path.slice(7))]));
    } else res.writeHead(404).end();
  } catch {
    res.writeHead(404).end();
  }
}).listen(0, '127.0.0.1');
await new Promise((done) => server.once('listening', done));

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(`http://localhost:${server.address().port}/`);
  await page.waitForFunction(() => typeof self.measure === 'function');
  const videos = [];
  for (const [index, file] of files.entries()) {
    const runs = { page: [], worker: [] };
    for (const path of ['page', 'worker'])
      for (let i = 0; i < repeat; i++)
        runs[path].push(
          await page.evaluate(
            ([u, p, e, f]) => self.measure(u, p, e, f),
            [`/video/${index}`, path, edge, fps],
          ),
        );
    const compare = (a, b) => {
      const n = Math.min(a.length, b.length);
      let different = 0;
      for (let k = 0; k < n; k++) if (a[k].key !== b[k].key || a[k].time !== b[k].time) different++;
      return { frames: n, countA: a.length, countB: b.length, different };
    };
    videos.push({
      file: basename(file),
      distinctKeys: new Set(runs.page[0].map((f) => f.key)).size,
      'repeat-page': compare(runs.page[0], runs.page[1] ?? runs.page[0]),
      'repeat-worker': compare(runs.worker[0], runs.worker[1] ?? runs.worker[0]),
      'page-worker': compare(runs.page[0], runs.worker[0]),
    });
  }
  const total = (kind) => {
    const frames = videos.reduce((s, v) => s + v[kind].frames, 0),
      different = videos.reduce((s, v) => s + v[kind].different, 0);
    return { frames, different, rate: frames ? different / frames : null };
  };
  console.log(
    JSON.stringify(
      {
        browser: `${browser.browserType().name()} ${browser.version()}`,
        date: new Date().toISOString().slice(0, 10),
        fps,
        edge,
        repeat,
        videos,
        total: {
          'repeat-page': total('repeat-page'),
          'repeat-worker': total('repeat-worker'),
          'page-worker': total('page-worker'),
        },
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
  server.close();
  await rm(work, { recursive: true, force: true });
}
