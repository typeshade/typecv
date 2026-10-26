// Phase 0 of the refactoring programme (`docs/library-structure.md`, section 5, "Baselines": the
// start payloads and record options of about 40 query strings). For each query string of
// `LAB_QUERIES` (`lab-page-module.mjs`) the real lab page (`SfmLab.tsx`) runs a sample video in
// Chromium with the real SfM worker. The worker imports the deterministic fake runtime of
// `lab-fake-runtime.mjs` from `/runtime/gpu.js`. The test records two things: the `start` (or
// `multiStart`) message that the page sends, and the `options` of the run record that the page
// builds. The fixture is `tests/fixtures/lab-run-options.json`; phase 7 keeps both identical.
//
// Masked: `options.keyframes`, a count of the run that depends on the decoded pixels, not on the
// query string.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { canonical, checkBaseline } from './lab-baseline.mjs';
import { createFakeRuntime } from './lab-fake-runtime.mjs';
import { LAB_QUERIES, LAB_QUERY_BASE } from './lab-page-module.mjs';
import { fixtureVideo, imageFixture } from './video-fixture.mjs';
import { runRecordProblem } from './src/lib/run-record.ts';

/** Pages that run at the same time. */
const PAGES = 3;

/** The page with a `Worker` that records the start message and runs the real worker. */
const harness = `
import React from 'react';
import {createRoot} from 'react-dom/client';
import SfmLab from './src/components/SfmLab.tsx';
const RealWorker = window.Worker;
window.startMessages = [];
window.Worker = class extends RealWorker {
  constructor(url, options) {
    const decode = String(url).includes('decode.worker');
    super(decode ? '/decode-worker.js' : '/sfm-worker.js', {type: 'module'});
    this.decode = decode;
  }
  postMessage(message, transfer) {
    // An undefined field keeps its key, with the tag of \`canonical\` (\`lab-baseline.mjs\`).
    if (!this.decode && (message?.type === 'start' || message?.type === 'multiStart'))
      window.startMessages.push(JSON.stringify(message, (key, value) => value === undefined ? {$: 'undefined'} : value));
    return transfer === undefined ? super.postMessage(message) : super.postMessage(message, transfer);
  }
};
createRoot(document.getElementById('app')).render(<SfmLab locale="en" />);
`;

/** The adapter of the worker: the real one would name the machine's GPU. */
const adapterShim = `Object.defineProperty(self.navigator, 'gpu', {configurable: true, value: {
  requestAdapter: async () => ({
    info: {vendor: 'typecv-test', architecture: 'fake', device: 'browser', description: 'Deterministic fake adapter'},
    isFallbackAdapter: false,
    limits: {maxBufferSize: 268435456, maxStorageBufferBindingSize: 134217728,
      maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256,
      maxStorageBuffersPerShaderStage: 10},
    features: new Set(['timestamp-query', 'shader-f16']),
  }),
}});
`;

/** The fake runtime as a module: the same function that the transcript harness compiles. */
function runtimeModule(spec) {
  const names = Object.keys(createFakeRuntime(spec).module);
  return (
    `const createFakeRuntime = ${createFakeRuntime.toString()};\n` +
    `const { module } = createFakeRuntime(${JSON.stringify(spec)}, () => {});\n` +
    `export const { ${names.join(', ')} } = module;\n`
  );
}

/** Two video samples of 320 × 180 pixels and 3 s that store a source video, and their set. */
const samples = ['a', 'b'].map((id) => ({
  id,
  kind: 'video',
  dir: `/sample/${id}/`,
  name: `Clip ${id.toUpperCase()}`,
  frames: 3,
  thumbnail: 'frame-00.jpg',
  size: [320, 180],
  description: { en: 'Test video', ko: '테스트 영상' },
  video: { name: `${id}.mp4`, times: [0, 1, 2], duration: 3 },
  source: {
    file: 'source.mp4',
    videoId: id.repeat(16),
    bytes: 1,
    width: 320,
    height: 180,
    codec: 'vp9',
  },
}));
const catalog = {
  format: 'typecv-lab-samples-v1',
  samples: [
    ...samples,
    {
      id: 'clips',
      kind: 'video-set',
      name: 'Two clips',
      members: ['a', 'b'],
      description: { en: 'Test set', ko: '테스트' },
    },
  ],
};

async function bundles() {
  const common = { bundle: true, write: false, format: 'esm', logLevel: 'silent' };
  const worker = (name, extra = {}) =>
    build({
      ...common,
      entryPoints: [new URL(`./src/workers/${name}`, import.meta.url).pathname],
      platform: 'browser',
      ...extra,
    });
  const [app, sfm, decode] = await Promise.all([
    build({
      ...common,
      stdin: { contents: harness, resolveDir: import.meta.dirname, loader: 'tsx' },
      jsx: 'automatic',
      define: { 'process.env.NODE_ENV': '"production"' },
    }),
    worker('sfm.worker.ts', { banner: { js: adapterShim } }),
    worker('decode.worker.ts'),
  ]);
  return {
    '/app.js': app.outputFiles[0].text,
    '/sfm-worker.js': sfm.outputFiles[0].text,
    '/decode-worker.js': decode.outputFiles[0].text,
    '/runtime/gpu.js': runtimeModule({ multi: {} }),
    '/runtime/internal/gpu.js': runtimeModule({ multi: {} }),
  };
}

/** The lab's files, the sample catalog, the clip and the fake runtime. */
async function serve() {
  const scripts = await bundles();
  // Three different frames, one per second (`video-fixture.mjs`, ffmpeg).
  const images = await Promise.all(
    [0, 3, 6].map((frame) => readFile(imageFixture('south-building', frame))),
  );
  const clip = await fixtureVideo(images, 320, 180);
  const css = await readFile(new URL('./src/styles/global.css', import.meta.url), 'utf8');
  const server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (Object.hasOwn(scripts, path))
      return res.writeHead(200, { 'Content-Type': 'text/javascript' }).end(scripts[path]);
    if (path === '/runtime/vocabulary/rootsift-32-v1.bin')
      return res.writeHead(200).end(Buffer.from([1, 2, 3, 4]));
    if (path === '/sample/index.json')
      return res
        .writeHead(200, { 'Content-Type': 'application/json' })
        .end(JSON.stringify(catalog));
    if (path.endsWith('.jpg'))
      return res.writeHead(200, { 'Content-Type': 'image/jpeg' }).end(images[0]);
    if (path.endsWith('.mp4')) {
      const match = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
      const start = match ? Number(match[1]) : 0;
      const end = match?.[2] ? Math.min(Number(match[2]), clip.length - 1) : clip.length - 1;
      res.writeHead(match ? 206 : 200, {
        'Content-Type': 'video/webm',
        'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1,
        ...(match ? { 'Content-Range': `bytes ${start}-${end}/${clip.length}` } : {}),
      });
      return res.end(clip.subarray(start, end + 1));
    }
    if (path === '/version.json')
      return res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"commit":"test"}');
    if (path.startsWith('/api/') || path.startsWith('/fonts/')) return res.writeHead(404).end();
    res
      .writeHead(200, { 'Content-Type': 'text/html' })
      .end(
        `<style>${css}</style><div id="app" class="lab-main"></div><script type="module" src="/app.js"></script>`,
      );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return server;
}

/** Run the page once for `search` and return what the baseline keeps of it. */
async function capture(browser, origin, kind, search) {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  try {
    await page.goto(`${origin}/?${search}`);
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await page.locator('.lab-topbar .ant-btn').click();
    await page
      .getByRole('button', {
        name: kind === 'multi' ? 'Load Two clips' : 'Load Clip A',
        exact: true,
      })
      .click();
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const pre = page.locator('.run-record details pre');
    await pre.waitFor({ state: 'attached', timeout: 180000 });
    const record = JSON.parse(await pre.textContent());
    const starts = (await page.evaluate(() => window.startMessages)).map((text) =>
      JSON.parse(text),
    );
    assert.equal(starts.length, 1, search);
    assert.equal(runRecordProblem(record), undefined, search);
    assert.deepEqual(errors, [], search);
    const { keyframes: _keyframes, ...options } = record.options;
    return {
      query: search,
      kind,
      failure: record.failure ? record.failure.message.split('\n')[0] : null,
      start: canonical(starts[0]),
      options,
    };
  } finally {
    await page.close();
  }
}

test(
  'the start payloads and record options of the lab query strings',
  { timeout: 900000 },
  async () => {
    const server = await serve();
    const origin = `http://127.0.0.1:${server.address().port}`;
    let browser;
    const entries = new Array(LAB_QUERIES.length);
    try {
      browser = await chromium.launch({
        executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
        args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
      });
      let next = 0;
      const runner = async () => {
        while (next < LAB_QUERIES.length) {
          const index = next++;
          const [kind, query, base = true] = LAB_QUERIES[index];
          const search = [base ? LAB_QUERY_BASE : '', query].filter(Boolean).join('&');
          entries[index] = await capture(browser, origin, kind, search);
        }
      };
      await Promise.all(Array.from({ length: PAGES }, runner));
    } finally {
      await browser?.close();
      await new Promise((resolve) => server.close(resolve));
    }
    await checkBaseline(
      new URL('../tests/fixtures/lab-run-options.json', import.meta.url),
      JSON.stringify(entries, null, 1) + '\n',
    );
  },
);
