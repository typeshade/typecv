import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';
import { densePly, denseMeshPly } from './src/lib/dense-cloud.ts';
import { reconstructionJson } from './src/lib/result-export.ts';
import { sampleRecord } from './record-fixture.mjs';

// The lab opens saved results without a run: from files on the visitor's disk and from a result
// card of the catalog. The catalog and the result files are served by this test, as the Worker
// serves them from the R2 bucket.
const directory = path.resolve(import.meta.dirname, 'dist');

const scene = {
  views: [0, 1].map((id) => ({
    id,
    pose: { R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: Float64Array.from([-id, 0, 0]) },
    camera: { fx: 500, fy: 500, cx: 320, cy: 180 },
  })),
  landmarks: [0, 1, 2].map((i) => ({
    trackId: i,
    xyz: Float64Array.from([i, 1, 5]),
    color: [1, 0, 0],
    observations: [],
  })),
};
const json = reconstructionJson({
  source: { name: 'demo', width: 640, height: 360 },
  assumedFocalRatio: 0.8,
  diagnostics: {},
  unregistered: [],
  analysisFrames: [],
  scene,
  sourceCamera: (view) => view.camera,
  runRecord: sampleRecord(),
});
const dense = densePly([
  {
    viewId: 0,
    positions: Float32Array.from([0, 0, 4, 1, 0, 4, 0, 1, 4, 1, 1, 4]),
    colors: new Uint8Array(12).fill(200),
  },
]);
const mesh = denseMeshPly({
  positions: Float32Array.from([0, 0, 4, 1, 0, 4, 0, 1, 4]),
  colors: new Uint8Array(9).fill(100),
  indices: Uint32Array.of(0, 1, 2),
});
const files = {
  'typecv-reconstruction.json': [Buffer.from(json), 'application/json'],
  'typecv-dense.ply': [Buffer.from(dense), 'application/octet-stream'],
  'typecv-mesh.ply': [Buffer.from(mesh), 'application/octet-stream'],
};
const entry = (id, size) => ({
  id,
  kind: 'result',
  name: 'Demo ' + id,
  dir: `/sample/results/${id}/`,
  files: { json: 'typecv-reconstruction.json', dense: 'typecv-dense.ply', mesh: 'typecv-mesh.ply' },
  size,
  source: 'pyramid',
  remote: true,
  run: { date: '2026-10-21', device: 'swiftshader', commit: 'abcdef1' },
});

test('the lab opens result files and a result card without a run', async () => {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const served = files[path.basename(url.pathname)];
    if (url.pathname.startsWith('/sample/results/demo/') && served) {
      res.setHeader('Content-Type', served[1]);
      res.end(served[0]);
      return;
    }
    const file = path.join(
      directory,
      url.pathname.endsWith('/') ? url.pathname + 'index.html' : url.pathname,
    );
    try {
      const content = await readFile(file);
      const type = {
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.json': 'application/json',
      }[path.extname(file)];
      res.setHeader('Content-Type', type ?? 'application/octet-stream');
      if (url.pathname === '/sample/index.json') {
        const catalog = JSON.parse(content);
        const size = files['typecv-reconstruction.json'][0].length + dense.length + mesh.length;
        catalog.samples.push(entry('demo', size), entry('gone'));
        res.end(JSON.stringify(catalog));
        return;
      }
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}/lab/`);
    const viewer = page.locator('.sfm-map .viewer');
    await page.waitForFunction(() => document.querySelector('.sfm-map .viewer')?.dataset.state);
    if ((await viewer.getAttribute('data-state')) !== 'ready') {
      // No WebGL in this browser: the parsers are held by result-load.test.mjs.
      console.log('skip: the viewer did not start');
      return;
    }
    // The cards: the existing samples stay, the two results are added, the second is not published.
    await page.waitForSelector('.sample-card[data-kind=result]');
    assert.equal(await page.locator('.sample-card[data-kind=result]').count(), 2);
    const card = page.locator('.sample-card[data-sample=demo]');
    assert.ok((await card.innerText()).includes('SwiftShader'));
    await page.getByRole('button', { name: 'View Demo gone' }).click();
    await page.getByRole('button', { name: 'View Demo gone' }).waitFor({ state: 'attached' });
    await page.waitForFunction(() =>
      /Not published yet/.test(document.querySelector('[data-sample=gone]').textContent),
    );
    assert.ok(await page.getByRole('button', { name: 'View Demo gone' }).isDisabled());

    // View: the files download, and the viewer shows cameras, the dense cloud and the mesh switch.
    await page.getByRole('button', { name: 'View Demo demo' }).click();
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.views === '2',
    );
    assert.equal(await viewer.getAttribute('data-views'), '2');
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.cloud === 'dense',
    );
    assert.equal(await viewer.getAttribute('data-dense-points'), '4');
    assert.equal(await viewer.getAttribute('data-mesh-faces'), '1');
    assert.ok(
      await page
        .locator('.result-info')
        .innerText()
        .then((t) => /2 cameras/.test(t) && /3 points/.test(t)),
    );
    assert.equal(await card.getAttribute('data-loaded'), 'true');

    // Files from disk: the mesh alone shows the mesh at once.
    await page.locator('input[aria-label="Result files"]').setInputFiles({
      name: 'typecv-mesh.ply',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from(mesh),
    });
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.cloud === 'mesh',
    );
    assert.equal(await viewer.getAttribute('data-views'), '0');
    // Several at once: the JSON and the dense cloud.
    await page.locator('input[aria-label="Result files"]').setInputFiles([
      {
        name: 'typecv-dense.ply',
        mimeType: 'application/octet-stream',
        buffer: Buffer.from(dense),
      },
      {
        name: 'typecv-reconstruction.json',
        mimeType: 'application/json',
        buffer: Buffer.from(json),
      },
    ]);
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.cloud === 'dense',
    );
    assert.equal(await viewer.getAttribute('data-views'), '2');
    // A file that is not a result is refused with a message.
    await page.locator('input[aria-label="Result files"]').setInputFiles({
      name: 'x.json',
      mimeType: 'application/json',
      buffer: Buffer.from('{}'),
    });
    await page.waitForFunction(() =>
      /Cannot open the result/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
