import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

test('the lab visualizes real fusion events, replaces temporary points and clears finished activity', async () => {
  const directory = path.join(import.meta.dirname, 'dist');
  const server = createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const file = path.join(directory, pathname.endsWith('/') ? pathname + 'index.html' : pathname);
    if (!file.startsWith(directory + path.sep)) return res.writeHead(403).end();
    try {
      res.setHeader(
        'Content-Type',
        {
          '.js': 'text/javascript',
          '.css': 'text/css',
          '.html': 'text/html',
          '.json': 'application/json',
          '.jpg': 'image/jpeg',
          '.svg': 'image/svg+xml',
        }[path.extname(file)] ?? 'application/octet-stream',
      );
      res.end(await readFile(file));
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    page.on('console', (message) => {
      if (message.type() === 'error' && message.text().includes('THREE'))
        errors.push(message.text());
    });
    await page.addInitScript(() => {
      const model = {
        views: [0, 1].map((id) => ({
          id,
          pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-id, 0, 0] },
          camera: { fx: 100, fy: 100, cx: 50, cy: 50 },
        })),
        landmarks: [0, 1, 2].map((trackId) => ({
          trackId,
          xyz: [trackId, 0, 4],
          color: [1, 0.5, 0.2],
          observations: [],
        })),
      };
      window.Worker = class {
        constructor() {
          window.__denseWorker = this;
        }
        send(message) {
          this.onmessage?.({ data: { id: this.id, ...message } });
        }
        postMessage(message) {
          this.id = message.id;
          queueMicrotask(() => {
            if (message.type === 'start') this.send({ type: 'ready' });
            if (message.type === 'frame')
              this.send({
                type: 'frameComplete',
                frameId: message.frame.id,
                elapsedMs: 1,
                keyframe: true,
                cameraRegistered: true,
                registered: 2,
                verifiedInliers: 20,
              });
            if (message.type === 'finish') {
              this.send({
                type: 'result',
                reconstruction: model,
                diagnostics: {},
                unregistered: [],
                stages: [{ operation: 'runTotal', elapsedMs: 1 }],
                featureCounts: [20],
                frameMs: [1],
                matchedPairs: 1,
                skippedFrames: 0,
                keyframes: 2,
                kernels: [],
                chunks: {},
                options: {},
                summary: [],
              });
              setTimeout(
                () =>
                  this.send({
                    type: 'denseActivity',
                    phase: 'integrate',
                    view: 1,
                    views: 2,
                    viewId: 1,
                    pass: 1,
                    passes: 2,
                    done: 25,
                    total: 100,
                    points: 0,
                    elapsedMs: 1000,
                    phaseMs: 0,
                  }),
                50,
              );
            }
          });
        }
        terminate() {}
      };
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/lab/?dense=1`);
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await page.locator('.lab-topbar .ant-btn').click();
    await page.getByRole('button', { name: /Load Pyramid benchmark/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 8);
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    const activity = page.locator('.sfm-dense-activity');
    await activity.waitFor();
    assert.match(await activity.textContent(), /Combining depths.*Camera #1/);
    assert.equal(await activity.locator('progress').getAttribute('value'), '25');
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.pulseView === '1',
    );
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'denseFusionPoints',
        viewId: -1,
        positions: Float32Array.of(0, 0, 4, 1, 0, 4, 2, 0, 4),
        colors: new Uint8Array(9).fill(180),
        normals: new Int16Array(6),
      }),
    );
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.denseDrawn === '3',
    );
    await page.getByRole('button', { name: /Display/ }).click();
    const denseSwitch = page
      .locator('.sfm-map-options label', { hasText: /^Dense$/ })
      .getByRole('switch');
    assert.equal(await denseSwitch.isEnabled(), true);
    await denseSwitch.click();
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'denseFusionPoints',
        viewId: -1,
        positions: Float32Array.of(3, 0, 4),
        colors: new Uint8Array(3).fill(180),
        normals: new Int16Array(2),
      }),
    );
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.denseDrawn === '4',
    );
    assert.equal(await denseSwitch.getAttribute('aria-checked'), 'false');
    await denseSwitch.click();
    await page.keyboard.press('Escape');
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'denseActivity',
        phase: 'sort',
        pass: 2,
        passes: 2,
        done: 0,
        total: 1,
        points: 3,
        elapsedMs: 2500,
        phaseMs: 0,
      }),
    );
    await page.waitForFunction(
      () => document.querySelector('.sfm-dense-activity')?.dataset.phase === 'sort',
    );
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'densePoints',
        viewId: -1,
        positions: Float32Array.of(0, 0, 4, 1, 0, 4),
        colors: new Uint8Array(6).fill(180),
        normals: new Int16Array(4),
      }),
    );
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.denseDrawn === '2',
    );
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'denseActivity',
        phase: 'clean',
        done: 0,
        total: 0,
        elapsedMs: 3000,
        phaseMs: 0,
      }),
    );
    await page.waitForFunction(
      () => document.querySelector('.sfm-dense-activity')?.dataset.phase === 'clean',
    );
    assert.equal(await activity.locator('progress').getAttribute('value'), null);
    await page.setViewportSize({ width: 375, height: 812 });
    const box = await activity.boundingBox();
    assert.ok(box.x >= 0 && box.x + box.width <= 375);
    await page.evaluate(() =>
      window.__denseWorker.send({
        type: 'denseResult',
        points: 2,
        finalPoints: 2,
        cancelled: false,
        failure: null,
        skipped: '',
        views: [],
        options: {},
        timing: { totalMs: 3000, stageMs: {} },
        memory: {},
        kernels: [],
        chunks: {},
      }),
    );
    await activity.waitFor({ state: 'detached' });
    await page.locator('.sfm-live').waitFor({ state: 'detached' });
    assert.match(await page.locator('.lab-status').textContent(), /Dense MVS completed/i);
    const downloading = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export dense PLY' }).click();
    const download = await downloading;
    const stream = await download.createReadStream();
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    assert.match(Buffer.concat(chunks).toString('utf8'), /element vertex 2\n/);
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
