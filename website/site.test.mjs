import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';
import { runRecordProblem } from './src/lib/run-record.ts';

const directory = path.resolve(import.meta.dirname, 'dist');
const version = JSON.parse(await readFile(path.join(directory, 'version.json'), 'utf8'));
const fileFor = (pathname) =>
  path.join(directory, pathname.endsWith('/') ? pathname + 'index.html' : pathname);

test('all generated pages link to real pages, anchors, local runtimes and records', async () => {
  const failures = [];
  for (const route of version.routes) {
    const html = await readFile(fileFor(route), 'utf8');
    assert.equal((html.match(/<h1[ >]/g) ?? []).length, 1, route);
    assert.ok(!/cfat_[a-z0-9]{20,}/i.test(html), 'Credential marker in static content');
    for (const match of html.matchAll(/(?:href|src)="(\/[^"]*)"/g)) {
      const url = new URL(match[1], 'https://cv.typeshade.dev');
      const file = fileFor(decodeURIComponent(url.pathname));
      assert.ok(file.startsWith(directory + path.sep));
      try {
        await access(file);
      } catch {
        failures.push(route + ' → ' + match[1]);
        continue;
      }
      if (url.hash && (url.pathname.endsWith('/') || url.pathname.endsWith('.html'))) {
        const target = await readFile(file, 'utf8');
        if (!target.includes(`id="${decodeURIComponent(url.hash.slice(1))}"`))
          failures.push(route + ' → ' + match[1]);
      }
    }
  }
  assert.deepEqual(failures, []);
  const runtime = await readFile(path.join(directory, 'runtime/core.js'), 'utf8');
  assert.ok(runtime.includes('verifyImagePair'));
  assert.equal(version.project, 'TypeCV');
});

test('public UI runs real CPU/WebGPU verification, handles missing GPU and fits mobile', async () => {
  const mime = {
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.json': 'application/json',
    '.svg': 'image/svg+xml',
    '.woff2': 'font/woff2',
    '.txt': 'text/plain',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const file = fileFor(decodeURIComponent(url.pathname));
    if (!file.startsWith(directory + path.sep)) {
      res.writeHead(403);
      res.end();
      return;
    }
    try {
      const content = await readFile(file);
      res.setHeader('Content-Type', mime[path.extname(file)] ?? 'application/octet-stream');
      res.end(content);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end(await readFile(path.join(directory, '404.html')));
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args:
        process.env.TYPECV_HARDWARE_GPU === '1'
          ? [
              '--enable-unsafe-webgpu',
              ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : []),
            ]
          : [
              '--enable-unsafe-webgpu',
              '--enable-unsafe-swiftshader',
              '--use-angle=swiftshader',
              '--use-vulkan=swiftshader',
              '--enable-features=Vulkan',
            ],
    });
    const page = await browser.newPage(),
      errors = [],
      requested = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('request', (r) => requested.push(new URL(r.url()).pathname));
    const url = `http://127.0.0.1:${server.address().port}`;
    await page.goto(url + '/ko/playground/');
    await page.waitForFunction(() => !!window.__typecvPlayground);
    await page.selectOption('[name=backend]', 'cpu');
    await page.click('#run');
    await page.waitForFunction(() =>
      ['success', 'error', 'rejected'].includes(document.querySelector('#status').dataset.state),
    );
    const cpu = await page.evaluate(() => window.__typecvPlayground.result);
    assert.ok(
      cpu?.backend === 'cpu' && cpu.quality >= 90 && cpu.errorPx < 0.2,
      JSON.stringify(cpu),
    );
    assert.ok(requested.includes('/runtime/core.js'));
    assert.ok(!requested.includes('/runtime/gpu.js'), 'CPU reference must not load TypeShade');
    await page.selectOption('[name=backend]', 'webgpu');
    await page.click('#run');
    await page.waitForFunction(
      () =>
        ['success', 'error', 'rejected'].includes(document.querySelector('#status').dataset.state),
      undefined,
      { timeout: 45000 },
    );
    const gpu = await page.evaluate(() => window.__typecvPlayground.result);
    assert.ok(
      gpu?.backend === 'webgpu' && gpu.quality >= 90 && gpu.errorPx < 0.2,
      JSON.stringify(gpu),
    );
    // The model generation runs as chunks under the time budget: each chunk is a submission. The
    // GPU decides each batch's work (TypeShade change 0075), and one scheduling word is read
    // after every fourth batch of a search, so at most one per four batches.
    assert.ok(gpu.computeSubmissions > 1 && gpu.modelBatches > 0, JSON.stringify(gpu));
    assert.ok(4 * gpu.controlWords <= gpu.modelBatches, JSON.stringify(gpu));
    assert.equal(gpu.applicationReadbackBytes, 624 + 4 * 128 + 4 * gpu.controlWords);
    assert.ok(gpu.dispatches > 0);
    assert.deepEqual(gpu.cpuStages, []);
    await page.locator('.ant-switch').click();
    const mode = await page.evaluate(() => document.documentElement.dataset.theme);
    await page.reload();
    await page.waitForFunction(
      () =>
        document.documentElement.dataset.theme === 'light' ||
        document.documentElement.dataset.theme === 'dark',
    );
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), mode);
    await page.setViewportSize({ width: 390, height: 844 });
    for (const route of [
      '/ko/',
      '/docs/gpu-initialization/',
      '/ko/playground/',
      '/ko/lab/',
      '/lab/probe/',
    ]) {
      await page.goto(url + route);
      assert.ok(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        route,
      );
    }
    await page.addInitScript(() => {
      window.__viewerContexts = 0;
      const seen = new WeakSet(),
        native = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (...args) {
        const context = Reflect.apply(native, this, args);
        if (/^webgl2?$/.test(args[0]) && context && !seen.has(context)) {
          seen.add(context);
          window.__viewerContexts++;
        }
        return context;
      };
    });
    await page.goto(url + '/lab/');
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await page.evaluate(() => {
      window.__viewerCanvas = document.querySelector('.sfm-map .viewer canvas');
    });
    assert.equal(await page.locator('.sfm-map .viewer').getAttribute('data-backend'), 'webgl');
    assert.equal(await page.locator('.sfm-map .viewer').getAttribute('data-points'), '0');
    assert.equal(await page.locator('.sfm-map .viewer').getAttribute('data-views'), '0');
    assert.equal(await page.evaluate(() => window.__viewerContexts), 1);
    const readableLab = () => {
      const heading = document.querySelector('.lab-panel h4');
      if (!heading) return false;
      const red = Number(getComputedStyle(heading).color.match(/[\d.]+/)?.[0]);
      return document.documentElement.dataset.theme === 'dark' ? red > 180 : red < 100;
    };
    await page.waitForFunction(readableLab);
    await page.locator('.ant-switch').click();
    await page.waitForFunction(readableLab);
    await page.locator('.ant-switch').click();
    await page.waitForFunction(readableLab);
    await page.evaluate(() => {
      window.__sfmTrace = [];
      window.__workerViewerStates = [];
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          window.__workerViewerStates.push(
            document.querySelector('.sfm-map .viewer')?.dataset.state,
          );
          this.addEventListener('message', ({ data }) => {
            if (['features', 'matches', 'snapshot', 'frameComplete', 'result'].includes(data.type))
              window.__sfmTrace.push({
                type: data.type,
                views: data.reconstruction?.views.length,
                points: data.points?.length,
                time: data.time,
                // The result's registration detail, for the assertion message when a view is missing.
                unregistered: data.unregistered,
                pairs: data.diagnostics?.pairVerification?.map((pair) => [
                  pair.first,
                  pair.second,
                  pair.inliers,
                ]),
                featureCounts: data.featureCounts,
                frameMs: data.frameMs,
              });
          });
        }
      };
    });
    assert.ok(await page.locator('input[type=file]').count());
    assert.ok(await page.locator('.ant-input-number-input').count());
    // The sample catalog lists every sample directory; the standard benchmark's pyramid is one card.
    await page.waitForFunction(() => document.querySelectorAll('.sample-card').length >= 7);
    assert.equal(await page.locator('.sample-card[data-kind=photos]').count(), 6);
    await page.getByRole('button', { name: /Load Pexels pyramid video/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 8);
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    const sfmStatus = await page.locator('.lab-status').textContent();
    assert.match(sfmStatus, /completed/i, sfmStatus);
    await page.waitForSelector('.viewer canvas', { timeout: 15000 });
    assert.ok(await page.locator('.viewer canvas').count());
    // The Photo toggle round trip: off hides the photo plane, on shows it again (owner's report,
    // 2026-10-20: it stayed hidden after a re-check).
    const photoToggle = page.locator('.lab-switch', { hasText: /^Photo$/ });
    const photoVisible = (state) => () =>
      document.querySelector('.sfm-map .viewer')?.dataset.photoVisible === state;
    await page.waitForFunction(photoVisible('shown'));
    await photoToggle.click();
    await page.waitForFunction(photoVisible('hidden'));
    await photoToggle.click();
    await page.waitForFunction(photoVisible('shown'));
    const trace = await page.evaluate(() => window.__sfmTrace);
    assert.ok(trace.some((event) => event.type === 'features' && event.points > 0));
    assert.ok(trace.some((event) => event.type === 'matches'));
    assert.ok(
      trace.findIndex((event) => event.type === 'snapshot') <
        trace.findIndex((event) => event.type === 'result'),
    );
    assert.ok(trace.filter((event) => event.type === 'snapshot').length > 1);
    assert.match(await page.locator('.sfm-throughput').innerText(), /8 frames processed/);
    const sfmSummary = await page.locator('.sfm-summary').innerText();
    const sfmStages = JSON.parse(await page.locator('.run-details details pre').textContent());
    const boundedStages = sfmStages.filter((stage) =>
      ['pairVerification', 'pnp', 'bundle'].includes(stage.operation),
    );
    assert.ok(boundedStages.length > 0);
    assert.ok(boundedStages.every((stage) => stage.dispatches >= stage.submissions));
    assert.ok(boundedStages.some((stage) => stage.dispatches > stage.submissions));
    assert.equal(sfmStages.find((stage) => stage.operation === 'runTotal').reusedDevice, false);
    // The run record: shown with its exact JSON, sent only after consent, validated like the collector.
    const posted = [];
    await page.route('**/api/records/', async (route) => {
      posted.push(route.request().postDataJSON());
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          id: '20260101000000000-deadbeef',
          receivedAt: '2026-01-01T00:00:00Z',
        }),
      });
    });
    const shownRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(
      runRecordProblem(shownRecord),
      undefined,
      JSON.stringify(shownRecord).slice(0, 400),
    );
    assert.equal(shownRecord.input.kind, 'sample');
    assert.equal(shownRecord.input.frames, 8);
    // With SIFT every consecutive pair of the pyramid sample verifies with well over a hundred
    // inliers, so all eight frames register; the message names the worker's result otherwise.
    assert.equal(
      shownRecord.result.registeredViews,
      8,
      JSON.stringify(trace.find((event) => event.type === 'result')),
    );
    assert.equal(shownRecord.timing.reusedDevice, false);
    assert.equal(shownRecord.benchmark, null);
    assert.ok(shownRecord.timing.frameMs.length === 8);
    assert.ok(Object.keys(shownRecord.timing.stages).includes('bundle'));
    // The chunk measurements of the time budget (TypeShade change 0074) reach the record and the page.
    assert.equal(shownRecord.timing.budgetMs, 250);
    const chunked = JSON.stringify(shownRecord.timing.chunks);
    assert.ok(shownRecord.timing.chunks.scorePairModels?.chunks > 0, chunked);
    // Six of the eight views register by PnP, whose scoring is a chunk step too.
    assert.ok(shownRecord.timing.chunks.scorePnP?.chunks > 0, chunked);
    assert.ok(shownRecord.timing.chunks.bundleCg?.chunks > 0, chunked);
    assert.ok(shownRecord.timing.chunks.triangulateTracks?.chunks > 0, chunked);
    assert.match(
      await page.locator('.run-record-chunks').innerText(),
      /Time budget 250 ms\n(.|\n)*scorePairModels: \d+ chunks?, longest \d+ ms, [\d.e-]+ ms\/unit/,
    );
    assert.ok(!JSON.stringify(shownRecord).includes('data:image'));
    assert.ok(await page.getByRole('button', { name: /Send record/ }).isDisabled());
    await page.locator('.run-record .ant-checkbox-input').check();
    await page.getByRole('button', { name: /Send record/ }).click();
    await page.waitForFunction(() =>
      /Sent: 20260101000000000-deadbeef/.test(
        document.querySelector('.run-record-status')?.textContent ?? '',
      ),
    );
    assert.equal(posted.length, 1);
    assert.equal(runRecordProblem(posted[0]), undefined);
    // The adapter identity reaches the record; SwiftShader reports its vendor and architecture.
    assert.ok(posted[0].gpu.vendor || posted[0].gpu.architecture, JSON.stringify(posted[0].gpu));
    await page.unroute('**/api/records/');
    assert.deepEqual(await page.evaluate(() => window.__workerViewerStates), ['ready']);
    assert.equal(
      await page.evaluate(
        () => document.querySelector('.sfm-map .viewer canvas') === window.__viewerCanvas,
      ),
      true,
    );
    // A successful job keeps its worker/device so the next job can reuse compiled pipelines.
    assert.equal(page.workers().length, 1);
    const retainedWorker = page.workers()[0];
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /completed/i);
    assert.equal(page.workers()[0], retainedWorker);
    assert.equal(page.workers().length, 1);
    const repeatedStages = JSON.parse(await page.locator('.run-details details pre').textContent());
    assert.equal(repeatedStages.find((stage) => stage.operation === 'runTotal').reusedDevice, true);
    assert.equal(
      await page.evaluate(
        () => document.querySelector('.sfm-map .viewer canvas') === window.__viewerCanvas,
      ),
      true,
    );
    assert.equal(await page.evaluate(() => window.__viewerContexts), 1);

    const prepared = await page.evaluate(async () => {
      const frames = [];
      for (const [id, image] of [...document.querySelectorAll('.frame-grid img')].entries()) {
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.getContext('2d').drawImage(image, 0, 0);
        frames.push({
          id,
          width: canvas.width,
          height: canvas.height,
          time: id,
          image: canvas.toDataURL('image/png'),
        });
      }
      return {
        format: 'typecv-sfm-video-v1',
        source: {
          name: '32k-input.mkv',
          width: 32768,
          height: Math.round((32768 * frames[0].height) / frames[0].width),
          duration: 8,
          fps: 1,
        },
        frames,
      };
    });
    await page.locator('input[type=file]').setInputFiles({
      name: '32k-input.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(prepared)),
    });
    await page.getByRole('button', { name: /Prepare analysis frames/ }).click();
    await page.waitForFunction(() =>
      /analysis frames are ready/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    assert.match(await page.locator('.lab-panel').innerText(), /Source 32768/);
    assert.equal(await page.locator('.frame-grid img').count(), 8);
    assert.equal(page.workers().length, 0, 'changing input terminates the retained worker');
    await page.waitForFunction(() => {
      const viewer = document.querySelector('.sfm-map .viewer');
      return (
        viewer?.dataset.points === '0' &&
        viewer.dataset.views === '0' &&
        viewer.dataset.selectedView === '' &&
        viewer.dataset.selectedPoint === '' &&
        !viewer.dataset.photo
      );
    });
    assert.equal(
      await page.evaluate(
        () => document.querySelector('.sfm-map .viewer canvas') === window.__viewerCanvas,
      ),
      true,
    );
    assert.equal(await page.evaluate(() => window.__viewerContexts), 1);
    // Decode an actual WebM containing the real sample footage through the video element.
    // GPU processing and worker messages remain real; no runtime or numerical stages are mocked.
    const videoBytes = await page.evaluate(
      async (images) => {
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 180;
        const ctx = canvas.getContext('2d');
        const bitmaps = await Promise.all(
          images.map(async (image) => createImageBitmap(await (await fetch(image)).blob())),
        );
        ctx.drawImage(bitmaps[0], 0, 0, 320, 180);
        const stream = canvas.captureStream(10);
        const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8' });
        const chunks = [];
        recorder.ondataavailable = ({ data }) => chunks.push(data);
        const stopped = new Promise((resolve) => {
          recorder.onstop = resolve;
        });
        recorder.start();
        for (const bitmap of bitmaps) {
          ctx.drawImage(bitmap, 0, 0, 320, 180);
          await new Promise((resolve) => setTimeout(resolve, 800));
          bitmap.close();
        }
        recorder.stop();
        await stopped;
        for (const track of stream.getTracks()) track.stop();
        return [...new Uint8Array(await new Blob(chunks).arrayBuffer())];
      },
      prepared.frames.map((frame) => frame.image),
    );
    await page.locator('input[type=file]').setInputFiles({
      name: 'real-pyramid-video.webm',
      mimeType: 'video/webm',
      buffer: Buffer.from(videoBytes),
    });
    await page.waitForFunction(() => document.querySelector('video')?.videoWidth === 320);
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /completed/i);
    assert.ok((await page.locator('.frame-grid img').count()) >= 2);
    assert.match(await page.locator('.sfm-timing').innerText(), /Video [1-9]/);
    // Ten frames per second of a slideshow: the repeated frames are not keyframes.
    assert.match(await page.locator('.sfm-throughput').innerText(), /[1-9]\d* frames skipped/);
    assert.ok(
      await page.locator('.sfm-analysis-canvas').evaluate((canvas) => canvas.width === 320),
    );
    const videoStages = JSON.parse(await page.locator('.run-details details pre').textContent());
    assert.ok(videoStages.filter((stage) => stage.operation === 'keyframe').length >= 2);
    assert.ok(
      videoStages.some((stage) => stage.operation === 'pnp' || stage.operation === 'bundle'),
    );
    // The COLMAP sample with the dataset's SIFT features: the mapper alone, held to COLMAP's cameras.
    // antd hides the radio input; its wrapper label takes the click.
    await page.locator('.ant-radio-button-wrapper', { hasText: /Dataset SIFT features/ }).click();
    await page.getByRole('button', { name: /Load COLMAP south-building \(12 photos\)/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 12);
    assert.match(await page.locator('.sfm-sample-credit').innerText(), /Christopher Zach/);
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /completed/i);
    const siftReference = page.locator('.sfm-reference');
    assert.equal(await siftReference.getAttribute('data-status'), 'aligned');
    assert.equal(await siftReference.getAttribute('data-common-views'), '12');
    const siftRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(siftRecord), undefined, JSON.stringify(siftRecord).slice(0, 400));
    assert.equal(siftRecord.input.kind, 'sample');
    assert.equal(siftRecord.input.name, 'COLMAP south-building (COLMAP SIFT)');
    assert.equal(siftRecord.result.registeredViews, 12);
    assert.equal(siftRecord.options.refineSharedFocal, false);
    assert.equal(siftRecord.reference.name, 'COLMAP south-building (COLMAP SIFT)');
    assert.equal(siftRecord.reference.status, 'aligned');
    assert.equal(siftRecord.reference.commonViews, 12);
    assert.ok(
      siftRecord.reference.medianRotationDeg < 1,
      'median rotation ' + siftRecord.reference.medianRotationDeg,
    );
    assert.ok(
      siftRecord.reference.medianCenterOfRadius < 0.05,
      'median center ' + siftRecord.reference.medianCenterOfRadius,
    );
    // The analysis resolution selector loads a sample's 1280 px preview set, and 640 again.
    await page.locator('.ant-radio-button-wrapper', { hasText: /1280 px/ }).click();
    await page.getByRole('button', { name: /Load COLMAP south-building \(12 photos\)/ }).click();
    await page.waitForFunction(() =>
      /photographs at 1280 × 960/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.locator('.ant-radio-button-wrapper', { hasText: /640 px/ }).click();
    // The same photographs through the GPU front end with the photo profile.
    await page.locator('.ant-radio-button-wrapper', { hasText: /GPU front end/ }).click();
    await page.getByRole('button', { name: /Load COLMAP south-building \(12 photos\)/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 12);
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /completed/i);
    const photoRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(photoRecord), undefined);
    assert.equal(photoRecord.input.name, 'COLMAP south-building');
    assert.equal(photoRecord.options.maxFeatures, 4096);
    assert.ok(photoRecord.result.featureCounts.every((count) => count > 200 && count <= 4096));
    // SIFT registers every photograph through the full GPU pipeline and agrees with COLMAP.
    assert.equal(photoRecord.result.registeredViews, 12, JSON.stringify(photoRecord.result));
    assert.equal(photoRecord.reference.status, 'aligned');
    assert.equal(photoRecord.reference.commonViews, 12);
    assert.ok(
      photoRecord.reference.medianRotationDeg < 1,
      'median rotation ' + photoRecord.reference.medianRotationDeg,
    );
    if (process.env.TYPECV_SITE_SCREENSHOTS) {
      await mkdir(process.env.TYPECV_SITE_SCREENSHOTS, { recursive: true });
      await page.setViewportSize({ width: 1440, height: 1100 });
      await page.screenshot({
        path: path.join(process.env.TYPECV_SITE_SCREENSHOTS, 'sfm-live.png'),
        fullPage: true,
      });
    }
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    const closed = new Promise((resolve) => page.workers()[0].once('close', resolve));
    await page.getByRole('button', { name: /^Cancel$/ }).click();
    await closed;
    await page.waitForFunction(() =>
      /cancelled/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    assert.equal(page.workers().length, 0);
    const failedViewer = await browser.newPage();
    await failedViewer.addInitScript(() => {
      // Real context creation succeeds; fail the first actual empty-scene clear.
      const native = WebGL2RenderingContext.prototype.clear;
      WebGL2RenderingContext.prototype.clear = function (...args) {
        Reflect.apply(native, this, args);
        throw new Error('Injected initial viewer render failure');
      };
    });
    await failedViewer.goto(url + '/lab/');
    await failedViewer.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'error',
    );
    assert.match(
      await failedViewer.locator('.sfm-map [role=alert]').innerText(),
      /Injected initial viewer render failure/,
    );
    assert.equal(await failedViewer.locator('.sfm-map .viewer canvas').count(), 0);
    await failedViewer.getByRole('button', { name: /Load Pexels pyramid video/ }).click();
    await failedViewer.waitForFunction(
      () => document.querySelectorAll('.frame-grid img').length === 8,
    );
    // A viewer that failed does not stop the run: the worker starts and the alert stays (Safari
    // after a GPU process loss has no WebGL context, and the GPU run is what the lab measures).
    const workerStarted = new Promise((resolve) => failedViewer.once('worker', resolve));
    await failedViewer.getByRole('button', { name: /Run GPU SfM/ }).click();
    await workerStarted;
    assert.equal(
      failedViewer.workers().length,
      1,
      'a failed viewer must not prevent the numerical worker from starting',
    );
    assert.match(
      await failedViewer.locator('.sfm-map [role=alert]').innerText(),
      /Injected initial viewer render failure/,
    );
    const workerClosed = new Promise((resolve) => failedViewer.workers()[0].once('close', resolve));
    await failedViewer.getByRole('button', { name: /^Cancel$/ }).click();
    await workerClosed;
    await failedViewer.waitForFunction(() =>
      /cancelled/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await failedViewer.close();
    if (process.env.TYPECV_SITE_SCREENSHOTS) {
      await mkdir(process.env.TYPECV_SITE_SCREENSHOTS, { recursive: true });
      await page.goto(url + '/ko/');
      await page.screenshot({
        path: path.join(process.env.TYPECV_SITE_SCREENSHOTS, 'mobile.png'),
        fullPage: true,
      });
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.screenshot({
        path: path.join(process.env.TYPECV_SITE_SCREENSHOTS, 'desktop.png'),
        fullPage: true,
      });
    }
    const unavailable = await browser.newPage();
    await unavailable.addInitScript(() =>
      Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }),
    );
    await unavailable.goto(url + '/playground/');
    await unavailable.click('#run');
    await unavailable.waitForFunction(
      () => document.querySelector('#status').dataset.state === 'error',
    );
    assert.equal(await unavailable.locator('[name=backend]').inputValue(), 'webgpu');
    assert.equal(await unavailable.evaluate(() => window.__typecvPlayground.result), undefined);
    assert.match(await unavailable.locator('#status').textContent(), /WebGPU is unavailable/);
    const missing = await page.goto(url + '/missing-route/');
    assert.equal(missing.status(), 404);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ cpu, gpu, sfmStatus, sfmSummary, pages: version.routes.length }));
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
