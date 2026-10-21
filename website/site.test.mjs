import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';
import { runRecordProblem } from './src/lib/run-record.ts';
import { reconstructionJson } from './src/lib/result-export.ts';

const directory = path.resolve(import.meta.dirname, 'dist');
const version = JSON.parse(await readFile(path.join(directory, 'version.json'), 'utf8'));
/** The lab's input drawer: opens it unless it is open already. */
async function openInput(page) {
  const body = page.locator('.lab-input-drawer .ant-drawer-body');
  if (await body.isVisible().catch(() => false)) return;
  await page.locator('.lab-topbar .ant-btn').click();
  await body.waitFor({ state: 'visible' });
}
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
    await page.locator('.header-tools .ant-switch').click();
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
    // The lab's runs keep the octave-0, unguided front end here: at the 640 px defaults (octave -1,
    // guided matching) a south-building run takes over 10 minutes on SwiftShader.
    // `website/sift-quality.test.mjs` holds the defaults themselves.
    await page.goto(url + '/lab/?octave=0&guided=0');
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
    await page.locator('.header-tools .ant-switch').click();
    await page.waitForFunction(readableLab);
    await page.locator('.header-tools .ant-switch').click();
    await page.waitForFunction(readableLab);
    // At desktop width the analysis column is 340-420 px and the 3D card takes the rest, with a
    // viewer as tall as the screen allows; the frame box and the viewport keep their sizes from
    // the idle state to a finished run.
    await page.setViewportSize({ width: 1440, height: 900 });
    const cardBoxes = () =>
      page.evaluate(() => {
        const box = (selector) => {
          const { width, height } = document.querySelector(selector).getBoundingClientRect();
          return { width: Math.round(width * 100) / 100, height: Math.round(height * 100) / 100 };
        };
        const stage = document.querySelector('.sfm-stream .sfm-stage');
        const video = stage.querySelector('video').getBoundingClientRect();
        const canvas = stage.querySelector('canvas').getBoundingClientRect();
        return {
          frame: box('.sfm-stream'),
          map: box('.sfm-map'),
          stage: box('.sfm-stream .sfm-stage'),
          viewport: box('.sfm-map .viewer'),
          frameTiles: box('.sfm-stream .sfm-tiles'),
          mapTiles: box('.sfm-map .sfm-tiles'),
          sameRect:
            video.x === canvas.x &&
            video.y === canvas.y &&
            video.width === canvas.width &&
            video.height === canvas.height,
        };
      });
    await page.waitForFunction(
      () =>
        innerWidth === 1440 &&
        getComputedStyle(document.querySelector('.sfm-live-grid')).gridTemplateColumns.split(' ')
          .length === 2,
    );
    const idleBoxes = await cardBoxes();
    assert.ok(
      idleBoxes.frame.width >= 340 && idleBoxes.frame.width <= 420,
      'the analysis column: ' + JSON.stringify(idleBoxes.frame),
    );
    assert.ok(
      idleBoxes.map.width > 2 * idleBoxes.frame.width,
      'the 3D card takes the rest: ' + JSON.stringify([idleBoxes.frame, idleBoxes.map]),
    );
    assert.ok(idleBoxes.viewport.height >= 460, 'the viewer is tall: ' + idleBoxes.viewport.height);
    assert.equal(idleBoxes.sameRect, true, 'the overlay canvas covers the video box');
    assert.equal(await page.locator('.sfm-stream video').count(), 1);
    assert.equal(await page.locator('.sfm-stream .sfm-stage canvas').count(), 1);
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
            if (
              [
                'features',
                'matches',
                'snapshot',
                'frameComplete',
                'result',
                'denseResult',
              ].includes(data.type)
            )
              window.__sfmTrace.push({
                type: data.type,
                cancelled: data.cancelled,
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
    assert.ok(await page.locator('input[name="file"]').count());
    assert.ok(await page.locator('.ant-input-number-input').count());
    // The sample catalog lists every sample directory; the standard benchmark's pyramid is one card.
    await openInput(page);
    await page.waitForFunction(() => document.querySelectorAll('.sample-card').length >= 7);
    assert.equal(await page.locator('.sample-card[data-kind=photos]').count(), 6);
    await page.getByRole('button', { name: /Load Pyramid benchmark/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 8);
    // The live view (`docs/lab-live-view.md`): record what the viewer and the map panel show while
    // the run is busy, since the run is over before the assertions read the page.
    await page.evaluate(() => {
      const seen = (window.__liveSeen = {
        added: 0,
        newViews: 0,
        pulse: new Set(),
        badge: false,
        inset: false,
      });
      const look = () => {
        const viewer = document.querySelector('.sfm-map .viewer');
        if (viewer) {
          seen.added = Math.max(seen.added, Number(viewer.dataset.added ?? 0));
          seen.newViews = Math.max(seen.newViews, Number(viewer.dataset.newViews ?? 0));
          if (viewer.dataset.pulse) seen.pulse.add(viewer.dataset.pulse);
        }
        seen.badge ||= !!document.querySelector('.sfm-map .sfm-live');
        seen.inset ||= !!document.querySelector('.sfm-map .sfm-inset canvas');
      };
      new MutationObserver(look).observe(document.body, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ['data-added', 'data-new-views', 'data-pulse'],
      });
    });
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
    const liveSeen = await page.evaluate(() => ({
      ...window.__liveSeen,
      pulse: [...window.__liveSeen.pulse],
    }));
    assert.ok(liveSeen.added > 0, JSON.stringify(liveSeen));
    assert.ok(liveSeen.newViews > 0, JSON.stringify(liveSeen));
    assert.ok(
      liveSeen.pulse.some((target) => target === 'all' || target === 'latest'),
      JSON.stringify(liveSeen),
    );
    assert.ok(liveSeen.badge && liveSeen.inset, JSON.stringify(liveSeen));
    // The selected camera, the last analysed frame, shows its 2D–3D links (3.5).
    assert.ok(Number(await page.locator('.sfm-map .viewer').getAttribute('data-links')) > 0);
    // Once the run is over the badge and the frame inset go, and nothing pulses.
    assert.equal(await page.locator('.sfm-map .sfm-live').count(), 0);
    assert.equal(await page.locator('.sfm-map .sfm-inset').count(), 0);
    await page.waitForSelector('.viewer canvas', { timeout: 15000 });
    assert.ok(await page.locator('.viewer canvas').count());
    // The Photo toggle round trip: off hides the photo plane, on shows it again (owner's report,
    // 2026-10-20: it stayed hidden after a re-check).
    // The switches live in the Display popover, which closes with Escape.
    await page.getByRole('button', { name: /Display/ }).click();
    const photoToggle = page.locator('.lab-switch', { hasText: /^Photo$/ });
    const photoVisible = (state) =>
      page.waitForFunction(
        (expected) => document.querySelector('.sfm-map .viewer')?.dataset.photoVisible === expected,
        state,
      );
    await photoVisible('shown');
    await photoToggle.click();
    await photoVisible('hidden');
    await photoToggle.click();
    await photoVisible('shown');
    // The mesh switch (`docs/dense-mvs.md` 3.13) is off by default, and disabled until a dense run
    // posts a mesh. The sparse run has none.
    // The display menu's switch; the run settings hold a second Mesh switch (with Dense MVS).
    const meshToggle = page.locator('.sfm-map-options .lab-switch', { hasText: /^Mesh$/ });
    assert.equal(await meshToggle.count(), 1);
    assert.equal(await meshToggle.getByRole('switch').isChecked(), false);
    assert.equal(await meshToggle.getByRole('switch').isDisabled(), true);
    assert.equal(await page.locator('.sfm-map .viewer').getAttribute('data-mesh-shown'), 'false');
    await page.keyboard.press('Escape');
    const trace = await page.evaluate(() => window.__sfmTrace);
    assert.ok(trace.some((event) => event.type === 'features' && event.points > 0));
    assert.ok(trace.some((event) => event.type === 'matches'));
    assert.ok(
      trace.findIndex((event) => event.type === 'snapshot') <
        trace.findIndex((event) => event.type === 'result'),
    );
    assert.ok(trace.filter((event) => event.type === 'snapshot').length > 1);
    assert.equal(await page.locator('[data-tile=processed] .sfm-tile-value').innerText(), '8');
    const sfmSummary = await page.locator('.sfm-tiles').nth(1).innerText();
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
    // The record waits folded under the run buttons; open it to read and send.
    await page.locator('#run-record > summary').click();
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
    // "Also send the result files" is off, and disabled here: this page has no collector config.
    const filesBox = page.getByRole('checkbox', { name: /Also send the result files/ });
    assert.equal(await filesBox.isChecked(), false);
    assert.ok(await filesBox.isDisabled());
    await page.locator('.run-record-consent .ant-checkbox-input').first().check();
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

    // `docs/dense-mvs.md` 5.3 (step 6): with `?dense=1` the worker posts the sparse result, then
    // runs the dense MVS on the same device. A cancel stops it between chunks and keeps the sparse
    // map and the worker. A whole dense run of this sample takes about 20 minutes on SwiftShader
    // ("Step 6 record"), so the test cancels it and the lab measurement covers the full run.
    await page.evaluate(() =>
      history.replaceState(null, '', location.pathname + location.search + '&dense=1'),
    );
    const denseWorker = page.workers()[0];
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /Starting the dense MVS|Dense MVS:|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 240000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /dense MVS|Dense MVS/);
    await page.getByRole('button', { name: /Cancel dense MVS/ }).click();
    await page.waitForFunction(
      () =>
        /Dense MVS (cancelled|completed|failed|skipped)/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      undefined,
      { timeout: 120000 },
    );
    assert.match(await page.locator('.lab-status').textContent(), /Dense MVS cancelled/);
    // The sparse result comes first, then one `denseResult` that says it was cancelled.
    const denseTrace = (await page.evaluate(() => window.__sfmTrace)).filter((event) =>
      ['result', 'denseResult'].includes(event.type),
    );
    assert.deepEqual(
      denseTrace.slice(-2).map((event) => [event.type, event.cancelled]),
      [
        ['result', undefined],
        ['denseResult', true],
      ],
    );
    assert.equal(page.workers()[0], denseWorker);
    assert.ok(Number(await page.locator('.sfm-map .viewer').getAttribute('data-points')) > 0);
    const denseRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(denseRecord), undefined);
    assert.equal(denseRecord.format, 'typecv-run-record-v5');
    assert.equal(denseRecord.result.registeredViews, 8);
    assert.equal(denseRecord.dense.status, 'cancelled', JSON.stringify(denseRecord.dense));
    assert.deepEqual(denseRecord.dense.options, {
      maxSources: 6,
      bestSources: 3,
      windowRadius: 5,
      windowStep: 2,
      photometricIterations: 8,
      geometricIterations: 4,
      minConsistentViews: 3,
      denseClean: true,
      denseMesh: false,
    });
    // 4.3: the device's bytes before the first MVS allocation are recorded.
    assert.ok(Number.isSafeInteger(denseRecord.dense.memory.liveBytesBefore));
    assert.equal(denseRecord.failure, null);
    await page.locator('#run-record').evaluate((details) => (details.open = true));
    assert.match(await page.locator('.run-record-chunks').innerText(), /Dense MVS cancelled/);
    await page.evaluate(() =>
      history.replaceState(null, '', location.pathname + location.search.replace('&dense=1', '')),
    );

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
    await openInput(page);
    await page.locator('input[name="file"]').setInputFiles({
      name: '32k-input.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(prepared)),
    });
    await page.getByRole('button', { name: /Prepare analysis frames/ }).click();
    await page.waitForFunction(() =>
      /analysis frames are ready/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    assert.match(await page.locator('.lab-run-column').innerText(), /Source 32768/);
    assert.equal(await page.locator('.frame-grid img').count(), 8);
    assert.equal(page.workers().length, 0, 'changing input terminates the retained worker');
    await page.waitForFunction(() => {
      const viewer = document.querySelector('.sfm-map .viewer');
      return (
        viewer?.dataset.points === '0' &&
        viewer.dataset.views === '0' &&
        viewer.dataset.selectedView === '' &&
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
    await openInput(page);
    await page.locator('input[name="file"]').setInputFiles({
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
    assert.match(await page.locator('[data-tile=time] .sfm-tile-value').innerText(), /^[1-9]/);
    // Ten frames per second of a slideshow: the repeated frames are not keyframes.
    assert.match(
      await page.locator('[data-tile=skipped] .sfm-tile-value').innerText(),
      /^[1-9]\d*$/,
    );
    // The page takes 4 frames per second of the upload; each 0.8 s slide still repeats.
    assert.match(
      (await page.locator('.sfm-sampling').textContent()) ?? '',
      /sampled at 4 frames\/s/,
    );
    assert.ok(
      await page.locator('.sfm-analysis-canvas').evaluate((canvas) => canvas.width === 320),
    );
    // The upload decoded in the decode worker, and the run record names that path.
    const videoRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(videoRecord), undefined);
    assert.equal(videoRecord.options.videoDecode, 'worker');
    assert.equal(videoRecord.options.videoSampleFps, 4);
    // One view: the canvas with the frame and its marks sits in the video's box, and no point of
    // either view is selectable.
    await page.waitForSelector('.sfm-analysis-canvas[data-shown=true][data-mode=marks]');
    assert.equal(
      await page.locator('.sfm-analysis-canvas').evaluate((c) => getComputedStyle(c).pointerEvents),
      'none',
      'clicks reach the video and its controls',
    );
    // Idle, the canvas holds the marks alone: most pixels are transparent, so the video shows.
    const opaque = await page.locator('.sfm-analysis-canvas').evaluate((c) => {
      const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      let count = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 0) count++;
      return count / (data.length / 4);
    });
    assert.ok(opaque < 0.25, 'the canvas draws no frame image when idle: ' + opaque);
    assert.match(
      await page.locator('[data-tile=resolution] .sfm-tile-value').innerText(),
      /320 × 180/,
    );
    const finishedBoxes = await cardBoxes();
    assert.equal(finishedBoxes.frame.width, idleBoxes.frame.width, 'the column keeps its width');
    assert.equal(finishedBoxes.map.width, idleBoxes.map.width, 'the 3D card keeps its width');
    assert.deepEqual(finishedBoxes.stage, idleBoxes.stage, 'the frame box keeps its size');
    assert.equal(finishedBoxes.sameRect, true, 'the overlay canvas covers the video box');
    assert.deepEqual(finishedBoxes.viewport, idleBoxes.viewport, 'the viewport keeps its size');
    assert.deepEqual(
      finishedBoxes.frameTiles,
      idleBoxes.frameTiles,
      'the tile grid keeps its size',
    );
    assert.deepEqual(finishedBoxes.mapTiles, idleBoxes.mapTiles, 'the tile grid keeps its size');
    await page.locator('.lab-switch', { hasText: 'Frame overlay' }).locator('.ant-switch').click();
    assert.equal(await page.locator('.sfm-analysis-canvas').getAttribute('data-shown'), 'true');
    assert.equal(
      await page.locator('.sfm-map .viewer').getAttribute('data-selected-point'),
      null,
      'the 3D view has no selected point',
    );
    const videoStages = JSON.parse(await page.locator('.run-details details pre').textContent());
    assert.ok(videoStages.filter((stage) => stage.operation === 'keyframe').length >= 2);
    assert.ok(
      videoStages.some((stage) => stage.operation === 'pnp' || stage.operation === 'bundle'),
    );
    // The COLMAP sample with the dataset's SIFT features: the mapper alone, held to COLMAP's cameras.
    // The card's checkbox loads the dataset's features instead of running the GPU front end.
    await openInput(page);
    const southBuilding = page.locator('[data-sample="south-building"]');
    await southBuilding.getByRole('checkbox').check();
    await openInput(page);
    await page.getByRole('button', { name: /Load south-building \(12 photos\)/ }).click();
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
    await openInput(page);
    await page.getByRole('button', { name: /Load south-building \(12 photos\)/ }).click();
    await page.waitForFunction(() =>
      /photographs at 1280 × 960/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.locator('.ant-radio-button-wrapper', { hasText: /640 px/ }).click();
    // The same photographs through the GPU front end with the photo profile.
    await openInput(page);
    await southBuilding.getByRole('checkbox').uncheck();
    await openInput(page);
    await page.getByRole('button', { name: /Load south-building \(12 photos\)/ }).click();
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
    await openInput(failedViewer);
    await failedViewer.getByRole('button', { name: /Load Pyramid benchmark/ }).click();
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

// Verifies: docs/multi-video.md step 6 (the sample set card, the upload rows and a set run). The
// catalog and the frames of the two sample videos are served by the test: the first four and the
// last four frames of the local pyramid sample. The run is cancelled after its first keyframe.
test('the lab shows a sample set card, loads its rows and starts a multi-video run', async () => {
  const frame = (n) => readFile(path.join(directory, 'sample/pyramid', `frame-0${n}.jpg`));
  const member = (id, first) => ({
    id,
    kind: 'video',
    dir: `/sample/${id}/`,
    name: `Set member ${id}`,
    frames: 4,
    size: [640, 360],
    thumbnail: 'frame-00.jpg',
    description: { en: id, ko: id },
    video: { name: id, times: [0, 1, 2, 3], duration: 4 },
    first,
  });
  const members = [member('set-a', 0), member('set-b', 4)];
  const catalog = {
    format: 'typecv-lab-samples-v1',
    samples: [
      ...members.map(({ first: _first, ...entry }) => entry),
      {
        id: 'set-ab',
        kind: 'video-set',
        name: 'Two members',
        members: ['set-a', 'set-b'],
        description: { en: 'Two videos.', ko: '영상 두 개.' },
      },
    ],
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const file = fileFor(decodeURIComponent(url.pathname));
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
          '.bin': 'application/octet-stream',
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
      args: [
        '--enable-unsafe-webgpu',
        '--enable-unsafe-swiftshader',
        '--use-angle=swiftshader',
        '--use-vulkan=swiftshader',
        '--enable-features=Vulkan',
      ],
    });
    const page = await browser.newPage(),
      errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.route('**/sample/index.json', (route) =>
      route.fulfill({ contentType: 'application/json', body: JSON.stringify(catalog) }),
    );
    await page.route(/\/sample\/set-[ab]\/frame-\d\d\.jpg$/, async (route) => {
      const [, id, n] = /sample\/(set-[ab])\/frame-0(\d)\.jpg$/.exec(route.request().url());
      const first = members.find((m) => m.id === id).first;
      await route.fulfill({ contentType: 'image/jpeg', body: await frame(first + Number(n)) });
    });
    const url = `http://127.0.0.1:${server.address().port}`;
    await page.goto(url + '/lab/');
    await openInput(page);
    const card = page.locator('[data-kind="video-set"]');
    await card.waitFor();
    assert.equal(await card.getAttribute('data-sample'), 'set-ab');
    assert.match(await card.innerText(), /2 videos/);
    // The member videos keep their own cards; the set card loads both as rows.
    assert.equal(await page.locator('[data-kind="video"]').count(), 2);
    await openInput(page);
    await card.getByRole('button', { name: /Load Two members/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.mv-row').length === 2);
    assert.match(await page.locator('.lab-status').innerText(), /2 videos are ready/);
    // Removing a row leaves one video; loading the set again brings it back.
    await page.getByRole('button', { name: /Remove Set member set-b/ }).click();
    assert.equal(await page.locator('.mv-row').count(), 1);
    await openInput(page);
    await card.getByRole('button', { name: /Load Two members/ }).click();
    await page.waitForFunction(() => document.querySelectorAll('.mv-row').length === 2);
    const started = new Promise((resolve) => page.once('worker', resolve));
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await started;
    await page.waitForFunction(() => document.querySelector('.mv-count'), undefined, {
      timeout: 240000,
    });
    assert.match(await page.locator('.mv-count').first().innerText(), /^\d+ kf$/);
    // Amendment 8: the stage list names the stage that runs, and the status line moves with time.
    const stages = page.locator('[data-testid="mv-stages"] li');
    assert.equal(await stages.count(), 6);
    assert.equal(
      await page.locator('[data-testid="mv-stages"] li[data-state="running"]').count(),
      1,
    );
    const first = await page.locator('.lab-status').innerText();
    assert.match(first, /total \d/);
    await page.waitForFunction(
      (before) => document.querySelector('.lab-status')?.textContent !== before,
      first,
      { timeout: 5000 },
    );
    await page.getByRole('button', { name: /^Cancel$/ }).click();
    await page.waitForFunction(() =>
      /cancelled/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});

// Verifies: a video sample that stores a source video decodes it by byte range at the analysis
// edge, and uses its frame files when the source cannot be read (docs/multi-video.md, Amendment
// 7). The source is a small VP9 clip served with Range; the second sample's source is missing.
test('a video sample reads its source video by byte range, or falls back to its frames', async () => {
  const frame = (n) => readFile(path.join(directory, 'sample/pyramid', `frame-0${n}.jpg`));
  const clip = await readFile(
    path.join(import.meta.dirname, '../tests/fixtures/source-clip-vp9.mp4'),
  );
  const entry = (id, videoId) => ({
    id,
    kind: 'video',
    dir: `/sample/${id}/`,
    name: `Source sample ${id}`,
    frames: 4,
    size: [640, 360],
    thumbnail: 'frame-00.jpg',
    description: { en: id, ko: id },
    video: { name: id, times: [0, 1, 2, 3], duration: 4 },
    source: {
      file: 'source.mp4',
      bytes: clip.length,
      videoId,
      width: 320,
      height: 180,
      codec: 'vp9',
    },
  });
  const catalog = {
    format: 'typecv-lab-samples-v1',
    samples: [
      entry('src-ok', '0123456789abcdef'),
      entry('src-missing', 'fedcba9876543210'),
      {
        id: 'src-set',
        kind: 'video-set',
        name: 'Source set',
        members: ['src-ok', 'src-missing'],
        description: { en: 'Two videos.', ko: '영상 두 개.' },
      },
    ],
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const file = fileFor(decodeURIComponent(url.pathname));
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
          '.bin': 'application/octet-stream',
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
      args: [
        '--enable-unsafe-webgpu',
        '--enable-unsafe-swiftshader',
        '--use-angle=swiftshader',
        '--use-vulkan=swiftshader',
        '--enable-features=Vulkan',
      ],
    });
    const page = await browser.newPage(),
      errors = [],
      ranges = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.route('**/sample/index.json', (route) =>
      route.fulfill({ contentType: 'application/json', body: JSON.stringify(catalog) }),
    );
    await page.route(/\/sample\/src-(ok|missing)\/frame-\d\d\.jpg$/, async (route) => {
      const n = /frame-0(\d)\.jpg$/.exec(route.request().url())[1];
      await route.fulfill({ contentType: 'image/jpeg', body: await frame(Number(n)) });
    });
    // The Worker's answer to a Range request, for the stored clip.
    await page.route('**/sample/src-ok/source.mp4', async (route) => {
      const header = route.request().headers().range;
      const match = /^bytes=(\d+)-(\d*)$/.exec(header ?? '');
      ranges.push(header ?? null);
      const first = match ? Number(match[1]) : 0;
      const last =
        match && match[2] ? Math.min(Number(match[2]), clip.length - 1) : clip.length - 1;
      if (first >= clip.length)
        return route.fulfill({
          status: 416,
          headers: { 'content-range': `bytes */${clip.length}` },
        });
      await route.fulfill({
        status: match ? 206 : 200,
        contentType: 'video/mp4',
        headers: {
          'accept-ranges': 'bytes',
          ...(match ? { 'content-range': `bytes ${first}-${last}/${clip.length}` } : {}),
        },
        body: clip.subarray(first, last + 1),
      });
    });
    await page.route('**/sample/src-missing/source.mp4', (route) => route.fulfill({ status: 404 }));
    const url = `http://127.0.0.1:${server.address().port}`;
    await page.goto(url + '/lab/?fps=1');
    const finished = () =>
      page.waitForFunction(
        () =>
          /completed|Reconstruction failed/.test(
            document.querySelector('.lab-status')?.textContent ?? '',
          ),
        undefined,
        { timeout: 240000 },
      );
    const recordOf = async () => {
      const record = JSON.parse(await page.locator('.run-record details pre').textContent());
      assert.equal(runRecordProblem(record), undefined);
      return record;
    };
    // The source is there and VP9 decodes in this Chromium: the run reads it by byte range.
    await openInput(page);
    await page.getByRole('button', { name: 'Load Source sample src-ok' }).click();
    await page.waitForFunction(() =>
      /\(source video\)/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    const fromSource = await recordOf();
    assert.equal(fromSource.options.videoSource, 'source');
    assert.equal(fromSource.options.videoSampleFps, 1);
    assert.ok(fromSource.options.videoDecode);
    // The 320 × 180 clip is under the 640 analysis edge: the frames keep the source size.
    assert.equal(fromSource.input.width, 320);
    assert.ok(ranges.length > 0 && ranges.every((r) => r === null || r.startsWith('bytes=')));
    // The source does not answer: the 640 pixel frames run, and the record says so.
    await openInput(page);
    await page.getByRole('button', { name: 'Load Source sample src-missing' }).click();
    await page.waitForFunction(() =>
      /\(frame files\)/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    const fromFrames = await recordOf();
    assert.equal(fromFrames.options.videoSource, 'frames');
    assert.equal(fromFrames.options.videoSampleFps, undefined);
    assert.equal(fromFrames.input.width, 640);
    // A video set: the first video is read from its source, the second from its frames, and
    // the record says both.
    await openInput(page);
    await page
      .locator('[data-kind="video-set"]')
      .getByRole('button', { name: /Load Source set/ })
      .click();
    await page.waitForFunction(() => document.querySelectorAll('.mv-row').length === 2);
    const rangesBefore = ranges.length;
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    const set = await recordOf();
    assert.ok(ranges.length > rangesBefore, 'the set read the source video by range');
    assert.equal(set.options.videoSource, 'mixed');
    assert.deepEqual(
      set.input.videos.map((v) => v.videoId),
      ['0123456789abcdef', 'fedcba9876543210'].sort(),
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});

// Verifies: the run record names the sample that ran, and its view count is the count of the
// reconstruction the lab exports for the same run (a bug fix: an owner run on 2026-10-21 recorded
// the default pyramid card for a run of another video sample, and 22 views for an export of 18).
test('the run record and the export of one run name the same sample and count the same views', async () => {
  const frame = (n) => readFile(path.join(directory, 'sample/pyramid', `frame-0${n}.jpg`));
  const entry = (id, name, count) => ({
    id,
    kind: 'video',
    dir: `/sample/${id}/`,
    name: `Card ${id}`,
    frames: count,
    size: [640, 360],
    thumbnail: 'frame-00.jpg',
    description: { en: id, ko: id },
    video: { name, times: [0, 1, 2, 3, 4, 5].slice(0, count), duration: count },
  });
  const catalog = {
    format: 'typecv-lab-samples-v1',
    samples: [
      entry('first-card', 'First card sample', 5),
      entry('second-card', 'Second card sample', 6),
    ],
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const file = fileFor(decodeURIComponent(url.pathname));
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
          '.bin': 'application/octet-stream',
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
      args: [
        '--enable-unsafe-webgpu',
        '--enable-unsafe-swiftshader',
        '--use-angle=swiftshader',
        '--use-vulkan=swiftshader',
        '--enable-features=Vulkan',
      ],
    });
    const page = await browser.newPage(),
      errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.route('**/sample/index.json', (route) =>
      route.fulfill({ contentType: 'application/json', body: JSON.stringify(catalog) }),
    );
    await page.route(/\/sample\/(first|second)-card\/frame-\d\d\.jpg$/, async (route) => {
      const n = /frame-0(\d)\.jpg$/.exec(route.request().url())[1];
      await route.fulfill({ contentType: 'image/jpeg', body: await frame(Number(n)) });
    });
    const url = `http://127.0.0.1:${server.address().port}`;
    const finished = async () => {
      await page.waitForFunction(
        () =>
          /completed|Reconstruction failed/.test(
            document.querySelector('.lab-status')?.textContent ?? '',
          ),
        undefined,
        { timeout: 420000 },
      );
      const status = await page.locator('.lab-status').textContent();
      assert.match(status, /completed/, status);
    };
    // A later run that sends its record and files: the files are that run's, not the run before.
    const posted = [],
      puts = {};
    await page.route('**/api/config/', (route) =>
      route.fulfill({ contentType: 'application/json', body: '{"files":true}' }),
    );
    await page.route('**/api/records/**', async (route) => {
      const request = route.request();
      if (request.method() === 'PUT') puts[request.url().split('/').pop()] = request.postData();
      else posted.push(request.postDataJSON());
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          id: '20260101000000000-deadbeef',
          receivedAt: '2026-01-01T00:00:00Z',
        }),
      });
    });
    await page.goto(url + '/lab/');
    // Select one card, then another, then run.
    await openInput(page);
    await page.getByRole('button', { name: 'Load Card first-card' }).click();
    await page.waitForFunction(() =>
      /sample frames/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await openInput(page);
    await page.getByRole('button', { name: 'Load Card second-card' }).click();
    await page.waitForFunction(() =>
      /sample frames/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    const record = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(record), undefined);
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: 'Export JSON' }).click(),
    ]);
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
    assert.equal(record.input.name, 'Second card sample');
    assert.equal(exported.source, 'Second card sample');
    assert.equal(record.result.registeredViews, exported.views.length);
    assert.equal(exported.runRecord?.result.registeredViews, exported.views.length);
    await openInput(page);
    await page.getByRole('button', { name: 'Load Card second-card' }).click();
    await page.waitForFunction(() =>
      /sample frames/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    await openInput(page);
    await page.getByRole('button', { name: 'Load Card first-card' }).click();
    await page.waitForFunction(() =>
      /5 sample frames/.test(document.querySelector('.lab-status')?.textContent ?? ''),
    );
    // Consent and the files box are set before the run: the record and its files are sent at its end.
    await page.locator('.run-record-consent .ant-checkbox-input').first().check();
    await page.getByRole('checkbox', { name: /Also send the result files/ }).check();
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await finished();
    await page.waitForFunction(() =>
      /Sent: /.test(document.querySelector('.run-record-status')?.textContent ?? ''),
    );
    const sent = posted.at(-1);
    assert.equal(sent.input.name, 'First card sample');
    assert.equal(sent.input.frames, 5);
    assert.ok(
      puts['typecv-reconstruction.json'],
      JSON.stringify({
        files: Object.keys(puts),
        status: await page.locator('.run-record-status').textContent(),
        record: sent.files,
      }),
    );
    const sentExport = JSON.parse(puts['typecv-reconstruction.json']);
    assert.equal(sentExport.source, 'First card sample');
    assert.equal(sentExport.analysisFrames.length, 5);
    assert.equal(sentExport.views.length, sent.result.registeredViews);
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});

test('the 3D card widens and goes full screen, and the mesh and dense switches say why they are off', async () => {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const file = fileFor(decodeURIComponent(url.pathname));
    if (!file.startsWith(directory + path.sep)) return res.writeHead(403).end();
    try {
      res.setHeader(
        'Content-Type',
        {
          '.js': 'text/javascript',
          '.css': 'text/css',
          '.html': 'text/html',
          '.json': 'application/json',
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
    });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } }),
      errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}/lab/`);
    const map = page.locator('section.sfm-map');
    // The 3D card takes the wider column beside the analysis column, and the whole row when wide.
    const half = (await map.boundingBox()).width;
    const wideSwitch = page
      .locator('.sfm-map-options label', { hasText: 'Wide 3D' })
      .getByRole('switch');
    await page.getByRole('button', { name: /Display/ }).click();
    await wideSwitch.click();
    assert.equal(await page.locator('.sfm-live-grid.map-wide').count(), 1);
    assert.ok((await map.boundingBox()).width > half * 1.3, 'the 3D card takes the full width');
    await wideSwitch.click();
    assert.equal(await page.locator('.sfm-live-grid.map-wide').count(), 0);
    await page.keyboard.press('Escape');

    // The View menu holds the camera, fit and preset views; off until a map exists.
    await page.getByRole('button', { name: /^View/ }).click();
    for (const name of [
      'Through the selected camera',
      'Whole map',
      'From above',
      'From the front',
      'From the side',
    ])
      assert.equal(await page.getByRole('menuitem', { name }).isDisabled(), true, name);
    await page.keyboard.press('Escape');

    // The mesh switch of the run settings waits for Dense MVS.
    const mesh = page.locator('label.lab-switch', { hasText: /^Mesh$/ }).first();
    assert.equal(await mesh.locator('button[role=switch]').isDisabled(), true);
    await page
      .locator('label.lab-switch', { hasText: 'Dense MVS' })
      .locator('button[role=switch]')
      .click();
    assert.equal(await mesh.locator('button[role=switch]').isDisabled(), false);

    // The display menu's dense switch is off with a reason.
    await page.getByRole('button', { name: /Display/ }).click();
    await page.locator('.sfm-map-options label', { hasText: 'Dense' }).hover();
    await page.getByText(/No dense cloud\. Turn on Dense MVS/).waitFor();

    // Full screen holds the 3D card, and the menu opens inside it.
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Full screen' }).click();
    await page.waitForFunction(() => document.fullscreenElement?.classList.contains('sfm-map'));
    await page.getByRole('button', { name: /Display/ }).click();
    assert.equal(
      await page.evaluate(
        () => !!document.fullscreenElement.querySelector('.ant-popover:not(.ant-popover-hidden)'),
      ),
      true,
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    server.close();
  }
});

// The Place menu (owner request 2026-10-21): a result whose ground is tilted 16.7 degrees about x
// is levelled by "Level automatically", "Reset" undoes it, and the handles take a mode.
test('the Place menu levels a tilted ground, resets, and shows the handles', async () => {
  // The mapper's frame: x right, y down, z forward. The ground is y = 2 + 0.3 z under three
  // cameras at y = 0; on screen (y up) it rises 0.3 per unit toward the viewer: atan 0.3.
  const landmarks = [];
  for (let i = 0; i < 40; i++)
    for (let k = 0; k < 40; k++) {
      const x = -4 + (8 * i) / 39,
        z = 2 + (8 * k) / 39;
      landmarks.push({
        trackId: landmarks.length,
        xyz: Float64Array.from([x, 2 + 0.3 * z, z]),
        color: [0.4, 0.6, 0.3],
        observations: [],
      });
    }
  const view = (id, x) => ({
    id,
    time: id,
    sequence: id,
    width: 640,
    height: 360,
    pose: { R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: Float64Array.from([-x, 0, 0]) },
    camera: { fx: 512, fy: 512, cx: 320, cy: 180 },
  });
  const json = reconstructionJson({
    source: { name: 'tilted', width: 640, height: 360, duration: 3 },
    assumedFocalRatio: 0.8,
    diagnostics: { meanReprojectionError: 0.4 },
    unregistered: [],
    analysisFrames: [0, 1, 2].map((id) => ({ id, time: id, width: 640, height: 360 })),
    scene: { views: [view(0, -2), view(1, 0), view(2, 2)], landmarks },
    sourceCamera: (v) => v.camera,
  });
  const entry = {
    id: 'tilted',
    kind: 'result',
    name: 'Tilted ground',
    dir: '/sample/results/tilted/',
    files: { json: 'typecv-reconstruction.json' },
    remote: true,
    run: { date: '2026-10-21', device: 'swiftshader', commit: 'abcdef1' },
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/sample/results/tilted/typecv-reconstruction.json') {
      res.setHeader('Content-Type', 'application/json');
      return res.end(json);
    }
    const file = fileFor(decodeURIComponent(url.pathname));
    if (!file.startsWith(directory + path.sep)) return res.writeHead(403).end();
    try {
      const body = await readFile(file);
      if (url.pathname === '/sample/index.json') {
        const catalog = JSON.parse(String(body));
        catalog.samples.unshift(entry);
        res.setHeader('Content-Type', 'application/json');
        return res.end(JSON.stringify(catalog));
      }
      res.setHeader(
        'Content-Type',
        {
          '.js': 'text/javascript',
          '.css': 'text/css',
          '.html': 'text/html',
          '.json': 'application/json',
        }[path.extname(file)] ?? 'application/octet-stream',
      );
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
    });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } }),
      errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}/lab/`);
    const viewer = page.locator('.sfm-map .viewer');
    const placement = async () =>
      JSON.parse((await viewer.getAttribute('data-placement')) ?? 'null');
    await openInput(page);
    await page.getByRole('button', { name: 'View Tilted ground' }).click();
    await page.waitForFunction(
      () => Number(document.querySelector('.sfm-map .viewer')?.dataset.points) === 1600,
    );
    await page.getByRole('button', { name: /^Place/ }).click();
    await page.getByRole('button', { name: 'Level automatically' }).click();
    await page.waitForFunction(
      () => !!document.querySelector('.sfm-map .viewer')?.dataset.placement,
    );
    const [, turn, scale] = await placement();
    const tilt = (Math.atan(0.3) * 180) / Math.PI;
    assert.ok(Math.abs(Math.abs(turn[0]) - tilt) < 0.5, 'turn x ' + turn[0] + ' against ' + tilt);
    assert.ok(Math.abs(turn[1]) < 0.5 && Math.abs(turn[2]) < 0.5, 'turn ' + turn);
    assert.equal(scale, 1);
    // The numbers show the placement.
    assert.equal(
      Math.abs(Number(await page.getByLabel('Turn ° x').inputValue())).toFixed(0),
      tilt.toFixed(0),
    );
    await page.getByRole('button', { name: 'Reset' }).click();
    await page.waitForFunction(() =>
      document
        .querySelector('.sfm-map .viewer')
        ?.dataset.placement?.startsWith('[[0,0,0],[0,0,0],1]'),
    );
    await page.locator('.placement-menu .ant-radio-button-wrapper', { hasText: 'Turn' }).click();
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.transform === 'rotate',
    );
    // Picking the ground shows the banner with its count; cancel ends it.
    await page.getByRole('button', { name: 'Pick three ground points' }).click();
    await page.getByText('Double-click a point on the ground (0 / 3)').waitFor();
    await page.locator('.sfm-pick-banner').getByRole('button', { name: 'Cancel' }).click();
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.picking === '-1',
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    server.close();
  }
});
