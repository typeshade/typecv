import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { inBrowser } from '../tests/typeshade/browser.mjs';

test('real single and multi-video workers report preparation before opening the device', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'typecv-preparation-'));
  try {
    const workerFile = join(directory, 'worker.js');
    const runtimeFile = join(directory, 'gpu.js');
    const vocabularyFile = join(directory, 'vocabulary.bin');
    await build({
      entryPoints: [new URL('./src/workers/sfm.worker.ts', import.meta.url).pathname],
      outfile: workerFile,
      bundle: true,
      format: 'esm',
      platform: 'browser',
    });
    await writeFile(
      runtimeFile,
      `
      self.postMessage({type:'testRuntimeLoaded'});
      export class ShadeDevice {
        static async open(options) {
          self.postMessage({type:'testDeviceOpening'});
          return {...options, backend:'webgpu', fp64Flavor:'float', statistics:{},
            kernelStatistics:{rows:[],dropped:0}, async dispose(){}};
        }
      }
      export class IncrementalSfmSession {}
      export const parseVocabulary = () => new Float32Array(32*128);
    `,
    );
    await writeFile(vocabularyFile, new Uint8Array(4));
    const { result, errors } = await inBrowser(
      async () => {
        const run = (multi) =>
          new Promise((resolve, reject) => {
            const worker = new Worker('/worker.js', { type: 'module' });
            const messages = [];
            worker.onerror = (error) => {
              worker.terminate();
              reject(new Error(error.message));
            };
            worker.onmessage = ({ data }) => {
              messages.push(data);
              if (data.type === 'error') {
                worker.terminate();
                reject(new Error(data.message));
              }
              if (data.type === (multi ? 'multiReady' : 'ready')) {
                worker.terminate();
                resolve(messages);
              }
            };
            worker.postMessage(
              multi
                ? { id: 42, type: 'multiStart', query: {}, analysisEdge: 640, firstOctave: 0 }
                : { id: 42, type: 'start', focalRatio: 0.8 },
            );
          });
        return { single: await run(false), multi: await run(true) };
      },
      undefined,
      {
        serve: {
          '/worker.js': workerFile,
          '/runtime/gpu.js': runtimeFile,
          '/runtime/internal/gpu.js': runtimeFile,
          '/runtime/vocabulary/rootsift-32-v1.bin': vocabularyFile,
        },
      },
    );
    assert.deepEqual(errors, []);
    for (const [mode, messages] of Object.entries(result)) {
      const phases = messages.filter((message) => message.type === 'preparing');
      assert.deepEqual(
        phases.map((message) => message.phase),
        mode === 'multi'
          ? ['runtime', 'vocabulary', 'adapter', 'device', 'session']
          : ['runtime', 'adapter', 'device', 'session'],
      );
      assert.ok(phases.every((message) => message.id === 42));
      const index = (type, phase) =>
        messages.findIndex(
          (message) => message.type === type && (!phase || message.phase === phase),
        );
      assert.ok(index('preparing', 'runtime') < index('testRuntimeLoaded'));
      assert.ok(index('preparing', 'device') < index('testDeviceOpening'));
      assert.ok(index('testDeviceOpening') < index('preparing', 'session'));
      assert.equal(messages.at(-1).type, mode === 'multi' ? 'multiReady' : 'ready');
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

// The real Lab, viewer and input loader; only the numerical worker is held at preparation.
const harness = `
import React from 'react';
import {createRoot} from 'react-dom/client';
import SfmLab from './src/components/SfmLab.tsx';
window.Worker = class {
  requests = [];
  constructor() { window.testWorker = this; }
  emit(message) { this.onmessage?.({data:{id:this.id,...message}}); }
  postMessage(message) {
    this.requests.push(message); this.id = message.id;
    if (message.type === 'start' || message.type === 'multiStart')
      queueMicrotask(() => this.emit({type:'preparing',phase:'runtime'}));
    if(window.autoMulti) queueMicrotask(() => {
      if(message.type === 'multiVideo') this.emit({type:'videoStarted'});
      if(message.type === 'multiFrame') this.emit({type:'frameComplete',keyframe:true,features:24,elapsedMs:1});
      if(message.type === 'multiEndVideo') this.emit({type:'videoComplete',videoId:message.videoId,keyframes:3,keptFrameIds:[0,1,2]});
      if(message.type === 'multiFinish') window.multiFinishHeld = true;
    });
  }
  terminate() { this.terminated = true; }
};
createRoot(document.getElementById('app')).render(<SfmLab locale={location.pathname.startsWith('/ko') ? 'ko' : 'en'} />);
`;

test('the Lab shows actual preparation, elapsed time, cancellation, errors and readiness', async () => {
  const bundle = await build({
    stdin: { contents: harness, resolveDir: import.meta.dirname, loader: 'tsx' },
    bundle: true,
    write: false,
    format: 'esm',
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"production"' },
  });
  const css = await readFile(new URL('./src/styles/global.css', import.meta.url));
  const jpeg = await readFile(
    new URL('./public/sample/south-building/frame-00.jpg', import.meta.url),
  );
  const otherJpeg = await readFile(
    new URL('./public/sample/south-building/frame-09.jpg', import.meta.url),
  );
  const samples = ['a', 'b'].map((id) => ({
    id,
    kind: 'video',
    name: `Prepared ${id}`,
    dir: `/sample/${id}/`,
    frames: 3,
    thumbnail: 'frame-00.jpg',
    size: [320, 180],
    description: { en: 'Test', ko: '테스트' },
    video: { name: `${id}.mp4`, times: [0, 1, 2], duration: 2 },
  }));
  const server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (path === '/app.js')
      return res
        .writeHead(200, { 'Content-Type': 'text/javascript' })
        .end(bundle.outputFiles[0].text);
    if (path === '/style.css') return res.writeHead(200, { 'Content-Type': 'text/css' }).end(css);
    if (path.endsWith('.jpg'))
      return res
        .writeHead(200, { 'Content-Type': 'image/jpeg' })
        .end(path.includes('/b/') ? otherJpeg : jpeg);
    if (path === '/sample/index.json')
      return res.writeHead(200, { 'Content-Type': 'application/json' }).end(
        JSON.stringify({
          format: 'typecv-lab-samples-v1',
          samples: [
            ...samples,
            {
              id: 'clips',
              kind: 'video-set',
              name: 'Prepared set',
              members: ['a', 'b'],
              description: { en: 'Test', ko: '테스트' },
            },
          ],
        }),
      );
    if (path === '/version.json')
      return res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"commit":"test"}');
    if (path.startsWith('/api/')) return res.writeHead(404).end();
    res
      .writeHead(200, { 'Content-Type': 'text/html' })
      .end(
        '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/style.css"><div id="app"></div><script type="module" src="/app.js"></script>',
      );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    });
    for (const multi of [false, true]) {
      const ko = multi;
      const page = await browser.newPage({
        viewport: { width: 1280, height: 900 },
        reducedMotion: multi ? 'reduce' : 'no-preference',
      });
      const errors = [];
      page.on('pageerror', (error) => errors.push(String(error)));
      await page.goto(`http://127.0.0.1:${server.address().port}/${ko ? 'ko/' : ''}`);
      await page.locator('.lab-topbar .ant-btn').click();
      await page
        .getByRole('button', {
          name: multi ? 'Load Prepared set' : 'Load Prepared a',
          exact: true,
        })
        .click();
      const run = page.getByRole('button', { name: ko ? '실행' : 'Run', exact: true });
      const stop = page.getByRole('button', { name: ko ? '정지' : 'Stop', exact: true });
      const preparation = page.getByTestId('gpu-preparation');
      await run.click();
      await page.waitForFunction(
        () => document.querySelector('[data-testid=gpu-preparation]')?.dataset.phase === 'runtime',
      );
      assert.equal(await preparation.locator('li').count(), multi ? 7 : 6);
      assert.equal(await preparation.getByRole('progressbar').getAttribute('aria-valuenow'), null);
      await page.evaluate(() => window.testWorker.emit({ type: 'preparing', phase: 'device' }));
      await page.waitForFunction(
        () => document.querySelector('[data-testid=gpu-preparation]')?.dataset.phase === 'device',
      );
      assert.equal(
        await preparation.locator('[aria-current=step]').innerText(),
        ko ? 'GPU 연결' : 'Connecting GPU',
      );
      await page.waitForFunction(
        () =>
          Number(document.querySelector('.lab-preparation-elapsed')?.dataset.elapsedSeconds) >= 1,
      );
      assert.equal(await page.locator('.lab-run-status > .ant-progress').count(), 0);
      if (multi) {
        await page.setViewportSize({ width: 390, height: 844 });
        await page.evaluate(() => {
          document.documentElement.dataset.theme = 'dark';
        });
        assert.equal(
          await preparation
            .locator('.anticon-spin svg')
            .evaluate((el) => getComputedStyle(el).animationName),
          'none',
        );
      }
      assert.equal(await preparation.evaluate((el) => el.scrollWidth <= el.clientWidth), true);
      if (process.env.TYPECV_PREPARATION_SCREENSHOT_DIR)
        await preparation.screenshot({
          path: join(
            process.env.TYPECV_PREPARATION_SCREENSHOT_DIR,
            multi ? 'preparation-mobile-dark-ko.png' : 'preparation-desktop-en.png',
          ),
        });
      await page.evaluate(() =>
        window.testWorker.emit({ type: 'error', message: 'Preparation fixture failure' }),
      );
      await preparation.waitFor({ state: 'hidden' });
      assert.match(await page.locator('.lab-status').innerText(), /Preparation fixture failure/);
      await run.click();
      await preparation.waitFor();
      await page.evaluate(() => {
        window.previousWorker = window.testWorker;
      });
      await stop.click();
      await preparation.waitFor({ state: 'hidden' });
      assert.equal(await page.evaluate(() => window.previousWorker.terminated), true);
      await run.click();
      await page.waitForFunction(
        () => document.querySelector('[data-testid=gpu-preparation]')?.dataset.phase === 'runtime',
      );
      await page.evaluate(() =>
        window.previousWorker.emit({ type: 'preparing', phase: 'session' }),
      );
      assert.equal(
        await preparation.getAttribute('data-phase'),
        'runtime',
        'a cancelled run cannot advance the next run',
      );
      await page.evaluate((multi) => {
        window.autoMulti = multi;
        window.testWorker.emit({ type: multi ? 'multiReady' : 'ready' });
      }, multi);
      await preparation.waitFor({ state: 'hidden' });
      if (multi) {
        await page.waitForFunction(() => window.multiFinishHeld);
        const panel = page.getByTestId('multi-video-activity');
        assert.equal(await panel.getAttribute('data-mode'), 'full');
        assert.equal(await panel.locator('.mv-activity-node').count(), 2);
        await page.evaluate(() => {
          const ids = window.testWorker.requests
            .filter((message) => message.type === 'multiVideo')
            .map((message) => message.video.videoId);
          window.activityIds = ids;
          const links = [
            {
              first: ids[0],
              second: ids[1],
              candidates: 4,
              matched: 0,
              matches: 0,
              verified: 0,
              accepted: 0,
              inliers: 0,
              active: false,
            },
          ];
          window.testWorker.emit({
            type: 'graphActivity',
            stage: 'candidates',
            links,
            components: [],
            limited: false,
            activeVideos: [],
          });
        });
        await panel
          .locator('.mv-activity-edge[data-state=candidate]')
          .waitFor({ state: 'attached' });
        await page.evaluate(() =>
          window.testWorker.emit({
            type: 'multiStage',
            stage: 'candidates',
            done: 4,
            total: 4,
            final: true,
          }),
        );
        assert.equal(
          await panel.getAttribute('data-stage'),
          'candidates',
          'a completed retrieval stage does not announce reconstruction early',
        );
        await page.evaluate(() =>
          window.testWorker.emit({
            type: 'graphActivity',
            stage: 'matching',
            links: [
              {
                first: window.activityIds[0],
                second: window.activityIds[1],
                candidates: 4,
                matched: 1,
                matches: 0,
                verified: 0,
                accepted: 0,
                inliers: 0,
                active: false,
              },
            ],
            components: [],
            limited: false,
            activeVideos: [],
          }),
        );
        await page.waitForFunction(
          () =>
            document.querySelector('[data-testid=multi-video-activity]')?.dataset.stage ===
            'matching',
        );
        assert.equal(
          await panel.locator('.mv-activity-edge').getAttribute('data-state'),
          'candidate',
          'processing a pair without correspondences does not display a matched connection',
        );
        await page.evaluate(() =>
          window.testWorker.emit({
            type: 'graphActivity',
            stage: 'matching',
            links: [
              {
                first: window.activityIds[0],
                second: window.activityIds[1],
                candidates: 4,
                matched: 2,
                matches: 80,
                verified: 0,
                accepted: 0,
                inliers: 0,
                active: true,
              },
            ],
            components: [],
            limited: false,
            activeVideos: window.activityIds,
          }),
        );
        await panel
          .locator('.mv-activity-edge[data-state=matched][data-active=true]')
          .waitFor({ state: 'attached' });
        await page.evaluate(() =>
          window.testWorker.emit({ type: 'multiStage', stage: 'matching', done: 2, total: 4 }),
        );
        assert.equal(
          await panel
            .locator('.mv-activity-edge')
            .evaluate((el) => getComputedStyle(el).animationName),
          'none',
          'reduced motion also disables graph flow',
        );
        await page.evaluate(() =>
          window.testWorker.emit({
            type: 'graphActivity',
            stage: 'verification',
            links: [
              {
                first: window.activityIds[0],
                second: window.activityIds[1],
                candidates: 4,
                matched: 4,
                matches: 160,
                verified: 4,
                accepted: 2,
                inliers: 70,
                active: false,
              },
            ],
            components: [],
            limited: false,
            activeVideos: [],
          }),
        );
        await panel
          .locator('.mv-activity-edge[data-state=verified]')
          .waitFor({ state: 'attached' });
        await page.evaluate(() =>
          window.testWorker.emit({ type: 'multiStage', stage: 'verification', done: 4, total: 4 }),
        );
        await panel.getByRole('button', { name: /Prepared b/ }).click();
        assert.match(await panel.locator('.mv-activity-detail').innerText(), /Prepared b/);
        assert.equal(await panel.locator('.mv-activity-thumbnails img').count(), 3);
        assert.equal(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth), true);
        if (process.env.TYPECV_PREPARATION_SCREENSHOT_DIR)
          await panel.screenshot({
            path: join(
              process.env.TYPECV_PREPARATION_SCREENSHOT_DIR,
              'multi-video-activity-mobile.png',
            ),
          });
        await page.setViewportSize({ width: 1280, height: 900 });
        if (process.env.TYPECV_PREPARATION_SCREENSHOT_DIR)
          await panel.screenshot({
            path: join(
              process.env.TYPECV_PREPARATION_SCREENSHOT_DIR,
              'multi-video-activity-desktop.png',
            ),
          });
        await page.evaluate(() => {
          window.testWorker.emit({
            type: 'multiStage',
            stage: 'reconstruction',
            done: 2,
            total: 6,
          });
          window.savedViewerCanvas = document.querySelector('.sfm-map .viewer canvas');
          window.testWorker.emit({
            type: 'graphActivity',
            stage: 'components',
            links: [
              {
                first: window.activityIds[0],
                second: window.activityIds[1],
                candidates: 4,
                matched: 4,
                matches: 160,
                verified: 4,
                accepted: 2,
                inliers: 70,
                active: false,
              },
            ],
            components: [
              { index: 0, videos: window.activityIds, keyframes: 6, reconstructed: true },
            ],
            limited: false,
            activeVideos: [],
          });
          window.testWorker.emit({
            type: 'model',
            partial: true,
            component: 0,
            registered: 2,
            reconstruction: {
              views: window.activityIds.map((videoId, id) => ({
                id,
                videoId,
                sequence: 0,
                width: 320,
                height: 180,
                pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-id, 0, -2] },
                camera: { fx: 200, fy: 200, cx: 160, cy: 90 },
              })),
              packed: {
                count: 1,
                trackIds: new Uint32Array([1]),
                xyz: new Float64Array([0, 0, 4]),
                rgb: new Uint8Array([160, 100, 60]),
                observations: 2,
              },
            },
          });
        });
        await page.waitForFunction(
          () =>
            document.querySelector('[data-testid=multi-video-activity]')?.dataset.mode ===
            'compact',
        );
        await page.locator('.sfm-map .viewer').scrollIntoViewIfNeeded();
        await page.waitForFunction(
          () => document.querySelector('.sfm-map .viewer')?.dataset.views === '2',
        );
        assert.equal(
          await page.evaluate(
            () => window.savedViewerCanvas === document.querySelector('.sfm-map .viewer canvas'),
          ),
          true,
          'the renderer persists when the graph contracts',
        );
        await panel.getByRole('button', { name: '영상 연결 그래프 보기' }).click();
        await panel
          .locator('.mv-activity-edge[data-state=verified]')
          .waitFor({ state: 'attached' });
        await panel.getByRole('button', { name: '3D 장면으로 돌아가기' }).click();
        await page.waitForFunction(
          () =>
            document.querySelector('[data-testid=multi-video-activity]')?.dataset.mode ===
            'compact',
        );
      }
      await stop.click();
      await run.waitFor();
      assert.equal(await page.getByTestId('multi-video-activity').count(), 0);
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
