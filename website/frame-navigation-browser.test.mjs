import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, readdir } from 'node:fs/promises';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { runRecordProblem } from './src/lib/run-record.ts';

// The real Lab, decoder and WebGL viewer; only SfM results are deterministic worker messages.
const harness = `
import React from 'react';
import {createRoot} from 'react-dom/client';
import * as THREE from 'three';
import SfmLab from './src/components/SfmLab.tsx';
THREE.Scene.prototype.onBeforeRender = function(renderer, scene, camera) {
  if (new URLSearchParams(location.search).has('viewerFailure')) throw new Error('Injected initial viewer render failure');
  let root;
  scene.traverse(object => { if (object.isPoints && object.material.size === 3) root ??= object.parent; });
  if (!root) return;
  window.project = id => {
    const view = window.testViews.find(view => view.id === id);
    const p = root.localToWorld(new THREE.Vector3(...view.pose.t.map(v => -v))).project(camera);
    const rect = renderer.domElement.getBoundingClientRect();
    return {x: rect.left + (p.x+1)*rect.width/2, y: rect.top + (1-p.y)*rect.height/2};
  };
  window.cameraDistance = id => {
    const view = window.testViews.find(view => view.id === id);
    return camera.position.distanceTo(root.localToWorld(new THREE.Vector3(...view.pose.t.map(v => -v))));
  };
};
window.Worker = class {
  frames = [];
  videos = [];
  constructor() { window.testWorker = this; }
  send(message) { this.onmessage?.({data: {id: this.id, ...message}}); }
  postMessage(m) {
    this.id = m.id;
    queueMicrotask(() => {
      if (m.type === 'start') { this.frames = []; this.videos = []; this.send({type:'ready'}); }
      if (m.type === 'multiStart') { this.frames = []; this.videos = []; this.dense = m.dense; this.send({type:'multiReady'}); }
      if (m.type === 'multiVideo') { this.videos.push(m.video.videoId); this.send({type:'videoStarted'}); }
      if (m.type === 'multiEndVideo') this.send({type:'videoComplete',videoId:m.videoId,keyframes:3});
      if (m.type === 'frame' || m.type === 'multiFrame') {
        if ((m.type === 'multiFrame' && window.holdMulti) || (m.type === 'frame' && window.holdSingle)) {
          this.heldFrame = m;
          if (m.type === 'multiFrame') window.multiHeld = true; else window.singleHeld = true;
          return;
        }
        this.frames.push({...m.frame, videoId:m.videoId});
        this.send({type:'frameComplete',frameId:m.frame.id,elapsedMs:1,keyframe:true,cameraRegistered:m.frame.id!==1,verifiedInliers:20});
      }
      if (m.type === 'finish' || m.type === 'multiFinish') {
        const multi = m.type === 'multiFinish';
        const views = this.frames.filter(f => multi ? f.id < 3 : f.id !== 1).map(f => {
          const v = multi ? this.videos.indexOf(f.videoId) : 0;
          return {id: multi ? 20+v*20+f.id : f.id, time:f.time, sequence:f.id, videoId:f.videoId,
            width:f.width,height:f.height,
            image:multi ? (v===1 && f.id===2 ? '/slow.jpg' : f.image) : undefined,
            pose:{R:[1,0,0,0,1,0,0,0,1],t:[-v*3-f.id*0.3, -v*0.5, -2-f.id*0.1]},
            camera:{fx:200,fy:200,cx:f.width/2,cy:f.height/2}};
        });
        window.testViews = views;
        const model = views => ({views,landmarks:[0,1,2].map(i => ({trackId:i,xyz:[i*2,1,4],color:[1,0.5,0.2],observations:views.map(v => ({viewId:v.id,featureId:i,pixel:[160,90]}))}))});
        if (!multi) this.send({type:'result', reconstruction:model(views), diagnostics:{}, unregistered:[], stages:[{operation:'runTotal',elapsedMs:1}], featureCounts:[20],frameMs:[1],matchedPairs:1,skippedFrames:0,keyframes:views.length,kernels:[],chunks:{},options:{},summary:[]});
        else {
          const components = [0,1].map(index => ({index,reconstructed:true,keyframes:index?3:6,videos:index?this.videos.slice(2):this.videos.slice(0,2)}));
          this.send({type:'graph',canonical:{},components,matrix:{order:this.videos,cells:this.videos.map(() => [0,0,0])},stats:{verifiedPairs:2}});
          this.send({type:'model',component:0,reconstruction:model(views.filter(v => v.id < 60))});
          this.send({type:'model',component:1,reconstruction:model(views.filter(v => v.id >= 60))});
          this.send({type:'result',stages:{},totalMs:1,keyframeCount:9,
            gpuKernels:[{name:'sparseKernel',groups:1,dispatches:1,timedGroups:1,gpuNs:1000000}],
            gpuDropped:1,
            summary:{videos:this.videos.map((videoId,i) => ({videoId,durationSeconds:8,focalRatio:0.8,keyframes:3,registeredViews:3,component:i<2?0:1,tags:['component']})),components:[],notConnected:[]}});
          if (this.dense) setTimeout(() => {
            const row = count => ({name:'tsdfIntegrate',groups:count,dispatches:count,timedGroups:count,gpuNs:count*2000000});
            for (const component of [0,1]) this.send({type:'denseResult',component,
              options:{maxSources:6,bestSources:3,windowRadius:5,windowStep:2,photometricIterations:8,geometricIterations:4,minConsistentViews:3,fusion:'tsdf'},
              views:[],points:0,timing:{totalMs:1,stageMs:{photometric:0,geometric:0,fusion:1}},
              gpuKernels:[row(component+1)]});
            this.send({type:'denseDone',gpuKernels:[row(2)],gpuDropped:2});
          }, 0);
        }
      }
    });
  }
  terminate() { this.terminated = true; }
};
createRoot(document.getElementById('app')).render(<SfmLab locale={location.pathname.startsWith('/ko') ? 'ko' : 'en'} />);
`;

const samples = ['a', 'b', 'c'].map((id) => ({
  id,
  kind: 'video',
  dir: `/sample/${id}/`,
  name: `Clip ${id.toUpperCase()}`,
  frames: 3,
  thumbnail: 'frame-00.jpg',
  size: [320, 180],
  description: { en: 'Test video', ko: '테스트 영상' },
  video: { name: `${id}.mp4`, times: [0, 1, 2], duration: 8 },
  source: {
    file: 'source.mp4',
    videoId: id.repeat(16),
    bytes: 1,
    width: 320,
    height: 180,
    codec: 'vp9',
  },
}));

async function selectTimeline(page, value) {
  const slider = page.getByRole('slider', { name: 'Frame timeline' });
  if (await slider.count()) {
    await slider.evaluate((input, value) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }, String(value));
  } else {
    await page.locator('video').evaluate((video, time) => {
      video.currentTime = time;
    }, value);
  }
  await page.locator('.sfm-map .viewer').scrollIntoViewIfNeeded();
}
async function stepFrame(page, key = 'ArrowRight') {
  await page.locator('.sfm-stream').focus();
  await page.keyboard.press(key);
}
async function sparseOnly(page) {
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Run settings' });
  await dialog.locator('label.lab-switch', { hasText: 'Dense MVS' }).getByRole('switch').click();
  await dialog.getByRole('button', { name: 'Close', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
}
async function expectSelection(page, id) {
  // The viewer pauses offscreen; bring it into view before checking rendered selection state.
  await page.locator('.sfm-map .viewer').scrollIntoViewIfNeeded();
  try {
    await page.waitForFunction((id) => {
      const viewer = document.querySelector('.sfm-map .viewer');
      const frame = document.querySelector('.sfm-analysis-canvas');
      return (
        viewer?.dataset.selectedView === String(id) &&
        viewer.dataset.photo === String(id) &&
        frame?.dataset.frame === String(id)
      );
    }, id);
  } catch (error) {
    const state = await page.evaluate(() => ({
      viewer: { ...document.querySelector('.sfm-map .viewer')?.dataset },
      frame: { ...document.querySelector('.sfm-analysis-canvas')?.dataset },
      status: document.querySelector('.lab-status')?.textContent,
      alerts: [...document.querySelectorAll('[role=alert]')].map((el) => el.textContent),
      selection: document.querySelector('.sfm-frame-selection')?.textContent,
      range: document.querySelectorAll('.lab-video-range').length,
      viewerRect: document.querySelector('.sfm-map .viewer')?.getBoundingClientRect().toJSON(),
      scrollY,
    }));
    throw new Error(`Frame ${id} selection failed: ${JSON.stringify(state)}`, { cause: error });
  }
}

test('completed videos keep native playback, frame selection, images and camera views connected across sources', async () => {
  const bundle = await build({
    stdin: { contents: harness, resolveDir: import.meta.dirname, loader: 'tsx' },
    bundle: true,
    write: false,
    format: 'esm',
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"production"' },
  });
  // Playwright's Chromium shell has no H.264 decoder. Use the existing VP9 fixture so
  // source and upload runs exercise native playback rather than prepared-frame fallback.
  const clip = await readFile(new URL('../tests/fixtures/source-clip-vp9.mp4', import.meta.url));
  const jpeg = await readFile(
    new URL('./public/sample/south-building/frame-00.jpg', import.meta.url),
  );
  const productionCss = await readFile(new URL('./src/styles/global.css', import.meta.url), 'utf8');
  const pendingImages = [];
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/app.js')
      return res
        .writeHead(200, { 'Content-Type': 'text/javascript' })
        .end(bundle.outputFiles[0].text);
    if (url.pathname === '/sample/index.json')
      return res.writeHead(200, { 'Content-Type': 'application/json' }).end(
        JSON.stringify({
          format: 'typecv-lab-samples-v1',
          samples: [
            ...samples,
            {
              ...samples[0],
              id: 'prepared',
              name: 'Prepared clip',
              dir: '/sample/prepared/',
              source: undefined,
            },
            {
              id: 'clips',
              kind: 'video-set',
              name: 'Three clips',
              members: ['a', 'b', 'c'],
              description: { en: 'Test set', ko: '테스트' },
            },
          ],
        }),
      );
    if (url.pathname === '/slow.jpg') {
      pendingImages.push(() => res.writeHead(200, { 'Content-Type': 'image/jpeg' }).end(jpeg));
      return;
    }
    if (url.pathname.endsWith('.jpg'))
      return res.writeHead(200, { 'Content-Type': 'image/jpeg' }).end(jpeg);
    if (url.pathname.endsWith('.mp4')) {
      const match = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
      const start = match ? Number(match[1]) : 0;
      const end = match?.[2] ? Math.min(Number(match[2]), clip.length - 1) : clip.length - 1;
      res.writeHead(match ? 206 : 200, {
        'Content-Type': 'video/mp4',
        'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1,
        ...(match ? { 'Content-Range': `bytes ${start}-${end}/${clip.length}` } : {}),
      });
      return res.end(clip.subarray(start, end + 1));
    }
    if (url.pathname === '/version.json')
      return res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"commit":"82c1dd3"}');
    if (url.pathname.startsWith('/api/')) return res.writeHead(404).end();
    res
      .writeHead(200, { 'Content-Type': 'text/html' })
      .end(
        `<style data-harness>${productionCss}</style><div id="app" class="lab-main"></div><script type="module" src="/app.js"></script>`,
      );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    });
    const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    const address = `http://127.0.0.1:${server.address().port}/?fps=1&decode=main`;
    for (const input of ['url', 'upload', 'prepared']) {
      await page.goto(address);
      await page.waitForFunction(
        () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
      );
      await sparseOnly(page);
      await page.locator('.lab-topbar .ant-btn').click();
      if (input === 'prepared')
        await page.getByRole('button', { name: 'Load Prepared clip', exact: true }).click();
      else if (input === 'url')
        await page.getByRole('button', { name: 'Load Clip A', exact: true }).click();
      else
        await page
          .locator('input[type=file]')
          .first()
          .setInputFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: clip });
      await page.getByRole('button', { name: 'Run', exact: true }).click();
      await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
      await selectTimeline(page, 0);
      await expectSelection(page, 0);
      await page.locator('.sfm-frame-selection').getByRole('switch').click();
      if (input === 'prepared') {
        await selectTimeline(page, 2);
        await expectSelection(page, 2);
        await page.waitForFunction(() => window.cameraDistance(2) < 1e-6);
        await page.getByRole('button', { name: 'Frame 1', exact: true }).click();
        await expectSelection(page, 0);
        assert.equal(await page.locator('video').getAttribute('src'), null);
        await page.getByRole('button', { name: 'Next frame', exact: true }).click();
        await page.waitForFunction(
          () => document.querySelector('.sfm-analysis-canvas')?.dataset.frame === '1',
        );
        await page.getByRole('button', { name: 'Previous frame', exact: true }).click();
        await expectSelection(page, 0);
        await page.getByRole('button', { name: 'Replay camera motion' }).click();
        await page.getByRole('button', { name: 'Next frame', exact: true }).click();
        assert.equal(await page.getByRole('button', { name: 'Replay camera motion' }).count(), 1);
        continue;
      }
      await selectTimeline(page, 3);
      await expectSelection(page, 3);
      await page.waitForFunction(() => window.cameraDistance(3) < 1e-6);
      assert.ok(Math.abs((await page.locator('video').evaluate((v) => v.currentTime)) - 3) < 0.02);
      await page.locator('video').evaluate((v) => {
        v.currentTime = 3.2;
      });
      await page.waitForFunction(() => !document.querySelector('video').seeking);
      await page.getByRole('button', { name: 'Frame 4', exact: true }).click();
      await page.waitForFunction(
        () => Math.abs(document.querySelector('video').currentTime - 3) < 0.02,
      );
      // A missing pose clears the selected camera and photo rather than keeping the old one.
      await selectTimeline(page, 1);
      await page.waitForFunction(
        () =>
          document.querySelector('.sfm-analysis-canvas')?.dataset.frame === '1' &&
          document.querySelector('.sfm-map .viewer')?.dataset.selectedView === '' &&
          !document.querySelector('.sfm-map .viewer')?.dataset.photo,
      );
      await page.locator('video').evaluate((v) => {
        v.currentTime = 4;
      });
      await expectSelection(page, 4);
      await page.locator('video').evaluate((v) => v.play());
      await page.waitForFunction(
        () => document.querySelector('.sfm-map .viewer')?.dataset.selectedView === '5',
      );
      assert.equal(
        await page.locator('video').evaluate((v) => v.paused),
        false,
        'playback must not pause itself when it selects a frame',
      );
      await page.locator('video').evaluate((v) => v.pause());
      assert.equal(
        await page.getByRole('button', { name: 'Previous frame', exact: true }).count(),
        0,
      );
      assert.equal(await page.getByRole('button', { name: 'Next frame', exact: true }).count(), 0);
      assert.equal(await page.getByRole('button', { name: 'Replay camera motion' }).count(), 0);
      assert.equal(await page.getByRole('slider', { name: 'Frame timeline' }).count(), 0);
      await selectTimeline(page, 3);
      await expectSelection(page, 3);
      await stepFrame(page);
      await expectSelection(page, 4);
      await stepFrame(page, 'ArrowLeft');
      await expectSelection(page, 3);
    }
    // A user-selected upload interval reaches the real decoder and keeps source times for browsing.
    await page.goto(address);
    await sparseOnly(page);
    await page.locator('.lab-topbar .ant-btn').click();
    await page
      .locator('input[type=file]')
      .first()
      .setInputFiles({ name: 'trimmed.mp4', mimeType: 'video/mp4', buffer: clip });
    const range = page.getByRole('region', { name: 'Analysis range', exact: true });
    await range.waitFor();
    const startTime = range.getByRole('spinbutton', { name: 'Analysis start time (seconds)' });
    const endTime = range.getByRole('spinbutton', { name: 'Analysis end time (seconds)' });
    await startTime.fill('2');
    await endTime.fill('5');
    await endTime.press('Tab');
    await page.waitForFunction(
      () => Math.abs(document.querySelector('video').currentTime - 5) < 0.02,
    );
    await page.evaluate(() => {
      window.holdSingle = true;
    });
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await page.waitForFunction(() => window.singleHeld);
    assert.equal(await startTime.isDisabled(), true);
    assert.equal(await endTime.isDisabled(), true);
    assert.equal(
      await page.getByTestId('run-progress').getByRole('progressbar').getAttribute('aria-valuenow'),
      '0',
      'the interval begins at zero progress, even when source time is nonzero',
    );
    await page.evaluate(() => {
      window.holdSingle = false;
      window.testWorker.postMessage(window.testWorker.heldFrame);
    });
    await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    const trimmed = await page.evaluate(() =>
      window.testWorker.frames.map(({ id, time, last }) => ({ id, time, last })),
    );
    assert.ok(trimmed.length >= 3);
    assert.ok(trimmed.every((frame) => frame.time >= 2 && frame.time < 5));
    assert.equal(trimmed[0].time, 2);
    assert.equal(trimmed.at(-1).last, true);
    const rangeRecord = JSON.parse(await page.locator('.run-record details pre').textContent());
    assert.equal(runRecordProblem(rangeRecord), undefined);
    assert.deepEqual(rangeRecord.input.range, { start: 2, end: 5 });
    assert.equal(rangeRecord.input.durationSeconds, 8);
    await selectTimeline(page, 2);
    await expectSelection(page, 0);
    await selectTimeline(page, 4);
    await expectSelection(page, 2);
    assert.ok(
      Math.abs((await page.locator('video').evaluate((video) => video.currentTime)) - 4) < 0.02,
    );
    if (process.env.TYPECV_RANGE_SCREENSHOT_DIR) {
      await range.screenshot({
        path: `${process.env.TYPECV_RANGE_SCREENSHOT_DIR}/video-range-desktop.png`,
      });
      await page.setViewportSize({ width: 390, height: 844 });
      await range.scrollIntoViewIfNeeded();
      assert.equal(
        await range.evaluate((element) => element.scrollWidth <= element.clientWidth),
        true,
      );
      await range.screenshot({
        path: `${process.env.TYPECV_RANGE_SCREENSHOT_DIR}/video-range-mobile.png`,
      });
      await page.setViewportSize({ width: 1500, height: 1000 });
    }
    await range.getByRole('button', { name: 'Select the whole video' }).click();
    assert.equal(Number(await startTime.inputValue()), 0);
    assert.equal(Number(await endTime.inputValue()), 8);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    assert.equal(await page.evaluate(() => window.testWorker.frames[0].time), 0);
    assert.ok(
      await page.evaluate((count) => window.testWorker.frames.length > count, trimmed.length),
    );
    assert.equal(
      JSON.parse(await page.locator('.run-record details pre').textContent()).input.range,
      undefined,
    );

    // Each uploaded video has an independent interval in the settings drawer.
    await page.goto(address);
    await sparseOnly(page);
    await page.locator('.lab-topbar .ant-btn').click();
    const secondClip = Buffer.concat([clip, Buffer.from([0, 0, 0, 8, 102, 114, 101, 101])]);
    await page
      .locator('input[type=file]')
      .first()
      .setInputFiles([
        { name: 'first.mp4', mimeType: 'video/mp4', buffer: clip },
        { name: 'second.mp4', mimeType: 'video/mp4', buffer: secondClip },
      ]);
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    const videoRows = page.locator('.mv-row');
    await videoRows.first().getByRole('region', { name: 'Analysis range' }).waitFor();
    for (const [index, start, end] of [
      [0, 1, 4],
      [1, 4, 7],
    ]) {
      const row = videoRows.nth(index);
      await row
        .getByRole('spinbutton', { name: 'Analysis start time (seconds)' })
        .fill(String(start));
      await row.getByRole('spinbutton', { name: 'Analysis end time (seconds)' }).fill(String(end));
    }
    await page
      .getByRole('dialog', { name: 'Run settings' })
      .getByRole('button', { name: 'Close', exact: true })
      .click();
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    const multiRange = await page.evaluate(() =>
      window.testWorker.videos.map((videoId) =>
        window.testWorker.frames
          .filter((frame) => frame.videoId === videoId)
          .map((frame) => frame.time),
      ),
    );
    assert.equal(multiRange.length, 2);
    assert.ok(multiRange[0].every((time) => time >= 1 && time < 4));
    assert.ok(multiRange[1].every((time) => time >= 4 && time < 7));
    const multiRangeRecord = JSON.parse(
      await page.locator('.run-record details pre').textContent(),
    );
    assert.equal(runRecordProblem(multiRangeRecord), undefined);
    assert.deepEqual(
      multiRangeRecord.input.videos.map((video) => video.range),
      [
        { start: 1, end: 4 },
        { start: 4, end: 7 },
      ],
    );
    await page.goto(address);
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await sparseOnly(page);
    await page.locator('.lab-topbar .ant-btn').click();
    await page.getByRole('button', { name: 'Load Three clips', exact: true }).click();
    await page.evaluate(() => {
      window.holdMulti = true;
    });
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await page.waitForFunction(() => window.multiHeld);
    const stop = page.getByRole('button', { name: 'Stop', exact: true });
    assert.equal(await stop.isEnabled(), true);
    await stop.click();
    await page.getByRole('button', { name: 'Run', exact: true }).waitFor();
    assert.equal(await page.evaluate(() => window.testWorker.terminated), true);
    await page.evaluate(() => {
      window.holdMulti = false;
    });
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    await expectSelection(page, 20);
    await page.getByRole('combobox', { name: 'Browse video' }).selectOption('b'.repeat(16));
    await expectSelection(page, 40);
    await page.locator('video').evaluate((v) => v.play());
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.selectedView === '41',
    );
    assert.equal(await page.locator('video').evaluate((v) => v.paused), false);
    await selectTimeline(page, 0);
    await expectSelection(page, 40);
    await stepFrame(page);
    await expectSelection(page, 41);
    assert.ok(Math.abs((await page.locator('video').evaluate((v) => v.currentTime)) - 1) < 0.02);
    await selectTimeline(page, 2);
    await page.waitForFunction(
      () =>
        document.querySelector('.sfm-map .viewer')?.dataset.selectedView === '42' &&
        !document.querySelector('.sfm-map .viewer')?.dataset.photo &&
        !document.querySelector('.sfm-analysis-canvas')?.dataset.frame,
    );
    // Click the camera that was actually drawn, including the map's placement transform.
    await page.locator('.sfm-map .viewer').scrollIntoViewIfNeeded();
    const point = await page.evaluate(() => window.project(22));
    await page.mouse.click(point.x, point.y);
    await expectSelection(page, 22);
    pendingImages.splice(0).forEach((release) => release());
    await page.waitForTimeout(100);
    assert.equal(
      await page.locator('.sfm-map .viewer').getAttribute('data-photo'),
      '22',
      'a late image cannot replace the newer selection',
    );
    assert.equal(
      await page.getByRole('combobox', { name: 'Browse video' }).inputValue(),
      'a'.repeat(16),
    );
    assert.match(await page.locator('.sfm-timeline-text').textContent(), /Frame 3 \/ 3.*2.00s/);
    await page.locator('.mv-components [data-component="1"]').click();
    await expectSelection(page, 60);
    assert.equal(
      await page.getByRole('combobox', { name: 'Browse video' }).inputValue(),
      'c'.repeat(16),
    );
    await page.locator('.sfm-frame-selection').getByRole('switch').click();
    await stepFrame(page);
    await expectSelection(page, 61);
    await page.waitForFunction(() => window.cameraDistance(61) < 1e-6);
    // Check the new controls in the production layout at a phone width.
    const assets = new URL('./dist/_astro/', import.meta.url);
    const stylesheet = (await readdir(assets)).find(
      (name) => name.startsWith('SiteLayout.') && name.endsWith('.css'),
    );
    await page.locator('style[data-harness]').evaluate((style) => style.remove());
    await page.locator('#app').evaluate((app) => {
      app.className = 'lab-main';
    });
    await page.addStyleTag({ content: await readFile(new URL(stylesheet, assets), 'utf8') });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForFunction(
      () =>
        getComputedStyle(document.querySelector('.sfm-live-grid')).gridTemplateColumns.split(' ')
          .length === 2,
    );
    await page.evaluate(() => window.scrollTo({ top: 0, left: 0, behavior: 'instant' }));
    const desktop = await page.evaluate(async () => {
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const stream = document.querySelector('.sfm-stream'),
        map = document.querySelector('.sfm-map');
      const frame = stream.querySelector('.sfm-stage').getBoundingClientRect();
      const viewer = map.querySelector('.viewer').getBoundingClientRect();
      const video = stream.querySelector('video').getBoundingClientRect();
      const overlay = stream.querySelector('canvas').getBoundingClientRect();
      return {
        frameTop: frame.top,
        viewerTop: viewer.top,
        streamHeight: stream.getBoundingClientRect().height,
        mapHeight: map.getBoundingClientRect().height,
        scrollHeight: stream.scrollHeight,
        clientHeight: stream.clientHeight,
        video: [video.width, video.height],
        overlay: [overlay.width, overlay.height],
      };
    });
    const layout = JSON.stringify(desktop);
    assert.ok(
      desktop.frameTop >= 0 &&
        desktop.frameTop < 900 &&
        desktop.viewerTop >= 0 &&
        desktop.viewerTop < 900,
      layout,
    );
    assert.ok(Math.abs(desktop.frameTop - desktop.viewerTop) < 8, layout);
    assert.ok(Math.abs(desktop.streamHeight - desktop.mapHeight) < 1, layout);
    assert.ok(desktop.scrollHeight <= desktop.clientHeight + 1, layout);
    assert.deepEqual(desktop.video, desktop.overlay, layout);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.sfm-video-selection').scrollIntoViewIfNeeded();
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      true,
    );
    for (const selector of ['.sfm-video-selection', '.sfm-frame-selection']) {
      const box = await page.locator(selector).boundingBox();
      assert.ok(box.x >= 0 && box.x + box.width <= 391, selector);
    }
    // The record exported by the real Lab combines two devices, and does not add the
    // first component's cumulative Dense snapshot to the last one's a second time.
    const profilePage = await browser.newPage();
    profilePage.on('pageerror', (error) => errors.push(String(error)));
    await profilePage.goto(address);
    await profilePage.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await profilePage.locator('.lab-topbar .ant-btn').click();
    await profilePage.getByRole('button', { name: 'Load Three clips', exact: true }).click();
    await profilePage.getByRole('button', { name: 'Run', exact: true }).click();
    await profilePage.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    await profilePage.locator('.lab-record-panel > summary').click();
    await profilePage.locator('#run-record > summary').click();
    const [downloaded] = await Promise.all([
      profilePage.waitForEvent('download'),
      profilePage.getByRole('button', { name: 'Download JSON', exact: true }).click(),
    ]);
    const record = JSON.parse(await readFile(await downloaded.path(), 'utf8'));
    assert.deepEqual(record.timing.gpu, [
      { name: 'sparseKernel', groups: 1, dispatches: 1, timedGroups: 1, gpuMs: 1 },
      { name: 'tsdfIntegrate', groups: 2, dispatches: 2, timedGroups: 2, gpuMs: 4 },
    ]);
    assert.equal(record.timing.gpuDropped, 3);
    assert.equal(runRecordProblem(record), undefined);
    await profilePage.close();
    // The explanatory alert and the original failure are both present. Select the original
    // diagnostic explicitly, and keep numerical work usable after the viewer failed.
    const failedViewer = await browser.newPage();
    failedViewer.on('pageerror', (error) => errors.push(String(error)));
    await failedViewer.goto(address + '&viewerFailure=1');
    await failedViewer.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'error',
    );
    const diagnostic = failedViewer
      .locator('.sfm-map [role=alert]')
      .filter({ hasText: 'Injected initial viewer render failure' });
    assert.equal(await failedViewer.locator('.sfm-map [role=alert]').count(), 2);
    assert.match(await diagnostic.innerText(), /Injected initial viewer render failure/);
    await sparseOnly(failedViewer);
    await failedViewer.locator('.lab-topbar .ant-btn').click();
    await failedViewer.getByRole('button', { name: 'Load Clip A', exact: true }).click();
    await failedViewer.getByRole('button', { name: 'Run', exact: true }).click();
    await failedViewer.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    assert.ok(await failedViewer.evaluate(() => window.testWorker.frames.length > 0));
    assert.match(await diagnostic.innerText(), /Injected initial viewer render failure/);
    await failedViewer.close();
    const koreanRange = await browser.newPage({
      viewport: { width: 390, height: 844 },
      reducedMotion: 'reduce',
    });
    koreanRange.on('pageerror', (error) => errors.push(String(error)));
    await koreanRange.goto(address.replace('/?', '/ko/?'));
    await koreanRange.locator('.lab-topbar .ant-btn').click();
    await koreanRange
      .locator('input[type=file]')
      .first()
      .setInputFiles({ name: 'korean.mp4', mimeType: 'video/mp4', buffer: clip });
    const koreanEditor = koreanRange.getByRole('region', { name: '분석 구간', exact: true });
    await koreanEditor.waitFor();
    const koreanStart = koreanEditor.getByRole('spinbutton', { name: '분석 시작 시간 (초)' });
    const koreanEnd = koreanEditor.getByRole('spinbutton', { name: '분석 종료 시간 (초)' });
    await koreanStart.fill('2.25');
    await koreanEnd.fill('1');
    await koreanEnd.press('Tab');
    assert.equal(
      Number(await koreanEnd.inputValue()),
      3.25,
      'the end cannot cross the start or make a sub-second interval',
    );
    await koreanEnd.fill('5.75');
    await koreanEnd.press('Tab');
    await koreanEditor.getByRole('slider', { name: '분석 시작 위치' }).focus();
    await koreanRange.keyboard.press('ArrowLeft');
    assert.equal(Number(await koreanStart.inputValue()), 2.24);
    await koreanRange.evaluate(() => {
      document.documentElement.dataset.theme = 'dark';
    });
    assert.equal(
      await koreanEditor.evaluate((element) => element.scrollWidth <= element.clientWidth),
      true,
    );
    if (process.env.TYPECV_RANGE_SCREENSHOT_DIR)
      await koreanEditor.screenshot({
        path: `${process.env.TYPECV_RANGE_SCREENSHOT_DIR}/video-range-mobile-ko.png`,
      });
    await koreanRange.locator('.lab-topbar .ant-btn').click();
    await koreanRange
      .locator('input[type=file]')
      .first()
      .setInputFiles({ name: 'replacement.mp4', mimeType: 'video/mp4', buffer: clip });
    await koreanEditor.waitFor();
    assert.equal(
      Number(await koreanStart.inputValue()),
      0,
      'changing input clears the old selection',
    );
    assert.equal(Number(await koreanEnd.inputValue()), 8);
    await koreanRange.close();
    assert.deepEqual(errors, []);
  } finally {
    pendingImages.splice(0).forEach((release) => release());
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
