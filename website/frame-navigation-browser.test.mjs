import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, readdir } from 'node:fs/promises';
import { build } from 'esbuild';
import { chromium } from 'playwright';

// The real Lab, decoder and WebGL viewer; only SfM results are deterministic worker messages.
const harness = `
import React from 'react';
import {createRoot} from 'react-dom/client';
import * as THREE from 'three';
import SfmLab from './src/components/SfmLab.tsx';
THREE.Scene.prototype.onBeforeRender = function(renderer, scene, camera) {
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
  send(message) { this.onmessage?.({data: {id: this.id, ...message}}); }
  postMessage(m) {
    this.id = m.id;
    queueMicrotask(() => {
      if (m.type === 'start') this.send({type:'ready'});
      if (m.type === 'multiStart') this.send({type:'multiReady'});
      if (m.type === 'multiVideo') { this.videos.push(m.video.videoId); this.send({type:'videoStarted'}); }
      if (m.type === 'multiEndVideo') this.send({type:'videoComplete',videoId:m.videoId,keyframes:3});
      if (m.type === 'frame' || m.type === 'multiFrame') {
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
          this.send({type:'result',stages:{},totalMs:1,keyframeCount:9,summary:{videos:this.videos.map((videoId,i) => ({videoId,keyframes:3,registeredViews:3,component:i<2?0:1,tags:['component']})),components:[],notConnected:[]}});
        }
      }
    });
  }
  terminate() {}
};
createRoot(document.getElementById('app')).render(<SfmLab />);
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
    codec: 'h264',
  },
}));

async function selectTimeline(page, value) {
  await page.getByRole('slider', { name: 'Frame timeline' }).evaluate((input, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, String(value));
}
async function expectSelection(page, id) {
  await page.waitForFunction((id) => {
    const viewer = document.querySelector('.sfm-map .viewer');
    const frame = document.querySelector('.sfm-analysis-canvas');
    return (
      viewer?.dataset.selectedView === String(id) &&
      viewer.dataset.photo === String(id) &&
      frame?.dataset.frame === String(id)
    );
  }, id);
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
  const clip = await readFile(new URL('../tests/fixtures/source-clip.mp4', import.meta.url));
  const jpeg = await readFile(
    new URL('./public/sample/south-building/frame-00.jpg', import.meta.url),
  );
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
        '<style data-harness>body{margin:0}.viewer{width:640px;height:420px}.sfm-map{width:640px}.sfm-stage{position:relative;width:640px;height:360px}.sfm-stage video,.sfm-analysis-canvas{position:absolute;width:100%;height:100%;object-fit:contain}.sfm-analysis-canvas{pointer-events:none}.sfm-analysis-canvas[data-shown=false]{display:none}.frame-grid{display:flex}.frame-grid img{width:60px}.sample-card img{width:100px}.sr-only{display:none}</style><div id="app"></div><script type="module" src="/app.js"></script>',
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
      await page.getByRole('button', { name: /Run GPU SfM/ }).click();
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
        () => Number(document.querySelector('[aria-label="Frame timeline"]').value) >= 5,
      );
      assert.equal(
        await page.locator('video').evaluate((v) => v.paused),
        false,
        'playback must not pause itself when it selects a frame',
      );
      await page.locator('video').evaluate((v) => v.pause());
      await page.getByRole('button', { name: 'Replay camera motion' }).click();
      await page.getByRole('button', { name: 'Previous frame', exact: true }).click();
      assert.equal(
        await page.getByRole('button', { name: 'Replay camera motion' }).count(),
        1,
        'manual navigation stops replay',
      );
    }
    await page.goto(address);
    await page.waitForFunction(
      () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
    );
    await page.locator('.lab-topbar .ant-btn').click();
    await page.getByRole('button', { name: 'Load Three clips', exact: true }).click();
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.getByRole('button', { name: 'Export JSON', exact: true }).waitFor();
    await expectSelection(page, 20);
    await page.getByRole('combobox', { name: 'Browse video' }).selectOption('b'.repeat(16));
    await expectSelection(page, 40);
    await page.locator('video').evaluate((v) => v.play());
    await page.waitForFunction(
      () => Number(document.querySelector('[aria-label="Frame timeline"]').value) >= 1,
    );
    assert.equal(await page.locator('video').evaluate((v) => v.paused), false);
    await selectTimeline(page, 0);
    await expectSelection(page, 40);
    await page.getByRole('button', { name: 'Next frame', exact: true }).click();
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
    assert.equal(await page.getByRole('slider', { name: 'Frame timeline' }).inputValue(), '2');
    await page.locator('.mv-components [data-component="1"]').click();
    await expectSelection(page, 60);
    assert.equal(
      await page.getByRole('combobox', { name: 'Browse video' }).inputValue(),
      'c'.repeat(16),
    );
    await page.locator('.sfm-frame-selection').getByRole('switch').click();
    await page.getByRole('button', { name: 'Next frame', exact: true }).click();
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
    assert.deepEqual(errors, []);
  } finally {
    pendingImages.splice(0).forEach((release) => release());
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
