import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { build } from 'esbuild';
import { chromium } from 'playwright';

// Exercise the real React viewer and compile every shader, with reported maps on a controlled
// clock. No reconstruction/GPU worker is needed to reproduce overlapping map deliveries.
const harness = `
import React from 'react';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import * as THREE from 'three';
import SfmViewer from './src/components/SfmViewer.tsx';
let now = 10;
performance.now = () => now * 1000;
THREE.Scene.prototype.onBeforeRender = function(renderer, scene) {
  window.layers = [];
  scene.traverse(object => { if (object.isPoints || object.isLineSegments) window.layers.push(object); });
};
const root = createRoot(document.getElementById('app'));
let props = { onSelectView() {}, onState(state) { window.viewerState = state; } };
window.drive = async (time, changes) => {
  now = time;
  props = { ...props, ...changes };
  flushSync(() => root.render(React.createElement(SfmViewer, props)));
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
};
window.layer = kind => window.layers.find(object =>
  kind === 'sparse' ? object.isPoints && object.material.size === 3 && !object.material.transparent :
  kind === 'dense' ? object.material.customProgramCacheKey() === 'typecv-dense-live-srgb' :
  kind === 'preview' ? object.isPoints && object.material.size === 2 && object.geometry.hasAttribute('aStarted') :
  kind === 'links' ? object.isPoints && object.material.size === 4 :
  kind === 'ghosts' ? object.isPoints && object.material.size === 3 && object.material.transparent :
  false);
window.attributes = kind => {
  const object = window.layer(kind), geometry = object.geometry;
  return Object.fromEntries(Object.entries(geometry.attributes).map(([name, attr]) => [name, Array.from(attr.array).slice(0, Number.isFinite(geometry.drawRange.count) ? geometry.drawRange.count * attr.itemSize : undefined)]));
};
window.map = x => ({
  views: [{ id: 1, pose: { R: [1,0,0,0,1,0,0,0,1], t: [0,0,0] }, camera: { fx: 100, fy: 100, cx: 50, cy: 50 } }],
  landmarks: [0,1,2].map(i => ({ trackId: i, xyz: [x + i, 0, 4], color: [1,0.5,0.2], observations: [{viewId: 1, featureId: i, pixel: [50+i,50]}] }))
});
window.chunk = x => ({viewId: 1, positions: Float32Array.of(x,0,4,x+1,0,4), colors: new Uint8Array(6).fill(180)});
window.preview = depth => ({ run: 1, done: false, views: new Map([[1, {
  viewId: 1, width: 2, height: 2, imageWidth: 100, imageHeight: 100,
  stage: 'photometric', depth: new Float32Array(4).fill(depth), rgb: new Uint8Array(12).fill(180)
}]]) });
window.drive(10, {});
`;

test('live viewer preserves overlapping transitions, animates dense arrivals and finishes the last map', async () => {
  const bundle = await build({
    stdin: { contents: harness, resolveDir: import.meta.dirname, loader: 'tsx' },
    bundle: true,
    write: false,
    format: 'esm',
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"production"' },
  });
  const server = createServer((req, res) => {
    res.setHeader('Content-Type', req.url === '/app.js' ? 'text/javascript' : 'text/html');
    res.end(
      req.url === '/app.js'
        ? bundle.outputFiles[0].text
        : '<style>html,body{margin:0}.viewer{width:800px;height:600px}</style><div id="app"></div><script type="module" src="/app.js"></script>',
    );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
    });
    for (const reducedMotion of ['no-preference', 'reduce']) {
      const page = await browser.newPage({ reducedMotion });
      const errors = [];
      page.on('pageerror', (error) => errors.push(String(error)));
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.goto(`http://127.0.0.1:${server.address().port}/`);
      await page.waitForFunction(() => window.viewerState?.status === 'ready');
      const results = await page.evaluate(async () => {
        await window.drive(10, { reconstruction: window.map(0), live: true, selectedViewId: 1 });
        const first = window.attributes('sparse');
        await window.drive(10.5, { reconstruction: window.map(0) });
        const second = window.attributes('sparse');
        await window.drive(10.6, { reconstruction: window.map(10) });
        await window.drive(11, { reconstruction: window.map(20) });
        const interrupted = window.attributes('sparse');
        const chunk = window.chunk(1);
        await window.drive(11.1, {
          dense: { run: 1, chunks: [chunk], count: 2 },
          densePreview: window.preview(4),
        });
        const dense = window.attributes('dense');
        const preview = window.attributes('preview');
        await window.drive(11.3, {
          densePreview: window.preview(5),
          selectedViewId: 1,
          photo: { id: 1, width: 100, height: 100 },
        });
        const nextPreview = window.attributes('preview');
        const nextChunk = window.chunk(2);
        await window.drive(11.4, { dense: { run: 1, chunks: [chunk, nextChunk], count: 4 } });
        const nextDense = window.attributes('dense');
        await window.drive(11.5, { reconstruction: window.map(30), live: false, dense: undefined });
        const final = window.attributes('sparse');
        await window.drive(12.4, { showCameras: false });
        const done = window.attributes('sparse');
        return { first, second, interrupted, dense, nextDense, preview, nextPreview, final, done };
      });
      assert.deepEqual(errors, [], reducedMotion);
      assert.equal(results.first.aBorn[0], results.second.aBorn[0]);
      assert.ok(Math.abs(results.dense.aBorn[0] - 11.1) < 1e-4);
      assert.equal(results.nextDense.aBorn[0], results.dense.aBorn[0]);
      assert.ok(results.nextDense.aBorn[2] > results.nextDense.aBorn[0]);
      assert.equal(results.preview.aBorn[0], results.nextPreview.aBorn[0]);
      if (reducedMotion === 'no-preference') {
        assert.ok(Math.abs(results.interrupted.aFrom[0] - 5) < 1e-4);
        assert.ok(Math.abs(results.final.aStarted[0] - 11.5) < 1e-4);
        assert.ok(results.final.aFrom[0] < results.final.position[0]);
      } else {
        assert.equal(results.interrupted.aFrom[0], 20);
        assert.equal(results.final.aFrom[0], 30);
      }
      assert.equal(results.done.position[0], 30);
      await page.close();
    }
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
