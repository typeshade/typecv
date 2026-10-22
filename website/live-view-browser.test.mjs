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
THREE.Scene.prototype.onBeforeRender = function(renderer, scene, camera) {
  window.cameraState = {
    position: camera.position.toArray(), rotation: camera.quaternion.toArray(), fov: camera.fov,
  };
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

test('the viewer preserves live map transitions and smooths frame camera navigation', async () => {
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
      // A frame change moves only the viewing camera. Test the actual rendered midpoint,
      // retargeting during a move, exact arrival, cancellation and reduced motion.
      const camera = await page.evaluate(async () => {
        const snapshot = () => structuredClone(window.cameraState);
        const model = {
          views: [
            {
              id: 1,
              height: 100,
              pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] },
              camera: { fx: 100, fy: 100, cx: 50, cy: 50 },
            },
            {
              id: 2,
              height: 100,
              pose: { R: [0, 1, 0, -1, 0, 0, 0, 0, 1], t: [-3, 2, 0] },
              camera: { fx: 50, fy: 50, cx: 50, cy: 50 },
            },
            {
              id: 3,
              height: 100,
              pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [2, 0, 0] },
              camera: { fx: 200, fy: 200, cx: 50, cy: 50 },
            },
          ],
          landmarks: window.map(0).landmarks.map((p) => ({
            ...p,
            observations: [1, 2, 3].map((viewId) => ({
              viewId,
              featureId: p.trackId,
              pixel: [50, 50],
            })),
          })),
        };
        const original = JSON.stringify(model);
        await window.drive(20, {
          reconstruction: model,
          live: false,
          selectedViewId: 1,
          photo: undefined,
          dense: undefined,
          densePreview: undefined,
        });
        const before = snapshot();
        await window.drive(20, { cameraView: { id: 1, nonce: 1 } });
        const firstStart = snapshot();
        await window.drive(20.4, {});
        const firstEnd = snapshot();
        await window.drive(21, { selectedViewId: 2, cameraView: { id: 2, nonce: 2 } });
        const secondStart = snapshot();
        await window.drive(21.15, {});
        const midpoint = snapshot();
        await window.drive(21.15, { selectedViewId: 3, cameraView: { id: 3, nonce: 3 } });
        const retargeted = snapshot();
        await window.drive(21.3, {});
        const retargetMidpoint = snapshot();
        await window.drive(21.5, {});
        const end = snapshot();
        await window.drive(22, { selectedViewId: 2, cameraView: { id: 2, nonce: 4 } });
        await window.drive(22.15, {});
        const beforeCancel = snapshot();
        await window.drive(22.15, { cameraView: undefined });
        await window.drive(22.7, {});
        const cancelled = snapshot();
        await window.drive(23, { selectedViewId: 999, cameraView: { id: 999, nonce: 5 } });
        const missing = snapshot();
        await window.drive(24, { selectedViewId: 1, cameraView: { id: 1, nonce: 6 } });
        await window.drive(24.15, {});
        return {
          before,
          firstStart,
          firstEnd,
          secondStart,
          midpoint,
          retargeted,
          retargetMidpoint,
          end,
          beforeCancel,
          cancelled,
          missing,
          unchanged: JSON.stringify(model) === original,
        };
      });
      const close = (actual, expected, label) => {
        assert.equal(actual.length, expected.length, label);
        for (let i = 0; i < actual.length; i++)
          assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6, `${label}: ${actual} vs ${expected}`);
      };
      const fov1 = (2 * Math.atan(0.5) * 180) / Math.PI;
      close(camera.firstEnd.position, [0, 0, 0], 'first camera center');
      close(camera.firstEnd.rotation, [0, 0, 0, 1], 'first orientation');
      close([camera.firstEnd.fov], [fov1], 'first field of view');
      if (reducedMotion === 'no-preference') {
        close(camera.firstStart.position, camera.before.position, 'no jump at first selection');
        close(camera.secondStart.position, [0, 0, 0], 'no jump on next frame');
        close(camera.midpoint.position, [1, -1.5, 0], 'position midpoint');
        close(
          camera.midpoint.rotation,
          [0, 0, -Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)],
          'rotation midpoint',
        );
        close([camera.midpoint.fov], [(fov1 + 90) / 2], 'field-of-view midpoint');
        close(
          camera.retargeted.position,
          camera.midpoint.position,
          'retarget from displayed position',
        );
        close(
          camera.retargeted.rotation,
          camera.midpoint.rotation,
          'retarget from displayed rotation',
        );
        close(camera.retargetMidpoint.position, [-0.5, -0.75, 0], 'retargeted midpoint');
      } else {
        close(camera.secondStart.position, [2, -3, 0], 'reduced motion moves immediately');
        close(camera.retargeted.position, [-2, 0, 0], 'reduced motion retargets immediately');
      }
      close(camera.end.position, [-2, 0, 0], 'exact destination');
      close(camera.end.rotation, [0, 0, 0, 1], 'exact destination orientation');
      close([camera.end.fov], [(2 * Math.atan(0.25) * 180) / Math.PI], 'exact destination lens');
      close(
        camera.cancelled.position,
        camera.beforeCancel.position,
        'disabled follow stops motion',
      );
      close(
        camera.missing.position,
        camera.cancelled.position,
        'missing camera does not create a pose',
      );
      assert.equal(camera.unchanged, true, 'reported poses remain unchanged');

      // A manual orbit gesture takes over immediately, without the tween resuming afterward.
      await page.mouse.move(20, 20);
      await page.mouse.down();
      await page.mouse.up();
      const manual = await page.evaluate(async () => {
        const before = structuredClone(window.cameraState);
        await window.drive(25, {});
        return {
          before,
          after: window.cameraState,
          motion: document.querySelector('.viewer').dataset.cameraMotion,
        };
      });
      close(manual.after.position, manual.before.position, 'manual navigation cancels motion');
      assert.equal(manual.motion, 'idle');
      assert.deepEqual(errors, [], reducedMotion);
      await page.close();
    }
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
