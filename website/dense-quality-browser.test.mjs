import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';

// Exercise the real worker request queue and dense context. Only numerical computation is
// replaced, so forwarded controls and the final report must agree without running SfM/MVS.
const runtime = `
export class ShadeDevice {
 static async open(options){return {...options,backend:'webgpu',fp64Flavor:'integer',statistics:{liveBytes:0,pooledBytes:0,applicationUploadBytes:0,applicationReadbackBytes:0},kernelStatistics:{rows:[],dropped:0},chunkRates:{},async dispose(){}};}
}
export class IncrementalSfmSession {
 activeViewIds=[];
 async add(frame){this.activeViewIds.push(frame.id);}
 async finish(){return {reconstruction:{views:[{id:10,pose:{R:[1,0,0,0,1,0,0,0,1],t:[0,0,0]},camera:{fx:8,fy:8,cx:3.5,cy:3.5}}],landmarks:[]},diagnostics:{},unregistered:[]};}
}
export async function reconstructDenseGpu(device,scene,images,options){
 self.postMessage({type:'testQualityReceived',quality:{boundaryAware:options.boundaryAware,minFiniteGain:options.minFiniteGain}});
 return {points:{count:0},views:[{id:10,status:'done'}]};
}
`;

test('single-video worker forwards quality controls and resets them between runs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'typecv-dense-quality-'));
  try {
    const worker = join(directory, 'worker.js'),
      gpu = join(directory, 'gpu.js');
    await build({
      entryPoints: [new URL('./src/workers/sfm.worker.ts', import.meta.url).pathname],
      outfile: worker,
      bundle: true,
      format: 'esm',
      platform: 'browser',
    });
    await writeFile(gpu, runtime);
    const { result, errors } = await inBrowser(
      async () => {
        const worker = new Worker('/worker.js', { type: 'module' }),
          messages = [],
          waiting = [];
        worker.onmessage = ({ data }) => {
          messages.push(data);
          for (const f of waiting.splice(0)) f();
        };
        const wait = async (predicate) => {
          while (!messages.some(predicate)) await new Promise((r) => waiting.push(r));
          return messages.find(predicate);
        };
        const send = async (message, type) => {
          worker.postMessage(message);
          return wait((m) => m.id === message.id && (m.type === type || m.type === 'error'));
        };
        const rows = [];
        try {
          const controls = [
            { boundaryAware: true, minFiniteGain: 0.02 },
            undefined,
            { boundaryAware: false, minFiniteGain: 0 },
          ];
          for (let i = 0; i < controls.length; i++) {
            const id = i + 1,
              start = await send(
                { type: 'start', id, focalRatio: 0.8, dense: true, denseQuality: controls[i] },
                'ready',
              );
            if (start.type === 'error') throw new Error(JSON.stringify(start));
            const frame = await send(
              {
                type: 'frame',
                id,
                frame: {
                  id: 10,
                  time: 1,
                  width: 8,
                  height: 8,
                  pixels: new Uint8ClampedArray(256),
                  precomputed: { keypoints: new Float64Array([2, 2]), pairs: [] },
                },
              },
              'frameComplete',
            );
            if (frame.type === 'error') throw new Error(JSON.stringify(frame));
            const sparse = await send({ type: 'finish', id }, 'result');
            if (sparse.type === 'error') throw new Error(JSON.stringify(sparse));
            const final = await wait(
              (m) => m.id === id && (m.type === 'denseResult' || m.type === 'error'),
            );
            rows.push({
              quality: messages.filter((m) => m.type === 'testQualityReceived').at(-1)?.quality,
              final,
            });
          }
          return rows;
        } finally {
          worker.terminate();
        }
      },
      null,
      { serve: { '/worker.js': worker, '/runtime/gpu.js': gpu }, timeoutMs: 60000 },
    );
    assert.deepEqual(errors, []);
    assert.equal(result.length, 3);
    for (const [i, expected] of [
      [0, { boundaryAware: true, minFiniteGain: 0.02 }],
      [2, { boundaryAware: false, minFiniteGain: 0 }],
    ]) {
      assert.deepEqual(result[i].quality, expected);
      assert.equal(result[i].final.failure, null);
      assert.equal(result[i].final.options.boundaryAware, expected.boundaryAware);
      assert.equal(result[i].final.options.minFiniteGain, expected.minFiniteGain);
    }
    assert.equal(result[1].quality.boundaryAware, undefined);
    assert.equal(result[1].quality.minFiniteGain, undefined);
    assert.equal(result[1].final.options.boundaryAware, undefined);
    assert.equal(result[1].final.options.minFiniteGain, undefined);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
