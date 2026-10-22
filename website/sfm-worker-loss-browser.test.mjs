import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { build } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';

// Run the real worker and request queue; replace only the numerical runtime to inject losses
// deterministically, including one after a completed frame and one followed by a readback error.
const runtime = `
let current, activeLoss = false;
const devices = [];
self.addEventListener('message', ({data}) => {
  if (data.type !== 'cancelDense' || !data.testAction) return;
  if (data.testAction === 'arm') activeLoss = true;
  if (data.testAction === 'lose') current.lose();
  if (data.testAction === 'oldLoss') devices[0].lose();
  if (data.testAction === 'duplicate') self.dispatchEvent(new ErrorEvent('error', {message:'late error'}));
  self.postMessage({id:data.id, type:'testAck', action:data.testAction});
});
export class ShadeDevice {
  static async open(options) {
    let lose;
    const device = {
      ...options, backend:'webgpu', boundedSubmissions:!!options.boundedSubmissions,
      fp64Flavor:options.fp64Flavor === 'auto' ? 'integer' : options.fp64Flavor,
      statistics:{liveBytes:0,pooledBytes:0,applicationUploadBytes:0,applicationReadbackBytes:0},
      kernelStatistics:{rows:[],dropped:0}, chunkRates:{},
      lost:new Promise(resolve => { lose = resolve; }),
      lose() { lose({reason:'destroyed',message:''}); },
      async dispose() { self.postMessage({type:'testDisposed'}); },
    };
    current = device; devices.push(device); return device;
  }
}
export class IncrementalSfmSession {
  activeViewIds = [];
  constructor(device, options) { this.device = device; this.options = options; }
  async add(frame) {
    this.options.onProgress({operation:'guided matching',index:10,phase:'start'});
    this.device.onProgress({kernel:'scatterFloatMatches',phase:'start'});
    this.device.onProgress({kernel:'scatterFloatMatches',phase:'complete'});
    this.activeViewIds.push(frame.id);
    if (activeLoss) {
      this.device.lose();
      await new Promise(resolve => setTimeout(resolve, 0));
      throw new Error('readback after device loss');
    }
    return undefined;
  }
  async finish() { return {reconstruction:{views:[],landmarks:[]},diagnostics:{},unregistered:[]}; }
}
`;

test('worker reports idle and active losses once, and ignores a replaced device', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'typecv-worker-loss-'));
  try {
    const workerFile = join(directory, 'worker.js');
    const runtimeFile = join(directory, 'gpu.js');
    await build({
      entryPoints: [new URL('./src/workers/sfm.worker.ts', import.meta.url).pathname],
      outfile: workerFile,
      bundle: true,
      format: 'esm',
      platform: 'browser',
    });
    await writeFile(runtimeFile, runtime);
    const { result, errors } = await inBrowser(
      async () => {
        const run = async (mode) => {
          const worker = new Worker('/worker.js', { type: 'module' });
          const messages = [],
            waiting = [];
          worker.onmessage = ({ data }) => {
            messages.push(data);
            for (const item of waiting.splice(0)) item();
          };
          const wait = async (predicate) => {
            while (!messages.some(predicate)) await new Promise((resolve) => waiting.push(resolve));
            return messages.find(predicate);
          };
          const send = async (message, type) => {
            worker.postMessage(message);
            return wait((m) => m.id === message.id && m.type === type);
          };
          const action = async (testAction, id = 1) => {
            worker.postMessage({ id, type: 'cancelDense', testAction });
            await wait((m) => m.type === 'testAck' && m.action === testAction);
          };
          try {
            await send({ type: 'start', id: 1, focalRatio: 0.8, fp64: 'integer' }, 'ready');
            if (mode === 'active') await action('arm');
            await send(
              {
                type: 'frame',
                id: 1,
                frame: {
                  id: 10,
                  time: 1,
                  width: 8,
                  height: 8,
                  pixels: new Uint8ClampedArray(256),
                  precomputed: { keypoints: new Float64Array([2, 2]), pairs: [] },
                },
              },
              mode === 'active' ? 'error' : 'frameComplete',
            );
            if (mode === 'idle') {
              await action('lose');
              await wait((m) => m.type === 'error');
              await action('duplicate');
            } else if (mode === 'active') {
              await wait((m) => m.type === 'testDisposed');
            } else {
              await send({ type: 'finish', id: 1 }, 'result');
              await send({ type: 'start', id: 2, focalRatio: 0.8, bounded: true }, 'ready');
              await action('oldLoss', 2);
            }
            return messages.filter((m) => m.type === 'error');
          } finally {
            worker.terminate();
          }
        };
        return {
          idle: await run('idle'),
          active: await run('active'),
          replaced: await run('replace'),
        };
      },
      undefined,
      { serve: { '/worker.js': workerFile, '/runtime/gpu.js': runtimeFile } },
    );
    assert.deepEqual(errors, []);
    assert.equal(result.idle.length, 1);
    const idle = result.idle[0].failure;
    assert.equal(idle.operation, 'waiting for next request');
    assert.equal(idle.kernel, '');
    assert.equal(idle.context.state, 'idle');
    assert.equal(idle.context.lastCompletedRequest.frameId, 10);
    assert.equal(idle.context.lastKernel.name, 'scatterFloatMatches');
    assert.equal(idle.context.lastKernel.inCurrentRequest, false);
    assert.equal(result.active.length, 1, 'readback rejection does not overwrite native loss');
    assert.equal(result.active[0].failure.reason, 'destroyed');
    assert.equal(result.active[0].failure.context.state, 'active');
    assert.equal(result.active[0].failure.context.request.frameId, 10);
    assert.equal(result.active[0].failure.kernel, 'scatterFloatMatches');
    assert.deepEqual(result.replaced, []);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
