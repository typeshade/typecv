// The transcript harness of phase 0 (`docs/library-structure.md`, section 5, "Lab transcripts"):
// the real SfM worker, bundled with esbuild, runs in a `vm` realm of its own. It is driven only
// through `onmessage`, as the page drives it. Its only other inputs are the runtime entries (the
// deterministic fake of `lab-fake-runtime.mjs`), the adapter, the vocabulary file, a clock and
// timers. The harness owns all of them, so a run gives the same transcript on every machine.
//
// The clock stands still while the worker handles a request. Between two requests the harness
// moves it on (by `stepMs`, or to `at`) and fires the timers that come due on the way. So a
// duration that the worker measures inside one request is 0, and one that spans requests is the
// time the harness gave it.
import vm from 'node:vm';
import { gunzipSync } from 'node:zlib';
import { build } from 'esbuild';
import { canonical, jsonLines, sha256 } from './lab-baseline.mjs';
import { createFakeRuntime } from './lab-fake-runtime.mjs';
import { fft2d } from './src/lib/phase-correlation.ts';

export const ORIGIN = 'https://lab.typecv.test';

/** A band-limited 256 × 256 texture from a fixed sequence, as `keyframe-gate.test.mjs` makes. */
function texture(seed, size = 256) {
  let s = seed >>> 0;
  const random = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
  const re = new Float64Array(size * size),
    im = new Float64Array(size * size);
  for (let i = 0; i < size * size; i++) re[i] = random() - 0.5;
  fft2d(re, im, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const gain = 1 / (1 + Math.hypot(Math.min(x, size - x), Math.min(y, size - y))) ** 1.5;
      re[y * size + x] *= gain;
      im[y * size + x] *= gain;
    }
  fft2d(re, im, size, true);
  let lo = Infinity,
    hi = -Infinity;
  for (const v of re) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return re.map((v) => Math.round((255 * (v - lo)) / (hi - lo)));
}
/** Two unrelated scenes: the phase gate correlates one with itself, not with the other. */
export const SCENE = texture(5),
  OTHER = texture(77);

/**
 * The RGBA pixels of a frame: `source` seen through a window that moved by `-shift`, so that its
 * content moved by `shift`. The alpha byte of the first pixel names the content index, which the
 * fake runtime reads to give the frame its features.
 */
export function framePixels(content, [dx, dy] = [0, 0], source = SCENE, width = 128, height = 96) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = source[(y + 64 - dy) * 256 + x + 64 - dx];
      rgba.set([v, v, 255 - v, 255], (y * width + x) * 4);
    }
  rgba[3] = content;
  return rgba;
}
const WORKER = new URL('./src/workers/sfm.worker.ts', import.meta.url).pathname;

/** The adapter that `navigator.gpu.requestAdapter()` gives the worker. */
export const ADAPTER = {
  info: {
    vendor: 'typecv-test',
    architecture: 'fake',
    device: 'transcript',
    description: 'Deterministic fake adapter',
  },
  isFallbackAdapter: false,
  limits: {
    maxBufferSize: 268435456,
    maxStorageBufferBindingSize: 134217728,
    maxComputeWorkgroupStorageSize: 32768,
    maxComputeInvocationsPerWorkgroup: 256,
    maxStorageBuffersPerShaderStage: 10,
  },
  features: new Set(['timestamp-query', 'shader-f16']),
};

/**
 * Bundle the worker as the browser runs it, as one script. The worker loads its runtime with
 * `import(url)`. esbuild lowers that to `require(url)` here, and the harness answers it.
 */
export async function bundleWorker() {
  const result = await build({
    entryPoints: [WORKER],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    write: false,
    logLevel: 'silent',
    supported: { 'dynamic-import': false },
  });
  return result.outputFiles[0].text;
}

/** Copy a message into the worker's realm, as structured cloning does for `postMessage`. */
const COPY = `(function copy(value) {
  if (value === null || typeof value !== 'object') return value;
  if (ArrayBuffer.isView(value))
    return new globalThis[Object.prototype.toString.call(value).slice(8, -1)](value);
  if (Array.isArray(value)) return value.map(copy);
  const out = {};
  for (const key of Object.keys(value)) out[key] = copy(value[key]);
  return out;
})`;

/** A gzip part of the depth archive: its bytes depend on the zlib build, its content does not. */
function encodeOut(data) {
  if (data && data.type === 'denseArtifact' && ArrayBuffer.isView(data.data)) {
    const inflated = gunzipSync(data.data);
    return canonical({
      ...data,
      data: { gzip: { length: inflated.length, sha256: sha256(inflated) } },
    });
  }
  return canonical(data);
}

/** The message types that end a request, as the page waits for them. */
function ends(message, dense) {
  const id = message.id;
  const reply = (types) => (out) =>
    out.id === id && (out.type === 'error' || types.includes(out.type));
  switch (message.type) {
    case 'start':
      return reply(['ready']);
    case 'frame':
      return (out) =>
        reply([])(out) || (out.type === 'frameComplete' && out.frameId === message.frame.id);
    case 'finish':
      return reply(dense ? ['denseResult'] : ['result']);
    case 'multiStart':
      return reply(['multiReady']);
    case 'multiVideo':
      return reply(['videoStarted']);
    case 'multiFrame':
      return (out) =>
        reply([])(out) ||
        (out.type === 'frameComplete' &&
          out.videoId === message.videoId &&
          out.frameId === message.frame.id);
    case 'multiEndVideo':
      return reply(['videoComplete']);
    case 'multiFinish':
      return reply(dense ? ['denseDone'] : ['result']);
    default:
      return undefined;
  }
}

export class WorkerHarness {
  /**
   * `code` is the bundle of `bundleWorker`, `spec` the fake runtime's scenario and `stepMs` the
   * time between two requests.
   */
  constructor(code, spec = {}, { stepMs = 250 } = {}) {
    this.events = [];
    this.outgoing = [];
    this.now = 0;
    this.stepMs = stepMs;
    this.timers = new Map();
    this.nextTimer = 1;
    this.listeners = new Map();
    this.dense = false;
    const call = (fn, args) => this.events.push({ t: 'call', fn, args: canonical(args) });
    const sandbox = {
      console,
      location: { origin: ORIGIN },
      navigator: {
        gpu: {
          requestAdapter: async () => {
            call('navigator.gpu.requestAdapter', []);
            return ADAPTER;
          },
        },
      },
      performance: { now: () => this.now },
      setTimeout: (callback, delay = 0) => {
        const id = this.nextTimer++;
        this.timers.set(id, { id, callback, delay, at: this.now + Math.max(0, delay) });
        return id;
      },
      clearTimeout: (id) => this.timers.delete(id),
      postMessage: (message, transfer) => {
        // As `postMessage` does: a structured clone, and the transfer list detaches its buffers.
        const data = structuredClone(message, { transfer: transfer ?? [] });
        this.outgoing.push(data);
        this.events.push({
          t: 'out',
          data: encodeOut(data),
          ...(transfer && transfer.length ? { transfer: transfer.length } : {}),
        });
      },
      addEventListener: (type, listener) =>
        this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]),
      require: (url) => {
        if (url === ORIGIN + '/runtime/gpu.js' || url === ORIGIN + '/runtime/internal/gpu.js') {
          call('import', [url.slice(ORIGIN.length)]);
          return this.runtime.module;
        }
        throw new Error('The worker imported an unknown module: ' + url);
      },
      fetch: async (url) => {
        call('fetch', [String(url)]);
        return { ok: true, arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer };
      },
      URL,
      TextEncoder,
      TextDecoder,
      AbortController,
      AbortSignal,
      DOMException,
      ReadableStream,
      CompressionStream,
      Response,
      Blob,
      crypto: globalThis.crypto,
      queueMicrotask,
    };
    sandbox.self = sandbox;
    this.sandbox = sandbox;
    this.context = vm.createContext(sandbox);
    const factory = vm.runInContext('(' + createFakeRuntime.toString() + ')', this.context);
    this.runtime = factory(spec, call);
    this.copy = vm.runInContext(COPY, this.context);
    vm.runInContext(code, this.context, { filename: 'sfm.worker.js' });
  }

  /** Move the clock to `target`, firing every timer that comes due on the way, in order. */
  async advance(target) {
    for (;;) {
      let next;
      for (const timer of this.timers.values())
        if (timer.at <= target && (!next || timer.at < next.at)) next = timer;
      if (!next) break;
      this.timers.delete(next.id);
      this.now = Math.max(this.now, next.at);
      this.events.push({ t: 'timer', at: this.now, delay: next.delay });
      next.callback();
      await this.idle();
    }
    this.now = Math.max(this.now, target);
  }

  /** Wait until `until` holds for a message posted since `from`, and the worker is quiet. */
  async idle(until, from = this.outgoing.length) {
    const deadline = Date.now() + 30000;
    let quiet = 0,
      seen = this.events.length;
    for (;;) {
      await new Promise((resolve) => setImmediate(resolve));
      if (this.events.length !== seen) {
        seen = this.events.length;
        quiet = 0;
        continue;
      }
      if (!until || this.outgoing.slice(from).some(until)) {
        if (++quiet >= 3) return;
      } else if (Date.now() > deadline)
        throw new Error('The worker did not answer:\n' + jsonLines(this.events.slice(-5)));
    }
  }

  /**
   * Send one message as the page does, at `at` ms or one step after the last request. The call
   * returns when the page would see the request end; with `wait: false`, when the worker is quiet.
   */
  async send(message, { at, wait = true } = {}) {
    await this.advance(at ?? this.now + this.stepMs);
    if (message.type === 'start' || message.type === 'multiStart')
      this.dense = message.dense === true;
    this.events.push({ t: 'in', at: this.now, data: canonical(message) });
    const from = this.outgoing.length;
    this.sandbox.onmessage({ data: this.copy(message) });
    await this.idle(wait ? ends(message, this.dense) : undefined, from);
  }

  /** Lose device `index` (in the order of `ShadeDevice.open`) while the worker is idle. */
  async lose(index, info = { reason: 'destroyed', message: 'synthetic idle loss' }) {
    this.events.push({ t: 'lose', at: this.now, device: index });
    this.runtime.control.lose(index, info);
    await this.idle();
  }

  /** Let a dense MVS that holds at `spec.denseHold` go on. */
  async release() {
    this.events.push({ t: 'release', at: this.now });
    this.runtime.control.release();
    await this.idle();
  }

  /** The transcript: every event as one JSON line, and the timers still waiting at the end. */
  transcript() {
    return jsonLines([
      ...this.events,
      { t: 'end', at: this.now, timers: [...this.timers.values()].map((t) => t.delay) },
    ]);
  }
}

/** The `start` message as the page writes it (`SfmLab.tsx`, `runSfM`), with `extra` on top. */
export const pageStart = (id, extra = {}) => ({
  type: 'start',
  id,
  focalRatio: 0.8,
  focalSearch: false,
  profile: 'video',
  bounded: false,
  fp64: undefined,
  budget: undefined,
  dense: false,
  denseGpuTimestamps: undefined,
  denseClean: false,
  denseFusion: 'tsdf',
  denseQuality: {},
  denseMesh: false,
  gate: 'legacy',
  firstOctave: 0,
  guided: false,
  clahe: false,
  features: 'sift',
  guidedKernel: 'scalar',
  orbTuning: {},
  klt: false,
  kltSetup: undefined,
  analysisEdge: 640,
  ...extra,
});

/** A `frame` message of 128 × 96 pixels at `time`; its content moved by `spec.shift[id]`. */
export const pageFrame = (job, id, time, spec, { source = SCENE, ...extra } = {}) => ({
  type: 'frame',
  id: job,
  frame: {
    id,
    width: 128,
    height: 96,
    time,
    pixels: framePixels(id, spec.shift?.[id] ?? [id * 4, 0], source),
    ...extra,
  },
});

/** The `multiStart` message as the page writes it (`SfmLab.tsx`, `runMultiVideo`). */
export const pageMultiStart = (id, extra = {}) => ({
  type: 'multiStart',
  id,
  query: { canonical: true, maxKeyframesPerVideo: 60, maxFeatures: undefined },
  analysisEdge: 640,
  firstOctave: 0,
  clahe: false,
  budget: undefined,
  fp64: undefined,
  dense: false,
  denseGpuTimestamps: undefined,
  denseClean: false,
  denseFusion: 'tsdf',
  denseQuality: {},
  denseMesh: false,
  ...extra,
});

/**
 * Feed one video as the page does: `multiVideo`, a `multiFrame` per frame and `multiEndVideo`.
 * `frames` are `[time, content, shift, source?]`; the last one is marked `last`.
 */
export async function pageVideo(h, id, videoId, label, frames) {
  await h.send({
    type: 'multiVideo',
    id,
    video: { videoId, label, durationSeconds: 2, focalRatio: 0.8 },
  });
  for (const [index, [time, content, shift, source = SCENE]] of frames.entries())
    await h.send({
      type: 'multiFrame',
      id,
      videoId,
      frame: {
        id: index,
        width: 128,
        height: 96,
        time,
        pixels: framePixels(content, shift, source),
        sourceWidth: 1280,
        sourceHeight: 960,
        image: `data:image/jpeg;base64,${label ?? 'upload'}${index}`,
        ...(index === frames.length - 1 ? { last: true } : {}),
      },
    });
  await h.send({ type: 'multiEndVideo', id, videoId });
}

/** The video ids of the multi-video cases. */
export const VIDEOS = ['a', 'b', 'c'].map((letter) => letter.repeat(16));

/** Three videos: the first two share a component, the third stands alone. */
export async function pageThreeVideos(h, id) {
  const [a, b, c] = VIDEOS;
  await pageVideo(h, id, a, 'sample-a', [
    [0, 0, [0, 0]],
    [0.1, 1, [0, 0]],
    [0.6, 2, [20, 0]],
    [1, 3, [22, 0]],
  ]);
  await pageVideo(h, id, b, null, [
    [0, 10, [0, 0]],
    // The other scene gives a weak peak: the job has no legacy rule and keeps the frame.
    [0.5, 11, [2, 0], OTHER],
    [1.6, 12, [4, 0]],
    [1.8, 13, [30, 0]],
  ]);
  await pageVideo(h, id, c, 'sample-c', [
    [0, 20, [0, 0]],
    [0.4, 21, [10, 0]],
    [1.2, 22, [24, 0]],
  ]);
}
