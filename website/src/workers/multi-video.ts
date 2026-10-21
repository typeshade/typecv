/// <reference lib="webworker" />
/**
 * The `multiVideo` job of the SfM worker (`docs/multi-video.md`, 6.2): the messages and the device.
 * `sfm.worker.ts` forwards the messages whose type starts with `multi` to `handleMultiVideo`. The
 * job itself, which a unit test runs in Node, is `lib/multi-video-job.ts`.
 *
 * The job has its own runtime entry, `/runtime/internal/gpu.js`, and its own device. The single
 * video job and its device are not touched. A run closes its device when it ends.
 */
import { MultiVideoJob, type JobApi, type JobFrame, type JobVideo } from '../lib/multi-video-job';
import { splitModel } from '../lib/model-chunks';
import { adapterFacts } from '../lib/adapter-info';
import type { MultiVideoQuery } from '../lib/multi-video';

export type MultiRequest =
  | {
      type: 'multiStart';
      id: number;
      query: MultiVideoQuery;
      analysisEdge: number;
      firstOctave: -1 | 0;
      clahe?: boolean;
      budget?: number;
      fp64?: 'float' | 'integer';
    }
  | { type: 'multiVideo'; id: number; video: JobVideo }
  | { type: 'multiFrame'; id: number; videoId: string; frame: JobFrame }
  | { type: 'multiEndVideo'; id: number; videoId: string }
  | { type: 'multiFinish'; id: number };

export const isMultiRequest = (message: { type: string }): message is MultiRequest =>
  message.type.startsWith('multi');

/**
 * The mapper's controls: the numbers of `LAB_OPTIONS` in `sfm.worker.ts` that matter to a map.
 * The WebGPU bundle adjustment has no per-group focal kernel (Amendment 4 of the record), so the
 * run shares one focal.
 */
const MAPPER = {
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
};
const VOCABULARY = '/runtime/vocabulary/rootsift-32-v1.bin';
const VOCABULARY_WORDS = 32;

let apiPromise: Promise<any> | undefined;
let active: { id: number; job: MultiVideoJob; device: any } | undefined;
let lastProgress = 0;

async function adapterInfo() {
  const requested = await navigator.gpu?.requestAdapter();
  if (!requested) return null;
  return adapterFacts(requested);
}

async function start(message: Extract<MultiRequest, { type: 'multiStart' }>) {
  if (active) throw new Error('Another multi-video job is already running.');
  apiPromise ??= import(
    /* @vite-ignore */ new URL('/runtime/internal/gpu.js', self.location.origin).href
  );
  const api = await apiPromise;
  const response = await fetch(VOCABULARY);
  if (!response.ok) throw new Error('Could not load the vocabulary ' + VOCABULARY);
  const vocabulary = api.parseVocabulary(
    new Uint8Array(await response.arrayBuffer()),
    VOCABULARY_WORDS,
  );
  const scale =
    ((message.analysisEdge * message.analysisEdge) / (640 * 640)) *
    (message.firstOctave === -1 ? 4 : 1);
  const id = message.id;
  const device = await api.ShadeDevice.open({
    backend: 'webgpu',
    fp64Flavor: message.fp64 ?? 'auto',
    timeBudgetMs: message.budget,
    workingBytes: Math.round(64 * 1024 ** 2 * scale),
    poolBytes: Math.round(32 * 1024 ** 2 * scale),
    solverCheckpoints: true,
    onProgress: ({ kernel, phase }: { kernel: string; phase: string }) => {
      if (performance.now() - lastProgress < 100) return;
      lastProgress = performance.now();
      self.postMessage({ id, type: 'progress', detail: kernel + ' (' + phase + ')' });
    },
  });
  device.lost?.then((info: any) => {
    if (active?.device !== device) return;
    self.postMessage({
      id,
      type: 'error',
      message:
        'WebGPU device lost during the multi-video job: ' + info.message + ' [' + info.reason + ']',
    });
  });
  const maxFeatures = message.query.maxFeatures ?? Math.min(16384, Math.round(4096 * scale));
  const job = new MultiVideoJob(
    api as JobApi,
    device,
    vocabulary,
    {
      query: message.query,
      maxFeatures,
      firstOctave: message.firstOctave,
      clahe: message.clahe,
      mapper: MAPPER,
      graph: { verification: MAPPER },
    },
    (out) => {
      for (const part of splitModel(out as any)) self.postMessage({ id, ...part });
    },
  );
  active = { id, job, device };
  self.postMessage({ id, type: 'multiReady', adapter: await adapterInfo(), maxFeatures });
}

async function close() {
  const current = active;
  active = undefined;
  await current?.device.dispose().catch(() => {});
}

/** Handle one `multi*` message. Errors go to the page as `error`, and the run's device closes. */
export async function handleMultiVideo(message: MultiRequest): Promise<void> {
  try {
    if (message.type === 'multiStart') return await start(message);
    if (active?.id !== message.id) return;
    const { job } = active;
    if (message.type === 'multiVideo') {
      job.video(message.video);
      self.postMessage({ id: message.id, type: 'videoStarted', videoId: message.video.videoId });
    } else if (message.type === 'multiFrame') await job.frame(message.videoId, message.frame);
    else if (message.type === 'multiEndVideo') job.endVideo(message.videoId);
    else {
      await job.finish();
      await close();
    }
  } catch (error) {
    const text = error instanceof Error ? error.stack || String(error) : String(error);
    // What the device timed before the failure reaches the failure record.
    self.postMessage({ id: message.id, type: 'error', message: text, ...active?.job.gpuFields() });
    await close();
  }
}

/** The page cancelled or reloaded: free the device. */
export const closeMultiVideo = close;
