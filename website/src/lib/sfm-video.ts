import { sampleTimestamps, VIDEO_SAMPLE_FPS } from './video-sampling';
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny';
import {
  DECODE_WINDOW,
  chooseDecodePath,
  detectDecodeEnvironment,
  type DecodePath,
  type DecodeRequest,
  type DecodeResponse,
} from './decode-protocol';

export type DecodedFrame = {
  time: number;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  pixels: Uint8ClampedArray;
  image: string;
};

/** Demux and decode actual presentation frames, with bounded decoder/canvas buffering. */
/** The analysis edges the lab offers: the long side of a frame in pixels. */
export const ANALYSIS_EDGES = [640, 1280] as const;
export type AnalysisEdge = (typeof ANALYSIS_EDGES)[number];

/** Reads the pixels and a JPEG data URL out of a decoded canvas (a DOM canvas or an offscreen one). */
export type FrameReadback = (
  canvas: HTMLCanvasElement | OffscreenCanvas,
  width: number,
  height: number,
) =>
  | { pixels: Uint8ClampedArray; image: string }
  | Promise<{ pixels: Uint8ClampedArray; image: string }>;

/**
 * The decoded canvas belongs to mediabunny, which creates its 2D context without
 * `willReadFrequently`, so reading it with `getImageData` is the slow GPU readback the browser
 * warns about. One scratch canvas per stream, with `willReadFrequently: true`, takes a copy
 * (`drawImage`) and is read instead.
 */
function createReadback(
  makeCanvas: (width: number, height: number) => HTMLCanvasElement | OffscreenCanvas,
  encode: (canvas: HTMLCanvasElement | OffscreenCanvas) => string | Promise<string>,
): FrameReadback {
  let scratch: HTMLCanvasElement | OffscreenCanvas | undefined;
  let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  return async (canvas, width, height) => {
    if (!scratch) {
      scratch = makeCanvas(width, height);
      ctx = scratch.getContext('2d', { willReadFrequently: true }) as typeof ctx;
    }
    ctx!.drawImage(canvas as CanvasImageSource, 0, 0, width, height);
    return { pixels: ctx!.getImageData(0, 0, width, height).data, image: await encode(scratch) };
  };
}

/** The readback of the page path (an `HTMLCanvasElement`). */
export const createPageReadback = (): FrameReadback =>
  createReadback(
    (width, height) => Object.assign(document.createElement('canvas'), { width, height }),
    (canvas) => (canvas as HTMLCanvasElement).toDataURL('image/jpeg', 0.8),
  );

/** The readback of the worker path (an `OffscreenCanvas` and a JPEG blob read as a data URL). */
export const createWorkerReadback = (): FrameReadback =>
  createReadback(
    (width, height) => new OffscreenCanvas(width, height),
    async (canvas) => {
      const blob = await (canvas as OffscreenCanvas).convertToBlob({
        type: 'image/jpeg',
        quality: 0.8,
      });
      return await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });
    },
  );

/** Demux and decode on the page's main thread. */
export function decodeVideoFrames(
  file: File,
  signal: AbortSignal,
  at?: number,
  edge: AnalysisEdge = 640,
  /** Analysis frames per second. Frames between the targets are never decoded or scaled. */
  fps: number = VIDEO_SAMPLE_FPS,
): AsyncGenerator<DecodedFrame> {
  return decodeFrameStream(file, signal, at, edge, fps, createPageReadback());
}

/** The decode and the sampling schedule, shared by the page and the decode worker. */
export async function* decodeFrameStream(
  file: File,
  signal: AbortSignal,
  at: number | undefined,
  edge: AnalysisEdge | number,
  fps: number,
  readback: FrameReadback,
): AsyncGenerator<DecodedFrame> {
  signal.throwIfAborted();
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const abort = () => input.dispose();
  signal.addEventListener('abort', abort, { once: true });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode())) {
      throw new Error(
        'Frame decoding is unavailable for this codec/browser. Use a supported MP4/WebM or prepared frame JSON.',
      );
    }
    const [sourceWidth, sourceHeight] = await Promise.all([
      track.getDisplayWidth(),
      track.getDisplayHeight(),
    ]);
    const scale = Math.min(1, edge / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    const sink = new CanvasSink(track, { width, height, fit: 'fill', poolSize: 2 });
    let targets: number[] | undefined;
    if (at === undefined) {
      const [first, end] = await Promise.all([track.getFirstTimestamp(), track.computeDuration()]);
      targets = sampleTimestamps(first, end, fps);
    }
    // A target timestamp selects the frame presented at it; the decoder seeks between targets.
    const frames = sink.canvasesAtTimestamps(targets ?? [at! + 1e-7]);
    let previous = -Infinity;
    for await (const frame of frames) {
      signal.throwIfAborted();
      // A variable-rate video can present one frame at two targets: analyse it once.
      if (!frame || frame.timestamp <= previous) continue;
      previous = frame.timestamp;
      const { pixels, image } = await readback(frame.canvas, width, height);
      yield { time: frame.timestamp, width, height, sourceWidth, sourceHeight, pixels, image };
    }
  } finally {
    signal.removeEventListener('abort', abort);
    input.dispose();
  }
}

/** The worker could not start or failed before its first frame: the caller decodes on the page. */
export class DecodeWorkerUnavailable extends Error {}

let decodeId = 0;

/** Decode in a dedicated worker. Throws `DecodeWorkerUnavailable` if it fails before a frame. */
export async function* decodeVideoFramesInWorker(
  file: File,
  signal: AbortSignal,
  edge: AnalysisEdge,
  fps: number,
): AsyncGenerator<DecodedFrame> {
  signal.throwIfAborted();
  const id = ++decodeId;
  const worker = new Worker(new URL('../workers/decode.worker.ts', import.meta.url), {
    type: 'module',
  });
  const queue: DecodeResponse[] = [];
  let wake: (() => void) | undefined;
  const push = (message: DecodeResponse) => {
    queue.push(message);
    wake?.();
  };
  worker.onmessage = (event: MessageEvent<DecodeResponse>) => {
    if (event.data.id === id) push(event.data);
  };
  worker.onerror = (event) =>
    push({
      type: 'error',
      id,
      message: 'Decode worker error: ' + (event.message || '(no message)'),
    });
  const send = (request: DecodeRequest) => worker.postMessage(request);
  const abort = () => {
    send({ type: 'cancel', id });
    push({ type: 'done', id });
  };
  signal.addEventListener('abort', abort, { once: true });
  let produced = 0;
  try {
    send({ type: 'decode', id, file, edge, fps, window: DECODE_WINDOW });
    for (;;) {
      while (queue.length === 0) await new Promise<void>((resolve) => (wake = resolve));
      const message = queue.shift()!;
      signal.throwIfAborted();
      if (message.type === 'done') return;
      if (message.type === 'error') {
        throw produced === 0
          ? new DecodeWorkerUnavailable(message.message)
          : new Error(message.message);
      }
      produced++;
      const { pixels, ...rest } = message.frame;
      yield { ...rest, pixels: new Uint8ClampedArray(pixels) };
      send({ type: 'next', id });
    }
  } finally {
    signal.removeEventListener('abort', abort);
    worker.terminate();
  }
}

/**
 * Decode a whole uploaded video for a run. The worker path is chosen by feature detection
 * (`chooseDecodePath`); a worker that fails before its first frame falls back to the page.
 * `info.path` names the path that produced the frames, for the run record.
 */
export async function* decodeUploadedFrames(
  file: File,
  signal: AbortSignal,
  edge: AnalysisEdge,
  fps: number,
  info: { path: DecodePath },
  search = '',
): AsyncGenerator<DecodedFrame> {
  info.path = chooseDecodePath(detectDecodeEnvironment(), search);
  if (info.path === 'worker') {
    try {
      yield* decodeVideoFramesInWorker(file, signal, edge, fps);
      return;
    } catch (error) {
      if (!(error instanceof DecodeWorkerUnavailable)) throw error;
      info.path = 'main-thread';
    }
  }
  yield* decodeVideoFrames(file, signal, undefined, edge, fps);
}
