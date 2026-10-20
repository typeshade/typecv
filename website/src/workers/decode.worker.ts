/// <reference lib="webworker" />
import { createWorkerReadback, decodeFrameStream } from '../lib/sfm-video';
import { FrameWindow, type DecodeRequest, type DecodeResponse } from '../lib/decode-protocol';

// Decodes an uploaded video off the page's main thread (see `lib/decode-protocol.ts`).
const windows = new Map<number, FrameWindow>();
const cancelled = new Set<number>();

function send(message: DecodeResponse, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(message, transfer);
}

async function run(request: Extract<DecodeRequest, { type: 'decode' }>) {
  const { id } = request;
  const slots = new FrameWindow(request.window);
  windows.set(id, slots);
  const controller = new AbortController();
  const poll = setInterval(() => cancelled.has(id) && controller.abort(), 20);
  try {
    for await (const frame of decodeFrameStream(
      request.file,
      controller.signal,
      undefined,
      request.edge,
      request.fps,
      createWorkerReadback(),
    )) {
      await slots.take();
      if (cancelled.has(id)) break;
      const pixels = frame.pixels.buffer as ArrayBuffer;
      send({ type: 'frame', id, frame: { ...frame, pixels } }, [pixels]);
    }
    if (!cancelled.has(id)) send({ type: 'done', id });
  } catch (error) {
    if (!cancelled.has(id)) {
      send({ type: 'error', id, message: error instanceof Error ? error.message : String(error) });
    }
  } finally {
    clearInterval(poll);
    windows.delete(id);
    cancelled.delete(id);
  }
}

self.onmessage = (event: MessageEvent<DecodeRequest>) => {
  const message = event.data;
  if (message.type === 'decode') void run(message);
  else if (message.type === 'next') windows.get(message.id)?.release();
  else {
    cancelled.add(message.id);
    windows.get(message.id)?.wake();
  }
};
