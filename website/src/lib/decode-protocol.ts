/**
 * The protocol between the lab page and the video decode worker (`workers/decode.worker.ts`),
 * and the choice between the worker path and the main-thread path. This module has no DOM and
 * no worker globals, so a unit test runs it in Node.
 *
 * The worker decodes the uploaded video with mediabunny on an `OffscreenCanvas`, reads the
 * pixels with `getImageData` and posts each frame with its pixel buffer transferred. The page
 * keeps the preview `<video>` and the UI. The page and the worker agree on a window of frames in
 * flight: the worker never runs more than `DECODE_WINDOW` frames ahead of what the page consumed.
 */

/** The decode paths the run record names (`options.videoDecode`). */
export type DecodePath = 'worker' | 'main-thread';

/** Frames the worker may hold unconsumed on the page (each is about 1 MB at 640 × 360). */
export const DECODE_WINDOW = 2;

export type DecodeRequest =
  | {
      type: 'decode';
      id: number;
      /** An uploaded file, or the URL of a video that the server answers by byte range. */
      source: File | string;
      edge: number;
      fps: number;
      window: number;
    }
  /** The page consumed one frame: the worker may send one more. */
  | { type: 'next'; id: number }
  | { type: 'cancel'; id: number };

export type WireFrame = {
  time: number;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  /** RGBA pixels. Transferred, not copied. */
  pixels: ArrayBuffer;
  /** A JPEG data URL of the frame, for the thumbnail strip and the overlay. */
  image: string;
};

export type DecodeResponse =
  | { type: 'frame'; id: number; frame: WireFrame }
  | { type: 'done'; id: number }
  | { type: 'error'; id: number; message: string };

/** What the browser offers to a decode worker. */
export type DecodeEnvironment = {
  worker: boolean;
  offscreenCanvas: boolean;
  /** Pixel readback from an `OffscreenCanvas` 2D context. */
  offscreenContext2d: boolean;
};

/** The `?decode=main` query override: force the main-thread path, for a measurement. */
export function decodeOverride(search = ''): DecodePath | undefined {
  const value = new URLSearchParams(search).get('decode');
  return value === 'main' ? 'main-thread' : value === 'worker' ? 'worker' : undefined;
}

/**
 * Choose the decode path by feature detection. The worker path needs a module worker and an
 * `OffscreenCanvas` with a 2D context; an older Safari without them decodes on the main thread.
 * `?decode=main` forces the main-thread path. `?decode=worker` does not force a path the browser
 * cannot run.
 */
export function chooseDecodePath(environment: DecodeEnvironment, search = ''): DecodePath {
  if (decodeOverride(search) === 'main-thread') return 'main-thread';
  return environment.worker && environment.offscreenCanvas && environment.offscreenContext2d
    ? 'worker'
    : 'main-thread';
}

/** Detect the environment in a page. */
export function detectDecodeEnvironment(): DecodeEnvironment {
  const offscreenCanvas = typeof OffscreenCanvas !== 'undefined';
  let offscreenContext2d = false;
  if (offscreenCanvas) {
    try {
      offscreenContext2d = !!new OffscreenCanvas(1, 1).getContext('2d', {
        willReadFrequently: true,
      });
    } catch {
      offscreenContext2d = false;
    }
  }
  return { worker: typeof Worker !== 'undefined', offscreenCanvas, offscreenContext2d };
}

/** Flow control for the worker: `take()` waits while the window is used up. */
export class FrameWindow {
  private free: number;
  private waiter: (() => void) | undefined;
  constructor(size: number) {
    this.free = Math.max(1, Math.floor(size));
  }
  /** Wait for a free slot and take it. */
  async take(): Promise<void> {
    while (this.free <= 0) await new Promise<void>((resolve) => (this.waiter = resolve));
    this.free--;
  }
  /** The page consumed a frame: give a slot back. */
  release(): void {
    this.free++;
    this.wake();
  }
  /** Wake a waiting `take()`. */
  wake(): void {
    const waiter = this.waiter;
    this.waiter = undefined;
    waiter?.();
  }
}
