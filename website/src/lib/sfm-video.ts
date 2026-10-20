import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny';

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

export async function* decodeVideoFrames(
  file: File,
  signal: AbortSignal,
  at?: number,
  edge: AnalysisEdge = 640,
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
    const frames = at === undefined ? sink.canvases(0) : sink.canvasesAtTimestamps([at + 1e-7]);
    for await (const frame of frames) {
      signal.throwIfAborted();
      if (!frame) continue;
      const canvas = frame.canvas as HTMLCanvasElement;
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      yield {
        time: frame.timestamp,
        width,
        height,
        sourceWidth,
        sourceHeight,
        pixels: ctx.getImageData(0, 0, width, height).data,
        image: canvas.toDataURL('image/jpeg', 0.8),
      };
    }
  } finally {
    signal.removeEventListener('abort', abort);
    input.dispose();
  }
}
