/**
 * Pre-sampling of an uploaded video: the lab analyses frames on a fixed time grid instead of
 * every decoded frame. The grid comes from the keyframe gate's own constants (`LAB_OPTIONS` of
 * `lib/lab-options.ts`):
 *
 * - A keyframe is forced after one second at the latest, so a grid finer than a few frames per
 *   second adds frames that no gap rule needs.
 * - The legacy motion gate needs 8 px of median displacement at 640 px (the phase gate: 4 % of the
 *   frame, `lib/keyframe-gate.ts`). At 4 frames per second a
 *   camera moving at 32 px/s (a slow walk) still reaches it every frame, and a faster camera
 *   reaches it sooner than the grid, so the grid, not the gate, then limits the density.
 *
 * Evidence: lab record 20261020121308655-a6fb6f3b decoded 566 frames of a 23.6 s video and
 * kept 27 keyframes (about 1 per 21 frames). At 4 frames per second the worker sees 95 frames.
 */

/** Default analysis frames per second for a video (one frame every 0.25 s). */
export const VIDEO_SAMPLE_FPS = 4;

/** The largest rate the `?fps=` override accepts: a measurement can ask for dense sampling. */
export const MAX_VIDEO_SAMPLE_FPS = 120;

/** The sampling rate from the page's query string (`?fps=<number>`), or the default. */
export function videoSampleFps(search = '', fallback = VIDEO_SAMPLE_FPS): number {
  const raw = new URLSearchParams(search).get('fps');
  const value = raw === null || raw.trim() === '' ? NaN : Number(raw);
  return Number.isFinite(value) && value > 0 && value <= MAX_VIDEO_SAMPLE_FPS ? value : fallback;
}

/**
 * The target timestamps in seconds, ascending, for a video that starts at `start` and lasts
 * until `end`. The grid is `start + i / fps` below `end`. The last frame of the video is added
 * when the grid stops more than half an interval before it, so the end of the video is always
 * covered without a near-duplicate pair.
 */
export function sampleTimestamps(start: number, end: number, fps: number): number[] {
  if (!(fps > 0) || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return Number.isFinite(start) ? [start] : [];
  }
  const step = 1 / fps;
  const times: number[] = [];
  for (let i = 0; start + i * step < end - 1e-9; i++) times.push(start + i * step);
  // The last frame starts a little before the end of the track.
  const last = end - Math.min(step / 4, 1e-3);
  if (last - times[times.length - 1] > step / 2) times.push(last);
  return times;
}
