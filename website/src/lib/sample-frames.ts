/**
 * The frame schedule of a video sample hosted in R2 (`website/samples/remote-videos.json`): two
 * frames per second over the whole clip, at least 24 and at most 48, evenly spaced from t = 0 to
 * a quarter second before the end. The publish script cuts the frames at these times and the
 * catalog lists the same times, so both read this one rule.
 */
export const VIDEO_SAMPLE_FPS = 2;
export const VIDEO_SAMPLE_MIN_FRAMES = 24;
export const VIDEO_SAMPLE_MAX_FRAMES = 48;

/** The frame count of a clip of `duration` seconds. */
export function videoSampleCount(duration: number): number {
  if (!(duration > 0)) throw new RangeError('A video sample needs a positive duration.');
  return Math.min(
    VIDEO_SAMPLE_MAX_FRAMES,
    Math.max(VIDEO_SAMPLE_MIN_FRAMES, Math.round(duration * VIDEO_SAMPLE_FPS)),
  );
}

/** The frame times of a clip of `duration` seconds, in seconds, rounded to 0.01 s. */
export function videoSampleTimes(duration: number): number[] {
  const count = videoSampleCount(duration);
  const end = Math.max(0, duration - 0.25);
  return Array.from({ length: count }, (_, i) => Math.round((i * end * 100) / (count - 1)) / 100);
}
