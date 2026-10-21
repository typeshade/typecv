/**
 * The input side of a multi-video run (`docs/multi-video.md`, 2.1 and 6.1): the `videoId` of an
 * uploaded file or a sample, the duration of a file and the frames of a sample. The functions need
 * a browser (`Blob`, `fetch`, a `<video>` element), so the lab calls them and a test reads the
 * rest of the run through `multi-video-job.ts`.
 */
import { videoIdOf } from '../../../src/reconstruction/identity.ts';
import type { SampleEntry } from './sample-catalog.ts';

/** The `videoId` of an uploaded file: size, first and last MiB (2.4). */
export const fileVideoId = (file: File): Promise<string> => videoIdOf(file);

/**
 * The `videoId` of a catalog sample. A sample has no video file in the browser: its content is
 * the frame files, so the id is the hash of their concatenation under the same rule as a file.
 */
export const sampleVideoId = (frames: readonly Blob[]): Promise<string> =>
  videoIdOf(new Blob([...frames]));

/** The duration of a video file in seconds, or 0 when the browser cannot read it in time. */
export function probeDuration(file: File, timeoutMs = 4000): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    const finish = (seconds: number) => {
      window.clearTimeout(timer);
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(seconds) ? seconds : 0);
    };
    const timer = window.setTimeout(() => finish(0), timeoutMs);
    video.preload = 'metadata';
    video.onloadedmetadata = () => finish(video.duration);
    video.onerror = () => finish(0);
    video.src = url;
  });
}

/** The frame files of a video sample, fetched in order. */
export async function fetchSampleFrames(
  entry: SampleEntry,
  signal: AbortSignal,
  onProgress?: (done: number, total: number) => void,
): Promise<{ blob: Blob; url: string; time: number }[]> {
  const times = entry.video?.times ?? [];
  if (times.length !== entry.frames) throw new Error('The sample lists no frame times.');
  const out: { blob: Blob; url: string; time: number }[] = [];
  for (let i = 0; i < times.length; i++) {
    signal.throwIfAborted();
    const url = entry.dir + 'frame-' + String(i).padStart(2, '0') + '.jpg';
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error('Could not load sample frame ' + i + ' of ' + entry.id);
    out.push({ blob: await response.blob(), url, time: times[i]! });
    onProgress?.(i + 1, times.length);
  }
  return out;
}
