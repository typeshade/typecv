/**
 * The input side of a multi-video run (`docs/multi-video.md`, 2.1 and 6.1): the `videoId` of an
 * uploaded file and the duration of a file. The functions need
 * a browser (`Blob`, `fetch`, a `<video>` element), so the lab calls them and a test reads the
 * rest of the run through `multi-video-job.ts`.
 */
import { videoIdOf } from '../../../src/reconstruction/identity.ts';

/** The `videoId` of an uploaded file: size, first and last MiB (2.4). */
export const fileVideoId = (file: File): Promise<string> => videoIdOf(file);

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
