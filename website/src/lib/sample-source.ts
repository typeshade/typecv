/**
 * Where a video sample's analysis frames come from. A remote video sample stores its source
 * video unchanged in the R2 bucket (`/sample/<id>/source.mp4`, `scripts/lab-remote-samples.mjs`)
 * next to 640 × 360 JPEG frames. The lab decodes the source video by byte range at the selected
 * analysis edge, the way it decodes an uploaded video, and uses the JPEG frames when the browser
 * cannot decode the codec (`sourceDecodable` in `sfm-video.ts`) or the source does not answer.
 *
 * This module has no DOM and no decoder, so a unit test runs it in Node.
 */
import type { SampleEntry } from './sample-catalog.ts';

/** The input a run took its frames from: the stored video, or the JPEG frame files. */
export type VideoSource = 'source' | 'frames';

/** The URL of a sample's source video, or undefined when the sample stores none. */
export function sourceUrl(entry: SampleEntry): string | undefined {
  return entry.source ? entry.dir + entry.source.file : undefined;
}

/**
 * The input kind of a sample: the stored video when the sample has one and the browser decodes
 * it, the frame files otherwise. A sample that stores no source video has no kind to record
 * (undefined).
 */
export function videoSourceOf(entry: SampleEntry, decodable: boolean): VideoSource | undefined {
  if (!entry.source) return undefined;
  return decodable ? 'source' : 'frames';
}

/** One value for the several videos of a run: `mixed` when the videos took different inputs. */
export function runVideoSource(
  sources: readonly (VideoSource | undefined)[],
): VideoSource | 'mixed' | undefined {
  const known = sources.filter((s): s is VideoSource => s !== undefined);
  if (known.length === 0) return undefined;
  return known.every((s) => s === known[0]) ? known[0] : 'mixed';
}

/**
 * The `videoId` of a sample row (`docs/multi-video.md`, Amendment 7). A sample that stores a
 * source video has the `videoId` of that file, which the publish script computed with the rule
 * of 2.4 and recorded in the catalog: it is the same for the source input and the frame input,
 * and the same as the id of the file uploaded from the same video. A sample without one has the
 * hash of its frame files, as in Amendment 6.
 */
export function catalogVideoId(entry: SampleEntry): string | undefined {
  return entry.source?.videoId;
}
