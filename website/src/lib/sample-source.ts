/** Source video metadata and selection for the lab's video-only examples.
 * JPEG thumbnails are previews; analysis always decodes the original video by byte range.
 * Historical run records may still name the earlier frame-file input.
 */
import type { SampleEntry } from './sample-catalog.ts';

/** The input a run took its frames from: the stored video, or the JPEG frame files. */
export type VideoSource = 'source' | 'frames';

/** The URL of a sample's source video, or undefined when the sample stores none. */
export function sourceUrl(entry: SampleEntry): string | undefined {
  return entry.source ? entry.dir + entry.source.file : undefined;
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

/** Require a real, decodable video before selecting a sample. No frame-file fallback. */
export async function requireSampleVideo(
  entry: SampleEntry,
  signal: AbortSignal,
  canDecode: (url: string, signal: AbortSignal) => Promise<boolean>,
): Promise<string> {
  signal.throwIfAborted();
  const url = sourceUrl(entry);
  if (entry.kind !== 'video' || !url || !entry.video || !entry.source?.videoId)
    throw new Error('This sample has no source video. Choose a video sample.');
  if (!(await canDecode(url, signal)))
    throw new Error(
      'Cannot read or decode the sample source video. Choose a supported video or try another browser.',
    );
  signal.throwIfAborted();
  return url;
}

/** Actual decoded dimensions, using the same no-upscaling rule as the video decoder. */
export function sampleAnalysisSize(entry: SampleEntry, edge: number): [number, number] {
  const source = entry.source;
  if (!source || !(source.width > 0 && source.height > 0))
    throw new Error('The sample has no source dimensions.');
  const scale = Math.min(1, edge / Math.max(source.width, source.height));
  return [
    Math.max(1, Math.round(source.width * scale)),
    Math.max(1, Math.round(source.height * scale)),
  ];
}
