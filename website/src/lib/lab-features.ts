/**
 * The rules of the lab's feature stage (`docs/library-structure.md`, section 3.4, `features`). The
 * feature minimum of a frame: the SfM worker (`processFrame`) and the multi-video job
 * (`MultiVideoJob.frame`) call it, and each rejects the frame in its own way. The feature cap of a
 * frame: both workers derive it when they start. The colours under the keypoints of a frame, and
 * the landmark colours that the viewer draws from them: the SfM worker (`sampleColors`, `scene`)
 * and the multi-video job (`MultiVideoJob.frame`, `MultiVideoJob.scene`) each keep their own
 * variant. The worker samples floats in 0..1, holds the colours of its newest frames and keeps the
 * colour of a track for the whole run. The job samples bytes, holds them with each keyframe and
 * looks a landmark's colour up on each call. This module has no DOM and no worker globals, so a
 * unit test runs it in Node.
 */

/** A frame with fewer detected features than this is rejected ("Too few detected features"). */
export const MIN_FEATURES = 8;

/** True when a frame's count of detected features is under `MIN_FEATURES`. */
export function tooFewFeatures(count: number): boolean {
  return count < MIN_FEATURES;
}

/** The SIFT front end's limit on the features of a frame: the feature cap stops here. */
export const MAX_FEATURE_CAP = 16384;

/**
 * The feature cap of a frame at `scale` (`analysisScale` of `working-budget.ts`): `base` features
 * at 640 px, grown with the pixel count up to `MAX_FEATURE_CAP`. A base of 4096 gives 4096 at
 * 640 px and 16384 at 1280 px. The SfM worker passes the `maxFeatures` of its controls, and the
 * multi-video worker 4096 when the query sets no cap.
 */
export function featureCap(base: number, scale: number): number {
  return Math.min(MAX_FEATURE_CAP, Math.round(base * scale));
}

/** The SfM worker holds the keypoint colours of this many newest frames for the landmark colours. */
export const KEYPOINT_COLOR_FRAMES = 64;

/** What the colour samplers read of a frame: its size and its RGBA pixels. */
type ColorFrame = {
  readonly width: number;
  readonly height: number;
  readonly pixels: ArrayLike<number>;
};

/** The RGBA offset of the pixel under keypoint `i`: its position rounded and clamped to the frame. */
function keypointPixel(frame: ColorFrame, points: Float64Array, i: number): number {
  const x = Math.max(0, Math.min(frame.width - 1, Math.round(points[i * 2]!)));
  const y = Math.max(0, Math.min(frame.height - 1, Math.round(points[i * 2 + 1]!)));
  return (y * frame.width + x) * 4;
}

/** The colour under each keypoint, for the point cloud: three floats in 0..1 for each keypoint. */
export function keypointColorFloats(frame: ColorFrame, points: Float64Array): Float32Array {
  const colors = new Float32Array((points.length / 2) * 3);
  for (let i = 0; i < points.length / 2; i++) {
    const offset = keypointPixel(frame, points, i);
    for (let channel = 0; channel < 3; channel++)
      colors[i * 3 + channel] = frame.pixels[offset + channel]! / 255;
  }
  return colors;
}

/** The colour under each keypoint, for the point cloud: three bytes for each keypoint. */
export function keypointColorBytes(frame: ColorFrame, points: Float64Array): Uint8Array {
  const colors = new Uint8Array((points.length / 2) * 3);
  for (let i = 0; i < points.length / 2; i++) {
    const offset = keypointPixel(frame, points, i);
    for (let c = 0; c < 3; c++) colors[i * 3 + c] = frame.pixels[offset + c]!;
  }
  return colors;
}

/** A colour that the viewer draws: red, green and blue in 0..1. */
export type Rgb = [number, number, number];

/** The value of each channel of the grey of a landmark that no keypoint colour covers. */
export const UNCOLORED_CHANNEL = 0.7;

/**
 * The colour of a landmark: the keypoint colour under its first observation whose view holds
 * colours, else grey. `colorsOf` gives the keypoint colours of a view, three for each feature:
 * floats in 0..1 (`float`, the SfM worker's) or bytes (`byte`, the multi-video job's), which are
 * divided by 255. With a `cache`, a track's colour is looked up once and kept there. The SfM worker
 * passes one cache for the whole run, so a track keeps its colour after its frames' colours are
 * gone. The multi-video job passes none.
 */
export function landmarkColor(
  point: {
    readonly trackId: number;
    readonly observations: readonly { readonly viewId: number; readonly featureId: number }[];
  },
  colorsOf: (viewId: number) => ArrayLike<number> | undefined,
  channels: 'float' | 'byte',
  cache?: Map<number, Rgb>,
): Rgb {
  let color = cache?.get(point.trackId);
  if (!color)
    for (const observation of point.observations) {
      const colors = colorsOf(observation.viewId);
      if (colors && observation.featureId * 3 + 2 < colors.length) {
        const at = observation.featureId * 3;
        color =
          channels === 'float'
            ? [colors[at]!, colors[at + 1]!, colors[at + 2]!]
            : [colors[at]! / 255, colors[at + 1]! / 255, colors[at + 2]! / 255];
        cache?.set(point.trackId, color);
        break;
      }
    }
  return color ?? [UNCOLORED_CHANNEL, UNCOLORED_CHANNEL, UNCOLORED_CHANNEL];
}
