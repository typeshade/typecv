/**
 * The SIFT front end's quality settings for a lab run, from the analysis edge and the page's
 * query string. Measured on 2026-10-20 against COLMAP at 640 px (pull requests #84 and #85):
 *
 * - The first octave of -1 upsamples each frame 2× before the pyramid: three times the
 *   landmarks, gerrard-hall's median rotation error 0.278° → 0.019°, three to six times the
 *   wall time on a software GPU. It is on at 640 px and off above, where a 1280 px frame
 *   would build a 2560 px pyramid.
 * - Guided matching matches each pair again along its epipolar lines: 16 to 36 % more
 *   landmarks for 12 to 20 % more time. It is on.
 *
 * - CLAHE (contrast-limited adaptive histogram equalization of the gray image before the
 *   pyramid) is off. `?clahe=1` turns it on, for bright low-contrast frames.
 *
 * `?octave=-1|0` and `?guided=1|0` override either setting, for a measurement.
 */
export type SiftQuality = {
  readonly firstOctave: -1 | 0;
  readonly guided: boolean;
  readonly clahe: boolean;
};

/** The longest analysis edge at which octave -1 is the default. */
export const UPSAMPLED_EDGE_LIMIT = 640;

export function siftQuality(analysisEdge: number, search = ''): SiftQuality {
  const query = new URLSearchParams(search);
  const octave = query.get('octave'),
    guided = query.get('guided');
  return {
    firstOctave:
      octave === '-1' ? -1 : octave === '0' ? 0 : analysisEdge <= UPSAMPLED_EDGE_LIMIT ? -1 : 0,
    guided: guided === '0' ? false : guided === '1' ? true : true,
    clahe: query.get('clahe') === '1',
  };
}

/**
 * The feature front end of a lab run from the page's query string: `?features=orb` runs the ORB
 * front end (FAST corners, rBRIEF descriptors, Hamming matching); anything else runs SIFT, the
 * default. The ORB front end ignores the SIFT settings above.
 */
export type FeatureFrontEnd = 'sift' | 'orb';

export function featureFrontEnd(search = ''): FeatureFrontEnd {
  return new URLSearchParams(search).get('features') === 'orb' ? 'orb' : 'sift';
}

/**
 * Measurement overrides of the ORB front end from the page's query string: `?orbFeatures=N`
 * (features per frame, 1 to 8000), `?orbRatio=R` (ratio test, above 0.1 and below 1) and
 * `?orbDistance=D` (absolute Hamming bound, 1 to 256). A missing or invalid value keeps the
 * lab's ORB default.
 */
export type OrbTuning = {
  readonly maxFeatures?: number;
  readonly ratio?: number;
  readonly maxDistance?: number;
};

export function orbTuning(search = ''): OrbTuning {
  const query = new URLSearchParams(search);
  const number = (key: string) => (query.has(key) ? Number(query.get(key)) : NaN);
  const features = number('orbFeatures'),
    ratio = number('orbRatio'),
    distance = number('orbDistance');
  return {
    ...(Number.isInteger(features) && features >= 1 && features <= 8000
      ? { maxFeatures: features }
      : {}),
    ...(ratio > 0.1 && ratio < 1 ? { ratio } : {}),
    ...(Number.isInteger(distance) && distance >= 1 && distance <= 256
      ? { maxDistance: distance }
      : {}),
  };
}

/**
 * KLT tracking for an uploaded video (`?klt=1`, `docs/klt-tracking.md` 5.2 and 6): the worker tracks
 * points from frame to frame and joins them to the SIFT keypoints of each keyframe. Off unless the
 * query says `klt=1`; `?klt=0` is the default spelled out. A photograph or sample input never uses it.
 */
export function kltTracking(search = ''): boolean {
  return new URLSearchParams(search).get('klt') === '1';
}
