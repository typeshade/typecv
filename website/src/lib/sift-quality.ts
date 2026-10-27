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
 * `?octave=-1|0` and `?guided=1|0` override either setting, for a measurement. `?guided=tiled`
 * keeps guided matching on and runs its tiled kernel (`guidedKernel`).
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
 * The WebGPU kernel of guided matching: scalar (`nearestFloatGuided`, one query per lane) by
 * default, tiled (`nearestFloatGuidedTiles`) with `?guided=tiled`. The matches are the same
 * bits. The tiled kernel was 3.7 to 4.9 times slower than the scalar one on SwiftShader, so it
 * is opt-in, for a measurement on a physical GPU (`docs/lab-run-records.md`).
 */
export type GuidedKernel = 'tiled' | 'scalar';

export function guidedKernel(search = ''): GuidedKernel {
  return new URLSearchParams(search).get('guided') === 'tiled' ? 'tiled' : 'scalar';
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

/**
 * The step 6 settings of a KLT run (`docs/klt-tracking.md` 3.4, 3.6 and 8, "Step 6 record"). Each one
 * is a measurement switch on top of `?klt=1`, and a run without it behaves as step 5 did.
 *
 * - `kltCapacity=<n>`: the tracks the tracker holds (1 to 8192). The lab's default is 512
 *   (`LAB_KLT_CAPACITY` of `lib/lab-options.ts`); the library's is 1024.
 * - `kltFill=1`: fill-in points at a join (3.4). The lab leaves them out by default, because they
 *   added nothing in the step 6 measurement.
 * - `kltMatching=tracks`: a pair of keyframes that the tracks cover with enough matches skips the
 *   descriptor matching and the guided matching.
 * - `kltGate=1`: with the legacy gate, the median displacement of the tracks decides the keyframe
 *   before SIFT runs.
 */
export interface KltSetup {
  capacity?: number;
  fillIn?: true;
  matching?: 'tracks';
  gate?: true;
}

export function kltSetup(search = ''): KltSetup | undefined {
  const query = new URLSearchParams(search),
    setup: KltSetup = {},
    capacity = Number(query.get('kltCapacity'));
  if (Number.isInteger(capacity) && capacity >= 1 && capacity <= 8192) setup.capacity = capacity;
  if (query.get('kltFill') === '1') setup.fillIn = true;
  if (query.get('kltMatching') === 'tracks') setup.matching = 'tracks';
  if (query.get('kltGate') === '1') setup.gate = true;
  return Object.keys(setup).length ? setup : undefined;
}
