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
 * `?octave=-1|0` and `?guided=1|0` override either setting, for a measurement.
 */
export type SiftQuality = { readonly firstOctave: -1 | 0; readonly guided: boolean };

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
  };
}
