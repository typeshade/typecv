/**
 * The lab preset: the mapper controls of the lab (docs/library-structure.md, section 5, phase 2,
 * and section 7, item 4). The page and both lab workers read it through
 * `website/src/lib/lab-options.ts`; the scripts that measure the lab's mapper import it here.
 * `tests/lab-preset.test.ts` holds it to its values.
 */

/** The six numbers of the lab's mapper (`LAB` of `scripts/measure-focal.mjs`). */
export const LAB_MAPPER_NUMBERS = {
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
} as const;

/**
 * The lab's mapper controls: the six numbers, and the shared focal refined for a video. A
 * photograph's profile turns the refinement off (`PHOTO_PROFILE` of the lab).
 */
export const LAB_MAPPER = {
  ...LAB_MAPPER_NUMBERS,
  refineSharedFocal: true,
} as const;

/** The window of the lab's streaming session (`IncrementalSfmSession`). */
export const LAB_WINDOW_SIZE = 32;
