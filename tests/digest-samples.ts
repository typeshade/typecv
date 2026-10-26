import { readFileSync } from 'node:fs';
import type { Camera } from '../src/geometry/camera.ts';
import type { PairMatches } from '../src/reconstruction/types.ts';

/**
 * The four lab samples that scripts/measure-focal.mjs reads, read as it reads them, and a frozen
 * copy of its `LAB` controls: the inputs of the characterization digests of
 * docs/library-structure.md, section 5, phase 0. The digests keep their own copy so that they
 * characterize the library alone: phase 2 moves the lab preset, and the compare job of CI runs
 * these files against an older library. tests/digests/measure-focal.test.ts holds the copy equal
 * to the script's.
 */
export const SAMPLES = ['south-building', 'gerrard-hall', 'person-hall', 'graham-hall'] as const;
export type Sample = (typeof SAMPLES)[number];

/** `LAB` of scripts/measure-focal.mjs (`LAB_OPTIONS` of website/src/workers/sfm.worker.ts) on 48e2711. */
export const LAB = Object.freeze({
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
});

export interface SampleFrame {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  readonly truth: Camera;
  readonly keypoints: Float64Array;
}

/** The sample's frames as mapper input: COLMAP's SIFT keypoints and raw matches (`loadSample` of the script). */
export function loadSample(name: Sample): { frames: SampleFrame[]; pairs: PairMatches[] } {
  const dir = new URL(`./fixtures/lab-images/${name}/`, import.meta.url);
  const sample = JSON.parse(readFileSync(new URL('sample.json', dir), 'utf8'));
  const bytes = readFileSync(new URL('colmap-sift.bin', dir));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const keypoints = new Float32Array(buffer, 0, sample.features.keypointBytes / 4);
  const matches = new Uint16Array(buffer, sample.features.keypointBytes);
  const blocks = new Map<number, { offset: number; count: number }>(
    sample.features.keypoints.map((b: { id: number; offset: number; count: number }) => [b.id, b]),
  );
  const frames = sample.frames.map(
    (frame: { id: number; width: number; height: number; camera: Camera }) => {
      const block = blocks.get(frame.id)!;
      return {
        id: frame.id,
        width: frame.width,
        height: frame.height,
        truth: frame.camera,
        keypoints: Float64Array.from(
          keypoints.subarray(block.offset, block.offset + 2 * block.count),
        ),
      };
    },
  );
  const pairs = sample.features.pairs.map(
    (p: { first: number; second: number; offset: number; count: number }) => ({
      first: p.first,
      second: p.second,
      matches: Uint32Array.from(matches.subarray(p.offset, p.offset + 2 * p.count)),
    }),
  );
  return { frames, pairs };
}

/** Every view at fx = fy = ratio × width (`camerasAt` of the script). */
export function camerasAt(frames: readonly SampleFrame[], ratio: number, distortion: boolean) {
  return frames.map((f) => ({
    id: f.id,
    keypoints: f.keypoints,
    camera: {
      fx: ratio * f.width,
      fy: ratio * f.width,
      cx: f.truth.cx,
      cy: f.truth.cy,
      k1: distortion ? f.truth.k1 : 0,
      k2: distortion ? f.truth.k2 : 0,
    },
  }));
}
