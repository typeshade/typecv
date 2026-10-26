// The inputs of the device-tier characterization cases (sfm-digest-cases.mjs), as plain data that
// a browser page can take: docs/library-structure.md, section 5, phase 0.
import { scene } from '../scene.ts';
import { refinementFixture } from '../fixtures/refinement.ts';
import { LAB, loadSample, SAMPLES } from '../digest-samples.ts';

export { SAMPLES };
/** The frames of a sample that the device tiers run: the first six, and the pairs among them. */
export const SAMPLE_FRAMES = 6;

const plainImages = (images) =>
  images.map((image) => ({
    id: image.id,
    camera: { ...image.camera },
    keypoints: [...image.keypoints],
  }));
const plainPairs = (pairs) =>
  pairs.map((pair) => ({ first: pair.first, second: pair.second, matches: [...pair.matches] }));
const scaled = (images, scale) =>
  images.map((image) => ({
    ...image,
    camera: { ...image.camera, fx: image.camera.fx * scale, fy: image.camera.fy * scale },
  }));

/** The cases on the scenes of tests/scene.ts: the option sets of the CPU-path cases and of streaming.test.mjs. */
export function sceneDeviceCases() {
  const distorted = scene({
    count: 90,
    views: 6,
    noise: 0.2,
    camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.03, k2: 0.004 },
  });
  const evicting = scene({ count: 28, views: 11, noise: 0.15, seed: 41 });
  const square = scene({
    count: 90,
    views: 6,
    noise: 0.1,
    camera: { fx: 850, fy: 850, cx: 480, cy: 270 },
  });
  const refinement = refinementFixture();
  const search = {
    ransacIterations: 128,
    bundleIterations: 4,
    finalBundleIterations: 8,
    refineSharedFocal: true,
    focalSearch: { views: 4 },
  };
  return [
    {
      name: 'IncrementalSfmSession/scene-90x6-distorted',
      kind: 'session',
      images: plainImages(distorted.inputs),
      pairs: plainPairs(distorted.pairs),
      options: { seed: 42, ransacIterations: 300 },
    },
    {
      name: 'IncrementalSfmSession/scene-28x11-window-8',
      kind: 'session',
      images: plainImages(evicting.inputs),
      pairs: plainPairs(evicting.pairs),
      options: {
        windowSize: 8,
        ransacIterations: 64,
        minInitialPoints: 8,
        minPnPInliers: 6,
        initialPair: [10, 20],
        bundleIterations: 1,
        finalBundleIterations: 3,
        historyBundleIterations: 4,
        seed: 42,
      },
    },
    {
      name: 'IncrementalSfmSession/scene-90x6-focal-1.2',
      kind: 'session',
      images: plainImages(scaled(square.inputs, 1.2)),
      pairs: plainPairs(square.pairs),
      options: search,
    },
    {
      name: 'reconstructIncrementalGpu/scene-90x6-distorted/global',
      kind: 'driver',
      images: plainImages(distorted.inputs),
      pairs: plainPairs(distorted.pairs),
      options: { seed: 42, ransacIterations: 300, initialPair: [10, 60] },
    },
    {
      name: 'reconstructIncrementalGpu/scene-90x6-focal-1.2/focal-search',
      kind: 'driver',
      images: plainImages(scaled(square.inputs, 1.2)),
      pairs: plainPairs(square.pairs),
      options: search,
    },
    {
      name: 'refineIncrementalGpu/refinement-fixture',
      kind: 'refine',
      initial: {
        images: plainImages(refinement.limited.inputs),
        pairs: plainPairs(refinement.limited.pairs),
      },
      images: plainImages(refinement.inputs),
      pairs: plainPairs(refinement.pairs),
      options: { ...refinement.controls, initialPair: [...refinement.controls.initialPair] },
    },
  ];
}

/** The session on the first frames of a sample, with the lab's controls (`LAB` of tests/digest-samples.ts) and its cameras. */
export function sampleDeviceCase(name) {
  const { frames, pairs } = loadSample(name);
  const kept = frames.slice(0, SAMPLE_FRAMES);
  const ids = new Set(kept.map((frame) => frame.id));
  return {
    name: `IncrementalSfmSession/${name}-first-${SAMPLE_FRAMES}/lab`,
    kind: 'session',
    images: kept.map((frame) => ({
      id: frame.id,
      camera: { ...frame.truth },
      keypoints: [...frame.keypoints],
    })),
    pairs: plainPairs(pairs.filter((pair) => ids.has(pair.first) && ids.has(pair.second))),
    options: { ...LAB },
  };
}
