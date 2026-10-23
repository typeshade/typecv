import { cameraAtRatio } from '../../src/reconstruction/focal-search.ts';
import { generalVideoFixture } from './general-video.ts';
import { orbitFixture } from '../typeshade/focal-search-cases.mjs';

/** Known shared focal, independent components; poses/depths stay on the evaluator side. */
export function focalComponentsFixture(start = 1.2, { points = 48, prefixViews = 3 } = {}) {
  const rotation = generalVideoFixture('rotation'),
    orbit = orbitFixture({ views: 6, points });
  const images = [
    ...rotation.input.images.slice(0, prefixViews),
    ...orbit.images.map((view) => ({ ...view, id: view.id + 10 })),
  ].map((view) => ({
    ...view,
    camera: cameraAtRatio(view.camera, start),
  }));
  const pairs = [
    ...rotation.input.pairs.filter((pair) => pair.first < prefixViews && pair.second < prefixViews),
    ...orbit.pairs.map((pair) => ({ ...pair, first: pair.first + 10, second: pair.second + 10 })),
  ];
  return {
    images,
    pairs,
    ratio: orbit.ratio,
    width: orbit.width,
    truth: {
      views: orbit.truth.views.map((v) => ({ ...v, id: v.id + 10 })),
      points: orbit.truth.points,
    },
  };
}
