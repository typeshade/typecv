import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  reconstructMultiVideoGpu,
  reconstructMultiVideoOwnedGpu,
} from '../../dist/internal/gpu.js';
import { quantizeDescriptors } from '../../src/reconstruction/view-graph.ts';
import { pathScene } from '../view-graph-scene.ts';

const OPTIONS = {
  graph: {
    retrievalK: 3,
    temporalNeighbours: 2,
    exclusionWindow: 2,
    reservePerVideoPair: 2,
    verification: { seed: 7, ransacIterations: 100 },
  },
  mapper: { seed: 7, ransacIterations: 100, finalBundleIterations: 4 },
};
const scene = pathScene(
  [
    { label: 'orbit', cloud: 0, views: 6, from: -30, to: 30, radius: 9, height: -1 },
    { label: 'ascent', cloud: 0, views: 6, from: 5, to: 40, radius: 9, height: -3 },
    { label: 'elsewhere', cloud: 1, views: 6, from: -30, to: 30, radius: 9, height: -1 },
  ],
  { points: 50 },
);
const input = () =>
  scene.keyframes.map(({ viewId: _viewId, descriptors, ...frame }) => ({
    ...frame,
    keypoints: frame.keypoints.slice(),
    global: frame.global.slice(),
    descriptors: quantizeDescriptors(descriptors),
  }));
const componentEvidence = (component) => {
  const { ms: _ms, ...evidence } = component;
  return evidence;
};
const edgeEvidence = ({
  first,
  second,
  crossVideo,
  inliers,
  homographyInliers,
  medianAngleDeg,
  seedEligible,
  source,
}) => ({
  first,
  second,
  crossVideo,
  inliers,
  homographyInliers,
  medianAngleDeg,
  seedEligible,
  source,
});
const graphEvidence = ({ edges, components, videos, stats }) => ({
  edges: edges.map(edgeEvidence),
  components,
  videos,
  stats,
});

test('owned multi-video exactly preserves graph, registration, poses, landmarks and callback evidence', async (t) => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const normalInput = input(),
      ownedInput = input(),
      externalArrays = ownedInput.flatMap((frame) => [
        frame.keypoints,
        frame.descriptors,
        frame.global,
      ]);
    const snapshots = externalArrays.map((array) => array.slice());
    const normalEvents = [],
      ownedEvents = [],
      releases = [];
    const normal = await reconstructMultiVideoGpu(device, normalInput, {
      ...OPTIONS,
      onEvent: (event) => normalEvents.push(structuredClone(event)),
    });
    const owned = await reconstructMultiVideoOwnedGpu(device, ownedInput, {
      ...OPTIONS,
      onEvent: (event) => ownedEvents.push(structuredClone(event)),
      onRelease: (event) => {
        if (event.stage === 'component')
          assert.ok(
            ownedEvents.some(
              (done) =>
                done.stage === 'componentDone' &&
                done.reconstruction.views.some((view) => event.viewIds.includes(view.id)),
            ),
            'component features retire only after its final independent model callback',
          );
        releases.push(event);
      },
    });
    assert.equal(ownedInput.length, 0);
    assert.equal(normalInput.length, 18, 'ordinary input container remains immutable');
    assert.deepEqual(ownedEvents, normalEvents);
    assert.deepEqual(
      owned.components.map(componentEvidence),
      normal.components.map(componentEvidence),
    );
    assert.deepEqual(graphEvidence(owned.graph), graphEvidence(normal.graph));
    assert.deepEqual(
      owned.keyframes,
      normal.keyframes.map(({ viewId, videoId, frameKey, time }) => ({
        viewId,
        videoId,
        frameKey,
        time,
      })),
    );
    assert.deepEqual(
      releases.map((event) => event.stage),
      ['descriptors', 'component', 'component'],
    );
    assert.ok(
      owned.hostRetention.featuresBeforeGraphBytes > owned.hostRetention.featuresAfterGraphBytes,
    );
    assert.equal(
      owned.hostRetention.featuresAfterGraphBytes,
      normalInput.reduce((sum, frame) => sum + frame.keypoints.byteLength, 0),
    );
    assert.equal(owned.hostRetention.featuresAfterMappingBytes, 0);
    assert.ok(owned.hostRetention.graphMatchesBeforeMappingBytes > 0);
    assert.equal(owned.hostRetention.graphMatchesAfterMappingBytes, 0);
    t.diagnostic(JSON.stringify({ logicalOwnedBackingBuffers: owned.hostRetention }));
    externalArrays.forEach((array, i) => assert.deepEqual(array, snapshots[i]));
    assert.ok(
      normal.keyframes[0].descriptors.length > 0,
      'the public graph descriptor getters keep their original contract',
    );
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});

test('owned empty and unreconstructed components retire all roots; failure consumes its input', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const empty = await reconstructMultiVideoOwnedGpu(device, []);
    assert.equal(empty.components.length, 0);
    assert.equal(empty.hostRetention.featuresAfterMappingBytes, 0);
    const smallInput = input().slice(0, 2);
    const small = await reconstructMultiVideoOwnedGpu(device, smallInput, OPTIONS);
    assert.ok(small.components.every((component) => !component.reconstructed));
    assert.equal(small.hostRetention.featuresAfterMappingBytes, 0);
    assert.equal(small.hostRetention.graphMatchesAfterMappingBytes, 0);
    const failedInput = input();
    const arrays = failedInput.flatMap((frame) => [
      frame.keypoints,
      frame.descriptors,
      frame.global,
    ]);
    const snapshots = arrays.map((array) => array.slice());
    await assert.rejects(
      reconstructMultiVideoOwnedGpu(device, failedInput, {
        onProgress() {
          throw new Error('stopped');
        },
      }),
      /stopped/,
    );
    assert.equal(failedInput.length, 0);
    arrays.forEach((array, i) => assert.deepEqual(array, snapshots[i]));
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});
