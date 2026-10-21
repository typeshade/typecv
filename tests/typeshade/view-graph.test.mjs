import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, gpuViewGraphBackend } from '../../dist/internal/gpu.js';
import { buildViewGraph, cpuViewGraphBackend } from '../../src/reconstruction/view-graph.ts';
import { pathScene } from '../view-graph-scene.ts';
import { extractPhotoFrames, hasFfmpeg, photoKeyframes } from './view-graph-photos.mjs';

// Verifies: docs/multi-video.md 3.4 (matching and verification on the device), 3.5, 4.1.

const OPTIONS = {
  retrievalK: 3,
  temporalNeighbours: 2,
  exclusionWindow: 2,
  reservePerVideoPair: 2,
  verification: { ransacIterations: 60 },
};

test('the GPU backend on the CPU tier gives the CPU backend components', async () => {
  const scene = pathScene(
    [
      { label: 'orbit', cloud: 0, views: 6, from: -30, to: 30, radius: 9, height: -1 },
      { label: 'ascent', cloud: 0, views: 6, from: 5, to: 40, radius: 9, height: -3 },
      { label: 'elsewhere', cloud: 1, views: 6, from: -30, to: 30, radius: 9, height: -1 },
    ],
    { points: 50 },
  );
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const gpu = await buildViewGraph(scene.keyframes, gpuViewGraphBackend(device), OPTIONS);
    const cpu = await buildViewGraph(scene.keyframes, cpuViewGraphBackend(), OPTIONS);
    assert.deepEqual(
      gpu.components.map((c) => c.viewIds),
      cpu.components.map((c) => c.viewIds),
    );
    assert.deepEqual(gpu.videos, cpu.videos);
    assert.deepEqual(gpu.candidates, cpu.candidates);
    assert.equal(gpu.components.length, 2);
    assert.deepEqual(gpu.videos.map((v) => v.tag).sort(), ['alone', 'connected', 'connected']);
    assert.ok(gpu.stats.candidatePairs < gpu.stats.exhaustivePairs);
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});

test(
  'photo samples: two halves of one scene join, another scene stands alone',
  { skip: !hasFfmpeg() && 'ffmpeg is not installed' },
  async () => {
    const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    try {
      const frames = await extractPhotoFrames(
        device,
        {
          'hall-a': { sample: 'gerrard-hall', frames: [0, 1, 2] },
          'hall-b': { sample: 'gerrard-hall', frames: [3, 4, 5] },
          other: { sample: 'person-hall', frames: [0, 1, 2] },
        },
        { scale: 0.5, maxFeatures: 512 },
      );
      const graph = await buildViewGraph(photoKeyframes(frames, 64), gpuViewGraphBackend(device), {
        ...OPTIONS,
        minComponentViews: 3,
        exclusionWindow: 1,
      });
      assert.equal(graph.stats.keyframes, 9);
      assert.ok(graph.stats.candidatePairs < graph.stats.exhaustivePairs);
      assert.ok(graph.stats.crossVideoEdges >= 1);
      const tags = graph.videos.map((v) => v.tag).sort();
      assert.deepEqual(tags, ['alone', 'connected', 'connected']);
      // No edge joins the two scenes.
      const scene = new Map(photoKeyframes(frames, 64).map((k, i) => [k.viewId, frames[i].sample]));
      for (const e of graph.edges) assert.equal(scene.get(e.first), scene.get(e.second));
    } finally {
      await device.dispose();
    }
  },
);
