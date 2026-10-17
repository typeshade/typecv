import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental, incrementalSteps } from '../dist/reconstruction/mapper.js';
import { poseFixture } from './typeshade/geometry-cases.mjs';

function sequenceFixture() {
  const data = poseFixture({ count: 40, noise: 0, outliers: 0, radial: false }),
    images = Array.from({ length: 4 }, (_, frame) => ({
      id: 10 + frame * 10,
      camera: data.camera,
      keypoints: Float64Array.from(
        Array.from({ length: 40 }, (_, i) => {
          const x = data.world[3 * i] - 0.3 * frame,
            y = data.world[3 * i + 1],
            z = data.world[3 * i + 2];
          return [
            (data.camera.fx * x) / z + data.camera.cx,
            (data.camera.fy * y) / z + data.camera.cy,
          ];
        }).flat(),
      ),
    })),
    matches = Uint32Array.from({ length: 80 }, (_, i) => Math.floor(i / 2)),
    pairs = [];
  for (let first = 0; first < images.length; first++)
    for (let second = first + 1; second < images.length; second++)
      pairs.push({ first: images[first].id, second: images[second].id, matches: matches.slice() });
  return { images, pairs };
}

const controls = {
  initialPair: [10, 20],
  ransacIterations: 128,
  bundleIterations: 2,
  finalBundleIterations: 4,
};

test('pairBatchSize preserves the CPU result and batches only the verification job envelope', () => {
  const data = sequenceFixture(),
    serial = reconstructIncremental(data.images, data.pairs, controls),
    batched = reconstructIncremental(data.images, data.pairs, { ...controls, pairBatchSize: 3 });
  assert.deepEqual(batched, serial);

  const machine = incrementalSteps(data.images, data.pairs, { ...controls, pairBatchSize: 3 });
  let step = machine.next();
  assert.equal(step.done, false);
  assert.equal(step.value.kind, 'verifyPairs');
  assert.equal(step.value.args.length, 3);
  assert.deepEqual(step.value.pairs, [
    [10, 20],
    [10, 30],
    [10, 40],
  ]);
  assert.throws(
    () => reconstructIncremental(data.images, data.pairs, { ...controls, pairBatchSize: 0 }),
    /Invalid incremental reconstruction inputs\/options/,
  );
});
