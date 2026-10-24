import test from 'node:test';
import assert from 'node:assert/strict';
import {
  reconstructIncremental,
  ReconstructionInitializationError,
} from '../src/reconstruction/mapper.ts';
import { reconstructIncrementalScene } from '../src/reconstruction/scene-mapper.ts';
import { InverseDepthTracker } from '../src/reconstruction/inverse-depth-tracker.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';

test('object and scene mappers publish admitted depth snapshots; default output stays unchanged', () => {
  const f = generalVideoFixture('translation'),
    controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 2 };
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const plain = run(f.input.images, f.input.pairs, controls);
    assert.ok(!('inverseDepth' in plain));
    const tracker = new InverseDepthTracker({ pixelNoiseStdDevPx: 0.25 }),
      published: number[] = [],
      tracked = run(f.input.images, f.input.pairs, {
        ...controls,
        depthTracking: tracker,
        onReconstruction: (m) => published.push(m.landmarks.length),
      });
    assert.equal(tracked.inverseDepth!.diagnostics.finite, 60);
    assert.equal(tracked.reconstruction.landmarks.length, 60);
    assert.deepEqual(published, [60]);
    assert.ok(tracked.inverseDepth!.tracks.every((p) => p.status === 'finite' && p.covariance));
    assert.ok(tracked.reconstruction.landmarks.every((p) => p.xyz.every(Number.isFinite)));
  }
});
test('initialization deferral preserves directions/history on both mapper protocols without a fake pose', () => {
  const f = generalVideoFixture('rotation');
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const tracker = new InverseDepthTracker();
    assert.throws(
      () => run(f.input.images, f.input.pairs, { depthTracking: tracker, ransacIterations: 128 }),
      (e) => {
        assert.ok(e instanceof ReconstructionInitializationError);
        assert.equal(e.inverseDepth!.diagnostics.bearingOnly, 60);
        assert.ok(
          e.inverseDepth!.tracks.every(
            (t) => !t.estimate && t.reason === 'translation-unavailable',
          ),
        );
        assert.ok(!('reconstruction' in e));
        return true;
      },
    );
    assert.equal(tracker.snapshot()!.diagnostics.uniqueObservations, 300);
  }
});

test('admitted-depth callbacks own their geometry instead of mutating returned maps', () => {
  const f = generalVideoFixture('translation');
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const tracker = new InverseDepthTracker({ pixelNoiseStdDevPx: 0.25 });
    let captured: ReturnType<typeof reconstructIncremental>['reconstruction'] | undefined;
    const result = run(f.input.images, f.input.pairs, {
      depthTracking: tracker,
      ransacIterations: 128,
      bundleIterations: 1,
      finalBundleIterations: 2,
      onReconstruction: (model) => {
        captured = structuredClone(model);
        model.views[0]!.pose.R[0] = NaN;
        model.landmarks[0]!.xyz[0] = NaN;
      },
    });
    assert.equal(captured!.landmarks.length, 60);
    assert.deepEqual(result.reconstruction, captured);
    assert.equal(tracker.snapshot()!.diagnostics.finite, 60);
  }
});
