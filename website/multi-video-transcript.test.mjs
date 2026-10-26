// Phase 0 of the refactoring programme (`docs/library-structure.md`, section 5, "Lab
// transcripts"): the transcripts of the multi-video job, `MultiVideoJob.finish` among them. Each
// case drives the real SfM worker through `onmessage` with the page's `multi*` messages and the
// fake runtime of `lab-fake-runtime.mjs`, so the job (`lib/multi-video-job.ts`) and its wiring
// (`workers/multi-video.ts`: the devices, the dense stage, the losses) both run. The fixtures
// are under `tests/fixtures/lab-transcripts/`. A structure-only phase leaves them unchanged.
import test from 'node:test';
import assert from 'node:assert/strict';
import { checkBaseline } from './lab-baseline.mjs';
import {
  VIDEOS,
  WorkerHarness,
  bundleWorker,
  framePixels,
  pageMultiStart,
  pageThreeVideos,
} from './lab-worker-harness.mjs';

const code = await bundleWorker();
const fixture = (name) =>
  new URL(`../tests/fixtures/lab-transcripts/multi-${name}.jsonl`, import.meta.url);

const [A, B, C] = VIDEOS;
const start = pageMultiStart;
const threeVideos = pageThreeVideos;

const posted = (h, type) => h.outgoing.filter((message) => message.type === type);

/** Run a case twice in new realms, check that the transcripts agree and compare the fixture. */
async function transcript(name, spec, script, check) {
  const runs = [];
  for (let i = 0; i < 2; i++) {
    const h = new WorkerHarness(code, spec);
    await script(h);
    if (i === 0) check?.(h);
    runs.push(h.transcript());
  }
  assert.equal(runs[1], runs[0], 'the transcript repeats');
  await checkBaseline(fixture(name), runs[0]);
}

test('two components, the gate, the feature minimum and the finish of the job', async () => {
  const spec = { features: { 21: 5 }, multi: { components: [[A, B], [C]] } };
  await transcript(
    'components',
    spec,
    async (h) => {
      await h.send(start(1));
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const skipped = posted(h, 'frameComplete').filter((m) => m.skipped);
      assert.ok(skipped.some((m) => m.skipped === 'Too few detected features'));
      assert.ok(skipped.some((m) => /minimum gap/.test(m.skipped)));
      const [result] = posted(h, 'result');
      assert.equal(result.summary.components.length, 2);
      assert.equal(posted(h, 'graph').length, 1);
    },
  );
});

test('the owned runtime, arrival-order ids, the cap per video and a failed component', async () => {
  const spec = {
    multi: { owned: true, components: [[A, B], [C]], errorComponent: 1, descriptorCache: true },
  };
  await transcript(
    'owned',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          query: { canonical: false, maxKeyframesPerVideo: 2, maxFeatures: 2048 },
          fp64: 'float',
          budget: 100,
          firstOctave: -1,
          clahe: true,
        }),
      );
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const [result] = posted(h, 'result');
      assert.equal(result.canonical, false);
      assert.ok(result.descriptorCache);
      assert.ok(posted(h, 'videoComplete').some((m) => m.dropped > 0));
      assert.ok(posted(h, 'model').some((m) => m.error));
    },
  );
});

test('the dense stage on each component, on a device of its own', async () => {
  const spec = { multi: { components: [[A, B], [C]] } };
  await transcript(
    'dense',
    spec,
    async (h) => {
      await h.send(
        start(1, {
          dense: true,
          denseClean: undefined,
          denseFusion: 'pixel',
          denseMesh: true,
          denseQuality: { minFiniteGain: 1 },
        }),
      );
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const results = posted(h, 'denseResult');
      assert.deepEqual(
        results.map((m) => m.component),
        [0, 1],
      );
      assert.ok(results.every((m) => m.failure === null));
      assert.equal(posted(h, 'denseDone').length, 1);
    },
  );
});

test('a dense device that does not open ends each component as failed', async () => {
  const spec = { openFailure: [1], multi: { components: [[A, B], [C]] } };
  await transcript(
    'dense-open-failure',
    spec,
    async (h) => {
      await h.send(start(1, { dense: true }));
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const results = posted(h, 'denseResult');
      assert.equal(results.length, 2);
      assert.ok(results.every((m) => /Could not open the device/.test(m.failure.message)));
    },
  );
});

test('time between the partial maps sends a held map by its timer', async () => {
  // Each registered view spends 200 ms of the harness clock.
  const spec = { multi: { components: [[A, B], [C]], viewMs: 200 } };
  await transcript(
    'throttles',
    spec,
    async (h) => {
      await h.send(start(1));
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      assert.ok(
        h.events.some((e) => e.t === 'timer'),
        'a held partial map goes out by its timer',
      );
      assert.ok(posted(h, 'model').filter((m) => m.partial).length > 1);
    },
  );
});

test('a dense device lost during a component ends it and the next one as failed', async () => {
  const spec = {
    loss: {
      call: 'reconstructDenseGpu',
      at: 0,
      info: { reason: 'unknown', message: 'dense loss' },
    },
    multi: { components: [[A, B], [C]] },
  };
  await transcript(
    'dense-loss',
    spec,
    async (h) => {
      await h.send(start(1, { dense: true }));
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const results = posted(h, 'denseResult');
      assert.equal(results.length, 2);
      assert.ok(
        results.every((m) => /WebGPU device lost during the dense MVS/.test(m.failure.message)),
      );
    },
  );
});

test('a failed reconstruction posts the error with the GPU totals', async () => {
  const spec = { multi: { components: [[A, B], [C]], failure: 'synthetic mapper failure' } };
  await transcript(
    'failure',
    spec,
    async (h) => {
      await h.send(start(1));
      await threeVideos(h, 1);
      await h.send({ id: 1, type: 'multiFinish' });
    },
    (h) => {
      const [error] = posted(h, 'error');
      assert.match(error.message, /synthetic mapper failure/);
      assert.ok(Array.isArray(error.gpuKernels));
    },
  );
});

test('a device loss during the frames is reported', async () => {
  const spec = {
    loss: { call: 'describeKeyframeGpu', at: 1 },
    multi: { components: [[A, B], [C]] },
  };
  await transcript(
    'loss',
    spec,
    async (h) => {
      await h.send(start(1));
      await h.send({
        id: 1,
        type: 'multiVideo',
        video: { videoId: A, label: 'sample-a', durationSeconds: 2, focalRatio: 0.8 },
      });
      // The page fails the run at the `error` of the second frame and sends nothing more.
      for (const [index, [time, content, shift]] of [
        [0, 0, [0, 0]],
        [0.6, 2, [20, 0]],
      ].entries())
        await h.send({
          id: 1,
          type: 'multiFrame',
          videoId: A,
          frame: { id: index, time, width: 128, height: 96, pixels: framePixels(content, shift) },
        });
    },
    (h) => {
      const [error] = posted(h, 'error');
      assert.match(error.message, /WebGPU device lost during the multi-video job/);
    },
  );
});
