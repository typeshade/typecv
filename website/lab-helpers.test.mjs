// Phase 5a of the refactoring programme (`docs/library-structure.md`, section 5, row 5): the table
// tests of the lab's pure helpers. Each helper was inline in `src/workers/sfm.worker.ts`,
// `src/workers/multi-video.ts` or `src/lib/multi-video-job.ts` on `main` at db7216a, the parent of
// phase 5a. Each table holds the helper's values and units, and rows on both sides of each branch.
// `inline` is the code of a site as it stood there, copied with its literals, and every row holds
// the helper to it. Where two sites differ, each variant has its own copy and its own rows.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BUDGET_REFERENCE_EDGE,
  REFERENCE_POOL_BYTES,
  REFERENCE_WORKING_BYTES,
  analysisScale,
  deviceBudgets,
  workingBytesFromAdapter,
} from './src/lib/working-budget.ts';
import { denseWorkingBytes } from './src/lib/dense-run.ts';
import {
  KEYPOINT_COLOR_FRAMES,
  MAX_FEATURE_CAP,
  UNCOLORED_CHANNEL,
  featureCap,
  keypointColorBytes,
  keypointColorFloats,
  landmarkColor,
} from './src/lib/lab-features.ts';
import { LAB_OPTIONS } from './src/lib/lab-options.ts';
import { medianMotionPx } from './src/lib/keyframe-gate.ts';
import { trimStages } from './src/lib/lab-ledger.ts';
import { MATCH_LINE_ENTRIES, frameCamera, matchLines } from './src/lib/lab-matching.ts';

const MiB = 1024 ** 2;

test('the budgets: 64 and 32 MiB at 640 px, 4 times at octave -1, half of maxBufferSize (start, runDenseStage)', () => {
  assert.equal(BUDGET_REFERENCE_EDGE, 640);
  assert.equal(REFERENCE_WORKING_BYTES, 64 * MiB);
  assert.equal(REFERENCE_POOL_BYTES, 32 * MiB);
  // `start` of both workers: the scale.
  const inlineScale = (analysisEdge, firstOctave) =>
    ((analysisEdge * analysisEdge) / (640 * 640)) * (firstOctave === -1 ? 4 : 1);
  // `start` of both workers: the budgets of the SfM device.
  const inlineSfm = (scale, maxBufferSize) => ({
    workingBytes: workingBytesFromAdapter(Math.round(64 * 1024 ** 2 * scale), maxBufferSize),
    poolBytes: Math.round(32 * 1024 ** 2 * scale),
  });
  // `runDenseStage` of the multi-video worker: the budgets of the dense device.
  const inlineDense = (scale, maxBufferSize) => ({
    workingBytes: denseWorkingBytes(Math.round(64 * 1024 ** 2 * scale), maxBufferSize),
    poolBytes: Math.round(32 * 1024 ** 2 * scale),
  });
  for (const [edge, octave, maxBufferSize, scale, workingBytes, poolBytes] of [
    [640, 0, undefined, 1, 64 * MiB, 32 * MiB],
    [640, -1, undefined, 4, 256 * MiB, 128 * MiB],
    [1280, 0, undefined, 4, 256 * MiB, 128 * MiB],
    [1280, -1, undefined, 16, 1024 * MiB, 512 * MiB],
    [320, 0, undefined, 0.25, 16 * MiB, 8 * MiB],
    [720, 0, undefined, 1.265625, 84934656, 42467328],
    [853, 0, undefined, 1.77638916015625, 119211459, 59605729],
    [640, 0, 0, 1, 64 * MiB, 32 * MiB],
    [640, 0, 128 * MiB, 1, 64 * MiB, 32 * MiB],
    [640, 0, 129 * MiB + 1, 1, 67633152, 32 * MiB],
    [640, 0, 256 * MiB, 1, 128 * MiB, 32 * MiB],
    [960, 0, 256 * MiB, 2.25, 144 * MiB, 72 * MiB],
    [640, 0, 2048 * MiB, 1, 1024 * MiB, 32 * MiB],
    [1280, 0, 2048 * MiB, 4, 1024 * MiB, 128 * MiB],
    [1280, -1, 2048 * MiB, 16, 1024 * MiB, 512 * MiB],
  ]) {
    const row = `${edge} px, octave ${octave}, maxBufferSize ${maxBufferSize}`;
    assert.equal(analysisScale(edge, octave), scale, row);
    assert.equal(inlineScale(edge, octave), scale, row + ', inline');
    const expected = { workingBytes, poolBytes };
    assert.deepEqual(deviceBudgets(scale, maxBufferSize), expected, row);
    assert.deepEqual(inlineSfm(scale, maxBufferSize), expected, row + ', inline');
    assert.deepEqual(
      deviceBudgets(scale, maxBufferSize, denseWorkingBytes),
      expected,
      row + ', dense',
    );
    assert.deepEqual(inlineDense(scale, maxBufferSize), expected, row + ', dense, inline');
  }
});

test('the feature cap: 4096 features at 640 px, at most 16384 (start of both workers)', () => {
  assert.equal(MAX_FEATURE_CAP, 16384);
  assert.equal(LAB_OPTIONS.maxFeatures, 4096);
  // sfm.worker.ts `start`: the base is the `maxFeatures` of the run's controls.
  const inlineSfm = (base, scale) => Math.min(16384, Math.round(base.maxFeatures * scale));
  // multi-video.ts `start`, when the query sets no cap: the base is 4096.
  const inlineMulti = (scale) => Math.min(16384, Math.round(4096 * scale));
  for (const [scale, cap] of [
    [0.25, 1024],
    [1, 4096],
    [1.265625, 5184],
    [1.77638916015625, 7276],
    [2.25, 9216],
    [3.99, 16343],
    [4, 16384],
    [4.01, 16384],
    [16, 16384],
  ]) {
    assert.equal(featureCap(4096, scale), cap, `scale ${scale}`);
    assert.equal(featureCap(LAB_OPTIONS.maxFeatures, scale), cap, `scale ${scale}, controls`);
    assert.equal(inlineSfm(LAB_OPTIONS, scale), cap, `scale ${scale}, inline`);
    assert.equal(inlineMulti(scale), cap, `scale ${scale}, multi-video, inline`);
  }
});

test('the median motion of the legacy rule: the median displacement of the matches, in the pixels of this frame (processFrame)', () => {
  // sfm.worker.ts `medianMotionPx`, as it stood.
  const inline = (matches, previousPoints, previous, points, frame) => {
    const distances = [];
    for (let i = 0; i < matches.length; i += 2) {
      const a = matches[i] * 2,
        b = matches[i + 1] * 2;
      distances.push(
        Math.hypot(
          points[b] - (previousPoints[a] * frame.width) / previous.width,
          points[b + 1] - (previousPoints[a + 1] * frame.height) / previous.height,
        ),
      );
    }
    if (!distances.length) return Infinity;
    distances.sort((x, y) => x - y);
    return distances[distances.length >> 1];
  };
  const size = (width, height) => ({ width, height });
  const f64 = (...values) => Float64Array.from(values);
  for (const [what, matches, previousPoints, previous, points, frame, motion] of [
    ['no match', [], f64(1, 1), size(128, 96), f64(1, 1), size(128, 96), Infinity],
    ['one match', [0, 0], f64(10, 10), size(128, 96), f64(13, 14), size(128, 96), 5],
    [
      'an odd count',
      [0, 0, 1, 1, 2, 2],
      f64(0, 0, 0, 0, 0, 0),
      size(9, 9),
      f64(3, 0, 1, 0, 2, 0),
      size(9, 9),
      2,
    ],
    [
      'an even count: the upper middle',
      [0, 0, 1, 1, 2, 2, 3, 3],
      f64(0, 0, 0, 0, 0, 0, 0, 0),
      size(9, 9),
      f64(4, 0, 1, 0, 3, 0, 2, 0),
      size(9, 9),
      3,
    ],
    [
      'the previous points scaled to this frame',
      [0, 0],
      f64(100, 200),
      size(1280, 960),
      f64(53, 104),
      size(640, 480),
      5,
    ],
    ['a scale in x only', [0, 0], f64(100, 40), size(200, 100), f64(50, 43), size(100, 100), 3],
    [
      'the indices of each frame',
      [1, 0, 0, 1],
      f64(0, 0, 7, 7),
      size(16, 16),
      f64(7, 11, 3, 4),
      size(16, 16),
      5,
    ],
  ]) {
    const args = [Uint32Array.from(matches), previousPoints, previous, points, frame];
    assert.equal(medianMotionPx(...args), motion, what);
    assert.equal(inline(...args), motion, what + ', inline');
  }
});

test('the detailed stage list: 400 entries, keyframe entries dropped last (record, registerFrame)', () => {
  assert.equal(LAB_OPTIONS.detailedStageLimit, 400);
  // sfm.worker.ts `trimStages`, as it stood, with `LAB_OPTIONS.detailedStageLimit` written out.
  const inline = (stages) => {
    let excess = stages.length - 400;
    for (let i = 0; excess > 0 && i < stages.length;)
      if (stages[i].operation === 'keyframe') i++;
      else {
        stages.splice(i, 1);
        excess--;
      }
    if (excess > 0) stages.splice(0, excess);
  };
  // A list of `n` entries; the entries at the indices of `keyframes` are keyframe entries.
  const list = (n, keyframes = () => false) =>
    Array.from({ length: n }, (_, id) => ({
      operation: keyframes(id) ? 'keyframe' : 'matches',
      id,
    }));
  const ids = (stages) => stages.map((stage) => stage.id);
  const range = (from, to) => Array.from({ length: to - from }, (_, i) => from + i);
  for (const [what, stages, kept] of [
    ['an empty list', list(0), []],
    ['400 entries', list(400), range(0, 400)],
    ['401 entries: the oldest goes', list(401), range(1, 401)],
    ['the oldest keyframe entries stay', list(401, (id) => id < 3), [0, 1, 2, ...range(4, 401)]],
    ['403 entries: three go', list(403, (id) => id % 2 === 0), [0, 2, 4, ...range(6, 403)]],
    ['only keyframe entries: the oldest go', list(402, () => true), range(2, 402)],
    ['keyframe entries go when the others are gone', list(405, (id) => id < 403), range(3, 403)],
  ]) {
    const helper = stages.map((stage) => ({ ...stage })),
      copy = stages.map((stage) => ({ ...stage }));
    trimStages(helper);
    inline(copy);
    assert.deepEqual(ids(helper), kept, what);
    assert.deepEqual(ids(copy), kept, what + ', inline');
    assert.deepEqual(helper, copy, what + ', the entries');
  }
});

test('the frame camera: the calibration, else the focal ratio of the source width (guidedMatches, registerFrame, MultiVideoJob.frame)', () => {
  // sfm.worker.ts `frameCamera`, as it stood, with the worker's job as an argument.
  const inlineWorker = (frame, job) => {
    const sourceWidth = frame.sourceWidth ?? frame.width;
    const sourceHeight = frame.sourceHeight ?? frame.height;
    const sourceFocal = sourceWidth * (job.session.focalSearch?.ratio ?? job.focal);
    return (
      frame.camera ?? {
        fx: (sourceFocal * frame.width) / sourceWidth,
        fy: (sourceFocal * frame.height) / sourceHeight,
        cx: (frame.width - 1) / 2,
        cy: (frame.height - 1) / 2,
        k1: 0,
        k2: 0,
      }
    );
  };
  // multi-video-job.ts `cameraOf`, as it stood.
  const inlineJob = (frame, focalRatio) => {
    const sourceWidth = frame.sourceWidth ?? frame.width;
    const sourceHeight = frame.sourceHeight ?? frame.height;
    const sourceFocal = sourceWidth * focalRatio;
    return (
      frame.camera ?? {
        fx: (sourceFocal * frame.width) / sourceWidth,
        fy: (sourceFocal * frame.height) / sourceHeight,
        cx: (frame.width - 1) / 2,
        cy: (frame.height - 1) / 2,
        k1: 0,
        k2: 0,
      }
    );
  };
  const calibration = { fx: 500, fy: 501, cx: 60, cy: 50, k1: -0.1, k2: 0.01 };
  const pinhole = (fx, fy, cx, cy) => ({ fx, fy, cx, cy, k1: 0, k2: 0 });
  for (const [what, frame, ratio, camera] of [
    ['a calibrated frame', { width: 128, height: 96, camera: calibration }, 0.8, calibration],
    // 128 x 0.8 x 96 / 96 rounds to 102.40000000000002, as the inline code did.
    [
      'no source size',
      { width: 128, height: 96 },
      0.8,
      pinhole(102.4, 102.40000000000002, 63.5, 47.5),
    ],
    [
      'a sampled frame',
      { width: 128, height: 96, sourceWidth: 1280, sourceHeight: 960 },
      0.8,
      pinhole(102.4, 102.4, 63.5, 47.5),
    ],
    [
      'a 1080p source at 640 x 360',
      { width: 640, height: 360, sourceWidth: 1920, sourceHeight: 1080 },
      1.2,
      pinhole(768, 768, 319.5, 179.5),
    ],
    [
      'another aspect ratio',
      { width: 640, height: 480, sourceWidth: 1920, sourceHeight: 1080 },
      1.2,
      pinhole(768, 1024, 319.5, 239.5),
    ],
    [
      'a source width without a source height',
      { width: 128, height: 96, sourceWidth: 256 },
      1,
      pinhole(128, 256, 63.5, 47.5),
    ],
  ]) {
    assert.deepEqual(frameCamera(frame, ratio), camera, what);
    assert.deepEqual(inlineJob(frame, ratio), camera, what + ', job, inline');
    // The worker's focal: the focal search's estimate once it ran, else the assumed focal.
    for (const job of [
      { session: { focalSearch: undefined }, focal: ratio },
      { session: { focalSearch: { ratio } }, focal: 0.5 },
    ]) {
      const workerRatio = job.session.focalSearch?.ratio ?? job.focal;
      assert.deepEqual(frameCamera(frame, workerRatio), camera, what + ', worker');
      assert.deepEqual(inlineWorker(frame, job), camera, what + ', worker, inline');
    }
    if (frame.camera)
      assert.equal(frameCamera(frame, ratio), frame.camera, what + ', the same object');
  }
});

test('the match lines: 160 entries at most, the earlier points scaled to this frame (postMatchLines)', () => {
  assert.equal(MATCH_LINE_ENTRIES, 160);
  // sfm.worker.ts `postMatchLines`, as it stood: the lines it posted.
  const inline = (otherPoints, otherFrame, points, frame, matches) => {
    const lines = [];
    for (let i = 0; i < Math.min(matches.length, 160); i += 2) {
      const a = matches[i] * 2,
        b = matches[i + 1] * 2;
      lines.push(
        (otherPoints[a] * frame.width) / otherFrame.width,
        (otherPoints[a + 1] * frame.height) / otherFrame.height,
        points[b],
        points[b + 1],
      );
    }
    return lines;
  };
  // 200 points: point i of the earlier frame is (i, 2 i), of this frame (i + 0.5, 2 i + 1).
  const earlier = Float64Array.from({ length: 400 }, (_, k) => (k % 2 ? k - 1 : k / 2));
  const later = Float64Array.from({ length: 400 }, (_, k) => (k % 2 ? k : k / 2 + 0.5));
  const pairs = (n) => Uint32Array.from({ length: 2 * n }, (_, k) => k >> 1);
  const size = (width, height) => ({ width, height });
  for (const [what, otherFrame, frame, matches, count, first] of [
    ['no match', size(128, 96), size(128, 96), pairs(0), 0, undefined],
    ['five matches', size(128, 96), size(128, 96), pairs(5), 5, [0, 0, 0.5, 1]],
    ['80 matches', size(128, 96), size(128, 96), pairs(80), 80, [0, 0, 0.5, 1]],
    ['81 matches: 80 drawn', size(128, 96), size(128, 96), pairs(81), 80, [0, 0, 0.5, 1]],
    ['200 matches: 80 drawn', size(128, 96), size(128, 96), pairs(200), 80, [0, 0, 0.5, 1]],
    [
      'the earlier frame twice as large',
      size(256, 192),
      size(128, 96),
      pairs(3),
      3,
      [0, 0, 0.5, 1],
    ],
    [
      'an odd count of entries',
      size(128, 96),
      size(128, 96),
      Uint32Array.from([1, 1, 2]),
      2,
      [1, 2, 1.5, 3],
    ],
  ]) {
    const lines = matchLines(earlier, otherFrame, later, frame, matches);
    assert.deepEqual(lines, inline(earlier, otherFrame, later, frame, matches), what + ', inline');
    assert.equal(lines.length, 4 * count, what);
    if (first) assert.deepEqual(lines.slice(0, 4), first, what + ', the first line');
  }
  // The second line of a scaled pair: point 1 of the earlier frame, (1, 2), is (0.5, 1) here.
  assert.deepEqual(
    matchLines(earlier, size(256, 192), later, size(128, 96), pairs(3)).slice(4, 8),
    [0.5, 1, 1.5, 3],
  );
});

test('the keypoint colours: the pixel under each keypoint, floats in the worker and bytes in the job (sampleColors)', () => {
  assert.equal(KEYPOINT_COLOR_FRAMES, 64);
  // sfm.worker.ts `sampleColors`, as it stood, without the window of 64 frames that it fed.
  const inlineFloats = (frame, points) => {
    const colors = new Float32Array((points.length / 2) * 3);
    for (let i = 0; i < points.length / 2; i++) {
      const x = Math.max(0, Math.min(frame.width - 1, Math.round(points[i * 2])));
      const y = Math.max(0, Math.min(frame.height - 1, Math.round(points[i * 2 + 1])));
      const offset = (y * frame.width + x) * 4;
      for (let channel = 0; channel < 3; channel++)
        colors[i * 3 + channel] = frame.pixels[offset + channel] / 255;
    }
    return colors;
  };
  // multi-video-job.ts `sampleColors`, as it stood.
  const inlineBytes = (frame, points) => {
    const colors = new Uint8Array((points.length / 2) * 3);
    for (let i = 0; i < points.length / 2; i++) {
      const x = Math.max(0, Math.min(frame.width - 1, Math.round(points[i * 2])));
      const y = Math.max(0, Math.min(frame.height - 1, Math.round(points[i * 2 + 1])));
      const offset = (y * frame.width + x) * 4;
      for (let c = 0; c < 3; c++) colors[i * 3 + c] = frame.pixels[offset + c];
    }
    return colors;
  };
  // A 3 x 2 frame: the pixel (x, y) has the channels 10 (3 y + x) + 1, + 2 and + 3.
  const pixels = new Uint8ClampedArray(3 * 2 * 4);
  for (let p = 0; p < 6; p++) pixels.set([10 * p + 1, 10 * p + 2, 10 * p + 3, 255], p * 4);
  const frame = { width: 3, height: 2, pixels };
  const rgb = (x, y) => [10 * (3 * y + x) + 1, 10 * (3 * y + x) + 2, 10 * (3 * y + x) + 3];
  for (const [what, points, under] of [
    ['no keypoint', [], []],
    ['the corners', [0, 0, 2, 1], [rgb(0, 0), rgb(2, 1)]],
    ['rounded to the nearest pixel', [1.4, 0.6, 0.5, 0.49], [rgb(1, 1), rgb(1, 0)]],
    ['clamped to the frame', [-5, 10, 7, -1], [rgb(0, 1), rgb(2, 0)]],
    ['halves round up', [1.5, 0.5], [rgb(2, 1)]],
  ]) {
    const keypoints = Float64Array.from(points);
    const floats = keypointColorFloats(frame, keypoints);
    const bytes = keypointColorBytes(frame, keypoints);
    assert.deepEqual(bytes, Uint8Array.from(under.flat()), what + ', bytes');
    assert.deepEqual(bytes, inlineBytes(frame, keypoints), what + ', bytes, inline');
    assert.deepEqual(
      floats,
      Float32Array.from(under.flat(), (v) => v / 255),
      what + ', floats',
    );
    assert.deepEqual(floats, inlineFloats(frame, keypoints), what + ', floats, inline');
  }
  // A keypoint that is not a number reads no pixel: NaN as a float, 0 as a byte, as before.
  const lost = Float64Array.from([NaN, 0]);
  assert.deepEqual(Array.from(keypointColorFloats(frame, lost)), [NaN, NaN, NaN]);
  assert.deepEqual(Array.from(inlineFloats(frame, lost)), [NaN, NaN, NaN]);
  assert.deepEqual(keypointColorBytes(frame, lost), Uint8Array.from([0, 0, 0]));
  assert.deepEqual(inlineBytes(frame, lost), Uint8Array.from([0, 0, 0]));
});

test('the landmark colours: the persistent cache of the worker, none in the job (scene, MultiVideoJob.scene)', () => {
  assert.equal(UNCOLORED_CHANNEL, 0.7);
  // sfm.worker.ts `scene`, as it stood: the colour of one landmark.
  const inlineWorker = (active, point) => {
    let color = active.pointColors.get(point.trackId);
    if (!color)
      for (const observation of point.observations) {
        const colors = active.colors.get(observation.viewId);
        if (colors && observation.featureId * 3 + 2 < colors.length) {
          color = Array.from(
            colors.subarray(observation.featureId * 3, observation.featureId * 3 + 3),
          );
          active.pointColors.set(point.trackId, color);
          break;
        }
      }
    return { ...point, color: color ?? [0.7, 0.7, 0.7] };
  };
  // multi-video-job.ts `scene`, as it stood: the colour of one landmark, and the map of each call.
  const inlineJob = (info, point, colorOf = new Map()) => {
    let color = colorOf.get(point.trackId);
    if (!color)
      for (const o of point.observations) {
        const k = info.get(o.viewId);
        if (k && o.featureId * 3 + 2 < k.colors.length) {
          color = [
            k.colors[o.featureId * 3] / 255,
            k.colors[o.featureId * 3 + 1] / 255,
            k.colors[o.featureId * 3 + 2] / 255,
          ];
          break;
        }
      }
    return { ...point, color: color ?? [0.7, 0.7, 0.7] };
  };
  const grey = [0.7, 0.7, 0.7];
  const point = (trackId, ...observations) => ({
    trackId,
    observations: observations.map(([viewId, featureId]) => ({ viewId, featureId })),
  });
  // View 1 holds two features, view 3 none; view 2 has no colours.
  const bytes = new Map([
    [1, Uint8Array.from([10, 20, 30, 40, 50, 60])],
    [3, new Uint8Array(0)],
  ]);
  const floats = new Map([...bytes].map(([id, b]) => [id, Float32Array.from(b, (v) => v / 255)]));
  const info = new Map([...bytes].map(([id, colors]) => [id, { colors }]));
  const asFloats = (b) => Array.from(Float32Array.from(b, (v) => v / 255));
  for (const [what, landmark, fromBytes] of [
    ['the first observation with colours', point(7, [2, 0], [1, 1]), [40, 50, 60]],
    ['the first of two that have colours', point(8, [1, 0], [1, 1]), [10, 20, 30]],
    ['a feature beyond the colours', point(9, [1, 2], [3, 0]), undefined],
    ['no observation', point(10), undefined],
  ]) {
    // The job: bytes divided by 255, no cache.
    const job = fromBytes ? fromBytes.map((v) => v / 255) : grey;
    assert.deepEqual(
      landmarkColor(landmark, (id) => info.get(id)?.colors, 'byte'),
      job,
      what,
    );
    assert.deepEqual(inlineJob(info, landmark).color, job, what + ', job, inline');
    // The worker: floats as they are, kept in the cache.
    const worker = fromBytes ? asFloats(fromBytes) : grey;
    const cache = new Map(),
      active = { pointColors: new Map(), colors: floats };
    assert.deepEqual(
      landmarkColor(landmark, (id) => floats.get(id), 'float', cache),
      worker,
      what,
    );
    assert.deepEqual(inlineWorker(active, landmark).color, worker, what + ', worker, inline');
    assert.deepEqual([...cache], [...active.pointColors], what + ', the cache');
  }
  // The worker keeps a track's colour after the colours of its frames are gone; the job does not.
  const track = point(11, [1, 0]);
  const cache = new Map(),
    active = { pointColors: new Map(), colors: new Map(floats) };
  const first = landmarkColor(track, (id) => active.colors.get(id), 'float', cache);
  assert.deepEqual(inlineWorker(active, track).color, first);
  active.colors.clear();
  assert.equal(
    landmarkColor(track, (id) => active.colors.get(id), 'float', cache),
    first,
  );
  assert.equal(inlineWorker(active, track).color, active.pointColors.get(11));
  assert.deepEqual(active.pointColors.get(11), first);
  const released = new Map([[1, { colors: new Uint8Array(0) }]]);
  assert.deepEqual(
    landmarkColor(track, (id) => released.get(id)?.colors, 'byte'),
    grey,
  );
  assert.deepEqual(inlineJob(released, track).color, grey);
  // A cached colour wins over the colours of the views, in the worker.
  const seeded = new Map([[7, [1, 0, 0]]]);
  assert.deepEqual(
    landmarkColor(point(7, [1, 1]), (id) => floats.get(id), 'float', seeded),
    [1, 0, 0],
  );
  assert.deepEqual(
    inlineWorker({ pointColors: new Map([[7, [1, 0, 0]]]), colors: floats }, point(7, [1, 1]))
      .color,
    [1, 0, 0],
  );
});
