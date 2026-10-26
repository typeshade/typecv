import test from 'node:test';
import assert from 'node:assert/strict';
import { rotationExp } from '../src/geometry/pose.ts';
import { scene } from './scene.ts';
import {
  HUBER_PX,
  MIN_PAIR_MATCHES,
  calibrationViewIds,
  firstLastTurnDeg,
  parseOptions,
  runAtRatio,
  runModel,
  scoreLine,
  scoreTable,
  sweepRow,
  tracksToInput,
  type ExportModel,
  type SweepRow,
} from '../scripts/measure-focal-sweep.mjs';

const camera = { fx: 500, fy: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };
const deg = Math.PI / 180;
/** The pixel that encodes a track and a view, so that a test can read a match back. */
const pixel = (track: number, viewId: number): [number, number] => [track, viewId + 0.25];

/**
 * Views 7, 3 and 5 (export order) at the times 0.5, 0 and 1 s. Tracks 0 to 39 see 7 and 3
 * (listed 7 first), tracks 100 to 130 see 3 and 5 (31 matches), tracks 200 to 231 see 5 and 7
 * (32 matches). Track 300 sees 3 twice and 7 once; track 400 sees 3 and the unknown view 9.
 */
function exportModel(): ExportModel {
  const views = [
    { id: 7, time: 0.5, width: 640, height: 480, camera },
    { id: 3, time: 0, width: 640, height: 480, camera },
    { id: 5, time: 1, width: 640, height: 480, camera },
  ];
  const track = (id: number, viewIds: number[]) => ({
    observations: viewIds.map((viewId) => ({ viewId, pixel: pixel(id, viewId) })),
  });
  const landmarks = [];
  for (let t = 0; t < 40; t++) landmarks.push(track(t, [7, 3]));
  for (let t = 100; t < 131; t++) landmarks.push(track(t, [3, 5]));
  for (let t = 200; t < 232; t++) landmarks.push(track(t, [5, 7]));
  landmarks.push({
    observations: [
      { viewId: 3, pixel: pixel(300, 3) },
      { viewId: 7, pixel: pixel(300, 7) },
      { viewId: 3, pixel: [-1, -1] },
    ],
  });
  landmarks.push(track(400, [3, 9]));
  return { views, landmarks };
}

test('tracks become keypoints and pair matches; a pair keeps 32 or more matches', () => {
  const exported = exportModel();
  assert.equal(MIN_PAIR_MATCHES, 32);
  // The images come in time order (3, 7, 5), the order in which the lab's session receives them.
  const order = calibrationViewIds(exported.views);
  assert.deepEqual(order, [3, 7, 5]);
  const { images, pairs, droppedPairs } = tracksToInput(exported, order);
  assert.deepEqual(
    images.map((image) => [image.id, image.keypoints.length / 2]),
    [
      [3, 40 + 31 + 1 + 1],
      [7, 40 + 32 + 1],
      [5, 31 + 32],
    ],
  );
  assert.equal(images[0]!.camera, camera);
  // The pair of 31 matches is dropped. The earlier view comes first, also when its id is the
  // higher one (7 before 5), and the pairs follow their second view in time order.
  assert.deepEqual(
    pairs.map((pair) => [pair.first, pair.second, pair.matches.length / 2]),
    [
      [3, 7, 41],
      [7, 5, 32],
    ],
  );
  assert.equal(droppedPairs, 1);
  // Each match joins the two keypoints of one track.
  const keypoints = new Map(images.map((image) => [image.id, image.keypoints]));
  for (const { first, second, matches } of pairs)
    for (let m = 0; m < matches.length; m += 2) {
      const a = keypoints.get(first)!,
        b = keypoints.get(second)!;
      assert.equal(a[2 * matches[m]!], b[2 * matches[m + 1]!]);
      assert.equal(a[2 * matches[m]! + 1], first + 0.25);
      assert.equal(b[2 * matches[m + 1]! + 1], second + 0.25);
    }
  // A track that repeats a view keeps its first observation there.
  assert.ok(!Array.from(keypoints.get(3)!).includes(-1));
  // A lower limit keeps the pair of 31. In a subset of views, a track keeps its keypoints there,
  // also one that it has in one view only, and its matches between those views. The given order
  // decides which view of a pair comes first.
  assert.deepEqual(
    tracksToInput(exported, order, { minMatches: 31 }).pairs.map((p) => [p.first, p.second]),
    [
      [3, 7],
      [3, 5],
      [7, 5],
    ],
  );
  const subset = tracksToInput(exported, [7, 3]);
  assert.deepEqual(
    subset.images.map((image) => [image.id, image.keypoints.length / 2]),
    [
      [7, 73],
      [3, 73],
    ],
  );
  assert.deepEqual(
    subset.pairs.map((pair) => [pair.first, pair.second]),
    [[7, 3]],
  );
});

test('the calibration subset is the first views by time', () => {
  const { views } = exportModel();
  assert.deepEqual(calibrationViewIds(views, 2), [3, 7]);
  assert.deepEqual(calibrationViewIds([{ id: 1, camera }, ...views], 4), [3, 7, 5, 1]);
  assert.deepEqual(calibrationViewIds([{ id: 1, camera }, ...views]), [3, 7, 5, 1]);
});

test('the turn is between the optical axes of the lowest and the highest view id', () => {
  const at = (id: number, angle: number) => ({ id, pose: { R: rotationExp([0, angle, 0]) } });
  const views = [at(12, (5 + 30) * deg), at(4, 5 * deg), at(8, 70 * deg)];
  assert.ok(Math.abs(firstLastTurnDeg(views) - 30) < 1e-9);
  assert.equal(firstLastTurnDeg([at(1, 0.3)]), 0);
  assert.ok(Number.isNaN(firstLastTurnDeg([])));
});

test('rows and the truncated score', () => {
  const result = {
    reconstruction: {
      views: [
        { id: 2, pose: { R: rotationExp([0, 0, 10 * deg]) } },
        { id: 1, pose: { R: rotationExp([10 * deg, 0, 0]) } },
      ],
      landmarks: [{ observations: [1, 2] }, { observations: [1, 2, 3] }],
    },
    diagnostics: {
      bundleAdjustments: [{ finalRmsePx: 0.9 }, { finalRmsePx: 0.25 }],
      meanReprojectionErrorPx: 0.2,
    },
  };
  const row = sweepRow(0.8, result, 1234.4);
  assert.deepEqual(
    { ...row, firstLastTurnDeg: 0 },
    {
      ratio: 0.8,
      views: 2,
      landmarks: 2,
      observations: 5,
      rmsePx: 0.25,
      meanErrorPx: 0.2,
      firstLastTurnDeg: 0,
      ms: 1234,
    },
  );
  assert.ok(Math.abs(row.firstLastTurnDeg - 10) < 1e-9);
  // n = 100, the most observations of the 5-view runs: 20 dropped observations cost 1.5² each.
  assert.equal(HUBER_PX, 1.5);
  const at = (ratio: number, observations: number, rmsePx: number): SweepRow => ({
    ...row,
    ratio,
    views: 5,
    observations,
    rmsePx,
  });
  const scored = scoreTable([at(0.5, 100, 0.2), at(0.8, 80, 0.1), { ratio: 1.2, error: 'x' }]);
  assert.deepEqual(
    scored.map((s) => s.ratio),
    [0.5, 0.8],
  );
  assert.ok(Math.abs(scored[0]!.score - 0.04) < 1e-12);
  assert.ok(Math.abs(scored[1]!.score - (80 * 0.01 + 20 * 2.25) / 100) < 1e-12);
  assert.equal(
    scoreLine(scored[1]!),
    'ratio 0.8 truncated score 0.4580 px² (rmse 0.100 px, observations 80)',
  );
});

test('the flags', () => {
  assert.deepEqual(parseOptions(['--model', 'm.json', '--ratios', '0.3,0.5835', '--views', '5']), {
    model: 'm.json',
    ratios: [0.3, 0.5835],
    views: 5,
    minMatches: 32,
    finalBundleIterations: 8,
    out: undefined,
    models: undefined,
  });
  const all = parseOptions([
    '--model=m.json',
    '--ratios=0.97',
    '--min-matches=16',
    '--final=32',
    '--out=o.json',
    '--models=dir',
  ]);
  assert.deepEqual(
    [all.views, all.minMatches, all.finalBundleIterations, all.out, all.models],
    [undefined, 16, 32, 'o.json', 'dir'],
  );
  assert.throws(() => parseOptions(['--model', 'm.json']), /--ratios/);
  assert.throws(() => parseOptions(['--model', 'm.json', '--ratios', '0.3,x']), /--ratios/);
  assert.throws(() => parseOptions(['--model', 'm.json', '--ratios', '1', '--views', '2.5']));
  assert.throws(() => parseOptions(['--model', 'm.json', '--ratios', '1', '--starts', '1']));
  // Number('') is 0; an empty value and zero are refused, as a negative one is.
  for (const flag of ['--views=', '--views=0', '--min-matches=', '--min-matches=-1', '--final= '])
    assert.throws(
      () => parseOptions(['--model', 'm.json', '--ratios', '1', flag]),
      /must be a positive whole number/,
      flag,
    );
});

test('one fixed-focal run of the mapper on the tracks of a small exact scene', () => {
  // tests/scene.ts: views 10, 20 and 30 of 40 exact points; fx 850, fy 870, cx 480, so that
  // the ratio 850 / 961 (961 = 2·cx + 1) gives the true camera.
  const truth = scene({ count: 40, views: 3 });
  const exported = {
    views: truth.reconstruction.views.map((v, i) => ({ ...v, time: i, width: 961, height: 541 })),
    landmarks: truth.reconstruction.landmarks,
  };
  const { images, pairs } = tracksToInput(
    exported,
    exported.views.map((v) => v.id),
  );
  assert.equal(pairs.length, 3);
  const run = runAtRatio(images, pairs, 850 / 961);
  assert.ok(run.reconstruction);
  const row = run.row as SweepRow;
  assert.deepEqual(
    [row.ratio, row.views, row.landmarks, row.observations],
    [850 / 961, 3, 40, 120],
  );
  assert.ok(row.rmsePx < 1e-6 && row.meanErrorPx < 1e-6);
  // The turn does not depend on the gauge of the map.
  assert.ok(Math.abs(row.firstLastTurnDeg - firstLastTurnDeg(truth.reconstruction.views)) < 1e-6);
  const model = JSON.parse(JSON.stringify(runModel(row.ratio, run.reconstruction, exported.views)));
  assert.equal(model.ratio, 850 / 961);
  // The views come in the mapper's registration order, with the export's time and width.
  assert.deepEqual(
    model.views
      .map((v: { id: number; time: number; width: number }) => [v.id, v.time, v.width])
      .sort((a: number[], b: number[]) => a[0]! - b[0]!),
    [
      [10, 0, 961],
      [20, 1, 961],
      [30, 2, 961],
    ],
  );
  assert.equal(model.views[0].pose.R.length, 9);
  assert.equal(model.landmarks.length, 40);
  assert.equal(model.landmarks[0].observations[0].pixel.length, 2);
  // Without pairs the mapper builds no map, and the row says why.
  const failed = runAtRatio(images, [], 1);
  assert.equal(failed.reconstruction, undefined);
  assert.match(String(failed.row.error), /^ReconstructionInitializationError/);
});
