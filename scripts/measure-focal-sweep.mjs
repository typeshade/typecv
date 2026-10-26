// Fixed-focal runs of the CPU reference mapper on the tracks of a lab export: the method of
// docs/auto-focal.md 2.1 for record 4a8aa2ff. Each run holds the focal at one ratio, and the
// runs are scored as the focal search scores its grid points (`scoreCandidates`,
// docs/auto-focal.md 3.2).
//
//   node scripts/measure-focal-sweep.mjs --model typecv-reconstruction.json --views 5 \
//     --ratios 0.3,0.798,2.8074
//   node scripts/measure-focal-sweep.mjs --model typecv-reconstruction.json --ratios 0.5835,0.97 \
//     --out sweep.json --models sweep-models
//
// --model file.json    the lab's export (`typecv-reconstruction.json`, typecv-reconstruction-v1):
//                      views (id, time, camera) and landmarks with their observations.
// --ratios a,b,...     the focal ratios (focal / width) of the runs, run in this order.
// --views n            only the first n views by time: the calibration subset of the search when
//                      the views come in time order and are match-connected (docs/auto-focal.md
//                      4.1, step 1). Default: every view.
// --min-matches n      the fewest matches that a pair keeps (default 32, docs/auto-focal.md 2.1).
// --final n            `finalBundleIterations` of the mapper (default 8, the lab's).
// --out file.json      the input size, the controls, the rows and the scored table.
// --models dir         writes the model of each run to <dir>/ratio-<ratio>.model.json: the
//                      registered views (with the export's time, width and height) and the
//                      landmarks. scripts/measure-mvs-sources.mjs --model reads it.
//
// The input: each track gives one keypoint in each selected view that it observes (its first
// observation there) and one match in each pair of those views, in the order of the export's
// landmarks. The images keep the export's view order. Each camera keeps the export's principal
// point and distortion; its focal is `cameraAtRatio` of src/reconstruction/focal-search.ts:
// fx = ratio × (2·cx + 1), with fy / fx kept. The mapper (`reconstructIncremental`) runs with the
// lab's numerical controls (`LAB` of scripts/measure-focal.mjs) and `refineSharedFocal: false`.
// Caveat: the tracks survived the lab's run at the export's focal, so they favour that focal.
// This is not a replay of the search's grid points: `focalSearchSteps` runs them with
// `finalBundleIterations` = `bundleIterations` (4 in the lab, `--final 4` here), with
// `rotationTracking: false` and with one shared pair verification per pair.
//
// Printed: the input size, one JSON row per run (registered views, landmarks, observations, RMS
// error of the last bundle adjustment, mean reprojection error, the turn between the optical axes
// of the views with the lowest and the highest id, time in ms), then the truncated score of each
// run in px²: among the runs that registered the same number of views, each observation that
// another run kept and this run dropped costs τ², with τ = maxErrorPx / 2 = 1.5 px, the Huber
// threshold of the mapper's bundle adjustment. The scores compare the runs of one call only.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { reconstructIncremental } from '../src/index.ts';
import { cameraAtRatio, scoreCandidates } from '../src/reconstruction/focal-search.ts';
import { LAB } from './measure-focal.mjs';

/** The fewest matches that a pair keeps (docs/auto-focal.md 2.1). */
export const MIN_PAIR_MATCHES = 32;
/** τ of the truncated score: the Huber threshold of the mapper's bundle adjustment. */
export const HUBER_PX = LAB.maxErrorPx / 2;

/** A positive integer flag value, or the fallback when the flag is absent. */
function integer(name, value, fallback) {
  if (value === undefined) return fallback;
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n < 0) throw new Error(`--${name} must be a whole number`);
  return n;
}

/** The command-line options. Throws on an unknown flag or an invalid value. */
export function parseOptions(argv) {
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    options: {
      model: { type: 'string' },
      ratios: { type: 'string' },
      views: { type: 'string' },
      'min-matches': { type: 'string' },
      final: { type: 'string' },
      out: { type: 'string' },
      models: { type: 'string' },
    },
  });
  if (!values.model || !values.ratios) throw new Error('--model and --ratios are required');
  const ratios = values.ratios.split(',').map(Number);
  if (!ratios.every((ratio) => ratio > 0 && Number.isFinite(ratio)))
    throw new Error('--ratios must be positive numbers, separated by commas');
  return {
    model: values.model,
    ratios,
    views: integer('views', values.views, undefined),
    minMatches: integer('min-matches', values['min-matches'], MIN_PAIR_MATCHES),
    finalBundleIterations: integer('final', values.final, LAB.finalBundleIterations),
    out: values.out,
    models: values.models,
  };
}

/**
 * The ids of the first `n` views by time: the calibration subset of the search. Equal times keep
 * the export's order. Views without a time come last.
 */
export function calibrationViewIds(views, n) {
  const time = (view) => (Number.isFinite(view.time) ? view.time : Infinity);
  return views
    .map((view, index) => ({ view, index }))
    .sort((a, b) => time(a.view) - time(b.view) || a.index - b.index)
    .slice(0, n)
    .map(({ view }) => view.id);
}

/**
 * The mapper input that the tracks of `exported` give in the views `viewIds`: the images (the
 * export's cameras, in the export's view order) and the pairs with at least `minMatches` matches.
 * A track gives one keypoint in each of its views in `viewIds` (the first observation of a view
 * that it repeats), also when it has only one view there, and one match in each pair of those
 * views, the lower view id first.
 */
export function tracksToInput(exported, viewIds, { minMatches = MIN_PAIR_MATCHES } = {}) {
  const ids = new Set(viewIds);
  const views = exported.views.filter((view) => ids.has(view.id));
  const keypoints = new Map(views.map((view) => [view.id, []]));
  const matches = new Map();
  for (const landmark of exported.landmarks) {
    const observations = landmark.observations
      .filter((observation) => keypoints.has(observation.viewId))
      .sort((a, b) => a.viewId - b.viewId);
    const track = [];
    for (const { viewId, pixel } of observations) {
      if (track.length && track.at(-1).viewId === viewId) continue;
      const list = keypoints.get(viewId);
      track.push({ viewId, index: list.length / 2 });
      list.push(pixel[0], pixel[1]);
    }
    for (let a = 0; a < track.length; a++)
      for (let b = a + 1; b < track.length; b++) {
        const key = `${track[a].viewId}:${track[b].viewId}`;
        let list = matches.get(key);
        if (!list) matches.set(key, (list = []));
        list.push(track[a].index, track[b].index);
      }
  }
  const pairs = [];
  let droppedPairs = 0;
  for (const [key, list] of matches) {
    if (list.length / 2 < minMatches) {
      droppedPairs++;
      continue;
    }
    const [first, second] = key.split(':').map(Number);
    pairs.push({ first, second, matches: Uint32Array.from(list) });
  }
  const images = views.map((view) => ({
    id: view.id,
    camera: view.camera,
    keypoints: Float64Array.from(keypoints.get(view.id)),
  }));
  return { images, pairs, droppedPairs };
}

/**
 * The angle in degrees between the optical axes (the third rows of R) of the views with the
 * lowest and the highest id. The lab's view ids are frame indices, so this is the turn from the
 * first registered frame to the last. NaN without a view.
 */
export function firstLastTurnDeg(views) {
  if (!views.length) return NaN;
  const sorted = [...views].sort((a, b) => a.id - b.id);
  const a = sorted[0].pose.R,
    b = sorted.at(-1).pose.R;
  const cross = Math.hypot(
    a[7] * b[8] - a[8] * b[7],
    a[8] * b[6] - a[6] * b[8],
    a[6] * b[7] - a[7] * b[6],
  );
  return (Math.atan2(cross, a[6] * b[6] + a[7] * b[7] + a[8] * b[8]) * 180) / Math.PI;
}

/** The row of one run from the mapper's result and its time in ms. */
export function sweepRow(ratio, result, ms) {
  const { reconstruction, diagnostics } = result;
  return {
    ratio,
    views: reconstruction.views.length,
    landmarks: reconstruction.landmarks.length,
    observations: reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
    rmsePx: diagnostics.bundleAdjustments.at(-1)?.finalRmsePx ?? Infinity,
    meanErrorPx: diagnostics.meanReprojectionErrorPx,
    firstLastTurnDeg: firstLastTurnDeg(reconstruction.views),
    ms: Math.round(ms),
  };
}

/** The truncated score (`scoreCandidates`) of each run that built a map, in the order of `rows`. */
export function scoreTable(rows, huberPx = HUBER_PX) {
  return scoreCandidates(
    rows
      .filter((row) => row.error === undefined)
      .map(({ ratio, views, rmsePx, observations }) => ({ ratio, views, rmsePx, observations })),
    huberPx,
  );
}

/** The printed line of one scored run. */
export function scoreLine(candidate) {
  const { ratio, score, rmsePx, observations } = candidate;
  return (
    `ratio ${ratio} truncated score ${score.toFixed(4)} px² ` +
    `(rmse ${rmsePx.toFixed(3)} px, observations ${observations})`
  );
}

/**
 * One run of the CPU reference mapper with the focal held at `ratio`: its row and its model, or a
 * row with the error when the mapper builds no map.
 */
export function runAtRatio(images, pairs, ratio, { finalBundleIterations } = {}) {
  const started = performance.now();
  try {
    const result = reconstructIncremental(
      images.map((image) => ({ ...image, camera: cameraAtRatio(image.camera, ratio) })),
      pairs,
      {
        ...LAB,
        finalBundleIterations: finalBundleIterations ?? LAB.finalBundleIterations,
        refineSharedFocal: false,
      },
    );
    return {
      row: sweepRow(ratio, result, performance.now() - started),
      reconstruction: result.reconstruction,
    };
  } catch (error) {
    return { row: { ratio, error: `${error?.name ?? 'Error'}: ${error?.message ?? error}` } };
  }
}

/** The model file of one run: its registered views, with the export's time, width and height. */
export function runModel(ratio, reconstruction, exportViews) {
  const source = new Map(exportViews.map((view) => [view.id, view]));
  return {
    ratio,
    views: reconstruction.views.map((view) => ({
      id: view.id,
      time: source.get(view.id)?.time,
      width: source.get(view.id)?.width,
      height: source.get(view.id)?.height,
      camera: view.camera,
      pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
    })),
    landmarks: reconstruction.landmarks.map((landmark) => ({
      xyz: Array.from(landmark.xyz),
      observations: landmark.observations.map(({ viewId, featureId, pixel }) => ({
        viewId,
        featureId,
        pixel: Array.from(pixel),
      })),
    })),
  };
}

function main(argv) {
  const options = parseOptions(argv);
  const exported = JSON.parse(readFileSync(options.model, 'utf8'));
  const viewIds =
    options.views === undefined
      ? exported.views.map((view) => view.id)
      : calibrationViewIds(exported.views, options.views);
  const { images, pairs, droppedPairs } = tracksToInput(exported, viewIds, options);
  const keypoints = images.reduce((n, image) => n + image.keypoints.length / 2, 0);
  console.log(
    `views ${images.length}, pairs ${pairs.length} (${droppedPairs} with fewer than ` +
      `${options.minMatches} matches dropped), keypoints ${keypoints}`,
  );
  if (options.models) mkdirSync(options.models, { recursive: true });
  const rows = [];
  for (const ratio of options.ratios) {
    const { row, reconstruction } = runAtRatio(images, pairs, ratio, options);
    rows.push(row);
    console.log(JSON.stringify(row));
    if (options.models && reconstruction)
      writeFileSync(
        join(options.models, `ratio-${ratio}.model.json`),
        JSON.stringify(runModel(ratio, reconstruction, exported.views)),
      );
  }
  const scored = scoreTable(rows);
  for (const candidate of scored) console.log(scoreLine(candidate));
  if (options.out)
    writeFileSync(
      options.out,
      JSON.stringify(
        {
          model: options.model,
          viewIds,
          pairs: pairs.length,
          droppedPairs,
          keypoints,
          minMatches: options.minMatches,
          controls: {
            ...LAB,
            finalBundleIterations: options.finalBundleIterations,
            refineSharedFocal: false,
          },
          huberPx: HUBER_PX,
          rows,
          scored,
        },
        null,
        1,
      ),
    );
}

if (import.meta.main)
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
