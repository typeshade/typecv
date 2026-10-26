// The source views that the dense MVS picks for each reference view, replayed on a lab export:
// `planMvsViews` of src/mvs/gpu/mvs-views.ts, the host view selection of docs/dense-mvs.md 3.2 and 3.3.
//
//   node scripts/measure-mvs-sources.mjs --model typecv-reconstruction.json --out sources.json
//   node scripts/measure-mvs-sources.mjs --model typecv-reconstruction.json --max-sources 4 \
//     --full-angle 8 --min-angle 2 --min-shared 32
//
// --model file.json    the lab's export (`typecv-reconstruction.json`, typecv-reconstruction-v1):
//                      views (id, time, width, height, camera, pose) and landmarks with their
//                      observations. The models of scripts/measure-focal-sweep.mjs --models load
//                      the same way.
// --max-sources n      `maxSources` of `MvsViewOptions` (3.2): the most sources of one reference
//                      (default 6, at most 6).
// --full-angle deg     `fullAngleDeg`, α0 of the angle weight g(α) = min(1, (α / α0)²) (default 5).
// --min-angle deg      `minAngleDeg`, αmin: g(α) = 0 below it (default 1; 0 turns it off).
// --min-shared n       `minSharedLandmarks`: a pair with fewer shared landmarks scores 0
//                      (default 16).
// --out file.json      the options, the normalization, the rows and the summary.
//
// A flag value must be a number; an empty value is refused. `planMvsViews` refuses a value out of
// its range, for example `--max-sources 0`, `--full-angle 0` or `--min-shared 0`.
//
// Printed per reference view: its id and time, its status, its depth range (normalized units),
// and its sources, best first, as id(keyframe gap, time gap, median triangulation angle,
// baseline / depth). The keyframe gap is the difference of the two views' positions in time order
// (1 for neighbours). The median angle is over the landmarks that both views observe. The
// baseline is the distance between the two camera centres, over the median depth of the
// landmarks that the reference observes. Then, over all source links: the median angle, keyframe
// gap and time gap, the largest keyframe gap, and the share of links whose median angle is under
// 5° and under 10°. The median of an even count is the upper one of the two middle values.
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { mvsCameraCenter, mvsDepth, planMvsViews } from '../src/gpu/mvs-views.ts';

/** The command-line options. The view options hold only the flags given. */
export function parseOptions(argv) {
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    options: {
      model: { type: 'string' },
      out: { type: 'string' },
      'max-sources': { type: 'string' },
      'full-angle': { type: 'string' },
      'min-angle': { type: 'string' },
      'min-shared': { type: 'string' },
    },
  });
  const flags = {
    maxSources: 'max-sources',
    fullAngleDeg: 'full-angle',
    minAngleDeg: 'min-angle',
    minSharedLandmarks: 'min-shared',
  };
  const viewOptions = {};
  for (const [option, flag] of Object.entries(flags)) {
    const text = values[flag];
    if (text === undefined) continue;
    // Number('') is 0: an empty value must not pass as zero.
    const value = text.trim() === '' ? NaN : Number(text);
    if (!Number.isFinite(value)) throw new Error(`--${flag} must be a number`);
    viewOptions[option] = value;
  }
  return { model: values.model, out: values.out, viewOptions };
}

/**
 * The views in time order: by time, equal times in input order, and the views without a time
 * last, in input order. The lab's session receives its keyframes in this order.
 */
export function timeOrder(views) {
  const time = (view) => (Number.isFinite(view.time) ? view.time : Infinity);
  return views
    .map((view, index) => ({ view, index }))
    .sort((a, b) => time(a.view) - time(b.view) || a.index - b.index)
    .map(({ view }) => view);
}

/** The position of each view id in time order (`timeOrder`). */
export function keyframeRanks(views) {
  return new Map(timeOrder(views).map((view, rank) => [view.id, rank]));
}

/** The middle one of the finite values, the upper one of two for an even count. NaN without one. */
function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? sorted[sorted.length >> 1] : NaN;
}

/** The angle in degrees between the 3-vectors `u` and `v`: atan2(|u × v|, u · v). */
export function vectorAngleDeg(u, v) {
  const cross = Math.hypot(
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  );
  return (Math.atan2(cross, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) * 180) / Math.PI;
}

/** The angle in degrees at `p` between the rays to the camera centres `a` and `b`. */
function triangulationAngleDeg(p, a, b) {
  return vectorAngleDeg(
    [a[0] - p[0], a[1] - p[1], a[2] - p[2]],
    [b[0] - p[0], b[1] - p[1], b[2] - p[2]],
  );
}

/**
 * The sources of each reference view in `plan` (from `planMvsViews(scene)`), measured on `scene`
 * in its own units: one row per input view, in input order, and the summary over all links.
 */
export function summarizeSources(scene, plan) {
  const { views, landmarks } = scene;
  const index = new Map(views.map((view, i) => [view.id, i]));
  const rank = keyframeRanks(views);
  const centers = views.map(mvsCameraCenter);
  // The landmarks that each view observes, once per view.
  const observed = views.map(() => new Set());
  landmarks.forEach((landmark, k) => {
    for (const { viewId } of landmark.observations) observed[index.get(viewId)]?.add(k);
  });
  const medianDepth = views.map((view, i) =>
    median([...observed[i]].map((k) => mvsDepth(view, landmarks[k].xyz)).filter((z) => z > 0)),
  );
  const rows = plan.views.map((entry) => {
    const r = index.get(entry.id),
      view = views[r];
    const sources = entry.sources.map((packed) => {
      const s = plan.packed[packed],
        source = views[s];
      const [fewer, more] =
        observed[r].size <= observed[s].size
          ? [observed[r], observed[s]]
          : [observed[s], observed[r]];
      const shared = [...fewer].filter((k) => more.has(k));
      const a = centers[r],
        b = centers[s];
      return {
        id: source.id,
        keyframeGap: Math.abs(rank.get(source.id) - rank.get(view.id)),
        timeGap: Math.abs(source.time - view.time),
        sharedLandmarks: shared.length,
        medianAngleDeg: median(shared.map((k) => triangulationAngleDeg(landmarks[k].xyz, a, b))),
        baselineOverDepth: Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / medianDepth[r],
      };
    });
    return {
      id: entry.id,
      time: view.time,
      status: entry.status,
      depthRange: entry.depthRange,
      landmarks: entry.landmarks,
      medianDepth: medianDepth[r],
      sources,
    };
  });
  const links = rows.flatMap((row) => row.sources);
  const angles = links.map((link) => link.medianAngleDeg);
  const share = (limit) =>
    links.length ? angles.filter((angle) => angle < limit).length / links.length : NaN;
  const summary = {
    views: rows.length,
    references: rows.filter((row) => row.status === 'ready').length,
    links: links.length,
    medianAngleDeg: median(angles),
    medianKeyframeGap: median(links.map((link) => link.keyframeGap)),
    medianTimeGap: median(links.map((link) => link.timeGap)),
    maxKeyframeGap: links.length
      ? links.reduce((max, link) => Math.max(max, link.keyframeGap), 0)
      : NaN,
    shareBelow5Deg: share(5),
    shareBelow10Deg: share(10),
  };
  return { rows, summary };
}

const fixed = (value, digits) => (Number.isFinite(value) ? value.toFixed(digits) : '-');

/** The printed line of one reference view. */
export function formatRow(row) {
  const range = row.depthRange ? row.depthRange.map((d) => d.toFixed(3)).join('-') : '-';
  const sources = row.sources.map(
    (s) =>
      `${s.id}(${s.keyframeGap}, ${fixed(s.timeGap, 1)} s, ${fixed(s.medianAngleDeg, 1)}°, ` +
      `${fixed(s.baselineOverDepth, 3)})`,
  );
  return [
    `${String(row.id).padStart(4)} ${fixed(row.time, 1).padStart(6)} s`,
    row.status,
    range,
    sources.join(' ') || '-',
  ].join(' | ');
}

/** The printed lines of the summary. */
export function formatSummary(summary) {
  return [
    `median over ${summary.links} source links: triangulation angle ` +
      `${fixed(summary.medianAngleDeg, 2)}°, keyframe gap ${summary.medianKeyframeGap}, time gap ` +
      `${fixed(summary.medianTimeGap, 2)} s; largest keyframe gap ${summary.maxKeyframeGap}`,
    `share of source links with a median triangulation angle under 5°: ` +
      `${fixed(summary.shareBelow5Deg, 3)}, under 10°: ${fixed(summary.shareBelow10Deg, 3)}`,
  ];
}

function main(argv) {
  const { model, out, viewOptions } = parseOptions(argv);
  if (!model) throw new Error('--model is required');
  const exported = JSON.parse(readFileSync(model, 'utf8'));
  const scene = { views: exported.views, landmarks: exported.landmarks };
  const plan = planMvsViews(scene, viewOptions);
  const { rows, summary } = summarizeSources(scene, plan);
  const { center, scale } = plan.normalization;
  console.log(
    `views ${summary.views} (${summary.references} with a depth map), landmarks ` +
      `${scene.landmarks.length}, normalization scale ${scale.toFixed(3)}, centre ` +
      `${[...center].map((x) => x.toFixed(3)).join(', ')}`,
  );
  console.log(
    `${'id'.padStart(4)} ${'time'.padStart(8)} | status | depth range (normalized) | sources, ` +
      'best first: id(keyframe gap, time gap, median triangulation angle, baseline / depth)',
  );
  for (const row of rows) console.log(formatRow(row));
  for (const line of formatSummary(summary)) console.log(line);
  if (out)
    writeFileSync(
      out,
      JSON.stringify(
        {
          model,
          viewOptions,
          normalization: { center: [...center], scale },
          rows,
          summary,
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
