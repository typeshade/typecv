/**
 * The reference side of the OpenCV comparison (`docs/klt-tracking.md` 7.3). It is called by
 * `scripts/klt-opencv-compare.py` and reads and writes plain files.
 *
 *   node scripts/klt-opencv-reference.mjs synthetic <dir> <width> <height> <dx> <dy>
 *     writes <dir>/a.gray and <dir>/b.gray (8-bit, row-major): the scene of 7.1 and its shift.
 *   node scripts/klt-opencv-reference.mjs track <dir> <width> <height>
 *     reads <dir>/a.gray, <dir>/b.gray and <dir>/points.json ([x, y, ...]) and writes
 *     <dir>/reference.json: the tracks of the reference (`trackKlt`, the defaults of the frame
 *     width) and the picks of the selection (`kltSelectReference`) on the first frame.
 *
 * It imports `dist/internal/gpu.js`, so `npm run build` runs first.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import {
  buildKltPyramid,
  kltEigenMapReference,
  kltCellGrid,
  kltOptionsFor,
  kltScharrGradient,
  kltSelectReference,
  trackKlt,
} from '../dist/internal/gpu.js';
import { renderScene } from '../tests/klt-cases.ts';

const [mode, dir, w, h, dx, dy] = process.argv.slice(2),
  width = Number(w),
  height = Number(h);
const toBytes = (gray) =>
  Uint8Array.from(gray, (v) => Math.round(Math.min(Math.max(v, 0), 1) * 255));
const fromBytes = (file) => Float32Array.from(readFileSync(file), (v) => v / 255);

if (mode === 'synthetic') {
  const scene = { width, height };
  writeFileSync(`${dir}/a.gray`, toBytes(renderScene(scene)));
  writeFileSync(`${dir}/b.gray`, toBytes(renderScene(scene, { dx: Number(dx), dy: Number(dy) })));
} else if (mode === 'track') {
  const options = kltOptionsFor(width),
    a = fromBytes(`${dir}/a.gray`),
    b = fromBytes(`${dir}/b.gray`),
    points = JSON.parse(readFileSync(`${dir}/points.json`, 'utf8')),
    pa = buildKltPyramid(a, width, height, options.levels),
    pb = buildKltPyramid(b, width, height, options.levels),
    result = trackKlt(pa, pb, points, options),
    eig = kltEigenMapReference(
      kltScharrGradient(pa, pa.gray).subarray(0, 2 * width * height),
      width,
      height,
      options.halfWindow,
    ),
    none = new Float32Array(
      2 *
        kltCellGrid(width, height, options.minDistancePx).cellsX *
        kltCellGrid(width, height, options.minDistancePx).cellsY,
    ).fill(-1),
    picks = kltSelectReference(eig, width, height, options, none).picks;
  const n = points.length / 2,
    out = { n, alive: [], x: [], y: [], reason: [], picks: picks.map((p) => [p.x, p.y]) };
  for (let k = 0; k < n; k++) {
    out.alive.push(result.alive[k]);
    out.x.push(result.records[4 * k]);
    out.y.push(result.records[4 * k + 1]);
    out.reason.push(result.reason[k]);
  }
  writeFileSync(`${dir}/reference.json`, JSON.stringify(out));
} else throw new Error('Usage: see the header of this file');
