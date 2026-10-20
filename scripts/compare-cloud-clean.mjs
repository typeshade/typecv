// Compares the CPU reference of the dense cloud cleanup with scikit-learn on one cloud
// (docs/dense-mvs.md, 3.12). Needs Python with numpy and scikit-learn (SKLEARN_PATH may name an
// extra import directory). Prints a JSON record of the agreement.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  dbscan,
  meanNeighbourDistances,
  voxelDownsample,
} from '../src/reconstruction/cloud-clean.ts';
import { cleanCloud } from '../tests/typeshade/cloud-clean-cases.mjs';

const python = process.env.PYTHON ?? 'python3';
const { cloud } = cleanCloud();
const k = 16,
  eps = 0.03,
  minPoints = 5,
  voxel = 0.05;
const dir = mkdtempSync(join(tmpdir(), 'cloud-clean-'));
writeFileSync(join(dir, 'cloud.f32'), Buffer.from(cloud.positions.buffer));
execFileSync(
  python,
  [
    '-I',
    new URL('./compare-cloud-clean.py', import.meta.url).pathname,
    join(dir, 'cloud.f32'),
    join(dir, 'out.json'),
    k,
    eps,
    minPoints,
    voxel,
  ],
  { stdio: 'inherit', env: process.env },
);
const reference = JSON.parse(readFileSync(join(dir, 'out.json'), 'utf8'));
const mine = meanNeighbourDistances(cloud.positions, k);
let worst = 0;
mine.forEach((d, i) => (worst = Math.max(worst, Math.abs(d - reference.meanDistances[i]) / d)));
const labels = dbscan(cloud.positions, { eps, minPoints });
const labelDifferences = labels.labels.reduce((n, l, i) => n + (l !== reference.labels[i]), 0);
const voxels = voxelDownsample(cloud, { voxelSize: voxel }).cloud.count;
console.log(
  JSON.stringify({
    sklearn: reference.sklearn,
    points: cloud.count,
    meanDistanceWorstRelativeDifference: worst,
    dbscanClusters: [labels.clusters, Math.max(...reference.labels) + 1],
    dbscanLabelDifferences: labelDifferences,
    voxelCount: [voxels, reference.voxelCount],
  }),
);
