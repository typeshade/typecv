import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { deviceLabel, resultEntryProblem, resultFileNames } from './src/lib/sample-catalog.ts';
import {
  catalogEntry,
  readManifest,
  uploadPlan,
  withResultEntries,
} from '../scripts/lab-results.mjs';

const manifestResult = (overrides = {}) => ({
  id: 'south-building-dense',
  name: 'South Building, dense',
  source: 'south-building',
  description: { en: 'A dense run.', ko: 'Dense 실행.' },
  files: {
    json: 'typecv-reconstruction.json',
    ply: 'typecv-reconstruction.ply',
    dense: 'typecv-dense.ply',
    mesh: 'typecv-mesh.ply',
  },
  thumbnail: 'thumbnail.jpg',
  size: 12345678,
  run: { date: '2026-10-20', device: 'swiftshader', commit: 'f00dcafe' },
  ...overrides,
});

test('a result entry has the schema the lab loads', () => {
  const entry = catalogEntry(manifestResult());
  assert.deepEqual(entry, {
    id: 'south-building-dense',
    kind: 'result',
    name: 'South Building, dense',
    dir: '/sample/results/south-building-dense/',
    files: manifestResult().files,
    thumbnail: 'thumbnail.jpg',
    size: 12345678,
    source: 'south-building',
    description: { en: 'A dense run.', ko: 'Dense 실행.' },
    remote: true,
    run: { date: '2026-10-20', device: 'swiftshader', commit: 'f00dcafe' },
  });
  assert.equal(resultEntryProblem(entry), undefined);
  // The View button downloads the JSON, the dense cloud and the mesh, not the sparse PLY.
  assert.deepEqual(resultFileNames(entry), [
    'typecv-reconstruction.json',
    'typecv-dense.ply',
    'typecv-mesh.ply',
  ]);
  // The least an entry needs: an id, a name, the JSON and the run.
  const least = catalogEntry({ id: 'a', name: 'A', files: { json: 'r.json' }, run: entry.run });
  assert.deepEqual(resultFileNames(least), ['r.json']);
});

test('a result entry with a wrong field is refused, and the field is named', () => {
  const entry = catalogEntry(manifestResult());
  const bad = (change) => resultEntryProblem({ ...entry, ...change });
  assert.equal(bad({ kind: 'video' }), 'kind');
  assert.equal(bad({ id: 'Has Space' }), 'id');
  assert.equal(bad({ dir: '/sample/other/' }), 'dir');
  assert.equal(bad({ files: {} }), 'files.json');
  assert.equal(bad({ files: { json: 'r.ply' } }), 'files.json');
  assert.equal(bad({ files: { json: 'r.json', dense: '../x.ply' } }), 'files.dense');
  assert.equal(bad({ files: { json: 'r.json', mesh: 'm.json' } }), 'files.mesh');
  assert.equal(bad({ remote: false }), 'remote');
  assert.equal(bad({ size: -1 }), 'size');
  assert.equal(bad({ run: { ...entry.run, device: 'RTX 4090' } }), undefined);
  assert.equal(bad({ run: { ...entry.run, device: '' } }), 'run.device');
  assert.equal(deviceLabel('swiftshader'), 'SwiftShader');
  assert.equal(deviceLabel('RTX 4090'), 'RTX 4090');
  assert.equal(bad({ run: { ...entry.run, date: '20 Oct' } }), 'run.date');
  assert.equal(bad({ run: { ...entry.run, commit: 'HEAD' } }), 'run.commit');
  assert.equal(bad({ run: undefined }), 'run');
  assert.throws(() => catalogEntry(manifestResult({ id: 'Bad Id' })), /bad field id/);
});

test('the catalog lists every manifest result and nothing else of that kind', () => {
  const manifest = readManifest();
  const catalog = JSON.parse(readFileSync(new URL('./public/sample/index.json', import.meta.url)));
  assert.deepEqual(withResultEntries(catalog, manifest), catalog, 'run `lab-results.mjs catalog`');
  for (const entry of catalog.samples.filter((e) => e.kind === 'result'))
    assert.equal(resultEntryProblem(entry), undefined, entry.id);
  // Results go after the samples, a stale entry goes away, a repeated id is refused.
  const two = { ...manifest, results: [manifestResult()] };
  const withOne = withResultEntries({ ...catalog, samples: [...catalog.samples] }, two);
  assert.equal(withOne.samples.at(-1).id, 'south-building-dense');
  assert.deepEqual(withResultEntries(withOne, manifest).samples, catalog.samples);
  assert.throws(
    () =>
      withResultEntries(catalog, { ...manifest, results: [manifestResult(), manifestResult()] }),
    /repeats a result id/,
  );
  // A sample id never collides with a result id.
  const ids = withOne.samples.map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length);
});

test('publish uploads the files of a result directory with their content types', () => {
  const dir = mkdtempSync(join(tmpdir(), 'result-'));
  try {
    assert.throws(() => uploadPlan(dir, 'run-1'), /no typecv-reconstruction\.json/);
    for (const name of [
      'typecv-reconstruction.json',
      'typecv-dense.ply',
      'typecv-mesh.ply',
      'thumbnail.jpg',
      'notes.txt',
    ])
      writeFileSync(join(dir, name), 'x');
    const plan = uploadPlan(dir, 'run-1');
    assert.deepEqual(
      plan.map(({ key, type }) => [key, type]),
      [
        ['sample/results/run-1/typecv-reconstruction.json', 'application/json'],
        ['sample/results/run-1/typecv-dense.ply', 'application/octet-stream'],
        ['sample/results/run-1/typecv-mesh.ply', 'application/octet-stream'],
        ['sample/results/run-1/thumbnail.jpg', 'image/jpeg'],
      ],
    );
    assert.throws(() => uploadPlan(dir, '../x'), /lowercase letters/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
