#!/usr/bin/env node
// The lab's precomputed results: finished reconstructions that open in the viewer without a run.
// `website/samples/results.json` names each one. The files live in the R2 bucket
// `typecv-samples` under `sample/results/<id>/`, and the site's Worker serves them from
// `/sample/results/<id>/...`.
//
//   node scripts/lab-results.mjs publish <dir> [--id <id>] [--dry-run]
//   node scripts/lab-results.mjs catalog
//
// `publish` uploads the files of one result directory with `wrangler r2 object put --remote`. It
// needs CLOUDFLARE_API_TOKEN (or a wrangler login). `<dir>` holds `typecv-reconstruction.json`
// and, optionally, `typecv-reconstruction.ply`, `typecv-dense.ply`, `typecv-mesh.ply` and one
// image `thumbnail.jpg`. The id is the directory name unless `--id` is given. `--dry-run` lists
// the uploads and sends nothing.
//
// `catalog` writes the entries of the manifest into `website/public/sample/index.json`, after the
// samples, and removes result entries that the manifest no longer names.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { resultEntryProblem } from '../website/src/lib/sample-catalog.ts';

const root = new URL('..', import.meta.url).pathname;
const MANIFEST = join(root, 'website/samples/results.json');
const CATALOG = join(root, 'website/public/sample/index.json');
/** `wrangler r2 object put` refuses a file over 300 MiB. */
const MAX_BYTES = 300 * 1024 * 1024;

/** The upload name of each file a result directory may hold, and its content type. */
const UPLOADS = [
  { name: 'typecv-reconstruction.json', type: 'application/json' },
  { name: 'typecv-reconstruction.ply', type: 'application/octet-stream' },
  { name: 'typecv-dense.ply', type: 'application/octet-stream' },
  { name: 'typecv-mesh.ply', type: 'application/octet-stream' },
  { name: 'thumbnail.jpg', type: 'image/jpeg' },
];

export function readManifest(path = MANIFEST) {
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  if (manifest.format !== 'typecv-lab-results-v1') throw new Error('Unknown manifest format.');
  if (!Array.isArray(manifest.results)) throw new Error('The manifest has no results list.');
  return manifest;
}

/** The catalog entry of one manifest result. */
export function catalogEntry(result) {
  const entry = {
    id: result.id,
    kind: 'result',
    name: result.name,
    dir: `/sample/results/${result.id}/`,
    files: result.files,
    ...(result.thumbnail && { thumbnail: result.thumbnail }),
    ...(result.size !== undefined && { size: result.size }),
    ...(result.source && { source: result.source }),
    ...(result.description && { description: result.description }),
    remote: true,
    run: result.run,
  };
  const problem = resultEntryProblem(entry);
  if (problem) throw new Error(`Result "${result.id}": bad field ${problem}.`);
  return entry;
}

/** The catalog with its result entries replaced by the manifest's, after the samples. */
export function withResultEntries(catalog, manifest) {
  const ids = manifest.results.map((result) => result.id);
  if (new Set(ids).size !== ids.length) throw new Error('The manifest repeats a result id.');
  const samples = catalog.samples.filter((entry) => entry.kind !== 'result');
  return { ...catalog, samples: [...samples, ...manifest.results.map(catalogEntry)] };
}

/** The uploads of one result directory: file, bucket key and content type. */
export function uploadPlan(dir, id) {
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(id))
    throw new Error(`The id "${id}" must be lowercase letters, digits and hyphens.`);
  const present = new Set(readdirSync(dir));
  if (!present.has(UPLOADS[0].name))
    throw new Error(`${dir} holds no ${UPLOADS[0].name}; nothing to publish.`);
  const plan = UPLOADS.filter((upload) => present.has(upload.name)).map((upload) => {
    const file = join(dir, upload.name);
    const size = statSync(file).size;
    if (size > MAX_BYTES) throw new Error(`${upload.name} is over 300 MiB; wrangler refuses it.`);
    return { file, size, key: `sample/results/${id}/${upload.name}`, type: upload.type };
  });
  return plan;
}

function upload(bucket, item) {
  const result = spawnSync(
    'npx',
    [
      ...['wrangler', 'r2', 'object', 'put', `${bucket}/${item.key}`],
      ...['--file', item.file, '--content-type', item.type],
      ...['--cache-control', 'public, max-age=86400', '--remote'],
    ],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] },
  );
  if (result.status !== 0)
    throw new Error(`Upload of ${item.key} failed:\n${result.stdout}\n${result.stderr}`);
}

function main() {
  const [command, ...rest] = process.argv.slice(2);
  if (command === 'catalog') {
    const manifest = readManifest();
    const catalog = JSON.parse(readFileSync(CATALOG, 'utf8'));
    writeFileSync(CATALOG, JSON.stringify(withResultEntries(catalog, manifest), null, 2) + '\n');
    console.log(`${manifest.results.length} result entries written to the catalog`);
    return;
  }
  if (command === 'publish') {
    const flag = rest.indexOf('--id');
    const dryRun = rest.includes('--dry-run');
    const dir = rest.find((arg, i) => !arg.startsWith('--') && rest[i - 1] !== '--id');
    if (!dir || !existsSync(dir)) {
      console.error('usage: lab-results.mjs publish <dir> [--id <id>] [--dry-run]');
      process.exitCode = 2;
      return;
    }
    const id = flag >= 0 ? rest[flag + 1] : basename(resolve(dir));
    const plan = uploadPlan(dir, id);
    const bucket = readManifest().bucket;
    for (const item of plan) {
      console.log(`${dryRun ? 'would upload' : 'uploading'} ${item.key} (${item.size} B)`);
      if (!dryRun) upload(bucket, item);
    }
    const total = plan.reduce((sum, item) => sum + item.size, 0);
    console.log(
      `${id}: ${plan.length} files, ${total} B${dryRun ? ' (dry run)' : ' uploaded'}.\n` +
        `Add "${id}" to website/samples/results.json (size ${total}), then run: ` +
        'node scripts/lab-results.mjs catalog',
    );
    return;
  }
  console.error('usage: lab-results.mjs publish <dir> [--id <id>] [--dry-run] | catalog');
  process.exitCode = 2;
}

if (import.meta.url === `file://${process.argv[1]}`) main();
