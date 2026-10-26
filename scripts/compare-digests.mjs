// Compare the characterization digests of a base and a head that one runner wrote: the compare
// job of CI (docs/library-structure.md, section 5, phase 0, and section 4, rule 2).
//
//   node scripts/compare-digests.mjs <base directory> <head directory>
//
// Each directory holds the JSON files that the digest tests write under `TYPECV_DIGEST_OUT`
// (tests/typeshade/digest-fixture.mjs, tests/typeshade/sfm-digests-webgpu.test.mjs): `{ cases: {
// <name>: <record> } }`. For every case of the head the command compares the fields `output`,
// `trace` and `requests`, and `plan` and `steps` where both records hold them, with the base's
// case of the same name. A difference fails the command (exit 1) and names the case, the fields
// and the parts of the output that changed.
//
// A case or a file that the base lacks is reported and does not fail: the head's tests run
// against the base's library, and a module that the base does not have makes a test file fail to
// load. A file that the head lacks fails, since the head must run every test. Files that ran on
// two Node versions fail too: the CPU-tier digests are not known to be equal across Node majors.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const FIELDS = ['output', 'trace', 'requests', 'plan', 'steps'];

const read = (file) => JSON.parse(readFileSync(file, 'utf8')).cases ?? {};
/** The configuration a digest file records (Node, V8, platform, commit), if any. */
const configurationOf = (file) => JSON.parse(readFileSync(file, 'utf8')).configuration;
const describe = (c) =>
  c ? `Node ${c.node} (V8 ${c.v8}, ${c.platform}) at ${c.commit}` : 'unknown';
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** The comparison of two digest directories: the differences, and what the base did not run. */
export function compareDigests(baseDirectory, headDirectory) {
  const differences = [];
  const notInBase = [];
  const onlyInBase = [];
  const files = existsSync(headDirectory)
    ? readdirSync(headDirectory)
        .filter((f) => f.endsWith('.json'))
        .sort()
    : [];
  for (const file of files) {
    const head = read(join(headDirectory, file));
    const basePath = join(baseDirectory, file);
    if (!existsSync(basePath)) {
      notInBase.push(`${file}: every case (${Object.keys(head).length})`);
      continue;
    }
    const base = read(basePath);
    for (const name of Object.keys(head).sort()) {
      if (!Object.hasOwn(base, name)) {
        notInBase.push(`${file}: ${name}`);
        continue;
      }
      const fields = FIELDS.filter(
        (key) =>
          (key in head[name] || key in base[name]) && !same(head[name][key], base[name][key]),
      );
      if (fields.length === 0) continue;
      const parts = Object.keys({ ...base[name].parts, ...head[name].parts }).filter(
        (key) => base[name].parts?.[key] !== head[name].parts?.[key],
      );
      differences.push(
        `${file}: ${name}: ${fields.join(', ')}` +
          (parts.length ? ` (output parts: ${parts.join(', ')})` : ''),
      );
    }
    for (const name of Object.keys(base).sort())
      if (!Object.hasOwn(head, name)) onlyInBase.push(`${file}: ${name}`);
  }
  const baseFiles = existsSync(baseDirectory)
    ? readdirSync(baseDirectory).filter((f) => f.endsWith('.json'))
    : [];
  const missingInHead = baseFiles.filter((f) => !files.includes(f)).sort();
  // One runner and one Node: the CPU-tier digests are not known to be equal across Node majors.
  const nodes = new Set();
  const configurations = {};
  for (const [side, directory, list] of [
    ['head', headDirectory, files],
    ['base', baseDirectory, baseFiles],
  ]) {
    const found = list
      .map((file) => configurationOf(join(directory, file)))
      .filter((c) => c !== undefined);
    configurations[side] = found[0];
    for (const c of found) nodes.add(c.node);
  }
  const mismatch = nodes.size > 1 ? [...nodes].sort() : [];
  return { files, differences, notInBase, onlyInBase, missingInHead, configurations, mismatch };
}

if (import.meta.main) {
  const [baseDirectory, headDirectory] = process.argv.slice(2);
  if (!baseDirectory || !headDirectory) {
    console.error('Usage: node scripts/compare-digests.mjs <base directory> <head directory>');
    process.exit(2);
  }
  const result = compareDigests(baseDirectory, headDirectory);
  const section = (title, lines) => {
    console.log(`${title}: ${lines.length}`);
    for (const line of lines) console.log(`  ${line}`);
  };
  console.log(`Head: ${describe(result.configurations.head)}`);
  console.log(`Base: ${describe(result.configurations.base)}`);
  console.log(`Digest files of the head: ${result.files.join(', ') || 'none'}`);
  section('Not run on the base (reported, not compared)', result.notInBase);
  section('Run on the base only (reported)', result.onlyInBase);
  section('Files the base wrote and the head did not', result.missingInHead);
  section('Cases whose digests differ between the base and the head', result.differences);
  if (result.files.length === 0) {
    console.error('The head wrote no digest file.');
    process.exitCode = 1;
  } else if (result.mismatch.length > 0) {
    console.error(`The digests ran on more than one Node version: ${result.mismatch.join(', ')}`);
    process.exitCode = 1;
  } else if (result.differences.length > 0 || result.missingInHead.length > 0) process.exitCode = 1;
}
