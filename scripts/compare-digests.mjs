// Compare the characterization digests of a base and a head that one runner wrote: the compare
// job of CI (docs/library-structure.md, section 5, phase 0, and section 4, rule 2).
//
//   node scripts/compare-digests.mjs <base directory> <head directory>
//     [--declared <file> --pull-request <number>]
//
// Each directory holds the JSON files that the digest tests write under `TYPECV_DIGEST_OUT`
// (tests/typeshade/digest-fixture.mjs, tests/typeshade/sfm-digests-webgpu.test.mjs): `{ cases: {
// <name>: <record> } }`. For every case of the head the command compares the fields `output`,
// `trace` and `requests`, and `plan` and `steps` where both records hold them, with the base's
// case of the same name. A difference fails the command (exit 1) and names the case, the fields
// and the parts of the output that changed.
//
// A pull request that changes behaviour declares the cases whose digests it changes in a
// declaration file, `tests/digests/expected-changes.json`:
//
//   { "pullRequest": 267, "reason": "<why the digests change>", "cases": ["<file>: <case>", …] }
//
// With `--declared <file> --pull-request <number>` (CI passes the pull request's number), a
// difference in a declared case of that pull request is reported and does not fail. A declaration
// for another pull request is ignored and reported, so a stale file changes nothing. A declared
// case whose digests are equal, and a declared case that the head did not run, are reported. A
// malformed declaration is an error (exit 2). Every undeclared difference fails.
//
// A case or a file that the base lacks is reported and does not fail: the head's tests run
// against the base's library, and a module that the base does not have makes a test file fail to
// load. A file that the head lacks fails, since the head must run every test. Files that ran on
// two Node versions fail too: the CPU-tier digests are not known to be equal across Node majors.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const FIELDS = ['output', 'trace', 'requests', 'plan', 'steps'];
const USAGE =
  'Usage: node scripts/compare-digests.mjs <base directory> <head directory> ' +
  '[--declared <file> --pull-request <number>]';

const read = (file) => JSON.parse(readFileSync(file, 'utf8')).cases ?? {};
/** The configuration a digest file records (Node, V8, platform, commit, flags), if any. */
const configurationOf = (file) => JSON.parse(readFileSync(file, 'utf8')).configuration;
/** The flags of the process, where the file records them (tests/typeshade/digest-fixture.mjs). */
const flags = (c) =>
  Array.isArray(c.execArgv) ? `; execArgv ${c.execArgv.join(' ') || 'none'}` : '';
const describe = (c) =>
  c ? `Node ${c.node} (V8 ${c.v8}, ${c.platform}${flags(c)}) at ${c.commit}` : 'unknown';
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Read a declaration of expected digest changes; undefined when the file does not exist. Throws
 * on a file that is not the declared shape, so that a malformed declaration never passes a gate.
 */
export function readDeclaration(file) {
  if (!existsSync(file)) return undefined;
  let parsed;
  try {
    parsed = JSON.parse(readFileSync(file, 'utf8'));
  } catch (error) {
    throw new Error(`${file}: not JSON (${error.message})`);
  }
  const problems = [];
  if (!Number.isInteger(parsed?.pullRequest) || parsed.pullRequest <= 0)
    problems.push('pullRequest must be a positive integer');
  if (typeof parsed?.reason !== 'string' || parsed.reason.trim() === '')
    problems.push('reason must be a non-empty string');
  const isCase = (c) => typeof c === 'string' && /^[^\s:]+\.json: \S/.test(c);
  if (!Array.isArray(parsed?.cases) || parsed.cases.length === 0 || !parsed.cases.every(isCase))
    problems.push('cases must be a non-empty array of "<file>.json: <case>" strings');
  if (problems.length > 0) throw new Error(`${file}: ${problems.join('; ')}`);
  return {
    pullRequest: parsed.pullRequest,
    reason: parsed.reason,
    cases: [...new Set(parsed.cases)],
  };
}

/**
 * The comparison of two digest directories: the differences, split into the declared and the
 * undeclared ones (`declared` lists the case keys `<file>: <case>` that the pull request
 * declares), and what the base did not run.
 */
export function compareDigests(baseDirectory, headDirectory, declared = []) {
  const differences = [];
  const differingKeys = [];
  const headKeys = new Set();
  const notInBase = [];
  const onlyInBase = [];
  const files = existsSync(headDirectory)
    ? readdirSync(headDirectory)
        .filter((f) => f.endsWith('.json'))
        .sort()
    : [];
  for (const file of files) {
    const head = read(join(headDirectory, file));
    for (const name of Object.keys(head)) headKeys.add(`${file}: ${name}`);
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
      differingKeys.push(`${file}: ${name}`);
    }
    for (const name of Object.keys(base).sort())
      if (!Object.hasOwn(head, name)) onlyInBase.push(`${file}: ${name}`);
  }
  const declaredSet = new Set(declared);
  const differing = new Set(differingKeys);
  const declaredDifferences = differences.filter((_, i) => declaredSet.has(differingKeys[i]));
  const undeclared = differences.filter((_, i) => !declaredSet.has(differingKeys[i]));
  const declaredEqual = declared.filter((key) => headKeys.has(key) && !differing.has(key));
  const declaredUnknown = declared.filter((key) => !headKeys.has(key));
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
  return {
    files,
    differences,
    declared: declaredDifferences,
    undeclared,
    declaredEqual,
    declaredUnknown,
    notInBase,
    onlyInBase,
    missingInHead,
    configurations,
    mismatch,
  };
}

if (import.meta.main) {
  const positional = [];
  let declaredFile;
  let pullRequest;
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--declared') declaredFile = args[++i];
    else if (args[i] === '--pull-request') pullRequest = Number(args[++i]);
    else positional.push(args[i]);
  }
  const [baseDirectory, headDirectory] = positional;
  const withDeclaration = declaredFile !== undefined || pullRequest !== undefined;
  if (
    !baseDirectory ||
    !headDirectory ||
    positional.length > 2 ||
    (withDeclaration && (!declaredFile || !Number.isInteger(pullRequest) || pullRequest <= 0))
  ) {
    console.error(USAGE);
    process.exit(2);
  }
  let declaration;
  try {
    declaration = withDeclaration ? readDeclaration(declaredFile) : undefined;
  } catch (error) {
    console.error(`Malformed declaration: ${error.message}`);
    process.exit(2);
  }
  const current = declaration?.pullRequest === pullRequest ? declaration : undefined;
  const result = compareDigests(baseDirectory, headDirectory, current?.cases ?? []);
  const section = (title, lines) => {
    console.log(`${title}: ${lines.length}`);
    for (const line of lines) console.log(`  ${line}`);
  };
  console.log(`Head: ${describe(result.configurations.head)}`);
  console.log(`Base: ${describe(result.configurations.base)}`);
  console.log(`Digest files of the head: ${result.files.join(', ') || 'none'}`);
  if (declaration && !current)
    console.log(
      `Declaration of pull request #${declaration.pullRequest} in ${declaredFile} ignored: ` +
        `this is pull request #${pullRequest}`,
    );
  if (current)
    console.log(
      `Declared changes of pull request #${pullRequest} (${declaredFile}): ` +
        `${current.cases.length} cases; reason: ${current.reason}`,
    );
  section('Not run on the base (reported, not compared)', result.notInBase);
  section('Run on the base only (reported)', result.onlyInBase);
  section('Files the base wrote and the head did not', result.missingInHead);
  if (current) {
    section('Cases whose digests differ as declared (reported)', result.declared);
    section('Declared cases whose digests are equal (reported)', result.declaredEqual);
    section('Declared cases that the head did not run (reported)', result.declaredUnknown);
    section('Cases whose digests differ and are not declared', result.undeclared);
  } else section('Cases whose digests differ between the base and the head', result.differences);
  if (result.files.length === 0) {
    console.error('The head wrote no digest file.');
    process.exitCode = 1;
  } else if (result.mismatch.length > 0) {
    console.error(`The digests ran on more than one Node version: ${result.mismatch.join(', ')}`);
    process.exitCode = 1;
  } else if (result.undeclared.length > 0 || result.missingInHead.length > 0) process.exitCode = 1;
}
