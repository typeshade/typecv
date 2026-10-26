// Invoked by profile-kernel-ir.mjs after bundling the compiler's TypeScript sources.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { compile, emitModuleAt, lowerWgsl, sourceSpanOf, workgroupShapeOf } from 'typeshade';
import { countOps, profileEmit } from 'typeshade/dev';

const { values } = parseArgs({
  options: {
    out: { type: 'string', default: '.typecv-build/kernel-ir' },
    kernel: { type: 'string', multiple: true },
  },
});
const targets = values.kernel ?? [
  'src/gpu/joint-focal.shade.ts#bundleSchurCamera',
  'src/geometry/gpu/pair-models.shade.ts#appendPairModelRows',
  'src/gpu/mvs.shade.ts#mvsPropagate',
  'src/gpu/tsdf.shade.ts#tsdfCarve',
  'src/gpu/tsdf.shade.ts#tsdfViewCarve',
];
const out = resolve(values.out);
mkdirSync(out, { recursive: true });
const sha = (text) => createHash('sha256').update(text).digest('hex');
const json = (value) =>
  JSON.stringify(value, (key, v) => (key === 'declRef' ? undefined : v), 2) + '\n';
function readDocument(file) {
  try {
    return readFileSync(file, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return undefined;
    throw error;
  }
}
function walk(value, visit) {
  if (Array.isArray(value)) {
    for (const child of value) walk(child, visit);
  } else if (value && typeof value === 'object') {
    visit(value);
    for (const [key, child] of Object.entries(value))
      if (!['type', 'span', 'declRef'].includes(key)) walk(child, visit);
  }
}
function callNames(f) {
  const calls = new Set();
  walk(f.body, (node) => {
    if (node.op === 'call') calls.add(node.fn);
  });
  return calls;
}
function reachable(module, entry) {
  const byName = new Map(module.funcs.map((f) => [f.name, f]));
  if (!byName.has(entry)) throw new Error(`Unknown function ${entry}`);
  const seen = new Set();
  const visit = (name) => {
    if (seen.has(name) || !byName.has(name)) return;
    seen.add(name);
    for (const call of callNames(byName.get(name))) visit(call);
  };
  visit(entry);
  return { ...module, funcs: module.funcs.filter((f) => seen.has(f.name)) };
}
function facts(module) {
  return {
    ops: countOps(module),
    functions: module.funcs.map((f) => {
      const detail = { loops: 0, branches: 0, indexSites: 0, atomicCallSites: 0 };
      walk(f.body, (node) => {
        if (node.s === 'for') detail.loops++;
        if (node.s === 'if' || node.s === 'switch') detail.branches++;
        if (node.op === 'index') detail.indexSites++;
        if (node.op === 'call' && /^atomic/.test(node.fn)) detail.atomicCallSites++;
      });
      const span = sourceSpanOf(f.body[0]);
      return {
        name: f.name,
        source: span && { file: relative(process.cwd(), span.file), line: span.line + 1 },
        ...(f.stage === 'compute' ? { workgroupShape: workgroupShapeOf(f) } : {}),
        ops: countOps({ ...module, funcs: [f] }),
        ...detail,
        calls: [...callNames(f)],
      };
    }),
  };
}

const modules = new Map();
const results = [];
for (const target of targets) {
  const [file, entry, extra] = target.split('#');
  if (!file || !entry || extra) throw new Error('--kernel requires file.shade.ts#entry');
  const absolute = resolve(file);
  if (!modules.has(absolute)) {
    const source = readFileSync(absolute, 'utf8');
    const dependencies = new Map([[relative(process.cwd(), absolute), sha(source)]]);
    const started = performance.now();
    const compiled = compile(source, {
      fileName: absolute,
      readDocument: (path) => {
        const text = readDocument(path);
        if (text !== undefined) dependencies.set(relative(process.cwd(), path), sha(text));
        return text;
      },
    });
    const compileMs = performance.now() - started;
    const errors = compiled.diagnostics.filter((d) => d.category === 'error');
    if (errors.length) throw new Error(json(errors));
    modules.set(absolute, { module: compiled.module, compileMs, dependencies });
  }
  const { module, compileMs, dependencies } = modules.get(absolute);
  const scoped = reachable(module, entry);
  const artifacts = {};
  const levels = {};
  for (const level of ['O0', 'O2']) {
    const ir = lowerWgsl(scoped, level);
    const code = emitModuleAt(scoped, level);
    const irText = json(ir);
    writeFileSync(resolve(out, `${entry}.${level}.ir.json`), irText);
    writeFileSync(resolve(out, `${entry}.${level}.wgsl`), code);
    artifacts[level] = {
      irSha256: sha(irText),
      wgslSha256: sha(code),
      wgslBytes: Buffer.byteLength(code),
    };
    levels[level] = facts(reachable(ir, entry));
  }
  writeFileSync(resolve(out, `${entry}.authored.ir.json`), json(scoped));
  results.push({
    file: relative(process.cwd(), absolute),
    entry,
    dependencies: Object.fromEntries(dependencies),
    compileMs,
    authored: facts(scoped),
    levels,
    emitProfile: profileEmit(scoped),
    artifacts,
  });
  process.stderr.write(
    `${entry}: O0 ${levels.O0.ops.total}, O2 ${levels.O2.ops.total} static ops\n`,
  );
}
const packageInfo = JSON.parse(readFileSync('package.json', 'utf8'));
const report = {
  methodology:
    'Actual source and imported shader files, pinned TypeShade front end and WGSL lowering. Static counts cover unique functions reachable from one entry, each counted once; they are not dynamic operations, hardware instructions or GPU timings. Emit profiles measure build-time lowering on this host, not browser pipeline compilation. Float fp64 flavor matches the NVIDIA records. Inspectable IR JSON is not a version-independent interchange format.',
  siteCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  typeshade: packageInfo.devDependencies.typeshade,
  results,
};
writeFileSync(resolve(out, 'summary.json'), json(report));
process.stdout.write(`${out}/summary.json\n`);
