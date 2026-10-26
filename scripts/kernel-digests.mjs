// Digests of the WGSL and GLSL that the build emits for each kernel module: docs/library-structure.md,
// section 4, rule 2, and section 5, phase 0. The gate of a structure phase holds
// tests/digests/kernels.json unchanged, except for the kernels that the phase names.
//
//   node scripts/kernel-digests.mjs           print the digests
//   node scripts/kernel-digests.mjs --write   write tests/digests/kernels.json
//
// The faces are the build's (scripts/kernel-faces.mjs). Each module is keyed by its file name,
// which a move keeps; two modules with one file name are an error. A kernel alias (`aliasTarget`)
// is not keyed: `kernelAliases` holds it to its target's digests. For each module:
//
//   wgsl         SHA-256 of the WGSL that the WebGPU tier dispatches: the module's WGSL (the
//                manifest's `wgsl`, the text each entry dispatches), then the WGSL of each kernel
//                function by name (`gpu.wgsl`, and `gpu.projected.wgsl` when it has one).
//   wgslInteger  the same under the 'integer' flavor of the emulated double (TypeShade change
//                0073); present when the module emulates f64.
//   glsl         SHA-256 of the WebGL2 tier's GLSL ES 3.00 programs: the pass program of each
//                compute entry (`gl.computes`), the program of each fragment and vertex entry, and
//                the loop programs of each kernel function (`gl.loops`), by name. An entry without
//                a program contributes the compiler's reason. Absent when the module has no entry.
//   entries, kernelFunctions, glslPrograms, withoutGlsl: counts, for the reader.
//
// No file path enters a digest. The manifest's source lines and a kernel function's fuse record
// carry the kernel's absolute path; neither is read.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, join, posix } from 'node:path';
import ts from 'typescript';
import { kernelFaces } from './kernel-faces.mjs';

const ROOT = join(import.meta.dirname, '..');
export const DIGEST_FILE = 'tests/digests/kernels.json';

const sha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');
const byName = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

/** One line of the generated module that holds a JSON value: `<prefix><json>;`. */
function generated(code, prefix) {
  const line = code.split('\n').find((l) => l.startsWith(prefix));
  return line === undefined ? undefined : JSON.parse(line.slice(prefix.length, -1));
}

/** The shader texts of one face, each a list of named sections. */
export function shaderSections(face) {
  const pack = generated(face.code, 'export default ');
  if (pack === undefined) throw new Error(`${face.path}: the generated module has no manifest`);
  const integer = generated(face.code, 'const __ts_wgsl_integer = ');
  const kernels = (face.exports ?? [])
    .filter((e) => e.kind === 'kernel')
    .sort((a, b) => byName(a.name, b.name));
  const wgsl = [['module', pack.wgsl]];
  const wgslInteger = integer === undefined ? [] : [['module', integer]];
  const glsl = [];
  if (pack.glsl)
    glsl.push(['module vertex', pack.glsl.vertex], ['module fragment', pack.glsl.fragment]);
  for (const [kind, table, field] of [
    ['compute', pack.gl?.computes, 'vertex'],
    ['fragment', pack.gl?.draws, 'fragment'],
    ['vertex', pack.gl?.vertices, 'vertex'],
  ])
    for (const name of Object.keys(table ?? {}).sort(byName)) {
      const program = table[name];
      glsl.push([`${kind} ${name}`, 'none' in program ? `none: ${program.none}` : program[field]]);
    }
  for (const { name, face: kernel } of kernels) {
    const gpu = kernel.gpu;
    wgsl.push([`kernel ${name}`, gpu ? gpu.wgsl : `none: ${kernel.noGpu}`]);
    if (gpu?.projected) wgsl.push([`kernel ${name} projected`, gpu.projected.wgsl]);
    if (gpu?.wgslInteger !== undefined) wgslInteger.push([`kernel ${name}`, gpu.wgslInteger]);
    if (gpu?.projected?.wgslInteger !== undefined)
      wgslInteger.push([`kernel ${name} projected`, gpu.projected.wgslInteger]);
    if (kernel.gl)
      kernel.gl.loops.forEach((loop, i) => glsl.push([`kernel ${name} loop ${i}`, loop.glsl]));
    else glsl.push([`kernel ${name}`, `none: ${kernel.noWebgl2}`]);
  }
  return { pack, kernels, wgsl, wgslInteger, glsl };
}

const joined = (sections) => sections.map(([title, text]) => `// ${title}\n${text}\n`).join('');

/** The digests of one face. */
export function digestFace(face) {
  const { pack, kernels, wgsl, wgslInteger, glsl } = shaderSections(face);
  const entries = pack.entries.length;
  return {
    wgsl: sha256(joined(wgsl)),
    ...(wgslInteger.length > 0 ? { wgslInteger: sha256(joined(wgslInteger)) } : {}),
    ...(entries + kernels.length > 0 ? { glsl: sha256(joined(glsl)) } : {}),
    entries,
    kernelFunctions: kernels.length,
    glslPrograms: glsl.filter(([, text]) => !text.startsWith('none: ')).length,
    withoutGlsl: glsl.filter(([, text]) => text.startsWith('none: ')).length,
  };
}

/** The TypeShade pin of package.json: the compiler commit the digests belong to. */
export function compilerPin(root = ROOT) {
  const spec = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).devDependencies
    .typeshade;
  return /\/archive\/([0-9a-f]{40})\.tar\.gz$/.exec(spec)?.[1] ?? spec;
}

/**
 * The configuration that writes the file: the Node and V8 versions, the platform and the commit.
 * The shader text is not known to depend on it: TypeShade emits a math builtin's call rather than
 * folding it with the host's `Math`; the record names it all the same.
 */
function configuration(root) {
  let commit = 'unknown';
  try {
    commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
  } catch {
    // Not a git checkout.
  }
  return {
    node: process.version,
    v8: process.versions.v8,
    platform: `${process.platform} ${process.arch}`,
    commit,
  };
}

/**
 * The target of a kernel alias, relative to the repository root, or undefined for any other
 * module. A kernel alias is a shader module whose only statement after its directive is one
 * `export * from` another shader module, named by a relative path (docs/library-structure.md,
 * section 4, rule 4, amendment 1). Phase 1b leaves one at each old kernel path that a module
 * outside src/ imports, and one beside the kernels that import a moved helper kernel from their
 * own directory.
 */
export function aliasTarget(path, text) {
  const statements = ts.createSourceFile(path, text, ts.ScriptTarget.Latest).statements;
  if (statements.length !== 2) return undefined;
  const [directive, exported] = statements;
  if (
    !ts.isExpressionStatement(directive) ||
    !ts.isStringLiteral(directive.expression) ||
    directive.expression.text !== 'use typeshade' ||
    !ts.isExportDeclaration(exported) ||
    exported.exportClause !== undefined ||
    exported.isTypeOnly ||
    exported.moduleSpecifier === undefined ||
    !ts.isStringLiteral(exported.moduleSpecifier)
  )
    return undefined;
  const specifier = exported.moduleSpecifier.text;
  if (!specifier.startsWith('.') || !specifier.endsWith('.shade.ts')) return undefined;
  return posix.normalize(posix.join(posix.dirname(path), specifier));
}

const aliasOf = (root, face) => aliasTarget(face.path, readFileSync(join(root, face.path), 'utf8'));

/** Each kernel alias, its target and whether its digests equal the target's, by alias path. */
export async function kernelAliases(root = ROOT, faces = undefined) {
  faces ??= await kernelFaces(root);
  const byPath = new Map(faces.map((face) => [face.path, face]));
  const same = (a, b) => JSON.stringify(digestFace(a)) === JSON.stringify(digestFace(b));
  return faces
    .map((face) => ({ face, target: aliasOf(root, face) }))
    .filter(({ target }) => target !== undefined)
    .map(({ face, target }) => ({
      alias: face.path,
      target,
      equal: byPath.has(target) && same(face, byPath.get(target)),
    }));
}

/** The contents of tests/digests/kernels.json. */
export async function kernelDigests(root = ROOT, faces = undefined) {
  faces ??= await kernelFaces(root);
  const modules = {};
  for (const face of faces) {
    if (aliasOf(root, face) !== undefined) continue;
    const name = basename(face.path);
    if (Object.hasOwn(modules, name))
      throw new Error(`Two kernel modules are named ${name}; the digests key a module by its name`);
    modules[name] = digestFace(face);
  }
  const sorted = Object.fromEntries(
    Object.keys(modules)
      .sort(byName)
      .map((k) => [k, modules[k]]),
  );
  return {
    update: 'node scripts/kernel-digests.mjs --write',
    configuration: configuration(root),
    compiler: compilerPin(root),
    modules: sorted,
  };
}

if (import.meta.main) {
  const text = `${JSON.stringify(await kernelDigests(), null, 2)}\n`;
  if (process.argv.includes('--write')) {
    writeFileSync(join(ROOT, DIGEST_FILE), text);
    console.log(`Wrote ${DIGEST_FILE}`);
  } else process.stdout.write(text);
}
