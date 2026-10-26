// The public declaration surface of the two package entry points, `typecv` (src/index.ts) and
// `typecv/gpu` (src/gpu/index.ts): docs/library-structure.md, section 4, rule 2, and section 5,
// phase 0. The gate of every structure phase holds tests/api-surface.txt unchanged, except for the
// additions the phase names.
//
//   node scripts/api-surface.mjs           print the surface
//   node scripts/api-surface.mjs --write   write tests/api-surface.txt
//
// The declarations are the ones `npm run build` emits for the package's `types`, made here in
// memory with the TypeScript compiler API, so no prior build is needed: `tsconfig.build.json` for
// `typecv` and `tsconfig.gpu.json` for `typecv/gpu`, whose kernel imports read the host views that
// scripts/kernel-faces.mjs compiles (the build's Vite plugin writes the same views to disk). A
// type error in either program fails the command, as it fails the build.
//
// For each entry the file lists every exported name, in code-point order, with its kind, and then
// the declaration text of each name: the declaration it resolves to through the re-exports, as the
// emitted `.d.ts` holds it. Then the declarations that the exported ones reference and the entry
// does not export, by name, found through the type references of the emitted declarations; a
// declaration outside the package (TypeShade, the DOM, the standard library) is named, not
// printed. The text leaves out what a move of a file changes and a consumer does not see:
// comments, the `export` and `declare` modifiers of a declaration, the specifier of an inline
// `import("./…")` type, and every file path.
import { writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import ts from 'typescript';
import { kernelFaces } from './kernel-faces.mjs';

const ROOT = join(import.meta.dirname, '..');
export const SURFACE_FILE = 'tests/api-surface.txt';

/** The two entries: the package path, the source of the barrel and the configuration that builds its types. */
export const ENTRIES = [
  { name: 'typecv', source: 'src/index.ts', config: 'tsconfig.build.json' },
  { name: 'typecv/gpu', source: 'src/gpu/index.ts', config: 'tsconfig.gpu.json' },
];

const slash = (path) => path.replaceAll('\\', '/');

function formatDiagnostics(diagnostics) {
  return ts.formatDiagnostics(diagnostics, {
    getCanonicalFileName: (name) => name,
    getCurrentDirectory: () => ROOT,
    getNewLine: () => '\n',
  });
}

/**
 * A compiler host whose files in `overlay` (absolute path → text) replace the disk, and which
 * hides every other file that `hidden` names.
 */
function overlayHost(options, overlay, hidden) {
  const host = ts.createCompilerHost(options, true);
  const fileExists = host.fileExists.bind(host);
  const readFile = host.readFile.bind(host);
  const getSourceFile = host.getSourceFile.bind(host);
  host.fileExists = (file) =>
    overlay.has(slash(file)) || (!hidden(slash(file)) && fileExists(file));
  host.readFile = (file) =>
    overlay.get(slash(file)) ?? (hidden(slash(file)) ? undefined : readFile(file));
  host.getSourceFile = (file, language, onError, create) => {
    const text = overlay.get(slash(file));
    if (text !== undefined) return ts.createSourceFile(file, text, language, true);
    return hidden(slash(file)) ? undefined : getSourceFile(file, language, onError, create);
  };
  host.writeFile = () => {
    throw new Error('scripts/api-surface.mjs writes no compiler output');
  };
  return host;
}

/** The parsed configuration of `config`, relative to `root`. */
function configOf(root, config) {
  const parsed = ts.getParsedCommandLineOfConfigFile(join(root, config), undefined, {
    ...ts.sys,
    onUnRecoverableConfigFileDiagnostic: (diagnostic) => {
      throw new Error(formatDiagnostics([diagnostic]));
    },
  });
  if (parsed.errors.length > 0) throw new Error(formatDiagnostics(parsed.errors));
  return parsed;
}

/**
 * The `.d.ts` files that the configuration's build emits, in memory: absolute path → text. The
 * views of the kernels stand in for the files that the Vite plugin writes beside each kernel.
 */
const emitted = new WeakMap();
export function emitDeclarations(root, config, views) {
  // One emission per configuration and set of views: the test reads it twice.
  const key = `${root}\0${config}`;
  const memo = emitted.get(views)?.get(key);
  if (memo !== undefined) return memo;
  const result = emitUncached(root, config, views);
  if (!emitted.has(views)) emitted.set(views, new Map());
  emitted.get(views).set(key, result);
  return result;
}

function emitUncached(root, config, views) {
  const parsed = configOf(root, config);
  const options = {
    ...parsed.options,
    noEmit: false,
    declaration: true,
    emitDeclarationOnly: true,
    declarationMap: false,
    sourceMap: false,
  };
  // A view on disk is left over from an interrupted build; the views of this run replace it.
  const hidden = (file) => file.endsWith('.shade.typeshade.ts');
  const host = overlayHost(options, views, hidden);
  const rootNames = parsed.fileNames.filter((file) => !hidden(slash(file)));
  const program = ts.createProgram({ rootNames, options, host });
  const errors = ts
    .getPreEmitDiagnostics(program)
    .filter((d) => d.category === ts.DiagnosticCategory.Error);
  if (errors.length > 0) throw new Error(`${config}:\n${formatDiagnostics(errors)}`);
  const outputs = new Map();
  const result = program.emit(
    undefined,
    (file, text) => {
      // scripts/build-gpu.mjs deletes the declarations of the views from dist/types.
      if (!file.endsWith('.shade.typeshade.d.ts')) outputs.set(slash(file), text);
    },
    undefined,
    true,
  );
  if (result.emitSkipped) throw new Error(`${config}: declaration emit skipped`);
  return { options, outputs, outDir: slash(options.outDir), rootDir: slash(options.rootDir) };
}

const printer = ts.createPrinter({ removeComments: true, newLine: ts.NewLineKind.LineFeed });

/** The kind of a declaration, as the surface names it. */
function kindOf(node) {
  if (ts.isFunctionDeclaration(node)) return 'function';
  if (ts.isClassDeclaration(node)) return 'class';
  if (ts.isInterfaceDeclaration(node)) return 'interface';
  if (ts.isTypeAliasDeclaration(node)) return 'type';
  if (ts.isEnumDeclaration(node)) return 'enum';
  if (ts.isModuleDeclaration(node)) return 'namespace';
  if (ts.isVariableDeclaration(node))
    return node.parent.flags & ts.NodeFlags.Const
      ? 'const'
      : node.parent.flags & ts.NodeFlags.Let
        ? 'let'
        : 'var';
  return ts.SyntaxKind[node.kind];
}

/** The text of a declaration: its statement printed without comments, modifiers or import paths. */
function declarationText(node) {
  const statement = ts.isVariableDeclaration(node) ? node.parent.parent : node;
  let text = printer.printNode(ts.EmitHint.Unspecified, statement, statement.getSourceFile());
  while (/^(export|default|declare) /.test(text)) text = text.replace(/^\w+ /, '');
  return text.replace(/import\("\.[^"]*"\)\./g, '');
}

/** The nodes in `node` that name another declaration: type references, heritage, `typeof`. */
function referencedNames(node) {
  const names = [];
  const visit = (child) => {
    if (ts.isTypeReferenceNode(child)) names.push(child.typeName);
    else if (ts.isExpressionWithTypeArguments(child)) names.push(child.expression);
    else if (ts.isTypeQueryNode(child)) names.push(child.exprName);
    else if (ts.isImportTypeNode(child) && child.qualifier) names.push(child.qualifier);
    ts.forEachChild(child, visit);
  };
  visit(node);
  return names;
}

/** The surface of one entry: its exports and the declarations they reference. */
function entrySurface(root, entry, views) {
  const { options, outputs, outDir, rootDir } = emitDeclarations(root, entry.config, views);
  const entryFile = slash(join(outDir, relative(rootDir, join(root, entry.source)))).replace(
    /\.ts$/,
    '.d.ts',
  );
  if (!outputs.has(entryFile)) throw new Error(`${entry.config} emitted no ${entryFile}`);
  // A second program reads the emitted declarations as a consumer of the package does. The disk
  // copies under the output directory, from an earlier build, stay out of it.
  const declarationOptions = {
    ...options,
    noEmit: true,
    declaration: false,
    emitDeclarationOnly: false,
  };
  const program = ts.createProgram({
    rootNames: [entryFile],
    options: declarationOptions,
    host: overlayHost(declarationOptions, outputs, (file) => file.startsWith(`${outDir}/`)),
  });
  const errors = ts
    .getPreEmitDiagnostics(program)
    .filter((d) => d.category === ts.DiagnosticCategory.Error);
  if (errors.length > 0)
    throw new Error(`${entry.name} declarations:\n${formatDiagnostics(errors)}`);
  const checker = program.getTypeChecker();
  const inPackage = (node) => outputs.has(slash(node.getSourceFile().fileName));
  const origin = (node) => {
    const file = slash(relative(root, node.getSourceFile().fileName));
    return file.startsWith('node_modules/') ? file.slice('node_modules/'.length) : file;
  };
  const resolve = (symbol) =>
    symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;

  const moduleSymbol = checker.getSymbolAtLocation(program.getSourceFile(entryFile));
  const exported = checker
    .getExportsOfModule(moduleSymbol)
    .map((symbol) => ({ name: symbol.getName(), symbol, target: resolve(symbol) }))
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const targets = new Set(exported.map(({ target }) => target));

  const describe = (symbol) =>
    (symbol.declarations ?? []).map((node) =>
      inPackage(node)
        ? { kind: kindOf(node), text: declarationText(node), node }
        : { kind: kindOf(node), external: origin(node), node },
    );
  // The declarations that the exported ones reach through their type references.
  const referenced = new Map();
  const queue = exported.flatMap(({ target }) => describe(target));
  while (queue.length > 0) {
    const { node } = queue.shift();
    if (!inPackage(node)) continue;
    for (const name of referencedNames(node)) {
      const found = checker.getSymbolAtLocation(name);
      if (found === undefined) continue;
      const target = resolve(found);
      if (
        target.flags & ts.SymbolFlags.TypeParameter ||
        targets.has(target) ||
        referenced.has(target)
      )
        continue;
      const declarations = describe(target);
      if (!declarations.some((d) => inPackage(d.node))) continue;
      referenced.set(target, { name: target.getName(), declarations });
      queue.push(...declarations);
    }
  }
  const typeOnly = (symbol) =>
    (symbol.declarations ?? []).some(
      (d) => ts.isExportSpecifier(d) && (d.isTypeOnly || d.parent.parent.isTypeOnly),
    );
  return {
    exported: exported.map(({ name, symbol, target }) => ({
      name,
      typeOnly: typeOnly(symbol),
      original: target.getName() === name ? undefined : target.getName(),
      declarations: describe(target),
    })),
    referenced: [...referenced.values()]
      .map(({ name, declarations }) => ({
        name,
        declarations,
        key: declarations.map((d) => d.text).join('\n'),
      }))
      .sort((a, b) =>
        a.name < b.name ? -1 : a.name > b.name ? 1 : a.key < b.key ? -1 : a.key > b.key ? 1 : 0,
      ),
  };
}

/** The lines of one name: a heading, then each declaration's text or origin. */
function nameLines(name, note, declarations) {
  const lines = [`-- ${name}${note}`];
  for (const d of declarations)
    lines.push(...(d.external ? [`${d.kind} from ${d.external}`] : d.text.split('\n')));
  return lines;
}

/** The host view of each kernel, at the path the Vite plugin writes it to: absolute path → text. */
export async function kernelViews(root = ROOT) {
  return new Map(
    (await kernelFaces(root)).map((face) => [
      slash(join(root, face.path.replace(/\.ts$/, '.typeshade.ts'))),
      face.view,
    ]),
  );
}

/** The text of tests/api-surface.txt. */
export async function apiSurface(root = ROOT, views = undefined) {
  views ??= await kernelViews(root);
  const lines = [
    '# The public declaration surface of typecv and typecv/gpu: docs/library-structure.md,',
    '# section 4, rule 2, and section 5, phase 0. Generated; do not edit.',
    '# Update: node scripts/api-surface.mjs --write',
  ];
  for (const entry of ENTRIES) {
    const surface = entrySurface(root, entry, views);
    const values = surface.exported.filter(
      (e) => !e.typeOnly && e.declarations.some((d) => !['interface', 'type'].includes(d.kind)),
    ).length;
    lines.push(
      '',
      `== ${entry.name} (${entry.source}): ${surface.exported.length} names, ${values} values, ` +
        `${surface.exported.length - values} types`,
      ...surface.exported.map(
        (e) =>
          `${e.name}: ${[...new Set(e.declarations.map((d) => d.kind))].join(', ')}` +
          `${e.typeOnly ? ' (type-only export)' : ''}`,
      ),
    );
    for (const e of surface.exported) {
      const note = e.original ? ` (declared as ${e.original})` : '';
      lines.push('', ...nameLines(e.name, note, e.declarations));
    }
    lines.push(
      '',
      `== ${entry.name}: ${surface.referenced.length} declarations that the names reference`,
    );
    for (const r of surface.referenced) lines.push('', ...nameLines(r.name, '', r.declarations));
  }
  return `${lines.join('\n')}\n`;
}

if (import.meta.main) {
  const text = await apiSurface();
  if (process.argv.includes('--write')) {
    writeFileSync(join(ROOT, SURFACE_FILE), text);
    console.log(`Wrote ${SURFACE_FILE}`);
  } else process.stdout.write(text);
}
