// The import graph of `src/` and `website/src/`: the runtime import cycles, the cycles that
// type-only imports add, and the imports that break the layering rule of
// docs/library-structure.md, section 3.1 (phase 0 of section 5).
//
//   node scripts/import-graph.mjs          the report
//   node scripts/import-graph.mjs --json   the same, as JSON
//
// An import is an `import` or `export … from` declaration, a dynamic `import()` of a string
// literal, or an `import('…')` type, whose relative specifier names a file under `src/` or
// `website/src/`. A `.astro` file contributes the imports of its frontmatter and of its inline
// scripts. A specifier resolves as written, then with `.ts`, `.tsx` or `/index.ts` added. A
// relative specifier that names no file is reported as unresolved, and the command exits 1.
//
// An import is type-only when TypeScript and esbuild erase it from the emitted module: `import
// type`, `export type`, an `import('…')` type, or named bindings that all carry `type`. Every
// other import is a runtime import.
//
// Cycles: the strongly connected groups of files (Tarjan), first over the runtime imports, then
// over every import. A group of the second search that is not a group of the first is a cycle
// that type-only imports add; its type-only imports are listed.
//
// Layering (section 3.1): an area imports only the areas below it, type-only imports included.
// `contracts` and `device` are at the bottom, `image` and `geometry` above them, `features` and
// `analysis` above those, `motion` above `features`, and `sfm` and `mvs` at the top. A file
// takes its area from the table of section 3.1, at the path it has today (TODAY below). The
// pieces that phase 1c moves (`chunkedPass`, `packRgba`, `kltLayout`, the object-model types of
// `reconstruction/types.ts`, `validateReconstruction`, `observation-paths.ts`) count in the area
// of the file that holds them today, since phase 1c moves them to remove the breaks. A file in an
// area directory (`src/<area>/…`, phase 1b) takes that area. The barrels `src/index.ts`,
// `src/gpu/index.ts` and `src/gpu/internal.ts` are entry points and have no area. A file of
// `src/` with no area is reported. The layering rule covers `src/` only.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, sep } from 'node:path';
import ts from 'typescript';

const ROOT = join(import.meta.dirname, '..');
const TREES = ['src', 'website/src'];

/** The areas of section 3.1, each with its level: an area imports only the areas below it. */
export const LEVELS = Object.freeze({
  contracts: 0,
  device: 0,
  image: 1,
  geometry: 1,
  features: 2,
  analysis: 2,
  motion: 3,
  sfm: 4,
  mvs: 4,
});

/** The entry points: they export every area and have none of their own. */
const ENTRIES = new Set(['src/index.ts', 'src/gpu/index.ts', 'src/gpu/internal.ts']);

const gpu = (...names) => names.map((name) => `src/gpu/${name}`);
const reconstruction = (...names) => names.map((name) => `src/reconstruction/${name}`);

/** The table of section 3.1 for today's files. `src/reconstruction` is `sfm` unless named. */
const TODAY = {
  device: gpu(
    'device.ts',
    'budget-plan.ts',
    'readback.ts',
    'scan.ts',
    'scan.shade.ts',
    'device-diagnostics.ts',
    'rgba-read-view.ts',
    'numerics.shade.ts',
    'projective.shade.ts',
  ),
  image: gpu('clahe.ts', 'clahe.shade.ts'),
  analysis: gpu('slic.ts', 'slic.shade.ts', 'hough.ts', 'hough.shade.ts'),
  features: gpu(
    'sift.ts',
    'sift.shade.ts',
    'orb.ts',
    'orb.shade.ts',
    'orb-pattern.ts',
    'orb-reference.ts',
    'features.ts',
    'features.shade.ts',
    'tiled-features.ts',
    'float-matching.ts',
    'kernels/float-matching.shade.ts',
    'global-descriptor.ts',
    'global-descriptor-gpu.ts',
    'global-descriptor-candidates.ts',
    'global-descriptor.shade.ts',
  ),
  geometry: [
    'src/sfm/linear.ts',
    'src/sfm/refinement.ts',
    'src/sfm/triangulation.ts',
    'src/sfm/two-view.ts',
    ...gpu(
      'pair-estimation.ts',
      'image-pair.ts',
      'two-view.ts',
      'homography.ts',
      'p3p.ts',
      'pnp.ts',
      'triangulation.ts',
      'observations.ts',
      'observation-layout.ts',
      'epipolar.ts',
      'pair-preparation.shade.ts',
      'pair-models.shade.ts',
      'consensus.shade.ts',
      'recovery.shade.ts',
      'p3p.shade.ts',
      'pnp.shade.ts',
      'triangulation.shade.ts',
      'observation-evaluation.shade.ts',
      'epipolar.shade.ts',
    ),
  ],
  motion: gpu('klt.ts', 'klt-tracker.ts', 'klt-device.ts', 'klt-pair-cache.ts', 'klt.shade.ts'),
  sfm: gpu(
    'mapper.ts',
    'bundle.ts',
    'bundle.shade.ts',
    'joint-focal.ts',
    'joint-focal.shade.ts',
    'bundle-camera-product.ts',
    'bundle-cg-schedule.ts',
    'inverse-depth.ts',
    'inverse-depth.shade.ts',
    'view-graph.ts',
    'view-graph-descriptors.ts',
    'multi-video.ts',
    'multi-video-lifetime.ts',
    'observation-expansion.ts',
    'observation-append.ts',
    'observation-consensus.ts',
    'observation-surface.ts',
  ),
  mvs: [
    ...gpu(
      'mvs.ts',
      'mvs.shade.ts',
      'mvs-views.ts',
      'mvs-residency.ts',
      'planar-depth.ts',
      'planar-depth-lifetime.ts',
      'planar-depth.shade.ts',
      'tsdf.ts',
      'tsdf-box-table.ts',
      'tsdf-upload-window.ts',
      'tsdf-view-channels.ts',
      'tsdf.shade.ts',
      'cloud-clean.ts',
      'cloud-knn.shade.ts',
      'cloud-voxel.shade.ts',
      'dense-mesh.ts',
      'dense-mesh.shade.ts',
    ),
    ...reconstruction(
      'tsdf.ts',
      'dense-mesh.ts',
      'cloud-clean.ts',
      'dense-label.ts',
      'marching-cubes-tables.ts',
      'dense-score.ts',
    ),
  ],
};
const TODAY_AREA = new Map(
  Object.entries(TODAY).flatMap(([area, files]) => files.map((file) => [file, area])),
);

/**
 * The area of a file of `src/` (a path relative to the repository root, with `/`): `'entry'`
 * for a barrel, undefined for a file with no area and for a file outside `src/`.
 */
export function areaOf(file) {
  if (ENTRIES.has(file)) return 'entry';
  const today = TODAY_AREA.get(file);
  if (today !== undefined) return today;
  const directory = /^src\/([^/]+)\//.exec(file)?.[1];
  if (directory !== undefined && Object.hasOwn(LEVELS, directory)) return directory;
  if (directory === 'reconstruction') return 'sfm';
  return undefined;
}

/** The source files of the two trees, relative to `root`, with `/`, sorted. */
export function sourceFiles(root = ROOT) {
  const files = [];
  const walk = (directory) => {
    for (const entry of readdirSync(join(root, directory), { withFileTypes: true })) {
      const path = `${directory}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      // A generated host view (`*.shade.typeshade.ts`) exists only during a build.
      else if (/\.(tsx?|mts|astro)$/.test(entry.name) && !/\.(d|typeshade)\.m?ts$/.test(entry.name))
        files.push(path);
    }
  };
  for (const tree of TREES) if (existsSync(join(root, tree))) walk(tree);
  return files.sort();
}

/** The TypeScript of a `.astro` file: its frontmatter and its inline scripts. */
function astroScripts(text) {
  const parts = [];
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
  if (front) parts.push(front[1]);
  for (const match of text.matchAll(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/g))
    if (!/\ssrc=/.test(match[1] ?? '')) parts.push(match[2]);
  return parts.join('\n');
}

/** The imports a file writes: specifier, type-only, form and line. */
export function importsOf(file, text) {
  const source = ts.createSourceFile(
    file,
    file.endsWith('.astro') ? astroScripts(text) : text,
    ts.ScriptTarget.Latest,
    true,
    file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const out = [];
  const add = (specifier, typeOnly, form, node) =>
    out.push({
      specifier,
      typeOnly,
      form,
      line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
    });
  const visit = (node) => {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      const clause = node.importClause;
      const named = clause?.namedBindings;
      const typeOnly =
        clause !== undefined &&
        (clause.isTypeOnly ||
          (clause.name === undefined &&
            named !== undefined &&
            ts.isNamedImports(named) &&
            named.elements.length > 0 &&
            named.elements.every((element) => element.isTypeOnly)));
      add(node.moduleSpecifier.text, typeOnly, 'import', node);
    } else if (
      ts.isExportDeclaration(node) &&
      node.moduleSpecifier &&
      ts.isStringLiteral(node.moduleSpecifier)
    ) {
      const named = node.exportClause;
      const typeOnly =
        node.isTypeOnly ||
        (named !== undefined &&
          ts.isNamedExports(named) &&
          named.elements.length > 0 &&
          named.elements.every((element) => element.isTypeOnly));
      add(node.moduleSpecifier.text, typeOnly, 'export', node);
    } else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      node.arguments.length > 0 &&
      ts.isStringLiteralLike(node.arguments[0])
    ) {
      add(node.arguments[0].text, false, 'dynamic', node);
    } else if (
      ts.isImportTypeNode(node) &&
      ts.isLiteralTypeNode(node.argument) &&
      ts.isStringLiteral(node.argument.literal)
    ) {
      add(node.argument.literal.text, true, 'type', node);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return out;
}

/** The file a relative specifier names, relative to `root`; undefined when none exists. */
function resolveSpecifier(root, from, specifier) {
  const base = join(dirname(from), specifier).split(sep).join('/');
  for (const candidate of [base, `${base}.ts`, `${base}.tsx`, `${base}/index.ts`]) {
    const path = join(root, candidate);
    if (existsSync(path) && statSync(path).isFile()) return candidate;
  }
  return undefined;
}

/**
 * The import graph: the files of the two trees, each import between two of them (`from`, `to`,
 * `typeOnly`, `form`, `line`), and each relative specifier that names no file.
 */
export function importGraph(root = ROOT) {
  const files = sourceFiles(root);
  const known = new Set(files);
  const imports = [];
  const unresolved = [];
  for (const from of files) {
    for (const found of importsOf(from, readFileSync(join(root, from), 'utf8'))) {
      if (!found.specifier.startsWith('.')) continue;
      const to = resolveSpecifier(root, from, found.specifier);
      if (to === undefined) unresolved.push({ from, specifier: found.specifier, line: found.line });
      // A file outside the two trees (a stylesheet, a script of `scripts/`) is no node.
      else if (known.has(to)) imports.push({ from, to, ...found });
    }
  }
  return { files, imports, unresolved };
}

/** The strongly connected groups of two or more files, or of one file that imports itself. */
function stronglyConnected(files, imports) {
  const next = new Map(files.map((file) => [file, []]));
  for (const { from, to } of imports) next.get(from).push(to);
  const index = new Map();
  const low = new Map();
  const stack = [];
  const onStack = new Set();
  const groups = [];
  let counter = 0;
  const connect = (file) => {
    index.set(file, counter);
    low.set(file, counter);
    counter++;
    stack.push(file);
    onStack.add(file);
    for (const to of next.get(file)) {
      if (!index.has(to)) {
        connect(to);
        low.set(file, Math.min(low.get(file), low.get(to)));
      } else if (onStack.has(to)) low.set(file, Math.min(low.get(file), index.get(to)));
    }
    if (low.get(file) !== index.get(file)) return;
    const group = [];
    let member;
    do {
      member = stack.pop();
      onStack.delete(member);
      group.push(member);
    } while (member !== file);
    if (group.length > 1 || next.get(file).includes(file)) groups.push(group.sort());
  };
  for (const file of files) if (!index.has(file)) connect(file);
  return groups.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}

/** The runtime import cycles: each a sorted list of files. */
export function runtimeCycles(graph) {
  return stronglyConnected(
    graph.files,
    graph.imports.filter((found) => !found.typeOnly),
  );
}

/**
 * The cycles that type-only imports add: each group of files that is strongly connected over
 * every import and not over the runtime imports, with the type-only imports between its files.
 */
export function typeOnlyCycles(graph) {
  const runtime = new Set(runtimeCycles(graph).map((group) => group.join(' ')));
  return stronglyConnected(graph.files, graph.imports)
    .filter((group) => !runtime.has(group.join(' ')))
    .map((files) => {
      const members = new Set(files);
      return {
        files,
        typeOnly: graph.imports.filter(
          (found) => found.typeOnly && members.has(found.from) && members.has(found.to),
        ),
      };
    });
}

/** The imports of `src/` that break the layering rule, each with the two areas. */
export function layeringBreaks(graph) {
  const breaks = [];
  for (const found of graph.imports) {
    const from = areaOf(found.from);
    const to = areaOf(found.to);
    if (from === undefined || to === undefined || from === 'entry' || to === 'entry') continue;
    if (from !== to && LEVELS[to] >= LEVELS[from])
      breaks.push({ ...found, fromArea: from, toArea: to });
  }
  return breaks;
}

/** The files of `src/` that have no area. */
export function unassignedFiles(graph) {
  return graph.files.filter((file) => file.startsWith('src/') && areaOf(file) === undefined);
}

/** The whole report, as the command prints it with `--json`. */
export function report(root = ROOT) {
  const graph = importGraph(root);
  return {
    files: graph.files.length,
    imports: graph.imports.length,
    runtimeImports: graph.imports.filter((found) => !found.typeOnly).length,
    unresolved: graph.unresolved,
    runtimeCycles: runtimeCycles(graph),
    typeOnlyCycles: typeOnlyCycles(graph),
    layeringBreaks: layeringBreaks(graph),
    unassigned: unassignedFiles(graph),
  };
}

/** The report as lines of text. */
export function formatReport(result) {
  const kind = (found) => (found.typeOnly ? 'type-only' : 'runtime  ');
  const lines = [
    'Import graph of src/ and website/src/ (docs/library-structure.md, section 3.1)',
    `${result.files} files, ${result.imports} imports, ${result.runtimeImports} of them at run time`,
    '',
    `Runtime import cycles: ${result.runtimeCycles.length}`,
    ...result.runtimeCycles.map((files) => `  ${files.join(' <-> ')}`),
    '',
    `Cycles that type-only imports add: ${result.typeOnlyCycles.length}`,
    ...result.typeOnlyCycles.flatMap(({ files, typeOnly }) => [
      `  ${files.join(' <-> ')}`,
      ...typeOnly.map((found) => `    type-only ${found.from}:${found.line} -> ${found.to}`),
    ]),
    '',
    `Imports that break the layering rule: ${result.layeringBreaks.length}, ` +
      `${result.layeringBreaks.filter((found) => !found.typeOnly).length} of them at run time`,
    ...result.layeringBreaks.map(
      (found) =>
        `  ${kind(found)} ${found.from}:${found.line} (${found.fromArea}) -> ` +
        `${found.to} (${found.toArea})`,
    ),
  ];
  if (result.unassigned.length > 0)
    lines.push(
      '',
      `Files of src/ with no area: ${result.unassigned.length}`,
      ...result.unassigned.map((file) => `  ${file}`),
    );
  if (result.unresolved.length > 0)
    lines.push(
      '',
      `Unresolved imports: ${result.unresolved.length}`,
      ...result.unresolved.map((found) => `  ${found.from}:${found.line} ${found.specifier}`),
    );
  return lines;
}

if (import.meta.main) {
  const result = report();
  if (process.argv.includes('--json')) console.log(JSON.stringify(result, null, 2));
  else for (const line of formatReport(result)) console.log(line);
  if (result.unresolved.length > 0) process.exitCode = 1;
}
