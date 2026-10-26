import test from 'node:test';
import assert from 'node:assert/strict';
import {
  areaOf,
  importGraph,
  importsOf,
  layeringBreaks,
  runtimeCycles,
  typeOnlyCycles,
  unassignedFiles,
} from '../scripts/import-graph.mjs';

// The baseline of docs/library-structure.md, section 3.1 ("The layers today") and section 5,
// phase 0: today's runtime import cycles and today's imports that break the layering rule. A
// phase that removes a cycle or a break changes this test, and its pull request names the change
// (section 4, rule 2). `node scripts/import-graph.mjs` prints the same report with line numbers.

const graph = importGraph();

test('every relative import names a file', () => {
  assert.deepEqual(graph.unresolved, []);
});

test('the runtime import cycles of src/ and website/src/ are the three of section 1', () => {
  assert.deepEqual(runtimeCycles(graph), [
    ['src/sfm/bundle/bundle-adjustment.ts', 'src/sfm/bundle/joint-focal.ts'],
    ['src/sfm/bundle/gpu/bundle.ts', 'src/sfm/bundle/gpu/joint-focal.ts'],
    ['src/sfm/focal/focal-search.ts', 'src/sfm/mapper/mapper.ts'],
  ]);
});

test('type-only imports add three cycles (sparse audit, section 3.6)', () => {
  assert.deepEqual(
    typeOnlyCycles(graph).map(({ files, typeOnly }) => ({
      files,
      typeOnly: typeOnly.map(({ from, to }) => `${from} -> ${to}`),
    })),
    [
      {
        files: ['src/geometry/refinement.ts', 'src/geometry/two-view.ts'],
        typeOnly: ['src/geometry/refinement.ts -> src/geometry/two-view.ts'],
      },
      {
        files: [
          'src/sfm/bundle/bundle-adjustment.ts',
          'src/sfm/bundle/bundle-directions.ts',
          'src/sfm/bundle/joint-focal.ts',
        ],
        typeOnly: ['src/sfm/bundle/bundle-directions.ts -> src/sfm/bundle/bundle-adjustment.ts'],
      },
      {
        files: [
          'src/sfm/depth/depth-publication.ts',
          'src/sfm/focal/focal-search.ts',
          'src/sfm/mapper/mapper.ts',
        ],
        typeOnly: ['src/sfm/depth/depth-publication.ts -> src/sfm/mapper/mapper.ts'],
      },
    ],
  );
});

test('every file of src/ has an area of section 3.1 or is an entry point', () => {
  assert.deepEqual(unassignedFiles(graph), []);
  assert.equal(areaOf('src/index.ts'), 'entry');
  assert.equal(areaOf('src/gpu/sift.ts'), 'features');
  assert.equal(areaOf('src/sfm/two-view.ts'), 'geometry');
  assert.equal(areaOf('src/reconstruction/types.ts'), 'sfm');
  // Phase 1b moves files into area directories: a moved file takes its directory's area.
  assert.equal(areaOf('src/features/gpu/sift.ts'), 'features');
  assert.equal(areaOf('src/sfm/bundle/gpu/bundle.ts'), 'sfm');
  assert.equal(areaOf('src/gpu/new-module.ts'), undefined);
});

test('eight imports break the layering rule, seven of them at run time (section 3.1)', () => {
  const breaks = layeringBreaks(graph).map(
    (found) =>
      `${found.typeOnly ? 'type-only' : 'runtime'} ${found.from} (${found.fromArea}) -> ` +
      `${found.to} (${found.toArea})`,
  );
  assert.deepEqual(breaks, [
    'runtime src/analysis/gpu/hough.ts (analysis) -> src/gpu/cloud-clean.ts (mvs)',
    'runtime src/analysis/gpu/hough.ts (analysis) -> src/features/gpu/sift.ts (features)',
    'runtime src/analysis/gpu/slic.ts (analysis) -> src/gpu/cloud-clean.ts (mvs)',
    'runtime src/analysis/gpu/slic.ts (analysis) -> src/features/gpu/sift.ts (features)',
    'runtime src/features/gpu/sift.ts (features) -> src/motion/gpu/klt.ts (motion)',
    'type-only src/geometry/gpu/observations.ts (geometry) -> src/reconstruction/types.ts (sfm)',
    'runtime src/geometry/gpu/observations.ts (geometry) -> src/sfm/bundle/bundle-adjustment.ts (sfm)',
    'runtime src/gpu/mvs.ts (mvs) -> src/reconstruction/observation-paths.ts (sfm)',
  ]);
});

test('an import is type-only exactly when TypeScript erases it', () => {
  const found = importsOf(
    'probe.ts',
    [
      "import type { A } from './a.ts';",
      "import { type B, type C } from './b.ts';",
      "import { type D, E } from './c.ts';",
      "import F from './d.ts';",
      "import './e.ts';",
      "export type { G } from './f.ts';",
      "export * from './g.ts';",
      "const h = await import('./h.ts');",
      "let i: import('./i.ts').I;",
    ].join('\n'),
  );
  assert.deepEqual(
    found.map(({ specifier, typeOnly, form }) => [specifier, typeOnly, form]),
    [
      ['./a.ts', true, 'import'],
      ['./b.ts', true, 'import'],
      ['./c.ts', false, 'import'],
      ['./d.ts', false, 'import'],
      ['./e.ts', false, 'import'],
      ['./f.ts', true, 'export'],
      ['./g.ts', false, 'export'],
      ['./h.ts', false, 'dynamic'],
      ['./i.ts', true, 'type'],
    ],
  );
});
