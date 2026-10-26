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
// phase 0: today's runtime import cycles and the imports that break the layering rule, none since
// phase 1c. A phase that removes a cycle or a break changes this test, and its pull request names
// the change (section 4, rule 2). `node scripts/import-graph.mjs` prints the same report with
// line numbers.

const graph = importGraph();

test('every relative import names a file', () => {
  assert.deepEqual(graph.unresolved, []);
});

test('the mapper-focal cycle is the only runtime import cycle (section 5, phase 1b)', () => {
  assert.deepEqual(runtimeCycles(graph), [
    ['src/sfm/focal/focal-search.ts', 'src/sfm/mapper/mapper.ts'],
  ]);
});

test('type-only imports add two cycles (sparse audit, section 3.6)', () => {
  assert.deepEqual(
    typeOnlyCycles(graph).map(({ files, typeOnly }) => ({
      files,
      typeOnly: typeOnly.map(({ from, to }) => `${from} -> ${to}`),
    })),
    [
      {
        files: [
          'src/sfm/bundle/bundle-adjustment.ts',
          'src/sfm/bundle/bundle-directions.ts',
          'src/sfm/bundle/bundle-shared.ts',
          'src/sfm/bundle/joint-focal.ts',
        ],
        typeOnly: [
          'src/sfm/bundle/bundle-directions.ts -> src/sfm/bundle/bundle-adjustment.ts',
          'src/sfm/bundle/bundle-shared.ts -> src/sfm/bundle/bundle-adjustment.ts',
          'src/sfm/bundle/joint-focal.ts -> src/sfm/bundle/bundle-adjustment.ts',
        ],
      },
      {
        files: [
          'src/sfm/config/incremental-controls.ts',
          'src/sfm/depth/depth-publication.ts',
          'src/sfm/focal/focal-search.ts',
          'src/sfm/mapper/mapper.ts',
          'src/sfm/pairs/pair-verdict.ts',
          'src/sfm/protocol/errors.ts',
        ],
        typeOnly: [
          'src/sfm/config/incremental-controls.ts -> src/sfm/mapper/mapper.ts',
          'src/sfm/depth/depth-publication.ts -> src/sfm/mapper/mapper.ts',
          'src/sfm/pairs/pair-verdict.ts -> src/sfm/mapper/mapper.ts',
          'src/sfm/protocol/errors.ts -> src/sfm/focal/focal-search.ts',
          'src/sfm/protocol/errors.ts -> src/sfm/mapper/mapper.ts',
        ],
      },
    ],
  );
});

test('every file of src/ has an area of section 3.1 or is an entry point', () => {
  assert.deepEqual(unassignedFiles(graph), []);
  assert.equal(areaOf('src/index.ts'), 'entry');
  // A file in an area directory takes that area.
  assert.equal(areaOf('src/features/gpu/sift.ts'), 'features');
  assert.equal(areaOf('src/sfm/bundle/gpu/bundle.ts'), 'sfm');
  assert.equal(areaOf('src/contracts/scene.ts'), 'contracts');
  assert.equal(areaOf('src/sfm/two-view.ts'), 'sfm');
  // An alias outside the area directories takes the area of the module it names.
  assert.equal(areaOf('src/gpu/sift.ts'), 'features');
  assert.equal(areaOf('src/gpu/bundle.shade.ts'), 'sfm');
  assert.equal(areaOf('src/reconstruction/types.ts'), 'contracts');
  assert.equal(areaOf('src/reconstruction/observation-paths.ts'), 'geometry');
  // The float-matching kernel stays at its path (#250, deviation 1).
  assert.equal(areaOf('src/gpu/kernels/float-matching.shade.ts'), 'features');
  assert.equal(areaOf('src/gpu/new-module.ts'), undefined);
});

test('no import breaks the layering rule (section 5, phase 1c)', () => {
  assert.deepEqual(layeringBreaks(graph), []);
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
