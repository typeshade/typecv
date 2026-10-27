// The page module of the lab for the baselines of phase 0 (`docs/library-structure.md`, section
// 5): the record builders `buildRunRecord` and `buildMultiRunRecord`, and the page's query
// helpers.
//
// The builders are in `src/lib/run-record/page.ts` since phase 3, with the browser globals as
// their `environment` argument. This file passes them the fixed `ENVIRONMENT`. Before phase 3 they
// were module functions of `src/components/SfmLab.tsx` that read these globals from the page's
// realm.
//
// The query helpers are module functions of `src/components/SfmLab.tsx` that the page does not
// export. So the test bundles the module with one added export line and evaluates it in a `vm`
// realm. The product code is not changed. Phase 7 replaces the query helpers with `run-options`.
// Then this file imports that module instead, and the tests that use it keep their fixtures.
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import vm from 'node:vm';
import { build } from 'esbuild';
import { buildMultiRunRecord, buildRunRecord } from './src/lib/run-record/page.ts';

const PAGE = new URL('./src/components/SfmLab.tsx', import.meta.url).pathname;

/** The names the bundle adds to the module's exports. */
export const PAGE_FUNCTIONS = [
  'boundedOverride',
  'budgetOverride',
  'denseOverride',
  'denseGpuTimestampsOverride',
  'denseMeshOverride',
  'denseCleanOverride',
  'denseFusionOverride',
  'fp64Override',
  'pageKeyframeGate',
  'pageSampleFps',
];

/**
 * The browser globals that the record builders read (`navigator`, `screen`,
 * `devicePixelRatio`): their `environment` argument (`RunRecordEnvironment`). They become the
 * record's `client` block.
 */
export const ENVIRONMENT = {
  navigator: {
    userAgent: 'Mozilla/5.0 (X11; Linux x86_64) TypeCV-Baseline/1.0',
    userAgentData: { platform: 'Linux' },
    language: 'en-US',
    hardwareConcurrency: 8,
    deviceMemory: 8,
    maxTouchPoints: 0,
  },
  screen: { width: 1920, height: 1080 },
  devicePixelRatio: 2,
};

/**
 * Bundle and evaluate the page module. The result holds the functions of `PAGE_FUNCTIONS`, the two
 * record builders with `environment` as their environment, and `setSearch(search)`, which sets
 * `location.search` for the query helpers.
 */
export async function loadPageModule(environment = ENVIRONMENT) {
  const source = await readFile(PAGE, 'utf8');
  const result = await build({
    stdin: {
      contents: source + `\nexport { ${PAGE_FUNCTIONS.join(', ')} };\n`,
      loader: 'tsx',
      sourcefile: PAGE,
      resolveDir: dirname(PAGE),
    },
    bundle: true,
    write: false,
    format: 'iife',
    globalName: 'LabPage',
    platform: 'browser',
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"production"' },
    logLevel: 'silent',
    // The page imports style sheets; they go to a second output file, which the test drops.
    outdir: 'lab-page',
  });
  const code = result.outputFiles.find((file) => file.path.endsWith('.js')).text;
  const sandbox = {
    console,
    URL,
    URLSearchParams,
    navigator: structuredClone(environment.navigator),
    screen: { ...environment.screen },
    devicePixelRatio: environment.devicePixelRatio,
    location: { search: '', pathname: '/', href: 'https://lab.typecv.test/' },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(code, context, { filename: 'SfmLab.js' });
  const page = context.LabPage;
  return {
    ...Object.fromEntries(PAGE_FUNCTIONS.map((name) => [name, page[name]])),
    buildRunRecord: (input) => buildRunRecord(environment, input),
    buildMultiRunRecord: (input) => buildMultiRunRecord(environment, input),
    setSearch(search) {
      sandbox.location = { ...sandbox.location, search };
    },
  };
}

/**
 * The query strings of the start payload baseline (`run-options.test.mjs`,
 * `run-options-browser.test.mjs`): `[kind, query, base]`. Each string carries
 * `LAB_QUERY_BASE` unless `base` is false.
 */
export const LAB_QUERY_BASE = 'decode=main&fps=1';
export const LAB_QUERIES = [
  ['single', ''],
  ['single', 'gate=legacy'],
  ['single', 'gate=phase'],
  ['single', 'gate=other'],
  ['single', 'octave=-1'],
  ['single', 'octave=0'],
  ['single', 'guided=0'],
  ['single', 'guided=1'],
  ['single', 'guided=tiled'],
  ['single', 'clahe=1'],
  ['single', 'features=orb'],
  ['single', 'features=orb&orbFeatures=2000&orbRatio=0.9&orbDistance=64'],
  ['single', 'features=orb&orbFeatures=9000&orbRatio=1&orbDistance=0'],
  ['single', 'klt=1'],
  ['single', 'klt=1&kltCapacity=1024&kltFill=1&kltMatching=tracks&kltGate=1&gate=legacy'],
  ['single', 'klt=0&kltCapacity=99999'],
  ['single', 'decode=main&fps=2', false],
  ['single', 'decode=main&fps=0', false],
  ['single', 'bounded=1'],
  ['single', 'fp64=float'],
  ['single', 'fp64=integer'],
  ['single', 'fp64=double'],
  ['single', 'budget=100'],
  ['single', 'budget=0'],
  ['single', 'budget=60001'],
  ['single', 'dense=1'],
  ['single', 'denseFusion=pixel'],
  ['single', 'denseFusion=tsdf&denseClean=1'],
  ['single', 'denseClean=0&denseFusion=pixel'],
  ['single', 'denseMesh=1'],
  ['single', 'gpuTimestamps=1'],
  ['single', 'denseBoundary=1&denseFiniteGain=0.5'],
  ['single', 'denseBoundary=2&denseFiniteGain=3'],
  ['single', 'fps=1', false],
  [
    'single',
    'gate=legacy&octave=-1&guided=tiled&clahe=1&bounded=1&fp64=integer&budget=500&denseMesh=1',
  ],
  ['multi', ''],
  ['multi', 'canonical=0'],
  ['multi', 'maxKeyframes=30'],
  ['multi', 'maxKeyframes=1'],
  ['multi', 'maxFeatures=2048'],
  ['multi', 'clahe=1&octave=-1'],
  ['multi', 'fp64=float&budget=100'],
  ['multi', 'denseFusion=pixel&denseMesh=1&gpuTimestamps=1'],
  ['multi', 'gate=legacy&klt=1&features=orb&bounded=1'],
];
