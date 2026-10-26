// The page module of the lab, `src/components/SfmLab.tsx`, for the baselines of phase 0
// (`docs/library-structure.md`, section 5): the record builders `buildRunRecord` and
// `buildMultiRunRecord`, and the page's query helpers. They are module functions that the page
// does not export. So the test bundles the module with one added export line and evaluates it
// in a `vm` realm whose browser globals are the fixed `ENVIRONMENT`. The product code is not
// changed.
//
// Phase 3 moves the builders to `run-record/assemble.ts` with an `environment` argument, and
// phase 7 replaces the query helpers with `run-options`. Then this file imports those modules
// instead, and the tests that use it keep their fixtures.
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import vm from 'node:vm';
import { build } from 'esbuild';

const PAGE = new URL('./src/components/SfmLab.tsx', import.meta.url).pathname;

/** The names the bundle adds to the module's exports. */
export const PAGE_FUNCTIONS = [
  'buildRunRecord',
  'buildMultiRunRecord',
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
 * `devicePixelRatio`). They become the record's `client` block.
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
 * Bundle and evaluate the page module. The result holds the functions of `PAGE_FUNCTIONS` and
 * `setSearch(search)`, which sets `location.search` for the query helpers.
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
    setSearch(search) {
      sandbox.location = { ...sandbox.location, search };
    },
  };
}
