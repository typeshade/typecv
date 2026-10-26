// Phase 0 of the refactoring programme (`docs/library-structure.md`, section 5, "Baselines": the
// start payloads and record options of about 40 query strings), the part that runs in Node. For
// each query string of `LAB_QUERIES` it records what the page's query helpers (`SfmLab.tsx`) and
// the exported parsers return. The browser test `run-options-browser.test.mjs` records what the
// page then sends and keeps for the same strings. The fixture is
// `tests/fixtures/lab-query-parsers.json`; phase 7 keeps the values identical.
import test from 'node:test';
import { canonical, checkBaseline } from './lab-baseline.mjs';
import { LAB_QUERIES, LAB_QUERY_BASE, loadPageModule } from './lab-page-module.mjs';
import { decodeOverride } from './src/lib/decode-protocol.ts';
import { denseQualityOptions } from './src/lib/dense-quality.ts';
import { keyframeGateMode } from './src/lib/keyframe-gate.ts';
import { multiVideoQuery } from './src/lib/multi-video.ts';
import {
  featureFrontEnd,
  guidedKernel,
  kltSetup,
  kltTracking,
  orbTuning,
  siftQuality,
} from './src/lib/sift-quality.ts';
import { videoSampleFps } from './src/lib/video-sampling.ts';

test('the page query helpers and the parsers of the lab query strings', async () => {
  const page = await loadPageModule();
  const rows = LAB_QUERIES.map(([kind, query, base = true]) => {
    const search = '?' + [base ? LAB_QUERY_BASE : '', query].filter(Boolean).join('&');
    page.setSearch(search);
    return {
      query: search.slice(1),
      kind,
      helpers: {
        boundedOverride: page.boundedOverride(),
        budgetOverride: page.budgetOverride(),
        denseOverride: page.denseOverride(),
        denseGpuTimestampsOverride: page.denseGpuTimestampsOverride(),
        denseMeshOverride: page.denseMeshOverride(),
        denseCleanOverride: page.denseCleanOverride(),
        denseFusionOverride: page.denseFusionOverride(),
        fp64Override: page.fp64Override(),
        pageKeyframeGate: page.pageKeyframeGate(),
        pageSampleFps: page.pageSampleFps(),
      },
      parsers: {
        // The first octave's default follows the analysis edge: both sides of 640 px.
        siftQuality320: siftQuality(320, search),
        siftQuality1280: siftQuality(1280, search),
        guidedKernel: guidedKernel(search),
        featureFrontEnd: featureFrontEnd(search),
        orbTuning: orbTuning(search),
        kltTracking: kltTracking(search),
        kltSetup: kltSetup(search),
        videoSampleFps: videoSampleFps(search),
        keyframeGateMode: keyframeGateMode(search),
        denseQualityOptions: denseQualityOptions(search),
        multiVideoQuery: multiVideoQuery(search),
        decodeOverride: decodeOverride(search),
      },
    };
  });
  await checkBaseline(
    new URL('../tests/fixtures/lab-query-parsers.json', import.meta.url),
    JSON.stringify(canonical(rows), null, 1) + '\n',
  );
});
