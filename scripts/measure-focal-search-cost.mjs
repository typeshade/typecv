// Cost of the automatic focal search (docs/auto-focal.md, section 8) on a headless WebGPU adapter
// (SwiftShader without a physical GPU): pair verifications, dispatches, GPU time per kernel and
// wall time of one `reconstructIncrementalGpu` run with the lab's controls.
//
//   npm run build
//   node scripts/measure-focal-search-cost.mjs --sample pyramid --search 1
//   node scripts/measure-focal-search-cost.mjs --sample south-building --search 0 --out cost.json
//
// --sample   pyramid | south-building | gerrard-hall | person-hall | graham-hall
// --search   1 runs with `focalSearch: true` (default), 0 without it
// --views    the search's subset size (default 5)
// --frames   use only the first N frames (default all)
// --start    the start focal / width (default 0.8)
// --out      write the result as JSON
//
// SwiftShader is a CPU emulation: its times rank work, they are not the times of a physical GPU.
import { writeFile } from 'node:fs/promises';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { LAB, camerasAt, loadPyramid, loadSample } from './measure-focal.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const sample = flag('--sample', 'pyramid');
const search = flag('--search', '1') === '1';
const views = Number(flag('--views', 5));
const start = Number(flag('--start', 0.8));
const out = flag('--out');
const frameLimit = flag('--frames') === undefined ? undefined : Number(flag('--frames'));

const { frames, pairs } = sample === 'pyramid' ? loadPyramid() : loadSample(sample);
const kept = frames.slice(0, frameLimit ?? frames.length);
const ids = new Set(kept.map((f) => f.id));
const images = camerasAt(kept, start, false).map((v) => ({
  id: v.id,
  camera: v.camera,
  keypoints: Array.from(v.keypoints),
}));
const edges = pairs
  .filter((p) => ids.has(p.first) && ids.has(p.second))
  .map((p) => ({ first: p.first, second: p.second, matches: Array.from(p.matches) }));

const { result, errors } = await inBrowser(
  async ({ images, edges, lab, search, views }) => {
    const api = await import('/dist/gpu.js');
    const device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
    const table = new Map();
    let gpuMs = 0,
      dispatches = 0,
      submissions = 0;
    const fuse = device.shade.fuse.bind(device.shade);
    device.shade.fuse = async (body, options) => {
      const report = await fuse(body, options);
      dispatches += report.dispatches;
      submissions += report.submissions;
      for (const g of report.groups) {
        const name = g.calls.join('+');
        const row = table.get(name) ?? { groups: 0, gpuNs: 0 };
        row.groups++;
        if (Number.isFinite(g.gpuTime)) {
          row.gpuNs += g.gpuTime;
          gpuMs += g.gpuTime / 1e6;
        }
        table.set(name, row);
      }
      return report;
    };
    const stages = new Map();
    const t0 = performance.now();
    try {
      const run = await api.reconstructIncrementalGpu(
        device,
        images.map((v) => ({
          ...v,
          keypoints: Float64Array.from(v.keypoints),
        })),
        edges.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
        {
          ...lab,
          refineSharedFocal: true,
          focalSearch: search ? { views } : undefined,
          onStage: (s) => {
            const row = stages.get(s.operation) ?? { calls: 0, ms: 0 };
            row.calls++;
            row.ms += s.elapsedMs;
            stages.set(s.operation, row);
          },
        },
      );
      const found = run.diagnostics.focalSearch;
      return {
        elapsedMs: performance.now() - t0,
        views: run.reconstruction.views.length,
        landmarks: run.reconstruction.landmarks.length,
        focal:
          run.reconstruction.views[0].camera.fx / (2 * run.reconstruction.views[0].camera.cx + 1),
        search: found && {
          ratio: found.ratio,
          gridRatio: found.gridRatio,
          candidates: found.candidates.map((c) => [
            +c.ratio.toFixed(4),
            c.views,
            +c.score.toFixed(5),
            c.observations,
          ]),
        },
        pairVerificationCalls: stages.get('pairVerification')?.calls ?? 0,
        stages: Object.fromEntries(
          [...stages].map(([k, v]) => [k, { calls: v.calls, ms: Math.round(v.ms) }]),
        ),
        dispatches,
        submissions,
        gpuMs: Math.round(gpuMs),
        kernels: [...table]
          .map(([name, r]) => ({ name, groups: r.groups, gpuMs: Math.round(r.gpuNs / 1e5) / 10 }))
          .sort((a, b) => b.gpuMs - a.gpuMs)
          .slice(0, 12),
      };
    } finally {
      await device.dispose();
    }
  },
  { images, edges, lab: LAB, search, views },
  { timeoutMs: 3600000 },
);
if (errors.length) console.error(errors);
const record = { sample, search, views, start, frames: kept.length, ...result };
console.log(JSON.stringify(record, null, 1));
if (out) await writeFile(out, JSON.stringify(record, null, 1));
