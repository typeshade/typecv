/** GPU counterpart of reconstruct-sequence.mjs: runs reconstructIncrementalGpu in Chromium. */
import fs from 'node:fs';
import path from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const [input, output, optionsPath] = process.argv.slice(2);
if (!input || !output)
  throw new Error(
    'Usage: node scripts/reconstruct-sequence-gpu.mjs sequence.json output [options.json]',
  );
if (fs.existsSync(output) && fs.readdirSync(output).length)
  throw new Error('Output directory must be empty to avoid stale model files');
const options = {
  seed: 42,
  ransacIterations: 750,
  ...(optionsPath ? JSON.parse(fs.readFileSync(optionsPath, 'utf8')) : {}),
};
const timeoutMs = Number(process.env.TYPECV_GPU_TIMEOUT_MS ?? 300000);
const { result, errors } = await inBrowser(
  async ({ options, hardware }) => {
    const api = await import('/dist/gpu.js');
    const data = await (await fetch('/sequence.json')).json();
    const views = data.views.map((v) => ({
      id: v.id,
      camera: v.camera,
      keypoints: Float64Array.from(v.keypoints),
    }));
    const pairs = data.pairs.map((p) => ({
      first: p.first,
      second: p.second,
      matches: Uint32Array.from(p.matches),
      ...(p.distances ? { distances: Float64Array.from(p.distances) } : {}),
    }));
    const adapter = await navigator.gpu.requestAdapter();
    const info = adapter.info ?? {};
    const device = await api.ShadeDevice.open({ backend: 'webgpu', solverCheckpoints: true });
    const started = performance.now();
    try {
      const result = await api.reconstructIncrementalGpu(device, views, pairs, options);
      const elapsedMs = performance.now() - started;
      return {
        adapter: {
          vendor: info.vendor,
          architecture: info.architecture,
          description: info.description,
          hardware,
        },
        elapsedMs,
        model: {
          views: result.reconstruction.views.map((v) => ({
            ...v,
            pose: { R: [...v.pose.R], t: [...v.pose.t] },
          })),
          landmarks: result.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
        },
        tracks: result.tracks.length,
        unregistered: result.unregistered,
        diagnostics: result.diagnostics,
        execution: {
          backend: result.execution.backend,
          stages: result.execution.stages.map((s) => ({
            operation: s.operation,
            elapsedMs: s.elapsedMs,
            dispatches: s.execution.report?.dispatches,
            submissions: s.execution.report?.submissions,
            problem: s.problem,
          })),
          gpuBundleAdjustments: result.execution.gpuBundleAdjustments,
        },
        statistics: device.statistics,
      };
    } catch (error) {
      return { failure: String(error), stack: error?.stack };
    } finally {
      await device.dispose();
    }
  },
  { options, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
  { timeoutMs, serve: { '/sequence.json': path.resolve(input) } },
);
if (errors.length) console.error(errors);
fs.mkdirSync(output, { recursive: true });
if (result.failure) {
  fs.writeFileSync(path.join(output, 'failure.json'), JSON.stringify(result, null, 2) + '\n');
  throw new Error(result.failure);
}
const summary = {
  source: JSON.parse(fs.readFileSync(input, 'utf8')).metadata,
  options,
  backend: result.execution.backend,
  adapter: result.adapter,
  elapsedMs: result.elapsedMs,
  registeredViews: result.model.views.length,
  landmarks: result.model.landmarks.length,
  observations: result.model.landmarks.reduce((sum, p) => sum + p.observations.length, 0),
  tracks: result.tracks,
  unregistered: result.unregistered,
  diagnostics: result.diagnostics,
  execution: result.execution,
  statistics: result.statistics,
};
fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
fs.writeFileSync(path.join(output, 'model.json'), JSON.stringify(result.model) + '\n');
console.log(
  JSON.stringify({
    backend: summary.backend,
    registered: summary.registeredViews,
    points: summary.landmarks,
    errorPx: summary.diagnostics.meanReprojectionErrorPx,
    elapsedMs: summary.elapsedMs,
  }),
);
