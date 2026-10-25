/**
 * Compare expanded and indexed fixed-scene observation evaluation on one synthetic scene.
 * Build baseline/candidate separately, then pass their standalone dist/gpu/index.js bundles.
 * No uploaded geometry is used. This is not a whole-SfM or actual-run RSS benchmark.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string', default: 'dist/gpu/index.js' },
    backend: { type: 'string', default: 'webgpu' },
    points: { type: 'string', default: '1024' },
    views: { type: 'string', default: '8' },
    pairs: { type: 'string', default: '3' },
    guarded: { type: 'boolean', default: false },
    mapper: { type: 'boolean', default: false },
    'timeout-ms': { type: 'string', default: '20000' },
    out: { type: 'string' },
  },
});
if (!values.baseline) throw new Error('--baseline must name a standalone baseline GPU bundle');
const points = Number(values.points),
  views = Number(values.views),
  pairs = Number(values.pairs),
  timeoutMs = Number(values['timeout-ms']);
if (
  ![points, views, pairs].every((n) => Number.isSafeInteger(n) && n > 0) ||
  views < 2 ||
  pairs > 5 ||
  points * views > 500_000
)
  throw new Error('Use 2+ views, positive integer counts, <=5 pairs and <=500,000 observations');
if (!['cpu', 'webgpu'].includes(values.backend)) throw new Error('--backend must be cpu or webgpu');
if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 60000)
  throw new Error('--timeout-ms must be 1000..60000');
const baseline = resolve(values.baseline),
  candidate = resolve(values.candidate);
const { result, errors } = await inBrowser(
  async ({ points, views, pairs, backend, mapper, guarded }) => {
    const variants = {
      baseline: await import('/observation-baseline.mjs'),
      indexed: await import('/observation-candidate.mjs'),
    };
    const { observationModel } = await import('/tests/typeshade/indexed-observations-cases.mjs');
    const model = observationModel({ points, views });
    const options = guarded ? { maxRelativeDepthStdDev: 0.1 } : {};
    const devices = {};
    const hash = async (value) =>
      [...new Uint8Array(await crypto.subtle.digest('SHA-256', value))]
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    const inputHash = await hash(new TextEncoder().encode(JSON.stringify(model)));
    const digest = async (result) => {
      const count = result.masks.reduce((sum, mask) => sum + mask.length, 0);
      const bytes = new Uint8Array(16 + count + (result.depthRejected?.length ?? 0));
      const data = new DataView(bytes.buffer);
      data.setFloat64(0, result.meanErrorPx, true);
      data.setFloat64(8, result.inlierCount, true);
      let at = 16;
      for (const mask of result.masks) {
        bytes.set(mask, at);
        at += mask.length;
      }
      if (result.depthRejected) bytes.set(result.depthRejected, at);
      return hash(bytes);
    };
    const rows = [];
    try {
      for (const [name, api] of Object.entries(variants))
        devices[name] = await api.ShadeDevice.open({ backend, fp64Flavor: 'float', poolBytes: 0 });
      // Compile and populate the same programs before timed alternating calls.
      for (const [name, api] of Object.entries(variants))
        await api.evaluateObservationsGpu(devices[name], model, 3, options);
      for (let pair = 0; pair < pairs; pair++)
        for (const name of pair % 2 ? ['indexed', 'baseline'] : ['baseline', 'indexed']) {
          const device = devices[name],
            api = variants[name];
          const start = performance.now();
          const pending = api.evaluateObservationsGpu(device, model, 3, options);
          const hostPreparationMs = performance.now() - start;
          const result = await pending;
          const elapsedMs = performance.now() - start;
          rows.push({
            pair,
            variant: name,
            elapsedMs,
            hostPreparationMs,
            digest: await digest(result),
            uploadBytes: result.execution.applicationUploadBytes,
            readbackBytes: result.execution.applicationReadbackBytes,
            dispatches: result.execution.report.dispatches,
            liveBytes: device.statistics.liveBytes,
          });
        }
      const mapperRows = [];
      if (mapper) {
        const { mappingFixture } = await import('/tests/typeshade/geometry-cases.mjs');
        for (const [name, api] of Object.entries(variants)) {
          const fixture = mappingFixture();
          const mapped = await api.reconstructIncrementalGpu(
            devices[name],
            fixture.images,
            fixture.pairs,
            {
              ransacIterations: 16,
              seed: 17,
              bundleIterations: 0,
              finalBundleIterations: 0,
            },
          );
          const bitWords = [];
          for (const view of mapped.reconstruction.views)
            for (const array of [view.pose.R, view.pose.t])
              bitWords.push(...new Uint32Array(array.buffer, array.byteOffset, array.length * 2));
          for (const point of mapped.reconstruction.landmarks)
            bitWords.push(
              ...new Uint32Array(point.xyz.buffer, point.xyz.byteOffset, point.xyz.length * 2),
            );
          const signature = {
            reconstruction: mapped.reconstruction,
            diagnostics: mapped.diagnostics,
            bitWords,
          };
          mapperRows.push({
            variant: name,
            digest: await hash(new TextEncoder().encode(JSON.stringify(signature))),
            views: mapped.reconstruction.views.length,
            points: mapped.reconstruction.landmarks.length,
            liveBytes: devices[name].statistics.liveBytes,
          });
        }
      }
      return {
        backend,
        points,
        views,
        observations: points * views,
        options,
        rows,
        mapper: mapperRows,
        inputUnchanged: inputHash === (await hash(new TextEncoder().encode(JSON.stringify(model)))),
        devices: Object.fromEntries(
          Object.entries(devices).map(([name, device]) => [name, device.info]),
        ),
      };
    } finally {
      for (const device of Object.values(devices)) await device.dispose();
    }
  },
  { points, views, pairs, backend: values.backend, mapper: values.mapper, guarded: values.guarded },
  {
    timeoutMs,
    serve: { '/observation-baseline.mjs': baseline, '/observation-candidate.mjs': candidate },
  },
);
assert.deepEqual(errors, []);
assert.equal(result.inputUnchanged, true);
for (const row of result.rows) {
  assert.equal(row.digest, result.rows[0].digest, 'projection, depth masks or mean bits changed');
  assert.equal(row.liveBytes, 0, 'active resident storage leaked');
}
for (const row of result.mapper) {
  assert.equal(
    row.digest,
    result.mapper[0].digest,
    'mapper numerical result or diagnostics changed',
  );
  assert.equal(row.liveBytes, 0);
}
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
result.summary = Object.fromEntries(
  ['baseline', 'indexed'].map((name) => {
    const rows = result.rows.filter((row) => row.variant === name);
    return [
      name,
      {
        medianMs: median(rows.map((row) => row.elapsedMs)),
        medianHostPreparationMs: median(rows.map((row) => row.hostPreparationMs)),
        uploadBytes: rows[0].uploadBytes,
        readbackBytes: rows[0].readbackBytes,
        dispatches: rows[0].dispatches,
      },
    ];
  }),
);
result.bundleSha256 = Object.fromEntries(
  await Promise.all(
    [
      ['baseline', baseline],
      ['indexed', candidate],
    ].map(async ([name, file]) => [
      name,
      createHash('sha256')
        .update(await readFile(file))
        .digest('hex'),
    ]),
  ),
);
const json = JSON.stringify(result, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), json);
process.stdout.write(json);
