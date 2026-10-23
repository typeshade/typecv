import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { transform } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { generalVideoFixture } from '../tests/fixtures/general-video.ts';
import { compareSeedGauge } from './geometry-comparison.mjs';

const arg = (n, f) => process.argv.find((x) => x.startsWith(`--${n}=`))?.slice(n.length + 3) ?? f;
const repeats = Number(arg('repeats', '6')),
  output = resolve(
    arg('output', 'docs/experiments/general-video/structure-quality-rgb-comparison.json'),
  );
if (!Number.isSafeInteger(repeats) || repeats < 2 || repeats > 8 || repeats % 2)
  throw new RangeError('Even --repeats=2..8 required');
const sha = (v) => createHash('sha256').update(v).digest('hex');
const cases = [],
  truth = [];
for (const [scenario, overlay] of [
  ['translation', false],
  ['rotation', false],
  ['translation', true],
]) {
  const root = resolve('data/general-video-fixtures', scenario),
    input = JSON.parse(await readFile(resolve(root, 'input/frames.json'))),
    calibration = JSON.parse(await readFile(resolve(root, 'known-calibration.json')));
  const frames = [];
  for (const f of input.frames) {
    const bytes = await readFile(resolve(root, 'input', f.image));
    assert.equal(sha(bytes), f.sha256);
    frames.push({
      ...f,
      camera: calibration.frames.find((c) => c.id === f.id).camera,
      png: bytes.toString('base64'),
    });
  }
  cases.push({ frames, overlay });
  truth.push({ scenario, views: generalVideoFixture(scenario).truth.views });
}
const joinFile = '/tmp/typecv-structure-quality-klt-join.mjs';
await writeFile(
  joinFile,
  (
    await transform(await readFile('website/src/lib/klt-join.ts', 'utf8'), {
      loader: 'ts',
      format: 'esm',
      target: 'es2022',
    })
  ).code,
);
const { result, errors } = await inBrowser(
  async ({ cases, repeats }) => {
    const api = await import('/dist/gpu.js'),
      core = await import('/dist/index.js'),
      { mergeTrackerPairs } = await import('/quality/klt-join.mjs');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const gpu = await adapter.requestDevice(),
      gpuErrors = [];
    gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
    const description = [
      adapter.info?.vendor,
      adapter.info?.architecture,
      adapter.info?.device,
      adapter.info?.description,
    ]
      .filter(Boolean)
      .join(' ');
    const device = await api.ShadeDevice.open({
      gpuDevice: gpu,
      boundedSubmissions: false,
      solverCheckpoints: true,
      maxModelsPerBatch: 32,
    });
    let peak = 0;
    for (const key of ['allocate', 'allocateU32', 'allocateF64']) {
      const original = device[key].bind(device);
      device[key] = (...args) => {
        const r = original(...args);
        peak = Math.max(peak, device.statistics.liveBytes);
        return r;
      };
    }
    const records = [];
    try {
      for (const entry of cases) {
        const before = device.statistics,
          started = performance.now(),
          handles = [],
          images = [],
          pairs = [],
          masks = [],
          regions = [],
          featureCounts = [];
        let tracker,
          decodeMs = 0,
          featureMs = 0,
          matchingMs = 0,
          kltMs = 0,
          kltAdded = 0;
        peak = before.liveBytes;
        try {
          for (const frame of entry.frames) {
            let start = performance.now();
            const bitmap = await createImageBitmap(
              new Blob([Uint8Array.from(atob(frame.png), (c) => c.charCodeAt(0))], {
                type: 'image/png',
              }),
            );
            const width = bitmap.width,
              height = bitmap.height,
              canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(bitmap, 0, 0);
            bitmap.close();
            const boxes = [];
            if (entry.overlay) {
              const left = 65 + 4 * frame.id,
                top = 65,
                w = 170,
                h = 130;
              // Deliberately independent image motion plus a fixed HUD; these are composited
              // regions, not recovered game meshes. Oracle bounds are declared as GT masks.
              for (const [x, y, bw, bh, seed] of [
                [left, top, w, h, 391],
                [0, 0, width, 30, 812],
              ]) {
                let s = seed;
                const tile = ctx.createImageData(bw, bh);
                for (let i = 0; i < tile.data.length; i += 4) {
                  s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
                  const v = s >>> 24;
                  tile.data[i] = v;
                  tile.data[i + 1] = (v * 3) % 256;
                  tile.data[i + 2] = 255 - v;
                  tile.data[i + 3] = 255;
                }
                ctx.putImageData(tile, x, y);
                boxes.push({
                  left: x / width,
                  top: y / height,
                  right: (x + bw) / width,
                  bottom: (y + bh) / height,
                });
              }
            }
            regions.push(boxes);
            const rgba = ctx.getImageData(0, 0, width, height).data;
            decodeMs += performance.now() - start;
            start = performance.now();
            const handle = await api.extractSiftGpuResident(device, rgba, width, height, {
              maxFeatures: 800,
              firstOctave: 0,
              grayForKlt: { levels: api.kltOptionsFor(width).levels },
            });
            handles.push(handle);
            const keypoints = await handle.readKeypoints(),
              image = { id: frame.id, camera: frame.camera, keypoints };
            images.push(image);
            featureCounts.push(keypoints.length / 2);
            masks.push(core.cameraMaskFromRegions(image, width, height, boxes, 'ground-truth', 6));
            featureMs += performance.now() - start;
            start = performance.now();
            tracker ??= await api.createKltTracker(device, width, height, {
              options: { capacity: 800 },
              fillIn: false,
            });
            await tracker.push({ sift: handle }, { frameId: frame.id, select: false });
            const joined = await tracker.join({
              frameId: frame.id,
              keypoints,
              views: images.slice(0, -1).map((v) => v.id),
            });
            kltMs += performance.now() - start;
            const descriptor = [];
            start = performance.now();
            for (let first = Math.max(0, frame.id - 3); first < frame.id; first++) {
              const m = await api.matchFloatDescriptorsGpu(
                device,
                handles[first].descriptors,
                handle.descriptors,
                { ratio: 0.8, mutual: true },
              );
              if (m.matches.length >= 32)
                descriptor.push({
                  first,
                  second: frame.id,
                  matches: m.matches,
                  distances: m.distances,
                });
            }
            matchingMs += performance.now() - start;
            const merged = mergeTrackerPairs(descriptor, joined.pairs);
            kltAdded += merged.added;
            pairs.push(...merged.pairs);
          }
        } finally {
          await tracker?.dispose();
          for (const handle of handles) await handle.dispose();
        }
        const frontendMs = performance.now() - started,
          after = device.statistics;
        const row = {
          overlay: entry.overlay,
          featureCounts,
          kltAdded,
          pairCount: pairs.length,
          regions,
          frontend: {
            elapsedMs: frontendMs,
            decodeMs,
            featureMs,
            matchingMs,
            kltMs,
            uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
            readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
            peakManagedBytes: peak,
          },
          runs: [],
        };
        const input = JSON.stringify({ images, pairs }),
          digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(input));
        row.mapperInputSha256 = [...new Uint8Array(digest)]
          .map((x) => x.toString(16).padStart(2, '0'))
          .join('');
        const modes = entry.overlay ? ['baseline', 'mask', 'quality'] : ['baseline', 'quality'];
        for (let round = -1; round < repeats; round++)
          for (const mode of round % 2 === 0 ? [...modes].reverse() : modes) {
            const before = device.statistics,
              stages = [];
            peak = before.liveBytes;
            const options = {
              ransacIterations: 256,
              bundleIterations: 2,
              finalBundleIterations: 4,
              ...(mode === 'quality'
                ? { pairRefinementIterations: 10, maxRelativeDepthStdDev: 0.1 }
                : {}),
              ...(entry.overlay && mode !== 'baseline' ? { cameraObservationMasks: masks } : {}),
              onStage: (s) =>
                stages.push({
                  operation: s.operation,
                  elapsedMs: s.elapsedMs,
                  uploadBytes: s.execution.applicationUploadBytes,
                  readbackBytes: s.execution.applicationReadbackBytes,
                }),
            };
            let mapped, failure;
            const start = performance.now();
            try {
              mapped = await api.reconstructIncrementalGpu(device, images, pairs, options);
            } catch (e) {
              if (e.name !== 'ReconstructionInitializationError') throw e;
              failure = e;
            }
            const elapsedMs = performance.now() - start,
              after = device.statistics,
              tracking = mapped?.tracking ?? failure?.tracking;
            if (after.liveBytes !== after.pooledBytes)
              throw new Error('Active mapper buffers leaked');
            row.runs.push({
              round,
              warmup: round < 0,
              mode,
              mappingMs: elapsedMs,
              composedPipelineMs: elapsedMs + frontendMs,
              status: mapped ? 'map' : 'initialization-rejected',
              registered: mapped?.reconstruction.views.length ?? 0,
              points: mapped?.reconstruction.landmarks.length ?? 0,
              pending: tracking?.untriangulatedTrackIds.length ?? null,
              depthDeferred: tracking?.depthDeferredTrackIds?.length ?? null,
              separatedRows:
                (
                  mapped?.observationSelection ?? failure?.observationSelection
                )?.separatedPairs.reduce((n, p) => n + p.matches.length / 2, 0) ?? 0,
              uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
              readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
              readCalls: after.reads - before.reads,
              peakManagedBytes: peak,
              activeBytesAfter: after.liveBytes - after.pooledBytes,
              stages,
              initialPair: mapped?.diagnostics.initialPair,
              model: mapped
                ? {
                    views: mapped.reconstruction.views.map((v) => ({
                      ...v,
                      pose: { R: [...v.pose.R], t: [...v.pose.t] },
                    })),
                    landmarks: mapped.reconstruction.landmarks.map((p) => ({
                      ...p,
                      xyz: [...p.xyz],
                    })),
                  }
                : null,
            });
          }
        if (JSON.stringify({ images, pairs }) !== input)
          throw new Error('RGB mapper input mutated');
        records.push(row);
      }
    } finally {
      await device.dispose();
      gpu.destroy();
    }
    return {
      description,
      software: /swiftshader|llvmpipe|software/i.test(description),
      userAgent: navigator.userAgent,
      gpuErrors,
      records,
      liveBuffersAfterDisposal: device.statistics.liveBuffers,
    };
  },
  { cases, repeats },
  { timeoutMs: 600000, serve: { '/quality/klt-join.mjs': joinFile } },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.gpuErrors, []);
assert.equal(result.liveBuffersAfterDisposal, 0);
const median = (v) => {
  const a = [...v].sort((x, y) => x - y);
  return (a[(a.length - 1) >> 1] + a[a.length >> 1]) / 2;
};
for (const [i, row] of result.records.entries()) {
  row.scenario = truth[i].scenario;
  for (const r of row.runs) {
    r.cameraTruth = r.model
      ? compareSeedGauge({ views: truth[i].views }, r.model, r.initialPair)
      : null;
    r.falseStructure = truth[i].scenario === 'rotation' && r.points > 0;
    r.viewIds = r.model?.views.map((v) => v.id) ?? [];
  }
  for (const run of row.runs)
    if (run.mode !== 'baseline' && run.model) {
      const baseline = row.runs.find((r) => r.mode === 'baseline' && r.round === run.round);
      if (!baseline?.model) continue;
      const ids = run.viewIds.filter((id) => baseline.viewIds.includes(id));
      const reference = { views: truth[i].views };
      run.commonFrames = {
        ids,
        baseline: compareSeedGauge(
          reference,
          { ...baseline.model, views: baseline.model.views.filter((v) => ids.includes(v.id)) },
          baseline.initialPair,
        ),
        candidate: compareSeedGauge(
          reference,
          { ...run.model, views: run.model.views.filter((v) => ids.includes(v.id)) },
          run.initialPair,
        ),
        note: 'Same fixed gauge for each whole map, no refit or per-frame alignment; omitted registration remains reported separately.',
      };
    }
  row.summary = Object.fromEntries(
    [...new Set(row.runs.map((r) => r.mode))].map((mode) => {
      const runs = row.runs.filter((r) => !r.warmup && r.mode === mode),
        r = runs[0];
      return [
        mode,
        {
          registered: r.registered,
          points: r.points,
          cameraTruth: r.cameraTruth,
          commonFrames: r.commonFrames,
          pending: r.pending,
          depthDeferred: r.depthDeferred,
          separatedRows: r.separatedRows,
          medianMappingMs: median(runs.map((x) => x.mappingMs)),
          medianComposedPipelineMs: median(runs.map((x) => x.composedPipelineMs)),
          medianUploadBytes: median(runs.map((x) => x.uploadBytes)),
          medianReadbackBytes: median(runs.map((x) => x.readbackBytes)),
          medianReadCalls: median(runs.map((x) => x.readCalls)),
          peakManagedBytes: Math.max(...runs.map((x) => x.peakManagedBytes)),
          falseStructure: runs.some((x) => x.falseStructure),
          allRunsSameCounts: runs.every(
            (x) => x.registered === r.registered && x.points === r.points,
          ),
        },
      ];
    }),
  );
  for (const run of row.runs) delete run.model;
  console.log(
    JSON.stringify({ scenario: row.scenario, overlay: row.overlay, summary: row.summary }),
  );
}
const report = {
  createdAt: new Date().toISOString(),
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  workingTreeDirty: true,
  node: process.version,
  nativeGpuMeasured: false,
  protocol: {
    repeats,
    warmupsPerMode: 1,
    order: 'alternating mode order; same device, frontend results and input per mode',
    frontend:
      'Actual Lab SIFT, KLT tracker/join and unchanged mergeTrackerPairs helper. No UI or video codec test. Frontend measured once and added to each mapper sample.',
    input:
      'RGB plus known K; composited foreground case additionally supplies explicitly ground-truth region masks. No GT pose or depth enters reconstruction.',
    evaluation:
      'GT camera stays in Node; one fixed seed gauge per complete map; no point geometry accuracy claim for unassociated RGB features.',
    limits:
      'Small controlled render and composited 2D foreground/HUD; not actual gameplay, learned segmentation, native GPU, MVS or Radiance performance. Managed allocations include pool and exclude driver and host memory.',
  },
  measurements: result,
  sourceSha256: Object.fromEntries(
    await Promise.all(
      [
        'scripts/benchmark-structure-quality-rgb.mjs',
        'src/gpu/mapper.ts',
        'src/gpu/triangulation.shade.ts',
        'src/gpu/observations.ts',
        'website/src/lib/klt-join.ts',
        'dist/gpu/index.js',
      ].map(async (p) => [p, sha(await readFile(p))]),
    ),
  ),
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
