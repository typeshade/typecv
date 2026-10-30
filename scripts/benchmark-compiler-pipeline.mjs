// Paired compiler-pin comparison over identical TypeCV sources and committed RGB inputs.
// Time the complete feature -> matching -> native SfM call, including its ordinary transfers.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { arch, cpus, platform, release } from 'node:os';
import { gzipSync } from 'node:zlib';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { LAB_MAPPER } from '../src/sfm/config/lab-preset.ts';
import { DEFAULT_SEED } from '../src/geometry/operator-defaults.ts';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    output: { type: 'string' },
    features: { type: 'string', default: 'sift' },
    frames: { type: 'string', default: '0,3,7' },
    pairs: { type: 'string', default: '5' },
    warmups: { type: 'string', default: '1' },
    reverse: { type: 'boolean', default: false },
    control: { type: 'boolean', default: false },
  },
});
assert.ok(
  values.baseline && values.output,
  '--baseline <checkout> and --output <json> are required',
);
assert.ok(['sift', 'brief'].includes(values.features), '--features must be sift or brief');
const root = resolve(import.meta.dirname, '..');
const baseline = resolve(values.baseline);
const output = resolve(values.output);
const pairs = Number(values.pairs),
  warmups = Number(values.warmups);
assert.ok(Number.isSafeInteger(pairs) && pairs >= 1 && pairs <= 31);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 1 && warmups <= 5);
const indices = values.frames.split(',').map(Number);
assert.ok(indices.length >= 3 && new Set(indices).size === indices.length);
assert.ok(
  indices.every(
    (i, n) => Number.isInteger(i) && i >= 0 && i < 8 && (n === 0 || i > indices[n - 1]),
  ),
);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const scriptSha256 = hash(await readFile(new URL(import.meta.url)));
const git = (cwd, ...args) => execFileSync('git', args, { cwd, encoding: 'utf8' }).trim();
// The pin and lock are the experimental variable. Source/configuration/input changes are not.
const sourceRevision = git(root, 'rev-parse', 'HEAD');
assert.equal(git(root, 'rev-parse', 'HEAD:src'), git(baseline, 'rev-parse', 'HEAD:src'));
for (const directory of [baseline, root])
  assert.equal(git(directory, 'status', '--porcelain', '--', 'src'), '', 'dirty library source');
for (const file of ['vite.gpu.config.ts', 'scripts/build-gpu.mjs', 'tsconfig.gpu.json'])
  assert.deepEqual(await readFile(join(root, file)), await readFile(join(baseline, file)), file);
const packages = await Promise.all(
  [baseline, root].map(async (dir) =>
    JSON.parse(await readFile(join(dir, 'package.json'), 'utf8')),
  ),
);
const locks = await Promise.all(
  [baseline, root].map(async (dir) =>
    JSON.parse(await readFile(join(dir, 'package-lock.json'), 'utf8')),
  ),
);
const dependencies = (lock) =>
  Object.fromEntries(
    Object.entries(lock.packages).filter(([name]) => name && name !== 'node_modules/typeshade'),
  );
assert.deepEqual(
  dependencies(locks[0]),
  dependencies(locks[1]),
  'non-compiler dependencies differ',
);
const serve = {},
  runtimes = {};
for (const [label, dir] of Object.entries({
  baseline,
  candidate: values.control ? baseline : root,
})) {
  const files = [];
  async function walk(relative) {
    for (const entry of (await readdir(join(dir, 'dist', relative), { withFileTypes: true })).sort(
      (a, b) => a.name.localeCompare(b.name),
    )) {
      const path = relative ? `${relative}/${entry.name}` : entry.name;
      if (entry.isDirectory()) await walk(path);
      else if (path.endsWith('.js')) {
        const file = join(dir, 'dist', path);
        serve[`/${label}/${path}`] = file;
        const bytes = await readFile(file);
        files.push({
          path,
          sha256: hash(bytes),
          bytes: bytes.length,
          gzipBytes: gzipSync(bytes, { level: 9 }).length,
        });
      }
    }
  }
  await walk('');
  assert.ok(
    files.some((f) => f.path === 'gpu/index.js'),
    'build both checkouts first',
  );
  runtimes[label] = {
    compiler: packages[label === 'baseline' || values.control ? 0 : 1].devDependencies.typeshade,
    files,
    sha256: hash(JSON.stringify(files)),
  };
}
if (values.control) assert.equal(runtimes.baseline.sha256, runtimes.candidate.sha256);
else {
  assert.notEqual(
    runtimes.baseline.compiler,
    runtimes.candidate.compiler,
    'compiler pins are identical',
  );
  assert.notEqual(
    runtimes.baseline.sha256,
    runtimes.candidate.sha256,
    'built runtimes are identical',
  );
}
const frames = [];
for (const [id, index] of indices.entries()) {
  const file = `tests/fixtures/lab-images/pyramid/frame-${String(index).padStart(2, '0')}.jpg`;
  const bytes = await readFile(join(root, file));
  assert.deepEqual(bytes, await readFile(join(baseline, file)), 'fixture differs');
  serve[`/input/${id}.jpg`] = join(root, file);
  frames.push({ id, index, file, sha256: hash(bytes) });
}

const { result, errors } = await inBrowser(
  async ({ frames, pairs, warmups, reverse, featureMode, mapper }) => {
    const importStarted = performance.now();
    const apis = {
      baseline: await import('/baseline/gpu.js'),
      candidate: await import('/candidate/gpu.js'),
    };
    const importMs = performance.now() - importStarted;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU unavailable');
    const description = [
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.device,
      adapter.info.description,
    ]
      .filter(Boolean)
      .join(' ');
    if (!/swiftshader/i.test(description))
      throw new Error(`Expected SwiftShader, got ${description}`);
    const gpu = await adapter.requestDevice(),
      validation = [];
    gpu.addEventListener('uncapturederror', (event) => validation.push(event.error.message));
    const preparationStarted = performance.now();
    const canvas = document.createElement('canvas'),
      context = canvas.getContext('2d', { willReadFrequently: true });
    for (const f of frames) {
      const bitmap = await createImageBitmap(await (await fetch(`/input/${f.id}.jpg`)).blob());
      try {
        canvas.width = f.width = bitmap.width;
        canvas.height = f.height = bitmap.height;
        context.drawImage(bitmap, 0, 0);
        f.pixels = context.getImageData(0, 0, f.width, f.height).data.slice();
      } finally {
        bitmap.close();
      }
    }
    const preparationMs = performance.now() - preparationStarted;
    const digest = async (value) =>
      Array.from(
        new Uint8Array(
          await crypto.subtle.digest(
            'SHA-256',
            typeof value === 'string' ? new TextEncoder().encode(value) : value,
          ),
        ),
      )
        .map((x) => x.toString(16).padStart(2, '0'))
        .join('');
    const json = (v) =>
      JSON.stringify(v, (_key, x) => {
        if (typeof x === 'number' && !Number.isFinite(x)) throw new Error('Nonfinite output');
        if (Object.is(x, -0)) return '__negative_zero__';
        return ArrayBuffer.isView(x) ? Array.from(x) : x;
      });
    const arrayDigest = (a) => digest(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
    const devices = {},
      rows = [];
    try {
      for (const name of ['baseline', 'candidate'])
        devices[name] = await apis[name].ShadeDevice.open({
          gpuDevice: gpu,
          backend: 'webgpu',
          poolBytes: 0,
          boundedSubmissions: false,
          solverCheckpoints: true,
          maxModelsPerBatch: 32,
        });
      async function run(version, round) {
        const api = apis[version],
          device = devices[version],
          handles = [],
          images = [],
          matches = [],
          stages = [];
        let featureMs = 0,
          matchMs = 0;
        const before = device.statistics,
          started = performance.now();
        try {
          for (const frame of frames) {
            let at = performance.now();
            const h =
              featureMode === 'sift'
                ? await api.extractSiftGpuResident(
                    device,
                    frame.pixels,
                    frame.width,
                    frame.height,
                    { maxFeatures: 4096, firstOctave: -1, clahe: false },
                  )
                : await api.extractFeaturesGpuResident(
                    device,
                    frame.pixels,
                    frame.width,
                    frame.height,
                    { threshold: 0.045, maxFeatures: 400, smoothDescriptors: false },
                  );
            handles.push(h);
            const keypoints = await h.readKeypoints();
            featureMs += performance.now() - at;
            if (h.count < 8 || h.report.tier !== 'webgpu')
              throw new Error('Empty or non-GPU feature work');
            images.push({
              id: frame.id,
              keypoints,
              camera: {
                // Match the committed pyramid benchmark's initial square-pixel calibration.
                fx: frame.width * 0.8,
                fy: frame.width * 0.8,
                cx: (frame.width - 1) / 2,
                cy: (frame.height - 1) / 2,
                k1: 0,
                k2: 0,
              },
            });
            for (let previous = Math.max(0, frame.id - 3); previous < frame.id; previous++) {
              at = performance.now();
              const m =
                featureMode === 'sift'
                  ? await api.matchFloatDescriptorsGpu(
                      device,
                      handles[previous].descriptors,
                      h.descriptors,
                      { ratio: 0.8, maxDistance: 1e19 },
                    )
                  : await api.matchResidentFeaturesGpu(device, handles[previous], h, {
                      ratio: 0.82,
                      maxDistance: 80,
                    });
              matchMs += performance.now() - at;
              if (m.report.tier !== 'webgpu') throw new Error('Non-GPU matching');
              if (m.matches.length >= 16)
                matches.push({
                  first: previous,
                  second: frame.id,
                  matches: m.matches,
                  distances: m.distances,
                });
            }
          }
          matches.sort((a, b) => a.second - a.first - (b.second - b.first) || a.first - b.first);
          if (!matches.length) throw new Error('No accepted matches');
          const at = performance.now();
          const mapping = await api.reconstructIncrementalGpu(device, images, matches, mapper);
          await gpu.queue.onSubmittedWorkDone();
          const mapperMs = performance.now() - at,
            elapsedMs = performance.now() - started,
            after = device.statistics;
          for (const s of mapping.execution.stages)
            stages.push({
              operation: s.operation,
              elapsedMs: s.elapsedMs,
              tier: s.execution.report.tier,
              dispatches: s.execution.report.dispatches,
              submissions: s.execution.report.submissions,
            });
          const model = mapping.reconstruction;
          if (
            model.views.length < 2 ||
            model.landmarks.length < 8 ||
            !(mapping.execution.gpuBundleAdjustments >= 1)
          )
            throw new Error('No nontrivial reconstruction/BA work');
          // Diagnostic hashes/readbacks and resource disposal are outside the timed interval.
          const featureHashes = [];
          for (const h of handles) {
            const details = featureMode === 'sift' ? undefined : await h.read();
            featureHashes.push({
              count: h.count,
              keypoints: await arrayDigest(await h.readKeypoints()),
              ...(featureMode === 'sift'
                ? {
                    details: await arrayDigest(await h.readDetails()),
                    descriptors: await arrayDigest(
                      await device.read(h.descriptors.descriptors, h.count * 128),
                    ),
                  }
                : {
                    scores: await arrayDigest(details.scores),
                    descriptors: await arrayDigest(details.descriptors),
                  }),
            });
          }
          return {
            version,
            round,
            elapsedMs,
            featureMs,
            matchMs,
            mapperMs,
            stages,
            uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
            readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
            output: {
              featureHashes,
              matchHash: await digest(json(matches)),
              modelHash: await digest(json(model)),
              viewIds: model.views.map((v) => v.id),
              landmarks: model.landmarks.length,
              observations: model.landmarks.reduce((sum, p) => sum + p.observations.length, 0),
              meanReprojectionErrorPx: mapping.diagnostics.meanReprojectionErrorPx,
              registrationOrder: mapping.diagnostics.registrationOrder,
              initialPair: mapping.diagnostics.initialPair,
              gpuBundleAdjustments: mapping.execution.gpuBundleAdjustments,
            },
          };
        } finally {
          for (const handle of handles) await handle.dispose();
          if (device.statistics.liveBytes !== 0) throw new Error('Live arrays after completed run');
        }
      }
      for (let round = -warmups; round < pairs; round++) {
        const order =
          (round + warmups + Number(reverse)) % 2
            ? ['candidate', 'baseline']
            : ['baseline', 'candidate'];
        for (const version of order) {
          console.log(`compiler pipeline ${featureMode} round ${round} ${version}`);
          const row = await run(version, round);
          rows.push(row);
          console.log(
            `completed ${version}: ${row.elapsedMs.toFixed(1)} ms, ${row.output.viewIds.length} views, ${row.output.landmarks} landmarks`,
          );
        }
      }
      return {
        adapter: description,
        userAgent: navigator.userAgent,
        importMs,
        preparationMs,
        frames: frames.map(({ pixels: _pixels, ...f }) => f),
        rows,
        validation,
      };
    } finally {
      for (const device of Object.values(devices)) await device.dispose();
      gpu.destroy();
    }
  },
  {
    frames,
    pairs,
    warmups,
    reverse: values.reverse,
    featureMode: values.features,
    mapper: { ...LAB_MAPPER, seed: DEFAULT_SEED },
  },
  { serve, timeoutMs: 7200000 },
);

const mismatches = [];
for (let round = -warmups; round < pairs; round++) {
  const a = result.rows.find((r) => r.round === round && r.version === 'baseline');
  const b = result.rows.find((r) => r.round === round && r.version === 'candidate');
  for (const field of Object.keys(a.output))
    if (
      !Object.hasOwn(b.output, field) ||
      JSON.stringify(a.output[field]) !== JSON.stringify(b.output[field])
    )
      mismatches.push({ round, field, baseline: a.output[field], candidate: b.output[field] });
}
const median = (xs) => {
  const sorted = xs.toSorted((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
};
const warm = result.rows.filter((r) => r.round >= 0),
  summary = {};
for (const field of ['elapsedMs', 'featureMs', 'matchMs', 'mapperMs']) {
  const pairedPercent = Array.from({ length: pairs }, (_, round) => {
    const a = warm.find((r) => r.round === round && r.version === 'baseline')[field];
    const b = warm.find((r) => r.round === round && r.version === 'candidate')[field];
    assert.ok(Number.isFinite(a) && a > 0 && Number.isFinite(b) && b > 0);
    return 100 * (b / a - 1);
  });
  summary[field] = {
    baseline: median(warm.filter((r) => r.version === 'baseline').map((r) => r[field])),
    candidate: median(warm.filter((r) => r.version === 'candidate').map((r) => r[field])),
    medianPairedPercent: median(pairedPercent),
    pairedPercent,
  };
}
await mkdir(dirname(output), { recursive: true });
await writeFile(
  output,
  JSON.stringify(
    {
      schemaVersion: 1,
      createdAt: new Date().toISOString(),
      sourceRevision,
      sourceTree: git(root, 'rev-parse', 'HEAD:src'),
      scriptSha256,
      node: process.version,
      machine: {
        platform: platform(),
        release: release(),
        arch: arch(),
        cpu: cpus()[0]?.model,
        cpuQuota: await readFile('/sys/fs/cgroup/cpu.max', 'utf8').then(
          (value) => value.trim(),
          () => null,
        ),
      },
      browser: process.env.TYPESHADE_CHROMIUM ?? 'playwright default',
      controls: {
        pairs,
        warmups,
        reverse: values.reverse,
        identicalCompilerControl: values.control,
        features: values.features,
        initialFocalWidthRatio: 0.8,
        mapper: { ...LAB_MAPPER, seed: DEFAULT_SEED },
        comparison: 'exact same-run output hashes and decisions',
      },
      scope:
        'Committed RGB frames through features, matching and native sparse SfM, including normal uploads/readbacks and a completion fence. Module import/JPEG decode reported separately; diagnostic hashes and disposal excluded. Shared-session first calls are not isolated startup measurements. No whole-video, UI, dense-MVS or hardware-GPU claim.',
      runtimes,
      errors,
      ...result,
      mismatches,
      summary,
    },
    null,
    2,
  ) + '\n',
);
console.log(JSON.stringify({ output, summary, mismatches }, null, 2));
assert.deepEqual(errors, []);
assert.deepEqual(result.validation, []);
assert.deepEqual(mismatches, [], 'Pipeline output differs; receipt retained');
