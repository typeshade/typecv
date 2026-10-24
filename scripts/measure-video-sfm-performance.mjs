// Original-video RGB replay of the default SIFT/matching/native-SfM library path.
// Native extraction is timed separately; this is not a browser-worker/UI latency benchmark.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { inBrowser } from '../tests/typeshade/browser.mjs';
const args = process.argv.slice(2),
  flag = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d),
  video = resolve(flag('--video')),
  baseline = resolve(flag('--baseline')),
  out = resolve(flag('--out')),
  repetitions = Number(flag('--repetitions', '3')),
  times = flag('--times', '0,2,4').split(',').map(Number),
  hash = (v) => createHash('sha256').update(v).digest('hex');
assert.ok(Number.isInteger(repetitions) && repetitions > 0 && repetitions <= 10);
assert.ok(times.length >= 2 && times.every((t) => Number.isFinite(t) && t >= 0));
await mkdir(out, { recursive: true });
const metadata = JSON.parse(
    execFileSync('ffprobe', [
      '-v',
      'error',
      '-select_streams',
      'v:0',
      '-show_entries',
      'stream=width,height,codec_name',
      '-of',
      'json',
      video,
    ]),
  ),
  source = metadata.streams[0],
  width = 640,
  height = Math.round((source.height * width) / source.width),
  frames = [],
  serve = { '/baseline/gpu.mjs': baseline },
  extractionStarted = performance.now();
for (const [id, time] of times.entries()) {
  const file = join(out, `${id}.private.rgba`);
  execFileSync('ffmpeg', [
    '-v',
    'error',
    '-y',
    '-ss',
    String(time),
    '-i',
    video,
    '-frames:v',
    '1',
    '-vf',
    `scale=${width}:${height}:flags=bilinear`,
    '-f',
    'rawvideo',
    '-pix_fmt',
    'rgba',
    file,
  ]);
  const bytes = await readFile(file);
  assert.equal(bytes.length, width * height * 4);
  serve[`/input/${id}.rgba`] = file;
  frames.push({ id, time, width, height, sha256: hash(bytes) });
}
const extractionMs = performance.now() - extractionStarted;
const { result, errors } = await inBrowser(
  async ({ frames, repetitions }) => {
    const baseline = await import('/baseline/gpu.mjs'),
      optimized = await import('/dist/gpu.js'),
      adapter = await navigator.gpu.requestAdapter(),
      description = [adapter.info.vendor, adapter.info.architecture, adapter.info.description].join(
        ' ',
      ),
      gpu = await adapter.requestDevice(),
      validation = [];
    gpu.addEventListener('uncapturederror', (e) => validation.push(e.error.message));
    for (const f of frames)
      f.pixels = new Uint8Array(await (await fetch(`/input/${f.id}.rgba`)).arrayBuffer());
    const digest = async (v) =>
      Array.from(
        new Uint8Array(
          await crypto.subtle.digest(
            'SHA-256',
            typeof v === 'string' ? new TextEncoder().encode(v) : v,
          ),
        ),
      )
        .map((x) => x.toString(16).padStart(2, '0'))
        .join('');
    const devices = {};
    for (const [name, api] of Object.entries({ baseline, optimized }))
      devices[name] = await api.ShadeDevice.open({
        gpuDevice: gpu,
        backend: 'webgpu',
        poolBytes: 0,
        boundedSubmissions: false,
        solverCheckpoints: true,
        maxModelsPerBatch: 32,
      });
    const rows = [];
    async function run(version, round) {
      const api = version === 'baseline' ? baseline : optimized,
        device = devices[version],
        handles = [],
        images = [],
        pairs = [],
        features = [],
        stages = [],
        before = device.statistics;
      let featureMs = 0,
        matchMs = 0,
        mapperMs = 0;
      const started = performance.now();
      try {
        for (const f of frames) {
          let at = performance.now();
          const h = await api.extractSiftGpuResident(device, f.pixels, f.width, f.height, {
            maxFeatures: 4096,
            firstOctave: -1,
            clahe: false,
          });
          handles.push(h);
          const keypoints = await h.readKeypoints();
          featureMs += performance.now() - at;
          images.push({
            id: f.id,
            keypoints,
            camera: {
              fx: f.width * 0.8,
              fy: f.width * 0.8,
              cx: (f.width - 1) / 2,
              cy: (f.height - 1) / 2,
              k1: 0,
              k2: 0,
            },
          });
          features.push({ count: h.count, requested: h.requested, octaves: h.octaves });
          for (let previous = 0; previous < f.id; previous++) {
            at = performance.now();
            const m = await api.matchFloatDescriptorsGpu(
              device,
              handles[previous].descriptors,
              h.descriptors,
              { ratio: 0.8, maxDistance: 1e19 },
            );
            matchMs += performance.now() - at;
            if (m.matches.length >= 16)
              pairs.push({
                first: previous,
                second: f.id,
                matches: m.matches,
                distances: m.distances,
              });
          }
        }
        const at = performance.now(),
          mapping = await api.reconstructIncrementalGpu(device, images, pairs, {
            ransacIterations: 500,
            maxErrorPx: 3,
            minInitialAngleDeg: 1,
            minTriangulationAngleDeg: 0.5,
            bundleIterations: 4,
            finalBundleIterations: 8,
            refineSharedFocal: true,
          });
        mapperMs = performance.now() - at;
        const elapsedMs = performance.now() - started,
          after = device.statistics;
        for (const s of mapping.execution.stages)
          stages.push({
            operation: s.operation,
            elapsedMs: s.elapsedMs,
            submissions: s.execution.report.submissions,
            dispatches: s.execution.report.dispatches,
          });
        // All validation reads/hashes are outside the pipeline timing and transfer delta.
        const featureHashes = [];
        for (const h of handles)
          featureHashes.push({
            details: await digest((await h.readDetails()).buffer),
            descriptors: await digest(
              (await device.read(h.descriptors.descriptors, h.count * 128)).buffer,
            ),
          });
        const model = mapping.reconstruction,
          modelHash = await digest(
            JSON.stringify({ views: model.views, landmarks: model.landmarks }),
          ),
          matchHash = await digest(JSON.stringify(pairs));
        return {
          version,
          round,
          elapsedMs,
          featureMs,
          matchMs,
          mapperMs,
          features,
          featureHashes,
          modelHash,
          matchHash,
          registeredViews: model.views.length,
          landmarks: model.landmarks.length,
          observations: model.landmarks.reduce((s, p) => s + p.observations.length, 0),
          meanReprojectionErrorPx: mapping.diagnostics.meanReprojectionErrorPx,
          stages,
          uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
          acceptedPairs: pairs.length,
        };
      } finally {
        for (const h of handles) await h.dispose();
        if (device.statistics.liveBytes !== 0) throw new Error('Live arrays after run');
      }
    }
    try {
      for (let round = -1; round < repetitions; round++)
        for (const name of round % 2 ? ['optimized', 'baseline'] : ['baseline', 'optimized']) {
          console.log(`video SfM ${round} ${name}`);
          rows.push(await run(name, round));
        }
    } finally {
      for (const d of Object.values(devices)) await d.dispose();
      gpu.destroy();
    }
    return {
      adapter: description,
      software: /swiftshader|software|llvmpipe/i.test(description),
      rows,
      validation,
    };
  },
  { frames, repetitions },
  { serve, timeoutMs: 1200000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validation, []);
for (let round = -1; round < repetitions; round++) {
  const a = result.rows.find((r) => r.round === round && r.version === 'baseline'),
    b = result.rows.find((r) => r.round === round && r.version === 'optimized');
  for (const field of [
    'features',
    'featureHashes',
    'matchHash',
    'modelHash',
    'registeredViews',
    'landmarks',
    'observations',
    'meanReprojectionErrorPx',
  ])
    assert.deepEqual(a[field], b[field], `${field}: round ${round}`);
}
await writeFile(
  join(out, 'summary.json'),
  JSON.stringify(
    {
      createdAt: new Date().toISOString(),
      baseline: '88295b4',
      videoSha256: hash(await readFile(video)),
      nativeSource: source,
      frames,
      extractionMs,
      controls: {
        maxFeatures: 4096,
        firstOctave: -1,
        matchRatio: 0.8,
        nativeMapper: true,
        workerUiMeasured: false,
      },
      scope:
        'Fixed timestamp subset from original video; native ffmpeg extraction separately timed. Default SIFT/match/native-SfM library pipeline; no browser decode/worker/UI/whole-video speed claim. Cold first call round -1, followed by alternating warm pairs. Diagnostic hashes outside measured intervals. No source pixels/geometry committed.',
      baselineHash: hash(await readFile(baseline)),
      candidateHash: hash(await readFile(new URL('../dist/gpu/index.js', import.meta.url))),
      ...result,
    },
    null,
    2,
  ) + '\n',
);
console.log(
  JSON.stringify({
    records: result.rows.map(
      ({
        version,
        round,
        elapsedMs,
        featureMs,
        matchMs,
        mapperMs,
        registeredViews,
        landmarks,
      }) => ({
        version,
        round,
        elapsedMs,
        featureMs,
        matchMs,
        mapperMs,
        registeredViews,
        landmarks,
      }),
    ),
  }),
);
