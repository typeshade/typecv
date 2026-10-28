// SIFT front end with and without CLAHE on a lab sample: features per frame, matches per pair,
// registered views. Runs in a headless browser on the WebGPU tier (SwiftShader unless
// TYPECV_HARDWARE_GPU=1). Usage:
//   node scripts/benchmark-sift-clahe.mjs --sample=pyramid --frames=8 [--edge=640] [--window=3]
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { LAB_MAPPER } from '../src/sfm/config/lab-preset.ts';

const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
};
const sample = arg('sample', 'pyramid'),
  frames = Number(arg('frames', '8')),
  edge = Number(arg('edge', '640')),
  windowSize = Number(arg('window', '3')),
  repeats = Number(arg('repeats', '2')),
  output = resolve(arg('output', `work/benchmark-results/sift-clahe-${sample}.json`));
if (!/^[a-z0-9-]+$/.test(sample)) throw new RangeError('--sample is a directory name');
const encoded = await Promise.all(
  Array.from({ length: frames }, (_, i) =>
    readFile(
      new URL(
        `../tests/fixtures/lab-images/${sample}/frame-${String(i).padStart(2, '0')}.jpg`,
        import.meta.url,
      ),
    ).then((buffer) => buffer.toString('base64')),
  ),
);

const { result, errors } = await inBrowser(
  async ({ encoded, edge, windowSize, repeats, labMapper }) => {
    const api = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const device = await api.ShadeDevice.open({
      gpuDevice: gpu,
      backend: 'webgpu',
      boundedSubmissions: false,
      solverCheckpoints: true,
      maxModelsPerBatch: 32,
      // The lab budget at 640 px with octave -1.
      workingBytes: 256 * 1024 ** 2,
      poolBytes: 128 * 1024 ** 2,
    });
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const decoded = [];
    for (const text of encoded) {
      const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
      const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
      const scale = Math.min(1, edge / Math.max(bitmap.width, bitmap.height));
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      decoded.push({
        width: canvas.width,
        height: canvas.height,
        pixels: context.getImageData(0, 0, canvas.width, canvas.height).data.slice(),
      });
      bitmap.close();
    }
    const adapterInfo = adapter.info;
    const run = async (clahe) => {
      const started = performance.now();
      const live = [],
        images = [],
        pairs = [],
        counts = [],
        pairMatches = [];
      let featureMs = 0;
      try {
        for (let i = 0; i < decoded.length; i++) {
          const frame = decoded[i],
            t0 = performance.now(),
            features = await api.extractSiftGpuResident(
              device,
              frame.pixels,
              frame.width,
              frame.height,
              {
                maxFeatures: 16384,
                firstOctave: -1,
                clahe,
              },
            );
          featureMs += performance.now() - t0;
          live.push(features);
          counts.push(features.count);
          images.push({
            id: i,
            camera: {
              fx: frame.width * 0.8,
              fy: frame.width * 0.8,
              cx: (frame.width - 1) / 2,
              cy: (frame.height - 1) / 2,
              k1: 0,
              k2: 0,
            },
            keypoints: await features.readKeypoints(),
          });
          for (let p = Math.max(0, i - windowSize); p < i; p++) {
            const match = await api.matchFloatDescriptorsGpu(
              device,
              live[p].descriptors,
              features.descriptors,
              { ratio: 0.8 },
            );
            pairMatches.push(match.matches.length / 2);
            if (match.matches.length / 2 >= 8)
              pairs.push({
                first: p,
                second: i,
                matches: match.matches,
                distances: match.distances,
              });
          }
        }
        let mapping;
        if (pairs.length)
          mapping = await api.reconstructIncrementalGpu(device, images, pairs, { ...labMapper });
        const mean = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
        return {
          clahe,
          featureCounts: counts,
          meanFeatures: mean(counts),
          meanMatchesPerPair: mean(pairMatches),
          acceptedPairs: pairs.length,
          registeredViews: mapping ? mapping.reconstruction.views.length : 0,
          landmarks: mapping ? mapping.reconstruction.landmarks.length : 0,
          meanReprojectionErrorPx: mapping ? mapping.diagnostics.meanReprojectionErrorPx : null,
          featureMs,
          totalMs: performance.now() - started,
        };
      } finally {
        for (const features of live) await features.dispose();
      }
    };
    const rows = [];
    // Order: a warm-up run (off) so that a cold shader cache costs neither, then off and on
    // alternating, `repeats` times each.
    await run(false);
    for (let r = 0; r < repeats; r++)
      for (const clahe of [false, true]) rows.push(await run(clahe));
    return {
      adapter: [adapterInfo?.vendor, adapterInfo?.architecture, adapterInfo?.description]
        .filter(Boolean)
        .join(' '),
      sizes: decoded.map((f) => `${f.width}x${f.height}`).slice(0, 1),
      rows,
    };
  },
  { encoded, edge, windowSize, repeats, labMapper: LAB_MAPPER },
  { timeoutMs: 3600000 },
);
if (errors.length) console.error(errors);
await mkdir(dirname(output), { recursive: true });
await writeFile(
  output,
  JSON.stringify({ date: new Date().toISOString(), sample, ...result }, null, 2),
);
console.log(JSON.stringify(result, null, 1));
