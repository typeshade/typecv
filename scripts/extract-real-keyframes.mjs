// Extracts the keyframes of real sample frames for `measure-multi-video.mjs --real file.json`
// (docs/multi-video.md, Amendment 8): SIFT and the global descriptor on the device, as the lab's
// worker does, from the JPEG frames of the sample bucket.
//
//   npm run build
//   node scripts/extract-real-keyframes.mjs --out set.json --samples desert-monument,chimney-rock-32291999 \
//     [--keyframes 10] [--features 1024]
import { writeFile } from 'node:fs/promises';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const out = flag('out', ''),
  samples = flag('samples', '').split(',').filter(Boolean),
  keyframes = Number(flag('keyframes', 10)),
  features = Number(flag('features', 1024)),
  base = flag('base', 'https://cv.typeshade.dev/sample');
if (!out || !samples.length) throw new Error('--out and --samples are required');

// Fetch here: the browser page has no network access to the bucket and no CORS.
const frames = {};
for (const sample of samples) {
  frames[sample] = [];
  for (let n = 0; n < 200; n++) {
    const response = await fetch(`${base}/${sample}/frame-${String(n).padStart(2, '0')}.jpg`);
    if (!response.ok) break;
    frames[sample].push(Buffer.from(await response.arrayBuffer()).toString('base64'));
  }
  console.error(`${sample}: ${frames[sample].length} frames`);
}

const { result, errors } = await inBrowser(
  async ({ frames, keyframes, features }) => {
    const api = await import('/dist/internal/gpu.js');
    const vocabulary = api.parseVocabulary(
      new Uint8Array(await (await fetch('/vlad.bin')).arrayBuffer()),
      32,
    );
    const adapter = await navigator.gpu?.requestAdapter();
    const gpu = await adapter.requestDevice();
    const device = await api.ShadeDevice.open({ gpuDevice: gpu, solverCheckpoints: true });
    const b64 = (a) => {
      let s = '';
      const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
      for (let i = 0; i < u8.length; i += 0x8000)
        s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
      return btoa(s);
    };
    const input = [];
    try {
      for (const [sample, list] of Object.entries(frames)) {
        const picks = Array.from({ length: keyframes }, (_, i) =>
          Math.round((i * (list.length - 1)) / Math.max(1, keyframes - 1)),
        );
        for (const [i, at] of [...new Set(picks)].entries()) {
          const bytes = Uint8Array.from(atob(list[at]), (c) => c.charCodeAt(0));
          const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
          const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          ctx.drawImage(bitmap, 0, 0);
          const { data, width, height } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
          const sift = await api.extractSiftGpuResident(device, data, width, height, {
            maxFeatures: features,
            firstOctave: 0,
            clahe: false,
          });
          try {
            if (sift.count < 8) continue;
            const keypoints = await sift.readKeypoints();
            const described = await api.describeKeyframeGpu(device, sift, vocabulary);
            input.push({
              videoId: sample,
              frameKey: `${sample}:${at}`,
              time: at * 0.5,
              camera: {
                fx: 0.8 * width,
                fy: 0.8 * width,
                cx: (width - 1) / 2,
                cy: (height - 1) / 2,
                k1: 0,
                k2: 0,
              },
              keypoints: b64(keypoints),
              descriptors: b64(described.descriptors),
              global: b64(described.global),
            });
          } finally {
            await sift.dispose();
          }
        }
      }
    } finally {
      await device.dispose();
    }
    return input;
  },
  { frames, keyframes, features },
  {
    timeoutMs: 3600000,
    serve: {
      '/vlad.bin': new URL('../src/gpu/vocabulary/rootsift-32-v1.bin', import.meta.url).pathname,
    },
  },
);
if (errors.length) console.error(errors);
await writeFile(out, JSON.stringify(result));
console.error(`${result.length} keyframes written to ${out}`);
