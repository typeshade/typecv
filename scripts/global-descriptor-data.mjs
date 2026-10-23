/**
 * The data stage of the global descriptor measurement (docs/multi-video.md, 3.2 and step 3):
 * the frames, their RootSIFT features and the lists of frames that the later stages use.
 *
 *   node scripts/global-descriptor-data.mjs download <work-dir>
 *   node scripts/global-descriptor-data.mjs extract <work-dir> [--shard=i/n] [--training]
 *
 * `download` fetches the sample frames that the catalog (`website/public/sample/index.json`) lists
 * from original videos at https://cv.typeshade.dev/sample/<id>/source.mp4 with curl and ffmpeg. `extract` runs the SIFT of the
 * CPU tier (`extractSiftGpuResident`, RootSIFT, at most 4096 features) on each frame and writes
 * `<work-dir>/features/<id>/frame-NN.sift`: a u32 count, then 2 f32 per keypoint (x, y), then
 * 128 f32 per descriptor. It needs `npm run build`, curl and ffmpeg (JPEG to RGBA).
 *
 * Which frames: every photo of the photo samples, every third frame (`TEST_STRIDE`) of the 15
 * videos of the Chimney Rock test set (`set` in `website/samples/remote-videos.json`), and every
 * fourth frame (`SAMPLE_STRIDE`) of the other video samples. The strides keep the exhaustive
 * matching of the recall measurement to some thousand pairs and the SIFT of the CPU tier to
 * about an hour.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
export const SAMPLE_BASE = 'https://cv.typeshade.dev/sample';
export const TEST_STRIDE = 3;
export const SAMPLE_STRIDE = 4;
/** The focal length of an uploaded video's frame, as a multiple of its width (the lab's default). */
export const VIDEO_FOCAL = 0.8;

/** Every frame of every video and photo sample: `{ id, kind, index, width, height, set, group, camera }`. */
export function listFrames() {
  const catalog = JSON.parse(readFileSync(join(root, 'website/public/sample/index.json'), 'utf8'));
  const manifest = JSON.parse(
    readFileSync(join(root, 'website/samples/remote-videos.json'), 'utf8'),
  );
  const labels = new Map(manifest.videos.map((v) => [v.id, v]));
  const frames = [];
  for (const entry of catalog.samples) {
    if (entry.kind !== 'video' && entry.kind !== 'photos') continue;
    const count = Array.isArray(entry.frames) ? entry.frames.length : entry.frames;
    const label = labels.get(entry.id);
    const testSet = label?.set !== undefined;
    let cameras;
    const local = join(root, 'tests/fixtures/lab-images', entry.id, 'sample.json');
    if (existsSync(local)) cameras = JSON.parse(readFileSync(local, 'utf8')).frames;
    for (let index = 0; index < count; index++) {
      if (entry.kind === 'video' && index % (testSet ? TEST_STRIDE : SAMPLE_STRIDE)) continue;
      const [width, height] = entry.size;
      frames.push({
        id: entry.id,
        kind: entry.kind,
        index,
        time: entry.video?.times?.[index],
        width,
        height,
        set: label?.set,
        group: label?.group,
        camera: cameras?.[index]?.camera ?? videoCamera(width, height),
      });
    }
  }
  return frames;
}

export function videoCamera(width, height) {
  return {
    fx: VIDEO_FOCAL * width,
    fy: VIDEO_FOCAL * width,
    cx: (width - 1) / 2,
    cy: (height - 1) / 2,
    k1: 0,
    k2: 0,
  };
}

export const frameName = (f) => `${f.id}/frame-${String(f.index).padStart(2, '0')}`;

/** Read a `.sift` file: `{ count, points: Float32Array(2n), descriptors: Float32Array(128n) }`. */
export function readSift(path) {
  const bytes = readFileSync(path);
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const count = new Uint32Array(copy, 0, 1)[0];
  return {
    count,
    points: new Float32Array(copy, 4, 2 * count),
    descriptors: new Float32Array(copy, 4 + 8 * count, 128 * count),
  };
}

function download(work) {
  const sources = new Map();
  for (const f of listFrames()) {
    const out = join(work, 'frames', `${frameName(f)}.jpg`);
    if (existsSync(out)) continue;
    mkdirSync(dirname(out), { recursive: true });
    let clip = sources.get(f.id);
    if (!clip) {
      clip = join(work, 'sources', `${f.id}.mp4`);
      mkdirSync(dirname(clip), { recursive: true });
      if (!existsSync(clip)) {
        const result = spawnSync(
          'curl',
          ['-sSLf', `${SAMPLE_BASE}/${f.id}/source.mp4`, '-o', clip],
          { stdio: 'inherit' },
        );
        if (result.status) throw new Error(`Could not download the original video ${f.id}.`);
      }
      sources.set(f.id, clip);
    }
    const result = spawnSync(
      'ffmpeg',
      [
        '-v',
        'error',
        '-ss',
        String(f.time),
        '-i',
        clip,
        '-frames:v',
        '1',
        '-vf',
        `scale=${f.width}:${f.height}`,
        '-q:v',
        '2',
        '-y',
        out,
      ],
      { stdio: 'inherit' },
    );
    if (result.status) throw new Error(`Could not decode ${frameName(f)} from its original video.`);
  }
  console.log(`Original videos decoded for ${sources.size} samples.`);
}

async function extract(work, shard, trainingOnly) {
  const { ShadeDevice, extractSiftGpuResident } = await import(join(root, 'dist/gpu.js'));
  const [index, total] = shard.split('/').map(Number);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  // `--training` skips the 15 videos of the test set: the vocabulary and the photo recall need no more.
  const frames = listFrames().filter((f) => !trainingOnly || f.set === undefined);
  let done = 0;
  try {
    for (let i = index; i < frames.length; i += total) {
      const f = frames[i];
      const out = join(work, 'features', `${frameName(f)}.sift`);
      if (existsSync(out)) continue;
      const decoded = spawnSync(
        'ffmpeg',
        [
          '-v',
          'error',
          '-i',
          join(work, 'frames', `${frameName(f)}.jpg`),
          '-f',
          'rawvideo',
          '-pix_fmt',
          'rgba',
          '-',
        ],
        { maxBuffer: 1 << 26 },
      );
      if (decoded.status) throw new Error(`ffmpeg failed on ${frameName(f)}`);
      const rgba = new Uint8Array(decoded.stdout);
      if (rgba.length !== f.width * f.height * 4)
        throw new Error(`${frameName(f)}: unexpected size`);
      const sift = await extractSiftGpuResident(device, rgba, f.width, f.height, {
        maxFeatures: 4096,
      });
      const points = await sift.readKeypoints();
      const descriptors = await device.enqueue(() =>
        device.read(sift.descriptors.descriptors, sift.count * 128),
      );
      const buffer = new ArrayBuffer(4 + 8 * sift.count + 512 * sift.count);
      new Uint32Array(buffer, 0, 1)[0] = sift.count;
      const kp = new Float32Array(buffer, 4, 2 * sift.count);
      for (let k = 0; k < 2 * sift.count; k++) kp[k] = points[k];
      new Float32Array(buffer, 4 + 8 * sift.count, 128 * sift.count).set(
        descriptors.subarray(0, 128 * sift.count),
      );
      mkdirSync(dirname(out), { recursive: true });
      writeFileSync(out, new Uint8Array(buffer));
      await sift.dispose();
      done++;
      if (done % 10 === 0) console.log(`shard ${shard}: ${done} frames`);
    }
  } finally {
    await device.dispose();
  }
  console.log(`shard ${shard}: done, ${done} new frames`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [command, work, ...rest] = process.argv.slice(2);
  if (!work)
    throw new Error('Usage: node scripts/global-descriptor-data.mjs download|extract <work-dir>');
  if (command === 'download') download(resolve(work));
  else if (command === 'extract') {
    const shard = rest.find((a) => a.startsWith('--shard='))?.slice(8) ?? '0/1';
    await extract(resolve(work), shard, rest.includes('--training'));
  } else throw new Error(`Unknown command ${command}`);
}
