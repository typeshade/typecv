#!/usr/bin/env node
// The lab's video samples that live in the R2 bucket, not in the repository.
// `website/samples/remote-videos.json` names each clip; `website/src/lib/sample-frames.ts`
// gives its frame times. The site's Worker serves `/sample/<id>/...` from the bucket.
//
//   node scripts/lab-remote-samples.mjs catalog   # write their entries into the catalog
//   node scripts/lab-remote-samples.mjs publish   # upload original videos and display thumbnails (CI)
//   node scripts/lab-remote-samples.mjs publish --only canyon,water-tower
//   node scripts/lab-remote-samples.mjs publish --videos-only   # the source videos, no frames
//   node scripts/lab-remote-samples.mjs publish --thumbnails-only # display previews, no source videos
//   node scripts/lab-remote-samples.mjs publish --update        # accept a changed source file
//
// `publish` needs curl, ffprobe, ffmpeg and CLOUDFLARE_API_TOKEN. It downloads one clip at a
// time into a temporary directory and deletes it after its files are uploaded. It stores the
// clip unchanged as `sample/<id>/source.mp4` and records its size, SHA-256 and `videoId` in the
// manifest (`source`). A recorded source that differs from the download stops the run unless
// `--update` is given.
import { spawnSync } from 'node:child_process';
import {
  mkdtempSync,
  openAsBlob,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
  readdirSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { videoSampleTimes } from '../website/src/lib/sample-frames.ts';
import { videoIdOf } from '../src/reconstruction/identity.ts';

const root = new URL('..', import.meta.url).pathname;
const MANIFEST = join(root, 'website/samples/remote-videos.json');
const CATALOG = join(root, 'website/public/sample/index.json');
const WIDTH = 640;
const HEIGHT = 360;

export function readManifest(path = MANIFEST) {
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  if (manifest.format !== 'typecv-remote-videos-v1') throw new Error('Unknown manifest format.');
  // `set` and `group` are optional. A video that has a set has a group, and the reverse.
  for (const video of manifest.videos)
    if ((video.set === undefined) !== (video.group === undefined))
      throw new Error(`${video.id}: set and group go together.`);
  return manifest;
}

/** The catalog entry of one remote clip. */
export function catalogEntry(video) {
  if (!video.source) throw new Error(`${video.id}: source video metadata is required.`);
  const times = videoSampleTimes(video.duration);
  return {
    id: video.id,
    kind: 'video',
    dir: `/sample/${video.id}/`,
    name: video.title,
    frames: times.length,
    size: [WIDTH, HEIGHT],
    thumbnail: 'thumbnail.jpg',
    remote: true,
    ...(video.source ? { source: sourceEntry(video.source) } : {}),
    description: video.description,
    video: {
      name: `${video.title} sample`,
      times,
      duration: Math.round(video.duration * 10) / 10,
    },
  };
}

/** The file name of a source video inside the clip's folder in the bucket. */
export const SOURCE_FILE = 'source.mp4';

/** The catalog's `source` of a clip: what the lab needs to open the stored video. */
export function sourceEntry(source) {
  const { file, bytes, videoId, width, height, codec } = source;
  return { file: file ?? SOURCE_FILE, bytes, videoId, width, height, codec };
}

/** The unrelated sample that joins group B in the distractor set (docs/multi-video.md, 9). */
export const DISTRACTOR_ID = 'water-tower';

/**
 * The `video-set` entries: sample sets that load several remote clips at once. The first three
 * come from the `set` and `group` fields of the manifest. The distractor set is group B plus
 * `DISTRACTOR_ID`.
 */
export function setEntries(manifest) {
  const members = (pick) => manifest.videos.filter(pick).map((video) => video.id);
  const set = (id, name, ids, en, ko) => ({
    id,
    kind: 'video-set',
    name: `${name} (${ids.length} videos)`,
    members: ids,
    description: { en, ko },
  });
  const inSet = (video) => video.set === 'chimney-rock';
  const a = members((video) => inSet(video) && video.group === 'A');
  const b = members((video) => inSet(video) && video.group === 'B');
  const all = members(inSet);
  const entries = [
    set(
      'chimney-rock-a',
      'Chimney Rock, group A',
      a,
      'Videos of Chimney Rock, Colorado, group A.',
      '콜로라도 침니 록 영상, 그룹 A.',
    ),
    set(
      'chimney-rock-b',
      'Chimney Rock, group B',
      b,
      'Videos of Chimney Rock, Colorado, group B.',
      '콜로라도 침니 록 영상, 그룹 B.',
    ),
    set(
      'chimney-rock',
      'Chimney Rock, all groups',
      all,
      'All videos of Chimney Rock, Colorado.',
      '콜로라도 침니 록 영상 전체.',
    ),
    set(
      'chimney-rock-distractor',
      'Chimney Rock group B and an unrelated video',
      [...b, DISTRACTOR_ID],
      'Group B plus a video of a water tower, which shows another place.',
      '그룹 B와 다른 장소를 담은 급수탑 영상.',
    ),
  ];
  return entries;
}

/** The catalog with every remote entry replaced or added after the first video entry. */
export function withRemoteEntries(catalog, manifest) {
  const remote = [...manifest.videos.map(catalogEntry), ...setEntries(manifest)];
  // Runnable examples are source videos. Saved reconstructions remain available to inspect.
  const results = catalog.samples.filter((entry) => entry.kind === 'result');
  return { ...catalog, samples: [...remote, ...results] };
}

/** Attribution stored with each original video and its display-only thumbnail. */
export function readme(video, manifest) {
  return (
    `Video by ${video.creator}.\n` +
    `Source: ${video.page}\n` +
    `License: ${manifest.license} (checked ${manifest.checked}).\n` +
    `${SOURCE_FILE} is the source video, unchanged. thumbnail.jpg is a display-only preview. ` +
    `Analysis decodes the source video at the selected resolution; there are no frame-file inputs.\n`
  );
}

function run(command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  if (result.status !== 0)
    throw new Error(`${command} ${args.join(' ')} failed:\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}

function upload(bucket, key, file, type) {
  run('npx', [
    'wrangler',
    'r2',
    'object',
    'put',
    `${bucket}/${key}`,
    '--file',
    file,
    '--content-type',
    type,
    '--cache-control',
    'public, max-age=86400',
    '--remote',
  ]);
}

/** The size, SHA-256, `videoId` and stream facts of a downloaded clip. */
async function describeSource(clip) {
  const hash = createHash('sha256');
  hash.update(readFileSync(clip));
  const probe = JSON.parse(
    run('ffprobe', [
      ...['-v', 'error', '-select_streams', 'v:0'],
      ...['-show_entries', 'stream=codec_name,width,height', '-of', 'json', clip],
    ]),
  ).streams[0];
  return {
    file: SOURCE_FILE,
    bytes: statSync(clip).size,
    sha256: hash.digest('hex'),
    videoId: await videoIdOf(await openAsBlob(clip)),
    width: probe.width,
    height: probe.height,
    codec: probe.codec_name,
  };
}

async function publish(manifest, options) {
  const { only, thumbnails = true, videos = true, update = false } = options;
  let added = 0;
  for (const video of manifest.videos) {
    if (only && !only.has(video.id)) continue;
    const work = mkdtempSync(join(tmpdir(), `sample-${video.id}-`));
    try {
      const clip = join(work, 'clip.mp4');
      run('curl', ['-sSLf', '-A', 'Mozilla/5.0', '-o', clip, video.file]);
      const probed = Number(
        run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', clip]),
      );
      if (!(Math.abs(probed - video.duration) < 0.5))
        throw new Error(
          `${video.id}: the clip lasts ${probed} s, the manifest says ${video.duration} s.`,
        );
      const uploaded = [];
      if (videos) {
        const source = await describeSource(clip);
        const recorded = video.source;
        if (recorded && !update && recorded.sha256 !== source.sha256)
          throw new Error(
            `${video.id}: the download has SHA-256 ${source.sha256}, the manifest records ` +
              `${recorded.sha256}. Pass --update to accept the new file.`,
          );
        upload(manifest.bucket, `sample/${video.id}/${SOURCE_FILE}`, clip, 'video/mp4');
        video.source = source;
        writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
        added += source.bytes;
        uploaded.push(`${SOURCE_FILE} ${source.bytes} bytes sha256 ${source.sha256}`);
      }
      if (thumbnails) {
        run('ffmpeg', [
          ...['-v', 'error', '-i', clip, '-frames:v', '1'],
          ...['-vf', `scale=${WIDTH}:${HEIGHT}`, '-q:v', '4', '-y'],
          join(work, 'thumbnail.jpg'),
        ]);
        writeFileSync(join(work, 'README.txt'), readme(video, manifest));
        rmSync(clip);
        for (const name of readdirSync(work).sort())
          upload(
            manifest.bucket,
            `sample/${video.id}/${name}`,
            join(work, name),
            name.endsWith('.jpg') ? 'image/jpeg' : 'text/plain; charset=utf-8',
          );
        uploaded.push('preview thumbnail');
      }
      console.log(`${video.id}: ${uploaded.join(', ')}`);
    } finally {
      rmSync(work, { recursive: true, force: true });
    }
  }
  if (videos) console.log(`source videos uploaded this run: ${added} bytes`);
}

async function main() {
  const [command] = process.argv.slice(2);
  const manifest = readManifest();
  if (command === 'catalog') {
    const catalog = JSON.parse(readFileSync(CATALOG, 'utf8'));
    writeFileSync(CATALOG, JSON.stringify(withRemoteEntries(catalog, manifest), null, 2) + '\n');
    console.log(
      `${manifest.videos.length} remote entries and ${setEntries(manifest).length} sets written to the catalog`,
    );
    return;
  }
  if (command === 'publish') {
    const flag = process.argv.indexOf('--only');
    const only = flag > 0 ? new Set(process.argv[flag + 1].split(',')) : undefined;
    const has = (name) => process.argv.includes(name);
    if (has('--frames-only'))
      throw new Error('Frame inputs were removed. Use --thumbnails-only for previews.');
    if (has('--videos-only') && has('--thumbnails-only'))
      throw new Error('--videos-only and --thumbnails-only exclude each other.');
    await publish(manifest, {
      only,
      thumbnails: !has('--videos-only'),
      videos: !has('--thumbnails-only'),
      update: has('--update'),
    });
    return;
  }
  console.error(
    'usage: lab-remote-samples.mjs catalog | publish [--only id,id] [--videos-only | --thumbnails-only] [--update]',
  );
  process.exitCode = 2;
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
