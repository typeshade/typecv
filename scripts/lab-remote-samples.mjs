#!/usr/bin/env node
// The lab's video samples that live in the R2 bucket, not in the repository.
// `website/samples/remote-videos.json` names each clip; `website/src/lib/sample-frames.ts`
// gives its frame times. The site's Worker serves `/sample/<id>/...` from the bucket.
//
//   node scripts/lab-remote-samples.mjs catalog   # write their entries into the catalog
//   node scripts/lab-remote-samples.mjs publish   # cut the frames and upload them (CI)
//   node scripts/lab-remote-samples.mjs publish --only canyon,water-tower
//
// `publish` needs curl, ffprobe, ffmpeg and CLOUDFLARE_API_TOKEN. It downloads one clip at a
// time into a temporary directory and deletes it after its frames are uploaded.
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { videoSampleTimes } from '../website/src/lib/sample-frames.ts';

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
  const times = videoSampleTimes(video.duration);
  return {
    id: video.id,
    kind: 'video',
    dir: `/sample/${video.id}/`,
    name: `${video.title} (${times.length} frames)`,
    frames: times.length,
    size: [WIDTH, HEIGHT],
    thumbnail: 'frame-00.jpg',
    remote: true,
    description: video.description,
    video: {
      name: `${video.title} sample`,
      times,
      duration: Math.round(video.duration * 10) / 10,
    },
  };
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
  const ids = new Set(remote.map((entry) => entry.id));
  const local = catalog.samples.filter((entry) => !ids.has(entry.id));
  const at = local.findIndex((entry) => entry.kind !== 'video');
  const samples = [...local.slice(0, at < 0 ? local.length : at), ...remote];
  if (at >= 0) samples.push(...local.slice(at));
  return { ...catalog, samples };
}

/** The README of one clip's frames, kept next to them in the bucket. */
export function readme(video, manifest) {
  return (
    `Derived preview frames from a video by ${video.creator}.\n` +
    `Source: ${video.page}\n` +
    `License: ${manifest.license} (checked ${manifest.checked}).\n` +
    `The source video is resized to ${WIDTH} × ${HEIGHT} JPEG frames, two per second, for an ` +
    `in-browser SfM demonstration.\n`
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

function publish(manifest, only) {
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
      const times = videoSampleTimes(video.duration);
      times.forEach((time, i) =>
        run('ffmpeg', [
          ...['-v', 'error', '-ss', String(time), '-i', clip, '-frames:v', '1'],
          ...['-vf', `scale=${WIDTH}:${HEIGHT}`, '-q:v', '4', '-y'],
          join(work, `frame-${String(i).padStart(2, '0')}.jpg`),
        ]),
      );
      rmSync(clip);
      writeFileSync(join(work, 'README.txt'), readme(video, manifest));
      for (const name of readdirSync(work).sort())
        upload(
          manifest.bucket,
          `sample/${video.id}/${name}`,
          join(work, name),
          name.endsWith('.jpg') ? 'image/jpeg' : 'text/plain; charset=utf-8',
        );
      console.log(`${video.id}: ${times.length} frames uploaded`);
    } finally {
      rmSync(work, { recursive: true, force: true });
    }
  }
}

function main() {
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
    publish(manifest, only);
    return;
  }
  console.error('usage: lab-remote-samples.mjs catalog | publish [--only id,id]');
  process.exitCode = 2;
}

if (import.meta.url === `file://${process.argv[1]}`) main();
