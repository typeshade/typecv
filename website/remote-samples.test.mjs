import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  VIDEO_SAMPLE_MAX_FRAMES,
  VIDEO_SAMPLE_MIN_FRAMES,
  videoSampleCount,
  videoSampleTimes,
} from './src/lib/sample-frames.ts';
import {
  DISTRACTOR_ID,
  catalogEntry,
  readManifest,
  setEntries,
  withRemoteEntries,
} from '../scripts/lab-remote-samples.mjs';
import { listsBucket, withoutBucket } from '../scripts/ensure-samples-bucket.mjs';
import { withoutBucket as withoutNamed } from '../scripts/r2-bucket.mjs';
import { handle } from './worker.mjs';

test('historical preview timestamps remain reproducible for fixture measurements', () => {
  assert.equal(videoSampleCount(11.66), VIDEO_SAMPLE_MIN_FRAMES);
  assert.equal(videoSampleCount(19.5), 39);
  assert.equal(videoSampleCount(29.73), VIDEO_SAMPLE_MAX_FRAMES);
  const times = videoSampleTimes(19.5);
  assert.equal(times[0], 0);
  assert.equal(times.at(-1), 19.25);
  const gaps = times.slice(1).map((t, i) => t - times[i]);
  assert.ok(Math.max(...gaps) - Math.min(...gaps) < 0.02, 'evenly spaced');
  assert.throws(() => videoSampleTimes(0), RangeError);
});

test('the catalog lists every remote clip exactly as the publish script cuts it', () => {
  const manifest = readManifest();
  const catalog = JSON.parse(readFileSync(new URL('./public/sample/index.json', import.meta.url)));
  assert.deepEqual(
    withRemoteEntries(catalog, manifest),
    catalog,
    'run `lab-remote-samples.mjs catalog`',
  );
  for (const video of manifest.videos) {
    const entry = catalog.samples.find((e) => e.id === video.id);
    assert.deepEqual(entry, catalogEntry(video));
    assert.equal(entry.video.times.length, entry.frames);
    assert.equal(entry.name, video.title);
  }
  assert.ok(
    catalog.samples.every(
      (e) => e.kind === 'video-set' || e.kind === 'result' || (e.kind === 'video' && e.source),
    ),
  );
  assert.ok(!catalog.samples.some((e) => e.id === 'pyramid' || e.kind === 'photos'));
  assert.ok(
    catalog.samples.filter((e) => e.kind === 'video').every((e) => e.thumbnail === 'thumbnail.jpg'),
  );
});

test('the Chimney Rock test set agrees across the manifest, the catalog and the sets', () => {
  const manifest = readManifest();
  const catalog = JSON.parse(readFileSync(new URL('./public/sample/index.json', import.meta.url)));
  // docs/multi-video.md, section 9: group A is 4 clips, group B is 11, the ids are Pexels ids.
  const pexels = (v) => v.page.match(/-(\d+)\/$/)?.[1];
  const group = (g) => manifest.videos.filter((v) => v.set === 'chimney-rock' && v.group === g);
  assert.deepEqual(group('A').map(pexels).sort(), ['32291999', '32292000', '32292001', '32292002']);
  assert.deepEqual(
    group('B').map(pexels).sort(),
    Array.from({ length: 11 }, (_, i) => String(32332978 + i)),
  );
  assert.equal(group('A').filter((v) => v.id === 'desert-monument').length, 1);
  assert.equal(group('B').filter((v) => v.id === 'colorado-rock').length, 1);
  for (const v of group('A').concat(group('B'))) {
    assert.equal(v.creator, 'EJ Merl');
    assert.match(v.file, new RegExp(`^https://videos\\.pexels\\.com/video-files/${pexels(v)}/`));
    assert.ok(v.duration > 0);
  }
  assert.equal(new Set(manifest.videos.map((v) => v.id)).size, manifest.videos.length);
  const sets = Object.fromEntries(setEntries(manifest).map((e) => [e.id, e]));
  assert.deepEqual(Object.keys(sets), [
    'chimney-rock-a',
    'chimney-rock-b',
    'chimney-rock',
    'chimney-rock-distractor',
  ]);
  assert.equal(sets['chimney-rock-a'].members.length, 4);
  assert.equal(sets['chimney-rock-b'].members.length, 11);
  assert.equal(sets['chimney-rock'].members.length, 15);
  assert.deepEqual(sets['chimney-rock-distractor'].members, [
    ...sets['chimney-rock-b'].members,
    DISTRACTOR_ID,
  ]);
  const samples = new Map(catalog.samples.map((e) => [e.id, e]));
  for (const set of Object.values(sets)) {
    assert.equal(samples.get(set.id).kind, 'video-set');
    for (const member of set.members) assert.equal(samples.get(member)?.kind, 'video', member);
  }
});

test('a manifest entry has a set and a group, or neither', () => {
  const dir = mkdtempSync(join(tmpdir(), 'manifest-'));
  try {
    const file = join(dir, 'm.json');
    const manifest = { format: 'typecv-remote-videos-v1', videos: [{ id: 'x', set: 's' }] };
    writeFileSync(file, JSON.stringify(manifest));
    assert.throws(() => readManifest(file), /set and group go together/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the deploy keeps the R2 binding only when the bucket is there', () => {
  assert.ok(listsBucket('Listing buckets...\nname:           typecv-samples\ncreation_date:  x\n'));
  assert.ok(!listsBucket('name: typecv-samples-old\n'));
  const config = readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8');
  assert.match(config, /"r2_buckets"/);
  const without = withoutBucket(config);
  assert.doesNotMatch(without, /typecv-samples/);
  assert.match(without, /typecv-records/);
  assert.match(without, /"kv_namespaces"/);
  const bare = withoutNamed(without, 'typecv-records');
  assert.doesNotMatch(bare, /r2_buckets/);
  assert.match(bare, /"kv_namespaces"/);
});

/** An in-memory R2 bucket with the subset of the binding the Worker uses. */
function fakeR2(objects) {
  return {
    async get(key) {
      if (!(key in objects)) return null;
      return {
        body: objects[key],
        httpEtag: '"e"',
        writeHttpMetadata(headers) {
          headers.set('content-type', 'image/jpeg');
        },
      };
    },
  };
}

test('the Worker serves /sample/ from R2 and falls back to the static assets', async () => {
  const assets = { fetch: async () => new Response('asset', { status: 200 }) };
  const env = { SAMPLES: fakeR2({ 'sample/canyon/thumbnail.jpg': 'jpeg' }), ASSETS: assets };
  const hit = await handle(new Request('https://cv.example/sample/canyon/thumbnail.jpg'), env);
  assert.equal(await hit.text(), 'jpeg');
  assert.equal(hit.headers.get('content-type'), 'image/jpeg');
  assert.match(hit.headers.get('cache-control'), /max-age/);
  const miss = await handle(new Request('https://cv.example/sample/pyramid/thumbnail.jpg'), env);
  assert.equal(await miss.text(), 'asset');
  const unbound = await handle(new Request('https://cv.example/sample/canyon/thumbnail.jpg'), {
    ASSETS: assets,
  });
  assert.equal(await unbound.text(), 'asset');
});

test('historical frame inputs are unavailable even if R2 still holds them', async () => {
  const env = {
    SAMPLES: fakeR2({ 'sample/canyon/frame-00.jpg': 'old frame' }),
    ASSETS: { fetch: async () => new Response('old static frame') },
  };
  const response = await handle(new Request('https://cv.example/sample/canyon/frame-00.jpg'), env);
  assert.equal(response.status, 404);
});
