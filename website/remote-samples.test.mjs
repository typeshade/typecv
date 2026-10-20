import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  VIDEO_SAMPLE_MAX_FRAMES,
  VIDEO_SAMPLE_MIN_FRAMES,
  videoSampleCount,
  videoSampleTimes,
} from './src/lib/sample-frames.ts';
import { catalogEntry, readManifest, withRemoteEntries } from '../scripts/lab-remote-samples.mjs';
import { listsBucket, withoutBucket } from '../scripts/ensure-samples-bucket.mjs';
import { withoutBucket as withoutNamed } from '../scripts/r2-bucket.mjs';
import { handle } from './worker.mjs';

test('a remote clip takes two frames a second, from 24 to 48, over its whole length', () => {
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
    assert.match(entry.name, new RegExp(`\\(${entry.frames} frames\\)$`));
  }
  // The standard benchmark's eight frames stay in the repository.
  assert.equal(catalog.samples.find((e) => e.id === 'pyramid').frames, 8);
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
  const env = { SAMPLES: fakeR2({ 'sample/canyon/frame-00.jpg': 'jpeg' }), ASSETS: assets };
  const hit = await handle(new Request('https://cv.example/sample/canyon/frame-00.jpg'), env);
  assert.equal(await hit.text(), 'jpeg');
  assert.equal(hit.headers.get('content-type'), 'image/jpeg');
  assert.match(hit.headers.get('cache-control'), /max-age/);
  const miss = await handle(new Request('https://cv.example/sample/pyramid/frame-00.jpg'), env);
  assert.equal(await miss.text(), 'asset');
  const unbound = await handle(new Request('https://cv.example/sample/canyon/frame-00.jpg'), {
    ASSETS: assets,
  });
  assert.equal(await unbound.text(), 'asset');
});
