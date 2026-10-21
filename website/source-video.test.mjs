import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ALL_FORMATS, EncodedPacketSink, Input, UrlSource } from 'mediabunny';
import { handle } from './worker.mjs';
import { readManifest, catalogEntry, sourceEntry } from '../scripts/lab-remote-samples.mjs';
import {
  catalogVideoId,
  runVideoSource,
  sourceUrl,
  videoSourceOf,
} from './src/lib/sample-source.ts';

const CLIP = new Uint8Array(
  readFileSync(new URL('../tests/fixtures/source-clip.mp4', import.meta.url)),
);
const URL_OF_CLIP = 'https://cv.example/sample/clip/source.mp4';

/** An R2 bucket that holds the fixture, with `head` and ranged `get` like the binding. */
function bucket() {
  const meta = {
    size: CLIP.byteLength,
    httpEtag: '"clip"',
    writeHttpMetadata: (headers) => headers.set('content-type', 'video/mp4'),
  };
  return {
    async head(key) {
      return key === 'sample/clip/source.mp4' ? meta : null;
    },
    async get(key, options) {
      if (key !== 'sample/clip/source.mp4') return null;
      const { offset, length } = options?.range ?? { offset: 0, length: CLIP.byteLength };
      return { ...meta, body: CLIP.slice(offset, offset + length) };
    },
  };
}

/** A fetch that the Worker answers, and a log of what it answered. */
function loggedFetch() {
  const log = { requests: 0, bytes: 0, statuses: [], ranges: [] };
  const fetchFn = async (input, init) => {
    const request = new Request(input, init);
    const response = await handle(request, { SAMPLES: bucket() });
    log.requests++;
    log.statuses.push(response.status);
    log.ranges.push(request.headers.get('range'));
    log.bytes += (await response.clone().arrayBuffer()).byteLength;
    return response;
  };
  return { log, fetchFn };
}

test('mediabunny reads a stored video through the Worker by byte range', async () => {
  const { log, fetchFn } = loggedFetch();
  const input = new Input({
    source: new UrlSource(URL_OF_CLIP, { fetchFn }),
    formats: ALL_FORMATS,
  });
  try {
    const track = await input.getPrimaryVideoTrack();
    assert.equal(await track.getDisplayWidth(), 320);
    assert.equal(await track.getDisplayHeight(), 180);
    assert.ok(Math.abs((await track.computeDuration()) - 8) < 0.1);
    // One packet near the end: the decoder seeks to the key frame before it.
    const sink = new EncodedPacketSink(track);
    const packet = await sink.getPacket(6.5);
    assert.ok(packet && packet.timestamp <= 6.5 && packet.timestamp > 6);
    const key = await sink.getKeyPacket(6.5);
    assert.ok(key && key.timestamp <= 6.5 && key.timestamp >= 6 - 1e-6);
  } finally {
    input.dispose();
  }
  assert.ok(log.statuses.includes(206), 'the Worker answered a range: ' + log.statuses);
  assert.ok(
    log.ranges.some((range) => range?.startsWith('bytes=')),
    'the reader asked for byte ranges',
  );
  assert.ok(log.statuses.every((status) => status === 200 || status === 206));
});

test('a source video that the bucket does not hold fails the read, so the lab uses its frames', async () => {
  const { fetchFn } = loggedFetch();
  const input = new Input({
    source: new UrlSource('https://cv.example/sample/none/source.mp4', {
      fetchFn,
      getRetryDelay: () => null,
    }),
    formats: ALL_FORMATS,
  });
  try {
    await assert.rejects(async () => {
      await input.getPrimaryVideoTrack();
    });
  } finally {
    input.dispose();
  }
});

test('the catalog records each clip source from the manifest', () => {
  const manifest = readManifest();
  const catalog = JSON.parse(readFileSync(new URL('./public/sample/index.json', import.meta.url)));
  for (const video of manifest.videos) {
    assert.ok(video.source, video.id + ' has a source record');
    assert.match(video.source.sha256, /^[0-9a-f]{64}$/);
    assert.match(video.source.videoId, /^[0-9a-f]{16}$/);
    assert.ok(Number.isInteger(video.source.bytes) && video.source.bytes > 0);
    assert.ok(video.source.width >= 1920 && video.source.height >= 1080);
    const entry = catalog.samples.find((e) => e.id === video.id);
    assert.deepEqual(entry.source, sourceEntry(video.source));
    assert.equal(entry.source.file, 'source.mp4');
    assert.equal(sourceUrl(entry), `/sample/${video.id}/source.mp4`);
    assert.equal(catalogVideoId(entry), video.source.videoId);
    assert.deepEqual(catalogEntry(video).source, entry.source);
  }
  // No two clips share a videoId, so no clip of the Chimney Rock set is a duplicate of another.
  const ids = manifest.videos.map((v) => v.source.videoId);
  assert.equal(new Set(ids).size, ids.length);
});

test('a sample takes its source video when the browser decodes it, its frames otherwise', () => {
  const stored = { dir: '/sample/x/', source: { file: 'source.mp4', videoId: 'a'.repeat(16) } };
  const plain = { dir: '/sample/y/' };
  assert.equal(videoSourceOf(stored, true), 'source');
  assert.equal(videoSourceOf(stored, false), 'frames');
  assert.equal(videoSourceOf(plain, true), undefined);
  assert.equal(sourceUrl(plain), undefined);
  assert.equal(catalogVideoId(plain), undefined);
  assert.equal(runVideoSource([]), undefined);
  assert.equal(runVideoSource([undefined, undefined]), undefined);
  assert.equal(runVideoSource(['source', 'source', undefined]), 'source');
  assert.equal(runVideoSource(['frames', 'frames']), 'frames');
  assert.equal(runVideoSource(['source', 'frames']), 'mixed');
});
