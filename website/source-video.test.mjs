import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { ALL_FORMATS, EncodedPacketSink, Input, UrlSource } from 'mediabunny';
import { legacyFrameKeys } from '../scripts/remove-lab-frame-inputs.mjs';
import { handle } from './worker.mjs';
import { readManifest, catalogEntry, sourceEntry } from '../scripts/lab-remote-samples.mjs';
import {
  catalogVideoId,
  requireSampleVideo,
  sampleAnalysisSize,
  runVideoSource,
  sourceUrl,
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

test('a source video that the bucket does not hold fails the read', async () => {
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

test('historical source labels remain readable', () => {
  assert.equal(runVideoSource([]), undefined);
  assert.equal(runVideoSource(['source', 'source']), 'source');
  assert.equal(runVideoSource(['source', 'frames']), 'mixed');
});

test('sample selection requires a decodable original and never substitutes frame files', async () => {
  const entry = {
    kind: 'video',
    dir: '/sample/x/',
    video: { duration: 8 },
    source: { file: 'source.mp4', videoId: 'a'.repeat(16), width: 3840, height: 2160 },
  };
  const signal = new AbortController().signal;
  const requests = [];
  assert.equal(
    await requireSampleVideo(entry, signal, async (url) => {
      requests.push(url);
      return true;
    }),
    '/sample/x/source.mp4',
  );
  assert.deepEqual(requests, ['/sample/x/source.mp4']);
  await assert.rejects(
    requireSampleVideo(entry, signal, async () => false),
    /Cannot read or decode/,
  );
  await assert.rejects(
    requireSampleVideo({ ...entry, source: undefined }, signal, async () => {
      throw new Error('must not fetch');
    }),
    /no source video/,
  );
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(
    requireSampleVideo(entry, abort.signal, async () => true),
    { name: 'AbortError' },
  );
  assert.deepEqual(sampleAnalysisSize(entry, 640), [640, 360]);
  assert.deepEqual(sampleAnalysisSize(entry, 1280), [1280, 720]);
  assert.deepEqual(
    sampleAnalysisSize({ ...entry, source: { width: 320, height: 180 } }, 1280),
    [320, 180],
  );
});

test('frame cleanup excludes source videos, thumbnails, result files and other objects', () => {
  const keys = legacyFrameKeys(readManifest());
  assert.ok(keys.length > 500);
  assert.equal(new Set(keys).size, keys.length);
  assert.ok(keys.every((key) => /^sample\/[a-z0-9-]+\/frame-\d+\.jpg$/.test(key)));
  assert.ok(!keys.some((key) => /source|thumbnail|results|records/.test(key)));
});

test('public samples contain no historical image sequences or photograph datasets', () => {
  const files = readdirSync(new URL('./public/sample/', import.meta.url), { recursive: true });
  assert.ok(!files.some((file) => /frame-\d+\.(?:jpg|png)$/.test(file)));
  assert.ok(
    !files.some((file) =>
      /^(?:pyramid|south-building|south-building-24|gerrard-hall|graham-hall|person-hall|tum-fr3-structure-texture-far)[/\\]/.test(
        file,
      ),
    ),
  );
});
