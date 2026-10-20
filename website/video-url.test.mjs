import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_VIDEO_URL_BYTES,
  checkVideoResponse,
  fetchVideoFile,
  formatVideoUrlProgress,
  parseVideoUrl,
  recordedVideoUrl,
  videoUrlFileName,
} from './src/lib/video-url.ts';

const headers = (values) => ({ get: (name) => values[name.toLowerCase()] ?? null });
const code = (fn) => {
  try {
    fn();
  } catch (error) {
    return error.code;
  }
  return 'none';
};

test('only http and https URLs parse', () => {
  assert.equal(parseVideoUrl(' https://example.com/a/b.mp4?token=1 ').pathname, '/a/b.mp4');
  assert.equal(
    code(() => parseVideoUrl('')),
    'empty',
  );
  assert.equal(
    code(() => parseVideoUrl('not a url')),
    'invalid',
  );
  assert.equal(
    code(() => parseVideoUrl('file:///etc/passwd')),
    'scheme',
  );
  assert.equal(
    code(() => parseVideoUrl('javascript:alert(1)')),
    'scheme',
  );
  assert.equal(
    code(() => parseVideoUrl('ftp://example.com/a.mp4')),
    'scheme',
  );
});

test('a watch page URL is refused as a page', () => {
  assert.equal(
    code(() => parseVideoUrl('https://www.youtube.com/watch?v=abc')),
    'page',
  );
  assert.equal(
    code(() => parseVideoUrl('https://youtu.be/abc')),
    'page',
  );
  assert.equal(
    code(() => parseVideoUrl('https://vimeo.com/123')),
    'page',
  );
  assert.equal(parseVideoUrl('https://notyoutube.com/a.mp4').host, 'notyoutube.com');
  assert.match(
    (() => {
      try {
        parseVideoUrl('https://youtu.be/x', true);
      } catch (e) {
        return e.message;
      }
    })(),
    /웹 페이지/,
  );
});

test('response checks: status, content type and size', () => {
  const ok = (h) => ({ ok: true, status: 200, headers: headers(h) });
  assert.deepEqual(
    checkVideoResponse(ok({ 'content-type': 'video/mp4', 'content-length': '10' })),
    {
      total: 10,
      type: 'video/mp4',
    },
  );
  assert.equal(
    checkVideoResponse(ok({ 'content-type': 'application/octet-stream' })).total,
    undefined,
  );
  assert.equal(
    code(() => checkVideoResponse({ ok: false, status: 404, headers: headers({}) })),
    'status',
  );
  assert.equal(
    code(() => checkVideoResponse(ok({ 'content-type': 'text/html; charset=utf-8' }))),
    'page',
  );
  assert.equal(
    code(() => checkVideoResponse(ok({ 'content-type': 'image/png' }))),
    'type',
  );
  assert.equal(
    code(() =>
      checkVideoResponse(
        ok({ 'content-type': 'video/mp4', 'content-length': String(MAX_VIDEO_URL_BYTES + 1) }),
      ),
    ),
    'size',
  );
});

test('the record keeps the origin and path, never the query or credentials', () => {
  const url = parseVideoUrl('https://user:pw@example.com:8080/v/clip.mp4?token=secret#t=3');
  assert.equal(recordedVideoUrl(url), 'https://example.com:8080/v/clip.mp4');
  assert.equal(videoUrlFileName(url), 'clip.mp4');
  assert.equal(videoUrlFileName(new URL('https://example.com/')), 'video');
  assert.ok(recordedVideoUrl(new URL('https://example.com/' + 'a'.repeat(500))).length <= 128);
});

test('progress text has bytes and a percent when the total is known', () => {
  assert.equal(
    formatVideoUrlProgress({ loaded: 512, total: undefined }, false),
    'Downloading: 512 B',
  );
  assert.equal(
    formatVideoUrlProgress({ loaded: 1024 * 1024, total: 4 * 1024 * 1024 }, false),
    'Downloading: 1.0 MiB / 4.0 MiB (25%)',
  );
});

test('fetchVideoFile reports progress and maps a network failure to the CORS message', async () => {
  const body = new Uint8Array(10).fill(7);
  const seen = [];
  const { file, url } = await fetchVideoFile('https://example.com/c.webm?x=1', {
    fetchImpl: async () =>
      new Response(body, { headers: { 'content-type': 'video/webm', 'content-length': '10' } }),
    onProgress: (p) => seen.push(p.loaded),
  });
  assert.equal(file.name, 'c.webm');
  assert.equal(file.size, 10);
  assert.equal(file.type, 'video/webm');
  assert.equal(url.search, '?x=1');
  assert.equal(seen.at(-1), 10);
  await assert.rejects(
    fetchVideoFile('https://example.com/c.webm', {
      fetchImpl: async () => {
        throw new TypeError('Failed to fetch');
      },
    }),
    (error) => error.code === 'cors' && /Access-Control-Allow-Origin/.test(error.message),
  );
});
