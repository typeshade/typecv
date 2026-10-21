// Cloudflare Worker in front of the static site: stores lab run records and probe records in KV,
// and serves the lab's video samples from R2 under /sample/. Every other path is served from
// the static assets binding.
import {
  RESULT_FILE_MAX_BYTES,
  RESULT_FILE_NAMES,
  RUN_RECORD_MAX_BYTES,
  runRecordProblem,
} from './src/lib/run-record.ts';
import { PROBE_RECORD_FORMAT, probeRecordProblem } from './src/lib/probe-record.ts';

const RECORD_PREFIX = 'record/';
const RATE_PREFIX = 'rate/';
const RATE_LIMIT_PER_HOUR = 30;
const LIST_LIMIT = 100;
// Result files: 4 per record, so 120 per hour is 30 records. A file counts apart from a record.
const FILE_RATE_PREFIX = 'rate-files/';
const FILE_RATE_LIMIT_PER_HOUR = 120;
// A file is accepted only for a record that is at most this old.
const FILE_WINDOW_MS = 10 * 60_000;
const RECORD_ID = /^[0-9]{17}-[0-9a-f]{8}$/;
const FILE_PREFIX = 'records/';

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });

async function sha256Hex(data) {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Same-origin writes only: a browser sends Origin on POST; a missing Origin is a non-browser client. */
function originAllowed(request, url) {
  const origin = request.headers.get('origin');
  return origin === null || origin === url.origin;
}

async function rateLimited(env, request, prefix = RATE_PREFIX, limit = RATE_LIMIT_PER_HOUR) {
  const address = request.headers.get('cf-connecting-ip') ?? 'unknown';
  const hour = Math.floor(Date.now() / 3_600_000);
  const key = prefix + (await sha256Hex(address + ':' + hour)).slice(0, 32);
  const used = Number((await env.RECORDS.get(key)) ?? 0);
  if (used >= limit) return true;
  await env.RECORDS.put(key, String(used + 1), { expirationTtl: 3600 });
  return false;
}

function authorized(request, env) {
  const token = env.RECORDS_READ_TOKEN;
  if (!token) return false;
  return request.headers.get('authorization') === `Bearer ${token}`;
}

/** The longest chunk of any key in ms, rounded, or null when the record has none. */
function longestChunk(chunks) {
  const longest = Object.values(chunks ?? {}).map((rate) => rate.longestMs);
  return longest.length ? Math.round(Math.max(...longest)) : null;
}

/** The listing metadata of a run record. */
function runMetadata(record, receivedAt) {
  return {
    receivedAt,
    gpu: record.gpu.description || record.gpu.architecture || record.gpu.vendor,
    input: record.input.kind,
    frames: record.input.frames,
    registered: record.result.registeredViews,
    totalMs: Math.round(record.timing.totalMs),
    reusedDevice: record.timing.reusedDevice,
    fp64Flavor: record.options.fp64Flavor,
    // v4: the device's time budget and the longest chunk of any key; absent in a v3 record.
    budgetMs: record.timing.budgetMs ?? null,
    longestChunkMs: longestChunk(record.timing.chunks),
    // The page's longest main-thread task and frame gap in ms; absent before 2026-10-21.
    longestTaskMs: record.timing.mainThread
      ? Math.round(record.timing.mainThread.longestTaskMs)
      : null,
    longestGapMs: record.timing.mainThread
      ? Math.round(record.timing.mainThread.longestGapMs)
      : null,
    benchmark: record.benchmark?.name ?? null,
    failure: record.failure ? record.failure.kernel || record.failure.operation : null,
    // v5: the dense MVS's status and points (`docs/dense-mvs.md` 5.3); null without it.
    denseStatus: record.dense?.status ?? null,
    densePoints: record.dense?.points ?? null,
  };
}

/**
 * The listing metadata of a probe record (`website/src/lib/probe-record.ts`). KV holds at most
 * 1024 bytes of metadata per key, so the GPU string and the list of hanging tracks are bounded.
 */
function probeMetadata(record, receivedAt) {
  const adapter = record.adapter,
    hanging = record.bisection?.hangingTracks ?? [];
  return {
    receivedAt,
    kind: 'probe',
    gpu: adapter
      ? (adapter.description || adapter.architecture || adapter.vendor).slice(0, 128)
      : '',
    fp64Flavor: record.steps.find((step) => step.fp64Flavor)?.fp64Flavor ?? null,
    boundedSubmissions: record.device.boundedSubmissions,
    variants: record.variants.map((variant) => variant.outcome),
    steps: record.steps.length,
    hangingTracks: hanging.length,
    hanging: hanging.slice(0, 8).map((track) => track.index),
    aborted: record.aborted !== null,
  };
}

async function postRecord(request, url, env) {
  if (!env.RECORDS) return json(503, { error: 'collector not configured' });
  if (!originAllowed(request, url)) return json(403, { error: 'origin' });
  const length = Number(request.headers.get('content-length') ?? 0);
  if (length > RUN_RECORD_MAX_BYTES) return json(413, { error: 'too large' });
  const body = await request.text();
  if (body.length > RUN_RECORD_MAX_BYTES) return json(413, { error: 'too large' });
  let record;
  try {
    record = JSON.parse(body);
  } catch {
    return json(400, { error: 'invalid JSON' });
  }
  const probe = record?.format === PROBE_RECORD_FORMAT;
  const problem = probe ? probeRecordProblem(record) : runRecordProblem(record);
  if (problem) return json(400, { error: 'invalid record', field: problem });
  if (await rateLimited(env, request)) return json(429, { error: 'rate limit' });
  const receivedAt = new Date().toISOString();
  const id = receivedAt.replace(/[-:.TZ]/g, '') + '-' + crypto.randomUUID().slice(0, 8);
  const stored = {
    ...record,
    server: {
      id,
      receivedAt,
      country: request.cf?.country ?? null,
      colo: request.cf?.colo ?? null,
    },
  };
  await env.RECORDS.put(RECORD_PREFIX + id, JSON.stringify(stored), {
    metadata: probe ? probeMetadata(record, receivedAt) : runMetadata(record, receivedAt),
  });
  return json(201, { id, receivedAt });
}

async function listRecords(request, url, env) {
  if (!env.RECORDS) return json(503, { error: 'collector not configured' });
  if (!authorized(request, env)) return json(401, { error: 'token' });
  const limit = Math.min(LIST_LIMIT, Math.max(1, Number(url.searchParams.get('limit') ?? 50)));
  const page = await env.RECORDS.list({
    prefix: RECORD_PREFIX,
    limit,
    cursor: url.searchParams.get('cursor') ?? undefined,
  });
  return json(200, {
    records: page.keys.map((key) => ({
      id: key.name.slice(RECORD_PREFIX.length),
      ...key.metadata,
    })),
    cursor: page.list_complete ? null : page.cursor,
  });
}

async function getRecord(request, env, id) {
  if (!env.RECORDS) return json(503, { error: 'collector not configured' });
  if (!authorized(request, env)) return json(401, { error: 'token' });
  if (!/^[0-9]{17}-[0-9a-f]{8}$/.test(id)) return json(404, { error: 'not found' });
  const value = await env.RECORDS.get(RECORD_PREFIX + id);
  if (value === null) return json(404, { error: 'not found' });
  return new Response(value, {
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

/** What the page may offer: whether result files have a store (`RECORD_FILES` bound). */
function config(env) {
  return json(200, { files: Boolean(env.RECORD_FILES && env.RECORDS) });
}

/**
 * `PUT /api/records/<id>/files/<name>`: store one result file of a record. The file must be one
 * that the record declared in `files` (same name, size and SHA-256), the record must be at most
 * `FILE_WINDOW_MS` old, and a stored file is never replaced.
 */
async function putFile(request, url, env, id, name) {
  if (!env.RECORD_FILES || !env.RECORDS) return json(503, { error: 'file store not configured' });
  if (!originAllowed(request, url)) return json(403, { error: 'origin' });
  if (!RECORD_ID.test(id)) return json(404, { error: 'not found' });
  if (!RESULT_FILE_NAMES.includes(name)) return json(400, { error: 'file name' });
  const length = Number(request.headers.get('content-length') ?? 0);
  if (length > RESULT_FILE_MAX_BYTES) return json(413, { error: 'too large' });
  const value = await env.RECORDS.get(RECORD_PREFIX + id);
  if (value === null) return json(404, { error: 'not found' });
  const record = JSON.parse(value);
  const age = Date.now() - Date.parse(record.server?.receivedAt);
  if (!(age >= 0 && age <= FILE_WINDOW_MS)) return json(403, { error: 'record too old' });
  const entry = Array.isArray(record.files) ? record.files.find((f) => f.name === name) : undefined;
  if (!entry || entry.skipped) return json(400, { error: 'file not declared' });
  if (await rateLimited(env, request, FILE_RATE_PREFIX, FILE_RATE_LIMIT_PER_HOUR))
    return json(429, { error: 'rate limit' });
  const key = FILE_PREFIX + id + '/' + name;
  if (await env.RECORD_FILES.head(key)) return json(409, { error: 'already stored' });
  const body = new Uint8Array(await request.arrayBuffer());
  if (body.byteLength > RESULT_FILE_MAX_BYTES) return json(413, { error: 'too large' });
  if (body.byteLength !== entry.bytes || (await sha256Hex(body)) !== entry.sha256)
    return json(400, { error: 'file does not match the record' });
  await env.RECORD_FILES.put(key, body, {
    httpMetadata: { contentType: 'application/octet-stream' },
  });
  return json(201, { id, name, bytes: body.byteLength });
}

/** `GET /api/records/<id>/files/`: the stored files of a record. Token required. */
async function listFiles(request, env, id) {
  if (!env.RECORD_FILES) return json(503, { error: 'file store not configured' });
  if (!authorized(request, env)) return json(401, { error: 'token' });
  if (!RECORD_ID.test(id)) return json(404, { error: 'not found' });
  const page = await env.RECORD_FILES.list({ prefix: FILE_PREFIX + id + '/' });
  return json(200, {
    files: page.objects.map((object) => ({
      name: object.key.slice((FILE_PREFIX + id + '/').length),
      bytes: object.size,
    })),
  });
}

/** `GET /api/records/<id>/files/<name>`: the bytes of one stored file. Token required. */
async function getFile(request, env, id, name) {
  if (!env.RECORD_FILES) return json(503, { error: 'file store not configured' });
  if (!authorized(request, env)) return json(401, { error: 'token' });
  if (!RECORD_ID.test(id) || !RESULT_FILE_NAMES.includes(name))
    return json(404, { error: 'not found' });
  const object = await env.RECORD_FILES.get(FILE_PREFIX + id + '/' + name);
  if (!object) return json(404, { error: 'not found' });
  return new Response(object.body, {
    headers: {
      'content-type': 'application/octet-stream',
      'content-disposition': 'attachment; filename="' + name + '"',
      'cache-control': 'no-store',
    },
  });
}

/**
 * The byte range a `Range` header asks of an object of `size` bytes: `{ offset, length }` for a
 * satisfiable single range, `'unsatisfiable'` for a range outside the object, and undefined when
 * the header is absent or is not a single `bytes=` range (the Worker then answers 200 with the
 * whole object, which RFC 9110 allows). A range that ends past the object is cut to its end.
 */
export function parseRange(header, size) {
  if (header === null || header === undefined) return undefined;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || (match[1] === '' && match[2] === '')) return undefined;
  if (match[1] === '') {
    // A suffix: the last N bytes.
    const suffix = Number(match[2]);
    if (suffix === 0 || size === 0) return 'unsatisfiable';
    const length = Math.min(suffix, size);
    return { offset: size - length, length };
  }
  const first = Number(match[1]);
  if (first >= size) return 'unsatisfiable';
  const last = match[2] === '' ? size - 1 : Math.min(Number(match[2]), size - 1);
  if (last < first) return undefined;
  return { offset: first, length: last - first + 1 };
}

/**
 * A lab sample file from the R2 bucket (`website/samples/remote-videos.json`), or undefined when
 * the bucket does not hold it: then the static assets answer, so the samples kept in the
 * repository keep working and a sample not published yet is a 404 from the assets.
 *
 * The route answers a single-range `Range` request with 206 and `Content-Range`, and 416 for a
 * range outside the object, so a browser reads a source video by byte ranges. Every answer
 * carries `Accept-Ranges: bytes`. HEAD answers the headers of GET.
 */
async function sampleObject(env, request, path) {
  const key = path.slice(1);
  if (key.includes('..')) return undefined;
  const ranged = request.method === 'GET' && request.headers.has('range');
  const head = request.method === 'HEAD' || ranged;
  const meta = head ? await env.SAMPLES.head(key) : undefined;
  if (head && !meta) return undefined;
  const range = ranged ? parseRange(request.headers.get('range'), meta.size) : undefined;
  const headers = new Headers();
  const finish = (object) => {
    object.writeHttpMetadata(headers);
    headers.set('etag', object.httpEtag);
    headers.set('accept-ranges', 'bytes');
    if (!headers.has('cache-control')) headers.set('cache-control', 'public, max-age=86400');
  };
  if (range === 'unsatisfiable') {
    finish(meta);
    headers.set('content-range', `bytes */${meta.size}`);
    return new Response(null, { status: 416, headers });
  }
  if (request.method === 'HEAD') {
    finish(meta);
    headers.set('content-length', String(meta.size));
    return new Response(null, { status: 200, headers });
  }
  const object = await env.SAMPLES.get(key, range ? { range } : undefined);
  if (!object) return undefined;
  finish(object);
  if (!range) return new Response(object.body, { headers });
  const end = range.offset + range.length - 1;
  headers.set('content-range', `bytes ${range.offset}-${end}/${meta.size}`);
  headers.set('content-length', String(range.length));
  return new Response(object.body, { status: 206, headers });
}

/** Route one request. Exported for tests; `fetch` below is the Worker entry. */
export async function handle(request, env) {
  const url = new URL(request.url);
  // The host redirects every path without a trailing slash to the path with one (a 301 the
  // browser follows as a GET), so the page posts to `/api/records/`; both spellings route alike.
  const path = url.pathname.replace(/\/+$/, '') || '/';
  if (path === '/api/records') {
    if (request.method === 'POST') return postRecord(request, url, env);
    if (request.method === 'GET') return listRecords(request, url, env);
    return json(405, { error: 'method' });
  }
  if (path === '/api/config')
    return request.method === 'GET' ? config(env) : json(405, { error: 'method' });
  const fileRoute = path.match(/^\/api\/records\/([^/]+)\/files(?:\/([^/]+))?$/);
  if (fileRoute) {
    const [, id, name] = fileRoute;
    if (name === undefined && request.method === 'GET') return listFiles(request, env, id);
    if (name !== undefined && request.method === 'GET') return getFile(request, env, id, name);
    if (name !== undefined && request.method === 'PUT') return putFile(request, url, env, id, name);
    return json(405, { error: 'method' });
  }
  const single = path.match(/^\/api\/records\/([^/]+)$/);
  if (single) {
    if (request.method !== 'GET') return json(405, { error: 'method' });
    return getRecord(request, env, single[1]);
  }
  if (path.startsWith('/api')) return json(404, { error: 'not found' });
  if (
    path.startsWith('/sample/') &&
    env.SAMPLES &&
    (request.method === 'GET' || request.method === 'HEAD')
  ) {
    const sample = await sampleObject(env, request, path);
    if (sample) return sample;
  }
  if (!env.ASSETS) return json(404, { error: 'not found' });
  return env.ASSETS.fetch(request);
}

export default { fetch: handle };
