// Cloudflare Worker in front of the static site: stores lab run records and probe records in KV.
// Every other path is served from the static assets binding.
import { RUN_RECORD_MAX_BYTES, runRecordProblem } from './src/lib/run-record.ts';
import { PROBE_RECORD_FORMAT, probeRecordProblem } from './src/lib/probe-record.ts';

const RECORD_PREFIX = 'record/';
const RATE_PREFIX = 'rate/';
const RATE_LIMIT_PER_HOUR = 30;
const LIST_LIMIT = 100;

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Same-origin writes only: a browser sends Origin on POST; a missing Origin is a non-browser client. */
function originAllowed(request, url) {
  const origin = request.headers.get('origin');
  return origin === null || origin === url.origin;
}

async function rateLimited(env, request) {
  const address = request.headers.get('cf-connecting-ip') ?? 'unknown';
  const hour = Math.floor(Date.now() / 3_600_000);
  const key = RATE_PREFIX + (await sha256Hex(address + ':' + hour)).slice(0, 32);
  const used = Number((await env.RECORDS.get(key)) ?? 0);
  if (used >= RATE_LIMIT_PER_HOUR) return true;
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
  const single = path.match(/^\/api\/records\/([^/]+)$/);
  if (single) {
    if (request.method !== 'GET') return json(405, { error: 'method' });
    return getRecord(request, env, single[1]);
  }
  if (path.startsWith('/api')) return json(404, { error: 'not found' });
  if (!env.ASSETS) return json(404, { error: 'not found' });
  return env.ASSETS.fetch(request);
}

export default { fetch: handle };
