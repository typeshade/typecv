import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash, webcrypto } from 'node:crypto';
import vm from 'node:vm';
import { sampleRecord } from './record-fixture.mjs';

const source = readFileSync(new URL('../scripts/typecv-send-backup.js', import.meta.url), 'utf8');
const globals = {
  Blob,
  Map,
  Set,
  URL,
  TextEncoder,
  Uint8Array,
  AbortController,
  crypto: webcrypto,
};
const api = vm.runInNewContext(source, globals, { timeout: 1000 });
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const origin = 'https://cv.typeshade.dev';
const now = Date.parse('2026-10-25T01:00:00Z');

function fixture() {
  const data = new Map(),
    rows = [];
  for (let i = 0; i < 47; i++) {
    const name = `typecv-depths-c000-${String(i).padStart(4, '0')}.tar.gz`;
    const bytes = new TextEncoder().encode('data ' + name);
    data.set(name, bytes);
    rows.push({
      name,
      expectedBytes: bytes.length,
      bytesWritten: bytes.length,
      status: 'saved',
      verification: 'stream-completed-and-byte-count-matched',
    });
  }
  for (const name of [
    'typecv-depths-c001-0000.tar.gz',
    'typecv-depths-c000.json',
    'typecv-depths-c001.json',
  ]) {
    const bytes = new TextEncoder().encode('content ' + name);
    data.set(name, bytes);
    rows.push({
      name,
      expectedBytes: bytes.length,
      bytesWritten: bytes.length,
      status: 'saved',
      verification: 'stream-completed-and-byte-count-matched',
    });
  }
  rows.push(
    ...['typecv-depths-c000-0047.tar.gz', 'typecv-depths-c001-0001.tar.gz'].map((name) => ({
      name,
      expectedBytes: 100,
      status: 'failed',
      error: 'NotReadableError',
    })),
  );
  const manifest = {
    format: 'typecv-depth-recovery-v1',
    createdAt: '2026-10-25T00:00:00Z',
    retainedFiles: rows.length,
    expectedBytes: rows.reduce((n, row) => n + row.expectedBytes, 0),
    allRetainedFilesSaved: false,
    sourceArchivesComplete: false,
    archiveScope: {
      known: true,
      expectedComponentIds: [0, 1],
      missingComponentManifests: [],
      unexpectedComponentManifests: [],
    },
    sourceArchives: [0, 1].map((id) => ({
      name: `typecv-depths-c00${id}.json`,
      captureComplete: false,
      captureOutcome: 'completed',
      captureError: 'TypeError: Failed to fetch',
      missingDuringCapture: id ? 357 : 714,
      missingRetainedParts: [],
    })),
    files: rows,
    note: 'Only saved entries are verified copies; original capture was incomplete.',
  };
  let reads = 0,
    active = 0,
    peak = 0;
  const outputs = new Map();
  const directory = {
    async getFileHandle(name, options) {
      if (name === 'typecv-depth-recovery.json')
        return { getFile: async () => new Blob([JSON.stringify(manifest)]) };
      if (options?.create)
        return {
          createWritable: async () => ({
            write: async (value) => outputs.set(name, value),
            close: async () => {},
          }),
        };
      if (!data.has(name)) throw Error('Missing file ' + name);
      return {
        async getFile() {
          const bytes = data.get(name);
          return {
            size: bytes.length,
            async arrayBuffer() {
              reads++;
              active++;
              peak = Math.max(peak, active);
              await new Promise((resolve) => setImmediate(resolve));
              active--;
              return bytes.slice().buffer;
            },
          };
        },
      };
    },
  };
  return { manifest, data, directory, outputs, stats: () => ({ reads, peak }) };
}

function legacyCollector({ fail } = {}) {
  let posts = 0,
    active = 0,
    peak = 0;
  const calls = [],
    stored = new Map(),
    records = new Map();
  const fetcher = async (url, init) => {
    calls.push({ url, method: init.method, bodyIsBytes: init.body instanceof Uint8Array });
    assert.equal(new URL(url).origin, origin);
    assert.equal(init.mode, 'same-origin');
    assert.equal(init.redirect, 'error');
    assert.equal(init.credentials, 'same-origin');
    if (init.method === 'POST') {
      const record = JSON.parse(init.body);
      assert.ok(Buffer.byteLength(init.body) <= 256 * 1024);
      for (const file of record.files) {
        assert.match(file.name, /^typecv-depths-c[0-9]{3}(?:-[0-9]{4}\.tar\.gz|\.json)$/);
        assert.ok(Number.isSafeInteger(file.bytes) && file.bytes <= 50 * 1024 ** 2);
        assert.match(file.sha256, /^[0-9a-f]{64}$/);
        assert.equal(file.skipped, undefined);
        assert.equal(file.part, undefined);
      }
      const id = '20261025010000000-' + String(++posts).padStart(8, '0');
      records.set(id, record);
      return new Response(JSON.stringify({ id, receivedAt: new Date(now).toISOString() }), {
        status: 201,
      });
    }
    assert.equal(init.headers['content-type'], 'application/octet-stream');
    assert.equal(init.body instanceof Uint8Array, true);
    active++;
    peak = Math.max(peak, active);
    try {
      await new Promise((resolve) => setImmediate(resolve));
      const [, id, name] = /\/api\/records\/([^/]+)\/files\/([^/]+)$/.exec(new URL(url).pathname);
      const entry = records.get(id).files.find((file) => file.name === name);
      assert.equal(init.body.byteLength, entry.bytes);
      assert.equal(sha(init.body), entry.sha256);
      const key = id + '/' + name;
      if (stored.has(key)) return new Response('{"error":"already stored"}', { status: 409 });
      const response = fail?.(name, stored, key, init.body);
      if (response) return response;
      stored.set(key, Buffer.from(init.body));
      return new Response('{}', { status: 201 });
    } finally {
      active--;
    }
  };
  return { fetcher, records, calls, stored, stats: () => ({ posts, peak }) };
}

test('50 saved depth files use legacy healthy manifests, sequential bytes and preserve original gaps', async () => {
  const input = fixture(),
    collector = legacyCollector();
  const inventory = await api.openBackup(input.directory);
  let consent = false;
  const transfer = api.createTransfer({
    record: sampleRecord(),
    inventory,
    origin,
    fetcher: collector.fetcher,
    getConsent: () => consent,
    now: () => now,
  });
  assert.equal(collector.calls.length, 0);
  assert.equal(input.stats().reads, 0);
  await assert.rejects(transfer.start(), /동의/);
  assert.equal(collector.calls.length, 0);
  consent = true;
  const receipt = await transfer.start();
  assert.equal(receipt.id, '20261025010000000-00000001');
  assert.equal(receipt.files.length, 50);
  assert.ok(receipt.files.every((file) => file.status === 'stored'));
  assert.equal(receipt.excluded.length, 2);
  assert.equal(receipt.sourceArchivesComplete, false);
  assert.equal(receipt.uploadWindowEndsAt, '2026-10-25T02:00:00.000Z');
  assert.equal(collector.stats().peak, 1);
  assert.deepEqual(input.stats(), { reads: 100, peak: 1 });
  const record = collector.records.get(receipt.id);
  assert.deepEqual(record.recovery.backup, input.manifest);
  assert.equal(record.recovery.incompleteOriginalCapture, true);
  assert.equal(record.recovery.missingRetainedFiles.length, 2);
  assert.match(record.recovery.scope, /primary.*not included/);
  assert.equal(collector.stored.size, 50);
});

test('one failed PUT does not stop later files, unknown success retries as409, manual retry uses the same ID', async () => {
  const input = fixture();
  let blocked = true,
    unknown = true;
  const first = input.manifest.files[0].name,
    second = input.manifest.files[1].name;
  const collector = legacyCollector({
    fail(name, stored, key, bytes) {
      if (name === first && blocked) return new Response('{"error":"busy"}', { status: 503 });
      if (name === second && unknown) {
        unknown = false;
        stored.set(key, Buffer.from(bytes));
        throw TypeError('response lost');
      }
    },
  });
  let time = now;
  const transfer = api.createTransfer({
    record: sampleRecord(),
    inventory: await api.openBackup(input.directory),
    origin,
    fetcher: collector.fetcher,
    getConsent: () => true,
    now: () => time,
  });
  const receipt = await transfer.start();
  assert.equal(receipt.status, 'partial');
  assert.equal(receipt.files[0].status, 'failed');
  assert.equal(receipt.files[0].attempts, 2);
  assert.equal(receipt.files[1].status, 'stored');
  assert.equal(receipt.files[1].attempts, 2);
  assert.equal(receipt.files.at(-1).status, 'stored');
  blocked = false;
  const previousPuts = collector.calls.filter((call) => call.method === 'PUT').length;
  await transfer.retry();
  assert.equal(collector.stats().posts, 1);
  assert.equal(collector.calls.filter((call) => call.method === 'PUT').length, previousPuts + 1);
  assert.ok(receipt.files.every((file) => file.status === 'stored'));
  time += 60 * 60_000;
  await assert.rejects(transfer.retry(), /1시간/);
});

test('invalid paths, duplicate names, size mismatch and false capture completion stop before network', async () => {
  const input = fixture();
  const clone = () => structuredClone(input.manifest);
  for (const mutate of [
    (value) => {
      value.files[0].name = '../secret';
    },
    (value) => {
      value.files[0].name = 'typecv-dense.ply';
    },
    (value) => {
      value.files[0].name = value.files[1].name;
    },
    (value) => {
      value.files[0].expectedBytes++;
    },
    (value) => {
      value.files[0].bytesWritten--;
    },
    (value) => {
      value.sourceArchivesComplete = true;
    },
    (value) => {
      value.files[0].expectedBytes = 51 * 1024 ** 2;
      value.files[0].bytesWritten = value.files[0].expectedBytes;
    },
  ]) {
    const value = clone();
    mutate(value);
    assert.throws(() => api.validateManifest(value));
  }
  input.data.set(input.manifest.files[0].name, new Uint8Array(1));
  await assert.rejects(api.openBackup(input.directory), /実際|실제 파일 크기/);
  assert.equal(input.stats().reads, 0);
  assert.throws(() => api.recordFromDocument({ querySelectorAll: () => [] }), /TypeCV Lab/);
  assert.throws(() => api.validateRecord({ format: 'typecv-run-record-v6' }), /JSON/);
});

test('read failures omit only unhealthy legacy declarations and preserve an honest recovery receipt', async () => {
  const input = fixture(),
    inventory = await api.openBackup(input.directory),
    collector = legacyCollector();
  const failed = input.manifest.files[0].name;
  const original = input.directory.getFileHandle;
  input.directory.getFileHandle = async (name, ...rest) => {
    if (name === failed)
      return {
        getFile: async () => ({
          size: input.data.get(name).length,
          arrayBuffer: async () => {
            throw new DOMException('File backing lost', 'NotReadableError');
          },
        }),
      };
    return original(name, ...rest);
  };
  const transfer = api.createTransfer({
    record: sampleRecord(),
    inventory,
    origin,
    fetcher: collector.fetcher,
    getConsent: () => true,
    now: () => now,
    onReceipt: async () => {
      throw Error('local receipt write failed');
    },
  });
  const receipt = await transfer.start();
  assert.equal(receipt.preparationFailures.length, 1);
  assert.equal(receipt.preparationFailures[0].name, failed);
  assert.equal(receipt.files.length, 49);
  assert.equal(receipt.status, 'partial');
  assert.match(receipt.receiptError, /receipt write failed/);
  const record = collector.records.get(receipt.id);
  assert.equal(record.files.length, 49);
  assert.equal(record.recovery.preparationFailures[0].name, failed);
  assert.ok(record.files.every((file) => file.sha256 && file.bytes !== undefined));
});

function documentFixture(record) {
  const nodes = [];
  const createElement = (tag) => {
    const node = {
      tag,
      style: {},
      children: [],
      events: new Map(),
      disabled: false,
      append(...children) {
        this.children.push(...children);
      },
      setAttribute() {},
      scrollIntoView() {},
      addEventListener(name, handler) {
        this.events.set(name, handler);
      },
      async click() {
        if (!this.disabled) return this.events.get('click')?.();
      },
    };
    nodes.push(node);
    return node;
  };
  return {
    nodes,
    body: createElement('body'),
    createElement,
    getElementById: (id) => nodes.find((node) => node.id === id),
    querySelectorAll: (selector) =>
      selector === '.lab-app' ? [{}] : record ? [{ textContent: JSON.stringify(record) }] : [],
  };
}

test('installing and choosing a folder never sends; only explicit checked Send starts current-origin requests', async () => {
  const input = fixture(),
    collector = legacyCollector(),
    document = documentFixture(sampleRecord());
  class FixedDate extends Date {
    static now() {
      return now;
    }
  }
  vm.runInNewContext(
    source,
    {
      ...globals,
      Date: FixedDate,
      document,
      location: { origin },
      window: { showDirectoryPicker: async () => input.directory },
      fetch: collector.fetcher,
      console,
      alert: (message) => {
        throw Error(message);
      },
      setTimeout,
    },
    { timeout: 1000 },
  );
  assert.equal(collector.calls.length, 0);
  const button = (text) =>
    document.nodes.find((node) => node.tag === 'button' && node.textContent === text);
  await button('저장된 백업 하위 폴더 선택').click();
  assert.equal(collector.calls.length, 0);
  const consent = document.nodes.find((node) => node.tag === 'input' && node.type === 'checkbox');
  assert.equal(consent.checked, false);
  await button('선택 파일 검증 후 전송').click();
  assert.equal(collector.calls.length, 0);
  consent.checked = true;
  consent.events.get('change')();
  await button('선택 파일 검증 후 전송').click();
  assert.equal(collector.stats().posts, 1);
  assert.equal(collector.stored.size, 50);
  assert.ok(
    input.outputs.size > 0,
    'a small receipt is durably saved without changing archive inputs',
  );
  assert.equal(input.data.size, 50);
});

test('a fresh Lab tab accepts its saved small run JSON without a reconstruction or network request', async () => {
  const input = fixture(),
    collector = legacyCollector(),
    document = documentFixture();
  class FixedDate extends Date {
    static now() {
      return now;
    }
  }
  vm.runInNewContext(
    source,
    {
      ...globals,
      Date: FixedDate,
      document,
      location: { origin },
      window: { showDirectoryPicker: async () => input.directory },
      fetch: collector.fetcher,
      console,
      alert: (message) => {
        throw Error(message);
      },
      setTimeout,
    },
    { timeout: 1000 },
  );
  const button = (text) =>
    document.nodes.find((node) => node.tag === 'button' && node.textContent === text);
  const fileInput = document.nodes.find((node) => node.tag === 'input' && node.type === 'file');
  const consent = document.nodes.find((node) => node.tag === 'input' && node.type === 'checkbox');
  assert.equal(button('저장된 백업 하위 폴더 선택').disabled, true);
  fileInput.files = [new Blob([JSON.stringify(sampleRecord())], { type: 'application/json' })];
  await fileInput.events.get('change')();
  assert.equal(consent.checked, false);
  assert.equal(collector.calls.length, 0);
  await button('저장된 백업 하위 폴더 선택').click();
  await button('선택 파일 검증 후 전송').click();
  assert.equal(collector.calls.length, 0);
  consent.checked = true;
  consent.events.get('change')();
  await button('선택 파일 검증 후 전송').click();
  assert.equal(collector.stats().posts, 1);
  assert.equal(collector.stored.size, 50);
  fileInput.files = [new Blob(['{"format":"wrong"}'])];
  await fileInput.events.get('change')();
  assert.equal(consent.checked, false);
  assert.equal(
    button('선택 파일 검증 후 전송').disabled,
    true,
    'changing records invalidates the existing plan',
  );
});

test('record file size is checked before reading and malformed or reconstruction JSON is rejected', async () => {
  let reads = 0;
  await assert.rejects(
    api.readRecordFile({
      size: 256 * 1024 + 1,
      text: async () => {
        reads++;
        return '{}';
      },
    }),
    /256 KiB/,
  );
  assert.equal(reads, 0);
  await assert.rejects(api.readRecordFile(new Blob(['not JSON'])));
  await assert.rejects(
    api.readRecordFile(new Blob(['{"format":"typecv-reconstruction-v1"}'])),
    /JSON/,
  );
  const record = await api.readRecordFile(new Blob([JSON.stringify(sampleRecord())]));
  assert.equal(record.format, 'typecv-run-record-v6');
});
