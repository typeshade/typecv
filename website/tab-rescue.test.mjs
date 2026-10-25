import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
// Execute the shipped standalone snippet in its no-document test branch. No
// reconstruction bundle, browser, network or current user-tab data is involved.
const source = readFileSync(new URL('../scripts/typecv-tab-rescue.js', import.meta.url), 'utf8');
const api = vm.runInNewContext(
  source,
  {
    Blob,
    Map,
    Set,
    TransformStream,
  },
  { timeout: 1000 },
);
const part = 'typecv-depths-c000-0000.tar.gz';
const manifest = 'typecv-depths-c000.json';
function doc(refs) {
  let next = null;
  for (const current of refs) next = { memoizedState: { current }, next };
  const parent = { memoizedState: next };
  const root = { __reactFiber$test: { memoizedState: null, return: parent } };
  return { querySelectorAll: () => [root] };
}
function fakeDirectory({ reportedSize } = {}) {
  const bytes = new Map();
  const aborted = new Set();
  return {
    bytes,
    aborted,
    async getFileHandle(name) {
      return {
        async createWritable() {
          const chunks = [];
          return new WritableStream({
            write(chunk) {
              chunks.push(Buffer.from(chunk));
            },
            close() {
              bytes.set(name, Buffer.concat(chunks));
            },
            abort() {
              aborted.add(name);
            },
          });
        },
        async getFile() {
          return new Blob([
            reportedSize === undefined
              ? (bytes.get(name) ?? Buffer.alloc(0))
              : new Uint8Array(reportedSize),
          ]);
        },
      };
    },
  };
}

test('strict React ref discovery snapshots payload references and refuses ambiguity', () => {
  const blob = new Blob(['a']);
  const original = new Map([[part, blob]]);
  const snapshot = api.artifactsFromDocument(doc([new Map([['unrelated', blob]]), original]));
  assert.notEqual(snapshot, original);
  assert.equal(snapshot.get(part), blob);
  assert.equal(original.size, 1);
  assert.throws(
    () => api.artifactsFromDocument({ querySelectorAll: () => [] }),
    /exactly one TypeCV Lab/,
  );
  original.clear();
  assert.equal(snapshot.get(part), blob, 'snapshot retains references if an app ref changes later');
  assert.throws(
    () => api.artifactsFromDocument(doc([snapshot, new Map([[part, blob]])])),
    /Ambiguous/,
  );
  assert.throws(
    () => api.artifactsFromDocument(doc([new Map([['evil/path', blob]])])),
    /No supported/,
  );
  assert.throws(
    () => api.artifactsFromDocument(doc([new Map([[part, new Uint8Array([1])]])])),
    /No supported/,
  );
  assert.equal(snapshot.get(part), blob);
});

test('copy uses a stream and verifies bytes without reading the complete Blob', async () => {
  const content = Uint8Array.from({ length: 512 * 1024 + 17 }, (_, i) => i % 251);
  class StreamOnlyBlob extends Blob {
    arrayBuffer() {
      throw new Error('Whole Blob read prohibited');
    }
    stream() {
      let at = 0;
      return new ReadableStream({
        pull(controller) {
          if (at === content.length) return controller.close();
          const end = Math.min(content.length, at + 16384);
          controller.enqueue(content.subarray(at, end));
          at = end;
        },
      });
    }
  }
  const blob = new StreamOnlyBlob([content]);
  const directory = fakeDirectory();
  const result = await api.copyBlob(directory, part, blob);
  assert.equal(result.bytesWritten, content.length);
  assert.deepEqual(directory.bytes.get(part), Buffer.from(content));
  assert.equal(blob.size, content.length);
});

test('unreadable data aborts only that output and later artifacts still save', async () => {
  class UnreadableBlob extends Blob {
    stream() {
      return new ReadableStream({
        start(controller) {
          controller.error(new DOMException('Backing file lost', 'NotReadableError'));
        },
      });
    }
  }
  const broken = new UnreadableBlob(['missing']);
  const good = new Blob(['still retained']);
  const entries = new Map([
    [part, broken],
    ['typecv-depths-c000-0001.tar.gz', good],
    [manifest, '{}'],
  ]);
  const directory = fakeDirectory();
  let notifications = 0;
  const rows = await api.saveEntries(entries, directory, () => notifications++);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].status, 'failed');
  assert.match(rows[0].error, /NotReadableError/);
  assert.equal(rows[1].status, 'saved');
  assert.equal(rows[2].status, 'saved');
  assert.equal(notifications, 3);
  assert.equal(directory.aborted.has(part), true);
  assert.equal(directory.bytes.get('typecv-depths-c000-0001.tar.gz').toString(), 'still retained');
  assert.equal(entries.get(part), broken, 'failed source reference preserved');
  assert.equal(entries.get('typecv-depths-c000-0001.tar.gz'), good);
});

test('truncated source streams are failed copies and do not stop later files', async () => {
  class TruncatedBlob extends Blob {
    stream() {
      return new Blob(['ab']).stream();
    }
  }
  const original = new TruncatedBlob(['abc']);
  const next = new Blob(['complete']);
  const directory = fakeDirectory();
  const entries = new Map([
    [part, original],
    ['typecv-depths-c000-0001.tar.gz', next],
  ]);
  const rows = await api.saveEntries(entries, directory);
  assert.equal(rows[0].status, 'failed');
  assert.match(rows[0].error, /byte count does not match/);
  assert.equal(rows[0].expectedBytes, 3);
  assert.equal(directory.bytes.get(part).byteLength, 2);
  assert.equal(rows[1].status, 'saved');
  assert.equal(rows[1].bytesWritten, next.size);
  assert.equal(entries.get(part), original);
  assert.equal(entries.get('typecv-depths-c000-0001.tar.gz'), next);
});

test('a completed source stream still fails when the saved file reports a partial write', async () => {
  const original = new Blob(['abc']);
  const directory = fakeDirectory({ reportedSize: 2 });
  await assert.rejects(api.copyBlob(directory, part, original), /byte count does not match/);
  assert.equal(directory.bytes.get(part).toString(), 'abc');
  assert.equal(original.size, 3);
});

test('cancellation does not open files or discard pending references', async () => {
  const controller = new AbortController();
  controller.abort();
  const blob = new Blob(['a']);
  let calls = 0;
  const directory = {
    getFileHandle() {
      calls++;
      throw Error('Must not write');
    },
  };
  const rows = await api.saveEntries(
    new Map([[part, blob]]),
    directory,
    undefined,
    controller.signal,
  );
  assert.equal(rows[0].status, 'not-attempted');
  assert.equal(calls, 0);
});

test('source completeness is distinct from copying all retained artifacts', () => {
  const json = JSON.stringify({
    format: 'typecv-depth-archive-v1',
    outcome: 'completed',
    complete: true,
    archives: [part, 'typecv-depths-c000-0001.tar.gz'],
    missing: [],
  });
  const info = api.sourceSummary(
    new Map([
      [manifest, json],
      [part, new Blob(['a'])],
    ]),
  );
  assert.equal(info[0].captureComplete, true);
  assert.equal(info[0].missingRetainedParts.join(), 'typecv-depths-c000-0001.tar.gz');
  const invalid = api.sourceSummary(new Map([[manifest, '{not JSON']]));
  assert.equal(invalid[0].captureComplete, false);
});

test('missing complete component manifests and unknown run scope never claim full source', () => {
  const record = {
    format: 'typecv-run-record-v6',
    input: {
      videos: [
        { component: 0, registeredViews: 240 },
        { component: 1, registeredViews: 93 },
      ],
    },
    result: { components: [{}, {}] },
    dense: { status: 'completed', views: 333 },
  };
  const retained = new Map([
    [
      manifest,
      JSON.stringify({ format: 'typecv-depth-archive-v1', complete: true, archives: [part] }),
    ],
    [part, new Blob(['a'])],
  ]);
  const scope = api.scopeFromRecord(record);
  assert.equal(scope.known, true);
  assert.equal(scope.expectedComponentIds.join(), '0,1');
  const incomplete = api.sourceCompleteness(retained, scope);
  assert.equal(incomplete.complete, false);
  assert.equal(incomplete.archiveScope.missingComponentManifests.join(), 'typecv-depths-c001.json');
  const unknown = api.sourceCompleteness(retained, api.scopeFromRecord(undefined));
  assert.equal(unknown.archiveScope.known, false);
  assert.equal(unknown.complete, false);
  retained.set(
    'typecv-depths-c001.json',
    JSON.stringify({
      format: 'typecv-depth-archive-v1',
      complete: true,
      archives: [],
    }),
  );
  assert.equal(api.sourceCompleteness(retained, scope).complete, true);
  retained.set(
    'typecv-depths-c003.json',
    JSON.stringify({
      format: 'typecv-depth-archive-v1',
      complete: true,
      archives: [],
    }),
  );
  const unexpected = api.sourceCompleteness(retained, scope);
  assert.equal(unexpected.complete, false);
  assert.equal(
    unexpected.archiveScope.unexpectedComponentManifests.join(),
    'typecv-depths-c003.json',
  );
});
