import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { CASES } from './src/lib/hardware-check/cases.ts';
import {
  failedCases,
  HARDWARE_CHECK_FORMAT,
  hardwareCheckProblem,
} from './src/lib/hardware-check/record.ts';

// The hardware check of `/lab/check/` (docs/lab-run-records.md, "Hardware check") in Chromium on
// SwiftShader, as CI runs the browser tests. The test bundles the check's worker, serves the
// runtime entries at `/runtime/` as the site does (website/scripts/prepare.mjs), and runs the
// check twice: every case, then the one case that a `resume` query would name. SwiftShader stands
// in for a device only to prove the mechanism: it holds the record to its validator, every case to
// its bound and to its second run, the flavor to the rule before TypeShade change 0085, and the
// CPU tier of the browser to the digests of Node (tests/digests/hardware-check.json). With
// TYPECV_CHECK_RECORD_OUT=<file> it also writes the record of the first check there: the source of
// tests/fixtures/hardware-check/swiftshader.json, the fixture of tests/compare-hardware-check.test.ts.

const root = fileURLToPath(new URL('..', import.meta.url));

/** The site's runtime as `/runtime/`: `dist/gpu.js` and the files of `dist/gpu` and `dist/internal`. */
async function runtime() {
  const serve = { '/runtime/gpu.js': join(root, 'dist/gpu.js') };
  for (const tree of ['gpu', 'internal'])
    for (const file of await readdir(join(root, 'dist', tree), { recursive: true }))
      if (file.endsWith('.js')) serve[`/runtime/${tree}/${file}`] = join(root, 'dist', tree, file);
  return serve;
}

test('the hardware check runs every case on WebGPU and builds a valid record', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'typecv-hardware-check-'));
  try {
    const workerFile = join(directory, 'worker.js');
    await build({
      entryPoints: [
        fileURLToPath(new URL('./src/workers/hardware-check.worker.ts', import.meta.url)),
      ],
      outfile: workerFile,
      bundle: true,
      format: 'esm',
      platform: 'browser',
      logLevel: 'silent',
    });
    const serve = {
      ...(await runtime()),
      '/worker.js': workerFile,
      '/sample/probe/south-building-pair-triangulation.json': join(
        root,
        'website/public/sample/probe/south-building-pair-triangulation.json',
      ),
    };
    const { result, errors } = await inBrowser(
      async () => {
        const check = (cases) =>
          new Promise((resolve, reject) => {
            const worker = new Worker('/worker.js', { type: 'module' });
            const lines = [];
            worker.onmessage = ({ data }) => {
              if (data.type === 'log') lines.push(data.line);
              else if (data.type === 'result') {
                worker.terminate();
                resolve({ record: data.result, lines });
              } else if (data.type === 'error') reject(new Error(data.message));
            };
            worker.onerror = (event) => reject(new Error(event.message));
            worker.postMessage({
              type: 'start',
              href: location.href,
              commit: 'test',
              typeshade: null,
              note: 'SwiftShader, website/hardware-check-browser.test.mjs',
              cases,
            });
          });
        return { all: await check([]), one: await check(['kernel-functions']) };
      },
      undefined,
      { timeoutMs: 1200000, serve },
    );
    assert.deepEqual(errors, []);
    const { record, lines } = result.all;
    if (process.env.TYPECV_CHECK_RECORD_OUT)
      await writeFile(process.env.TYPECV_CHECK_RECORD_OUT, JSON.stringify(record, null, 2) + '\n');
    assert.equal(record.format, HARDWARE_CHECK_FORMAT);
    assert.equal(hardwareCheckProblem(record), undefined);
    assert.ok(JSON.stringify(record).length < 256 * 1024);
    assert.deepEqual(
      record.cases.map((c) => c.name),
      CASES.map((c) => c.name),
    );
    assert.deepEqual(failedCases(record), [], lines.join('\n'));
    for (const c of record.cases) {
      assert.equal(c.outcome, 'completed', c.name);
      assert.equal(c.pass, true, c.name);
      assert.equal(c.repeatable, true, c.name);
      assert.equal(c.loss, null, c.name);
      assert.ok(c.gpu.length > 0 && c.coldMs >= 0 && c.warmMs >= 0, c.name);
    }
    assert.equal(record.aborted, null);
    assert.equal(record.resume, null);
    // SwiftShader is not an Apple GPU: both paths select the float flavor, as the rule did.
    assert.equal(record.flavor.selected, 'float');
    assert.equal(record.flavor.selectedDefault, 'float');
    assert.equal(record.flavor.same, true);
    assert.equal(record.flavor.info.provenance, 'device');
    assert.equal(record.adapter.architecture, 'swiftshader');
    // The CPU tier of the browser gives the digests of Node's CPU tier, on the same inputs.
    const fixture = JSON.parse(
      await readFile(join(root, 'tests/digests/hardware-check.json'), 'utf8'),
    ).cases;
    for (const c of record.cases) {
      assert.equal(c.digests.input, fixture[c.name].input, `${c.name}: the input`);
      assert.equal(c.digests.cpu, fixture[c.name].cpu, `${c.name}: the CPU tier`);
    }
    // A query names the cases to run, as `resume` does after a loss.
    assert.deepEqual(
      result.one.record.cases.map((c) => [c.name, c.outcome, c.pass]),
      [['kernel-functions', 'completed', true]],
    );
    console.log(
      JSON.stringify(
        record.cases.map((c) => ({
          name: c.name,
          coldMs: c.coldMs,
          warmMs: c.warmMs,
          deviceEqualsCpu: c.digests.device === c.digests.cpu,
          maxAbs: c.agreement.maxAbs,
          maxRel: c.agreement.maxRel,
        })),
      ),
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
