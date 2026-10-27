#!/usr/bin/env node
// Compare a hardware-check record of `/lab/check/` (docs/lab-run-records.md, "Hardware check")
// with its baselines, and print one Markdown table to cite in a pull request or an amendment.
//
//   node scripts/compare-hardware-check.mjs --record lab-records/<id>.json
//   node scripts/compare-hardware-check.mjs --record lab-records/<id>.json \
//     --against lab-records/<earlier id>.json \
//     --benchmark lab-records/<cold id>.json --benchmark lab-records/<warm id>.json \
//     --json comparison.json
//
// --record file.json     the check record, as scripts/export-run-records.mjs wrote it or as the
//                        page's "Download JSON" saved it.
// --baseline file.json   the CPU-tier digests of Node (default tests/digests/hardware-check.json).
// --against file.json    an earlier check record of the same device: each case's answer on the
//                        device must have the same digest (the gate of a change that must not
//                        change a result). A record of another device is reported, not compared.
// --benchmark file.json  a run record of the standard benchmark (repeatable), reported beside.
// --json out.json        also write the rows and the verdict.
//
// Each row has a verdict: PASS, FAIL, WARN, INFO, SKIP or n.a. A row fails when the flavor that
// the TypeShade device selected differs from TypeCV's rule before change 0085 (recomputed here
// from the recorded signals), when a case did not complete, is outside its bound or gave two
// answers on one device, when an answer differs from the earlier record of the same device, and
// when a benchmark run failed. A row warns when the browser's CPU tier, or the input the page
// built, differs from Node's baseline: that is the browser's JavaScript engine or another commit,
// not the device. The exit code is 0 when no row fails, 1 when one fails, and 2 when an input
// cannot be read or is not the record it should be.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { previousFp64Flavor } from '../website/src/lib/hardware-check/flavor.ts';
import { hardwareCheckProblem } from '../website/src/lib/hardware-check/record.ts';
import { runRecordProblem } from '../website/src/lib/run-record.ts';

export const DEFAULT_BASELINE = fileURLToPath(
  new URL('../tests/digests/hardware-check.json', import.meta.url),
);
const short = (digest) => (typeof digest === 'string' ? digest.slice(0, 12) : 'none');
const identity = (a) =>
  [a?.vendor, a?.architecture, a?.device, a?.description].map((s) => s ?? '').join('|');

/** The rows of the comparison. `cpu` is the baseline file's `cases`. */
export function compareHardwareCheck(record, { baseline = {}, against, benchmarks = [] } = {}) {
  const rows = [];
  const row = (check, expected, observed, verdict) =>
    rows.push({ check, expected, observed, verdict });
  const a = record.adapter;
  row(
    'record',
    'a hardware-check record',
    `${record.server?.id ?? 'not stored'}, site ${short(record.site.commit)}, TypeShade ${short(record.site.typeshade)}, ${record.startedAt}`,
    'INFO',
  );
  row(
    'device',
    "the adapter and the visitor's note",
    (a
      ? [a.vendor, a.architecture, a.description].filter(Boolean).join(' ') +
        (a.fallbackAdapter ? ' (fallback adapter)' : '')
      : 'no adapter') + (record.note ? `; note: ${record.note}` : '; no note'),
    'INFO',
  );
  const f = record.flavor;
  const previous = previousFp64Flavor({
    handedDevice: true,
    deviceAdapterInfo: f.signals.deviceAdapterInfo,
    userAgent: f.signals.userAgent,
  });
  const previousRequested = previousFp64Flavor({
    handedDevice: false,
    requestedAdapterInfo: f.signals.requestedAdapterInfo,
    userAgent: f.signals.userAgent,
  });
  for (const [label, selected, rule] of [
    ["flavor, the lab's path", f.selected, previous],
    ['flavor, the default path', f.selectedDefault, previousRequested],
  ])
    row(
      label,
      `the rule before change 0085: ${rule}`,
      selected === null
        ? `no device opened${f.error ? ': ' + f.error.split('\n')[0] : ''}`
        : selected,
      selected === rule ? 'PASS' : 'FAIL',
    );
  const consistent =
    f.previous === previous &&
    f.previousRequested === previousRequested &&
    f.same ===
      (f.selected === null || f.selectedDefault === null
        ? null
        : f.selected === previous && f.selectedDefault === previousRequested);
  row(
    "flavor, the page's arithmetic",
    'the rule recomputed from the recorded signals',
    consistent ? 'equal' : 'different',
    consistent ? 'PASS' : 'FAIL',
  );
  const earlier = against;
  const sameDevice =
    earlier !== undefined &&
    identity(earlier.adapter) === identity(a) &&
    earlier.client.userAgent === record.client.userAgent;
  if (earlier !== undefined)
    row(
      'earlier record',
      'a record of the same device',
      `${earlier.server?.id ?? earlier.startedAt}, site ${short(earlier.site.commit)}` +
        (sameDevice ? '' : ': another adapter or browser, so its answers are not compared'),
      sameDevice ? 'INFO' : 'n.a.',
    );
  for (const c of record.cases) {
    const name = c.name;
    if (c.outcome === 'skipped') {
      row(name, 'runs', `skipped: ${c.skipped}`, 'SKIP');
      continue;
    }
    row(
      `${name}: outcome`,
      'completed without a device loss',
      c.outcome +
        (c.loss
          ? ` (${c.loss.reason}: ${c.loss.message.split('\n')[0]}; last kernel ${c.loss.lastKernel})`
          : '') +
        (c.error ? ` (${c.error.split('\n')[0]})` : ''),
      c.outcome === 'completed' ? 'PASS' : 'FAIL',
    );
    if (c.agreement) {
      const measured = c.agreement.maxAbs ?? c.agreement.maxRel ?? c.agreement.mismatches;
      row(
        `${name}: bound`,
        `${c.bound.kind} ${c.bound.value} (${c.bound.source})`,
        `${measured} over ${c.agreement.compared} values` +
          (c.agreement.problems.length ? `; ${c.agreement.problems.join('; ')}` : ''),
        c.pass ? 'PASS' : 'FAIL',
      );
    }
    if (c.repeatable !== null)
      row(
        `${name}: second run`,
        'the same answer on the same device',
        c.repeatable ? 'the same' : 'a different answer',
        c.repeatable ? 'PASS' : 'FAIL',
      );
    const base = baseline[name];
    for (const [label, key] of [
      ['input', 'input'],
      ['CPU tier', 'cpu'],
    ])
      row(
        `${name}: ${label}`,
        `Node's baseline ${short(base?.[key])}`,
        short(c.digests[key]),
        base?.[key] === undefined || c.digests[key] === null
          ? 'n.a.'
          : base[key] === c.digests[key]
            ? 'PASS'
            : 'WARN',
      );
    if (c.digests.device !== null)
      row(
        `${name}: device bits`,
        "the CPU tier's bits",
        c.digests.device === c.digests.cpu
          ? 'equal'
          : 'different in ' +
              (Object.keys(c.digests.deviceParts ?? {})
                .filter((key) => c.digests.deviceParts[key] !== c.digests.cpuParts?.[key])
                .join(', ') || 'the whole answer'),
        'INFO',
      );
    if (earlier !== undefined && sameDevice) {
      const e = earlier.cases.find((x) => x.name === name);
      row(
        `${name}: against the earlier record`,
        short(e?.digests.device),
        short(c.digests.device),
        !e?.digests.device || !c.digests.device
          ? 'n.a.'
          : e.digests.device === c.digests.device
            ? 'PASS'
            : 'FAIL',
      );
    }
    const gpuMs = c.gpu.reduce((sum, g) => sum + (g.gpuMs ?? 0), 0);
    const timed = c.gpu.some((g) => g.gpuMs !== null);
    row(
      `${name}: time`,
      'measured',
      `${c.coldMs ?? '?'} ms cold, ${c.warmMs ?? '?'} ms warm, ${c.cpuMs ?? '?'} ms on the CPU tier; ` +
        `${timed ? gpuMs.toFixed(3) + ' ms GPU' : 'no GPU time'}, ` +
        `${c.gpu.reduce((sum, g) => sum + g.dispatches, 0)} dispatches`,
      'INFO',
    );
  }
  if (record.aborted)
    row(
      'check',
      'every case runs',
      `${record.aborted}; resume: ${record.resume ?? 'none'}`,
      'FAIL',
    );
  for (const b of benchmarks) {
    const id = b.server?.id ?? b.createdAt;
    const sameBenchmarkDevice =
      identity(b.gpu) === identity(a) && b.client.userAgent === record.client.userAgent;
    const longest = Math.max(0, ...Object.values(b.timing.chunks ?? {}).map((r) => r.longestMs));
    row(
      `benchmark ${id}`,
      `${b.benchmark?.name ?? 'no benchmark'} ${b.benchmark?.pass ?? ''}`.trim(),
      b.failure
        ? `failed at ${b.failure.kernel || b.failure.operation}: ${b.failure.message.split('\n')[0]}`
        : `${Math.round(b.timing.totalMs)} ms, ${b.result.registeredViews} of ${b.input.frames} views, ` +
            `${b.result.landmarks} landmarks, ${b.result.meanReprojectionErrorPx ?? '?'} px, ` +
            `fp64 ${b.options.fp64Flavor}, longest chunk ${Math.round(longest)} ms`,
      b.failure ? 'FAIL' : 'INFO',
    );
    row(
      `benchmark ${id}: flavor`,
      `the check's selection: ${f.selected ?? 'none'}`,
      sameBenchmarkDevice ? b.options.fp64Flavor : 'another adapter or browser',
      !sameBenchmarkDevice ? 'n.a.' : b.options.fp64Flavor === f.selected ? 'PASS' : 'WARN',
    );
  }
  return rows;
}

/** 0 when no row fails, 1 when one fails. */
export const verdict = (rows) => (rows.some((r) => r.verdict === 'FAIL') ? 1 : 0);

/** The rows as a Markdown table, with a count of each verdict. */
export function markdown(rows) {
  const cell = (text) => String(text).replaceAll('|', '\\|').replaceAll('\n', ' ');
  const counts = {};
  for (const r of rows) counts[r.verdict] = (counts[r.verdict] ?? 0) + 1;
  return [
    '| Check | Expected | Observed | Verdict |',
    '| --- | --- | --- | --- |',
    ...rows.map(
      (r) => `| ${cell(r.check)} | ${cell(r.expected)} | ${cell(r.observed)} | ${r.verdict} |`,
    ),
    '',
    `${rows.length} rows: ` +
      Object.entries(counts)
        .map(([v, n]) => `${n} ${v}`)
        .join(', '),
  ].join('\n');
}

/** Read and parse a JSON file, or throw a message that names it. */
function readJson(file, what) {
  try {
    return JSON.parse(readFileSync(file, 'utf8'));
  } catch (error) {
    throw new Error(`Cannot read the ${what} ${file}: ${error.message}`);
  }
}

function main() {
  const { values } = parseArgs({
    options: {
      record: { type: 'string' },
      baseline: { type: 'string', default: DEFAULT_BASELINE },
      against: { type: 'string' },
      benchmark: { type: 'string', multiple: true, default: [] },
      json: { type: 'string' },
    },
  });
  if (!values.record) throw new Error('--record <file> is required');
  const record = readJson(values.record, 'record');
  const problem = hardwareCheckProblem(record);
  if (problem)
    throw new Error(`${values.record} is not a hardware-check record (field ${problem})`);
  const against = values.against ? readJson(values.against, 'earlier record') : undefined;
  const earlierProblem = against && hardwareCheckProblem(against);
  if (earlierProblem)
    throw new Error(`${values.against} is not a hardware-check record (field ${earlierProblem})`);
  const benchmarks = values.benchmark.map((file) => {
    const run = readJson(file, 'benchmark record');
    const runProblem = runRecordProblem(run);
    if (runProblem) throw new Error(`${file} is not a run record (field ${runProblem})`);
    return run;
  });
  const baseline = readJson(values.baseline, 'baseline').cases ?? {};
  const rows = compareHardwareCheck(record, { baseline, against, benchmarks });
  console.log(markdown(rows));
  if (values.json)
    writeFileSync(
      values.json,
      JSON.stringify({ record: record.server?.id ?? null, verdict: verdict(rows), rows }, null, 2) +
        '\n',
    );
  return verdict(rows);
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  try {
    process.exitCode = main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}
