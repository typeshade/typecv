// Compiler probes: the verdicts of the pinned TypeShade compiler on the kernel shapes that TypeCV
// cares about.
//
// What it is. Each probe compiles one small "use typeshade" program with the compiler that
// package.json pins. It holds the verdict that the compiler gives today: a GPU loop and its
// tiers, the CPU tier with TS8070 and its proof rule, a silent CPU fallback, a refusal with its
// code, or an entry that compiles, with its shape on each target and its results on the CPU
// tiers. The probes test no TypeCV code. They hold the compiler limits that TypeCV's refactoring
// programme waits on, and the capabilities that a refactor may use. So a pin bump that closes a
// gap fails a probe, and the failure names the conversion step that may open.
//
// The record:
// - docs/library-structure.md: section 6 (rows C1 to C13 and the "Not gaps" list), section 3.6
//   and the phase 9 and 10 rows of section 5;
// - docs/audits/2026-10-26-kernel-style.md: section 0.1 (probes P1 to P30) and section 6 (the
//   gaps, ranked).
// Both are dated records. This suite does not change them. K1 to K6 have no dated record yet:
// the pull request that added them holds their table.
//
// Measured on 2026-10-26: typeshade 725b2969 (MEASURED_AT, the package.json pin; package version
// 0.1.0), Node 22.22.0, the CPU tier (no WebGPU device). The pin bump from d36ac502, where #243
// first measured the suite, read every probe again (#246; docs/library-structure.md, amendment 2).
// No verdict changed. K1 to K6 were measured on 2026-10-27 with the same pin and Node. Their WGSL
// and their WebGL2 pass programs were read, not run.
//
// How to read a failure. The test name gives the probe, the row it serves, the shape and the
// verdict held. The assertion message adds what changed and what the change means. After a pin
// bump, a failure is a signal, not a TypeCV defect. Read the compiler change, then open the
// TypeCV step that the message names, under its own record.
//
// Changing an expected verdict. A probe's expected verdict changes only in a pull request that
// cites the compiler change (the pin bump, with the TypeShade change or issue) and the TypeCV row
// of the probe. MEASURED_AT moves with the pin, in the same pull request.
//
// Ids. P1 to P30 are the audit's. P7, P10, P14 to P16, P18 and P19 are the names that the audit's
// harness gave to shapes that the audit's table groups under P6, P9, P11 and P13. A letter splits
// the shapes of one audit row (P5b, P9b, P20a to P20g, P30a, P30b). S1 to S6 are this suite's
// own: the rows that the audit did not probe (C2, C5, C7, the max half of C9, the exported
// constants of the Not gaps list) and the max reduction of the KLT maximum. K1 to K6 are this
// suite's own too: the loops of src/mvs/gpu/cloud-knn.shade.ts. K3b and K6b are neighbours of K3
// and K6.
//
// Rows not probed. C4 (a proved loop cut into timed chunks): on the CPU tier a chunk is the whole
// range by design (surface §65), so only a WebGPU device can observe a cut. C10 to C13 (the
// timestamp switch, the loss record, the chunk-rate epoch, the profiling scope): each needs a
// WebGPU device or asks for an API that does not exist yet, so a probe would only test an absence.
//
// The proof rule of a TS8070. The warning carries no rule number. Its first sentence is worded per
// proof rule (surface §65, the table after "A loop the proof refuses runs on the CPU"), and
// RULE_WORDING maps those words to R1 to R6. A probe asserts the code, the line of the loop and
// the rule, not the full text.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build } from 'esbuild';
import { MARK, SOURCES } from './compiler-probes.sources.mjs';

const root = path.resolve(import.meta.dirname, '../..');

/** The typeshade commit whose verdicts the probes hold. */
const MEASURED_AT = '725b2969a9cac37400980c2c0058680368407602';
const AT = `typeshade ${MEASURED_AT.slice(0, 8)}`;

/**
 * Bundle an ES module that imports `typeshade` and load it, as the other compiler tests do. The
 * package ships TypeScript sources, which Node does not strip under node_modules.
 */
async function load(contents) {
  const bundle = await build({
    stdin: { contents, resolveDir: root },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const m = { exports: {} };
  new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(
    createRequire(import.meta.url),
    m,
    m.exports,
  );
  return m.exports;
}

let compilerBundle;
/** `compile()`, the oracle, and `hostFace()`: the host face that the Vite plugin writes (§64). */
const compiler = () =>
  (compilerBundle ??= load(
    "export { compile, compileModule } from 'typeshade';\n" +
      "export { hostFace } from 'typeshade/language-service';",
  ));

const compiled = new Map();
/** The compile result and the host face of the probe source `name`, made once. */
function compileProbe(name) {
  if (!compiled.has(name))
    compiled.set(
      name,
      compiler().then(({ compile, hostFace }) => {
        const source = SOURCES[name];
        const fileName = `${name}.shade.ts`;
        return {
          source,
          result: compile(source, { fileName }),
          face: hostFace(source, { fileName }),
        };
      }),
    );
  return compiled.get(name);
}

/** The one-based line that MARK ends in `source`. */
function markedLine(source) {
  const at = source.split('\n').findIndex((line) => line.trimEnd().endsWith(MARK));
  assert.notEqual(at, -1, 'the probe source marks no line');
  return at + 1;
}

const kernelOf = (face, name) => face.exports?.find((e) => e.kind === 'kernel' && e.name === name);
const codesOf = (diagnostics) => diagnostics.map((d) => d.code);
const listed = (diagnostics) =>
  diagnostics.map((d) => `${d.code} on line ${d.line}: ${d.message}`).join('; ') || 'none';

/**
 * The words of each proof rule in the first sentence of a TS8070 warning: the table of surface
 * §65, and the other refusals that `kernel-loops.ts` words.
 */
const RULE_WORDING = {
  R1: /is a while loop|does not step through a range of indices/,
  R2: /returns from inside it|breaks out of it/,
  R3: /which the next iteration reads|two iterations can share|also pushes onto it/,
  R4: /another iteration (writes|stores)|the loop pushes onto|value push returns|the loop adds to|order the iterations run in/,
  R5: /calls "[^"]*", which writes/,
  R6: /would print in another order|is not a workgroup/,
};

/** The one rule whose words `message` has, or undefined. */
function ruleOf(message) {
  const rules = Object.keys(RULE_WORDING).filter((rule) => RULE_WORDING[rule].test(message));
  return rules.length === 1 ? rules[0] : undefined;
}

// The verdicts.

/** A GPU loop on WebGPU, on WebGL2 when named, and on the CPU tier as every kernel's fallback. */
const gpu = (...tiers) => ({ kind: 'gpu', webgl2: tiers.includes('webgl2') });
/** The CPU tier only, with a TS8070 warning on the marked loop for `rule`. */
const cpuWarned = (rule) => ({ kind: 'cpu', code: 'TS8070', rule });
/** The CPU tier only, with no diagnostic. */
const cpuSilent = { kind: 'cpu' };
/** A compile error on the marked line: no WGSL and no host face. */
const refused = (code) => ({ kind: 'refused', code });

function verdictText(v) {
  if (v.kind === 'gpu')
    return v.webgl2 ? 'GPU loop on WebGPU and WebGL2' : 'GPU loop on WebGPU, not WebGL2';
  if (v.kind === 'refused') return `refused, ${v.code} error`;
  return v.code === undefined
    ? 'CPU tier only, no diagnostic'
    : `CPU tier only, ${v.code} ${v.rule}`;
}

/** The message of a failed assertion: what changed, the verdict held, and what the change means. */
const says =
  (p) =>
  (what, signal = p.signal) =>
    `${p.id} (${p.row}; ${p.shape}): ${what.replace(/\.$/, '')}. Held at ${AT}: ${p.held}. ` +
    `If this changed: ${signal}.`;

/** Assert the verdict of probe `p` on its compile result and its host face. */
function holdVerdict(p, { source, result, face }) {
  const say = says(p);
  const v = p.verdict;
  const codes = codesOf(result.diagnostics);
  // The build reads the host face, and the tests read compile(): both must give one verdict.
  assert.deepEqual(codesOf(face.diagnostics), codes, say('hostFace() and compile() disagree'));
  if (v.kind === 'refused') {
    assert.deepEqual(codes, [v.code], say(`the diagnostics are ${listed(result.diagnostics)}`));
    const [d] = result.diagnostics;
    assert.equal(d.category, 'error', say(`${v.code} is a ${d.category} now`));
    assert.equal(d.line, markedLine(source), say(`${v.code} is on line ${d.line} now`));
    assert.equal(result.wgsl, undefined, say('the module emits WGSL now'));
    assert.equal(face.exports, undefined, say('the host face has exports now'));
    return;
  }
  assert.notEqual(
    result.wgsl,
    undefined,
    say(`the module does not compile: ${listed(result.diagnostics)}`),
  );
  const kernel = kernelOf(face, p.fn);
  assert.ok(kernel, say(`the host face has no kernel function "${p.fn}"`));
  const { gpu: onGpu, gl, noGpu, noWebgl2 } = kernel.face;
  if (v.kind === 'gpu') {
    assert.deepEqual(codes, [], say(`the diagnostics are ${listed(result.diagnostics)}`));
    assert.ok(onGpu, say(`it runs on the CPU tier now (${noGpu})`));
    assert.equal(
      gl !== undefined,
      v.webgl2,
      say(
        `its WebGL2 tier changed (${noWebgl2 ?? 'it runs there now'})`,
        'no TypeCV step opens, since TypeCV opens only webgpu and cpu; update the expected tiers',
      ),
    );
    return;
  }
  assert.equal(onGpu, undefined, say('it lowers to a GPU loop now'));
  assert.equal(gl, undefined, say('it runs on WebGL2 now'));
  if (v.code === undefined) {
    assert.deepEqual(codes, [], say(`a diagnostic comes with it: ${listed(result.diagnostics)}`));
    return;
  }
  assert.deepEqual(codes, [v.code], say(`the diagnostics are ${listed(result.diagnostics)}`));
  const [d] = result.diagnostics;
  assert.equal(d.category, 'warning', say(`${v.code} is a ${d.category} now`));
  assert.equal(d.line, markedLine(source), say(`${v.code} is on line ${d.line}, not on the loop`));
  const rule = ruleOf(d.message);
  assert.equal(
    rule,
    v.rule,
    say(`${v.code} reads as ${rule ?? 'no single rule'} now: ${d.message}`),
  );
}

// The signals: what a change of verdict means for TypeCV.

const C1_CLOSED =
  'C1 closed: a helper may take a storage array, so phase 10 step C1 (one BA module, after R03) may open';
const C6_CLOSED = (form) =>
  `C6 closed for ${form}: the maps that need it may convert in phase 10 step "C6 and C7" once C7 is closed too (S2)`;
const C9_CLOSED = (op) =>
  `C9 closed (typeshade/typeshade#761): an integer ${op} scatter lowers, which no step of section 5 needs by the kernels' source (the KLT maximum is a reduction, S6; the Hough vote a += scatter, P20a)`;
const REGRESSION = (what) => `a compiler regression: ${what}; report it to TypeShade`;

// The probes of the audit (section 0.1).

const PROBES = [
  {
    id: 'P1',
    row: 'C8',
    shape: 'rgbaToGray as a loop over offset..offset+count',
    source: 'p1',
    fn: 'rgbaToGrayK',
    verdict: gpu('webgpu', 'webgl2'),
    also: 'the call takes no workgroup count and no indirect count',
    signal: REGRESSION('the form of the phase 9 map conversions no longer lowers'),
    more({ face }, say) {
      const c8 =
        'C8 closed: a kernel call may take its trip count from a Resident, so the 12 indirect pair-model and PnP entries could become kernel functions after an amendment of section 3.6';
      const k = kernelOf(face, 'rgbaToGrayK');
      assert.deepEqual(
        k.face.params.map((q) => q.name),
        ['offset', 'count', 'rgba', 'gray'],
        say('the call takes more than the authored parameters', c8),
      );
      const declarations = face.view.split('\n').filter((l) => l.includes('function rgbaToGrayK('));
      assert.ok(declarations.length > 0, say('the host view declares no rgbaToGrayK', c8));
      for (const line of declarations)
        assert.doesNotMatch(line, /indirect|workgroups/, say('the call takes a count now', c8));
    },
  },
  {
    id: 'P2',
    row: 'C1',
    shape: 'a loop calls a helper that reads a module storage binding',
    source: 'p2',
    fn: 'mapK',
    verdict: cpuSilent,
    also: 'the reason names the binding "lut"',
    signal:
      'a kernel loop may reach a module binding now; re-read C1, since a helper that reads a binding by name could then serve kernel functions',
    more({ face }, say) {
      const why = kernelOf(face, 'mapK').face.noGpu ?? '';
      assert.match(why, /"lut"/, say(`the CPU reason no longer names "lut": ${why}`));
    },
  },
  {
    id: 'P3',
    row: 'C1',
    shape: "a local arrow function captures the kernel's array<f32>",
    source: 'p3',
    fn: 'blurK',
    verdict: gpu('webgpu'),
    signal: REGRESSION(
      'a local function over the kernel array, the in-function workaround of C1, no longer lowers',
    ),
  },
  {
    id: 'P4',
    row: 'C1',
    shape: 'a plain helper takes array<f32>',
    source: 'p4',
    verdict: refused('TS8020'),
    signal: C1_CLOSED,
  },
  {
    id: 'P5',
    row: 'C3',
    shape: 'a one-iteration loop writes state[0]',
    source: 'p5',
    fn: 'acceptA',
    verdict: cpuWarned('R3'),
    signal:
      'the proof accepts a constant index in a one-iteration loop, so the single-invocation steps of section 3.6 gain a direct kernel form; their device-side control still waits for C3',
  },
  {
    id: 'P5b',
    row: 'C3',
    shape: 'the same step written at state[i * 2 + c]',
    source: 'p5b',
    fn: 'acceptB',
    verdict: gpu('webgpu'),
    signal: REGRESSION('the affine form of a single-invocation step no longer lowers'),
  },
  {
    id: 'P6',
    row: 'C3',
    shape: 'the trip count is read from counts[0] before the loop',
    source: 'p6',
    fn: 'boundFromArray',
    verdict: cpuSilent,
    signal:
      'C3 closed for loop bounds: a trip count read from a device array lowers, so phase 10 step C3 may open, and C8 (which shares the lowering) may follow',
  },
  {
    id: 'P7',
    row: 'C3',
    shape: 'a guard if (i >= counts[0]) continue inside the loop',
    source: 'p7',
    fn: 'guardInLoop',
    verdict: gpu('webgpu', 'webgl2'),
    signal: REGRESSION(
      'the guard form, which runs over the capacity and skips past a device count, no longer lowers',
    ),
  },
  {
    id: 'P8',
    row: 'Not gaps',
    shape: 'workgroup memory and workgroupBarrier() in the loop',
    source: 'p8',
    fn: 'tiled',
    verdict: cpuWarned('R6'),
    also: 'the remedy names a @compute entry',
    signal:
      "the proof accepts workgroup memory or a barrier in a kernel loop, so section 3.6's list of what only an entry provides needs a review",
    more({ result }, say) {
      assert.match(
        result.diagnostics[0].message,
        /@compute/,
        say('the remedy names no @compute entry'),
      );
    },
  },
  {
    id: 'P9',
    row: 'C3',
    shape: 'out.push(i) with return out.length',
    source: 'p9',
    fn: 'strong',
    verdict: gpu('webgpu'),
    also: 'the push runs as count, scan and store',
    signal: REGRESSION('push no longer lowers, which phase 10 step C3 (the compactions) needs'),
    more({ face }, say) {
      const [loop] = kernelOf(face, 'strong').face.gpu.loops;
      assert.ok(
        loop.push?.count && loop.push?.scan,
        say(
          'the push has no count and scan steps',
          'the push lowering changed (surface §65); check that it keeps the sequential order that phase 10 step C3 relies on',
        ),
      );
    },
  },
  {
    id: 'P10',
    row: 'Not gaps',
    shape: 'nested 2D loops write out[y * w + x]',
    source: 'p10',
    fn: 'grid',
    verdict: gpu('webgpu'),
    signal: REGRESSION('a nested 2D loop no longer lowers, against the Not gaps list of section 6'),
  },
  {
    id: 'P16',
    row: 'C6',
    shape: 'a record out[i * 4 + k] over a nested k < 4',
    source: 'p16',
    fn: 'records',
    verdict: gpu('webgpu'),
    signal: REGRESSION(
      'the record form i * W + k no longer lowers, which the planar depth and SLIC conversions need',
    ),
  },
  {
    id: 'P9b',
    row: 'C6',
    shape: 'a nested record at (i * slots + s) * 4 + c',
    source: 'p9b',
    fn: 'records2',
    verdict: cpuWarned('R3'),
    signal: C6_CLOSED('a nested affine record'),
  },
  {
    id: 'P11',
    row: 'Not gaps',
    shape: 'an array<f64> map',
    source: 'p11',
    fn: 'scaleD',
    verdict: gpu('webgpu'),
    also: 'the f64 is emulated, under both flavors',
    signal: REGRESSION('f64 in a kernel function no longer lowers, against the Not gaps list'),
    more({ result, face }, say) {
      const emulation =
        'the f64 emulation of kernel functions changed (surface §39); re-read audit section 5 before an f64 conversion';
      const { gpu: g } = kernelOf(face, 'scaleD').face;
      assert.ok(
        g.wgslInteger !== undefined && g.guard !== undefined,
        say('one flavor of the emulated double is missing', emulation),
      );
      assert.ok(
        result.determinism.some(
          (e) => e.kind === 'emulated' && e.elem === 'f64' && e.where.includes('scaleD'),
        ),
        say('the determinism report lists no emulated f64 operation in scaleD', emulation),
      );
    },
  },
  {
    id: 'P18',
    row: 'C9',
    shape: 'best[keys[i]] = min(best[keys[i]], v) on u32',
    source: 'p18',
    fn: 'minScatter',
    verdict: cpuWarned('R3'),
    signal: C9_CLOSED('min'),
  },
  {
    id: 'P19',
    row: 'none: the TSDF integrate, audit section 3',
    shape: 'an f32 scatter sums[keys[i]] += v',
    source: 'p19',
    fn: 'addScatter',
    verdict: cpuWarned('R3'),
    also: 'the determinism report lists its tree order',
    signal:
      'an f32 scatter lowers to the GPU; no row of section 6 asks for it, and the fixed-point TSDF integrate could use it only by an owner decision on the bits (audit section 5)',
    more({ result }, say) {
      assert.ok(
        result.determinism.some(
          (e) => e.kind === 'order' && e.elem === 'f32' && e.where.includes('addScatter'),
        ),
        say('the determinism report lists no order row for the f32 scatter'),
      );
    },
  },
  {
    id: 'P13',
    row: 'Not gaps',
    shape: 'a kernel function calls a kernel function',
    source: 'p13',
    verdict: refused('TS8099'),
    signal:
      'a kernel function may call another now, so the Not gaps entry of section 6 (fuse composes the calls on the host) needs a review',
  },
  {
    id: 'P14',
    row: 'phase 9',
    shape: '@workgroup_size([16]) on a kernel function',
    source: 'p14',
    fn: 'scaled',
    verdict: gpu('webgpu', 'webgl2'),
    also: 'its loop runs at workgroup 16',
    signal: REGRESSION("a converted map can no longer keep its entry's workgroup size"),
    more({ face }, say) {
      const [loop] = kernelOf(face, 'scaled').face.gpu.loops;
      assert.equal(loop.wg, 16, say(`the loop runs at workgroup ${loop.wg}`));
    },
  },
  {
    id: 'P15',
    row: 'phase 9',
    shape: 'a struct control parameter',
    source: 'p15',
    fn: 'gainK',
    verdict: gpu('webgpu'),
    signal: REGRESSION(
      'a struct control parameter no longer lowers, so converted maps would pass scalars',
    ),
  },
  {
    id: 'P20a',
    row: 'Not gaps',
    shape: 'an integer scatter bins[keys[i]] += 1',
    source: 'p20a',
    fn: 'countScatter',
    verdict: gpu('webgpu'),
    also: 'an atomicAdd on array<atomic<u32>>',
    signal: REGRESSION(
      'the integer scatter no longer lowers, which the Hough vote and the CLAHE histogram need',
    ),
    more({ face }, say) {
      const atomics =
        'the integer scatter lowers in another way now (surface §65); re-read it before the Hough vote conversion';
      const { wgsl } = kernelOf(face, 'countScatter').face.gpu;
      assert.match(
        wgsl,
        /\bbins\s*:\s*array<atomic<u32>>/,
        say('bins is no atomic array', atomics),
      );
      assert.match(wgsl, /\batomicAdd\(/, say('the WGSL has no atomicAdd', atomics));
    },
  },
  {
    id: 'P20b',
    row: 'C6',
    shape: 'one write at out[stride * i]',
    source: 'p20b',
    fn: 'strided',
    verdict: gpu('webgpu'),
    signal: REGRESSION('a write at a loop-invariant stride no longer lowers'),
  },
  {
    id: 'P20c',
    row: 'C6',
    shape: 'writes at out[stride * i] and out[stride * i + 1]',
    source: 'p20c',
    fn: 'stridedPlus',
    verdict: cpuWarned('R3'),
    signal: C6_CLOSED('fields at a runtime stride'),
  },
  {
    id: 'P20d',
    row: 'C6',
    shape: 'a write at out[base + i]',
    source: 'p20d',
    fn: 'offsetWrite',
    verdict: cpuWarned('R3'),
    signal: C6_CLOSED("an injective offset (SIFT's blurOut[outOffset + i] converts as written)"),
  },
  {
    id: 'P20e',
    row: 'phase 10: exact reductions',
    shape: 'xs.findIndex((x) => x > 0.5)',
    source: 'p20e',
    fn: 'argBest',
    verdict: gpu('webgpu'),
    also: 'a min reduction of the index',
    signal: REGRESSION('findIndex no longer lowers to a reduction'),
    more({ face }, say) {
      const [loop] = kernelOf(face, 'argBest').face.gpu.loops;
      assert.deepEqual(
        loop.reduce?.vars.map((v) => [v.op, v.scalar]),
        [['min', 'i32']],
        say('the loop is no min reduction of an i32 index'),
      );
    },
  },
  {
    id: 'P20f',
    row: 'phase 10: floating-point reductions, an owner decision',
    shape: 'xs.reduce((a, x) => a + x * x, 0.)',
    source: 'p20f',
    fn: 'sumSq',
    verdict: gpu('webgpu'),
    also: 'an f32 + reduction in the tree order',
    signal: REGRESSION(
      'a floating-point reduce no longer lowers, or its order row is gone; re-read audit section 5',
    ),
    more({ result, face }, say) {
      const [loop] = kernelOf(face, 'sumSq').face.gpu.loops;
      assert.deepEqual(
        loop.reduce?.vars.map((v) => [v.op, v.scalar]),
        [['+', 'f32']],
        say('the loop is no + reduction of an f32'),
      );
      assert.ok(
        result.determinism.some(
          (e) => e.kind === 'order' && e.elem === 'f32' && e.where.includes('sumSq'),
        ),
        say('the determinism report lists no order row for the sum'),
      );
    },
  },
  {
    id: 'P20g',
    row: 'C9',
    shape: 'const k = keys[i]; best[k] = min(best[k], v) on u32',
    source: 'p20g',
    fn: 'minScatter2',
    verdict: cpuWarned('R3'),
    signal: C9_CLOSED('min'),
  },
  {
    id: 'P30a',
    row: 'phase 10: inverse depth, after R03',
    shape: 'array<S> of f64 fields in and out, with continue',
    source: 'p30a',
    fn: 'linearizeK',
    verdict: gpu('webgpu'),
    signal: REGRESSION('the struct-array form of the inverse-depth conversion no longer lowers'),
  },
  {
    id: 'P30b',
    row: 'C6',
    shape: 'a record written in two forms in one loop',
    source: 'p30b',
    fn: 'recordWrites',
    verdict: cpuWarned('R3'),
    signal: C6_CLOSED('two record forms in one loop (the 23-word inverse-depth record)'),
  },
];

test(`the pin: the probes hold the verdicts of ${AT}, the compiler package.json pins`, async () => {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
  const pin = /\/archive\/([a-f0-9]{40})\.tar\.gz$/.exec(pkg.devDependencies.typeshade)?.[1];
  assert.equal(
    pin,
    MEASURED_AT,
    `package.json pins typeshade ${pin}, and the probes hold the verdicts of ${MEASURED_AT}. ` +
      'Read each failing probe as its message says, then set MEASURED_AT to the new pin in the ' +
      'same pull request, which cites the compiler change and the TypeCV rows (the header).',
  );
});

/** Register the test of probe `p`: its verdict, then its further checks. */
function probeTest(p) {
  p.held = p.also === undefined ? verdictText(p.verdict) : `${verdictText(p.verdict)}; ${p.also}`;
  test(`${p.id} (${p.row}) ${p.shape}: ${p.held}`, async () => {
    const c = await compileProbe(p.source);
    holdVerdict(p, c);
    await p.more?.(c, says(p));
  });
}

for (const p of PROBES) probeTest(p);

// The rows that the audit did not probe.

const S1 = {
  id: 'S1',
  row: 'C2',
  shape: 'a @compute entry with workgroup memory and workgroupBarrier()',
  held: 'the host CPU tier refuses it with a TypeError (Rule 8.24), the oracle runs it in passes',
  signal:
    'C2 closed: the host CPU tier runs barrier entries in passes, so phase 10 step C2 (the 18 oracle twins) may open once item 10 is decided yes, the 12 sparse twins after R03',
};
test(`${S1.id} (${S1.row}) ${S1.shape}: ${S1.held}`, async () => {
  const say = says(S1);
  const { compileModule } = await compiler();
  const { result, face } = await compileProbe('s1');
  const diagnostics = `the diagnostics are ${listed(result.diagnostics)}`;
  assert.deepEqual(codesOf(result.diagnostics), [], say(diagnostics));
  const entry = face.exports.find((e) => e.name === 'blockSum');
  const twin = face.exports.find((e) => e.name === 'blockSumOracle');
  assert.equal(entry?.kind, 'compute', say('blockSum is no compute entry in the host face'));
  assert.match(entry.entry.barrier ?? '', /workgroupBarrier\(\)/, say('the face names no barrier'));
  assert.equal(twin?.entry.barrier, undefined, say('the barrier-free twin reaches a barrier'));

  // The generated module, as a bundle ships it, with the CPU tier made the only tier.
  const host = await load(`${face.code}\nexport { configure } from 'typeshade/runtime';`);
  host.configure({ prefer: ['cpu'] });
  const xs = Float32Array.from({ length: 130 }, (_, i) => i + 1);
  const expected = [2080, 6176, 259];
  const twinSums = new Float32Array(3);
  await host.blockSumOracle({ xs, sums: twinSums }, 3);
  assert.deepEqual(
    Array.from(twinSums),
    expected,
    say('the twin gives other sums on the CPU tier'),
  );
  await assert.rejects(
    async () => host.blockSum({ xs, sums: new Float32Array(3) }, 3),
    (e) => e instanceof TypeError && /workgroupBarrier/.test(e.message),
    say('the host CPU tier runs the barrier entry now, or refuses it in other terms'),
  );

  // The compiler's oracle runs the same entry in passes (surface §25): the gap is the host tier's.
  const oracle = compileModule(result.module);
  const oracleSums = new Float32Array(3);
  oracle.setBinding('xs', xs);
  oracle.setBinding('sums', oracleSums);
  const report = oracle.dispatch('blockSum', 3);
  assert.deepEqual(Array.from(oracleSums), expected, say('the oracle gives other sums'));
  assert.ok(report.barrierPhases > 0, say('the oracle ran no barrier phase'));
});

/** The body of WGSL function `name`, between its braces. */
function entryBody(wgsl, name) {
  const start = wgsl.indexOf(`fn ${name}(`);
  if (start < 0) return undefined;
  const open = wgsl.indexOf('{', start);
  for (let i = open, depth = 0; i < wgsl.length; i++) {
    if (wgsl[i] === '{') depth++;
    else if (wgsl[i] === '}' && --depth === 0) return wgsl.slice(open + 1, i);
  }
  return undefined;
}

const S2 = {
  id: 'S2',
  row: 'C7',
  shape: "the WGSL of P1's converted map",
  held: 'the loop entry returns past the trip count before the iteration (kernel-lower.ts:756-767)',
  signal:
    'C7 closed: the map scaffold no longer returns early, so phase 10 step "C6 and C7" may open for the entries that phase 9 left (SLIC, mesh, planarFit, the KLT maximum, the Hough vote and collect), with a hardware run while A1 is open, and triangulateTracks may leave the entries of section 3.6',
};
test(`${S2.id} (${S2.row}) ${S2.shape}: ${S2.held}`, async () => {
  const say = says(S2);
  const { face } = await compileProbe('p1');
  const lowered = kernelOf(face, 'rgbaToGrayK')?.face.gpu;
  assert.ok(lowered, say('P1 no longer lowers, so no map entry is left to read'));
  const [loop] = lowered.loops;
  const body = entryBody(lowered.wgsl, loop.entry);
  assert.ok(body, say(`the WGSL has no entry ${loop.entry}`));
  const early = /if\s*\(([^{}]*)\)\s*\{\s*return;\s*\}/.exec(body);
  const store = body.search(/\bgray\s*\[/);
  assert.ok(
    early !== null && />=/.test(early[1]) && /\b_n\b/.test(early[1]) && early.index < store,
    say('the entry no longer returns past the trip count before its first store'),
  );
});

const S3 = {
  id: 'S3',
  row: 'C9',
  shape: 'const k = keys[i]; best[k] = max(best[k], v) on u32',
  source: 's3',
  fn: 'maxScatter',
  verdict: cpuWarned('R3'),
  signal: C9_CLOSED('max'),
};
probeTest(S3);

const S4 = {
  id: 'S4',
  row: 'Not gaps: surface §64',
  shape: 'exported u32 and f32 constants (the Hough and TSDF copies)',
  held: 'the host face exports each, and the generated module gives the values the hosts copy',
  signal:
    "a host's import of a shader constant no longer resolves or gives another number, so the constant step of phase 9 (hough.shade.ts, tsdf.shade.ts) needs a record first",
};
test(`${S4.id} (${S4.row}) ${S4.shape}: ${S4.held}`, async () => {
  const say = says(S4);
  const { result, face } = await compileProbe('s4');
  const diagnostics = `the diagnostics are ${listed(result.diagnostics)}`;
  assert.deepEqual(codesOf(result.diagnostics), [], say(diagnostics));
  const types = Object.fromEntries(
    face.exports.filter((e) => e.kind === 'const').map((e) => [e.name, e.type.t]),
  );
  assert.deepEqual(
    types,
    { MAX_SEGMENTS: 'u32', SEGMENT_STRIDE: 'u32', TIE: 'f32', EMPTY: 'u32' },
    say('the host face exports other constants'),
  );
  // The values of hough.ts (lines 108-110) and tsdf.ts (line 121): TIE stays the double 1e-5.
  const host = await load(face.code);
  assert.deepEqual(
    [host.MAX_SEGMENTS, host.SEGMENT_STRIDE, host.TIE, host.EMPTY],
    [16, 65, 1e-5, 0xffffffff],
    say('the host reads other values'),
  );
});

const S5 = {
  id: 'S5',
  row: 'C5',
  shape: 'Resident.write(value, { offset }) on the CPU tier',
  held: 'the write replaces the whole array, and the offset is ignored',
  signal:
    'C5 closed: a partial write lands at its offset, so mvsCopyF32, mvsCopyU32 and their staging buffers could go, in a step that section 5 does not schedule yet (dense, phase 11)',
};
test(`${S5.id} (${S5.row}) ${S5.shape}: ${S5.held}`, async () => {
  const say = says(S5);
  const { resident } = await load("export { resident } from 'typeshade/runtime';");
  const r = resident(new Float32Array([1, 2, 3, 4]));
  r.write(new Float32Array([9]), { offset: 2 });
  assert.deepEqual(Array.from(await r.read()), [9], say('the write kept part of the old array'));
  r.destroy();
});

const S6 = {
  id: 'S6',
  row: 'phase 10: the KLT maximum, after C7',
  shape: 'best = max(best, xs[i]) from let best = 0.',
  source: 's6',
  fn: 'maxReduce',
  verdict: gpu('webgpu'),
  also: 'an f32 max reduction that keeps the 0 floor on the CPU tier',
  signal: REGRESSION(
    'the max reduction no longer lowers or loses its 0 floor, which the KLT maximum step and its claim of no bit change (audit section 5) need',
  ),
  async more({ face }, say) {
    const [loop] = kernelOf(face, 'maxReduce').face.gpu.loops;
    assert.deepEqual(
      loop.reduce?.vars.map((v) => [v.name, v.op, v.scalar]),
      [['best', 'max', 'f32']],
      say('the loop is no max reduction of best'),
    );
    const host = await load(`${face.code}\nexport { configure } from 'typeshade/runtime';`);
    host.configure({ prefer: ['cpu'] });
    assert.equal(await host.maxReduce(new Float32Array([-3, -1, -2])), 0, say('no 0 floor'));
    assert.equal(await host.maxReduce(new Float32Array([1, 5, 2])), 5, say('another maximum'));
  },
};
probeTest(S6);

// The loops of src/mvs/gpu/cloud-knn.shade.ts: what a refactor of knnMeanDistance may use (K1 to
// K5), and the write that keeps it an entry (K6). Both CPU tiers: the oracle, which runs an entry
// in the phased order of the WebGL2 tier (change 0054), and the host CPU tier of the generated
// module.

/** The points of the K probes, on a line. */
const POINTS = [0, 1, 2, 5, 9];
/** Each point's three smallest squared distances to the other points, summed. */
const SUMS = [30, 18, 14, 41, 129];
/** `100 * r + j` for the first point `j` within the smallest ring `r` of 1 to 4. */
const RINGS = [101, 100, 101, 302, 403];
/** The visit order of K6 and K6b: lane `l` takes point `VISIT[l]`. */
const VISIT = [3, 0, 4, 1, 2];

const sumBindings = () => ({ xs: Float32Array.from(POINTS), out: new Float32Array(5) });
const ringBindings = () => ({ xs: Float32Array.from(POINTS), out: new Uint32Array(5) });

/**
 * Assert the verdict of K entry `p`. It compiles with no diagnostic. Its WGSL and the GLSL of its
 * WebGL2 pass program have the shapes `p.wgsl` and `p.glsl`. The determinism report is empty, and
 * both CPU tiers write `p.expect`.
 */
async function holdEntry(p, { result, face }) {
  const say = says(p);
  const codes = codesOf(result.diagnostics);
  assert.deepEqual(codesOf(face.diagnostics), codes, say('hostFace() and compile() disagree'));
  assert.deepEqual(codes, [], say(`the diagnostics are ${listed(result.diagnostics)}`));
  for (const shape of p.wgsl) assert.match(result.wgsl, shape, say(`the WGSL lacks ${shape}`));
  const entry = face.exports.find((e) => e.kind === 'compute' && e.name === p.fn)?.entry;
  assert.ok(entry, say(`the host face has no compute entry "${p.fn}"`));
  assert.equal(entry.barrier, undefined, say(`the entry reaches ${entry.barrier}`));
  assert.equal(entry.noCpu, undefined, say(`the host CPU tier refuses it: ${entry.noCpu}`));
  assert.ok(entry.gl, say(`it has no WebGL2 pass program: ${entry.noGl}`));
  assert.deepEqual(
    entry.gl.cuts,
    {},
    say('its pass program stops at a barrier, an atomic or a log'),
  );
  for (const shape of p.glsl) assert.match(entry.gl.vertex, shape, say(`the GLSL lacks ${shape}`));
  assert.deepEqual(result.determinism, [], say('the determinism report lists an operation'));

  const { compileModule } = await compiler();
  const oracle = compileModule(result.module);
  const bindings = p.bindings();
  for (const [name, value] of Object.entries(bindings)) oracle.setBinding(name, value);
  oracle.dispatch(p.fn, 1);
  assert.deepEqual(Array.from(bindings.out), p.expect, say('the oracle writes other values'));

  // The generated module, as a bundle ships it, with the CPU tier made the only tier.
  const host = await load(`${face.code}\nexport { configure } from 'typeshade/runtime';`);
  host.configure({ prefer: ['cpu'] });
  const own = p.bindings();
  await host[p.fn](own, 1);
  assert.deepEqual(Array.from(own.out), p.expect, say('the host CPU tier writes other values'));
}

/** Register the test of K entry `p`. */
function entryTest(p) {
  test(`${p.id} (${p.row}) ${p.shape}: ${p.held}`, async () => {
    await holdEntry(p, await compileProbe(p.source));
  });
}

/** What kernel function `name` of `face` writes for POINTS and VISIT on the CPU tier, k = 3. */
async function onCpuTier(face, name) {
  const host = await load(`${face.code}\nexport { configure } from 'typeshade/runtime';`);
  host.configure({ prefer: ['cpu'] });
  const sums = new Float32Array(5);
  await host[name](3, 5, Float32Array.from(POINTS), Uint32Array.from(VISIT), sums);
  return Array.from(sums);
}

const K1 = {
  id: 'K1',
  row: 'none: the insertion in cloud-knn.shade.ts',
  shape: 'a helper takes a local array<f32, 32> as @inout',
  source: 'k1',
  fn: 'insertByRef',
  held: 'compiles; WGSL passes &best to a ptr<function> parameter, the WebGL2 pass program to an inout one, and both CPU tiers write the sums',
  wgsl: [
    /fn insertSorted\(best: ptr<function, array<f32, 32>>, k: u32, d2: f32\)/,
    /\binsertSorted\(&best, 3u, /,
  ],
  glsl: [
    /void insertSorted\(inout float\[32\] best, uint k, float d2\)/,
    /\binsertSorted\(\w+, 3u, /,
  ],
  bindings: sumBindings,
  expect: SUMS,
  signal: REGRESSION(
    'a helper can no longer take the list by reference (Rule 8.25), so the insertion of the k-NN stays written twice',
  ),
};
entryTest(K1);

const K2 = {
  id: 'K2',
  row: 'none: the bucket scan in cloud-knn.shade.ts',
  shape: 'a helper takes the list as @inout, reads two bindings by name and hands the list on',
  source: 'k2',
  fn: 'scanByRef',
  held: 'compiles; both helpers keep a ptr<function> parameter in WGSL and an inout one in GLSL, and both CPU tiers write the sums',
  wgsl: [
    /fn scanBucket\(best: ptr<function, array<f32, 32>>, k: u32, i: u32, bucket: u32\)/,
    /\binsertSorted\(best, k, /,
    /\bscanBucket\(&best, 3u, /,
  ],
  glsl: [
    /void scanBucket\(inout float\[32\] best, uint k, uint i, uint bucket\)/,
    /\binsertSorted\(best, k, /,
  ],
  bindings: () => ({ ...sumBindings(), bucketStart: Uint32Array.from([0, 2, 5]) }),
  expect: SUMS,
  signal: REGRESSION(
    'a helper can no longer read the bindings and hand the list on by reference, so the bucket scan of the k-NN stays in the entry',
  ),
};
entryTest(K2);

const K3 = {
  id: 'K3',
  row: 'none: the ring loop in cloud-knn.shade.ts',
  shape: 'break of an inner loop, and of the outer loop by a flag',
  source: 'k3',
  fn: 'ringBreak',
  held: 'compiles; WGSL keeps both breaks, the WebGL2 pass program keeps the loops, and both CPU tiers write the first ring',
  wgsl: [/\bbreak;[\s\S]*\bbreak;/],
  glsl: [/for \(uint r = 1u; \(r <= 4u\); r = \(r \+ 1u\)\)/, /for \(uint j = 0u; /],
  bindings: ringBindings,
  expect: RINGS,
  signal: REGRESSION(
    'break no longer runs alike on every tier, so the ring loop of the k-NN keeps its done flag',
  ),
};
entryTest(K3);

const K3b = {
  id: 'K3b',
  row: 'none: the ring loop in cloud-knn.shade.ts',
  shape: 'a labelled break of the outer loop',
  source: 'k3b',
  verdict: refused('TS8099'),
  also: 'the remedy names a flag, or a function that returns',
  signal:
    'a labelled break compiles now, which surface §52 refuses; the k-NN could leave its ring loop from the bucket loop',
  more({ result }, say) {
    assert.match(
      result.diagnostics[0].message,
      /with a flag, or hoist the inner loop into a function/,
      say(`the remedy names neither a flag nor a function: ${result.diagnostics[0].message}`),
    );
  },
};
probeTest(K3b);

const K4 = {
  id: 'K4',
  row: 'none: the insertion in cloud-knn.shade.ts',
  shape: 'a while loop with a data-dependent trip count',
  source: 'k4',
  fn: 'insertWhile',
  held: 'compiles; WGSL and GLSL spell it as a for over the counter _w, and both CPU tiers write the sums',
  wgsl: [/for \(var _w\w*: i32 = 0; \(\(s > 0u\) && \(best\[\(s - 1u\)\] > d2\)\);/],
  glsl: [/for \(int _w\w* = 0; \(\(s > 0u\) && \(\w*best\[\(s - 1u\)\] > d2\)\);/],
  bindings: sumBindings,
  expect: SUMS,
  signal: REGRESSION(
    'a while with a data-dependent exit no longer runs alike on every tier (Rule 7.5), so the insertion of the k-NN keeps its 31 steps',
  ),
};
entryTest(K4);

const K5 = {
  id: 'K5',
  row: 'none: the ring loop in cloud-knn.shade.ts',
  shape: 'a return from inside the nested loops of an entry with no barrier',
  source: 'k5',
  fn: 'ringReturn',
  held: 'compiles with no diagnostic; it has a WebGL2 pass program, and both CPU tiers write the first ring',
  wgsl: [/\breturn;/],
  glsl: [],
  bindings: ringBindings,
  expect: RINGS,
  signal: REGRESSION(
    "a return inside the loops of an entry no longer runs alike on every tier; TypeCV's own A1 rule allows it only for a condition of the item's data",
  ),
};
entryTest(K5);

const K6 = {
  id: 'K6',
  row: 'phase 10: k-NN stays an entry',
  shape: 'a kernel function writes sums[visitOrder[lane]]',
  source: 'k6',
  fn: 'knnPermuted',
  verdict: cpuWarned('R3'),
  also: 'the warning names "sums[i]" and the counter "lane"; the CPU tier writes the sums',
  signal:
    'a permuted write lowers now, so knnMeanDistance could become a kernel function with its visit order, a step that section 5 does not schedule (its phase 10 row: C6 does not cover a permutation)',
  async more({ result, face }, say) {
    const { message } = result.diagnostics[0];
    assert.match(message, /"sums\[i\]"/, say(`the warning no longer names "sums[i]": ${message}`));
    assert.match(message, /"lane"/, say(`the remedy no longer names "lane": ${message}`));
    assert.deepEqual(
      await onCpuTier(face, 'knnPermuted'),
      SUMS,
      say('the CPU tier writes other sums'),
    );
  },
};
probeTest(K6);

const K6b = {
  id: 'K6b',
  row: 'phase 10: k-NN stays an entry',
  shape: 'the same function writes sums[lane], and the host applies the visit order',
  source: 'k6b',
  fn: 'knnByLane',
  verdict: gpu('webgpu', 'webgl2'),
  also: 'the CPU tier writes the sums in lane order',
  signal: REGRESSION('the lane-indexed form of the k-NN no longer lowers'),
  async more({ face }, say) {
    assert.deepEqual(
      await onCpuTier(face, 'knnByLane'),
      VISIT.map((i) => SUMS[i]),
      say('the CPU tier writes other sums'),
    );
  },
};
probeTest(K6b);
