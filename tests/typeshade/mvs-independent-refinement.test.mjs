import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build } from 'esbuild';
import {
  REFINEMENT_SENTINEL,
  legacyRefinementSource,
  refinementAnalyticalSource,
  refinementFixture,
  refinementAnalyticalFixture,
  refinementPlaneFixture,
  refinementCalls,
  refinementState,
  bindRefinement,
  runRefinementCalls,
} from './mvs-independent-refinement-cases.mjs';

async function compiler() {
  const b = await build({
    stdin: {
      contents: "export {compile,compileModuleJs} from 'typeshade';",
      resolveDir: path.resolve(import.meta.dirname, '../..'),
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const m = { exports: {} };
  new Function('require', 'module', 'exports', b.outputFiles[0].text)(
    createRequire(import.meta.url),
    m,
    m.exports,
  );
  return m.exports;
}

let prepared;
function modules() {
  return (prepared ??= (async () => {
    const c = await compiler(),
      source = await readFile(new URL('../../src/gpu/mvs.shade.ts', import.meta.url), 'utf8'),
      old = legacyRefinementSource(source),
      parsed = {};
    for (const [name, shader] of Object.entries({
      current: source,
      old,
      analytical: refinementAnalyticalSource(source),
      oldAnalytical: refinementAnalyticalSource(old),
    })) {
      parsed[name] = c.compile(shader);
      assert.deepEqual(
        parsed[name].diagnostics.filter((d) => d.category === 'error'),
        [],
        name,
      );
    }
    return { c, parsed };
  })());
}

function assertExpected(fixture) {
  const b = fixture.bindings,
    depth = b.mvsControl.geometric ? b.mvsGeoDepth : b.mvsDepth;
  assert.equal(depth[fixture.pixel], Math.fround(fixture.expected.depth), fixture.id + ' depth');
  assert.equal(b.mvsNormal[fixture.pixel], fixture.expected.word, fixture.id + ' normal word');
}

test('disabled scalar/cache/streamed search is exactly the frozen e70f70a shader, including omitted CPU flag', async () => {
  const { c, parsed } = await modules();
  let checked = 0,
    validCosts = 0,
    improvements = 0;
  for (const precision of ['f64', 'f32']) {
    const current = c.compileModuleJs(parsed.current.module, { precision }),
      old = c.compileModuleJs(parsed.old.module, { precision });
    for (const geometric of [0, 1])
      for (const flag of [0, undefined])
        for (const mode of ['scalar', 'cached', 'stream'])
          for (const reverse of [false, true]) {
            const fixture = refinementFixture(current, { count: 3, geometric }),
              a = structuredClone(fixture.bindings),
              b = structuredClone(fixture.bindings);
            // Exercise valid propagated neighbor planes as well as random/local candidates.
            for (const values of [a, b]) {
              values.mvsDepth[fixture.pixel + 1] = values.mvsGeoDepth[fixture.pixel + 1] = 3.7;
              values.mvsNormal[fixture.pixel + 1] = fixture.anchorWord;
            }
            if (flag === undefined) delete a.mvsControl.independentRefinement;
            delete b.mvsControl.independentRefinement;
            bindRefinement(current, a);
            assert.ok(a.mvsCameras[12] > 0 && a.mvsCameras[13] > 0, 'positive packed K');
            assert.deepEqual(current.fns.referenceRay(16, 12), [0, 0, 1], 'finite center ray');
            const stats = current.fns.referenceWindow(16, 12),
              initialCost = current.fns.hypothesisCost(
                16,
                12,
                4,
                current.fns.decodeNormal(fixture.anchorWord),
                stats,
              );
            assert.equal(stats.ok, 1, 'valid textured reference window');
            assert.ok(initialCost < 2, 'finite real source evidence, not an invalid-window no-op');
            validCosts++;
            a.mvsCost[fixture.pixel] = b.mvsCost[fixture.pixel] = initialCost;
            const storedInitialCost = a.mvsCost[fixture.pixel];
            const calls = refinementCalls(a, mode, reverse);
            runRefinementCalls(current, a, calls);
            runRefinementCalls(old, b, calls);
            assert.deepEqual(
              refinementState(a),
              refinementState(b),
              `${precision}/${geometric}/${flag}/${mode}`,
            );
            assert.deepEqual(a.mvsPartial, b.mvsPartial);
            if (a.mvsCost[fixture.pixel] < storedInitialCost) improvements++;
            checked++;
          }
  }
  assert.equal(checked, 48);
  assert.equal(validCosts, 48);
  assert.ok(improvements > 0, 'legacy candidates improve real finite hypotheses');
});

test('actual scalar/cache/stream kernels preserve the good component, fixed anchor and first-wins ties', async () => {
  const { c, parsed } = await modules();
  const rows = [];
  for (const precision of ['f64', 'f32']) {
    const m = c.compileModuleJs(parsed.analytical.module, { precision });
    for (const geometric of [0, 1])
      for (const kind of ['depth-only', 'normal-only', 'anchor-order', 'joint-tie', 'anchor-tie']) {
        const outputs = [];
        for (const mode of ['scalar', 'cached', 'stream']) {
          const f = refinementAnalyticalFixture(m, kind, { independentRefinement: 1, geometric });
          if (kind === 'depth-only' || kind === 'normal-only') {
            const target = f.bindings.refinementTarget,
              cost = (d, n) => m.fns.refinementAnalyticalCost(d, n),
              joint = m.fns.streamHypothesis(f.element, 0),
              oldNormal = m.fns.decodeNormal(f.anchorWord),
              newNormal = m.fns.decodeNormal(joint.word),
              anchorCost = cost(f.anchorDepth, oldNormal);
            assert.ok(
              cost(joint.depth, newNormal) > anchorCost,
              'joint worsens the correct component',
            );
            assert.equal(cost(target.depth, target.normal), 0, 'independent component minimum');
          }
          runRefinementCalls(m, f.bindings, refinementCalls(f.bindings, mode));
          assertExpected(f);
          outputs.push(refinementState(f.bindings));
        }
        assert.deepEqual(outputs[1], outputs[0], `${precision}/${geometric}/${kind}: cached`);
        assert.deepEqual(outputs[2], outputs[0], `${precision}/${geometric}/${kind}: streamed`);
        rows.push({ precision, geometric, kind });
      }
  }
  assert.equal(rows.length, 20);
});

test('phase3 uses separate three-candidate rows, keeps partial sentinels and cannot re-anchor during selection', async () => {
  const { c, parsed } = await modules();
  for (const precision of ['f64', 'f32']) {
    const m = c.compileModuleJs(parsed.analytical.module, { precision }),
      f = refinementAnalyticalFixture(m, 'anchor-order', { independentRefinement: 1, count: 3 }),
      b = f.bindings,
      L = b.mvsControl.bestSources;
    for (let k = 0; k < f.count; k++) b.mvsCost[f.pixel + 2 * k] = 1;
    const calls = refinementCalls(b, 'stream').slice(3),
      expected = [];
    bindRefinement(m, b);
    for (let e = f.element; e < f.element + f.count; e++)
      for (let h = 0; h < 3; h++) {
        const hyp = m.fns.streamHypothesis(e, h),
          cost = Math.fround(
            m.fns.refinementAnalyticalCost(hyp.depth, m.fns.decodeNormal(hyp.word)),
          );
        expected.push(...Array(L).fill(cost));
      }
    runRefinementCalls(m, b, calls.slice(0, 2));
    assert.deepEqual([...b.mvsPartial.slice(0, expected.length)], expected);
    assert.ok(b.mvsPartial.slice(expected.length).every((v) => v === REFINEMENT_SENTINEL));
    const before = structuredClone(refinementState(b));
    assert.equal(before[0][f.pixel], f.anchorDepth, 'cost accumulation never updates the anchor');
    runRefinementCalls(m, b, calls.slice(2));
    assertExpected(f);
    assert.equal(b.mvsCost[f.pixel], 0.25);
  }
});

test('independent physical plane renderer and real NCC exhibit depth-only and normal-only winners', async () => {
  const { c, parsed } = await modules();
  const rows = [];
  for (const precision of ['f64', 'f32']) {
    const m = c.compileModuleJs(parsed.current.module, { precision });
    for (const kind of ['depth-only', 'normal-only']) {
      const outputs = [];
      for (const mode of ['scalar', 'cached', 'stream']) {
        const f = refinementPlaneFixture(m, kind, { independentRefinement: 1 });
        assert.ok(
          f.oracle.jointCost > f.oracle.anchorCost,
          `${precision}/${kind}: joint perturbs a correct component`,
        );
        assert.ok(
          f.oracle.independentCost < f.oracle.anchorCost,
          `${precision}/${kind}: independent plane improves NCC`,
        );
        assert.ok(
          f.oracle.randomCost >= f.oracle.anchorCost,
          `${precision}/${kind}: random leaves this anchor`,
        );
        runRefinementCalls(m, f.bindings, refinementCalls(f.bindings, mode));
        assertExpected(f);
        outputs.push(refinementState(f.bindings));
        if (mode === 'scalar') rows.push({ precision, kind, ...f.oracle });
      }
      // Streamed source costs are stored as f32 even in the f64 CPU oracle. Admission/depth and
      // quantized normals are exact; only the aggregated cost may differ by one f32 rounding.
      for (const output of outputs.slice(1)) {
        assert.deepEqual(output.slice(0, 3), outputs[0].slice(0, 3));
        for (let i = 0; i < output[3].length; i++)
          assert.ok(Math.abs(output[3][i] - outputs[0][3][i]) <= 2 ** -23);
      }
    }
  }
  console.log(JSON.stringify({ independentPhysicalPlanes: rows }));
});

test('public CPU depth API forwards the opt-in through both passes and budgeted streamed chunks', async () => {
  const { ShadeDevice, planMvsViews, mvsDepthMapsGpu } = await import('../../dist/internal/gpu.js');
  const { mvsScene } = await import('./mvs-cases.mjs');
  const scene = mvsScene({ width: 33, height: 25, landmarks: 40 }),
    plan = planMvsViews(scene),
    runs = [];
  assert.equal(plan.packed.length, 5);
  for (const options of [
    {},
    { independentRefinement: false },
    { independentRefinement: true },
    { independentRefinement: true, streamGroup: 1 },
    { independentRefinement: true, residentBytes: 40_000, chunkPixels: 17 },
  ]) {
    const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
    try {
      const result = await mvsDepthMapsGpu(device, scene, plan, scene.images, {
        photometricIterations: 1,
        geometricIterations: 1,
        readStages: true,
        ...options,
      });
      assert.equal(device.statistics.liveBuffers, 0);
      assert.ok(result.views.some((v) => v.photometric.cost.some((cost) => cost < 2)));
      runs.push(result);
    } finally {
      await device.dispose();
    }
  }
  assert.deepEqual(runs[1].views, runs[0].views, 'default and explicit false');
  assert.deepEqual(runs[3].views, runs[2].views, 'on window and one-source groups');
  assert.deepEqual(runs[4].views, runs[2].views, 'on budget/partial/range tails');
  assert.equal(runs[4].residency.geometric.mode, 'streamed');
  assert.ok(runs[4].residency.geometric.elements < Math.ceil(33 / 2) * 25);
  assert.ok(runs[4].residency.peakBytes <= 40_000);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const independentRefinement of [0, 1, 'true', null])
      await assert.rejects(
        async () => mvsDepthMapsGpu(device, scene, plan, scene.images, { independentRefinement }),
        /independentRefinement must be a boolean/,
      );
    assert.equal(device.statistics.liveBuffers, 0);
  } finally {
    await device.dispose();
  }
});
