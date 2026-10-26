import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, readdir, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { inBrowser } from './browser.mjs';
import { mvsScene, depthMetrics } from './mvs-cases.mjs';
import {
  ShadeDevice,
  planMvsViews,
  mvsPhotometricGpu,
  decodeMvsNormal,
} from '../../dist/internal/gpu.js';

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const plain = ({ depth, normal, cost }) => ({
  depth: Array.from(depth),
  normal: Array.from(normal),
  cost: Array.from(cost),
});

function agreement(gpu, host, range) {
  const kept = (r, i) => r.cost[i] <= 0.9 && r.depth[i] >= range[0] && r.depth[i] <= range[1];
  const relativeErrors = [],
    partedCosts = [];
  for (let i = 0; i < host.depth.length; i++)
    if (kept(gpu, i) && kept(host, i)) {
      const error = Math.abs(gpu.depth[i] - host.depth[i]) / host.depth[i];
      relativeErrors.push(error);
      if (error > 1e-3) partedCosts.push(Math.abs(gpu.cost[i] - host.cost[i]));
    }
  assert.ok(relativeErrors.length > 0, 'agreement requires shared kept pixels');
  relativeErrors.sort((a, b) => a - b);
  partedCosts.sort((a, b) => a - b);
  return {
    keptBoth: relativeErrors.length,
    max: relativeErrors.at(-1),
    within1e3: relativeErrors.filter((e) => e <= 1e-3).length / relativeErrors.length,
    within1e2: relativeErrors.filter((e) => e <= 1e-2).length / relativeErrors.length,
    partedMedianCost: partedCosts.length ? partedCosts[Math.floor(partedCosts.length / 2)] : 0,
  };
}
function assertFinalAgreement(actual, expected, range) {
  const result = agreement(actual, expected, range);
  // Existing tests/typeshade/mvs-webgpu.test.mjs tolerances, unchanged.
  assert.ok(result.within1e2 >= 0.99, JSON.stringify(result));
  assert.ok(result.within1e3 >= 0.95, JSON.stringify(result));
  assert.ok(result.partedMedianCost <= 0.01, JSON.stringify(result));
  return result;
}
function assertLifetime(row) {
  assert.deepEqual(row.validation, []);
  assert.deepEqual(row.uncaptured, []);
  assert.deepEqual(row.lost, []);
  const byId = new Map(row.records.map((record) => [record.id, record]));
  for (const snapshot of row.snapshots) {
    assert.equal(snapshot.liveBuffers, 0, snapshot.name);
    assert.equal(snapshot.liveBytes, 0, snapshot.name);
    assert.equal(
      snapshot.retained.length,
      1,
      'one GPUDevice-scoped dispatch ring, no other retained buffers',
    );
    const ring = byId.get(snapshot.retained[0]);
    assert.equal(ring.usage, 72);
    assert.ok(ring.size >= 4096);
    assert.equal(ring.label, '');
    assert.equal(ring.mappedAtCreation, false);
    assert.match(ring.stack, /\bat ringFor\b/);
    assert.match(ring.stack, /\bat uniformsOf\b/);
  }
  const retained = row.records.filter((record) => !record.destroyed);
  assert.equal(retained.length, 1);
  assert.equal(retained[0].invalidatedByDevice, true);
  assert.ok(row.records.every((record) => record.destroyed || record === retained[0]));
  assert.equal(row.native.deviceDestroyCalls, 1);
  assert.ok(row.native.pipelines > 0 && row.native.dispatches > 0 && row.native.readbacks > 0);
}

test('independent MVS perturbations on actual WebGPU preserve OFF and satisfy existing synthetic parity/accuracy bounds', async () => {
  const scene = mvsScene(),
    plan = planMvsViews(scene),
    entry = plan.views.find((v) => v.id === 2),
    range = entry.depthRange;
  const packedRange = [plan.cameras[entry.index * 24 + 18], plan.cameras[entry.index * 24 + 19]];
  const input = {
      views: scene.views.map((v) => ({
        ...v,
        pose: { R: Array.from(v.pose.R), t: Array.from(v.pose.t) },
      })),
      landmarks: scene.landmarks.map((p) => ({ ...p, xyz: Array.from(p.xyz) })),
      images: [...scene.images].map(([id, image]) => [
        id,
        { ...image, rgba: Array.from(image.rgba) },
      ]),
    },
    inputSHA = sha(JSON.stringify(input));
  const cpuDevice = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  let cpu;
  try {
    cpu = await mvsPhotometricGpu(cpuDevice, scene, plan, scene.images, entry.index, {
      independentRefinement: true,
      snapshots: true,
    });
  } finally {
    await cpuDevice.dispose();
  }
  const temporary = await mkdtemp(join(tmpdir(), 'typecv-independent-refinement-'));
  const serve = {},
    legacy = process.env.TYPECV_MVS_LEGACY_DIST,
    legacyHashes = [];
  let legacyManifestSHA;
  try {
    if (legacy) {
      const root = resolve(legacy);
      async function walk(directory) {
        for (const item of await readdir(directory, { withFileTypes: true })) {
          const path = join(directory, item.name);
          if (item.isDirectory()) await walk(path);
          else {
            const name = relative(root, path).replaceAll('\\', '/');
            serve['/legacy-dist/' + name] = path;
            legacyHashes.push({ path, sha256: sha(await readFile(path)) });
          }
        }
      }
      await walk(root);
      assert.ok(serve['/legacy-dist/internal/gpu.js'], 'frozen legacy internal module is required');
      const manifestPath =
          process.env.TYPECV_MVS_LEGACY_MANIFEST ?? resolve(root, '../manifest.json'),
        manifestBytes = await readFile(manifestPath),
        manifest = JSON.parse(manifestBytes);
      legacyManifestSHA = sha(manifestBytes);
      assert.equal(manifest.files.length, legacyHashes.length, 'complete frozen legacy tree');
      for (const file of manifest.files) {
        assert.ok(file.path.startsWith('dist/'));
        const path = resolve(root, file.path.slice(5)),
          bytes = await readFile(path);
        assert.equal(bytes.byteLength, file.bytes, path);
        assert.equal(sha(bytes), file.sha256, path);
      }
    }
    await build({
      entryPoints: [
        fileURLToPath(new URL('./mvs-independent-refinement-gpu-host.mjs', import.meta.url)),
      ],
      bundle: true,
      platform: 'browser',
      format: 'esm',
      outfile: join(temporary, 'host.mjs'),
    });
    serve['/private-independent-refinement/host.mjs'] = join(temporary, 'host.mjs');
    const { result, errors } = await inBrowser(
      async (argument) => {
        const { runIndependentPhotometricGpu } =
          await import('/private-independent-refinement/host.mjs');
        return runIndependentPhotometricGpu(argument);
      },
      {
        hardware: process.env.TYPECV_HARDWARE_GPU === '1',
        legacyUrl: legacy ? '/legacy-dist/internal/gpu.js' : undefined,
        input,
        inputSHA,
      },
      { serve, timeoutMs: 600000 },
    );
    if (process.env.TYPECV_REFINEMENT_GPU_RESULT)
      await writeFile(
        process.env.TYPECV_REFINEMENT_GPU_RESULT,
        JSON.stringify({ result, errors, legacyHashes, legacyManifestSHA }, null, 2) + '\n',
      );
    assert.deepEqual(errors, []);
    assert.equal(result.inputSHA, inputSHA);
    const rows = Object.fromEntries(result.rows.map((row) => [row.name, row]));
    assert.deepEqual(rows.default.final, rows.off.final);
    assert.deepEqual(rows.default.snapshots, rows.off.snapshots);
    assert.ok(
      rows.on.snapshots.some((hash, i) => hash !== rows.off.snapshots[i]),
      'the ON option must exercise a different search on this fixed synthetic scene',
    );
    if (legacy) {
      assert.deepEqual(
        rows.off.final,
        rows['frozen-legacy'].final,
        'OFF must reproduce the separately loaded frozen old runtime',
      );
      assert.deepEqual(rows.off.snapshots, rows['frozen-legacy'].snapshots);
      assert.deepEqual(rows['off-stream'].final, rows['frozen-legacy-stream'].final);
      assert.deepEqual(rows['off-stream'].snapshots, rows['frozen-legacy-stream'].snapshots);
    }
    assert.deepEqual(
      rows.on.final,
      rows['on-scalar-pose'].final,
      'existing pose-cache bit-exact contract',
    );
    for (const row of result.rows) {
      assert.ok(row.final.depth.every(Number.isFinite));
      assert.ok(row.final.cost.every(Number.isFinite));
      assert.ok(row.final.depth.every((d) => d >= packedRange[0] && d <= packedRange[1]));
      assert.ok(row.final.cost.every((c) => c >= 0 && c <= 2));
    }
    assert.equal(rows.on.residency.photometric.mode, 'window');
    assert.equal(rows['on-stream'].residency.photometric.mode, 'streamed');
    assert.equal(rows['on-stream'].residency.photometric.slots, 2);
    assert.deepEqual(rows.on.residency, rows.off.residency);
    assert.deepEqual(rows['on-stream'].residency, rows['off-stream'].residency);
    const init = agreement(rows.on.early[0], plain(cpu.snapshots[0]), range);
    const first = agreement(rows.on.early[1], plain(cpu.snapshots[1]), range);
    assert.ok(init.max <= 1e-3, JSON.stringify(init));
    assert.ok(first.within1e3 >= 0.999, JSON.stringify(first));
    const final = assertFinalAgreement(rows.on.final, plain(cpu), range);
    const streamed = assertFinalAgreement(rows['on-stream'].final, rows.on.final, range);
    const accuracy = {};
    for (const name of ['on', 'on-stream']) {
      accuracy[name] = depthMetrics(
        rows[name].final,
        scene.truth[2],
        plan.normalization.scale,
        range,
        decodeMvsNormal,
      );
      assert.ok(accuracy[name].median <= 0.01);
      assert.ok(accuracy[name].kept >= 0.8);
      assert.ok(accuracy[name].medianNormalDeg <= 16);
    }
    const hostAccuracy = depthMetrics(
      cpu,
      scene.truth[2],
      plan.normalization.scale,
      range,
      decodeMvsNormal,
    );
    assert.ok(Math.abs(accuracy.on.median - hostAccuracy.median) <= 0.0005);
    assert.ok(Math.abs(accuracy.on.kept - hostAccuracy.kept) <= 0.01);
    for (const row of result.lifetimes) assertLifetime(row);
    for (const item of legacyHashes)
      assert.equal(sha(await readFile(item.path)), item.sha256, item.path);
    console.log(
      JSON.stringify({
        independentRefinementWebGPU: {
          description: result.description,
          frozenLegacyCompared: !!legacy,
          init,
          first,
          final,
          streamed,
          accuracy,
          hostAccuracy,
        },
      }),
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});
