// Exported sparse-observation replay only. No source pixels or landmark geometry are committed.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const args = process.argv.slice(2),
  flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const frozen = JSON.parse(
  await readFile(
    new URL('../docs/experiments/general-video/inverse-depth-scaling-inputs.json', import.meta.url),
    'utf8',
  ),
);
if (flag('--worker')) {
  let source, record;
  if (flag('--worker') === 'synthetic') {
    const count = Number(flag('--synthetic-views')),
      { inverseDepthLongScene } = await import('../tests/typeshade/inverse-depth-cases.mjs');
    source = inverseDepthLongScene(count);
    // Translate each local group with the flight, keeping its observations in the camera FOV.
    source.landmarks = source.landmarks.map((p) => {
      const xyz = p.xyz.slice();
      xyz[0] += 0.03 * p.observations[0].viewId;
      const observations = p.observations.map((o) => {
        const v = source.views[o.viewId],
          k = v.camera,
          q = xyz.map((x, i) => x + v.pose.t[i]),
          x = q[0] / q[2],
          y = q[1] / q[2],
          d = 1 + k.k1 * (x * x + y * y);
        return { ...o, pixel: [k.fx * x * d + k.cx, k.fy * y * d + k.cy] };
      });
      return { ...p, xyz, observations };
    });
    record = {
      record: `synthetic-local-${count}`,
      sha256: createHash('sha256').update(JSON.stringify(source)).digest('hex'),
    };
  } else {
    const bytes = await readFile(flag('--worker'));
    record = frozen.records.find((r) => r.record === flag('--record'));
    if (createHash('sha256').update(bytes).digest('hex') !== record.sha256)
      throw new Error('Frozen input hash changed');
    source = JSON.parse(bytes);
  }
  const model = {
      views: source.views.map((v) => ({
        id: v.id,
        camera: v.camera,
        pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
      })),
      landmarks: source.landmarks.map((p) => ({
        trackId: p.trackId,
        xyz: Float64Array.from(p.xyz),
        observations: p.observations,
      })),
    },
    mode = flag('--solver'),
    module = await import(
      pathToFileURL(resolve(flag(mode === 'baseline' ? '--baseline' : '--candidate'))).href
    ),
    { inverseRangeUncertainty, inverseDepthJacobian } =
      await import('../src/geometry/inverse-depth.ts');
  globalThis.gc?.();
  const started = performance.now(),
    result = module.bundleAdjustInverseDepth(model, [], {
      iterations: Number(flag('--iterations')),
      solver: mode === 'baseline' ? undefined : mode,
      maxCameraParameters: 1536,
    }),
    elapsedMs = performance.now() - started,
    views = new Map(result.reconstruction.views.map((v) => [v.id, v])),
    pointValues = [];
  let admitted = 0;
  for (const p of result.estimates) {
    const range = inverseRangeUncertainty(p.point, p.covariance);
    let maxError = 0;
    for (const o of p.observations) {
      const v = views.get(o.viewId),
        j = inverseDepthJacobian(v.camera, v.pose, p.point);
      maxError = Math.max(
        maxError,
        j ? Math.hypot(j.pixel[0] - o.pixel[0], j.pixel[1] - o.pixel[1]) : Infinity,
      );
    }
    if (
      range.stddev !== undefined &&
      range.mean > 3 * range.stddev &&
      range.stddev / range.mean <= 0.2 &&
      maxError <= 3 &&
      p.observations.length >= 3
    )
      admitted++;
    pointValues.push({
      id: p.trackId,
      rhoZ: p.point.parameters[2],
      variances: p.covariance && [p.covariance[0], p.covariance[4], p.covariance[8]],
      ...(args.includes('--full-values')
        ? {
            parameters: Array.from(p.point.parameters),
            covariance: p.covariance ? Array.from(p.covariance) : null,
            bearingCovariance: p.bearingCovariance ? Array.from(p.bearingCovariance) : null,
            reference: {
              R: Array.from(p.point.reference.R),
              t: Array.from(p.point.reference.t),
            },
            referenceViewId: p.referenceViewId ?? null,
            directionOnly: !!p.directionOnly,
            informationDeferred: !!p.informationDeferred,
            observationCount: p.observations.length,
          }
        : {}),
    });
  }
  console.log(
    JSON.stringify({
      record: record.record,
      sha256: record.sha256,
      mode,
      iterations: Number(flag('--iterations')),
      views: model.views.length,
      points: model.landmarks.length,
      elapsedMs,
      maxRssKiB: process.resourceUsage().maxRSS,
      diagnostics: result.diagnostics,
      uncertainty: result.uncertainty,
      admitted,
      pointValues,
      ...(args.includes('--full-values')
        ? {
            cameraValues: result.reconstruction.views.map((v) => ({
              id: v.id,
              camera: {
                fx: v.camera.fx,
                fy: v.camera.fy,
                cx: v.camera.cx,
                cy: v.camera.cy,
                k1: v.camera.k1 ?? 0,
                k2: v.camera.k2 ?? 0,
              },
              R: Array.from(v.pose.R),
              t: Array.from(v.pose.t),
            })),
          }
        : {}),
    }),
  );
} else {
  if (!flag('--records') || !flag('--out'))
    throw new Error(
      'Usage: --records <record directory root> --out <directory> [--baseline <frozen bundle>] [--case <record>] [--modes dense,sparse] [--iterations 0,1] [--synthetic-views <count>]',
    );
  const root = resolve(flag('--records')),
    out = resolve(flag('--out')),
    results = [],
    comparisons = [];
  await mkdir(out, { recursive: true });
  const { build } = await import('esbuild'),
    candidate = join(out, '.candidate.mjs');
  await build({
    entryPoints: [
      new URL('../src/reconstruction/inverse-depth-bundle.ts', import.meta.url).pathname,
    ],
    outfile: candidate,
    bundle: true,
    platform: 'node',
    format: 'esm',
  });
  const hash = (bytes) => createHash('sha256').update(bytes).digest('hex'),
    candidateHash = hash(await readFile(candidate)),
    baselineHash = flag('--baseline') ? hash(await readFile(flag('--baseline'))) : undefined;
  const cases = [
    ...frozen.records,
    ...(flag('--synthetic-views')
      ? [{ record: `synthetic-local-${flag('--synthetic-views')}`, synthetic: true }]
      : []),
  ];
  for (const record of cases) {
    if (flag('--case') && flag('--case') !== record.record) continue;
    for (const iteration of flag('--iterations', '0,1').split(',')) {
      const modes = flag(
        '--modes',
        flag('--baseline') ? 'baseline,dense,sparse,auto' : 'dense,sparse,auto',
      ).split(',');
      const rows = [];
      for (const mode of modes) {
        console.error(`Starting ${record.record} / ${mode} / iterations ${iteration}`);
        const command = [
          '--expose-gc',
          import.meta.filename,
          '--worker',
          record.synthetic
            ? 'synthetic'
            : join(root, 'record-' + record.record, 'typecv-reconstruction.json'),
          '--record',
          record.record,
          '--solver',
          mode,
          '--iterations',
          iteration,
          '--candidate',
          candidate,
        ];
        if (record.synthetic) command.push('--synthetic-views', flag('--synthetic-views'));
        if (flag('--baseline')) command.push('--baseline', resolve(flag('--baseline')));
        let value;
        try {
          value = JSON.parse(
            execFileSync(process.execPath, command, {
              encoding: 'utf8',
              maxBuffer: 16 * 1024 ** 2,
              timeout: 300000,
            }),
          );
        } catch (e) {
          value = {
            record: record.record,
            mode,
            iterations: Number(iteration),
            error: String(e.stderr || e.message).slice(0, 1200),
          };
        }
        await writeFile(
          join(out, `${record.record}-${mode}-${iteration}.json`),
          JSON.stringify(value),
        );
        rows.push(value);
        results.push(value);
        console.error(JSON.stringify({ ...value, pointValues: undefined }));
      }
      for (const [left, right] of [
        ['baseline', 'auto'],
        ['dense', 'sparse'],
      ]) {
        const a = rows.find((r) => r.mode === left),
          b = rows.find((r) => r.mode === right);
        if (!a?.pointValues || !b?.pointValues) continue;
        const reference = new Map(a.pointValues.map((p) => [p.id, p])),
          errors = [];
        let maxRhoDifference = 0,
          unmatchedTracks = 0;
        for (const p of b.pointValues) {
          const q = reference.get(p.id);
          if (!q) {
            unmatchedTracks++;
            continue;
          }
          maxRhoDifference = Math.max(maxRhoDifference, Math.abs(p.rhoZ - q.rhoZ));
          if (p.variances && q.variances)
            errors.push(...p.variances.map((v, i) => Math.abs(v / q.variances[i] - 1)));
        }
        errors.sort((a, b) => a - b);
        comparisons.push({
          record: record.record,
          iterations: Number(iteration),
          left,
          right,
          unmatchedTracks,
          commonMarginalCoordinates: errors.length,
          maxRhoDifference,
          varianceRelativeErrorMedian: errors[Math.floor(errors.length / 2)] ?? null,
          varianceRelativeErrorP95: errors[Math.floor(errors.length * 0.95)] ?? null,
          varianceRelativeErrorMax: errors.at(-1) ?? null,
          leftAdmitted: a.admitted,
          rightAdmitted: b.admitted,
        });
      }
    }
  }
  await writeFile(
    join(out, 'summary.json'),
    JSON.stringify(
      {
        protocol: frozen.protocol,
        sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], {
          cwd: new URL('..', import.meta.url),
          encoding: 'utf8',
        }).trim(),
        candidateHash,
        baselineHash,
        runtime: process.version,
        platform: process.platform,
        results: results.map(({ pointValues, ...row }) => row),
        comparisons,
      },
      null,
      2,
    ) + '\n',
  );
}
