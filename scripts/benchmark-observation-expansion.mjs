import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { generalVideoFixture, renderGeneralVideoFrame } from '../tests/fixtures/general-video.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { alignPoses } from '../website/src/lib/pose-alignment.ts';
import { compareSeedGauge } from './geometry-comparison.mjs';
import { evaluateExpansionCohorts } from './observation-expansion-evaluation.mjs';

const option = (key, fallback) =>
  process.argv.find((v) => v.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const backend = option('backend', 'webgpu');
if (!['cpu', 'webgpu'].includes(backend)) throw new Error('backend must be cpu or webgpu');
const refinement = option('refinement', 'resume');
if (!['resume', 'rebuild'].includes(refinement))
  throw new Error('refinement must be resume or rebuild');
const booleanOption = (name, fallback) => {
  const value = option(name, fallback);
  if (!['0', '1'].includes(value)) throw new Error(`${name} must be 0 or 1`);
  return value === '1';
};
const geometryPatchWarp = booleanOption('patch-warp', '0'),
  pyramidCache = booleanOption('pyramid-cache', '0'),
  surfacePatchWarp = booleanOption('surface-warp', '0'),
  multiViewConsistency = booleanOption('multi-view', '0');
const output = resolve(
  option('output', `docs/experiments/general-video/observation-expansion-${backend}.json`),
);
const cases = [],
  evaluation = [];
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
for (const scenario of ['translation', 'translation-omission', 'rotation', 'low-parallax']) {
  const fixture = generalVideoFixture(scenario === 'rotation' ? 'rotation' : 'translation'),
    views = fixture.truth.views
      .slice(0, 4)
      .map((v) =>
        scenario === 'low-parallax'
          ? { ...v, pose: { R: v.pose.R, t: v.pose.t.map((x) => x * 0.001) } }
          : v,
      ),
    rendered = views.map((v) => renderGeneralVideoFrame(v));
  // Only the rendered RGB and declared calibration are sent to the reconstruction process.
  // Poses, depths, renderer identities and geometry-fixture correspondences stay in Node.
  cases.push({
    name: `render-${scenario}`,
    omitDetections: scenario === 'translation-omission',
    frames: rendered.map((f) => ({
      id: f.id,
      width: f.width,
      height: f.height,
      camera: f.camera,
      rgba: Array.from(f.rgba),
    })),
  });
  evaluation.push({ views, rendered });
}
const cache = JSON.parse(await readFile('tests/fixtures/graham-hall-lab-matches.json', 'utf8')),
  meta = JSON.parse(await readFile('tests/fixtures/lab-images/graham-hall/sample.json', 'utf8'));
cases.push({
  name: 'graham-hall-cached-rgb-features',
  frames: await Promise.all(
    meta.frames.map(async (f, i) => {
      const jpeg = await readFile(resolve('tests/fixtures/lab-images/graham-hall', f.file));
      return {
        id: f.id,
        width: f.width,
        height: f.height,
        camera: cache.frames[i].camera,
        keypoints: cache.frames[i].keypoints,
        jpeg: jpeg.toString('base64'),
        sha256: sha(jpeg),
      };
    }),
  ),
  pairs: cache.pairs,
});
evaluation.push({ references: meta.frames.map((f) => ({ id: f.id, ...f.reference })) });
const video = option('video', '');
if (video) {
  const preparationStarted = performance.now();
  const crop = option('video-crop', '1080:1440:0:480');
  if (!/^\d+:\d+:\d+:\d+$/.test(crop)) throw new Error('video-crop must be width:height:x:y');
  const start = Number(option('video-start', '8')),
    count = 6,
    fps = 10,
    width = 360,
    [cw, ch, cropX, cropY] = crop.split(':').map(Number),
    height = 2 * Math.round((width * ch) / cw / 2),
    validRect = option('video-valid', '0,32,360,230').split(',').map(Number);
  if (
    !Number.isFinite(start) ||
    start < 0 ||
    validRect.length !== 4 ||
    !validRect.every(Number.isFinite)
  )
    throw new Error('Invalid video window');
  const bytes = execFileSync(
    'ffmpeg',
    [
      '-v',
      'error',
      '-ss',
      String(start),
      '-i',
      resolve(video),
      '-vf',
      `fps=${fps},crop=${crop},scale=${width}:${height}`,
      '-frames:v',
      String(count),
      '-f',
      'rawvideo',
      '-pix_fmt',
      'rgba',
      '-',
    ],
    { maxBuffer: 1 << 27 },
  );
  const frameBytes = width * height * 4;
  if (bytes.length !== count * frameBytes) throw new Error('Video window is incomplete');
  cases.push({
    name: 'user-video-window',
    mask: 'manual background rectangle',
    videoWindow: {
      seekSeconds: start,
      analysisFps: fps,
      frames: count,
      crop,
      preparationMs: performance.now() - preparationStarted,
      sourceToAnalysisPixels: [
        width / cw,
        0,
        (width / cw) * (0.5 - cropX) - 0.5,
        0,
        height / ch,
        (height / ch) * (0.5 - cropY) - 0.5,
        0,
        0,
        1,
      ],
      timing: 'resampled 10 fps analysis frames after seek; no physical velocity inferred',
      calibration: 'assumed focal 0.8 * analysis width; not calibrated or ground truth',
    },
    frames: Array.from({ length: count }, (_, id) => ({
      id,
      width,
      height,
      validRect,
      camera: { fx: width * 0.8, fy: width * 0.8, cx: (width - 1) / 2, cy: (height - 1) / 2 },
      rgba: Array.from(bytes.subarray(id * frameBytes, (id + 1) * frameBytes)),
    })),
  });
  evaluation.push({ video: true });
}
const selected = option('cases', '').split(',').filter(Boolean);
const chosen = cases.filter((c) => !selected.length || selected.includes(c.name));

const { result, errors } = await inBrowser(
  async ({
    cases,
    backend,
    refinement,
    geometryPatchWarp,
    pyramidCache,
    surfacePatchWarp,
    multiViewConsistency,
  }) => {
    const api = await import('/dist/gpu.js'),
      core = await import('/dist/index.js');
    let adapter, gpu;
    if (backend === 'webgpu') {
      adapter = await navigator.gpu.requestAdapter();
      gpu = await adapter.requestDevice();
    }
    const device = await api.ShadeDevice.open({
      ...(gpu ? { gpuDevice: gpu } : { backend: 'cpu' }),
      poolBytes: 0,
    });
    const controls = {
      ransacIterations: 512,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: false,
    };
    const plainModel = (model) => ({
      views: model.views.map((v) => ({
        ...v,
        pose: { R: Array.from(v.pose.R), t: Array.from(v.pose.t) },
      })),
      landmarks: model.landmarks.map((p) => ({ ...p, xyz: Array.from(p.xyz) })),
    });
    function summarize(r, images, elapsedMs) {
      const model = r.reconstruction,
        observations = model.landmarks.reduce((n, p) => n + p.observations.length, 0),
        coverage = images.map((image) => {
          const cells = new Set();
          for (const p of model.landmarks)
            for (const o of p.observations)
              if (o.viewId === image.id)
                cells.add(
                  `${Math.floor((o.pixel[0] * 4) / image.width)}:${Math.floor((o.pixel[1] * 4) / image.height)}`,
                );
          return { id: image.id, cellsOf16: cells.size };
        });
      return {
        registeredFrames: model.views.length,
        landmarks: model.landmarks.length,
        observations,
        tracks: r.tracks.length,
        trackedObservations: r.tracks.reduce((n, t) => n + t.observations.length, 0),
        pendingTracks: r.tracking?.untriangulatedTrackIds.length ?? 0,
        meanTrackLength: r.tracks.length
          ? r.tracks.reduce((n, t) => n + t.observations.length, 0) / r.tracks.length
          : 0,
        meanLandmarkLength: model.landmarks.length ? observations / model.landmarks.length : 0,
        coverage,
        meanReprojectionErrorPx: r.diagnostics.meanReprojectionErrorPx,
        initialPair: r.diagnostics.initialPair,
        conflicts: r.diagnostics.conflictingTrackEdges,
        elapsedMs,
        model: plainModel(model),
      };
    }
    function reconstruct(images, pairs, initial) {
      const start = performance.now();
      try {
        const r = initial
          ? core.refineIncremental(images, pairs, initial, controls)
          : core.reconstructIncremental(images, pairs, controls);
        return { r, tracking: r.tracking, report: summarize(r, images, performance.now() - start) };
      } catch (e) {
        if (e.name !== 'ReconstructionInitializationError') throw e;
        return {
          tracking: e.tracking,
          report: {
            registeredFrames: 0,
            landmarks: 0,
            observations: 0,
            tracks: e.tracking?.tracks.length ?? 0,
            trackedObservations:
              e.tracking?.tracks.reduce((n, t) => n + t.observations.length, 0) ?? 0,
            pendingTracks: e.tracking?.untriangulatedTrackIds.length ?? 0,
            initialization: 'deferred',
            reason: e.message,
            elapsedMs: performance.now() - start,
          },
        };
      }
    }
    const out = [];
    try {
      for (const entry of cases) {
        const allStart = performance.now(),
          handles = [],
          frames = [],
          pairs = [];
        let decodeMs = 0,
          featureMs = 0,
          matchingMs = 0;
        try {
          for (const source of entry.frames) {
            let start = performance.now(),
              rgba;
            if (source.rgba) rgba = Uint8ClampedArray.from(source.rgba);
            else {
              const bitmap = await createImageBitmap(
                new Blob([Uint8Array.from(atob(source.jpeg), (c) => c.charCodeAt(0))], {
                  type: 'image/jpeg',
                }),
              );
              try {
                const canvas = new OffscreenCanvas(bitmap.width, bitmap.height),
                  context = canvas.getContext('2d');
                context.drawImage(bitmap, 0, 0);
                rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
              } finally {
                bitmap.close();
              }
            }
            const gray = Float32Array.from(
              { length: source.width * source.height },
              (_, i) =>
                (0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]) / 255,
            );
            decodeMs += performance.now() - start;
            let keypoints;
            if (source.keypoints) keypoints = Float64Array.from(source.keypoints);
            else {
              start = performance.now();
              const h = await api.extractFeaturesGpuPyramidResident(
                device,
                rgba,
                source.width,
                source.height,
                { maxFeatures: 600, threshold: 0.045, smoothDescriptors: true },
              );
              handles.push(h);
              keypoints = await h.readKeypoints();
              featureMs += performance.now() - start;
            }
            const validMask =
              source.validRect &&
              Uint8Array.from({ length: source.width * source.height }, (_, i) => {
                const x = i % source.width,
                  y = Math.floor(i / source.width),
                  [left, top, right, bottom] = source.validRect;
                return x >= left && x < right && y >= top && y < bottom ? 1 : 0;
              });
            frames.push({
              id: source.id,
              camera: source.camera,
              width: source.width,
              height: source.height,
              gray,
              keypoints,
              ...(validMask ? { validMask } : {}),
            });
          }
          if (entry.pairs)
            pairs.push(...entry.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })));
          else
            for (let b = 1; b < frames.length; b++)
              for (let a = Math.max(0, b - 2); a < b; a++) {
                const start = performance.now(),
                  m = await api.matchResidentFeaturesGpu(device, handles[a], handles[b], {
                    ratio: 0.82,
                    maxDistance: 80,
                  });
                pairs.push({
                  first: frames[a].id,
                  second: frames[b].id,
                  matches: m.matches,
                  distances: m.distances,
                });
                matchingMs += performance.now() - start;
              }
        } finally {
          for (const h of handles) await h.dispose();
        }
        for (const pair of pairs) {
          const a = frames.find((f) => f.id === pair.first),
            b = frames.find((f) => f.id === pair.second);
          if (!a.validMask && !b.validMask) continue;
          const matches = [],
            distances = [],
            valid = (frame, k) =>
              !frame.validMask ||
              frame.validMask[
                Math.round(frame.keypoints[2 * k + 1]) * frame.width +
                  Math.round(frame.keypoints[2 * k])
              ];
          for (let k = 0; k < pair.matches.length; k += 2)
            if (valid(a, pair.matches[k]) && valid(b, pair.matches[k + 1])) {
              matches.push(pair.matches[k], pair.matches[k + 1]);
              distances.push(pair.distances?.[k / 2] ?? 0);
            }
          pair.matches = Uint32Array.from(matches);
          pair.distances = Float64Array.from(distances);
        }
        if (entry.omitDetections) {
          // Controlled detector dropout, selected by detector row only, never truth identities.
          const remap = new Map();
          for (const frame of frames) {
            const map = new Map(),
              kept = [];
            for (let f = 0; f < frame.keypoints.length / 2; f++) {
              if ((f + frame.id) % 2) continue;
              map.set(f, kept.length / 2);
              kept.push(frame.keypoints[2 * f], frame.keypoints[2 * f + 1]);
            }
            remap.set(frame.id, map);
            frame.keypoints = Float64Array.from(kept);
          }
          for (const pair of pairs) {
            const a = remap.get(pair.first),
              b = remap.get(pair.second),
              matches = [],
              distances = [];
            for (let k = 0; k < pair.matches.length; k += 2)
              if (a.has(pair.matches[k]) && b.has(pair.matches[k + 1])) {
                matches.push(a.get(pair.matches[k]), b.get(pair.matches[k + 1]));
                distances.push(pair.distances?.[k / 2] ?? 0);
              }
            pair.matches = Uint32Array.from(matches);
            pair.distances = Float64Array.from(distances);
          }
        }
        const baseline = reconstruct(frames, pairs),
          baselineTotalMs = performance.now() - allStart;
        const expansion = await api.expandObservationsGpu(device, frames, pairs, {
          geometryPatchWarp,
          pyramidCache,
          surfacePatchWarp,
          multiViewConsistency,
          radius: 1,
          maxSeedsPerFrame: 128,
          maxNewPerFrame: 256,
          klt: { levels: 4, halfWindow: 5, maxIterations: 20 },
          verifiedPairs: baseline.r?.verifiedPairs,
          initialModel: baseline.r?.reconstruction,
          rotations: baseline.tracking?.rotations,
        });
        const afterFrames = expansion.images.map((image, i) => ({ ...frames[i], ...image })),
          improved = reconstruct(
            afterFrames,
            expansion.candidatePairs,
            refinement === 'resume' ? baseline.r : undefined,
          ),
          known = new Set(
            baseline.r?.reconstruction.landmarks.flatMap((p) =>
              p.observations.map((o) => `${o.viewId}:${o.featureId}`),
            ) ?? [],
          ),
          newLandmarks =
            improved.r?.reconstruction.landmarks.filter(
              (p) => !p.observations.some((o) => known.has(`${o.viewId}:${o.featureId}`)),
            ).length ?? 0;
        out.push({
          name: entry.name,
          mask: entry.mask ?? 'none',
          videoWindow: entry.videoWindow,
          frontend: { decodeMs, featureMs, matchingMs, cached: !!entry.pairs },
          inputFeatureCounts: frames.map((f) => f.keypoints.length / 2),
          outputFeatureCounts: afterFrames.map((f) => f.keypoints.length / 2),
          baseline: baseline.report,
          expanded: improved.report,
          newLandmarksWithoutBaselineObservation: newLandmarks,
          expansion: expansion.diagnostics,
          totalMs: performance.now() - allStart,
          baselineTotalMs,
        });
        console.log(
          JSON.stringify({
            case: entry.name,
            before: baseline.report.landmarks,
            after: improved.report.landmarks,
            added: expansion.diagnostics.newCoordinates,
          }),
        );
      }
      return {
        backend,
        adapter: adapter
          ? {
              vendor: adapter.info.vendor,
              architecture: adapter.info.architecture,
              description: adapter.info.description,
              fallback: adapter.info.isFallbackAdapter,
            }
          : null,
        controls,
        diagnostics: device.diagnostics,
        statistics: device.statistics,
        cases: out,
      };
    } finally {
      await device.dispose();
    }
  },
  {
    cases: chosen,
    backend,
    refinement,
    geometryPatchWarp,
    pyramidCache,
    surfacePatchWarp,
    multiViewConsistency,
  },
  { timeoutMs: 900000 },
);
if (errors.length) throw new Error(JSON.stringify(errors));
for (const row of result.cases) {
  const index = cases.findIndex((c) => c.name === row.name),
    truth = evaluation[index];
  row.cohortComparison = evaluateExpansionCohorts(row.baseline, row.expanded, truth);
  for (const name of ['baseline', 'expanded']) {
    const report = row[name];
    if (!report.model) continue;
    if (truth.references)
      report.poseReference = alignPoses(
        report.model.views.map((v) => ({ id: v.id, ...v.pose })),
        truth.references,
      );
    else if (!truth.video) {
      report.poseTruth = compareSeedGauge({ views: truth.views }, report.model, report.initialPair);
      const anchor = truth.views.find((v) => v.id === report.initialPair[0]),
        second = truth.views.find((v) => v.id === report.initialPair[1]),
        center = (pose) =>
          [0, 1, 2].map(
            (i) => -pose.R[i] * pose.t[0] - pose.R[3 + i] * pose.t[1] - pose.R[6 + i] * pose.t[2],
          ),
        origin = center(anchor.pose),
        other = center(second.pose),
        scale = Math.hypot(...other.map((x, i) => x - origin[i])),
        errors = [];
      for (const point of report.model.landmarks) {
        const o = point.observations[0],
          view = truth.views.find((v) => v.id === o.viewId),
          frame = truth.rendered.find((v) => v.id === o.viewId),
          x = Math.round(o.pixel[0]),
          y = Math.round(o.pixel[1]);
        if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) continue;
        const depth = frame.depthZ[y * frame.width + x];
        if (!(depth > 0 && Number.isFinite(depth))) continue;
        const world = [0, 1, 2].map(
          (i) =>
            origin[i] +
            scale *
              (anchor.pose.R[i] * point.xyz[0] +
                anchor.pose.R[3 + i] * point.xyz[1] +
                anchor.pose.R[6 + i] * point.xyz[2]),
        );
        const z =
          view.pose.R[6] * world[0] +
          view.pose.R[7] * world[1] +
          view.pose.R[8] * world[2] +
          view.pose.t[2];
        errors.push(Math.abs(z - depth) / depth);
      }
      errors.sort((a, b) => a - b);
      report.depthTruth = {
        method:
          'camera +Z at first observed pixel; nearest rendered pixel depth; same fixed seed gauge as camera evaluation',
        evaluatedPoints: errors.length,
        medianRelativeError: errors[Math.floor(errors.length * 0.5)] ?? null,
        p90RelativeError: errors[Math.floor(errors.length * 0.9)] ?? null,
        fractionAbove10Percent: errors.length
          ? errors.filter((e) => e > 0.1).length / errors.length
          : null,
        limitation:
          'nearest pixel quantization and surface boundaries are included; no source point identity is supplied',
      };
    }
    delete report.model;
  }
}
const receipt = {
  baselineCommit: 'e4e5046cf5722ce1c4d731bdd47de2b301d188bd',
  implementationCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  sourceHashes: Object.fromEntries(
    await Promise.all(
      [
        'src/gpu/observation-expansion.ts',
        'src/gpu/observation-surface.ts',
        'src/gpu/observation-consensus.ts',
        'scripts/benchmark-observation-expansion.mjs',
        'scripts/observation-expansion-evaluation.mjs',
      ].map(async (path) => [path, sha(await readFile(path))]),
    ),
  ),
  anchorSelection:
    'estimated view angle, magnification and camera orientation; nearest target playback index when geometry is unavailable',
  recordedAt: new Date().toISOString(),
  trackingOptions: { geometryPatchWarp, pyramidCache, surfacePatchWarp, multiViewConsistency },
  refinement:
    refinement === 'resume'
      ? 'existing accepted map and seed gauge carried into core.refineIncremental; no new seed selection'
      : 'control: rebuild from the same expanded candidates and allow a different seed',
  inputs: chosen.map((c) => ({
    name: c.name,
    omission: !!c.omitDetections,
    videoWindow: c.videoWindow,
    mask: c.mask ?? 'none',
    frames: c.frames.map((f) => ({
      id: f.id,
      width: f.width,
      height: f.height,
      validRect: f.validRect,
      sha256: f.sha256 ?? sha(Uint8Array.from(f.rgba)),
    })),
  })),
  mode: 'RGB only; calibration provenance per input; geometry runs on CPU; LK and fresh feature/matching on selected TypeShade backend',
  calibration: chosen.map((c) => ({
    name: c.name,
    source:
      c.videoWindow?.calibration ??
      (c.name.startsWith('graham-') ? 'supplied COLMAP calibration' : 'known synthetic K'),
  })),
  inputTruth:
    'no poses, depths, object transforms or truth correspondences supplied; labels evaluated outside browser',
  limits:
    'One expansion pass; local patch model; cached Graham Hall detector/matcher time excluded; no physical GPU or full gameplay claim',
  ...result,
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt, null, 2));
