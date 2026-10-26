// A deterministic fake of the lab's runtime entries, `/runtime/gpu.js` and
// `/runtime/internal/gpu.js`, for the baselines of phase 0 (`docs/library-structure.md`, section
// 5). The worker reaches the GPU only through these entries, so the fake stands at that boundary:
// the devices that `ShadeDevice.open` returns, the feature front ends, the matchers, the mapper
// session, the KLT tracker, the multi-video reconstruction and the dense MVS.
//
// `createFakeRuntime` is self-contained: the transcript harness compiles its source text inside
// the worker's realm, and the browser test writes the same text into a module. It reads no clock
// and no random source. Every value comes from `spec` and from fixed sequences.
//
// The frame identity: the alpha byte of the first pixel of a frame names its content index. The
// harness writes it; the fake reads it to give each frame its features.

/**
 * Make the fake runtime. `spec` is plain data (see the transcript tests), and `log(name, args)`
 * receives every call that the worker makes into the runtime. The result holds `module`, the
 * namespace that the worker imports, and `control`, the harness's handle on the devices.
 */
export function createFakeRuntime(spec = {}, log = () => {}) {
  const s = spec;
  const KERNEL_NS = 125000;
  // Scene points: fixed positions in the unit square from a linear congruential sequence.
  let seed = 0x2545f491;
  const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const BASE = [];
  for (let i = 0; i < 512; i++) BASE.push([random(), random()]);
  const quarter = (value) => Math.round(value * 4) / 4;
  const featureCount = (index) =>
    s.features && s.features[index] !== undefined ? s.features[index] : (s.defaultFeatures ?? 24);
  const shiftOf = (index) => (s.shift && s.shift[index] ? s.shift[index] : [index * 4, 0]);
  const contentOf = (pixels) => pixels[3];
  function keypointsOf(index, width, height) {
    const n = featureCount(index),
      [dx, dy] = shiftOf(index),
      out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      out[2 * i] = quarter(8 + BASE[i][0] * (width - 16) + dx);
      out[2 * i + 1] = quarter(8 + BASE[i][1] * (height - 16) + dy);
    }
    return out;
  }

  // Calls: every call is logged; `s.loss` and `s.fail` name a call by its name and its count.
  const counts = new Map();
  async function enter(name, args) {
    log(name, args);
    const at = counts.get(name) ?? 0;
    counts.set(name, at + 1);
    if (s.loss && s.loss.call === name && s.loss.at === at) {
      lose(devices.length - 1, s.loss.info);
      // A loss arrives between two submissions: its handler runs before the call goes on.
      await null;
      if (s.loss.reject) throw new Error(s.loss.reject);
    }
    if (s.fail && s.fail.call === name && s.fail.at === at) throw new Error(s.fail.message);
  }

  // Devices.
  const devices = [];
  const state = new WeakMap();
  function report(device, names) {
    const timed = state.get(device)?.timed === true;
    return {
      tier: 'webgpu',
      dispatches: names.length * 2,
      submissions: 1,
      groups: names.map((name) => (timed ? { name, gpuTime: KERNEL_NS } : { name })),
    };
  }
  /** Run named kernels on a device: progress, the per-kernel table and the byte counters. */
  function run(device, names, upload = 0, readback = 0) {
    const d = state.get(device);
    if (d) {
      for (const name of names) {
        d.onProgress?.({ kernel: name, phase: 'start' });
        const row = d.rows.get(name) ?? {
          name,
          groups: 0,
          dispatches: 0,
          timedGroups: 0,
          gpuNs: 0,
        };
        row.groups += 1;
        row.dispatches += 2;
        if (d.timed) {
          row.timedGroups += 1;
          row.gpuNs += KERNEL_NS;
        }
        d.rows.set(name, row);
        d.onProgress?.({ kernel: name, phase: 'complete' });
      }
      d.stats.applicationUploadBytes += upload;
      d.stats.applicationReadbackBytes += readback;
      d.stats.liveBytes += 4096 * names.length;
      d.chunks.scorePairModels.chunks += names.length;
    }
    return {
      report: report(device, names),
      execution: { applicationUploadBytes: upload, applicationReadbackBytes: readback },
    };
  }
  function lose(index, info) {
    const device = devices[index];
    state.get(device)?.lose(info ?? { reason: 'destroyed', message: 'synthetic device loss' });
  }
  const ShadeDevice = {
    async open(options) {
      const attempt = devices.length;
      await enter('ShadeDevice.open', [options]);
      if (s.openFailure && s.openFailure.includes(attempt))
        throw new Error('synthetic device open failure');
      const { onProgress, ...settings } = options;
      const timed = options.gpuTimestamps !== false && s.timestamps !== false;
      const d = {
        onProgress,
        timed,
        rows: new Map(),
        stats: {
          liveBytes: 0,
          pooledBytes: 0,
          applicationUploadBytes: 0,
          applicationReadbackBytes: 0,
        },
        chunks: { scorePairModels: { msPerUnit: 0.5, chunks: 0, longestMs: 12, lastUnits: 64 } },
        disposed: false,
        lose: undefined,
      };
      const device = {
        ...settings,
        backend: 'webgpu',
        boundedSubmissions: options.boundedSubmissions === true,
        fp64Flavor:
          options.fp64Flavor === 'auto' || options.fp64Flavor === undefined
            ? (s.autoFlavor ?? 'float')
            : options.fp64Flavor,
        timeBudgetMs: options.timeBudgetMs ?? 250,
        info: { webgpu: { features: timed ? ['shader-f16', 'timestamp-query'] : ['shader-f16'] } },
        get statistics() {
          return { ...d.stats };
        },
        get kernelStatistics() {
          if (d.disposed) throw new Error('kernel statistics of a disposed device');
          return { rows: [...d.rows.values()].map((row) => ({ ...row })), dropped: 0 };
        },
        get chunkRates() {
          if (d.disposed) throw new Error('chunk rates of a disposed device');
          return Object.fromEntries(
            Object.entries(d.chunks).map(([key, rate]) => [key, { ...rate }]),
          );
        },
        resetKernelStatistics() {
          log('device.resetKernelStatistics', [attempt]);
          d.rows.clear();
        },
        lost: new Promise((resolve) => {
          d.lose = resolve;
        }),
        async dispose() {
          log('device.dispose', [attempt]);
          d.disposed = true;
        },
      };
      state.set(device, d);
      devices.push(device);
      return device;
    },
  };

  // Feature front ends.
  function features(kind, device, pixels, width, height, options, kernels) {
    const index = contentOf(pixels),
      n = featureCount(index),
      points = keypointsOf(index, width, height),
      { report: r, execution } = run(device, kernels, pixels.length, 8);
    return {
      count: n,
      index,
      descriptors: { kind, index },
      ...(options && options.grayForKlt
        ? { gray: { index, levels: options.grayForKlt.levels } }
        : {}),
      report: r,
      execution,
      async readKeypoints() {
        log(kind + '.readKeypoints', [index]);
        if (state.get(device)) state.get(device).stats.applicationReadbackBytes += n * 16;
        return points;
      },
      async dispose() {
        log(kind + '.dispose', [index]);
      },
    };
  }
  async function extractSiftGpuResident(device, pixels, width, height, options) {
    await enter('extractSiftGpuResident', [contentOf(pixels), width, height, options]);
    return features('sift', device, pixels, width, height, options, [
      'siftPyramid',
      'siftExtrema',
      'siftDescribe',
    ]);
  }
  async function extractOrbGpuResident(device, pixels, width, height, options) {
    await enter('extractOrbGpuResident', [contentOf(pixels), width, height, options]);
    return features('orb', device, pixels, width, height, options, ['orbFast', 'orbDescribe']);
  }
  async function extractFeaturesGpuPyramidResident(device, pixels, width, height, options) {
    await enter('extractFeaturesGpuPyramidResident', [contentOf(pixels), width, height, options]);
    return features('corner', device, pixels, width, height, options, ['fastCorners']);
  }

  // Matchers: feature i of every frame is scene point i. The plain matcher drops one match in
  // ten, the guided matcher drops none; `s.matchLimit['a:b']` caps the matches of a pair.
  function matchesBetween(a, b, guided) {
    const n = Math.min(featureCount(a), featureCount(b)),
      rows = [],
      distances = [];
    for (let i = 0; i < n; i++) {
      if (!guided && (i * 7 + a * 3 + b) % 10 === 0) continue;
      rows.push(i, i);
      distances.push(((i * 13 + a + b) % 50) / 100);
    }
    const limit = s.matchLimit ? s.matchLimit[a + ':' + b] : undefined;
    if (limit !== undefined) {
      rows.length = Math.min(rows.length, 2 * limit);
      distances.length = rows.length / 2;
    }
    return { matches: Uint32Array.from(rows), distances: Float64Array.from(distances) };
  }
  async function matchFloatDescriptorsGpu(device, first, second, options) {
    await enter('matchFloatDescriptorsGpu', [first, second, options]);
    const guided = !!(options && options.epipolar);
    const { report: r, execution } = run(
      device,
      [guided ? 'nearestFloatGuided' : 'nearestFloat'],
      0,
      64,
    );
    return { ...matchesBetween(first.index, second.index, guided), report: r, execution };
  }
  async function matchResidentFeaturesGpu(device, first, second, options) {
    await enter('matchResidentFeaturesGpu', [first.index, second.index, options]);
    const { report: r, execution } = run(device, ['hammingMatch'], 0, 64);
    return { ...matchesBetween(first.index, second.index, false), report: r, execution };
  }
  let essentialCalls = 0;
  async function estimateEssentialRansacGpu(device, pixels, first, second, options) {
    await enter('estimateEssentialRansacGpu', [pixels, first, second, options]);
    run(device, ['essentialRansac']);
    if (s.essentialFailure && s.essentialFailure.includes(essentialCalls++)) {
      const error = new Error('No essential matrix has enough inliers');
      error.name = 'GeometryEstimationError';
      throw error;
    }
    return { E: [0, -0.25, 0.5, 0.25, 0, -1, -0.5, 1, 0], inliers: pixels.length / 4 };
  }
  function unprojectPixel(camera, pixel) {
    return [(pixel[0] - camera.cx) / camera.fx, (pixel[1] - camera.cy) / camera.fy];
  }

  // The mapper session of a single-video run.
  class IncrementalSfmSession {
    constructor(device, controls) {
      log('new IncrementalSfmSession', [controls]);
      this.device = device;
      this.controls = controls;
      this.added = [];
      this.cameras = new Map();
      this.views = new Map();
      this.points = new Map();
      this.rejected = new Map();
      this.verifications = [];
      this.focalSearch = undefined;
    }
    get activeViewIds() {
      return this.added.slice(-(s.window ?? 32));
    }
    stage(operation, kernels, elapsedMs) {
      const { report: r, execution } = run(this.device, kernels, 64, 32);
      this.controls.onStage?.({ operation, elapsedMs, execution: { report: r, ...execution } });
    }
    register(id) {
      const [dx, dy] = shiftOf(id);
      this.views.set(id, {
        id,
        camera: { ...this.cameras.get(id) },
        pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-dx / 100, -dy / 100, 0] },
      });
    }
    triangulate(pair) {
      for (let k = 0; k < pair.matches.length; k += 2) {
        const a = pair.matches[k],
          b = pair.matches[k + 1];
        let point = this.points.get(a);
        if (!point) {
          point = {
            trackId: a,
            xyz: [quarter(BASE[a][0] * 4 - 2), quarter(BASE[a][1] * 3 - 1.5), 5 + (a % 7) / 2],
            observations: [],
          };
          this.points.set(a, point);
        }
        for (const [viewId, featureId] of [
          [pair.first, a],
          [pair.second, b],
        ])
          if (!point.observations.some((o) => o.viewId === viewId))
            point.observations.push({ viewId, featureId });
      }
    }
    model() {
      return {
        views: [...this.views.values()].map((view) => ({
          id: view.id,
          camera: { ...view.camera },
          pose: { R: [...view.pose.R], t: [...view.pose.t] },
        })),
        landmarks: [...this.points.values()].map((point) => ({
          trackId: point.trackId,
          xyz: [...point.xyz],
          observations: point.observations.map((o) => ({ ...o })),
        })),
      };
    }
    unregistered(final) {
      const out = [...this.rejected].map(([viewId, reason]) => ({ viewId, reason }));
      if (final)
        for (const id of this.added)
          if (!this.views.has(id) && !this.rejected.has(id))
            out.push({ viewId: id, reason: 'Not registered' });
      return out;
    }
    async add(image, pairs) {
      await enter('session.add', [image, pairs]);
      this.added.push(image.id);
      this.cameras.set(image.id, image.camera);
      this.controls.onProgress?.({
        operation: 'verifyPair',
        index: this.added.length,
        phase: 'start',
      });
      const verifications = [],
        verified = [];
      for (const pair of pairs) {
        const kept = [];
        for (let k = 0; k < pair.matches.length; k += 2)
          if ((pair.matches[k] + pair.first + pair.second) % 6 !== 0)
            kept.push(pair.matches[k], pair.matches[k + 1]);
        const inliers = kept.length / 2,
          accepted = inliers >= (s.minInliers ?? 8);
        verifications.push({
          first: pair.first,
          second: pair.second,
          matches: pair.matches.length / 2,
          inliers,
          homographyInliers: Math.floor(inliers / 2),
          medianAngleDeg: 2.5,
          seedEligible: accepted,
          ...(accepted ? {} : { reason: 'Too few inliers' }),
        });
        if (accepted)
          verified.push({
            first: pair.first,
            second: pair.second,
            matches: Uint32Array.from(kept),
          });
      }
      this.verifications.push(...verifications);
      this.stage('verifyPair', ['pairModels', 'scorePairModels'], 3);
      let registered = false;
      if (!this.views.size) {
        const seedPair = verified.find((pair) => this.cameras.has(pair.first));
        if (seedPair) {
          this.register(seedPair.first);
          this.register(image.id);
          this.triangulate(seedPair);
          registered = true;
        }
      } else {
        const links = verified.filter((pair) => this.views.has(pair.first));
        if (links.length) {
          this.register(image.id);
          for (const pair of links) this.triangulate(pair);
          registered = true;
        } else this.rejected.set(image.id, 'No verified pair with a registered view');
      }
      if (!this.views.size) return undefined;
      if (registered) {
        this.controls.onProgress?.({ operation: 'pnp', index: this.added.length, phase: 'start' });
        this.stage('pnp', ['p3pHypotheses', 'scorePnP'], 2);
        this.stage('triangulation', ['triangulateTracks'], 1);
        this.stage('bundle', ['bundleCg'], 4);
        if (this.controls.focalSearch && !this.focalSearch)
          this.focalSearch = {
            ratio: s.focalRatio ?? 0.82,
            gridRatio: 0.8,
            views: [...this.views.keys()],
          };
        this.controls.onReconstruction?.(this.model());
      }
      return {
        reconstruction: this.model(),
        diagnostics: { pairVerification: verifications, meanReprojectionErrorPx: 0.5 },
        unregistered: this.unregistered(false),
        verifiedPairs: verified,
      };
    }
    async finish() {
      await enter('session.finish', []);
      // A session that never initialized fails before its final refinement.
      if (s.finishFailure && !this.views.size) {
        const pair = this.verifications[0];
        const error = new Error('No initial pair passed the verification');
        error.name = 'ReconstructionInitializationError';
        error.pairVerification = this.verifications.slice();
        error.focalSearch = {
          omittedAttempts: 0,
          attempts: [
            {
              ratio: 0.7,
              views: this.added.length,
              pairs: this.verifications.length,
              eligiblePairs: 0,
              ...(pair ? { bestPair: pair } : {}),
              reasons: [{ reason: 'Too few inliers', pairs: this.verifications.length }],
              failure: 'No stable pair',
            },
          ],
        };
        throw error;
      }
      this.controls.onProgress?.({ operation: 'final', index: this.added.length, phase: 'start' });
      this.stage('final', ['bundleCg', 'bundleSchur'], 7);
      return {
        reconstruction: this.model(),
        diagnostics: { meanReprojectionErrorPx: 0.5, pairVerification: this.verifications.slice() },
        unregistered: this.unregistered(true),
      };
    }
  }

  // The KLT tracker: track k follows scene point k through the frames it is pushed.
  async function createKltTracker(device, width, height, options) {
    await enter('createKltTracker', [width, height, options]);
    const capacity = (options && options.options && options.options.capacity) || 1024;
    const t = { pushes: 0, joins: 0, frame: 0, rejected: 0, matches: 0 };
    const alive = () => Math.min(capacity, s.kltTracks ?? 64);
    return {
      async push(input, pushOptions) {
        await enter('klt.push', [input.sift ? { sift: input.sift.index } : 'rgba', pushOptions]);
        run(device, ['kltTrack'], input.sift ? 0 : width * height * 4, 0);
        t.pushes++;
        t.frame = pushOptions.frameId;
      },
      async join({ frameId, keypoints, views }) {
        await enter('klt.join', [frameId, keypoints.length / 2, views]);
        run(device, ['kltJoin'], 0, 256);
        t.joins++;
        const pairs = [];
        for (const first of views) {
          const n = Math.min(alive(), featureCount(first), keypoints.length / 2),
            rows = [],
            tracks = [];
          for (let k = 0; k < n; k++) {
            // One track in eleven disagrees with the descriptors: it points at the next feature.
            rows.push(k, k % 11 === 5 ? (k + 1) % n : k);
            tracks.push(k);
          }
          t.matches += n;
          pairs.push({
            first,
            second: frameId,
            matches: Uint32Array.from(rows),
            tracks: Uint32Array.from(tracks),
          });
        }
        return { pairs };
      },
      tracks() {
        const n = alive(),
          [dx, dy] = shiftOf(t.frame),
          x = new Float64Array(n),
          y = new Float64Array(n);
        for (let k = 0; k < n; k++) {
          x[k] = quarter(8 + BASE[k][0] * (width - 16) + dx);
          y[k] = quarter(8 + BASE[k][1] * (height - 16) + dy);
        }
        return { count: n, id: Uint32Array.from({ length: n }, (_, k) => k), x, y };
      },
      async reject(ids) {
        await enter('klt.reject', [ids]);
        t.rejected += ids.length;
      },
      stats() {
        // The shape of `KltStats` (`src/gpu/klt-tracker.ts`): 8 end reasons, 6 length bins.
        return {
          pushes: t.pushes,
          joins: t.joins,
          alive: alive(),
          ended: [0, 1, 0, 0, 0, 0, 0, t.rejected],
          joined: t.matches,
          seeded: alive(),
          filledIn: 0,
          filtered: 2,
          matches: t.matches,
          rejected: t.rejected,
          lengths: [1, 0, 2, t.pushes, 0, 0],
          meanLived: 2.5,
          msPush: 1.5,
          msSelect: 0.5,
          msJoin: 2,
        };
      },
      async dispose() {
        log('klt.dispose', []);
      },
    };
  }
  const kltOptionsFor = (width) => ({ levels: width >= 640 ? 4 : 3 });
  function kltReject(verdicts) {
    const ids = [];
    for (const [track, verdict] of verdicts)
      if (verdict.outlier > 0 && verdict.inlier === 0) ids.push(track);
    return Uint32Array.from(ids.sort((a, b) => a - b));
  }

  // The dense MVS: two depth previews and one point batch per view, and the result.
  async function reconstructDenseGpu(device, scene, images, options) {
    await enter('reconstructDenseGpu', [scene, images, options]);
    const views = scene.views,
      total = views.length;
    for (const stage of ['photometric', 'geometric']) {
      for (let k = 0; k < total; k++) {
        if (options.signal && options.signal.aborted) throw new Error('aborted');
        run(device, [stage === 'photometric' ? 'mvsPropagate' : 'mvsGeometric']);
        const viewId = views[k].id;
        options.onView?.({
          type: 'depth',
          viewId,
          stage,
          ...(stage === 'photometric' ? { iteration: 1 } : {}),
          width: 4,
          height: 3,
          depth: Float32Array.from({ length: 12 }, (_, i) => 1 + i / 8 + k),
        });
        options.onProgress?.({ stage, viewId, done: k + 1, total });
        if (stage === 'geometric' && s.depthDiagnostics && options.onDepthDiagnostic)
          await options.onDepthDiagnostic({
            viewId,
            stage: 'geometric-filtered',
            width: 4,
            height: 3,
            depthScaleToModel: 1,
            depth: Float32Array.from({ length: 12 }, (_, i) => 2 + i / 4),
            confidence: Float32Array.from({ length: 12 }, (_, i) => i / 12),
          });
      }
      if (s.denseHold === stage) await new Promise((resolve) => (control.release = resolve));
    }
    let count = 0;
    for (let k = 0; k < total; k++) {
      if (options.signal && options.signal.aborted) throw new Error('aborted');
      run(device, ['tsdfIntegrate']);
      options.onFusionProgress?.({ phase: 'fusion', done: k + 1, total, viewId: views[k].id });
      const n = 3;
      count += n;
      options.onView?.({
        type: 'points',
        viewId: views[k].id,
        count: n,
        positions: Float32Array.from({ length: 3 * n }, (_, i) => i / 4 + k),
        colors: Uint8Array.from({ length: 3 * n }, (_, i) => (i * 40 + k * 7) % 256),
        normals: Int16Array.from({ length: 3 * n }, (_, i) => (i % 3 === 2 ? 32767 : 0)),
      });
    }
    options.onFusionPreview?.({
      positions: new Float32Array(3),
      colors: new Uint8Array(3),
      normals: new Int16Array(3),
    });
    return {
      views: views.map((view, k) => ({
        id: view.id,
        status: 'done',
        sources: Math.max(0, total - 1),
        depthPixels: 12 - k,
        points: 3,
      })),
      points: {
        count,
        positions: Float32Array.from({ length: 3 * count }, (_, i) => i / 8),
        colors: Uint8Array.from({ length: 3 * count }, (_, i) => (i * 11) % 256),
        normals: Int16Array.from({ length: 3 * count }, (_, i) => (i % 3 === 2 ? 32767 : 0)),
      },
      residency: { photometric: { bytes: 1 << 20 }, geometric: { bytes: 1 << 21 } },
      ...(options.planarDepth ? { planar: { filled: 5 } } : {}),
    };
  }
  async function cleanDenseCloudGpu(device, points, options) {
    await enter('cleanDenseCloudGpu', [points.count, options]);
    run(device, ['cloudKnn', 'cloudVoxel']);
    const keep = Math.max(1, points.count - 2);
    return {
      cloud: {
        count: keep,
        positions: points.positions.slice(0, 3 * keep),
        colors: points.colors.slice(0, 3 * keep),
        normals: points.normals.slice(0, 3 * keep),
      },
      report: { before: points.count, after: keep, outliers: 1, voxels: keep, clusters: 1 },
    };
  }
  async function meshDenseCloudGpu(device, points) {
    await enter('meshDenseCloudGpu', [points.count]);
    run(device, ['meshSurface']);
    return {
      mesh: {
        positions: points.positions.slice(0, 9),
        colors: points.colors.slice(0, 9),
        indices: Uint32Array.from([0, 1, 2]),
      },
      report: { vertices: 3, triangles: 1, voxelSize: 0.05 },
    };
  }

  // The internal entry of the multi-video job.
  function parseVocabulary(bytes, words) {
    log('parseVocabulary', [bytes.length, words]);
    return Float32Array.from({ length: words * 128 }, (_, i) => (i % 17) / 16);
  }
  async function describeKeyframeGpu(device, sift, vocabulary) {
    await enter('describeKeyframeGpu', [sift.index, vocabulary.length]);
    run(device, ['quantizeDescriptors', 'vladAssign']);
    const n = sift.count;
    return {
      descriptors: Uint8Array.from({ length: n * 128 }, (_, i) => (i * 31 + sift.index) % 256),
      global: Float32Array.from({ length: 32 }, (_, i) => ((i + sift.index) % 9) / 8),
    };
  }
  async function reconstructMultiVideo(name, device, keyframes, options) {
    await enter(name, [keyframes, options]);
    const m = s.multi ?? {};
    const order = keyframes.map((_, i) => i);
    if (options.canonical)
      order.sort((a, b) => {
        const ka = keyframes[a].videoId + ':' + keyframes[a].time,
          kb = keyframes[b].videoId + ':' + keyframes[b].time;
        return ka < kb ? -1 : ka > kb ? 1 : 0;
      });
    const viewOf = new Array(keyframes.length);
    order.forEach((input, rank) => (viewOf[input] = rank));
    const viewIds = keyframes.map((_, i) => viewOf[i]);
    options.onEvent?.({ stage: 'keyframes', viewIds });
    options.onProgress?.('retrieval');
    // Candidate pairs: consecutive keyframes of the input order.
    const pairs = [];
    for (let i = 1; i < keyframes.length; i++)
      pairs.push({ first: viewIds[i - 1], second: viewIds[i] });
    const activity = options.graph && options.graph.onActivity;
    activity?.({ stage: 'candidates', phase: 'complete', pairs });
    options.onEvent?.({ stage: 'candidates', total: pairs.length });
    activity?.({ stage: 'matching', phase: 'start', pairs });
    activity?.({
      stage: 'matching',
      phase: 'complete',
      pairs: pairs.map((pair) => ({ ...pair, matches: 30 })),
    });
    options.onEvent?.({ stage: 'matching', done: pairs.length, total: pairs.length });
    const videoOfView = new Map(keyframes.map((k, i) => [viewIds[i], k.videoId]));
    const groups = m.components ?? [[...new Set(keyframes.map((k) => k.videoId))]];
    const componentOf = (videoId) => groups.findIndex((group) => group.includes(videoId));
    const edges = pairs
      .filter(
        (pair) =>
          componentOf(videoOfView.get(pair.first)) === componentOf(videoOfView.get(pair.second)),
      )
      .map((pair, k) => ({ ...pair, inliers: 40 + k, seedEligible: k % 2 === 0 }));
    activity?.({ stage: 'verification', phase: 'start', pairs });
    activity?.({
      stage: 'verification',
      phase: 'complete',
      pairs: pairs.map((pair) => {
        const edge = edges.find((e) => e.first === pair.first && e.second === pair.second);
        return { ...pair, inliers: edge ? edge.inliers : 5, accepted: !!edge };
      }),
    });
    options.onEvent?.({ stage: 'verification', done: pairs.length, total: pairs.length });
    const components = groups.map((videos, index) => {
      const ids = keyframes
        .map((k, i) => (videos.includes(k.videoId) ? viewIds[i] : -1))
        .filter((id) => id >= 0);
      return {
        index,
        videoIds: videos,
        viewIds: ids.sort((a, b) => a - b),
        reconstructed: ids.length >= 2,
      };
    });
    activity?.({ stage: 'components', components });
    options.onEvent?.({ stage: 'components', edges: edges.length });
    options.onEvent?.({
      stage: 'plan',
      sizes: components.filter((c) => c.reconstructed).map((c) => c.viewIds.length),
    });
    const results = [];
    for (const component of components) {
      if (!component.reconstructed) {
        results.push({ ...component, reconstruction: undefined });
        continue;
      }
      if (m.errorComponent === component.index) {
        results.push({ ...component, error: 'No initial pair in the component' });
        continue;
      }
      const views = [],
        points = new Map();
      for (const viewId of component.viewIds) {
        const k = keyframes[order[viewId]];
        run(device, ['p3pHypotheses', 'scorePnP']);
        options.mapper?.onStage?.({
          operation: views.length < 2 ? 'initialize' : 'pnp',
          elapsedMs: 2,
          execution: { report: report(device, ['scorePnP']) },
        });
        views.push({
          id: viewId,
          camera: { ...k.camera },
          pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-viewId / 10, 0, 0] },
        });
        const n = Math.min(12, k.keypoints.length / 2);
        for (let i = 0; i < n; i++) {
          const point = points.get(i) ?? {
            trackId: component.index * 1000 + i,
            xyz: [quarter(BASE[i][0] * 4 - 2), quarter(BASE[i][1] * 3 - 1.5), 6],
            observations: [],
          };
          point.observations.push({ viewId, featureId: i });
          points.set(i, point);
        }
        if (views.length >= 2)
          options.onEvent?.({
            stage: 'reconstruction',
            component: component.index,
            registered: views.length,
            reconstruction: {
              views: views.map((v) => ({ ...v })),
              landmarks: [...points.values()].map((p) => ({
                ...p,
                observations: p.observations.slice(),
              })),
            },
          });
      }
      const reconstruction = { views, landmarks: [...points.values()] };
      options.onEvent?.({
        stage: 'componentDone',
        component: component.index,
        registered: views.length,
        reconstruction,
        meanReprojectionErrorPx: 0.75,
      });
      options.onRelease?.({ stage: 'descriptors', viewIds: component.viewIds });
      options.onRelease?.({ stage: 'scene', viewIds: component.viewIds });
      results.push({
        ...component,
        reconstruction,
        meanReprojectionErrorPx: 0.75,
        registrationOrder: component.viewIds.slice(),
      });
    }
    if (m.failure) throw new Error(m.failure);
    const internal = new Map();
    for (const edge of edges) {
      const videoId = videoOfView.get(edge.first);
      if (videoId === videoOfView.get(edge.second))
        internal.set(videoId, (internal.get(videoId) ?? 0) + 1);
    }
    const videoIds = [...new Set(keyframes.map((k) => k.videoId))];
    return {
      canonical: options.canonical,
      graph: {
        edges,
        components,
        stats: {
          keyframes: keyframes.length,
          candidatePairs: pairs.length,
          verifiedPairs: pairs.length,
          edges: edges.length,
          exhaustivePairs: (keyframes.length * (keyframes.length - 1)) / 2,
        },
        timing: { retrievalMs: 1, pairMatchingMs: 2, pairVerificationMs: 3, componentsMs: 0.5 },
        videos: videoIds.map((videoId) => {
          const index = componentOf(videoId);
          const group = groups[index] ?? [];
          return {
            videoId,
            component: index,
            tag:
              group.length > 1
                ? 'connected'
                : components[index]?.reconstructed
                  ? 'alone'
                  : 'not connected',
          };
        }),
      },
      keyframes: keyframes.map((k, i) => ({
        viewId: viewIds[i],
        videoId: k.videoId,
        frameKey: k.frameKey,
        time: k.time,
      })),
      components: results,
      timing: { reconstructionMs: 4 },
      hostRetention: {
        featuresBeforeGraphBytes: 4096,
        featuresAfterGraphBytes: 2048,
        featuresAfterMappingBytes: 0,
        graphMatchesBeforeMappingBytes: 1024,
        graphMatchesAfterMappingBytes: 0,
      },
      ...(m.descriptorCache
        ? {
            descriptorCache: {
              hits: 3,
              misses: 2,
              uncached: 0,
              decodes: 2,
              decodedBytes: 1024,
              applicationUploadBytes: 2048,
              evictions: 0,
              entries: 2,
              residentBytes: 512,
              peakEntries: 2,
              peakResidentBytes: 512,
              limitBytes: 4096,
            },
          }
        : {}),
    };
  }

  const control = {
    devices,
    lose,
    /** Set while the dense MVS holds at `s.denseHold`: call it to let the MVS go on. */
    release: undefined,
  };
  const module = {
    ShadeDevice,
    IncrementalSfmSession,
    extractSiftGpuResident,
    extractOrbGpuResident,
    extractFeaturesGpuPyramidResident,
    matchFloatDescriptorsGpu,
    matchResidentFeaturesGpu,
    estimateEssentialRansacGpu,
    unprojectPixel,
    createKltTracker,
    kltOptionsFor,
    kltReject,
    reconstructDenseGpu,
    cleanDenseCloudGpu,
    meshDenseCloudGpu,
    parseVocabulary,
    describeKeyframeGpu,
    reconstructMultiVideoGpu: (device, keyframes, options) =>
      reconstructMultiVideo('reconstructMultiVideoGpu', device, keyframes, options),
    ...(s.multi && s.multi.owned
      ? {
          reconstructMultiVideoOwnedGpu: (device, keyframes, options) =>
            reconstructMultiVideo('reconstructMultiVideoOwnedGpu', device, keyframes, options),
        }
      : {}),
  };
  return { module, control };
}
