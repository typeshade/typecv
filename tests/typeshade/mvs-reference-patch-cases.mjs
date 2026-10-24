// The actual shader's original sourceCost branch is the independent scalar oracle. These
// test-only probes run both branches through TypeShade's f64 and f32 CPU code generators.
const PROBE = `
declare const referencePatchCoverage: storage<array<u32>, 'read_write'>;
export function referencePatchProbe(x: u32, y: u32, reuse: bool): array<f32, 29> {
  const stats: MvsWindow = referenceWindow(i32(x), i32(y));
  let patch: MvsReferencePatch = {
    values: fill<f32, 121>(0.),
    weights: fill<f32, 121>(0.),
  };
  if (reuse && stats.ok !== 0) prepareReferencePatch(patch, i32(x), i32(y), stats.center);
  let out: array<f32, 29> = fill<f32, 29>(0.);
  out[0] = stats.mean;
  out[1] = stats.variance;
  out[2] = stats.weight;
  out[3] = stats.center;
  out[4] = f32(stats.ok);
  for (let h: u32 = 0; h < 4; h++) {
    const X: vec3 = referenceRay(f32(x), f32(y)) * (2.5 + f32(h) * 0.9);
    const n: vec3 = normalize(vec3(f32(h) * 0.04, -f32(h) * 0.03, -1.));
    for (let k: u32 = 0; k < 6; k++) {
      const s: u32 = mvsSources[1 + k];
      if (reuse && stats.ok !== 0)
        out[5 + h * 6 + k] = sourceCostCached(patch, s, i32(x), i32(y), X, n, stats);
      else out[5 + h * 6 + k] = sourceCost(s, i32(x), i32(y), X, n, stats);
    }
  }
  return out;
}
`;

// Keep the pre-culling scalar control flow as an independent test-only oracle.
function legacyPropagation(source) {
  const start = source.indexOf('function mvsPropagateAt('),
    open = source.indexOf('{', start);
  if (start < 0 || open < 0) throw new Error('MVS propagation oracle site changed');
  let depth = 1,
    end = open + 1;
  for (; end < source.length && depth; end++) {
    if (source[end] === '{') depth++;
    else if (source[end] === '}') depth--;
  }
  const guard = '  if (stats.ok === 0 && bestCost <= NO_COST) return;';
  const body = source.slice(start, end);
  if (body.split(guard).length !== 2) throw new Error('MVS culling oracle site changed');
  return (
    body.replace('function mvsPropagateAt(', 'function mvsPropagateLegacyAt(').replace(guard, '') +
    `
export function mvsPropagateLegacyOracle(gid: vec3u): void {
  mvsPropagateLegacyAt(gid, referenceWindow, hypothesisCost);
}
`
  );
}

// Count the actual production call sites rather than reconstructing their guards in JS.
// This instrumentation is appended/replaced only in the CPU test's in-memory shader.
function instrumentCoverage(source) {
  source = source.replace(/\r\n/g, '\n');
  const replaceOnce = (needle, replacement) => {
    if (source.split(needle).length !== 2)
      throw new Error('MVS coverage instrumentation site changed');
    source = source.replace(needle, replacement);
  };
  const injectEntry = (name, statements) => {
    const declaration = `function ${name}(`;
    if (source.split(declaration).length !== 2) throw new Error('MVS coverage function changed');
    const at = source.indexOf('{', source.indexOf(declaration)) + 1;
    source = source.slice(0, at) + '\n' + statements + source.slice(at);
  };
  replaceOnce(
    '  if (stats.ok === 0 && bestCost <= NO_COST) return;',
    '  if (stats.ok === 0 && bestCost <= NO_COST) { referencePatchCoverage[8]++; return; }',
  );
  injectEntry('sourceCost', '  referencePatchCoverage[3]++;');
  injectEntry(
    'sourceCostCached',
    `  referencePatchCoverage[2]++;
  if (stats.ok === 0) referencePatchCoverage[6]++;`,
  );
  const window =
    '    const cost: f32 = evaluateHypothesis(x, y, depth, n, stats);\n    if (cost < bestCost) {';
  replaceOnce(
    window,
    `    if (active && stats.ok !== 0) referencePatchCoverage[0]++;
${window}
      if (active) referencePatchCoverage[1]++;`,
  );
  const streamed = '        const s: u32 = mvsSources[mvsControl.state * SOURCE_SLOTS + 1 + k];';
  replaceOnce(
    streamed,
    `        if (active && phase === 2 && h < 8 && k === 0) referencePatchCoverage[4]++;
${streamed}`,
  );
  const select = '      if (c < cost) {\n        depth = hyp.depth;';
  replaceOnce(
    select,
    `      if (c < cost) {
        if (active && h < 8) referencePatchCoverage[5]++;
        depth = hyp.depth;`,
  );
  return source;
}

function sameNumbers(a, b, precision, label) {
  const ArrayType = precision === 'f32' ? Float32Array : Float64Array;
  const x = ArrayBuffer.isView(a) ? a : ArrayType.from(a),
    y = ArrayBuffer.isView(b) ? b : ArrayType.from(b);
  const xb = new Uint8Array(x.buffer, x.byteOffset, x.byteLength),
    yb = new Uint8Array(y.buffer, y.byteOffset, y.byteLength);
  if (xb.length !== yb.length || xb.some((v, i) => v !== yb[i]))
    throw new Error(`Reference patch changed ${label}`);
}

function bindings({
  radius = 5,
  step = 2,
  bilateral = false,
  texture = 'normal',
  reference = 0,
} = {}) {
  const W = 17,
    H = 13,
    P = W * H,
    N = 7;
  const axis = Math.floor((2 * radius) / step) + 1;
  const cameras = new Float32Array(N * 24),
    gray = new Float32Array(N * P);
  for (let s = 0; s < N; s++) {
    cameras.set(
      [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        -0.16 * s,
        0,
        0,
        18,
        19,
        (W - 1) / 2,
        (H - 1) / 2,
        W,
        H,
        2,
        8,
        0.16 * s,
        0,
        0,
        s,
      ],
      s * 24,
    );
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const amplitude = texture === 'flat' ? 0 : texture === 'near threshold' ? 0.0014 : 0.2;
        gray[s * P + y * W + x] =
          0.5 + amplitude * (Math.sin((x + s * 0.2) * 0.7) + Math.cos(y * 0.6));
      }
  }
  if (texture === 'holes' || texture === 'center hole') {
    gray[reference * P + 6 * W + 9] = -1;
    for (let s = 0; s < N; s++) gray[s * P + 3] = -1;
  }
  if (texture === 'center hole') gray[reference * P + 6 * W + 8] = -1;
  const sources = [N - 1, ...Array.from({ length: N }, (_, s) => s).filter((s) => s !== reference)];
  const seeds = new Float32Array(P * 3);
  for (let i = 0; i < P; i++) seeds.set([i % W, Math.floor(i / W), 4], i * 3);
  return {
    mvsControl: {
      width: W,
      height: H,
      reference,
      state: 0,
      color: 0,
      iteration: 1,
      seed: 117,
      bestSources: 3,
      windowRadius: radius,
      windowStep: step,
      axisSamples: axis,
      minSamples: Math.ceil((axis * axis) / 4),
      bilateral: Number(bilateral),
      spatialFalloff: 1 / 8,
      colorFalloff: 1 / 0.02,
      minAngleCos: Math.cos(Math.PI / 180),
      perturbDepth: 0.25,
      perturbAngle: Math.PI / 6,
      geometric: 0,
      geometricWeight: 0.3,
      geometricMaxCost: 3,
      view: reference,
      relativePoseCache: 0,
    },
    mvsRange: { first: 0, end: Math.ceil(W / 2) * H },
    mvsCameras: cameras,
    mvsGray: gray,
    mvsSources: Uint32Array.from(sources),
    mvsSeeds: seeds,
    mvsDepth: new Float32Array(N * P).fill(4),
    mvsGeoDepth: new Float32Array(N * P).fill(4),
    mvsNormal: new Uint32Array(N * P),
    mvsCost: new Float32Array(P).fill(2),
    mvsPartial: new Float32Array(P * 27),
    streamControl: { phase: 2, start: 1, first: 0, stride: 27, costs: 3, sources: 6 },
    filterControl: {
      maxCost: 0.9,
      minConsistent: 3,
      maxError: 1,
      minAngleCos: Math.cos(Math.PI / 180),
      minConfidence: 0,
      writeConfidence: 1,
    },
  };
}

function bind(module, values) {
  for (const [name, value] of Object.entries(values)) module.setBinding(name, value);
}

function stateArrays(values) {
  return [values.mvsDepth, values.mvsGeoDepth, values.mvsNormal, values.mvsCost, values.mvsPartial];
}

function facingNormals(module, values) {
  const axisWord = module.fns.encodeNormal([0, 0, -1]);
  if (axisWord !== 0x7fff7fff) throw new Error('Shader -Z octahedral word changed');
  const axis = module.fns.decodeNormal(axisWord);
  if (!(axis[2] < -0.999)) throw new Error('Shader axis normal does not face the camera');
  const words = [
    [0, 0, -1],
    [0.06, -0.03, -1],
    [-0.08, 0.05, -1],
  ].map((normal) => {
    const length = Math.hypot(...normal);
    const word = module.fns.encodeNormal(normal.map((value) => value / length));
    const decoded = module.fns.decodeNormal(word);
    // The largest reference ray coordinates in these fixtures are below 0.5 in magnitude.
    if (!(decoded[2] + 0.5 * (Math.abs(decoded[0]) + Math.abs(decoded[1])) < 0))
      throw new Error('Varied shader normal does not face every fixture ray');
    return word;
  });
  for (let i = 0; i < values.mvsNormal.length; i++) values.mvsNormal[i] = words[i % words.length];
}

export function mvsReferencePatchCases(compiler, source) {
  const parsed = compiler.compile(instrumentCoverage(source) + legacyPropagation(source) + PROBE);
  const errors = parsed.diagnostics.filter((d) => d.category === 'error');
  if (errors.length) throw new Error(JSON.stringify(errors));
  const rows = [];
  for (const precision of ['f64', 'f32']) {
    const module = compiler.compileModuleJs(parsed.module, { precision });
    const coverage = new Uint32Array(9);
    module.setBinding('referencePatchCoverage', coverage);
    let probes = 0,
      validSourceCosts = 0,
      invalidWindows = 0;
    for (let radius = 1; radius <= 5; radius++)
      for (let step = 1; step <= radius; step++) {
        for (const bilateral of [false, true]) {
          const values = bindings({ radius, step, bilateral });
          bind(module, values);
          for (const [x, y] of [
            [0, 0],
            [8, 6],
            [16, 12],
            [1, 6],
            [8, 1],
          ]) {
            const original = module.fns.referencePatchProbe(x, y, false);
            const cached = module.fns.referencePatchProbe(x, y, true);
            sameNumbers(
              original,
              cached,
              precision,
              `${precision}/${radius}/${step}/${bilateral}/${x}/${y}`,
            );
            validSourceCosts += original.slice(5).filter((cost) => cost < 2).length;
            invalidWindows += Number(original[4] === 0);
            probes++;
          }
        }
      }
    // Rebind a different reference slot, change texture validity, and alternate cached/original
    // calls in one module. This also checks that invocation-private state cannot cross calls.
    for (const texture of ['flat', 'near threshold', 'holes', 'center hole'])
      for (const reference of [0, 3]) {
        const values = bindings({ texture, reference, bilateral: true, step: 1 });
        bind(module, values);
        for (const [x, y] of [
          [0, 0],
          [8, 6],
          [16, 12],
        ]) {
          const cached = module.fns.referencePatchProbe(x, y, true);
          const original = module.fns.referencePatchProbe(x, y, false);
          sameNumbers(
            original,
            cached,
            precision,
            `${precision}/${texture}/slot${reference}/${x}/${y}`,
          );
          if (texture === 'center hole' && x === 8 && y === 6 && original[4] !== 0)
            throw new Error('Missing-center probe must reject reference statistics');
          invalidWindows += Number(original[4] === 0);
          probes++;
        }
      }
    if (!validSourceCosts || !invalidWindows)
      throw new Error('Probe fixtures lack valid costs or rejected windows');

    let invalidPropagationCalls = 0,
      invalidFallbackCalls = 0;
    for (const texture of ['flat', 'center hole'])
      for (const geometric of [0, 1])
        for (const initialCost of [1, 2, 3, NaN])
          for (const color of [0, 1]) {
            const original = bindings({ texture });
            facingNormals(module, original);
            original.mvsControl.geometric = geometric;
            original.mvsControl.color = color;
            original.mvsCost.fill(initialCost);
            const cached = structuredClone(original),
              index = 6 * 9 + 4;
            bind(module, original);
            module.fns.mvsPropagateLegacyOracle([index, 0, 0]);
            bind(module, cached);
            const cullsBefore = coverage[8];
            module.fns.mvsPropagate([index, 0, 0], [index % 64, 0, 0]);
            if (initialCost <= 2) {
              if (coverage[8] !== cullsBefore + 1) throw new Error('Invalid window did not cull');
            } else {
              if (coverage[8] !== cullsBefore)
                throw new Error('Invalid nonstandard cost was culled');
              invalidFallbackCalls++;
            }
            stateArrays(original).forEach((array, i) =>
              sameNumbers(
                array,
                stateArrays(cached)[i],
                precision,
                `${precision}/invalid/${texture}/${geometric}/${initialCost}/${color}/${i}`,
              ),
            );
            invalidPropagationCalls++;
          }

    let propagationCalls = 0,
      streamedCalls = 0,
      geometricCachedCalls = 0,
      bilateralCachedCalls = 0,
      bilateralPropagationCalls = 0,
      photometricUnweightedCachedCalls = 0,
      streamedCachedCalls = 0;
    for (const config of [{}, { step: 1, bilateral: true }, { texture: 'holes', reference: 3 }]) {
      for (const geometric of [0, 1]) {
        const original = bindings(config);
        facingNormals(module, original);
        const cached = structuredClone(original);
        original.mvsControl.geometric = cached.mvsControl.geometric = geometric;
        const halfPixels = original.mvsRange.end;
        for (const color of [0, 1]) {
          original.mvsControl.color = cached.mvsControl.color = color;
          for (const index of [
            0,
            1,
            4,
            Math.floor(halfPixels / 2),
            halfPixels - 1,
            halfPixels,
            halfPixels + 2,
          ]) {
            bind(module, original);
            module.fns.mvsPropagateLegacyOracle([index, 0, 0]);
            bind(module, cached);
            const cachedBefore = coverage[2];
            module.fns.mvsPropagate([index, 0, 0], [index % 64, 0, 0]);
            if (geometric) geometricCachedCalls += coverage[2] - cachedBefore;
            if (original.mvsControl.bilateral) {
              bilateralCachedCalls += coverage[2] - cachedBefore;
              bilateralPropagationCalls++;
            } else if (!geometric) photometricUnweightedCachedCalls += coverage[2] - cachedBefore;
            stateArrays(original).forEach((array, i) =>
              sameNumbers(
                array,
                stateArrays(cached)[i],
                precision,
                `${precision}/propagation/${geometric}/${color}/${index}/${i}`,
              ),
            );
            propagationCalls++;
          }
        }
        for (const phase of [0, 1, 2, 3, 4, 5]) {
          for (const values of [original, cached]) {
            values.streamControl.phase = phase;
            values.streamControl.stride = phase === 2 ? 27 : phase === 5 ? 4 : 3;
            values.mvsRange.end = phase === 2 || phase === 3 ? halfPixels : 17 * 13;
            values.mvsPartial.fill(0);
            if (phase === 2) values.mvsCost.fill(2);
          }
          for (const start of [1, 0]) {
            original.streamControl.start = cached.streamControl.start = start;
            for (const index of [
              0,
              1,
              Math.floor(halfPixels / 2),
              halfPixels - 1,
              original.mvsRange.end,
            ]) {
              bind(module, original);
              module.fns.mvsStreamCostsOracle([index, 0, 0]);
              bind(module, cached);
              const cachedBefore = coverage[2];
              module.fns.mvsStreamCosts([index, 0, 0], [index % 64, 0, 0]);
              streamedCachedCalls += coverage[2] - cachedBefore;
              sameNumbers(
                original.mvsPartial,
                cached.mvsPartial,
                precision,
                `${precision}/stream/${geometric}/${phase}/${start}/${index}`,
              );
              streamedCalls++;
            }
          }
          if (phase === 2) {
            for (const index of [
              0,
              1,
              Math.floor(halfPixels / 2),
              halfPixels - 1,
              original.mvsRange.end,
            ]) {
              bind(module, original);
              module.fns.mvsStreamSelect([index, 0, 0]);
              bind(module, cached);
              module.fns.mvsStreamSelect([index, 0, 0]);
              stateArrays(original).forEach((array, i) =>
                sameNumbers(
                  array,
                  stateArrays(cached)[i],
                  precision,
                  `${precision}/stream-select/${geometric}/${index}/${i}`,
                ),
              );
            }
          }
        }
      }
    }
    rows.push({
      precision,
      probes,
      validSourceCosts,
      invalidWindows,
      propagationCalls,
      invalidPropagationCalls,
      invalidFallbackCalls,
      streamedCalls,
      coverage: {
        windowNeighborsEvaluated: coverage[0],
        windowNeighborsAccepted: coverage[1],
        cachedSourceCalls: coverage[2],
        uncachedSourceCalls: coverage[3],
        streamedNeighborsEvaluated: coverage[4],
        streamedNeighborsAccepted: coverage[5],
        invalidStatsCachedCalls: coverage[6],
        invalidWindowsCulled: coverage[8],
        geometricCachedCalls,
        bilateralCachedCalls,
        bilateralPropagationCalls,
        photometricUnweightedCachedCalls,
        streamedCachedCalls,
      },
    });
  }
  return rows;
}
