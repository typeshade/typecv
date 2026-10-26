import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

export const LEGACY_MVS_SHA256 = 'daa5201e4064498123e318e33bdfa8d9e889670197b0fff65e5b979d596a68ef';

function replaceOnce(source, before, after) {
  assert.equal(source.split(before).length, 2, `unique compatibility site: ${before.slice(0, 80)}`);
  return source.replace(before, after);
}

function removeBlock(source, marker) {
  assert.equal(source.split(marker).length, 2, `unique feature block: ${marker}`);
  const start = source.indexOf(marker),
    open = source.indexOf('{', start);
  let depth = 1,
    end = open + 1;
  for (; depth && end < source.length; end++) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
  }
  assert.equal(depth, 0);
  assert.equal(source[end], '\n');
  return source.slice(0, start) + source.slice(end + 1);
}

/** Restore the immutable e70f70a shader, not a second JS implementation of the search. */
export function legacyRefinementSource(source) {
  let old = replaceOnce(
    source,
    '  // 1: also test depth-only and normal-only local perturbations from the same best anchor.\n' +
      '  independentRefinement: u32;\n',
    '',
  );
  old = replaceOnce(
    old,
    '  const anchorDepth: f32 = bestDepth;\n  const anchorWord: u32 = bestWord;\n',
    '',
  );
  old = removeBlock(old, "  // Reuse exactly the joint candidate's draws.");
  old = removeBlock(old, "    // The phase's state is the fixed anchor for all groups");
  old = replaceOnce(
    old,
    '  if (streamControl.phase === 3 && mvsControl.independentRefinement === 1) return 3;\n',
    '',
  );
  old = removeBlock(old, '    if (mvsControl.independentRefinement === 1) {');
  assert.equal(createHash('sha256').update(old).digest('hex'), LEGACY_MVS_SHA256);
  return old;
}

function replaceFunctionBody(source, name, body) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, name);
  const open = source.indexOf('{', start);
  let depth = 1,
    end = open + 1;
  for (; depth && end < source.length; end++) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
  }
  assert.equal(depth, 0);
  return source.slice(0, open + 1) + '\n' + body + '\n' + source.slice(end - 1);
}

const ANALYTICAL_COST = `
class RefinementTarget {
  depth: f32;
  depthWeight: f32;
  normalWeight: f32;
  mode: u32;
  normal: vec3;
  anchorDepth: f32;
  anchorNormal: vec3;
}
declare const refinementTarget: uniform<RefinementTarget>;
function refinementAnalyticalCost(depth: f32, n: vec3): f32 {
  if (refinementTarget.mode !== 0) {
    const d0: bool = abs(depth - refinementTarget.anchorDepth) < 0.00001;
    const dp: bool = abs(depth - refinementTarget.depth) < 0.00001;
    const n0: bool = length(n - refinementTarget.anchorNormal) < 0.00001;
    const np: bool = length(n - refinementTarget.normal) < 0.00001;
    if (d0 && n0) return select(1., 0.25, refinementTarget.mode === 3);
    if ((dp && np) || (dp && n0) || (d0 && np)) {
      if (refinementTarget.mode === 2 || refinementTarget.mode === 3) return 0.25;
      if (dp && np) return 0.75;
      if (dp && n0) return 0.5;
      return 0.25;
    }
    return 1.9;
  }
  const d: f32 = depth - refinementTarget.depth;
  const dn: vec3 = n - refinementTarget.normal;
  return min(1.9, refinementTarget.depthWeight * d * d + refinementTarget.normalWeight * dot(dn, dn));
}
`;

/** Only the source-cost oracle changes; actual selection, RNG, cache and streamed kernels run. */
export function refinementAnalyticalSource(source) {
  for (const name of ['sourceCost', 'sourceCostCached', 'termCost'])
    source = replaceFunctionBody(source, name, '  return refinementAnalyticalCost(X.z, n);');
  return source + ANALYTICAL_COST;
}

export const REFINEMENT_WIDTH = 33;
export const REFINEMENT_HEIGHT = 25;
export const REFINEMENT_SENTINEL = -12345.25;

/** A small real shader binding set; only the selected red pixel(s) initially hold a plane. */
export function refinementFixture(module, options = {}) {
  const W = REFINEMENT_WIDTH,
    H = REFINEMENT_HEIGHT,
    P = W * H,
    N = 3,
    x = 16,
    y = 12,
    element = y * Math.ceil(W / 2) + x / 2,
    count = options.count ?? 1,
    pixel = y * W + x,
    near = 2,
    far = 8,
    cameras = new Float32Array(N * 24),
    gray = new Float32Array(N * P),
    depth = new Float32Array(N * P),
    geometricDepth = new Float32Array(N * P),
    normal = new Uint32Array(N * P),
    cost = new Float32Array(P).fill(2),
    word = module.fns.encodeNormal([0, 0, -1]);
  for (let s = 0; s < N; s++) {
    const cx = s === 0 ? 0 : s === 1 ? 0.6 : -0.6;
    cameras.set(
      [1, 0, 0, 0, 1, 0, 0, 0, 1, -cx, 0, 0, 30, 30, 16, 12, W, H, near, far, cx, 0, 0, s],
      s * 24,
    );
    for (let row = 0; row < H; row++)
      for (let col = 0; col < W; col++)
        gray[s * P + row * W + col] = 0.5 + 0.2 * Math.sin(col * 0.6 + row * 0.7);
    if (s !== 0) depth.fill(4, s * P, (s + 1) * P);
  }
  for (let k = 0; k < count; k++) {
    depth[pixel + 2 * k] = geometricDepth[pixel + 2 * k] = 4;
    normal[pixel + 2 * k] = word;
  }
  const flag = options.independentRefinement ?? 0,
    bindings = {
      mvsControl: {
        width: W,
        height: H,
        reference: 0,
        color: 0,
        iteration: 1,
        seed: options.seed ?? 117,
        bestSources: 2,
        windowRadius: 5,
        windowStep: 2,
        axisSamples: 6,
        minSamples: 9,
        bilateral: 0,
        boundaryAware: 0,
        minFiniteGain: 0,
        spatialFalloff: 1 / 50,
        colorFalloff: 12.5,
        minAngleCos: Math.cos(Math.PI / 180),
        perturbDepth: options.perturbDepth ?? 0.01,
        perturbAngle: options.perturbAngle ?? 0.5,
        geometric: options.geometric ?? 0,
        geometricWeight: 0.3,
        geometricMaxCost: 3,
        view: 0,
        state: 0,
        relativePoseCache: 0,
        independentRefinement: flag,
      },
      mvsRange: { first: element, end: element + count },
      mvsCameras: cameras,
      mvsSources: Uint32Array.of(2, 1, 2, 0, 0, 0, 0),
      mvsGray: gray,
      mvsDepth: depth,
      mvsGeoDepth: geometricDepth,
      mvsNormal: normal,
      mvsCost: cost,
      mvsSeeds: new Float32Array(3),
      mvsPartial: new Float32Array(count * 9 * 2 + 4).fill(REFINEMENT_SENTINEL),
      streamControl: {
        phase: 3,
        start: 1,
        first: element,
        stride: flag === 1 ? 6 : 2,
        costs: 2,
        sources: 2,
      },
      filterControl: {
        maxCost: 0.9,
        minConsistent: 2,
        maxError: 1,
        minAngleCos: Math.cos((3 * Math.PI) / 180),
        minConfidence: 0,
        writeConfidence: 0,
      },
    };
  return { id: 'refinement', bindings, pixel, element, count, anchorDepth: 4, anchorWord: word };
}

export function bindRefinement(module, bindings) {
  for (const [key, value] of Object.entries(bindings)) module.setBinding(key, value);
}

/** Commands use actual production entries and can also be serialized by the WebGPU driver. */
export function refinementCalls(bindings, path, reverse = false) {
  const count = bindings.mvsRange.end - bindings.mvsRange.first;
  if (path !== 'stream')
    return [
      { entry: path === 'cached' ? 'mvsPropagate' : 'mvsPropagateOracle', updates: {}, count },
    ];
  const calls = [],
    L = bindings.mvsControl.bestSources,
    groups = reverse ? [2, 1] : [1, 2];
  for (const phase of [2, 3]) {
    const stride = L * (phase === 2 ? 9 : bindings.mvsControl.independentRefinement === 1 ? 3 : 1);
    for (let g = 0; g < groups.length; g++)
      calls.push({
        entry: 'mvsStreamCostsOracle',
        updates: {
          mvsSources: Uint32Array.of(1, groups[g], 0, 0, 0, 0, 0),
          streamControl: {
            phase,
            start: g === 0 ? 1 : 0,
            first: bindings.mvsRange.first,
            stride,
            costs: L,
            sources: 2,
          },
        },
        count,
      });
    calls.push({ entry: 'mvsStreamSelect', updates: {}, count });
  }
  return calls;
}

export function runRefinementCalls(module, bindings, calls) {
  for (const call of calls) {
    Object.assign(bindings, structuredClone(call.updates));
    bindRefinement(module, bindings);
    // Include a masked tail lane; it must not write the next pixel or partial row.
    for (let lane = 0; lane <= call.count; lane++)
      module.fns[call.entry]([lane, 0, 0], [lane % 64, 0, 0]);
  }
  return bindings;
}

export function refinementState(bindings) {
  return [bindings.mvsDepth, bindings.mvsGeoDepth, bindings.mvsNormal, bindings.mvsCost];
}

export function refinementCandidates(module, fixture) {
  bindRefinement(module, fixture.bindings);
  const joint = module.fns.streamHypothesis(fixture.element, 0),
    normal = module.fns.decodeNormal(joint.word),
    anchorNormal = module.fns.decodeNormal(fixture.anchorWord),
    randomDepth = module.fns.randomDepth(fixture.pixel, 0),
    ray = module.fns.referenceRay(16, 12),
    randomWord = module.fns.encodeNormal(module.fns.randomNormal(fixture.pixel, 1, ray));
  return {
    jointDepth: joint.depth,
    jointWord: joint.word,
    normal,
    anchorNormal,
    randomDepth,
    randomNormal: module.fns.decodeNormal(randomWord),
  };
}

/** Closed-form component losses establish the winners without duplicating the search. */
export function refinementAnalyticalFixture(module, kind, options = {}) {
  const f = refinementFixture(module, options),
    c = refinementCandidates(module, f),
    normalDistance = (a, b) => a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0),
    dd = (c.jointDepth - f.anchorDepth) ** 2,
    nd = normalDistance(c.normal, c.anchorNormal),
    target = {
      depth: c.jointDepth,
      normal: c.normal,
      anchorDepth: f.anchorDepth,
      anchorNormal: c.anchorNormal,
      depthWeight: 1,
      normalWeight: 1,
      mode: 0,
    };
  assert.ok(dd > 1e-9 && nd > 1e-9, 'fixed draws perturb both components');
  if (kind === 'depth-only') {
    target.normal = c.anchorNormal;
    target.normalWeight = Math.max(
      (4 * dd) / nd,
      (4 * dd) / normalDistance(c.randomNormal, c.anchorNormal),
    );
  } else if (kind === 'normal-only') {
    target.depth = f.anchorDepth;
    target.depthWeight = Math.max((4 * nd) / dd, (4 * nd) / (c.randomDepth - f.anchorDepth) ** 2);
  } else target.mode = kind === 'anchor-order' ? 1 : kind === 'joint-tie' ? 2 : 3;
  f.id = kind;
  f.bindings.refinementTarget = target;
  bindRefinement(module, f.bindings);
  const stats = module.fns.referenceWindow(16, 12);
  f.bindings.mvsCost[f.pixel] = module.fns.hypothesisCost(
    16,
    12,
    f.anchorDepth,
    c.anchorNormal,
    stats,
  );
  f.expected = {
    depth: kind === 'depth-only' || kind === 'joint-tie' ? c.jointDepth : f.anchorDepth,
    word: kind === 'depth-only' || kind === 'anchor-tie' ? f.anchorWord : c.jointWord,
    jointDepth: c.jointDepth,
    jointWord: c.jointWord,
  };
  return f;
}

// Independent physical ray/plane renderer. It does not call the shader's warping/cost code.
export function renderRefinementPlane(bindings, depth, normal) {
  const W = REFINEMENT_WIDTH,
    H = REFINEMENT_HEIGHT,
    P = W * H,
    plane = normal[2] * depth;
  for (let s = 0; s < 3; s++) {
    const cx = bindings.mvsCameras[s * 24 + 20];
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const u = (x - 16) / 30,
          v = (y - 12) / 30,
          z = (plane - normal[0] * cx) / (normal[0] * u + normal[1] * v + normal[2]),
          X = cx + u * z,
          Y = v * z;
        bindings.mvsGray[s * P + y * W + x] =
          0.5 +
          0.18 * Math.sin(1.7 * X + 0.8 * Y) +
          0.12 * Math.cos(1.1 * X - 2.3 * Y) +
          0.08 * Math.sin(3.1 * X + 1.4 * Y);
        if (s !== 0) bindings.mvsDepth[s * P + y * W + x] = z;
      }
  }
}

/** Physical NCC cases: the current plane already has one exact component. The depth-only
 * case uses a custom depth/angle pair to isolate that component; the normal-only case uses
 * the host's first iteration (depth fraction 1/2, angle pi/12). These are kernel contract
 * probes, not claims about the shipped search's quality on arbitrary images.
 */
export function refinementPlaneFixture(module, kind, options = {}) {
  const f = refinementFixture(module, {
      ...options,
      seed: options.seed ?? 0,
      perturbDepth: options.perturbDepth ?? (kind === 'depth-only' ? 0.01 : 0.5),
      perturbAngle: options.perturbAngle ?? Math.PI / 12,
    }),
    c = refinementCandidates(module, f),
    depth = kind === 'depth-only' ? c.jointDepth : f.anchorDepth,
    normal = kind === 'depth-only' ? c.anchorNormal : c.normal;
  // A fronto-parallel correct plane has exact integer source translations, avoiding an
  // interpolation-error oracle. The tilted plane uses the same independent physical renderer.
  if (kind === 'depth-only')
    for (const s of [1, 2]) {
      const cx = ((s === 1 ? 3 : -3) * depth) / 30;
      f.bindings.mvsCameras[s * 24 + 9] = -cx;
      f.bindings.mvsCameras[s * 24 + 20] = cx;
    }
  renderRefinementPlane(f.bindings, depth, normal);
  bindRefinement(module, f.bindings);
  const stats = module.fns.referenceWindow(16, 12),
    cost = (d, n) => module.fns.hypothesisCost(16, 12, d, n, stats);
  f.id = 'physical-' + kind;
  f.bindings.mvsCost[f.pixel] = cost(f.anchorDepth, c.anchorNormal);
  f.oracle = {
    perturbDepth: f.bindings.mvsControl.perturbDepth,
    perturbAngle: f.bindings.mvsControl.perturbAngle,
    hostFirstIteration: kind === 'normal-only',
    depth,
    normal,
    anchorCost: cost(f.anchorDepth, c.anchorNormal),
    jointCost: cost(c.jointDepth, c.normal),
    independentCost: cost(depth, normal),
    randomCost: cost(c.randomDepth, c.randomNormal),
  };
  f.expected = {
    depth,
    word: kind === 'depth-only' ? f.anchorWord : c.jointWord,
    jointDepth: c.jointDepth,
    jointWord: c.jointWord,
  };
  return f;
}
