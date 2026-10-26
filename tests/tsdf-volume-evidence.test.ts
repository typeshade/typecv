import test from 'node:test';
import assert from 'node:assert/strict';
import {
  encodeTsdfNormal,
  fuseTsdf,
  planTsdfVolume,
  type TsdfInput,
  type TsdfView,
} from '../src/reconstruction/tsdf.ts';

const W = 8;
const P = W * W;
const OPTIONS = { visibility: 'view-weighted' } as const;

function plane(): TsdfView {
  return {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    center: [0, 0, 0],
    fx: 1000,
    fy: 1000,
    cx: 3.5,
    cy: 3.5,
    depth: new Float32Array(P).fill(2),
    normal: new Uint32Array(P).fill(encodeTsdfNormal([0, 0, -1])),
    confidence: new Float32Array(P).fill(1),
    rgba: new Uint32Array(P).fill(0xff808080),
  };
}

function remote(confidence: number): TsdfView {
  return {
    ...plane(),
    center: [1000, 1000, 0],
    depth: Float32Array.from({ length: P }, (_, i) => (i === 36 ? 2 : 0)),
    confidence: new Float32Array(P).fill(confidence),
  };
}

const input = (...views: TsdfView[]): TsdfInput => ({ width: W, height: W, views });

/** Per-view footprints grow with the view list; the existing volume must remain identical. */
function grid(volume: NonNullable<ReturnType<typeof planTsdfVolume>>) {
  const { footprints, ...rest } = volume;
  return { ...rest, existingFootprint: footprints[0] };
}

test('nonpositive-confidence samples cannot coarsen the trusted TSDF grid or change its surface', () => {
  const trusted = plane();
  const original = structuredClone(trusted);
  const baseline = input(trusted);
  const baselinePlan = planTsdfVolume(baseline, OPTIONS)!;
  const baselinePoints = fuseTsdf(baseline, OPTIONS);
  assert.equal(baselinePlan.samples, 64);
  assert.equal(baselinePlan.voxel, Math.fround(0.004));
  assert.equal(baselinePlan.keys, 903);
  assert.equal(baselinePoints.count, 16);
  for (const confidence of [0, -0, -1]) {
    const unsupported = remote(confidence);
    const before = structuredClone(unsupported);
    const candidate = input(trusted, unsupported);
    assert.deepEqual(grid(planTsdfVolume(candidate, OPTIONS)!), grid(baselinePlan));
    const points = fuseTsdf(candidate, OPTIONS);
    for (const key of [
      'positions',
      'normals',
      'colors',
      'weights',
      'free',
      'freeWeights',
      'tallies',
    ] as const)
      assert.deepEqual(points[key], baselinePoints[key], `${confidence}: ${key}`);
    assert.equal(points.count, baselinePoints.count);
    assert.deepEqual(unsupported, before);
    assert.deepEqual(trusted, original);
  }
  const absent = { ...remote(0), depth: new Float32Array(P) };
  assert.deepEqual(grid(planTsdfVolume(input(trusted, absent), OPTIONS)!), grid(baselinePlan));
  assert.deepEqual(fuseTsdf(input(trusted, absent), OPTIONS).positions, baselinePoints.positions);
  assert.deepEqual(OPTIONS, { visibility: 'view-weighted' });
});

test('unsupported positive depths cannot set the footprint percentile or the volume origin', () => {
  const trusted = plane();
  const near = {
    ...plane(),
    depth: new Float32Array(P).fill(0.001),
    confidence: new Float32Array(P),
  };
  const baseline = planTsdfVolume(input(trusted), OPTIONS)!;
  assert.deepEqual(grid(planTsdfVolume(input(trusted, near), OPTIONS)!), grid(baseline));
});

test('all nonpositive-confidence depths produce no planned volume or extracted surface', () => {
  for (const confidence of [0, -0, -1]) {
    const scene = input({ ...plane(), confidence: new Float32Array(P).fill(confidence) });
    const before = structuredClone(scene);
    assert.equal(planTsdfVolume(scene, OPTIONS), undefined);
    const points = fuseTsdf(scene, OPTIONS);
    assert.equal(points.count, 0);
    assert.equal(points.positions.length, 0);
    assert.equal(points.volume, undefined);
    assert.deepEqual(scene, before);
  }
});

test('positive or absent confidence keeps legacy supported volume planning', () => {
  const supported = remote(1);
  const baseline = planTsdfVolume(input(plane()), OPTIONS)!;
  const withSupported = planTsdfVolume(input(plane(), supported), OPTIONS)!;
  assert.equal(withSupported.samples, 65);
  assert.equal(withSupported.voxel, Math.fround(0.008));
  assert.equal(withSupported.keys, 2200079028);
  assert.ok(withSupported.voxel > baseline.voxel, 'supported remote evidence must still count');
  const { confidence: _confidence, ...implicit } = supported;
  assert.deepEqual(planTsdfVolume(input(plane(), implicit), OPTIONS), withSupported);
});

test('planning retains positive f32 confidence at rounding and grazing boundaries', () => {
  const supportedPlan = planTsdfVolume(input(plane(), remote(1)), OPTIONS)!;
  const confidences = [
    2 ** -149,
    Math.fround((0.5 - 2 ** -24) / 4096),
    Math.fround(0.5 / 4096),
    Math.fround((0.5 + 2 ** -23) / 4096),
  ];
  const normals = [
    encodeTsdfNormal([0, 0, -1]),
    encodeTsdfNormal([1, 0, 0]),
    encodeTsdfNormal([Math.sqrt(1 - 0.15 ** 2), 0, -0.15]),
  ];
  for (const confidence of confidences)
    for (const word of normals) {
      const candidate = remote(confidence);
      candidate.normal.fill(word);
      assert.ok(candidate.confidence![36]! > 0);
      // The host planner does not duplicate angle or fixed-point calculations from a shader.
      assert.deepEqual(planTsdfVolume(input(plane(), candidate), OPTIONS), supportedPlan);
    }
});
