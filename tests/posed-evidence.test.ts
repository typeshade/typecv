import test from 'node:test';
import assert from 'node:assert/strict';
import { identityPose, projectPoint } from '../src/geometry/pose.ts';
import { triangulateTrackDetailed } from '../src/geometry/multiview.ts';
import {
  POSED_EVIDENCE_VERSION,
  preparePosedEvidence,
  buildPosedEvidenceGraph,
  type PosedEvidenceInput,
  type PreparedPosedEvidence,
  type PosedPairProof,
  type EvidenceExclusions,
} from '../src/reconstruction/posed-evidence.ts';
import type { PairMatches } from '../src/reconstruction/types.ts';

function fixture(): PosedEvidenceInput {
  const camera = { fx: 800, fy: 800, cx: 320, cy: 240, k1: 0, k2: 0 };
  const views = [90, 10, 40, 70].map((id, i) => ({
    id,
    camera: { ...camera },
    pose: { ...identityPose(), t: Float64Array.of(-i * 0.06, 0, 0) },
  }));
  const images = views.map((view) => ({
    id: view.id,
    camera: { ...camera },
    keypoints: Float64Array.from(
      [0.1, 0.16].flatMap((x) => [
        ...projectPoint(camera, view.pose, Float64Array.of(x, -0.15, 10))!,
      ]),
    ),
  }));
  return {
    images,
    views: [views[2]!, views[0]!, views[1]!],
    pairs: [
      {
        first: 90,
        second: 10,
        matches: Uint32Array.of(0, 0, 1, 1),
        distances: Float64Array.of(4, 7),
      },
      { first: 10, second: 40, matches: Uint32Array.of(0, 0), distances: Float64Array.of(5) },
      { first: 90, second: 10, matches: Uint32Array.of(0, 0), distances: Float64Array.of(4) },
      { first: 40, second: 70, matches: Uint32Array.of(1, 0) },
    ],
    controls: { maxErrorPx: 3, minAngleDeg: 0.5, maxPairHypotheses: 32, refinementIterations: 10 },
    limits: { maxViews: 8, maxPairs: 8, maxMatchRows: 32 },
  };
}
function exclusions(): EvidenceExclusions {
  return { currentOwned: new Set(), archiveOwned: new Set(), rejected: new Set() };
}
function proofFor(prepared: PreparedPosedEvidence): PosedPairProof {
  return {
    version: POSED_EVIDENCE_VERSION,
    inputIdentity: prepared.identity,
    observationsPerRow: 2,
    statuses: prepared.rows.map(
      (row) =>
        triangulateTrackDetailed(
          row.observations.map((o) => ({
            camera: o.camera,
            pose: { R: Float64Array.from(o.pose.R), t: Float64Array.from(o.pose.t) },
            pixel: o.pixel,
          })),
          prepared.controls,
        ).status,
    ),
  };
}

test('preparation retains every raw row/duplicate and stable source order, including unposed rows', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input);
  assert.equal(prepared.source.rows.length, 5);
  assert.equal(prepared.source.pairs.length, 4);
  assert.equal(prepared.rows.length, 4);
  assert.deepEqual(
    prepared.rows.map((row) => row.sourceRow),
    [0, 1, 2, 3],
  );
  assert.deepEqual(
    prepared.source.rows.map((row) => [row.pairIndex, row.matchRow]),
    [
      [0, 0],
      [0, 1],
      [1, 0],
      [2, 0],
      [3, 0],
    ],
  );
  assert.deepEqual(
    prepared.source.views.map((view) => view.id),
    [90, 10, 40, 70],
  );
  assert.equal(prepared.source.views.find((view) => view.id === 70)!.registered, undefined);
  assert.equal(prepared.source.rows[4]!.second.viewId, 70);
});

test('actual low-angle pair proofs retain a three-view candidate chain without admitting depth', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  assert.deepEqual(Array.from(proof.statuses), [4, 4, 4, 4]);
  const result = buildPosedEvidenceGraph(prepared, input, proof, exclusions());
  assert.equal(result.authority, 'candidate-only');
  assert.equal(result.rows.filter((row) => row.candidate).length, 4);
  assert.equal(result.graph.redundantEdges, 1);
  const track = result.graph.tracks.find((t) => t.observations.length === 3)!;
  assert.deepEqual(
    track.observations.map((o) => `${o.viewId}:${o.featureId}`),
    ['10:0', '40:0', '90:0'],
  );
  assert.equal('xyz' in track, false);
  assert.equal('landmarks' in result.graph, false);
  const lookup = new Map(input.views.map((view) => [view.id, view]));
  const numerical = triangulateTrackDetailed(
    track.observations.map((o) => {
      const view = lookup.get(o.viewId)!,
        image = input.images.find((v) => v.id === o.viewId)!;
      return {
        camera: view.camera,
        pose: view.pose,
        pixel: [image.keypoints[o.featureId * 2]!, image.keypoints[o.featureId * 2 + 1]!] as const,
      };
    }),
    input.controls,
  );
  assert.ok(numerical.status === 0, 'Only a separate robust multiview result can establish depth');
});

test('missing/status1/status3 proofs remain pending and status0/4 only supplies candidate edges', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const pending = buildPosedEvidenceGraph(prepared, input, undefined, exclusions());
  assert.deepEqual(
    pending.rows.map((row) => row.status),
    [null, null, null, null],
  );
  assert.equal(pending.graph.tracks.length, 0);
  const mixed = buildPosedEvidenceGraph(
    prepared,
    input,
    { ...proof, statuses: [0, 1, 3, 4] },
    exclusions(),
  );
  assert.deepEqual(
    mixed.rows.map((row) => row.candidate),
    [true, false, false, true],
  );
  assert.deepEqual(mixed.evidence.source, pending.evidence.source);
});

test('fresh current/archive ownership and rejection exclude rows without deleting source evidence', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const original = buildPosedEvidenceGraph(prepared, input, proof, exclusions());
  const blocked = buildPosedEvidenceGraph(prepared, input, proof, {
    currentOwned: new Set(['90:0']),
    archiveOwned: new Set(['90:1']),
    rejected: new Set(['40:0']),
  });
  assert.equal(blocked.graph.tracks.length, 0);
  assert.equal(blocked.rows[0]!.excluded.currentOwned, true);
  assert.equal(blocked.rows[1]!.excluded.archiveOwned, true);
  assert.equal(blocked.rows[2]!.excluded.rejected, true);
  assert.deepEqual(blocked.evidence.source, original.evidence.source);
  assert.deepEqual(buildPosedEvidenceGraph(prepared, input, proof, exclusions()), original);
});

test('a compatible conflicting bridge does not persist a union or erase the original row', () => {
  const base = fixture(),
    input = {
      ...base,
      pairs: [
        ...base.pairs,
        { first: 90, second: 40, matches: Uint32Array.of(0, 1), distances: Float64Array.of(1) },
      ],
    };
  const prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  assert.ok(Array.from(proof.statuses).every((status) => status === 4));
  const conflicted = buildPosedEvidenceGraph(prepared, input, proof, exclusions());
  assert.ok(conflicted.graph.conflictingEdges > 0);
  assert.equal(conflicted.rows.filter((row) => row.candidate).length, prepared.rows.length);
  assert.equal(conflicted.evidence.source.rows.length, 6);
  const rebuilt = buildPosedEvidenceGraph(prepared, input, proof, {
    ...exclusions(),
    rejected: new Set(['90:0']),
  });
  assert.equal(rebuilt.graph.conflictingEdges, 0);
  assert.deepEqual(rebuilt.evidence.source, conflicted.evidence.source);
  assert.notDeepEqual(rebuilt.graph, conflicted.graph);
});

test('proof shape/status/version and exact two-observation binding fail closed', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  for (const invalid of [
    { ...proof, version: 'wrong-version' },
    { ...proof, inputIdentity: proof.inputIdentity + 'changed' },
    { ...proof, observationsPerRow: 3 },
    { ...proof, statuses: [4] },
    { ...proof, statuses: [4, 4, 2, 4] },
    { ...proof, statuses: [4, 4, NaN, 4] },
    { ...proof, statuses: new Array(4) },
  ])
    assert.throws(
      () => buildPosedEvidenceGraph(prepared, input, invalid as PosedPairProof, exclusions()),
      RangeError,
    );
});

test('proof statuses copy only bounded indexed entries without consuming an iterator', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const statuses = {
    length: 4,
    0: 0,
    1: 4,
    2: 0,
    3: 4,
    [Symbol.iterator]() {
      throw new Error('Proof iterator must not be consumed');
    },
  };
  const result = buildPosedEvidenceGraph(prepared, input, { ...proof, statuses }, exclusions());
  assert.deepEqual(
    result.rows.map((row) => row.status),
    [0, 4, 0, 4],
  );
  assert.ok(result.rows.every((row) => row.candidate));
});

test('pose/camera/referenced pixel/count/order/control changes reject stale asynchronous proofs', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const variants: PosedEvidenceInput[] = [];
  const moved = structuredClone(input);
  moved.views[0]!.pose.t[0] += 0.001;
  variants.push(moved);
  const camera = {
    ...input,
    views: input.views.map((view, i) =>
      i ? view : { ...view, camera: { ...view.camera, fx: view.camera.fx + 1 } },
    ),
  };
  variants.push(camera);
  variants.push({
    ...input,
    views: input.views.map((view, i) =>
      i ? view : { ...view, camera: { ...view.camera, k1: -0 } },
    ),
  });
  const distance = structuredClone(input);
  distance.pairs[0]!.distances![0] = 3.9;
  variants.push(distance);
  const pixel = structuredClone(input);
  pixel.images[0]!.keypoints[0] += 0.01;
  variants.push(pixel);
  variants.push({
    ...input,
    images: input.images.map((image, i) =>
      i ? image : { ...image, keypoints: Float64Array.of(...image.keypoints, 100, 100) },
    ),
  });
  variants.push({ ...input, pairs: [...input.pairs].reverse() });
  variants.push({
    ...input,
    pairs: input.pairs.map((pair, i) =>
      i ? pair : { ...pair, matches: Uint32Array.of(1, 1, 0, 0) },
    ),
  });
  for (const controls of [
    { ...input.controls, maxErrorPx: 4 },
    { ...input.controls, minAngleDeg: 0.6 },
    { ...input.controls, maxPairHypotheses: 16 },
    { ...input.controls, refinementIterations: 11 },
  ])
    variants.push({ ...input, controls });
  for (const current of variants)
    assert.throws(
      () => buildPosedEvidenceGraph(prepared, current, proof, exclusions()),
      /Stale posed evidence/,
    );
});

test('unreferenced finite pixels and registered-view array order do not change numerical proof inputs', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const current = structuredClone(input);
  current.images.find((v) => v.id === 70)!.keypoints[2] += 10;
  const reordered = { ...current, views: [...current.views].reverse() };
  assert.equal(preparePosedEvidence(reordered).identity, prepared.identity);
  assert.deepEqual(
    buildPosedEvidenceGraph(prepared, reordered, proof, exclusions()),
    buildPosedEvidenceGraph(prepared, input, proof, exclusions()),
  );
});

test('snapshots/provenance/graph have no mutable input or proof aliases', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const result = buildPosedEvidenceGraph(prepared, input, proof, exclusions());
  const originalPixel = prepared.source.rows[0]!.first.pixel[0];
  input.images[0]!.keypoints[0] += 10;
  (proof.statuses as number[])[0] = 3;
  assert.equal(prepared.source.rows[0]!.first.pixel[0], originalPixel);
  assert.equal(result.rows[0]!.status, 4);
  assert.throws(
    () => ((prepared.source.rows[0]!.first.pixel as unknown as number[])[0] = 99),
    TypeError,
  );
  assert.throws(() => ((prepared.rows[0]!.observations[0]!.pose.R as number[])[0] = 99), TypeError);
  assert.throws(() => (result.graph.tracks as unknown[]).push({}), TypeError);
});

test('explicit bounds count unposed rows and empty/duplicate pair records before copies', () => {
  const input = fixture();
  assert.throws(
    () => preparePosedEvidence({ ...input, limits: { ...input.limits, maxMatchRows: 4 } }),
    /match-row bound/,
  );
  const emptyPairs = [...input.pairs, { first: 90, second: 10, matches: new Uint32Array(0) }];
  assert.throws(
    () =>
      preparePosedEvidence({
        ...input,
        pairs: emptyPairs,
        limits: { ...input.limits, maxPairs: 4 },
      }),
    /view\/pair bounds/,
  );
  const guarded = fixture();
  Object.defineProperty(guarded.images[0], 'keypoints', {
    get() {
      throw new Error('Copied before admission');
    },
  });
  assert.throws(
    () => preparePosedEvidence({ ...guarded, limits: { ...guarded.limits, maxMatchRows: 1 } }),
    /match-row bound/,
  );
  for (const limits of [
    { ...input.limits, maxViews: 0 },
    { ...input.limits, maxPairs: 1.5 },
    { ...input.limits, maxMatchRows: Number.MAX_SAFE_INTEGER + 1 },
  ])
    assert.throws(() => preparePosedEvidence({ ...input, limits }), RangeError);
});

test('core identity/pixel/self-pair validation rejects malformed evidence without dropping it', () => {
  const input = fixture();
  const duplicateImages = { ...input, images: [...input.images, input.images[0]!] };
  const duplicatePoses = { ...input, views: [...input.views, input.views[0]!] };
  const selfPair = { ...input, pairs: [{ first: 90, second: 90, matches: Uint32Array.of(0, 0) }] };
  const badFeature = {
    ...input,
    pairs: [{ ...input.pairs[0]!, matches: Float64Array.of(-1, 0) } as unknown as PairMatches],
  };
  const badPixel = structuredClone(input);
  badPixel.images[0]!.keypoints[0] = NaN;
  for (const invalid of [duplicateImages, duplicatePoses, selfPair, badFeature, badPixel])
    assert.throws(() => preparePosedEvidence(invalid), RangeError);
});

test('retirement requires a new snapshot; the old complete source evidence remains intact', () => {
  const input = fixture(),
    prepared = preparePosedEvidence(input),
    proof = proofFor(prepared);
  const current = {
    ...input,
    images: input.images.filter((v) => v.id !== 70),
    pairs: input.pairs.filter((p) => p.first !== 70 && p.second !== 70),
  };
  assert.throws(
    () => buildPosedEvidenceGraph(prepared, current, proof, exclusions()),
    /Stale posed evidence/,
  );
  const next = preparePosedEvidence(current),
    nextProof = proofFor(next);
  assert.equal(next.source.rows.length, 4);
  assert.equal(prepared.source.rows.length, 5);
  assert.equal(
    buildPosedEvidenceGraph(next, current, nextProof, exclusions()).graph.tracks.length,
    2,
  );
});
