/**
 * The cases of the hardware check (`docs/lab-run-records.md`, "Hardware check"). A case runs one
 * family of kernels on a small fixed input, through the site's runtime: `api.gpu` is
 * `/runtime/gpu.js` and `api.internal` is `/runtime/internal/gpu.js` in the browser, and
 * `dist/gpu.js` and `dist/internal/gpu.js` in Node. Each case gives its answer, the input that the
 * answer comes from, and the comparison of a device's answer with the reference and the bound of
 * the family's WebGPU test (`bound.source`). `run.ts` runs a case on the device and on the CPU
 * tier and digests both answers. The answer holds no tier name, plan report or time, so that the
 * CPU tier's digest in a browser equals Node's (`tests/digests/hardware-check.json`).
 *
 * The inputs of the family tests are reused where they are free of the JavaScript engine's own
 * functions. The CLAHE input of the test uses `Math.sin` and `Math.cos`, whose last bit may differ
 * between engines, so the check builds its own CLAHE input from integer and rational arithmetic
 * only.
 */
import { planarDepthCase, planarInput } from '../../../../tests/typeshade/planar-depth-cases.mjs';
import { cleanCloud } from '../../../../tests/typeshade/cloud-clean-cases.mjs';
import {
  fixture as epipolarFixture,
  reference as epipolarReference,
} from '../../../../tests/typeshade/cases.mjs';
import type { HardwareCheckBound } from './record.ts';

/** The probe's job: `tests/fixtures/south-building-pair-triangulation.json`, as the site serves it. */
export interface TriangulationFixture {
  readonly options: { readonly maxErrorPx: number; readonly minAngleDeg: number };
  readonly views: readonly {
    readonly camera: Readonly<Record<string, number>>;
    readonly pose: { readonly R: readonly number[]; readonly t: readonly number[] };
  }[];
  readonly pixels: readonly (readonly number[])[];
  readonly expectedStatus: readonly number[];
}
/** What a case needs: the two runtime entries and the triangulation fixture. */
export interface CheckApis {
  readonly gpu: any;
  readonly internal: any;
  readonly fixture: TriangulationFixture;
}
/** The comparison of a device's answer with the reference: the record's `agreement` and more. */
export interface CaseComparison {
  readonly pass: boolean;
  readonly maxAbs: number | null;
  readonly maxRel: number | null;
  readonly mismatches: number;
  readonly compared: number;
  readonly problems: readonly string[];
  readonly summary: Readonly<Record<string, number>>;
}
export interface CheckCaseDefinition {
  readonly name: string;
  /** The kernel entries the case runs. */
  readonly family: readonly string[];
  /** The runtime entry whose `ShadeDevice` runs the case. */
  readonly entry: 'gpu' | 'internal';
  readonly bound: HardwareCheckBound;
  /** The case's input, for its digest. */
  input(api: CheckApis): unknown;
  /** The answer on `device`. */
  run(api: CheckApis, device: any): Promise<unknown>;
  /** A device's answer against the reference and the CPU tier's answer. */
  compare(api: CheckApis, device: any, cpu: any): CaseComparison;
}

/** The active lanes of the last workgroup of 64 lanes, for `n` invocations. */
const lastLanes = (n: number) => n % 64 || 64;
const comparison = (
  values: Partial<CaseComparison> & { pass: boolean; compared: number },
): CaseComparison => ({
  maxAbs: null,
  maxRel: null,
  mismatches: 0,
  summary: {},
  ...values,
  problems: (values.problems ?? []).slice(0, 8),
});

/**
 * The CLAHE input: 131 x 97 pixels (12707, so the last workgroup of 64 is partial), a bright and
 * low-contrast image of two triangle waves with the test's linear congruential noise.
 */
function claheInput() {
  const width = 131,
    height = 97,
    gray = new Float32Array(width * height);
  // A triangle wave in [-1, 1] of the given period, from integer arithmetic.
  const wave = (t: number, period: number) =>
    (4 * Math.abs((t % period) - period / 2)) / period - 1;
  let state = 99;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      gray[y * width + x] =
        0.82 + 0.03 * wave(x, 56) * wave(y, 44) + 0.012 * (((state >>> 8) % 1000) / 1000 - 0.5);
    }
  return { width, height, gray };
}
const CLAHE_OPTIONS = [true, { tiles: [5, 3], clipLimit: 0.02 }] as const;

/**
 * A planar-depth input whose pixel count is not a multiple of 64 (18 x 13 = 234): the 'plane'
 * mode of `planarInput` in `tests/typeshade/planar-depth-cases.mjs`, with a hole of 6 x 6 pixels in
 * the first view. The normal is packed without `Math.hypot`: the octahedral form divides by the
 * sum of the components, so the length cancels.
 */
function planarPartialInput() {
  const width = 18,
    height = 13,
    P = width * height,
    cx = (width - 1) / 2,
    cy = (height - 1) / 2;
  const sum = 0.15 + 0.08 + 1;
  const snorm = (v: number) => Math.round(v * 32767) & 0xffff;
  const normal = (snorm(1 - Math.abs(0.08 / sum)) | (snorm(1 - Math.abs(0.15 / sum)) << 16)) >>> 0;
  const views = [-0.1, 0, 0.1].map((center, index) => {
    const depth = new Float32Array(P);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        depth[y * width + x] =
          (2 + 0.15 * center) / (1 - (0.15 * (x - cx)) / 32 - (0.08 * (y - cy)) / 32);
    if (index === 0)
      for (let y = 4; y < 10; y++) for (let x = 5; x < 11; x++) depth[y * width + x] = 0;
    return {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      center: [center, 0, 0],
      fx: 32,
      fy: 32,
      cx,
      cy,
      depth,
      confidence: Float32Array.from(depth, (d) => (d > 0 ? 1 : 0)),
      normal: new Uint32Array(P).fill(normal),
      rgba: new Uint32Array(P).fill(0xff808080),
    };
  });
  return { width, height, views };
}

/** The modes of `planarDepthCase` (`tests/typeshade/planar-depth-cases.mjs`), in its order. */
const PLANAR_MODES = [
  'plane',
  'rotated',
  'weak-support',
  'limited-overlap',
  'occluded',
  'one-source',
  'repeated-source',
  'color-edge',
  'depth-edge',
];

/** The probe's tracks `first` to `end` as `triangulateTracksGpu` takes them. */
function tracksOf(f: TriangulationFixture, first: number, end: number) {
  return f.pixels.slice(first, end).map((pixels) =>
    f.views.map((view, k) => ({
      camera: view.camera,
      pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
      pixel: [pixels[2 * k], pixels[2 * k + 1]],
    })),
  );
}
/** The A1 runs: a workgroup of 16 lanes with 8 idle ones, then the full workgroup that holds them. */
const A1_RUNS = [
  ['8 idle lanes, simplest variant', 1664, 1672, { refinementIterations: 0, maxPairHypotheses: 1 }],
  ['8 idle lanes, the lab job', 1664, 1672, {}],
  ['full workgroup, the lab job', 1656, 1672, {}],
] as const;

/** The voxel representatives as member lists: the owner of a voxel depends on the scheduling. */
function partition(rep: ArrayLike<number>): string[] {
  const groups = new Map<number, number[]>();
  for (let i = 0; i < rep.length; i++) groups.set(rep[i], [...(groups.get(rep[i]) ?? []), i]);
  return [...groups.values()].map((g) => g.join()).sort();
}

export const CASES: readonly CheckCaseDefinition[] = [
  {
    name: 'clahe',
    family: ['claheHistogram', 'claheMapTiles', 'claheApply'],
    entry: 'gpu',
    bound: { kind: 'abs', value: 2.5e-4, source: 'tests/typeshade/clahe-webgpu.test.mjs' },
    input: () => ({ ...claheInput(), options: CLAHE_OPTIONS }),
    async run(api, device) {
      const { width, height, gray } = claheInput();
      const images: Float32Array[] = [];
      for (const options of CLAHE_OPTIONS)
        images.push(
          Float32Array.from(await api.gpu.equalizeGrayGpu(device, gray, width, height, options)),
        );
      return { images };
    },
    compare(api, device, cpu) {
      const { width, height, gray } = claheInput();
      let worst = 0,
        compared = 0;
      const problems: string[] = [];
      CLAHE_OPTIONS.forEach((options, k) => {
        const reference = api.gpu.equalizeGrayCpu(gray, width, height, options);
        for (let i = 0; i < reference.length; i++) {
          const d = Math.abs(device.images[k][i] - reference[i]);
          if (!(d <= worst)) worst = Number.isFinite(d) ? d : Infinity;
          compared++;
        }
        if (device.images[k].length !== cpu.images[k].length) problems.push('image size ' + k);
      });
      if (!Number.isFinite(worst)) problems.push('a value is not finite');
      return comparison({
        pass: problems.length === 0 && worst < 2.5e-4,
        maxAbs: Number.isFinite(worst) ? worst : null,
        compared,
        problems,
        summary: { pixels: width * height, lastWorkgroupLanes: lastLanes(width * height) },
      });
    },
  },
  {
    name: 'planar-depth',
    family: ['planarFit', 'planarPropose', 'planarVerify', 'planarFinish'],
    entry: 'internal',
    bound: { kind: 'abs', value: 1e-4, source: 'tests/typeshade/planar-depth-webgpu.test.mjs' },
    input: () => ({
      modes: PLANAR_MODES.map((mode) => planarInput(mode).input),
      partial: planarPartialInput(),
    }),
    async run(api, device) {
      // The family test's case opens a device of its own; it gets the check's device instead.
      const r: any = await planarDepthCase(
        { ...api.internal, ShadeDevice: { open: async () => device } },
        device.backend,
      );
      const partial = await api.internal.completePlanarDepths(device, planarPartialInput());
      const view = partial.input.views[0];
      return {
        modes: r.results.map((row: any) => ({
          mode: row.mode,
          filled: row.filled,
          worstRelativeError: row.worstRelativeError,
          worstNormalError: row.worstNormalError,
          changedExisting: row.changedExisting,
          changedInput: row.changedInput,
          depth: Float32Array.from(row.depth),
          confidence: Float32Array.from(row.confidence),
        })),
        rules: {
          budgetRejected: r.budgetRejected,
          cancelled: r.cancelled,
          selectedFilled: r.selectedFilled,
          boundedFilled: r.boundedFilled,
          liveBytes: r.liveBytes,
        },
        partial: {
          filled: partial.report.filled,
          depth: Float32Array.from(view.depth),
          confidence: Float32Array.from(view.confidence),
          normal: Uint32Array.from(view.normal),
        },
      };
    },
    compare(_api, r, cpu) {
      const problems: string[] = [];
      const rules = r.rules;
      if (rules.liveBytes !== 0) problems.push('live bytes after the case');
      if (!rules.budgetRejected) problems.push('a budget of 1 byte was not refused');
      if (!rules.cancelled) problems.push('an aborted signal was not honoured');
      if (rules.selectedFilled !== 0) problems.push('selected sources filled');
      if (rules.boundedFilled !== 36)
        problems.push('bounded dispatches filled ' + rules.boundedFilled);
      let worst = 0,
        compared = 0;
      const depths = (a: Float32Array, b: Float32Array, label: string) => {
        if (a.length !== b.length) return problems.push(label + ' size');
        for (let k = 0; k < a.length; k++) {
          compared++;
          if (a[k] > 0 !== b[k] > 0) problems.push(label + ' hole ' + k);
          const d = Math.abs(a[k] - b[k]);
          if (!(d <= worst)) worst = Number.isFinite(d) ? d : Infinity;
        }
      };
      r.modes.forEach((row: any, i: number) => {
        const expected = ['plane', 'weak-support', 'rotated'].includes(row.mode)
          ? 36
          : row.mode === 'limited-overlap'
            ? 24
            : 0;
        if (row.changedInput || row.changedExisting) problems.push(row.mode + ' changed a pixel');
        if (row.filled !== expected) problems.push(row.mode + ' filled ' + row.filled);
        if (!(row.worstRelativeError < 1e-4) || !(row.worstNormalError < 1e-4))
          problems.push(row.mode + ' error');
        depths(row.depth, cpu.modes[i].depth, row.mode);
      });
      if (r.partial.filled !== cpu.partial.filled)
        problems.push('18 x 13 filled ' + r.partial.filled + ', CPU tier ' + cpu.partial.filled);
      depths(r.partial.depth, cpu.partial.depth, '18 x 13');
      return comparison({
        pass: problems.length === 0 && worst < 1e-4,
        maxAbs: Number.isFinite(worst) ? worst : null,
        compared,
        problems,
        summary: {
          modes: r.modes.length,
          filled: r.modes.reduce((n: number, row: any) => n + row.filled, 0),
          partialPixels: 18 * 13,
          partialFilled: r.partial.filled,
          lastWorkgroupLanes: lastLanes(18 * 13),
        },
      });
    },
  },
  {
    name: 'cloud-clean',
    family: ['knnMeanDistance', 'voxelClaim'],
    entry: 'internal',
    bound: { kind: 'rel', value: 1e-6, source: 'tests/typeshade/cloud-clean-webgpu.test.mjs' },
    input: () => cleanCloud().cloud,
    async run(api, device) {
      const { cloud } = cleanCloud();
      const primitives = api.internal.gpuCleanPrimitives(device);
      const distances = Float32Array.from(await primitives.meanDistances(cloud.positions, 16));
      // 3000 points: the last workgroup of the k-NN entry has 56 active lanes.
      const partial = Float32Array.from(
        await primitives.meanDistances(cloud.positions.subarray(0, 3000 * 3), 16),
      );
      const frame = api.internal.voxelFrame(cloud.positions, 0.05);
      const voxels = partition(await primitives.voxelRepresentatives(cloud.positions, frame));
      const { before, afterOutliers, afterVoxel, after, clusters } = (
        await api.internal.cleanCloudWith(primitives, cloud)
      ).report;
      return {
        distances,
        partial,
        voxels,
        report: { before, afterOutliers, afterVoxel, after, clusters },
      };
    },
    compare(_api, r, cpu) {
      const problems: string[] = [];
      let worst = 0,
        compared = 0;
      for (const key of ['distances', 'partial'] as const) {
        if (r[key].length !== cpu[key].length) problems.push(key + ' size');
        for (let i = 0; i < r[key].length; i++) {
          compared++;
          const d = Math.abs(r[key][i] - cpu[key][i]) / cpu[key][i];
          if (!(d <= worst)) worst = Number.isFinite(d) ? d : Infinity;
        }
      }
      if (r.voxels.join('|') !== cpu.voxels.join('|')) problems.push('voxel partition');
      for (const key of ['before', 'afterOutliers', 'afterVoxel', 'after', 'clusters'])
        if (r.report[key] !== cpu.report[key]) problems.push('report ' + key);
      return comparison({
        pass: problems.length === 0 && worst < 1e-6,
        maxRel: Number.isFinite(worst) ? worst : null,
        compared,
        problems,
        summary: {
          points: r.distances.length,
          partialPoints: r.partial.length,
          lastWorkgroupLanes: lastLanes(r.partial.length),
          after: r.report.after,
          clusters: r.report.clusters,
        },
      });
    },
  },
  {
    name: 'a1-triangulation',
    family: ['prepareTriangulation', 'triangulateTracks'],
    entry: 'gpu',
    bound: {
      kind: 'exact',
      value: 0,
      source: 'docs/hardware-observations.md, A1 (probe records 14349a11 and 4b85f996)',
    },
    input: (api) => ({
      options: api.fixture.options,
      views: api.fixture.views,
      pixels: api.fixture.pixels.slice(1656, 1672),
    }),
    async run(api, device) {
      const out: Record<string, { first: number; status: number[]; xyz: (number[] | null)[] }> = {};
      for (const [label, first, end, override] of A1_RUNS) {
        const result = await api.gpu.triangulateTracksGpu(
          device,
          tracksOf(api.fixture, first, end),
          { ...api.fixture.options, ...override },
        );
        out[label] = {
          first,
          status: Array.from(result.status as Uint32Array),
          xyz: result.tracks.map((t: any) => (t ? Array.from(t.xyz as Float64Array) : null)),
        };
      }
      return out;
    },
    compare(api, r, cpu) {
      const problems: string[] = [];
      let compared = 0;
      for (const [label, run] of Object.entries(r) as [string, any][])
        run.status.forEach((s: number, i: number) => {
          compared++;
          const track = run.first + i;
          if (s !== api.fixture.expectedStatus[track])
            problems.push(`${label}: track ${track} status ${s}`);
          if (s !== cpu[label].status[i]) problems.push(`${label}: track ${track} against the CPU`);
        });
      return comparison({
        pass: problems.length === 0,
        mismatches: problems.length,
        compared,
        problems,
        summary: { runs: A1_RUNS.length, idleLanes: 8 },
      });
    },
  },
  {
    name: 'kernel-functions',
    family: [
      'normalizeMatches',
      'sampsonBatch',
      'inlierFlags',
      'msacCosts',
      'blockStats',
      'modelStats',
      'selectBest',
      'winningMask',
    ],
    entry: 'gpu',
    bound: {
      kind: 'abs',
      value: 0.002,
      source: 'tests/typeshade/webgpu.test.mjs (EpipolarSession, 513 matches)',
    },
    input: () => epipolarFixture(513),
    async run(api, device) {
      // The kernel functions of `EpipolarSession`: the compiler's map lowering over 513 matches,
      // so the last workgroup is partial. At a pin without TypeShade change 0096 each lane starts
      // with the early return past the trip count; with 0096, with its iteration scaffold.
      const data = epipolarFixture(513);
      const session = await api.gpu.EpipolarSession.create(
        device,
        data.pixels,
        data.first,
        data.second,
      );
      try {
        const scored = await session.score(data.models, data.threshold, {
          readMask: true,
          readStatistics: true,
        });
        return {
          bestIndex: scored.bestIndex,
          inlierCount: scored.inlierCount,
          msacCost: scored.msacCost,
          mask: Uint8Array.from(scored.mask),
          modelStatistics: Float32Array.from(scored.modelStatistics),
        };
      } finally {
        await session.dispose();
      }
    },
    compare(_api, r) {
      const expected = epipolarReference(epipolarFixture(513));
      const problems: string[] = [];
      if (r.bestIndex !== 2) problems.push('best model ' + r.bestIndex);
      if (Array.from(r.mask).join() !== expected.masks[2].join()) problems.push('inlier mask');
      const h = expected.counts.length;
      let worst = 0;
      for (let i = 0; i < h; i++) {
        if (r.modelStatistics[i] !== expected.counts[i]) problems.push('count of model ' + i);
        const d = Math.abs(r.modelStatistics[h + i] - expected.costs[i]);
        if (!(d <= worst)) worst = Number.isFinite(d) ? d : Infinity;
      }
      return comparison({
        pass: problems.length === 0 && worst < 0.002,
        maxAbs: Number.isFinite(worst) ? worst : null,
        compared: 2 * h + r.mask.length,
        problems,
        summary: { matches: 513, lastWorkgroupLanes: lastLanes(513), models: h },
      });
    },
  },
];
