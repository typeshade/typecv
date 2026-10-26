/**
 * Characterization digests of the sparse pipeline on the CPU path: docs/library-structure.md,
 * section 5, phase 0. A case runs one entry point on one input and records the canonical digest of
 * its whole output (tests/typeshade/digest.mjs) and of its per-request trace: the kind and the
 * sizes of each request that the mapper yields to its driver, and the name of each error that the
 * driver passes back. The tests compare the digests with fixtures in tests/digests/.
 *
 * The trace comes from a driver loop of the test (`drive`), which yields the requests of the entry
 * point's generator to the entry point's own executor with the entry point's error rule. On the
 * synthetic scenes the test also runs the entry point itself and checks that its output equals the
 * traced run's; on the samples the traced run stands in for it, to halve the time.
 *
 * tests/typeshade/digest-fixture.mjs reads and writes the fixtures: `TYPECV_DIGEST_UPDATE=1`
 * records the digests of a run, and `TYPECV_DIGEST_OUT=<directory>` writes them there instead of
 * comparing them.
 */
import assert from 'node:assert/strict';
import { reconstructIncremental, refineIncremental } from '../src/index.ts';
import type {
  ImageFeatures,
  IncrementalOptions,
  PairMatches,
  SparseReconstruction,
} from '../src/index.ts';
import {
  executeReconstructionJob,
  incrementalSteps,
  refinementModel,
} from '../src/reconstruction/mapper.ts';
import type { IncrementalResult, ReconstructionJob } from '../src/reconstruction/mapper.ts';
import {
  HostSceneDriver,
  reconstructIncrementalScene,
  sceneSteps,
} from '../src/reconstruction/scene-mapper.ts';
import type { SceneRequest } from '../src/reconstruction/scene-mapper.ts';
import { componentInput, reconstructComponent } from '../src/reconstruction/component-mapper.ts';
import { buildViewGraph, cpuViewGraphBackend } from '../src/reconstruction/view-graph.ts';
import type {
  ViewGraph,
  ViewGraphBackend,
  ViewGraphKeyframe,
  ViewGraphOptions,
} from '../src/reconstruction/view-graph.ts';
import { focalSearchControls, focalSearchSteps } from '../src/reconstruction/focal-search.ts';
import type { FocalSearchOptions } from '../src/reconstruction/focal-search.ts';
import { camerasAt, LAB, loadSample, SAMPLES, type Sample } from './digest-samples.ts';
import { digest, digestParts } from './typeshade/digest.mjs';
import { scene } from './scene.ts';
import { refinementFixture } from './fixtures/refinement.ts';
import { pathScene } from './view-graph-scene.ts';

/** One step of a trace: a request's kind and sizes, or `throw` and the error's name. */
export type TraceStep = readonly (string | number | boolean)[];
export interface CaseRun {
  readonly output: unknown;
  readonly trace: readonly TraceStep[];
}
export interface Case {
  readonly name: string;
  /** The traced run. */
  readonly run: () => Promise<CaseRun> | CaseRun;
  /** The entry point's own run, whose output must equal the traced run's. */
  readonly direct?: () => Promise<unknown> | unknown;
}
/** What a fixture records of one case. */
export interface CaseDigest {
  readonly output: string;
  readonly parts: Readonly<Record<string, string>>;
  readonly trace: string;
  readonly requests: Readonly<Record<string, number>>;
  readonly summary: Readonly<Record<string, number | string>>;
}

const observations = (model: SparseReconstruction) =>
  model.landmarks.reduce((n, point) => n + point.observations.length, 0);

/** A request of the object-model protocol (`ReconstructionJob`): its kind and sizes. */
export function describeJob(job: ReconstructionJob): TraceStep {
  switch (job.kind) {
    case 'verifyPair':
      return [job.kind, job.args[0].length / 4];
    case 'pnp':
      return [job.kind, job.args[1].length / 2];
    case 'triangulation':
      return [
        job.kind,
        job.observations.length,
        job.observations.reduce((n, track) => n + track.length, 0),
      ];
    case 'observations':
      return [
        job.kind,
        job.stage,
        job.model.views.length,
        job.model.landmarks.length,
        observations(job.model),
      ];
    case 'bundle': {
      const [model, options] = job.args;
      return [
        job.kind,
        model.views.length,
        model.landmarks.length,
        observations(model),
        options?.variableViewIds?.length ?? -1,
        options?.iterations ?? -1,
      ];
    }
  }
}

/** A request of the scene protocol (`SceneRequest`): its kind and sizes. */
export function describeRequest(job: SceneRequest): TraceStep {
  switch (job.kind) {
    case 'verifyPair':
      return [job.kind, job.args[0].length / 4];
    case 'load':
      return [job.kind, job.model.views.length, job.model.landmarks.length];
    case 'register':
      return [job.kind, job.viewId];
    case 'continue':
      return [job.kind, job.camera, job.points.length];
    case 'triangulate':
      return [
        job.kind,
        job.groups.length,
        job.groups.reduce((n, group) => n + group.cameras.length, 0),
      ];
    case 'filter':
      return [job.kind, job.depthOnly ?? false];
    case 'pnp':
      return [job.kind, job.points.length];
    case 'bundle':
      return [
        job.kind,
        job.iterations,
        job.anchor,
        job.scale,
        job.variable?.length ?? -1,
        job.subset,
        job.refineSharedFocal,
        job.directionalGroups?.length ?? 0,
      ];
    case 'measure':
    case 'snapshot':
      return [job.kind];
  }
}

/**
 * Drive a mapper generator with `execute` and record its trace. With `passErrors` an error of
 * `execute` goes back into the generator, as `reconstructIncremental` and
 * `reconstructIncrementalScene` do; without it the error ends the run, as `reconstructComponent`
 * does.
 */
export function drive<Job, Result>(
  machine: Generator<Job, Result, unknown>,
  execute: (job: Job) => unknown,
  describe: (job: Job) => TraceStep,
  trace: TraceStep[],
  passErrors = true,
): Result {
  let step = machine.next();
  while (!step.done) {
    trace.push(describe(step.value));
    let value: unknown;
    try {
      value = execute(step.value);
    } catch (error) {
      trace.push(['throw', error instanceof Error ? error.name : typeof error]);
      if (!passErrors) throw error;
      step = machine.throw(error);
      continue;
    }
    step = machine.next(value);
  }
  return step.value;
}

/** The traced run of a driver; an error that ends the run is its output. */
function traced(run: (trace: TraceStep[]) => unknown): CaseRun {
  const trace: TraceStep[] = [];
  try {
    return { output: run(trace), trace };
  } catch (error) {
    return { output: { error }, trace };
  }
}

/** The output of a direct call; an error is its output. */
function caught(run: () => unknown): unknown {
  try {
    return run();
  } catch (error) {
    return { error };
  }
}

const jobs = (job: ReconstructionJob) => executeReconstructionJob(job);

/** `reconstructIncremental`: `incrementalSteps` on the CPU executor, errors passed back. */
export function incrementalCase(
  name: string,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
): Case {
  return {
    name: `reconstructIncremental/${name}`,
    run: () =>
      traced((trace) => drive(incrementalSteps(images, pairs, options), jobs, describeJob, trace)),
    direct: () => caught(() => reconstructIncremental(images, pairs, options)),
  };
}

/** `reconstructIncrementalScene`: `sceneSteps` on `HostSceneDriver`, errors passed back. */
export function sceneCase(
  name: string,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
): Case {
  return {
    name: `reconstructIncrementalScene/${name}`,
    run: () =>
      traced((trace) => {
        const driver = new HostSceneDriver();
        return drive(
          sceneSteps(images, pairs, options),
          (job) => driver.execute(job),
          describeRequest,
          trace,
        );
      }),
    direct: () => caught(() => reconstructIncrementalScene(images, pairs, options)),
  };
}

/**
 * `refineIncremental` of an accepted map: `incrementalSteps` resuming it, as the entry point
 * calls it. The output holds the accepted map (`initial`) and the refined one (`refined`).
 */
export function refineCase(
  name: string,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  initial: () => IncrementalResult,
  options: IncrementalOptions,
): Case {
  let accepted: IncrementalResult | undefined;
  const start = () => (accepted ??= initial());
  return {
    name: `refineIncremental/${name}`,
    run: () => {
      const first = start();
      const run = traced((trace) => {
        const model = refinementModel(images, pairs, first, options);
        const controls = { ...options, initialPair: first.diagnostics.initialPair };
        return drive(
          incrementalSteps(images, pairs, controls, model, undefined, true, 2, true),
          jobs,
          describeJob,
          trace,
        );
      });
      return { output: { initial: first, refined: run.output }, trace: run.trace };
    },
    direct: () => ({
      initial: start(),
      refined: caught(() => refineIncremental(images, pairs, start(), options)),
    }),
  };
}

/** `reconstructComponent` of component `index`: its input on the CPU executor, no error passed back. */
export function componentCase(
  name: string,
  graph: () => Promise<{ graph: ViewGraph; keyframes: readonly ViewGraphKeyframe[] }>,
  index: number,
  options: IncrementalOptions,
): Case {
  return {
    name: `reconstructComponent/${name}`,
    run: async () => {
      const built = await graph();
      const run = traced((trace) => {
        const input = componentInput(built.graph, built.keyframes, index, options);
        return drive(
          incrementalSteps(input.images, input.pairs, input.options),
          jobs,
          describeJob,
          trace,
          false,
        );
      });
      // The graph is the component's input; its digest names a change of the view graph apart.
      const { edges, components, videos, stats, candidates } = built.graph;
      return {
        output: { graph: { edges, components, videos, stats, candidates }, result: run.output },
        trace: run.trace,
      };
    },
    direct: async () => {
      const built = await graph();
      const { edges, components, videos, stats, candidates } = built.graph;
      return {
        graph: { edges, components, videos, stats, candidates },
        result: caught(() => reconstructComponent(built.graph, built.keyframes, index, options)),
      };
    },
  };
}

/** The focal search itself: `focalSearchSteps` on the CPU executor, errors passed back. */
export function focalSearchCase(
  name: string,
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
  search: true | FocalSearchOptions,
): Case {
  return {
    name: `focalSearchSteps/${name}`,
    run: () =>
      traced((trace) =>
        drive(
          focalSearchSteps(images, pairs, options, focalSearchControls(search)!),
          jobs,
          describeJob,
          trace,
        ),
      ),
  };
}

/**
 * The view graph of a set of views whose matches are known: one video, every pair a temporal
 * candidate, and a backend that answers each match with the known matches. Verification runs as
 * `buildViewGraph` runs it, on the CPU executor.
 */
export function knownMatchGraph(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: ViewGraphOptions = {},
) {
  const known = new Map(pairs.map((p) => [`${p.first}:${p.second}`, p.matches]));
  const keyframes: ViewGraphKeyframe[] = images.map((image, time) => ({
    viewId: image.id,
    videoId: 'video',
    frameKey: `frame-${image.id}`,
    time,
    camera: image.camera,
    keypoints: image.keypoints,
    descriptors: new Float32Array(image.keypoints.length * 64),
    global: new Float32Array(1),
  }));
  const cpu = cpuViewGraphBackend();
  const backend: ViewGraphBackend = {
    async match(first, others) {
      return others.map((other) => {
        const forward = known.get(`${first.viewId}:${other.viewId}`);
        const backward = known.get(`${other.viewId}:${first.viewId}`);
        let matches = new Uint32Array(0);
        if (forward) matches = forward.slice();
        else if (backward) {
          matches = new Uint32Array(backward.length);
          for (let i = 0; i < backward.length; i += 2) {
            matches[i] = backward[i + 1]!;
            matches[i + 1] = backward[i]!;
          }
        }
        return { matches, distances: new Float64Array(matches.length / 2) };
      });
    },
    execute: cpu.execute,
  };
  let graph: Promise<ViewGraph> | undefined;
  return async () => ({
    graph: await (graph ??= buildViewGraph(keyframes, backend, {
      temporalNeighbours: images.length,
      retrievalK: 0,
      reservePerVideoPair: 0,
      ...options,
    })),
    keyframes,
  });
}

/** The inputs of `tests/scene.ts` with every camera's focal scaled by `scale`. */
function scaledFocal(images: readonly ImageFeatures[], scale: number): ImageFeatures[] {
  return images.map((image) => ({
    ...image,
    camera: { ...image.camera, fx: image.camera.fx * scale, fy: image.camera.fy * scale },
  }));
}

/** The synthetic cases, on the scenes of tests/scene.ts (the option sets of tests/scene-mapper.test.ts). */
export function sceneCases(): Case[] {
  const distorted = scene({
    count: 90,
    views: 6,
    noise: 0.2,
    camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.03, k2: 0.004 },
  });
  const optionSets: [string, IncrementalOptions][] = [
    ['global', { seed: 42, ransacIterations: 300, initialPair: [10, 60] }],
    [
      'local-subset',
      {
        seed: 42,
        ransacIterations: 300,
        localBundleSubset: true,
        globalBundleFrequency: 4,
        localBundleSize: 2,
      },
    ],
    ['shared-focal', { seed: 7, ransacIterations: 300, refineSharedFocal: true }],
  ];
  const refinement = refinementFixture();
  const square = scene({
    count: 90,
    views: 6,
    noise: 0.1,
    camera: { fx: 850, fy: 850, cx: 480, cy: 270 },
  });
  const searchOptions: IncrementalOptions = {
    ransacIterations: 128,
    bundleIterations: 4,
    finalBundleIterations: 8,
    refineSharedFocal: true,
  };
  const component = scene({ count: 80, views: 6, noise: 0.1, seed: 17 });
  return [
    ...optionSets.flatMap(([label, options]) => [
      incrementalCase(`scene-90x6-distorted/${label}`, distorted.inputs, distorted.pairs, options),
      sceneCase(`scene-90x6-distorted/${label}`, distorted.inputs, distorted.pairs, options),
    ]),
    refineCase(
      'refinement-fixture',
      refinement.inputs,
      refinement.pairs,
      () =>
        reconstructIncremental(
          refinement.limited.inputs,
          refinement.limited.pairs,
          refinement.controls,
        ),
      refinement.controls,
    ),
    incrementalCase(
      'scene-90x6-focal-1.2/focal-search',
      scaledFocal(square.inputs, 1.2),
      square.pairs,
      { ...searchOptions, focalSearch: { views: 4 } },
    ),
    focalSearchCase(
      'scene-90x6-focal-1.2',
      scaledFocal(square.inputs, 1.2),
      square.pairs,
      searchOptions,
      { views: 4 },
    ),
    componentCase(
      'scene-80x6-known-matches',
      knownMatchGraph(component.inputs, component.pairs, {
        verification: { seed: 7, ransacIterations: 200 },
      }),
      0,
      { seed: 7, ransacIterations: 200, finalBundleIterations: 10 },
    ),
  ];
}

/** The heavier synthetic cases: a longer sequence and the view-graph scene of tests/multi-video-mapper.test.ts. */
export function largerSceneCases(): Case[] {
  const long = scene({ count: 90, views: 12, noise: 0.1 });
  const paths = pathScene([
    { label: 'orbit', cloud: 0, views: 7, from: -30, to: 30, radius: 9, height: -1 },
    { label: 'ascent', cloud: 0, views: 7, from: 5, to: 40, radius: 9, height: -3 },
    { label: 'lateral', cloud: 0, views: 7, from: -45, to: -5, radius: 10, height: -2 },
    { label: 'elsewhere', cloud: 1, views: 7, from: -30, to: 30, radius: 9, height: -1 },
  ]);
  let graph: Promise<ViewGraph> | undefined;
  const pathGraph = async () => ({
    graph: await (graph ??= buildViewGraph(paths.keyframes, cpuViewGraphBackend(), {
      retrievalK: 4,
      temporalNeighbours: 2,
      exclusionWindow: 3,
      reservePerVideoPair: 2,
      verification: { seed: 7, ransacIterations: 200 },
    })),
    keyframes: paths.keyframes,
  });
  const mapper = { seed: 7, ransacIterations: 200, finalBundleIterations: 10 };
  return [
    incrementalCase('scene-90x12/defaults', long.inputs, long.pairs, {}),
    sceneCase('scene-90x12/defaults', long.inputs, long.pairs, {}),
    componentCase('path-scene/component-0', pathGraph, 0, mapper),
    componentCase('path-scene/component-1', pathGraph, 1, mapper),
  ];
}

export { SAMPLES };

/**
 * The cases of one sample, with the lab's controls (`LAB` of tests/digest-samples.ts): the
 * calibrated map, the scene mapper, a refinement that adds the pairs more than two frames apart,
 * a component of the view graph of the sample's own matches, and the focal search from a start of
 * 1.2 × width.
 */
export function sampleCases(name: Sample): Case[] {
  const { frames, pairs } = loadSample(name);
  const calibrated: ImageFeatures[] = frames.map((frame) => ({
    id: frame.id,
    keypoints: frame.keypoints,
    camera: { ...frame.truth },
  }));
  const order = new Map(frames.map((frame, i) => [frame.id, i]));
  const near = pairs.filter((p) => Math.abs(order.get(p.first)! - order.get(p.second)!) <= 2);
  const cases = [
    incrementalCase(`${name}/lab`, calibrated, pairs, LAB),
    sceneCase(`${name}/lab`, calibrated, pairs, LAB),
    refineCase(
      `${name}/lab-near-then-all`,
      calibrated,
      pairs,
      () => reconstructIncremental(calibrated, near, LAB),
      LAB,
    ),
    componentCase(
      `${name}/lab-known-matches`,
      knownMatchGraph(calibrated, pairs, {
        verification: { ransacIterations: LAB.ransacIterations },
      }),
      0,
      LAB,
    ),
    focalSearchCase(`${name}/start-1.2`, camerasAt(frames, 1.2, true), pairs, LAB, true),
  ];
  // The traced run stands in for the entry point: the synthetic cases check that they agree.
  return cases.map(({ name, run }) => ({ name, run }));
}

const count = (steps: readonly TraceStep[]) => {
  const out: Record<string, number> = {};
  for (const step of steps) out[String(step[0])] = (out[String(step[0])] ?? 0) + 1;
  return Object.fromEntries(Object.entries(out).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
};

/** A few counts of an output, for the reader of a fixture. */
function summary(output: unknown): Record<string, number | string> {
  const value = output as {
    error?: Error;
    result?: unknown;
    reconstruction?: SparseReconstruction;
    unregistered?: readonly unknown[];
    ratio?: number;
    views?: readonly number[];
  };
  if (value.error instanceof Error) return { error: value.error.name };
  if (value.result !== undefined) return summary(value.result);
  if ((value as { refined?: unknown }).refined !== undefined)
    return summary((value as { refined?: unknown }).refined);
  if (value.reconstruction)
    return {
      views: value.reconstruction.views.length,
      landmarks: value.reconstruction.landmarks.length,
      observations: observations(value.reconstruction),
      unregistered: value.unregistered?.length ?? 0,
    };
  if (typeof value.ratio === 'number') return { views: value.views?.length ?? 0 };
  return {};
}

/** Run a case: the digests of its output and its trace. */
export async function characterize(entry: Case): Promise<CaseDigest> {
  const { output, trace } = await entry.run();
  const parts = await digestParts(output);
  if (entry.direct) {
    const direct = await digest(await entry.direct());
    assert.equal(
      direct,
      parts.digest,
      `${entry.name}: the entry point's output differs from the traced run's; the driver of ` +
        'tests/characterization.ts no longer reproduces it',
    );
  }
  return {
    output: parts.digest,
    parts: parts.parts,
    trace: await digest(trace),
    requests: count(trace.filter((step) => step[0] !== 'throw')),
    summary: { ...summary(output), throws: trace.filter((step) => step[0] === 'throw').length },
  };
}
