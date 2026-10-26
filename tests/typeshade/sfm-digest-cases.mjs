// The device-tier characterization cases of docs/library-structure.md, section 5, phase 0:
// `IncrementalSfmSession` (each keyframe added, then `finish`), `reconstructIncrementalGpu` (the GPU
// driver of `incrementalSteps`, with and without the focal search) and `refineIncrementalGpu`.
// sfm-digests.test.mjs runs them on the CPU device and sfm-digests-webgpu.test.mjs on WebGPU, on
// the inputs of sfm-digest-inputs.mjs. The module runs in Node and in a browser page.
//
// A case records:
//   output   the digest of every result, its `execution` report left out (device counts and
//            times); a session's output is each `add` result, the `finish` result and the focal
//            search's result;
//   trace    the digest of the mapper's requests as the driver reports them: each `onProgress`
//            event (phase, operation) and each bundle stage's problem size (`onStage`);
//   plan     (with `plans`) the plan reports of the numerical stages: for each stage its
//            operation, dispatches, submissions and the call list of each group; each pair
//            verification's consensus counts; the groups of each kernel (`kernelStatistics`); and
//            the device's reads, read waves and application bytes;
//   steps    (with `plans`) the sequence of kernel step names the device reports (`onProgress`).
// The plan and the steps depend on the chunk sizes, which WebGPU learns from measured times, so
// only the CPU device, whose chunks hold the whole work, records them. The CPU tier reports the
// call lists and the groups, and 0 dispatches and 0 submissions: it dispatches no workgroup.
import { digest, digestParts } from './digest.mjs';

const typedImages = (images) =>
  images.map((image) => ({ ...image, keypoints: Float64Array.from(image.keypoints) }));
const typedPairs = (pairs) =>
  pairs.map((pair) => ({ ...pair, matches: Uint32Array.from(pair.matches) }));

/** A result without its execution report. */
function outputOf(result) {
  if (result === undefined) return null;
  const { execution: _execution, ...rest } = result;
  return rest;
}
const stagesOf = (result) => result?.execution?.stages ?? [];
const sorted = (record) =>
  Object.fromEntries(
    Object.keys(record)
      .sort()
      .map((key) => [key, record[key]]),
  );

/** The entry points each kind of case calls. */
const ENTRY_POINTS = {
  session: ['IncrementalSfmSession'],
  driver: ['reconstructIncrementalGpu'],
  refine: ['reconstructIncrementalGpu', 'refineIncrementalGpu'],
};

/**
 * Run one case of sfm-digest-inputs.mjs on a device that `open(onKernel)` opens. Undefined when
 * the library lacks an entry point of the case: the compare job then reports the case as not run
 * on the base.
 */
export async function runDeviceCase(api, open, data, { plans = false } = {}) {
  if (ENTRY_POINTS[data.kind].some((name) => typeof api[name] !== 'function')) return undefined;
  const steps = [];
  const device = await open((event) => {
    if (event.phase === 'start') steps.push(event.kernel);
  });
  const trace = [];
  const options = (base) => ({
    ...base,
    onProgress: (event) => trace.push([event.phase, event.operation]),
    onStage: (stage) => {
      if (stage.problem)
        trace.push([
          'problem',
          stage.operation,
          stage.problem.views,
          stage.problem.landmarks,
          stage.problem.observations,
        ]);
    },
  });
  const results = [];
  let output;
  try {
    const images = typedImages(data.images),
      pairs = typedPairs(data.pairs);
    if (data.kind === 'session') {
      const session = new api.IncrementalSfmSession(device, options(data.options));
      const adds = [];
      for (const image of images) {
        const active = new Set(session.activeViewIds);
        const added = await session.add(
          image,
          pairs.filter(
            (p) =>
              (p.first === image.id && active.has(p.second)) ||
              (p.second === image.id && active.has(p.first)),
          ),
        );
        results.push(added);
        adds.push(outputOf(added));
      }
      const finished = await session.finish();
      results.push(finished);
      output = { adds, finish: outputOf(finished), focalSearch: session.focalSearch ?? null };
    } else if (data.kind === 'driver') {
      const result = await api.reconstructIncrementalGpu(
        device,
        images,
        pairs,
        options(data.options),
      );
      results.push(result);
      output = outputOf(result);
    } else {
      const initial = await api.reconstructIncrementalGpu(
        device,
        typedImages(data.initial.images),
        typedPairs(data.initial.pairs),
        options(data.options),
      );
      const refined = await api.refineIncrementalGpu(
        device,
        images,
        pairs,
        initial,
        options(data.options),
      );
      results.push(initial, refined);
      output = { initial: outputOf(initial), refined: outputOf(refined) };
    }
  } catch (error) {
    output = { error: { name: error?.name, message: error?.message } };
  }
  const kernels = device.kernelStatistics;
  const statistics = device.statistics;
  await device.dispose();
  const parts = await digestParts(output);
  const requests = {};
  for (const step of trace)
    if (step[0] === 'start') requests[step[1]] = (requests[step[1]] ?? 0) + 1;
  const record = {
    output: parts.digest,
    parts: parts.parts,
    trace: await digest(trace),
    requests: sorted(requests),
    summary: summaryOf(output),
  };
  if (!plans) return record;
  const stages = results.flatMap(stagesOf);
  const reports = stages.map((stage) => [
    stage.operation,
    stage.execution.report.dispatches,
    stage.execution.report.submissions,
    stage.execution.report.groups.map((group) => group.calls.join('+')),
    stage.execution.applicationUploadBytes,
    stage.execution.applicationReadbackBytes,
  ]);
  const consensus = results
    .flatMap((result) => result?.execution?.pairConsensus ?? [])
    .map((c) => [
      c.hypothesisBatches,
      c.hypothesesScored,
      c.scoreDispatches,
      c.scoreSubmissions,
      c.applicationUploadBytes,
      c.applicationReadbackBytes,
    ]);
  return {
    ...record,
    plan: {
      digest: await digest({ reports, consensus }),
      stages: stages.length,
      groups: stages.reduce((n, stage) => n + stage.execution.report.groups.length, 0),
      dispatches: stages.reduce((n, stage) => n + stage.execution.report.dispatches, 0),
      submissions: stages.reduce((n, stage) => n + stage.execution.report.submissions, 0),
      reads: statistics.reads,
      readWaves: statistics.readWaves,
      uploadBytes: statistics.applicationUploadBytes,
      readbackBytes: statistics.applicationReadbackBytes,
      kernels: sorted(Object.fromEntries(kernels.rows.map((row) => [row.name, row.groups]))),
      kernelsDropped: kernels.dropped,
    },
    steps: { digest: await digest(steps), count: steps.length },
  };
}

function summaryOf(output) {
  if (output?.error) return { error: output.error.name ?? 'Error' };
  const last = output?.finish ?? output?.refined ?? output;
  const model = last?.reconstruction;
  if (!model) return {};
  return {
    views: model.views.length,
    landmarks: model.landmarks.length,
    observations: model.landmarks.reduce((n, point) => n + point.observations.length, 0),
  };
}
