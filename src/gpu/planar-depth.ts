import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice, GpuStep, ShadePlanReport } from './device.ts';
import type { TsdfInput, TsdfView } from '../reconstruction/tsdf.ts';
import { tsdfVisibilityGroups } from '../reconstruction/tsdf.ts';
import { chunkedPass as basePass } from './cloud-clean.ts';
import { planarFit, planarPropose, planarVerify, planarFinish } from './planar-depth.shade.ts';

/** Keep each dispatch within the adapter's one-dimensional workgroup limit. */
function* chunkedPass(
  device: ShadeDevice,
  key: string,
  name: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  const limit = Math.max(64, Math.floor(device.maxKernelElements / 64) * 64);
  for (let first = 0; first < total; first += limit) {
    const offset = first;
    yield* basePass(device, key, name, Math.min(limit, total - first), (range, groups) =>
      dispatch({ first: offset + range.first, end: offset + range.end }, groups),
    );
  }
}

export interface PlanarDepthReport {
  readonly filled: number;
  readonly perView: readonly number[];
  readonly peakBytes: number;
  readonly report: ShadePlanReport;
}

function camera(view: TsdfView) {
  const R = view.rotation,
    C = view.center;
  return {
    fx: view.fx,
    fy: view.fy,
    cx: view.cx,
    cy: view.cy,
    r0: R[0]!,
    r1: R[1]!,
    r2: R[2]!,
    r3: R[3]!,
    r4: R[4]!,
    r5: R[5]!,
    r6: R[6]!,
    r7: R[7]!,
    r8: R[8]!,
    centerX: C[0]!,
    centerY: C[1]!,
    centerZ: C[2]!,
  };
}

/** Frozen filtered maps are the sole geometric evidence; fills never validate other fills. */
export async function completePlanarDepths(
  device: ShadeDevice,
  input: TsdfInput,
  options: {
    residentBytes?: number;
    signal?: AbortSignal;
    sources?: readonly (readonly number[])[];
  } = {},
): Promise<{ input: TsdfInput; report: PlanarDepthReport }> {
  const { width, height } = input,
    P = width * height;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    input.views.some(
      (v) =>
        v.depth.length !== P ||
        v.normal.length !== P ||
        v.rgba.length !== P ||
        (v.confidence !== undefined && v.confidence.length !== P),
    )
  )
    throw new RangeError('Invalid planar depth input');
  options.signal?.throwIfAborted();
  if (
    options.sources &&
    (options.sources.length !== input.views.length ||
      options.sources.some((indices) =>
        indices.some((v) => !Number.isSafeInteger(v) || v < 0 || v >= input.views.length),
      ))
  )
    throw new RangeError('Invalid planar source views');
  if (input.views.length < 3)
    return {
      input,
      report: {
        filled: 0,
        perView: input.views.map(() => 0),
        peakBytes: 0,
        report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
      },
    };
  const tilesX = Math.ceil(width / 16),
    tilesY = Math.ceil(height / 16);
  const bytes = 44 * P + 4 * 14 * tilesX * tilesY;
  const s = device.statistics;
  const available = device.workingBytes - (s.liveBytes - s.pooledBytes);
  if (bytes > Math.min(available, options.residentBytes ?? available))
    throw new RangeError('Planar depth completion exceeds the available working budget');
  options.signal?.throwIfAborted();
  const views: TsdfView[] = [];
  const perView: number[] = [];
  const reports: ShadePlanReport[] = [];
  const visibilityGroups = tsdfVisibilityGroups(input);
  let ones: Float32Array | undefined;
  const confidenceOf = (v: TsdfView) => v.confidence ?? (ones ??= new Float32Array(P).fill(1));
  for (const [reference, view] of input.views.entries()) {
    options.signal?.throwIfAborted();
    const owned: (Resident<Float32Array> | Resident<Uint32Array>)[] = [];
    const f32 = (n: number, values?: Float32Array) => {
      const r = device.allocate(n, values);
      owned.push(r);
      return r;
    };
    const u32 = (n: number, values?: Uint32Array) => {
      const r = device.allocateU32(n, values);
      owned.push(r);
      return r;
    };
    let successful = false;
    try {
      const confidence = confidenceOf(view);
      const planar = {
        width,
        height,
        tilesX,
        tilesY,
        ...camera(view),
        k1: view.k1 ?? 0,
        k2: view.k2 ?? 0,
        radialLimit: view.radialLimit ?? 1e30,
      };
      const rgba = u32(P, view.rgba),
        depth = f32(P, view.depth),
        normal = u32(P, view.normal),
        trust = f32(P, confidence);
      const planes = f32(14 * tilesX * tilesY);
      const candidate = f32(P),
        candidateNormal = u32(P),
        candidateConfidence = f32(P),
        support = u32(P);
      const sourceDepth = f32(P),
        sourceConfidence = f32(P);
      const supportGroup = u32(P);
      const common = {
        planarDepth: depth,
        planarNormal: normal,
        planarConfidence: trust,
        planarCandidate: candidate,
        planarCandidateNormal: candidateNormal,
        planarCandidateConfidence: candidateConfidence,
        planarSupport: support,
      };
      const run = async (steps: Iterable<GpuStep>) => {
        function* checked() {
          for (const step of steps) {
            options.signal?.throwIfAborted();
            yield step;
          }
        }
        reports.push(await device.planSteps(checked()));
      };
      await run(
        chunkedPass(device, 'planarFit', 'planarFit', tilesX * tilesY, (planarRange, groups) =>
          planarFit(
            {
              planar,
              planarRange,
              planarDepth: depth,
              planarConfidence: trust,
              planarRgba: rgba,
              planarPlanes: planes,
            },
            groups,
          ),
        ),
      );
      await run(
        chunkedPass(device, 'planarPropose', 'planarPropose', P, (planarRange, groups) =>
          planarPropose(
            {
              planar,
              planarRange,
              planarDepth: depth,
              planarRgba: rgba,
              planarPlanes: planes,
              planarCandidate: candidate,
              planarCandidateNormal: candidateNormal,
              planarCandidateConfidence: candidateConfidence,
            },
            groups,
          ),
        ),
      );
      const allowed = options.sources ? new Set(options.sources[reference]) : undefined;
      const groups = visibilityGroups
        .filter((g) => !g.includes(reference))
        .map((g) => (allowed ? g.filter((v) => allowed.has(v)) : g))
        .filter((g) => g.length > 0);
      for (const [group, indices] of groups.entries())
        for (const index of indices) {
          const source = input.views[index]!;
          sourceDepth.write(source.depth);
          sourceConfidence.write(confidenceOf(source));
          const planarSource = { ...camera(source), group: group + 1 };
          await run(
            chunkedPass(device, 'planarVerify', 'planarVerify', P, (planarRange, groups) =>
              planarVerify(
                {
                  planar,
                  planarSource,
                  planarRange,
                  planarCandidate: candidate,
                  planarSupport: support,
                  planarSupportGroup: supportGroup,
                  planarSourceDepth: sourceDepth,
                  planarSourceConfidence: sourceConfidence,
                },
                groups,
              ),
            ),
          );
        }
      await run(
        chunkedPass(device, 'planarFinish', 'planarFinish', P, (planarRange, groups) =>
          planarFinish({ ...common, planarRange }, groups),
        ),
      );
      options.signal?.throwIfAborted();
      const completed = (await device.read(candidate, P)).slice(0, P);
      const completedNormal = (await device.read(candidateNormal, P)).slice(0, P);
      const completedConfidence = (await device.read(candidateConfidence, P)).slice(0, P);
      let filled = 0;
      for (let i = 0; i < P; i++) if (!(view.depth[i]! > 0) && completed[i]! > 0) filled++;
      perView.push(filled);
      views.push({
        ...view,
        depth: completed,
        normal: completedNormal,
        confidence: completedConfidence,
      });
      successful = true;
    } finally {
      for (const r of owned) successful ? device.recycle(r) : device.drop(r);
    }
  }
  return {
    input: { ...input, views },
    report: {
      filled: perView.reduce((n, v) => n + v, 0),
      perView,
      peakBytes: bytes,
      report: {
        tier: device.backend,
        dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
        submissions: reports.reduce((n, r) => n + r.submissions, 0),
        groups: reports.flatMap((r) => r.groups),
      },
    },
  };
}
