import type { DenseChunk, DenseMeshData } from './dense-cloud';
import { parsePly, type Ply } from './ply-parse.ts';
import { RUN_RECORD_FORMATS, runRecordProblem, type RunRecord } from './run-record.ts';
import type { Scene, SceneView, ScenePoint } from './sfm-scene';

/**
 * Open a finished reconstruction without running the pipeline: the files the lab exports
 * (`result-export.ts`) go in, and the scene, the dense cloud and the mesh the viewer draws come out.
 *
 * - `typecv-reconstruction.json`: cameras, sparse landmarks with colors and, when it has one, the
 *   run record.
 * - `typecv-run-record.json`: the run record alone.
 * - `typecv-dense.ply` (binary, no faces): the dense cloud.
 * - `typecv-mesh.ply` (faces): the mesh.
 * - `typecv-reconstruction.ply` (ASCII, no faces): the sparse points, used when no JSON is given.
 */

export type ResultFile = { name: string; data: Uint8Array };

export type LoadedResult = {
  /** Cameras and sparse landmarks; undefined when no file gave any. */
  scene?: Scene;
  source?: { name: string; width: number; height: number; duration?: number };
  runRecord?: RunRecord;
  diagnostics?: unknown;
  unregistered: { viewId: number; reason: string }[];
  /** The dense points, one chunk per PLY file; the viewer reads the chunks in order. */
  dense?: DenseChunk[];
  mesh?: Pick<DenseMeshData, 'positions' | 'colors' | 'indices'>;
  /** What the loader did not use, one line each. */
  notes: string[];
};

/** The gray of a point or vertex that the file gives no color. */
const NO_COLOR = 180;

function bad(message: string): never {
  throw new RangeError(message);
}

function numbers(value: unknown, length: number, what: string): number[] {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    !value.every((v) => typeof v === 'number' && Number.isFinite(v))
  )
    bad(what + ' must be ' + length + ' finite numbers');
  return value as number[];
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function camera(value: any, what: string): SceneView['camera'] {
  const out: SceneView['camera'] = {
    fx: optionalNumber(value?.fx) ?? bad(what + ': fx'),
    fy: optionalNumber(value?.fy) ?? bad(what + ': fy'),
    cx: optionalNumber(value?.cx) ?? bad(what + ': cx'),
    cy: optionalNumber(value?.cy) ?? bad(what + ': cy'),
  };
  const k1 = optionalNumber(value.k1),
    k2 = optionalNumber(value.k2);
  if (k1 !== undefined) out.k1 = k1;
  if (k2 !== undefined) out.k2 = k2;
  return out;
}

/** A parsed reconstruction JSON as a scene. Throws a `RangeError` that names the bad field. */
export function sceneFromJson(json: any): Scene {
  if (!Array.isArray(json?.views) || !Array.isArray(json?.landmarks))
    bad('The JSON is not a reconstruction: it has no views and landmarks.');
  const views: SceneView[] = json.views.map((v: any, i: number): SceneView => {
    const what = 'views[' + i + ']';
    const id = optionalNumber(v?.id) ?? bad(what + ': id');
    const view: SceneView = {
      id,
      pose: {
        R: Float64Array.from(numbers(v.pose?.R, 9, what + '.pose.R')),
        t: Float64Array.from(numbers(v.pose?.t, 3, what + '.pose.t')),
      },
      camera: camera(v.camera, what + '.camera'),
    };
    for (const key of ['time', 'sequence', 'width', 'height'] as const) {
      const value = optionalNumber(v[key]);
      if (value !== undefined) view[key] = value;
    }
    if (typeof v.image === 'string') view.image = v.image;
    return view;
  });
  const landmarks: ScenePoint[] = json.landmarks.map((p: any, i: number): ScenePoint => {
    const what = 'landmarks[' + i + ']';
    const point: ScenePoint = {
      trackId: optionalNumber(p?.trackId) ?? i,
      xyz: Float64Array.from(numbers(p?.xyz, 3, what + '.xyz')),
      observations: Array.isArray(p.observations) ? p.observations : [],
    };
    if (p.color !== undefined && p.color !== null)
      point.color = numbers(p.color, 3, what + '.color') as [number, number, number];
    return point;
  });
  return { views, landmarks };
}

/** The sparse landmarks of an ASCII PLY: one point per vertex, colors from bytes to 0..1. */
export function sceneFromPly(ply: Ply): Scene {
  const n = ply.positions.length / 3;
  const landmarks: ScenePoint[] = [];
  for (let i = 0; i < n; i++) {
    landmarks.push({
      trackId: i,
      xyz: Float64Array.from(ply.positions.subarray(3 * i, 3 * i + 3)),
      color: ply.colors
        ? [ply.colors[3 * i] / 255, ply.colors[3 * i + 1] / 255, ply.colors[3 * i + 2] / 255]
        : undefined,
      observations: [],
    });
  }
  return { views: [], landmarks };
}

function colorsOf(ply: Ply): Uint8Array {
  return ply.colors ?? new Uint8Array(ply.positions.length).fill(NO_COLOR);
}

/** The dense cloud of a PLY without faces, as one chunk. */
export function denseFromPly(ply: Ply, viewId = 0): DenseChunk {
  return { viewId, positions: ply.positions, colors: colorsOf(ply) };
}

/** The mesh of a PLY with faces. */
export function meshFromPly(ply: Ply): NonNullable<LoadedResult['mesh']> {
  if (!ply.indices) bad('The PLY has no faces.');
  return { positions: ply.positions, colors: colorsOf(ply), indices: ply.indices };
}

function isJson(file: ResultFile): boolean {
  if (/\.json$/i.test(file.name)) return true;
  for (const byte of file.data.subarray(0, 64)) {
    if (byte === 0x20 || byte === 0x0a || byte === 0x0d || byte === 0x09) continue;
    return byte === 0x7b;
  }
  return false;
}

/**
 * Read the files of one result. The order of the files does not matter. A file that cannot be
 * read throws a `RangeError` that names it; a file that is read but not used adds a note.
 */
export function loadResultFiles(files: readonly ResultFile[]): LoadedResult {
  const result: LoadedResult = { unregistered: [], notes: [] };
  const sparsePly: { name: string; ply: Ply }[] = [];
  for (const file of files) {
    try {
      if (isJson(file)) {
        const json = JSON.parse(new TextDecoder().decode(file.data));
        if ((RUN_RECORD_FORMATS as readonly unknown[]).includes(json?.format)) {
          const problem = runRecordProblem(json);
          if (problem) bad('The run record has a bad field: ' + problem);
          result.runRecord = json;
          continue;
        }
        if (result.scene) {
          result.notes.push(file.name + ': a second reconstruction JSON, not used');
          continue;
        }
        result.scene = sceneFromJson(json);
        const video = json.sourceVideo;
        if (video && typeof video.name === 'string')
          result.source = {
            name: video.name,
            width: optionalNumber(video.width) ?? 0,
            height: optionalNumber(video.height) ?? 0,
            duration: optionalNumber(video.duration),
          };
        result.diagnostics = json.diagnostics;
        if (Array.isArray(json.unregistered)) result.unregistered = json.unregistered;
        if (json.runRecord !== undefined && json.runRecord !== null) {
          const problem = runRecordProblem(json.runRecord);
          if (problem)
            result.notes.push(file.name + ': the run record has a bad field: ' + problem);
          else result.runRecord = json.runRecord;
        }
        continue;
      }
      const ply = parsePly(file.data);
      if (ply.indices) {
        if (result.mesh) result.notes.push(file.name + ': a second mesh, not used');
        else result.mesh = meshFromPly(ply);
      } else if (ply.format === 'ascii') sparsePly.push({ name: file.name, ply });
      else (result.dense ??= []).push(denseFromPly(ply, result.dense?.length ?? 0));
    } catch (error) {
      throw new RangeError(file.name + ': ' + (error as Error).message, { cause: error });
    }
  }
  if (sparsePly.length) {
    if (result.scene)
      for (const { name } of sparsePly)
        result.notes.push(name + ': the sparse PLY repeats the JSON, not used');
    else result.scene = sceneFromPly(sparsePly[0].ply);
  }
  if (!result.scene && !result.dense && !result.mesh)
    bad('The files hold no reconstruction, dense cloud or mesh.');
  return result;
}
