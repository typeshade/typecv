/**
 * The lab's sample catalog, `website/public/sample/index.json`. Two kinds of entry live in it:
 * a sample (frames to run the pipeline on: `video` or `photos`) and a result (files of a finished
 * run to open in the viewer without running it: `result`).
 */

export type SampleEntry = {
  id: string;
  kind: 'video' | 'photos';
  dir: string;
  name: string;
  frames: number;
  /** The width and height of a frame in the 640 px set; a preview set of edge E scales both. */
  size?: [number, number];
  thumbnail: string;
  features?: boolean;
  description: { en: string; ko: string };
  video?: { name: string; times: number[]; duration: number };
  /** The analysis edges the sample ships previews for; 640 only when absent. A preview set of
   *  edge E other than 640 lives under `dir + E + '/'` with its own `sample.json`. */
  edges?: number[];
  /** The frames live in the R2 bucket (`website/samples/remote-videos.json`), not in the
   *  repository; until the publish workflow has run they are missing. */
  remote?: boolean;
};

/** The files of a result, each a name inside the entry's `dir`. */
export type ResultFiles = {
  /** `typecv-reconstruction.json`: cameras, sparse points and, when it has one, the run record. */
  json: string;
  /** `typecv-reconstruction.ply`: the sparse cloud, kept for download. The viewer does not fetch it. */
  ply?: string;
  /** `typecv-dense.ply`: the dense cloud. */
  dense?: string;
  /** `typecv-mesh.ply`: the mesh. */
  mesh?: string;
};

/** The run a result came from. */
export type ResultRun = {
  /** The day of the run, `YYYY-MM-DD`. */
  date: string;
  /** What ran it, a short label: `swiftshader` for the software rasterizer, else a GPU name. */
  device: string;
  /** The commit of the lab or library that ran it (7 to 40 hex digits). */
  commit: string;
};

/**
 * A precomputed result. `dir` is `/sample/results/<id>/` and the files are served by the Worker
 * from the R2 bucket `typecv-samples` (`scripts/lab-results.mjs`).
 */
export type ResultEntry = {
  id: string;
  kind: 'result';
  name: string;
  dir: string;
  files: ResultFiles;
  /** An image inside `dir`; the card shows a blank tile when absent. */
  thumbnail?: string;
  /** The size of the files in bytes, summed; the card shows it in the tooltip. */
  size?: number;
  /** The `id` of the sample the result was computed from, shown as a tag. */
  source?: string;
  description?: { en: string; ko: string };
  /** Always true: the files are in the bucket, not in the repository. */
  remote: true;
  run: ResultRun;
};

/**
 * A set of remote video samples that the lab loads together (`docs/multi-video.md`, 6.3). Each
 * `members` item is the `id` of a `video` entry. The lab shows no card for it until step 6.
 */
export type VideoSetEntry = {
  id: string;
  kind: 'video-set';
  name: string;
  members: string[];
  description: { en: string; ko: string };
};

export type CatalogEntry = SampleEntry | ResultEntry | VideoSetEntry;

export type SampleCatalog = { format: 'typecv-lab-samples-v1'; samples: CatalogEntry[] };

export function isVideoSet(entry: CatalogEntry): entry is VideoSetEntry {
  return entry.kind === 'video-set';
}

export function isResult(entry: CatalogEntry): entry is ResultEntry {
  return entry.kind === 'result';
}

const ID = /^[a-z0-9][a-z0-9-]{0,63}$/;
const FILE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const DEVICE = /^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$/;

/** The tag text of a device: `swiftshader` is spelled SwiftShader, any other label stays. */
export function deviceLabel(device: string): string {
  return device.toLowerCase() === 'swiftshader' ? 'SwiftShader' : device;
}

/** The first problem of a result entry, or undefined when it is valid. */
export function resultEntryProblem(entry: unknown): string | undefined {
  const e = entry as Partial<ResultEntry> | null;
  if (!e || typeof e !== 'object') return 'not an object';
  if (e.kind !== 'result') return 'kind';
  if (typeof e.id !== 'string' || !ID.test(e.id)) return 'id';
  if (typeof e.name !== 'string' || !e.name || e.name.length > 120) return 'name';
  if (e.dir !== `/sample/results/${e.id}/`) return 'dir';
  const files = e.files;
  if (!files || typeof files !== 'object') return 'files';
  if (typeof files.json !== 'string' || !FILE.test(files.json) || !files.json.endsWith('.json'))
    return 'files.json';
  for (const key of ['ply', 'dense', 'mesh'] as const) {
    const name = files[key];
    if (
      name !== undefined &&
      (typeof name !== 'string' || !FILE.test(name) || !name.endsWith('.ply'))
    )
      return 'files.' + key;
  }
  if (e.thumbnail !== undefined && (typeof e.thumbnail !== 'string' || !FILE.test(e.thumbnail)))
    return 'thumbnail';
  if (e.size !== undefined && !(Number.isInteger(e.size) && e.size >= 0)) return 'size';
  if (e.source !== undefined && (typeof e.source !== 'string' || !ID.test(e.source)))
    return 'source';
  if (
    e.description !== undefined &&
    (typeof e.description?.en !== 'string' || typeof e.description?.ko !== 'string')
  )
    return 'description';
  if (e.remote !== true) return 'remote';
  const run = e.run;
  if (!run || typeof run !== 'object') return 'run';
  if (
    typeof run.date !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}$/.test(run.date) ||
    Number.isNaN(Date.parse(run.date))
  )
    return 'run.date';
  if (typeof run.device !== 'string' || !DEVICE.test(run.device)) return 'run.device';
  if (typeof run.commit !== 'string' || !/^[0-9a-f]{7,40}$/.test(run.commit)) return 'run.commit';
  return undefined;
}

/** The files the View button downloads: the JSON, the dense cloud and the mesh. The sparse PLY
 *  repeats the JSON's points, so the viewer does not fetch it. */
export function resultFileNames(entry: ResultEntry): string[] {
  const { json, dense, mesh } = entry.files;
  return [json, dense, mesh].filter((name): name is string => !!name);
}
