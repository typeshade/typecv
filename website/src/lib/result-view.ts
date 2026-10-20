import type { DenseCloud, DenseMeshData } from './dense-cloud';
import { loadResultFiles, type LoadedResult, type ResultFile } from './result-load.ts';
import { resultFileNames, type ResultEntry } from './sample-catalog.ts';
import type { RunRecord } from './run-record.ts';

/**
 * Everything a page needs to put a finished result in `SfmViewer`, with no React in it: the lab
 * mounts it today, and a page of its own can mount it later. `fetchResultFiles` downloads the files
 * of a catalog entry, `resultView` turns the loaded files into the viewer's props and the tags.
 */

/** What the tags above the viewer show. */
export type ResultInfo = {
  name: string;
  entry?: ResultEntry;
  record?: RunRecord;
  cameras: number;
  sparse: number;
  dense: number;
  faces: number;
  notes: string[];
};

export type ResultView = {
  scene: LoadedResult['scene'];
  source: LoadedResult['source'];
  diagnostics: unknown;
  unregistered: LoadedResult['unregistered'];
  dense?: DenseCloud;
  mesh?: DenseMeshData;
  /** A mesh without a dense cloud is shown at once. */
  showMesh: boolean;
  info: ResultInfo;
};

/** Thrown when the entry's JSON is not in the bucket: the result is not published yet. */
export class ResultMissingError extends Error {
  readonly url: string;
  readonly status: number;
  constructor(url: string, status: number) {
    super(status + ' ' + url);
    this.url = url;
    this.status = status;
  }
}

/** The viewer's props for a loaded result. `run` is the dense cloud's and mesh's run number. */
export function resultView(
  loaded: LoadedResult,
  name: string,
  run: number,
  entry?: ResultEntry,
): ResultView {
  const dense = loaded.dense;
  const count = dense?.reduce((sum, chunk) => sum + chunk.positions.length / 3, 0) ?? 0;
  return {
    scene: loaded.scene,
    source: loaded.source,
    diagnostics: loaded.diagnostics,
    unregistered: loaded.unregistered,
    dense: dense && { run, chunks: dense, count },
    mesh: loaded.mesh && { run, ...loaded.mesh },
    showMesh: !!loaded.mesh && !dense,
    info: {
      name,
      entry,
      record: loaded.runRecord,
      cameras: loaded.scene?.views.length ?? 0,
      sparse: loaded.scene?.landmarks.length ?? 0,
      dense: count,
      faces: (loaded.mesh?.indices.length ?? 0) / 3,
      notes: loaded.notes,
    },
  };
}

/** Read local files (a file picker's `FileList`) and build the view. */
export async function resultViewFromFiles(
  picked: readonly File[],
  run: number,
): Promise<ResultView> {
  const files = await Promise.all(
    picked.map(async (file) => ({
      name: file.name,
      data: new Uint8Array(await file.arrayBuffer()),
    })),
  );
  return resultView(loadResultFiles(files), picked.map((file) => file.name).join(', '), run);
}

/**
 * Download the files of an entry one after another. `onProgress(done, total)` counts bytes when
 * the entry gives its `size`, files otherwise. A missing JSON throws `ResultMissingError`; a
 * missing dense or mesh file is skipped.
 */
export async function fetchResultFiles(
  entry: ResultEntry,
  onProgress: (done: number, total: number) => void,
  fetchFile: typeof fetch = fetch,
): Promise<ResultFile[]> {
  const names = resultFileNames(entry);
  const bytes = entry.size !== undefined;
  const total = entry.size ?? names.length;
  let done = 0;
  const files: ResultFile[] = [];
  for (const name of names) {
    const url = entry.dir + name;
    const response = await fetchFile(url);
    if (!response.ok || !response.body) {
      if (name === entry.files.json) throw new ResultMissingError(url, response.status);
      continue;
    }
    const parts: Uint8Array[] = [];
    let length = 0;
    const reader = response.body.getReader();
    for (;;) {
      const { done: finished, value } = await reader.read();
      if (finished) break;
      parts.push(value);
      length += value.length;
      if (bytes) {
        done = Math.min(total, done + value.length);
        onProgress(done, total);
      }
    }
    const data = new Uint8Array(length);
    let at = 0;
    for (const part of parts) {
      data.set(part, at);
      at += part.length;
    }
    if (!bytes) onProgress(++done, total);
    files.push({ name, data });
  }
  return files;
}
