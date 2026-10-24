import { denseMeshPly, densePly, type DenseChunk, type DenseMeshData } from './dense-cloud.ts';
import { reconstructionPly } from './result-export.ts';
import type { ResultFilePayload } from './send-record.ts';
import type { Scene } from './sfm-scene.ts';

export interface ResultComponent {
  readonly index: number;
  readonly scene: Scene;
  readonly points?: readonly DenseChunk[];
  readonly mesh?: DenseMeshData;
}

/** Capture references once; serialize only when the sequential sender requests a file. */
export function resultFileSources(
  components: readonly ResultComponent[],
  artifacts: Iterable<readonly [string, Blob | string]>,
  json: (scene: Scene) => string,
): ResultFilePayload[] {
  const files: ResultFilePayload[] = [];
  const multiple = components.length > 1;
  const seen = new Set<number>();
  for (const component of [...components].sort((a, b) => a.index - b.index)) {
    const { index } = component;
    if (!Number.isSafeInteger(index) || index < 0 || index > 999 || seen.has(index))
      throw new RangeError('Invalid result component index');
    seen.add(index);
    const suffix = multiple ? '-c' + String(index).padStart(3, '0') : '';
    const name = (kind: 'reconstruction' | 'dense' | 'mesh', format: 'json' | 'ply') =>
      `typecv-${kind}${suffix}.${format}` as ResultFilePayload['name'];
    const scene = { views: [...component.scene.views], landmarks: [...component.scene.landmarks] };
    const points = [...(component.points ?? [])];
    const mesh = component.mesh;
    files.push(
      { name: name('reconstruction', 'json'), data: () => json(scene) },
      { name: name('reconstruction', 'ply'), data: () => reconstructionPly(scene) },
    );
    if (points.length) files.push({ name: name('dense', 'ply'), data: () => densePly(points) });
    if (mesh) files.push({ name: name('mesh', 'ply'), data: () => denseMeshPly(mesh) });
  }
  for (const [name, data] of artifacts)
    files.push({ name: name as ResultFilePayload['name'], data });
  return files;
}
