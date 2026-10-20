import type { RunRecord } from './run-record';
import type { Scene, SceneView } from './sfm-scene';

/**
 * The lab's reconstruction exports (`typecv-reconstruction.json` and `.ply`). `result-load.ts`
 * reads them back, so the writer and the reader live side by side and a test holds them equal.
 */
export const RECONSTRUCTION_FORMAT = 'typecv-reconstruction-v1' as const;

/** The color of a camera center in the sparse PLY. */
export const CAMERA_PLY_COLOR = [255, 180, 84] as const;

export type ReconstructionJsonInput = {
  source?: { name: string; width: number; height: number; duration?: number };
  assumedFocalRatio: number;
  diagnostics: unknown;
  unregistered: { viewId: number; reason: string }[];
  /** The analysis frames without their pixels. */
  analysisFrames: object[];
  scene: Scene;
  /** The view's camera in the source's pixels; the view's own camera when it has no frame. */
  sourceCamera: (view: SceneView) => SceneView['camera'];
  /** The run's record, when the page has one: the summary the loader shows. */
  runRecord?: RunRecord;
};

/** The reconstruction as JSON: views with their cameras and poses, and every landmark. */
export function reconstructionJson(input: ReconstructionJsonInput): string {
  const { scene } = input;
  return JSON.stringify(
    {
      format: RECONSTRUCTION_FORMAT,
      source: input.source?.name,
      sourceVideo: input.source,
      assumedFocalRatio: input.assumedFocalRatio,
      diagnostics: input.diagnostics,
      unregistered: input.unregistered,
      analysisFrames: input.analysisFrames,
      views: scene.views.map((view) => ({
        ...view,
        sourceCamera: input.sourceCamera(view),
        pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
      })),
      landmarks: scene.landmarks.map((point) => ({ ...point, xyz: Array.from(point.xyz) })),
      runRecord: input.runRecord,
    },
    null,
    2,
  );
}

/** The center of a camera in the world frame: -R^T t, with R row-major. */
export function cameraCenter({ R, t }: SceneView['pose']): [number, number, number] {
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

/** The sparse cloud as an ASCII PLY: every landmark, then one orange vertex per camera. */
export function reconstructionPly(scene: Scene): string {
  const vertices = [
    ...scene.landmarks.map(
      (point) =>
        Array.from(point.xyz).join(' ') +
        ' ' +
        (point.color ?? [0.7, 0.7, 0.7]).map((v) => Math.round(v * 255)).join(' '),
    ),
    ...scene.views.map(
      (view) => cameraCenter(view.pose).join(' ') + ' ' + CAMERA_PLY_COLOR.join(' '),
    ),
  ];
  return (
    [
      'ply',
      'format ascii 1.0',
      'element vertex ' + vertices.length,
      'property float x',
      'property float y',
      'property float z',
      'property uchar red',
      'property uchar green',
      'property uchar blue',
      'end_header',
      ...vertices,
    ].join('\n') + '\n'
  );
}
