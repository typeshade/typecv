import { validateCamera } from '../../geometry/camera.ts';
import type { Camera } from '../../geometry/camera.ts';
import { validatePose } from '../../geometry/pose.ts';
import type { Pose } from '../../geometry/pose.ts';
import type { Landmark, RegisteredView, SparseReconstruction } from '../../reconstruction/types.ts';

/** Doubles per camera pose: row-major R, then t. */
export const POSE_STRIDE = 12;
/** Doubles per camera model: fx, fy, cx, cy, k1, k2. */
export const MODEL_STRIDE = 6;
/** Doubles per point. */
export const POINT_STRIDE = 3;
/** Doubles per observation pixel. */
export const PIXEL_STRIDE = 2;
/** Bit 0 of a point or observation flag: alive. */
export const ALIVE = 1;
/** Bit 0 of a camera flag: variable in the current bundle window. */
export const VARIABLE = 1;

/** Observations of one group in one order: `order[start[i] .. start[i + 1])` belong to group `i`. */
export interface SceneOrder {
  readonly start: Uint32Array;
  readonly order: Uint32Array;
}
export interface SceneCapacity {
  readonly cameras?: number;
  readonly points?: number;
  readonly observations?: number;
}

function grown<T extends Float64Array | Uint32Array | Int32Array>(array: T, capacity: number): T {
  const next = new (array.constructor as new (length: number) => T)(capacity);
  next.set(array);
  return next;
}

/**
 * The map as typed arrays with the layout of `docs/gpu-resident-scene.md`: cameras, points and
 * observations with a count and a capacity each. Cameras are appended and never removed. Points
 * and observations are appended, flagged dead by a filter, and dropped by `compact()`. The arrays
 * are the public state: a driver reads and writes them in place through the indices this class
 * hands out, and `snapshot()` builds the object model on request only.
 */
export class HostScene {
  cameraCount = 0;
  pointCount = 0;
  observationCount = 0;
  /** View id of each camera. */
  cameraIds: Int32Array;
  cameraPoses: Float64Array;
  cameraModels: Float64Array;
  cameraFlags: Uint32Array;
  /** Track id of each point. */
  pointTracks: Int32Array;
  points: Float64Array;
  pointFlags: Uint32Array;
  /** Alive observations of each point, kept current by the observation writes. */
  pointObservations: Uint32Array;
  obsCamera: Uint32Array;
  obsPoint: Uint32Array;
  /** Feature id of each observation, for the snapshot; the device never reads it. */
  obsFeature: Uint32Array;
  obsPixel: Float64Array;
  obsFlags: Uint32Array;
  private readonly cameraOf = new Map<number, number>();
  private readonly pointOf = new Map<number, number>();

  constructor(capacity: SceneCapacity = {}) {
    const cameras = capacity.cameras ?? 16,
      points = capacity.points ?? 1024,
      observations = capacity.observations ?? 4096;
    if (
      ![cameras, points, observations].every((n) => Number.isSafeInteger(n) && n >= 1) ||
      cameras > 2 ** 31
    )
      throw new RangeError('Scene capacities must be positive safe integers');
    this.cameraIds = new Int32Array(cameras);
    this.cameraPoses = new Float64Array(cameras * POSE_STRIDE);
    this.cameraModels = new Float64Array(cameras * MODEL_STRIDE);
    this.cameraFlags = new Uint32Array(cameras);
    this.pointTracks = new Int32Array(points);
    this.points = new Float64Array(points * POINT_STRIDE);
    this.pointFlags = new Uint32Array(points);
    this.pointObservations = new Uint32Array(points);
    this.obsCamera = new Uint32Array(observations);
    this.obsPoint = new Uint32Array(observations);
    this.obsFeature = new Uint32Array(observations);
    this.obsPixel = new Float64Array(observations * PIXEL_STRIDE);
    this.obsFlags = new Uint32Array(observations);
  }
  get cameraCapacity(): number {
    return this.cameraIds.length;
  }
  get pointCapacity(): number {
    return this.pointTracks.length;
  }
  get observationCapacity(): number {
    return this.obsCamera.length;
  }

  /** Camera index of a registered view, if registered. */
  cameraIndex(viewId: number): number | undefined {
    return this.cameraOf.get(viewId);
  }
  /** Point index of a track's alive point, if it has one. */
  pointIndex(trackId: number): number | undefined {
    return this.pointOf.get(trackId);
  }
  /** Append a camera. Throws on a view already registered or an invalid camera or pose. */
  registerCamera(viewId: number, camera: Camera, pose: Pose): number {
    if (!Number.isSafeInteger(viewId) || viewId < 0 || viewId > 0x7fffffff)
      throw new RangeError('View id must be a nonnegative 31-bit integer');
    if (this.cameraOf.has(viewId)) throw new RangeError(`View ${viewId} is already registered`);
    validateCamera(camera);
    validatePose(pose);
    if (this.cameraCount === this.cameraCapacity) this.growCameras(this.cameraCapacity * 2);
    const index = this.cameraCount++;
    this.cameraIds[index] = viewId;
    this.cameraFlags[index] = 0;
    this.setPose(index, pose);
    this.setCamera(index, camera);
    this.cameraOf.set(viewId, index);
    return index;
  }
  setPose(index: number, pose: Pose): void {
    this.checkCamera(index);
    this.cameraPoses.set(pose.R, index * POSE_STRIDE);
    this.cameraPoses.set(pose.t, index * POSE_STRIDE + 9);
  }
  setCamera(index: number, camera: Camera): void {
    this.checkCamera(index);
    this.cameraModels.set(
      [camera.fx, camera.fy, camera.cx, camera.cy, camera.k1 ?? 0, camera.k2 ?? 0],
      index * MODEL_STRIDE,
    );
  }
  /** A copy of a camera's pose. */
  pose(index: number): Pose {
    this.checkCamera(index);
    const base = index * POSE_STRIDE;
    return {
      R: this.cameraPoses.slice(base, base + 9),
      t: this.cameraPoses.slice(base + 9, base + 12),
    };
  }
  /** A copy of a camera's model. Zero distortion is reported as absent, as the inputs spell it. */
  camera(index: number): Camera {
    this.checkCamera(index);
    const m = this.cameraModels,
      base = index * MODEL_STRIDE,
      camera: { -readonly [K in keyof Camera]: Camera[K] } = {
        fx: m[base]!,
        fy: m[base + 1]!,
        cx: m[base + 2]!,
        cy: m[base + 3]!,
      };
    if (m[base + 4] !== 0) camera.k1 = m[base + 4]!;
    if (m[base + 5] !== 0) camera.k2 = m[base + 5]!;
    return camera;
  }

  /** Append an alive point for a track. Throws while the track has an alive point. */
  addPoint(trackId: number, xyz: ArrayLike<number>): number {
    if (!Number.isSafeInteger(trackId) || trackId < 0 || trackId > 0x7fffffff)
      throw new RangeError('Track id must be a nonnegative 31-bit integer');
    if (this.pointOf.has(trackId)) throw new RangeError(`Track ${trackId} already has a point`);
    if (xyz.length !== POINT_STRIDE || !Array.from(xyz).every(Number.isFinite))
      throw new RangeError('A point is three finite numbers');
    if (this.pointCount === this.pointCapacity) this.growPoints(this.pointCapacity * 2);
    const index = this.pointCount++;
    this.pointTracks[index] = trackId;
    this.points.set(xyz, index * POINT_STRIDE);
    this.pointFlags[index] = ALIVE;
    this.pointObservations[index] = 0;
    this.pointOf.set(trackId, index);
    return index;
  }
  setPoint(index: number, xyz: ArrayLike<number>): void {
    this.checkPoint(index);
    if (xyz.length !== POINT_STRIDE) throw new RangeError('A point is three numbers');
    this.points.set(xyz, index * POINT_STRIDE);
  }
  point(index: number): Float64Array {
    this.checkPoint(index);
    return this.points.slice(index * POINT_STRIDE, index * POINT_STRIDE + POINT_STRIDE);
  }
  pointAlive(index: number): boolean {
    this.checkPoint(index);
    return (this.pointFlags[index]! & ALIVE) !== 0;
  }
  /** Flag a point dead with every observation it owns. Its track may get a new point later. */
  removePoint(index: number): void {
    if (!this.pointAlive(index)) return;
    this.pointFlags[index] = 0;
    this.pointOf.delete(this.pointTracks[index]!);
    for (let i = 0; i < this.observationCount; i++)
      if (this.obsPoint[i] === index) this.obsFlags[i] = 0;
    this.pointObservations[index] = 0;
  }

  /** Append an alive observation of an alive point by a registered camera. */
  addObservation(
    cameraIndex: number,
    pointIndex: number,
    featureId: number,
    x: number,
    y: number,
  ): number {
    this.checkCamera(cameraIndex);
    if (!this.pointAlive(pointIndex)) throw new RangeError(`Point ${pointIndex} is not alive`);
    if (!Number.isSafeInteger(featureId) || featureId < 0 || featureId > 0xffffffff)
      throw new RangeError('Feature id must be a nonnegative 32-bit integer');
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new RangeError('Pixel must be finite');
    if (this.observationCount === this.observationCapacity)
      this.growObservations(this.observationCapacity * 2);
    const index = this.observationCount++;
    this.obsCamera[index] = cameraIndex;
    this.obsPoint[index] = pointIndex;
    this.obsFeature[index] = featureId;
    this.obsPixel[index * PIXEL_STRIDE] = x;
    this.obsPixel[index * PIXEL_STRIDE + 1] = y;
    this.obsFlags[index] = ALIVE;
    this.pointObservations[pointIndex]!++;
    return index;
  }
  observationAlive(index: number): boolean {
    this.checkObservation(index);
    return (this.obsFlags[index]! & ALIVE) !== 0;
  }
  /** Flag an observation dead. The point keeps its other observations. */
  removeObservation(index: number): void {
    if (!this.observationAlive(index)) return;
    this.obsFlags[index] = 0;
    this.pointObservations[this.obsPoint[index]!]!--;
  }

  /** Alive observations, point-major, in arrival order within a point. */
  pointOrder(): SceneOrder {
    return this.order(this.pointCount, this.obsPoint);
  }
  /** Alive observations, camera-major, in arrival order within a camera. */
  cameraOrder(): SceneOrder {
    return this.order(this.cameraCount, this.obsCamera);
  }
  private order(groups: number, key: Uint32Array): SceneOrder {
    const start = new Uint32Array(groups + 1);
    for (let i = 0; i < this.observationCount; i++)
      if (this.obsFlags[i]! & ALIVE) start[key[i]! + 1]!++;
    for (let g = 0; g < groups; g++) start[g + 1]! += start[g]!;
    const order = new Uint32Array(start[groups]!),
      cursor = start.slice(0, groups);
    for (let i = 0; i < this.observationCount; i++)
      if (this.obsFlags[i]! & ALIVE) order[cursor[key[i]!]!++] = i;
    return { start, order };
  }

  /**
   * Drop the dead points and observations and renumber the rest in order. Returns what was
   * dropped and, for each point index before the call, its index after it or -1. Point indices
   * handed out before the call are stale after it; track ids are not.
   */
  compact(): { pointsRemoved: number; observationsRemoved: number; pointRemap: Int32Array } {
    const pointMap = new Int32Array(this.pointCount).fill(-1);
    let pointsKept = 0;
    for (let p = 0; p < this.pointCount; p++) {
      if (!(this.pointFlags[p]! & ALIVE)) continue;
      const to = pointsKept++;
      pointMap[p] = to;
      if (to !== p) {
        this.pointTracks[to] = this.pointTracks[p]!;
        this.points.copyWithin(
          to * POINT_STRIDE,
          p * POINT_STRIDE,
          p * POINT_STRIDE + POINT_STRIDE,
        );
        this.pointFlags[to] = ALIVE;
        this.pointObservations[to] = this.pointObservations[p]!;
        this.pointOf.set(this.pointTracks[to]!, to);
      }
    }
    let kept = 0;
    for (let i = 0; i < this.observationCount; i++) {
      if (!(this.obsFlags[i]! & ALIVE)) continue;
      const point = pointMap[this.obsPoint[i]!]!;
      if (point < 0) continue;
      const to = kept++;
      if (to !== i) {
        this.obsCamera[to] = this.obsCamera[i]!;
        this.obsFeature[to] = this.obsFeature[i]!;
        this.obsPixel[to * PIXEL_STRIDE] = this.obsPixel[i * PIXEL_STRIDE]!;
        this.obsPixel[to * PIXEL_STRIDE + 1] = this.obsPixel[i * PIXEL_STRIDE + 1]!;
        this.obsFlags[to] = ALIVE;
      }
      this.obsPoint[to] = point;
    }
    const result = {
      pointsRemoved: this.pointCount - pointsKept,
      observationsRemoved: this.observationCount - kept,
      pointRemap: pointMap,
    };
    this.pointCount = pointsKept;
    this.observationCount = kept;
    return result;
  }

  /** The alive cameras, points and observations as the object model, copied. */
  snapshot(): SparseReconstruction {
    const views: RegisteredView[] = [];
    for (let c = 0; c < this.cameraCount; c++)
      views.push({ id: this.cameraIds[c]!, camera: this.camera(c), pose: this.pose(c) });
    const { start, order } = this.pointOrder(),
      landmarks: Landmark[] = [];
    for (let p = 0; p < this.pointCount; p++) {
      if (!(this.pointFlags[p]! & ALIVE)) continue;
      const observations = [];
      for (let k = start[p]!; k < start[p + 1]!; k++) {
        const i = order[k]!;
        observations.push({
          viewId: this.cameraIds[this.obsCamera[i]!]!,
          featureId: this.obsFeature[i]!,
          pixel: [this.obsPixel[i * PIXEL_STRIDE]!, this.obsPixel[i * PIXEL_STRIDE + 1]!] as const,
        });
      }
      landmarks.push({ trackId: this.pointTracks[p]!, xyz: this.point(p), observations });
    }
    return { views, landmarks };
  }
  /** A scene holding a model: views in order, landmarks in order, observations in order. */
  static fromReconstruction(model: SparseReconstruction, capacity: SceneCapacity = {}): HostScene {
    const observations = model.landmarks.reduce((n, p) => n + p.observations.length, 0);
    const scene = new HostScene({
      cameras: Math.max(capacity.cameras ?? 0, model.views.length, 1),
      points: Math.max(capacity.points ?? 0, model.landmarks.length, 1),
      observations: Math.max(capacity.observations ?? 0, observations, 1),
    });
    for (const view of model.views) scene.registerCamera(view.id, view.camera, view.pose);
    for (const point of model.landmarks) {
      const index = scene.addPoint(point.trackId, point.xyz);
      for (const o of point.observations) {
        const camera = scene.cameraIndex(o.viewId);
        if (camera === undefined)
          throw new RangeError(`Observation of unregistered view ${o.viewId}`);
        scene.addObservation(camera, index, o.featureId, o.pixel[0], o.pixel[1]);
      }
    }
    return scene;
  }

  private checkCamera(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.cameraCount)
      throw new RangeError(`Camera index ${index} out of range`);
  }
  private checkPoint(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.pointCount)
      throw new RangeError(`Point index ${index} out of range`);
  }
  private checkObservation(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.observationCount)
      throw new RangeError(`Observation index ${index} out of range`);
  }
  private growCameras(capacity: number): void {
    this.cameraIds = grown(this.cameraIds, capacity);
    this.cameraPoses = grown(this.cameraPoses, capacity * POSE_STRIDE);
    this.cameraModels = grown(this.cameraModels, capacity * MODEL_STRIDE);
    this.cameraFlags = grown(this.cameraFlags, capacity);
  }
  private growPoints(capacity: number): void {
    this.pointTracks = grown(this.pointTracks, capacity);
    this.points = grown(this.points, capacity * POINT_STRIDE);
    this.pointFlags = grown(this.pointFlags, capacity);
    this.pointObservations = grown(this.pointObservations, capacity);
  }
  private growObservations(capacity: number): void {
    this.obsCamera = grown(this.obsCamera, capacity);
    this.obsPoint = grown(this.obsPoint, capacity);
    this.obsFeature = grown(this.obsFeature, capacity);
    this.obsPixel = grown(this.obsPixel, capacity * PIXEL_STRIDE);
    this.obsFlags = grown(this.obsFlags, capacity);
  }
}
