import { useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import {
  cameraPosition,
  depthColor,
  fitGrid,
  gridChanged,
  type GridFit,
  type Scene as SceneData,
  type SceneView,
} from '../lib/sfm-scene';
import {
  DENSE_POINT_BUDGET,
  denseDrawnCount,
  denseStride,
  type DenseChunk,
  type DenseCloud,
  type DenseMeshData,
} from '../lib/dense-cloud';
import { denseGeometry } from '../lib/dense-geometry';
import { dominantPlane, orientTowards, planeFromPoints, quantile } from '../lib/scene-align';
import {
  Change,
  PULSE_HOLD_MS,
  addedViews,
  flightOrigin,
  landmarkChanges,
  partnerView,
  viewLinks,
  type PositionIndex,
  type PulseTarget,
} from '../lib/live-view';

/** The placement of the map on the ground grid: a move, a turn (degrees, XYZ order) and a
 *  uniform scale, in the scene's frame (y up). */
export type MapPlacement = {
  position: [number, number, number];
  rotationDeg: [number, number, number];
  scale: number;
};
/** What the viewer reports about the placement: the values, and the ground points picked so far
 *  (-1 when it is not picking). `message` says why the last command did nothing. */
export type PlacementState = MapPlacement & { picking: number; message?: string };

export type ViewerState =
  { status: 'initializing' } | { status: 'ready' } | { status: 'error'; message: string };
type Props = {
  reconstruction?: SceneData;
  selectedViewId?: number;
  photo?: { id: number; image?: string; width: number; height: number };
  /** Look through a registered camera: the orbit camera takes the view's pose and field of view.
   *  A new `nonce` applies it again for the same view. */
  cameraView?: { id: number; nonce: number };
  /** A new value refits the orbit camera to the whole map. */
  fitNonce?: number;
  /** A preset view of the whole map: from above, from the front or from the side. A new `nonce`
   *  applies it again. */
  viewPreset?: { kind: 'top' | 'front' | 'side'; nonce: number };
  /** Draw the frusta, the trajectory and the selected camera's arrows (true by default). */
  showCameras?: boolean;
  /** Draw the selected camera's photo plane (true by default). */
  showPhoto?: boolean;
  /** A light background with dark grid lines, instead of the dark one. */
  lightBackground?: boolean;
  /** `orbit` (the default): drag turns about the target, the wheel zooms to the cursor, a double
   *  click moves the target to a camera. `fly`: drag turns the camera in place, W A S D
   *  and Q E move it along its axes (Shift four times as fast), the wheel scales the speed. */
  navigation?: 'orbit' | 'fly';
  /** The dense cloud of the run (`docs/dense-mvs.md` section 6). It grows as chunks arrive. */
  dense?: DenseCloud;
  /** Draw the dense cloud in place of the sparse landmarks while it has points (true by default). */
  showDense?: boolean;
  /** The most dense points drawn (section 6). Over it the viewer draws every k-th point. */
  densePointBudget?: number;
  /** The mesh of the run (`docs/dense-mvs.md` section 3.13). */
  mesh?: DenseMeshData;
  /** Draw the mesh in place of the points (false by default). */
  showMesh?: boolean;
  /**
   * The color of the cameras of each video, by `videoId` (a CSS color; `docs/multi-video.md` 6.3).
   * A view of another video, or without a `videoId`, keeps the default color.
   */
  videoColors?: ReadonlyMap<string, string>;
  /** The handles on the map: none (the default), move, turn or scale. */
  transformMode?: 'off' | 'translate' | 'rotate' | 'scale';
  /**
   * A placement command; a new `nonce` runs it again. `reset` undoes the placement, `ground`
   * puts the map's lowest points on the grid with its middle above the origin, `auto` levels the
   * plane that holds most points (the cameras above it), `pick` levels the plane of the next three
   * double-clicked points, and `set` applies `value`.
   */
  placementCommand?: {
    kind: 'reset' | 'ground' | 'auto' | 'pick' | 'set';
    nonce: number;
    value?: MapPlacement;
  };
  onPlacement?: (state: PlacementState) => void;
  /**
   * A run is in flight (`docs/lab-live-view.md`): what a new map adds glows, what it moves
   * slides to its new place, and what it drops fades out in red. Off, a new map is drawn as it is.
   */
  live?: boolean;
  /**
   * The stage the worker reported last, and when (`performance.now()`). While it is recent the
   * cameras its work touches pulse: every camera for bundle adjustment, the newest for the other
   * stages of a frame.
   */
  activity?: { target: PulseTarget; at: number };
  /**
   * Every feature detected in a view's frame, x and y in analysis pixels. When `viewId` is the
   * selected view, the features are drawn in gray on the plane of its photo.
   */
  features?: { viewId: number; points: ArrayLike<number> };
  /** Draw the selected view's 2D–3D links (`docs/lab-live-view.md` 3.5; true by default). */
  showLinks?: boolean;
  onSelectView: (id: number) => void;
  onState: (state: ViewerState) => void;
};
const EMPTY_SCENE: SceneData = { views: [], landmarks: [] };
/** The live view's timings in seconds (`docs/lab-live-view.md`). */
const TWEEN_S = 0.65;
const GLOW_S = 1.6;
const GHOST_S = 1.2;
/** The birth time of what was there before: long ago, so it does not glow. */
const OLD = -1e6;
/** A pulse alone redraws at most this often, to leave the GPU to the run. */
const PULSE_FRAME_MS = 50;

/**
 * The live view's uniforms, shared by the sparse points and the camera lines. The shader moves a
 * vertex from `aFrom` to its place over `TWEEN_S` after `uTweenStart`, lights what was born less
 * than `GLOW_S` ago, and brightens the pulsing cameras (`aLatest` marks the newest).
 */
function liveMaterial(
  material: THREE.PointsMaterial | THREE.LineBasicMaterial,
  uniforms: Record<string, THREE.IUniform<number>>,
  kind: 'points' | 'lines',
): void {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        'void main() {',
        'attribute vec3 aFrom;\nattribute float aBorn;\n' +
          (kind === 'lines' ? 'attribute float aLatest;\n' : '') +
          'uniform float uNow;\nuniform float uTweenStart;\n' +
          'uniform float uPulseAll;\nuniform float uPulseLatest;\n' +
          'void main() {\n' +
          `  float liveGlow = 1.0 - clamp((uNow - aBorn) / ${GLOW_S.toFixed(2)}, 0.0, 1.0);`,
      )
      .replace(
        '#include <color_vertex>',
        '#include <color_vertex>\n' +
          '  vColor.rgb = mix(vColor.rgb, vec3(1.0, 0.92, 0.55), liveGlow);\n' +
          (kind === 'lines'
            ? '  float livePulse = max(uPulseAll, uPulseLatest * aLatest);\n' +
              '  vColor.rgb = mix(vColor.rgb, vec3(1.0), 0.6 * livePulse);\n'
            : ''),
      )
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n' +
          `  float liveTween = smoothstep(0.0, 1.0, clamp((uNow - uTweenStart) / ${TWEEN_S.toFixed(2)}, 0.0, 1.0));\n` +
          '  transformed = mix(aFrom, transformed, liveTween);',
      );
    if (kind === 'points')
      shader.vertexShader = shader.vertexShader.replace(
        'gl_PointSize = size;',
        'gl_PointSize = size * (1.0 + 1.2 * liveGlow);',
      );
  };
  material.customProgramCacheKey = () => 'typecv-live-' + kind;
}
const PHOTO_CACHE = 8;

/** sRGB byte to linear float, so a point color costs one table lookup instead of a Color object. */
const SRGB_TO_LINEAR = Float32Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

/** A float attribute that grows by doubling and uploads only the written prefix. */
class GrowingAttribute {
  attribute: THREE.BufferAttribute;
  array: Float32Array;
  constructor(
    private geometry: THREE.BufferGeometry,
    private name: string,
    private itemSize: number,
    capacity = 1024,
  ) {
    this.array = new Float32Array(capacity * itemSize);
    this.attribute = new THREE.BufferAttribute(this.array, itemSize);
    this.attribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, this.attribute);
  }
  ensure(count: number): void {
    if (count * this.itemSize <= this.array.length) return;
    let capacity = this.array.length / this.itemSize;
    while (capacity < count) capacity *= 2;
    this.array = new Float32Array(capacity * this.itemSize);
    this.attribute = new THREE.BufferAttribute(this.array, this.itemSize);
    this.attribute.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute(this.name, this.attribute);
  }
  commit(count: number): void {
    this.attribute.clearUpdateRanges();
    this.attribute.addUpdateRange(0, count * this.itemSize);
    this.attribute.needsUpdate = true;
  }
}

/** World-space camera center and axes (`cameraPosition`, `cameraAxes`), without allocations. */
function cameraFrame(view: SceneView, center: THREE.Vector3, axes: THREE.Vector3[]): void {
  const { R, t } = view.pose;
  center.set(
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  );
  // The pose is world-to-camera (x_c = R x_w + t), so camera axis k in the world frame is
  // Rᵀ e_k: row k of R (`cameraAxes`). Column k is world axis k in the camera frame, which is
  // what the frusta and the forward arrow were drawn with before 2026-10-20.
  for (let k = 0; k < 3; k++) axes[k].set(R[3 * k], R[3 * k + 1], R[3 * k + 2]);
}
function cameraMatrix(view: SceneView): THREE.Matrix4 {
  const { R, t } = view.pose;
  return new THREE.Matrix4()
    .set(R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1)
    .invert();
}
const viewOrder = (view: SceneView) => view.sequence ?? view.time ?? view.id;
const consecutive = (previous: SceneView, view: SceneView) =>
  // A multi-video scene joins the cameras of one video only (`docs/multi-video.md` 6.3).
  previous.videoId === view.videoId &&
  (view.sequence !== undefined
    ? previous.sequence! + 1 === view.sequence
    : previous.id + 1 === view.id);

export default function SfmViewer(props: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const disposeRef = useRef<() => void>(() => {});
  const refreshRef = useRef<() => void>(() => {});
  const report = useCallback((state: ViewerState) => {
    const parent = host.current;
    // Detached-host cleanup only disposes resources; the parent aborts its readiness waiter.
    if (!parent) return;
    parent.dataset.state = state.status;
    if (state.status === 'error') parent.dataset.error = state.message;
    else delete parent.dataset.error;
    latest.current.onState(state);
  }, []);
  const initialize = useCallback(() => {
    const parent = host.current!;
    let disposed = false;
    const releases: (() => void)[] = [];
    const own = <T extends { dispose: () => void }>(resource: T): T => {
      releases.push(() => resource.dispose());
      return resource;
    };
    const dispose = () => {
      if (disposed) return;
      disposed = true;
      for (const release of releases.reverse()) {
        try {
          release();
        } catch (error) {
          console.error(error);
        }
      }
    };
    disposeRef.current = dispose;
    const fail = (error: unknown) => {
      dispose();
      report({ status: 'error', message: '3D viewer failed: ' + String(error) });
    };
    report({ status: 'initializing' });
    const renderer = own(new THREE.WebGLRenderer({ antialias: true }));
    releases.push(() => renderer.domElement.remove());
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setClearColor('#0b1020', 1);
    let lightBackground = false;
    parent.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    // The mapper's world is the first camera's frame: x right, y down, z forward (image rows grow
    // downwards). three.js draws y up, so every object of the map lives in a root turned half a
    // turn about x (y -> -y, z -> -z). A view then stands upright and looks along -z, and the
    // map's coordinates stay the mapper's for picking and for the record.
    const root = new THREE.Group();
    root.rotation.x = Math.PI;
    // The visitor's placement of the map on the grid: `placement` turns and scales about its own
    // origin, which `pivot` keeps at the map's middle, and the grid stays in the scene's frame.
    const placement = new THREE.Group(),
      pivot = new THREE.Group();
    placement.add(pivot);
    pivot.add(root);
    scene.add(placement);
    placement.updateMatrixWorld(true);
    const DEFAULT_FOV = 48;
    const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.001, 100000);
    const controls = own(new OrbitControls(camera, renderer.domElement));
    controls.enableDamping = true;
    // The wheel zooms toward the point under the cursor, so a visitor dives into one part of a
    // small-scale map instead of toward a far target; the distance bounds follow the extent.
    controls.zoomToCursor = true;
    controls.addEventListener('start', () => (cameraTouched = true));
    renderer.domElement.tabIndex = 0;
    renderer.domElement.style.outline = 'none';
    // Fly mode's state (the handlers are below the frame loop that reads it).
    let fly = false;
    let flySpeed = 1;
    const flyKeys = new Set<string>();
    const flyForward = new THREE.Vector3(),
      flyRight = new THREE.Vector3();

    // Persistent geometry: the point cloud, the camera frusta and the trajectory grow in place.
    const cloudGeometry = own(new THREE.BufferGeometry());
    const positions = new GrowingAttribute(cloudGeometry, 'position', 3);
    const colors = new GrowingAttribute(cloudGeometry, 'color', 3);
    // The live view (`docs/lab-live-view.md`): where each point was in the previous map, and when
    // it was born. The uniforms drive every live material at once.
    const cloudFrom = new GrowingAttribute(cloudGeometry, 'aFrom', 3);
    const cloudBorn = new GrowingAttribute(cloudGeometry, 'aBorn', 1);
    const liveUniforms = {
      uNow: { value: 0 },
      uTweenStart: { value: OLD },
      uPulseAll: { value: 0 },
      uPulseLatest: { value: 0 },
    };
    const clock = () => performance.now() / 1000;
    const reducedMotion =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cloudMaterial = own(
      new THREE.PointsMaterial({ vertexColors: true, size: 3, sizeAttenuation: false }),
    );
    liveMaterial(cloudMaterial, liveUniforms, 'points');
    const cloud = new THREE.Points(cloudGeometry, cloudMaterial);
    cloud.frustumCulled = false;
    root.add(cloud);
    const frustumGeometry = own(new THREE.BufferGeometry());
    const frusta = new GrowingAttribute(frustumGeometry, 'position', 3, 16 * 8 * 2);
    // The line colors come from the vertices: the default orange, or the color of a video.
    const lineMaterial = own(new THREE.LineBasicMaterial({ vertexColors: true }));
    liveMaterial(lineMaterial, liveUniforms, 'lines');
    const frustumColors = new GrowingAttribute(frustumGeometry, 'color', 3, 16 * 8 * 2);
    const frustumFrom = new GrowingAttribute(frustumGeometry, 'aFrom', 3, 16 * 8 * 2);
    const frustumBorn = new GrowingAttribute(frustumGeometry, 'aBorn', 1, 16 * 8 * 2);
    const frustumLatest = new GrowingAttribute(frustumGeometry, 'aLatest', 1, 16 * 8 * 2);
    const defaultLine = new THREE.Color('#ffb454');
    const lineColor = new THREE.Color();
    const frustumLines = new THREE.LineSegments(frustumGeometry, lineMaterial);
    frustumLines.frustumCulled = false;
    root.add(frustumLines);
    const pathGeometry = own(new THREE.BufferGeometry());
    const path = new GrowingAttribute(pathGeometry, 'position', 3, 16 * 2);
    const pathColors = new GrowingAttribute(pathGeometry, 'color', 3, 16 * 2);
    const pathFrom = new GrowingAttribute(pathGeometry, 'aFrom', 3, 16 * 2);
    const pathBorn = new GrowingAttribute(pathGeometry, 'aBorn', 1, 16 * 2);
    const pathLatest = new GrowingAttribute(pathGeometry, 'aLatest', 1, 16 * 2);
    const pathLines = new THREE.LineSegments(pathGeometry, lineMaterial);
    pathLines.frustumCulled = false;
    root.add(pathLines);
    cloudGeometry.setDrawRange(0, 0);
    // The points a new map dropped: red marks that fade out over GHOST_S.
    const ghostGeometry = own(new THREE.BufferGeometry());
    ghostGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
    const ghostMaterial = own(
      new THREE.PointsMaterial({
        color: '#ff5a4f',
        size: 4,
        sizeAttenuation: false,
        transparent: true,
        depthWrite: false,
      }),
    );
    const ghosts = new THREE.Points(ghostGeometry, ghostMaterial);
    ghosts.frustumCulled = false;
    ghosts.visible = false;
    root.add(ghosts);
    let ghostsAt = OLD;
    // What the previous map held, for the comparison with the next one.
    let previousPoints: PositionIndex = new Map();
    let previousViews = new Set<number>();
    let previousFrusta = new Map<number, Float32Array>();
    let previousCenters = new Map<number, [number, number, number]>();
    let latestViewId: number | undefined;
    /** The registered views in the order they appeared, for the start of a flight (3.6). */
    let registration: number[] = [];
    /** The live view animates until this time (seconds of `clock`). */
    let animateUntil = OLD;
    // The dense cloud (section 6): a second `Points` with its own geometry, 12 B of position and 3 B
    // of normalized color per drawn point. The vertex shader turns the sRGB bytes linear, as the
    // sparse cloud's table does. Growth makes a new geometry and disposes the old one, so a
    // replaced attribute frees its GPU buffer at once.
    const denseMaterial = own(
      new THREE.PointsMaterial({ vertexColors: true, size: 2, sizeAttenuation: false }),
    );
    denseMaterial.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        '#include <color_vertex>',
        '#include <color_vertex>\n' +
          'vColor.rgb = mix(vColor.rgb / 12.92, pow((vColor.rgb + 0.055) / 1.055, vec3(2.4)),' +
          ' step(vec3(0.04045), vColor.rgb));',
      );
    };
    denseMaterial.customProgramCacheKey = () => 'typecv-dense-srgb';
    const dense = {
      geometry: new THREE.BufferGeometry(),
      positions: new Float32Array(0),
      colors: new Uint8Array(0),
      /** The run, the chunks read, the stride, the points read and the points drawn. */
      run: NaN,
      chunks: 0,
      stride: 1,
      read: 0,
      drawn: 0,
    };
    const denseCloud = new THREE.Points(dense.geometry, denseMaterial);
    denseCloud.frustumCulled = false;
    root.add(denseCloud);
    // The mesh (section 3.13): vertex colors as sRGB bytes, turned linear in the vertex shader like
    // the dense points. The colors come from the photos and already hold their light, so the
    // lights give a face toward the camera its color unchanged and a grazing face 70 % of it:
    // an ambient term and a light at the camera, each scaled by pi for the Lambert term. They
    // hang off the camera and the scene, not the root, whose half turn about x would flip them.
    const meshMaterial = own(
      new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }),
    );
    meshMaterial.onBeforeCompile = denseMaterial.onBeforeCompile;
    meshMaterial.customProgramCacheKey = () => 'typecv-mesh-srgb';
    const meshAmbient = new THREE.AmbientLight(0xffffff, 0.7 * Math.PI);
    const headLight = new THREE.DirectionalLight(0xffffff, 0.3 * Math.PI);
    headLight.position.set(0, 0, 0);
    headLight.target.position.set(0, 0, -1);
    camera.add(headLight, headLight.target);
    scene.add(camera, meshAmbient);
    const meshObject = new THREE.Mesh(new THREE.BufferGeometry(), meshMaterial);
    meshObject.frustumCulled = false;
    meshObject.visible = false;
    root.add(meshObject);
    releases.push(() => meshObject.geometry.dispose());
    let shownMeshData: DenseMeshData | undefined;
    const meshUpdate = (data: DenseMeshData | undefined) => {
      meshObject.geometry.dispose();
      const geometry = new THREE.BufferGeometry();
      if (data) {
        geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3, true));
        geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
        geometry.computeVertexNormals();
      }
      meshObject.geometry = geometry;
    };
    /** A new geometry of `capacity` points that keeps the drawn prefix. */
    const denseCapacity = (capacity: number) => {
      const positions = new Float32Array(capacity * 3),
        colors = new Uint8Array(capacity * 3);
      positions.set(dense.positions.subarray(0, dense.drawn * 3));
      colors.set(dense.colors.subarray(0, dense.drawn * 3));
      dense.geometry.dispose();
      dense.geometry = denseGeometry(positions, colors);
      dense.geometry.setDrawRange(0, dense.drawn);
      dense.positions = positions;
      dense.colors = colors;
      denseCloud.geometry = dense.geometry;
    };
    /** Drop every dense point and the GPU buffers that held them. */
    const denseClear = () => {
      dense.chunks = dense.read = dense.drawn = 0;
      dense.stride = 1;
      denseCapacity(1024);
    };
    denseCapacity(1024);
    releases.push(() => dense.geometry.dispose());
    /** Append the points of a chunk whose global index is a multiple of the stride. */
    const denseAppend = (chunk: DenseChunk) => {
      const { positions, colors } = chunk,
        n = positions.length / 3,
        k = dense.stride,
        p = dense.positions,
        c = dense.colors;
      let at = dense.drawn * 3;
      for (let i = (k - (dense.read % k)) % k; i < n; i += k, at += 3) {
        p[at] = positions[3 * i];
        p[at + 1] = positions[3 * i + 1];
        p[at + 2] = positions[3 * i + 2];
        c[at] = colors[3 * i];
        c[at + 1] = colors[3 * i + 1];
        c[at + 2] = colors[3 * i + 2];
      }
      dense.read += n;
      dense.drawn = at / 3;
    };
    /**
     * Follow the page's dense cloud: a new run or a new stride starts again from the first chunk,
     * and otherwise the new chunks append. Only the appended range uploads.
     */
    const denseUpdate = (cloud: DenseCloud | undefined, budget: number) => {
      if (!cloud || !cloud.chunks.length) {
        if (dense.chunks || dense.positions.length > 3072) denseClear();
        dense.run = cloud?.run ?? NaN;
        return;
      }
      const stride = denseStride(cloud.count, budget);
      if (
        cloud.run !== dense.run ||
        stride !== dense.stride ||
        cloud.chunks.length < dense.chunks
      ) {
        denseClear();
        dense.run = cloud.run;
        dense.stride = stride;
      }
      if (dense.chunks === cloud.chunks.length) return;
      const from = dense.drawn;
      const needed = denseDrawnCount(cloud.count, stride);
      if (needed * 3 > dense.positions.length) {
        let capacity = dense.positions.length / 3;
        while (capacity < needed) capacity *= 2;
        denseCapacity(Math.min(capacity, Math.max(needed, budget)));
      }
      for (; dense.chunks < cloud.chunks.length; dense.chunks++)
        denseAppend(cloud.chunks[dense.chunks]);
      for (const name of ['position', 'color']) {
        const attribute = dense.geometry.getAttribute(name) as THREE.BufferAttribute;
        attribute.clearUpdateRanges();
        attribute.addUpdateRange(from * 3, (dense.drawn - from) * 3);
        attribute.needsUpdate = true;
      }
      dense.geometry.setDrawRange(0, dense.drawn);
    };
    frustumGeometry.setDrawRange(0, 0);
    pathGeometry.setDrawRange(0, 0);

    // Persistent selection objects; only their transforms and visibility change.
    const forwardArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(),
      1,
      '#68ffb1',
    );
    const motionArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(),
      1,
      '#69b1ff',
    );
    const photoMaterial = own(
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.85 }),
    );
    // A 1 × 1 placeholder keeps the material on its textured program from the start, so the first
    // photo of a run swaps textures without a shader compile.
    const placeholder = own(new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1));
    placeholder.needsUpdate = true;
    photoMaterial.map = placeholder;
    const photoPlane = new THREE.Mesh(own(new THREE.PlaneGeometry(1, 1)), photoMaterial);
    photoPlane.matrixAutoUpdate = false;
    for (const object of [forwardArrow, motionArrow, photoPlane]) {
      object.visible = false;
      root.add(object);
    }
    // The selected view's 2D–3D links (`docs/lab-live-view.md` 3.5) and its partner's (3.6): the
    // features on the plane of each photo, the linked ones in depth color, and a line from each
    // to its 3D point. The ends carry the live attributes, so they slide and grow (3.6).
    const featureGeometry = own(new THREE.BufferGeometry());
    featureGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
    const featureMarks = new THREE.Points(
      featureGeometry,
      own(
        new THREE.PointsMaterial({
          color: '#a3a3a3',
          size: 2,
          sizeAttenuation: false,
          transparent: true,
          opacity: 0.8,
        }),
      ),
    );
    type LinkSet = { ends: THREE.Points; lines: THREE.LineSegments };
    const linkSet = (size: number, opacity: number): LinkSet => {
      const endGeometry = own(new THREE.BufferGeometry());
      const lineGeometry = own(new THREE.BufferGeometry());
      for (const geometry of [endGeometry, lineGeometry])
        for (const [name, size] of [
          ['position', 3],
          ['color', 3],
          ['aFrom', 3],
          ['aBorn', 1],
          ['aLatest', 1],
        ] as const)
          geometry.setAttribute(name, new THREE.Float32BufferAttribute([], size));
      const endMaterial = own(
        new THREE.PointsMaterial({ vertexColors: true, size, sizeAttenuation: false }),
      );
      liveMaterial(endMaterial, liveUniforms, 'points');
      const lineMaterial = own(
        new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity,
          depthWrite: false,
        }),
      );
      liveMaterial(lineMaterial, liveUniforms, 'lines');
      return {
        ends: new THREE.Points(endGeometry, endMaterial),
        lines: new THREE.LineSegments(lineGeometry, lineMaterial),
      };
    };
    const selectedLinks = linkSet(4, 0.6),
      partnerLinks = linkSet(3, 0.3);
    const linkObjects = [
      featureMarks,
      selectedLinks.ends,
      selectedLinks.lines,
      partnerLinks.ends,
      partnerLinks.lines,
    ];
    for (const object of linkObjects) {
      object.visible = false;
      object.frustumCulled = false;
      // Drawn after the photo, so a mark on its plane is not hidden by it.
      object.renderOrder = 2;
      root.add(object);
    }
    const linkColor = new THREE.Color();
    /** A pixel of `view` on its photo's plane, just in front of the photo, in map coordinates. */
    const planeOf = (view: SceneView) => {
      const toWorld = cameraMatrix(view);
      const depth = frustumDepth * 1.4 * 0.995;
      const { fx, fy, cx, cy } = view.camera;
      return (u: number, v: number, out: THREE.Vector3) =>
        out.set((depth * (u - cx)) / fx, (depth * (v - cy)) / fy, depth).applyMatrix4(toWorld);
    };
    // The ends drawn last, by view, track and end (0 on the plane, 1 at the point).
    let previousEnds = new Map<string, [number, number, number]>();
    const hideLinks = () => {
      for (const object of linkObjects) object.visible = false;
      parent.dataset.links = parent.dataset.partnerLinks = '0';
      parent.dataset.partner = '';
      previousEnds = new Map();
    };
    const planePoint = new THREE.Vector3();
    /** Fill one link set; returns whether an end is new or moved. */
    const fillLinks = (
      set: LinkSet,
      view: SceneView,
      links: ReturnType<typeof viewLinks>,
      colorOf: (i: number) => THREE.Color,
      live: boolean,
      now: number,
      ends: Map<string, [number, number, number]>,
    ): boolean => {
      const n = links.trackIds.length;
      const place = planeOf(view);
      const pos = new Float32Array(n * 3),
        col = new Float32Array(n * 3),
        from = new Float32Array(n * 3),
        born = new Float32Array(n),
        lpos = new Float32Array(n * 6),
        lcol = new Float32Array(n * 6),
        lfrom = new Float32Array(n * 6),
        lborn = new Float32Array(n * 2);
      let changed = false;
      for (let i = 0; i < n; i++) {
        place(links.pixels[2 * i]!, links.pixels[2 * i + 1]!, planePoint);
        const plane: [number, number, number] = [planePoint.x, planePoint.y, planePoint.z];
        const point: [number, number, number] = [
          links.points[3 * i]!,
          links.points[3 * i + 1]!,
          links.points[3 * i + 2]!,
        ];
        const key = view.id + ':' + links.trackIds[i];
        const before = [previousEnds.get(key + ':0'), previousEnds.get(key + ':1')];
        ends.set(key + ':0', plane);
        ends.set(key + ':1', point);
        // A new link grows from its pixel to its point; a known one slides from where it was.
        const start = live ? [before[0] ?? plane, before[1] ?? before[0] ?? plane] : [plane, point];
        if (live && (!before[0] || !before[1])) changed = true;
        else if (live)
          for (let k = 0; k < 3; k++)
            if (start[0]![k] !== plane[k] || start[1]![k] !== point[k]) changed = true;
        const c = colorOf(i);
        pos.set(plane, 3 * i);
        col.set([c.r, c.g, c.b], 3 * i);
        from.set(start[0]!, 3 * i);
        born[i] = live && !before[0] ? now : OLD;
        lpos.set(plane, 6 * i);
        lpos.set(point, 6 * i + 3);
        lcol.set([c.r, c.g, c.b, c.r, c.g, c.b], 6 * i);
        lfrom.set(start[0]!, 6 * i);
        lfrom.set(start[1]!, 6 * i + 3);
        lborn[2 * i] = lborn[2 * i + 1] = born[i]!;
      }
      const write = (
        geometry: THREE.BufferGeometry,
        name: string,
        data: Float32Array,
        size: number,
      ) => geometry.setAttribute(name, new THREE.Float32BufferAttribute(data, size));
      write(set.ends.geometry, 'position', pos, 3);
      write(set.ends.geometry, 'color', col, 3);
      write(set.ends.geometry, 'aFrom', from, 3);
      write(set.ends.geometry, 'aBorn', born, 1);
      write(set.ends.geometry, 'aLatest', new Float32Array(n), 1);
      write(set.lines.geometry, 'position', lpos, 3);
      write(set.lines.geometry, 'color', lcol, 3);
      write(set.lines.geometry, 'aFrom', lfrom, 3);
      write(set.lines.geometry, 'aBorn', lborn, 1);
      write(set.lines.geometry, 'aLatest', new Float32Array(n * 2), 1);
      set.ends.visible = set.lines.visible = n > 0;
      return changed;
    };
    const showLinks = (view: SceneView, model: SceneData, state: Props) => {
      if (state.showLinks === false) return hideLinks();
      const live = state.live === true && !reducedMotion;
      const now = clock();
      const place = planeOf(view);
      const features = state.features?.viewId === view.id ? state.features.points : undefined;
      const marks = new Float32Array(features ? (features.length / 2) * 3 : 0);
      for (let i = 0; features && i < features.length / 2; i++) {
        place(features[2 * i]!, features[2 * i + 1]!, planePoint);
        marks.set([planePoint.x, planePoint.y, planePoint.z], 3 * i);
      }
      featureGeometry.setAttribute('position', new THREE.Float32BufferAttribute(marks, 3));
      featureMarks.visible = marks.length > 0;
      const links = viewLinks(model, view.id);
      const n = links.depths.length;
      // The frame card's depth colors: the 5th to 95th percentile of this view's depths.
      const sorted = Array.from(links.depths).sort((a, b) => a - b);
      const range: [number, number] = n
        ? [sorted[Math.floor((n - 1) * 0.05)]!, sorted[Math.ceil((n - 1) * 0.95)]!]
        : [0, 1];
      const colors = new Map<number, THREE.Color>();
      for (let i = 0; i < n; i++) {
        const [r, g, b] = depthColor(links.depths[i]!, range);
        colors.set(links.trackIds[i]!, new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace));
      }
      const ends = new Map<string, [number, number, number]>();
      let changed = fillLinks(
        selectedLinks,
        view,
        links,
        (i) => colors.get(links.trackIds[i]!)!,
        live,
        now,
        ends,
      );
      // The partner camera (3.6): its observations of the same tracks, in the same colors.
      const partnerId = n ? partnerView(model, view.id) : undefined;
      const partner = model.views.find((candidate) => candidate.id === partnerId);
      if (partner) {
        const shared = viewLinks(model, partner.id, new Set(links.trackIds));
        changed =
          fillLinks(
            partnerLinks,
            partner,
            shared,
            (i) => colors.get(shared.trackIds[i]!) ?? linkColor.set('#a3a3a3'),
            live,
            now,
            ends,
          ) || changed;
        parent.dataset.partnerLinks = String(shared.trackIds.length);
      } else {
        partnerLinks.ends.visible = partnerLinks.lines.visible = false;
        parent.dataset.partnerLinks = '0';
      }
      parent.dataset.partner = partner ? String(partner.id) : '';
      parent.dataset.links = String(n);
      previousEnds = ends;
      // A change starts a tween unless one runs now; then the links join it.
      if (live && changed) {
        if (now - liveUniforms.uTweenStart.value >= TWEEN_S) liveUniforms.uTweenStart.value = now;
        animateUntil = Math.max(animateUntil, now + Math.max(TWEEN_S, GLOW_S));
      }
    };
    releases.push(() => {
      forwardArrow.dispose();
      motionArrow.dispose();
    });
    // Decoded photo textures, keyed by their data URL; a small cache covers timeline scrubbing.
    const textures = new Map<string, THREE.Texture>();
    const dropTexture = (key: string) => {
      textures.get(key)?.dispose();
      textures.delete(key);
    };
    releases.push(() => {
      for (const key of [...textures.keys()]) dropTexture(key);
    });
    let photoGeneration = 0;
    const showPhoto = (view: SceneView, image: string, width: number, height: number) => {
      const generation = ++photoGeneration;
      const apply = (texture: THREE.Texture) => {
        if (disposed || generation !== photoGeneration) return;
        const depth = frustumDepth * 1.4;
        const { fx, fy, cx, cy } = view.camera;
        photoMaterial.map = texture;
        photoMaterial.needsUpdate = true;
        // Image y increases downward in the calibrated camera coordinate system.
        const local = new THREE.Matrix4().makeRotationX(Math.PI);
        local.setPosition(
          (depth * ((width - 1) / 2 - cx)) / fx,
          (depth * ((height - 1) / 2 - cy)) / fy,
          depth,
        );
        const size = new THREE.Matrix4().makeScale((depth * width) / fx, (depth * height) / fy, 1);
        photoPlane.matrix.copy(cameraMatrix(view)).multiply(local).multiply(size);
        photoPlane.visible = true;
        parent.dataset.photo = String(view.id);
        requestRender();
      };
      const cached = textures.get(image);
      if (cached) {
        apply(cached);
        return;
      }
      fetch(image)
        .then((response) => response.blob())
        // `flipY` has no effect on an ImageBitmap texture (three.js uploads it as is), so the
        // bitmap itself is flipped at creation: the photo plane's rows then run as the plane's
        // UVs expect, and the photo reads upright through the camera (owner's report, 2026-10-20).
        .then((blob) =>
          createImageBitmap(blob, { colorSpaceConversion: 'none', imageOrientation: 'flipY' }),
        )
        .then((bitmap) => {
          if (disposed || generation !== photoGeneration) {
            bitmap.close();
            return;
          }
          const texture = new THREE.Texture(bitmap);
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.flipY = false;
          texture.needsUpdate = true;
          textures.set(image, texture);
          if (textures.size > PHOTO_CACHE) dropTexture(textures.keys().next().value!);
          apply(texture);
        })
        .catch((error) => console.error(error));
    };

    let extent = 1;
    let frustumDepth = 0.05;
    let fitted = false;
    /** What the last automatic framing drew, and whether the visitor has moved the camera since
     *  the run began (then the framing stays theirs). */
    let framedKind = '',
      framedCount = 0,
      cameraTouched = false;
    // The ground grid is hidden until the first points arrive, so its program compiles with the
    // others before the viewer is ready. It is rebuilt (and the old one disposed) when the fitted
    // extent of the points and cameras moves by more than 20 %.
    const makeGrid = (size: number, divisions: number) => {
      const made = new THREE.GridHelper(size, divisions, '#35415d', '#202b43');
      made.visible = false;
      scene.add(made);
      return made;
    };
    const removeGrid = (old: THREE.GridHelper) => {
      scene.remove(old);
      old.geometry.dispose();
      (old.material as THREE.Material).dispose();
    };
    let grid = makeGrid(1, 10);
    let gridFit: GridFit | undefined;
    let gridSource: { positions: ArrayLike<number>; count: number; views: SceneView[] } | undefined;
    /** The grid follows the robust extent of what is drawn (mesh, dense or sparse) in the scene's
     *  ground plane (y = 0), after the placement. The cameras do not count: a drone's path is far
     *  wider than what it filmed. */
    const refitGrid = () => {
      if (!gridSource) return;
      const { array, count } = drawnPoints();
      const stride = Math.max(1, Math.floor(count / 4096));
      const ground = new Float64Array(2 * Math.ceil(count / stride));
      const heights: number[] = [];
      const world = new THREE.Vector3();
      root.updateMatrixWorld(true);
      for (let i = 0, k = 0; i < count; i += stride, k++) {
        world
          .set(array[3 * i]!, array[3 * i + 1]!, array[3 * i + 2]!)
          .applyMatrix4(root.matrixWorld);
        ground[2 * k] = world.x;
        ground[2 * k + 1] = world.z;
        heights.push(world.y);
      }
      // The grid lies under what is drawn (its 2nd percentile of height): a floor, not a sheet
      // through the map. "Put on the grid" moves that height to 0.
      const floor = heights.length ? quantile(heights, 0.02) : 0;
      const next = fitGrid(ground);
      if (next && gridChanged(gridFit, next)) {
        const old = grid;
        grid = makeGrid(next.size, next.divisions);
        grid.position.set(next.x, floor, next.z);
        grid.visible = true;
        gridFit = next;
        removeGrid(old);
      }
      grid.position.y = floor;
    };
    releases.push(() => removeGrid(grid));
    const center = new THREE.Vector3(),
      axes = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
      corner = new THREE.Vector3(),
      previousCenter = new THREE.Vector3(),
      bounds = new THREE.Box3();
    const update = (model: SceneData) => {
      if (!model.views.length && !model.landmarks.length) {
        photoGeneration++;
        fitted = false;
        framedKind = '';
        framedCount = 0;
        cameraTouched = false;
        extent = 1;
        grid.visible = false;
        gridFit = undefined;
        gridSource = undefined;
        // A new run starts unplaced.
        placement.position.set(0, 0, 0);
        placement.quaternion.identity();
        placement.scale.setScalar(1);
        pivot.position.set(0, 0, 0);
        placement.updateMatrixWorld(true);
        picking = -1;
        picked.length = 0;
        showPicked();
        announce();
        camera.position.set(0, 0, 5);
        controls.target.set(0, 0, 0);
        controls.update();
        previousPoints = new Map();
        previousViews = new Set();
        previousFrusta = new Map();
        previousCenters = new Map();
        latestViewId = undefined;
        registration = [];
        ghosts.visible = false;
        animateUntil = OLD;
      }
      // The live view compares the new map with the one drawn before (`docs/lab-live-view.md`).
      const live = latest.current.live === true;
      const now = clock();
      const priorPoints = previousPoints;
      const pointChanges = landmarkChanges(priorPoints, model.landmarks, extent * 1e-4);
      previousPoints = pointChanges.index;
      const newViews = addedViews(previousViews, model);
      previousViews = new Set(model.views.map((view) => view.id));
      for (const view of model.views) if (newViews.has(view.id)) latestViewId = view.id;
      registration = registration.filter((id) => previousViews.has(id));
      for (const view of [...model.views].sort((a, b) => viewOrder(a) - viewOrder(b)))
        if (!registration.includes(view.id)) registration.push(view.id);
      // A new point flies from its pixel in the newest camera that sees it (3.6): drawn index, landmark index.
      const flights: number[] = [];
      if (latestViewId !== undefined && !previousViews.has(latestViewId)) latestViewId = undefined;
      latestViewId ??= model.views.reduce<SceneView | undefined>(
        (best, view) => (!best || viewOrder(view) > viewOrder(best) ? view : best),
        undefined,
      )?.id;
      let tween = live && !reducedMotion && pointChanges.moved > 0;
      if (live && pointChanges.removed.length) {
        ghostGeometry.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(pointChanges.removed, 3),
        );
        ghostsAt = now;
        ghosts.visible = true;
      }
      // Points and colors in one pass; bounds without per-point allocations.
      positions.ensure(model.landmarks.length);
      colors.ensure(model.landmarks.length);
      cloudFrom.ensure(model.landmarks.length);
      cloudBorn.ensure(model.landmarks.length);
      const pf = cloudFrom.array,
        pb = cloudBorn.array;
      bounds.makeEmpty();
      let count = 0;
      const p = positions.array,
        c = colors.array;
      for (let i = 0; i < model.landmarks.length; i++) {
        const point = model.landmarks[i],
          x = point.xyz[0],
          y = point.xyz[1],
          z = point.xyz[2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        const at = count * 3;
        p[at] = x;
        p[at + 1] = y;
        p[at + 2] = z;
        const change = pointChanges.changes[i];
        const before = change === Change.Moved ? priorPoints.get(point.trackId) : undefined;
        pf[at] = before ? before[0] : x;
        pf[at + 1] = before ? before[1] : y;
        pf[at + 2] = before ? before[2] : z;
        pb[count] = live && change === Change.Added ? now : OLD;
        if (live && !reducedMotion && change === Change.Added) flights.push(count, i);
        const color = point.color ?? [0.7, 0.7, 0.7];
        c[at] = SRGB_TO_LINEAR[Math.round(Math.min(1, Math.max(0, color[0])) * 255)];
        c[at + 1] = SRGB_TO_LINEAR[Math.round(Math.min(1, Math.max(0, color[1])) * 255)];
        c[at + 2] = SRGB_TO_LINEAR[Math.round(Math.min(1, Math.max(0, color[2])) * 255)];
        bounds.min.x = Math.min(bounds.min.x, x);
        bounds.min.y = Math.min(bounds.min.y, y);
        bounds.min.z = Math.min(bounds.min.z, z);
        bounds.max.x = Math.max(bounds.max.x, x);
        bounds.max.y = Math.max(bounds.max.y, y);
        bounds.max.z = Math.max(bounds.max.z, z);
        count++;
      }
      positions.commit(count);
      colors.commit(count);
      cloudFrom.commit(count);
      cloudBorn.commit(count);
      cloudGeometry.setDrawRange(0, count);
      // The framing box is robust: the 2nd to 98th percentile on each axis, so a few far points
      // do not shrink the map to a dot.
      if (count >= 50) {
        const stride = Math.max(1, Math.floor(count / 20000));
        const axis: number[] = [];
        for (let a = 0; a < 3; a++) {
          axis.length = 0;
          for (let i = 0; i < count; i += stride) axis.push(p[3 * i + a]!);
          bounds.min.setComponent(a, quantile(axis, 0.02));
          bounds.max.setComponent(a, quantile(axis, 0.98));
        }
      }
      const size = bounds.isEmpty()
        ? new THREE.Vector3(5, 5, 5)
        : bounds.getSize(new THREE.Vector3());
      extent = Math.max(size.x, size.y, size.z, 0.1);
      cloudGeometry.boundingSphere = new THREE.Sphere(
        bounds.isEmpty() ? new THREE.Vector3() : bounds.getCenter(new THREE.Vector3()),
        extent,
      );
      // Frusta (eight segments per camera) and the trajectory between consecutive frames.
      const ordered =
        model.views.length > 1
          ? [...model.views].sort(
              (a, b) =>
                (a.videoId ?? '').localeCompare(b.videoId ?? '') || viewOrder(a) - viewOrder(b),
            )
          : model.views;
      frusta.ensure(ordered.length * 16);
      path.ensure(ordered.length * 2);
      frustumColors.ensure(ordered.length * 16);
      pathColors.ensure(ordered.length * 2);
      for (const attribute of [frustumFrom, frustumBorn, frustumLatest])
        attribute.ensure(ordered.length * 16);
      for (const attribute of [pathFrom, pathBorn, pathLatest])
        attribute.ensure(ordered.length * 2);
      const f = frusta.array,
        q = path.array,
        fc = frustumColors.array,
        qc = pathColors.array,
        ff = frustumFrom.array,
        fb = frustumBorn.array,
        fl = frustumLatest.array,
        qf = pathFrom.array,
        qb = pathBorn.array,
        ql = pathLatest.array;
      const priorFrusta = previousFrusta,
        priorCenters = previousCenters;
      previousFrusta = new Map();
      previousCenters = new Map();
      const moveLimit = extent * 1e-4;
      const colorOf = (view: SceneView) => {
        const css =
          view.videoId === undefined ? undefined : latest.current.videoColors?.get(view.videoId);
        return css ? lineColor.set(css) : defaultLine;
      };
      let frustumVertices = 0,
        pathVertices = 0;
      // A frustum's depth follows the spacing of consecutive cameras (a third of the median
      // step), bounded by a twentieth of the map's extent, so cameras close to the scene stay
      // small against it.
      const steps: number[] = [];
      for (let index = 1; index < ordered.length; index++) {
        cameraFrame(ordered[index - 1], previousCenter, axes);
        cameraFrame(ordered[index], center, axes);
        if (consecutive(ordered[index - 1], ordered[index]))
          steps.push(center.distanceTo(previousCenter));
      }
      steps.sort((a, b) => a - b);
      const medianStep = steps.length ? steps[Math.floor(steps.length / 2)] : 0;
      frustumDepth = medianStep > 0 ? Math.min(extent * 0.05, medianStep * 0.35) : extent * 0.05;
      const depth = frustumDepth;
      for (let index = 0; index < ordered.length; index++) {
        const view = ordered[index];
        cameraFrame(view, center, axes);
        const previous = ordered[index - 1];
        const born = live && newViews.has(view.id) ? now : OLD;
        const isLatest = view.id === latestViewId ? 1 : 0;
        previousCenters.set(view.id, [center.x, center.y, center.z]);
        if (previous && consecutive(previous, view)) {
          q[pathVertices * 3] = previousCenter.x;
          q[pathVertices * 3 + 1] = previousCenter.y;
          q[pathVertices * 3 + 2] = previousCenter.z;
          q[pathVertices * 3 + 3] = center.x;
          q[pathVertices * 3 + 4] = center.y;
          q[pathVertices * 3 + 5] = center.z;
          const ends = [priorCenters.get(previous.id), priorCenters.get(view.id)];
          for (let v = 0; v < 2; v++) {
            const at = (pathVertices + v) * 3;
            const before = live ? ends[v] : undefined;
            for (let k = 0; k < 3; k++) qf[at + k] = before ? before[k]! : q[at + k]!;
            qb[pathVertices + v] = born;
            ql[pathVertices + v] = isLatest;
          }
          const pc = colorOf(view);
          for (let v = 0; v < 2; v++) {
            qc[(pathVertices + v) * 3] = pc.r;
            qc[(pathVertices + v) * 3 + 1] = pc.g;
            qc[(pathVertices + v) * 3 + 2] = pc.b;
          }
          pathVertices += 2;
        }
        previousCenter.copy(center);
        const nx = (depth * (view.camera.cx + 0.5)) / view.camera.fx;
        const ny = (depth * (view.camera.cy + 0.5)) / view.camera.fy;
        const cornersAt = frustumVertices * 3;
        const corners: number[] = [];
        for (const [sx, sy] of [
          [-1, -1],
          [1, -1],
          [1, 1],
          [-1, 1],
        ]) {
          corner
            .copy(center)
            .addScaledVector(axes[0], sx * nx)
            .addScaledVector(axes[1], sy * ny)
            .addScaledVector(axes[2], depth);
          corners.push(corner.x, corner.y, corner.z);
        }
        for (let i = 0; i < 4; i++) {
          const a = i * 3,
            b = ((i + 1) % 4) * 3,
            at = cornersAt + i * 12;
          f[at] = center.x;
          f[at + 1] = center.y;
          f[at + 2] = center.z;
          f[at + 3] = corners[a];
          f[at + 4] = corners[a + 1];
          f[at + 5] = corners[a + 2];
          f[at + 6] = corners[a];
          f[at + 7] = corners[a + 1];
          f[at + 8] = corners[a + 2];
          f[at + 9] = corners[b];
          f[at + 10] = corners[b + 1];
          f[at + 11] = corners[b + 2];
        }
        const vc = colorOf(view);
        for (let v = 0; v < 16; v++) {
          fc[(frustumVertices + v) * 3] = vc.r;
          fc[(frustumVertices + v) * 3 + 1] = vc.g;
          fc[(frustumVertices + v) * 3 + 2] = vc.b;
        }
        // The live view: the frustum slides from its previous place, and a new one glows.
        const placed = f.slice(cornersAt, cornersAt + 48);
        previousFrusta.set(view.id, placed);
        const before = live ? priorFrusta.get(view.id) : undefined;
        for (let k = 0; k < 48; k++) {
          ff[cornersAt + k] = before ? before[k]! : placed[k]!;
          if (before && !tween && Math.abs(before[k]! - placed[k]!) > moveLimit)
            tween = !reducedMotion;
        }
        for (let v = 0; v < 16; v++) {
          fb[frustumVertices + v] = born;
          fl[frustumVertices + v] = isLatest;
        }
        frustumVertices += 16;
      }
      frusta.commit(frustumVertices);
      frustumColors.commit(frustumVertices);
      path.commit(pathVertices);
      pathColors.commit(pathVertices);
      for (const attribute of [frustumFrom, frustumBorn, frustumLatest])
        attribute.commit(frustumVertices);
      for (const attribute of [pathFrom, pathBorn, pathLatest]) attribute.commit(pathVertices);
      // The flights (3.6): a new point starts on the image plane of the newest view that observes
      // it, at its observed pixel, so it travels along that pixel's viewing ray.
      if (flights.length) {
        const newestFirst = [...registration].reverse();
        const planes = new Map<number, ReturnType<typeof planeOf>>();
        const byId = new Map(model.views.map((view) => [view.id, view]));
        let flown = 0;
        for (let k = 0; k < flights.length; k += 2) {
          const drawn = flights[k]!,
            origin = flightOrigin(model.landmarks[flights[k + 1]!]!, newestFirst);
          const view = origin && byId.get(origin.viewId);
          if (!origin || !view) continue;
          let place = planes.get(view.id);
          if (!place) planes.set(view.id, (place = planeOf(view)));
          place(origin.pixel[0], origin.pixel[1], corner);
          pf[drawn * 3] = corner.x;
          pf[drawn * 3 + 1] = corner.y;
          pf[drawn * 3 + 2] = corner.z;
          flown++;
        }
        cloudFrom.commit(count);
        if (flown) tween = true;
        parent.dataset.flights = String(flown);
      } else parent.dataset.flights = '0';
      // A move starts the tween now; without one the old tween has long ended.
      if (tween) liveUniforms.uTweenStart.value = now;
      else if (!live) liveUniforms.uTweenStart.value = OLD;
      if (live && (tween || pointChanges.added || newViews.size || pointChanges.removed.length))
        animateUntil = now + Math.max(TWEEN_S, GLOW_S, GHOST_S);
      parent.dataset.added = String(live ? pointChanges.added : 0);
      parent.dataset.moved = String(live ? pointChanges.moved : 0);
      parent.dataset.removed = String(live ? pointChanges.removed.length / 3 : 0);
      parent.dataset.newViews = String(live ? newViews.size : 0);
      frustumGeometry.setDrawRange(0, frustumVertices);
      pathGeometry.setDrawRange(0, pathVertices);
      parent.dataset.points = String(model.landmarks.length);
      parent.dataset.views = String(model.views.length);
      // Fit once, so later map updates preserve the user's orbit and zoom.
      if (!fitted && count) {
        fitted = true;
        fit();
      }
      gridSource = { positions: p, count, views: model.views };
      refitGrid();
    };
    /** The orbit camera along one axis of the scene, at twice the extent from the map's middle. */
    const preset = (kind: 'top' | 'front' | 'side') => {
      cameraTouched = true;
      const { middle, extent } = framing();
      camera.up.set(0, 1, 0);
      camera.fov = DEFAULT_FOV;
      camera.updateProjectionMatrix();
      const d = extent * 2;
      const offset =
        kind === 'top'
          ? new THREE.Vector3(0, d, d * 1e-3)
          : kind === 'front'
            ? new THREE.Vector3(0, 0, d)
            : new THREE.Vector3(d, 0, 0);
      camera.position.copy(middle).add(offset);
      controls.target.copy(middle);
      controls.update();
    };
    // ---- Placement on the grid -------------------------------------------------------------
    const UP = new THREE.Vector3(0, 1, 0);
    let picking = -1;
    const picked: THREE.Vector3[] = [];
    const pickedGeometry = own(new THREE.BufferGeometry());
    const pickedMarks = new THREE.Points(
      pickedGeometry,
      own(
        new THREE.PointsMaterial({
          color: '#ffd23f',
          size: 10,
          sizeAttenuation: false,
          depthTest: false,
        }),
      ),
    );
    pickedMarks.renderOrder = 10;
    pickedMarks.frustumCulled = false;
    scene.add(pickedMarks);
    const showPicked = () => {
      pickedGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(
          picked.flatMap((v) => [v.x, v.y, v.z]),
          3,
        ),
      );
      pickedMarks.visible = picked.length > 0;
    };
    /** The whole placement (`placement` then `pivot`): a move, a turn and a scale about the scene's
     *  origin, so the numbers do not depend on where the turn's center sits. */
    const totalPlacement = () => {
      placement.updateMatrix();
      pivot.updateMatrix();
      const total = new THREE.Matrix4().multiplyMatrices(placement.matrix, pivot.matrix);
      const position = new THREE.Vector3(),
        quaternion = new THREE.Quaternion(),
        scale = new THREE.Vector3();
      total.decompose(position, quaternion, scale);
      return { position, quaternion, scale: scale.x };
    };
    const placementState = (message?: string): PlacementState => {
      const total = totalPlacement();
      const euler = new THREE.Euler().setFromQuaternion(total.quaternion, 'XYZ');
      // Rounded, with -0 shown as 0.
      const round = (v: number, digits: number) => Number(v.toFixed(digits)) + 0;
      return {
        position: [
          round(total.position.x, 4),
          round(total.position.y, 4),
          round(total.position.z, 4),
        ],
        rotationDeg: [
          round(THREE.MathUtils.radToDeg(euler.x), 2),
          round(THREE.MathUtils.radToDeg(euler.y), 2),
          round(THREE.MathUtils.radToDeg(euler.z), 2),
        ],
        scale: round(total.scale, 4),
        picking,
        ...(message ? { message } : {}),
      };
    };
    const announce = (message?: string) => {
      const state = placementState(message);
      parent.dataset.placement = JSON.stringify([state.position, state.rotationDeg, state.scale]);
      parent.dataset.picking = String(picking);
      latest.current.onPlacement?.(state);
    };
    /** Keep the map where it is and move the turn's center to `center` (scene frame). */
    const setPivot = (center: THREE.Vector3) => {
      placement.updateMatrixWorld(true);
      const total = new THREE.Matrix4().multiplyMatrices(placement.matrix, pivot.matrix);
      placement.position.copy(center);
      placement.updateMatrix();
      const rest = new THREE.Matrix4().copy(placement.matrix).invert().multiply(total);
      pivot.position.setFromMatrixPosition(rest);
      placement.updateMatrixWorld(true);
    };
    /** The map's middle in the scene frame. */
    /** The middle and the size of what is drawn (mesh, dense or sparse) in the scene frame: the
     *  2nd to 98th percentile box, so the view frames the cloud on screen, not stray points. */
    const framing = (): { middle: THREE.Vector3; extent: number } => {
      root.updateMatrixWorld(true);
      const { array, count } = drawnPoints();
      const box = new THREE.Box3();
      if (count >= 50) {
        const stride = Math.max(1, Math.floor(count / 20000));
        const axis: number[] = [];
        for (let a = 0; a < 3; a++) {
          axis.length = 0;
          for (let i = 0; i < count; i += stride) axis.push(array[3 * i + a]!);
          box.min.setComponent(a, quantile(axis, 0.02));
          box.max.setComponent(a, quantile(axis, 0.98));
        }
      } else box.copy(bounds);
      if (box.isEmpty()) return { middle: root.localToWorld(new THREE.Vector3()), extent };
      const size = box.getSize(new THREE.Vector3());
      return {
        middle: root.localToWorld(box.getCenter(new THREE.Vector3())),
        extent: Math.max(size.x, size.y, size.z, 1e-3) * placement.scale.x,
      };
    };
    const mapMiddle = () => framing().middle;
    /** Apply `change` (a scene-frame transform) on top of the placement. */
    const applyChange = (change: THREE.Matrix4) => {
      placement.updateMatrix();
      const next = new THREE.Matrix4().multiplyMatrices(change, placement.matrix);
      next.decompose(placement.position, placement.quaternion, placement.scale);
      const s = placement.scale.x;
      placement.scale.setScalar(s);
      placement.updateMatrixWorld(true);
    };
    /** Turn `normal` (scene frame) to up about `point`, then lift `point` to the grid. */
    const level = (normal: THREE.Vector3, point: THREE.Vector3) => {
      const turn = new THREE.Matrix4().makeRotationFromQuaternion(
        new THREE.Quaternion().setFromUnitVectors(normal.clone().normalize(), UP),
      );
      const change = new THREE.Matrix4()
        .makeTranslation(point.x, 0, point.z)
        .multiply(turn)
        .multiply(new THREE.Matrix4().makeTranslation(-point.x, -point.y, -point.z));
      applyChange(change);
    };
    /** The drawn points of the map (mesh, dense or sparse) and their count, in the map's frame. */
    const drawnPoints = (): { array: ArrayLike<number>; count: number } => {
      if (meshObject.visible && shownMeshData)
        return { array: shownMeshData.positions, count: shownMeshData.positions.length / 3 };
      if (denseCloud.visible && dense.drawn) return { array: dense.positions, count: dense.drawn };
      const attribute = cloudGeometry.getAttribute('position');
      return {
        array: (attribute?.array as ArrayLike<number>) ?? [],
        count: Math.min(attribute?.count ?? 0, cloudGeometry.drawRange.count),
      };
    };
    /** The camera centers in the scene frame (x, y, z per view). */
    const sceneCenters = (): number[] => {
      root.updateMatrixWorld(true);
      const out: number[] = [];
      const at = new THREE.Vector3();
      for (const view of shown?.views ?? []) {
        at.fromArray(cameraPosition(view)).applyMatrix4(root.matrixWorld);
        out.push(at.x, at.y, at.z);
      }
      return out;
    };
    /** Put the lowest points (the 2nd percentile of height) on the grid, the middle above 0. */
    const groundMap = () => {
      const { array, count } = drawnPoints();
      if (!count) return;
      root.updateMatrixWorld(true);
      const stride = Math.max(1, Math.floor(count / 20000));
      const heights: number[] = [];
      const at = new THREE.Vector3();
      for (let i = 0; i < count; i += stride) {
        at.set(array[3 * i]!, array[3 * i + 1]!, array[3 * i + 2]!).applyMatrix4(root.matrixWorld);
        heights.push(at.y);
      }
      const middle = mapMiddle();
      applyChange(
        new THREE.Matrix4().makeTranslation(-middle.x, -quantile(heights, 0.02), -middle.z),
      );
    };
    const finishPick = () => {
      const plane = planeFromPoints(
        picked[0]!.toArray(),
        picked[1]!.toArray(),
        picked[2]!.toArray(),
      );
      picking = -1;
      picked.length = 0;
      showPicked();
      if (!plane) return announce('line');
      const oriented = orientTowards(plane, sceneCenters());
      level(new THREE.Vector3(...oriented.normal), new THREE.Vector3(...oriented.point));
      setPivot(mapMiddle());
      refitGrid();
      announce();
    };
    const runPlacement = (command: NonNullable<Props['placementCommand']>) => {
      let message: string | undefined;
      if (command.kind !== 'pick') {
        picking = -1;
        picked.length = 0;
        showPicked();
      }
      if (command.kind === 'reset') {
        placement.position.set(0, 0, 0);
        placement.quaternion.identity();
        placement.scale.setScalar(1);
        pivot.position.set(0, 0, 0);
        placement.updateMatrixWorld(true);
      } else if (command.kind === 'ground') groundMap();
      else if (command.kind === 'auto') {
        const { array, count } = drawnPoints();
        root.updateMatrixWorld(true);
        const world = new Float64Array(3 * count);
        const at = new THREE.Vector3();
        for (let i = 0; i < count; i++) {
          at.set(array[3 * i]!, array[3 * i + 1]!, array[3 * i + 2]!).applyMatrix4(
            root.matrixWorld,
          );
          world[3 * i] = at.x;
          world[3 * i + 1] = at.y;
          world[3 * i + 2] = at.z;
        }
        const found = dominantPlane(world, count);
        if (!found) message = 'few';
        else {
          const oriented = orientTowards(found, sceneCenters());
          level(new THREE.Vector3(...oriented.normal), new THREE.Vector3(...oriented.point));
          groundMap();
        }
      } else if (command.kind === 'pick') {
        picking = 0;
        picked.length = 0;
        showPicked();
      } else if (command.kind === 'set' && command.value) {
        // The values are the whole placement; the turn's center (`pivot`) stays where it is.
        const { position, rotationDeg, scale } = command.value;
        const total = new THREE.Matrix4().compose(
          new THREE.Vector3().fromArray(position),
          new THREE.Quaternion().setFromEuler(
            new THREE.Euler(
              ...(rotationDeg.map((d) => THREE.MathUtils.degToRad(d)) as [number, number, number]),
              'XYZ',
            ),
          ),
          new THREE.Vector3().setScalar(scale > 0 ? scale : 1),
        );
        pivot.updateMatrix();
        total
          .multiply(new THREE.Matrix4().copy(pivot.matrix).invert())
          .decompose(placement.position, placement.quaternion, placement.scale);
        placement.updateMatrixWorld(true);
      }
      if (command.kind !== 'pick' && command.kind !== 'set') setPivot(mapMiddle());
      refitGrid();
      announce(message);
    };
    // The handles: TransformControls on the placement group. A drag holds the orbit; the scale
    // handles keep the scale uniform (the axis that moved most sets it).
    const handles = new TransformControls(camera, renderer.domElement);
    const handlesHelper = handles.getHelper();
    scene.add(handlesHelper);
    releases.push(() => {
      handles.detach();
      handles.dispose();
      scene.remove(handlesHelper);
    });
    handles.enabled = false;
    handlesHelper.visible = false;
    let handlesMode: NonNullable<Props['transformMode']> = 'off';
    handles.addEventListener('dragging-changed', (event) => {
      controls.enabled = !event.value && !fly;
      if (!event.value) {
        refitGrid();
        announce();
      }
    });
    let lastScale = 1;
    handles.addEventListener('objectChange', () => {
      if (handlesMode === 'scale') {
        const { x, y, z } = placement.scale;
        const before = lastScale;
        const moved = [x, y, z].reduce((a, b) =>
          Math.abs(b - before) > Math.abs(a - before) ? b : a,
        );
        placement.scale.setScalar(Math.max(1e-6, moved));
        lastScale = placement.scale.x;
      }
      announce();
    });
    handles.addEventListener('change', () => requestRender());
    const setHandles = (mode: NonNullable<Props['transformMode']>) => {
      if (mode === handlesMode) return;
      handlesMode = mode;
      if (mode === 'off') {
        handles.detach();
        handles.enabled = false;
        handlesHelper.visible = false;
      } else {
        setPivot(mapMiddle());
        lastScale = placement.scale.x;
        handles.attach(placement);
        handles.setMode(mode);
        handles.enabled = true;
        handlesHelper.visible = true;
      }
      parent.dataset.transform = mode;
    };
    /** The orbit camera above and behind the map, looking at its middle, with the default lens. */
    const fit = () => {
      // The middle and size of what is drawn; the orbit camera lives in the scene.
      const { middle, extent: size } = framing();
      camera.up.set(0, 1, 0);
      camera.fov = DEFAULT_FOV;
      camera.updateProjectionMatrix();
      controls.minDistance = Math.min(extent, size) * 0.002;
      controls.maxDistance = Math.max(extent, size) * 50;
      camera.position.copy(middle).add(new THREE.Vector3(size * 0.6, size * 0.55, size * 1.2));
      controls.target.copy(middle);
      controls.update();
    };
    /**
     * Look through a registered camera: the orbit camera at the view's center, looking along its
     * forward axis with its up vector (image rows grow downwards), with the view's vertical field
     * of view. The orbit target sits at the median depth of the landmarks this view observes (half
     * an extent ahead without any), so orbiting and zooming afterwards turn about what it saw.
     */
    const lookThrough = (view: SceneView, model: SceneData) => {
      cameraTouched = true;
      const { R, t } = view.pose;
      const depths: number[] = [];
      for (const point of model.landmarks) {
        if (!point.observations.some((o) => o.viewId === view.id)) continue;
        const X = point.xyz;
        const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
        if (z > 0 && Number.isFinite(z)) depths.push(z);
      }
      depths.sort((a, b) => a - b);
      const ahead = depths.length ? depths[depths.length >> 1] : extent * 0.5;
      cameraFrame(view, center, axes);
      const height = (view.height ?? Math.round(view.camera.cy * 2 + 1)) || 1;
      camera.fov = (2 * Math.atan(height / 2 / view.camera.fy) * 180) / Math.PI;
      camera.updateProjectionMatrix();
      camera.up
        .copy(root.localToWorld(axes[1].clone().negate().add(center)))
        .sub(root.localToWorld(center.clone()));
      camera.position.copy(root.localToWorld(center.clone()));
      controls.target.copy(root.localToWorld(center.clone().addScaledVector(axes[2], ahead)));
      controls.update();
    };
    const updateSelected = (state: Props) => {
      const model = state.reconstruction ?? EMPTY_SCENE;
      const view = model.views.find((v) => v.id === state.selectedViewId);
      parent.dataset.selectedView = String(view?.id ?? '');
      forwardArrow.visible = motionArrow.visible = false;
      if (view) {
        cameraFrame(view, center, axes);
        forwardArrow.position.copy(center);
        forwardArrow.setDirection(axes[2]);
        forwardArrow.setLength(extent * 0.12);
        forwardArrow.visible = true;
        // Consecutive input frames only: never draw motion across a failed pose.
        const previous = model.views.find((v) =>
          view.sequence !== undefined ? v.sequence === view.sequence - 1 : v.id === view.id - 1,
        );
        if (previous) {
          cameraFrame(previous, previousCenter, axes);
          const delta = center.clone().sub(previousCenter);
          if (delta.length() > 1e-9) {
            motionArrow.position.copy(previousCenter);
            motionArrow.setDirection(delta.clone().normalize());
            motionArrow.setLength(delta.length());
            motionArrow.visible = true;
          }
        }
        showLinks(view, model, state);
        const photo = state.photo?.id === view.id ? state.photo : view;
        if (photo.image) {
          const width = photo.width ?? Math.round(view.camera.cx * 2 + 1);
          const height = photo.height ?? Math.round(view.camera.cy * 2 + 1);
          showPhoto(view, photo.image, width, height);
        } else {
          photoGeneration++;
          photoPlane.visible = false;
          delete parent.dataset.photo;
        }
      } else {
        photoGeneration++;
        photoPlane.visible = false;
        delete parent.dataset.photo;
        hideLinks();
      }
    };

    // Render on demand: a change of data, selection, size or orbit schedules one frame, and the
    // frame schedules another only while the orbit damping is still moving.
    let shown: SceneData | undefined;
    let shownDense: DenseCloud | undefined;
    let shownBudget = NaN;
    let selectedProps: Props | undefined;
    let appliedCameraView: Props['cameraView'] | undefined;
    let appliedFit: number | undefined;
    let appliedPreset: Props['viewPreset'];
    let appliedPlacement: Props['placementCommand'];
    let animation = 0;
    let pending = false;
    releases.push(() => cancelAnimationFrame(animation));
    const frame = () => {
      animation = 0;
      pending = false;
      if (disposed) return;
      try {
        const model = latest.current.reconstruction ?? EMPTY_SCENE;
        if (shown !== model) {
          shown = model;
          update(shown);
          selectedProps = undefined;
        }
        // The dense cloud follows the page's chunks, and with points it replaces the sparse cloud.
        const budget = latest.current.densePointBudget ?? DENSE_POINT_BUDGET;
        if (latest.current.dense !== shownDense || budget !== shownBudget) {
          shownDense = latest.current.dense;
          shownBudget = budget;
          denseUpdate(shownDense, budget);
        }
        if (latest.current.mesh !== shownMeshData) {
          shownMeshData = latest.current.mesh;
          meshUpdate(shownMeshData);
        }
        const meshShown = !!shownMeshData?.indices.length && latest.current.showMesh === true;
        meshObject.visible = meshShown;
        const denseShown = dense.drawn > 0 && latest.current.showDense !== false && !meshShown;
        denseCloud.visible = denseShown;
        cloud.visible = !denseShown && !meshShown;
        parent.dataset.meshFaces = String((shownMeshData?.indices.length ?? 0) / 3);
        parent.dataset.meshShown = String(meshShown);
        parent.dataset.densePoints = String(dense.read);
        parent.dataset.denseDrawn = String(dense.drawn);
        parent.dataset.denseStride = String(dense.stride);
        const kind = meshShown ? 'mesh' : denseShown ? 'dense' : 'sparse';
        parent.dataset.cloud = kind;
        // What is drawn changed (sparse to dense to mesh), or it doubled: the grid follows it, and
        // the orbit camera frames it again until the visitor moves the camera.
        const drawnCount =
          kind === 'mesh'
            ? (meshObject.geometry.index?.count ?? 0)
            : kind === 'dense'
              ? dense.drawn
              : cloudGeometry.drawRange.count;
        if (shown && (kind !== framedKind || drawnCount > 2 * framedCount)) {
          framedKind = kind;
          framedCount = drawnCount;
          refitGrid();
          if (!cameraTouched && fitted) fit();
        }
        if (
          !selectedProps ||
          selectedProps.selectedViewId !== latest.current.selectedViewId ||
          selectedProps.photo !== latest.current.photo ||
          selectedProps.showPhoto !== latest.current.showPhoto ||
          selectedProps.features !== latest.current.features ||
          selectedProps.showLinks !== latest.current.showLinks
        ) {
          selectedProps = latest.current;
          updateSelected(selectedProps);
        }
        if (latest.current.fitNonce !== appliedFit) {
          appliedFit = latest.current.fitNonce;
          if (appliedFit !== undefined && shown && (shown.views.length || shown.landmarks.length))
            fit();
        }
        if (latest.current.viewPreset !== appliedPreset) {
          appliedPreset = latest.current.viewPreset;
          if (appliedPreset && shown && (shown.views.length || shown.landmarks.length))
            preset(appliedPreset.kind);
        }
        setHandles(latest.current.transformMode ?? 'off');
        if (latest.current.placementCommand !== appliedPlacement) {
          appliedPlacement = latest.current.placementCommand;
          if (appliedPlacement) runPlacement(appliedPlacement);
        }
        if (latest.current.cameraView !== appliedCameraView) {
          appliedCameraView = latest.current.cameraView;
          const view =
            appliedCameraView && shown?.views.find((v) => v.id === appliedCameraView!.id);
          if (view && shown) lookThrough(view, shown);
        }
        // The navigation mode. Leaving fly mode puts the orbit target ahead of the camera at the
        // distance it had, so orbiting continues about what is in view.
        const flyNow = latest.current.navigation === 'fly';
        if (flyNow !== fly) {
          const distance = Math.max(extent * 0.01, camera.position.distanceTo(controls.target));
          fly = flyNow;
          controls.enabled = !fly;
          if (!fly) {
            camera.getWorldDirection(flyForward);
            controls.target.copy(camera.position).addScaledVector(flyForward, distance);
          }
          parent.dataset.navigation = fly ? 'fly' : 'orbit';
        }
        // The display options: cameras, the photo plane and the background.
        const showCameras = latest.current.showCameras !== false;
        frustumLines.visible = pathLines.visible = showCameras;
        if (!showCameras) forwardArrow.visible = motionArrow.visible = false;
        // The photo plane is shown by `updateSelected` (which runs again when the option turns
        // on) and hidden here while the option is off.
        if (latest.current.showPhoto === false) photoPlane.visible = false;
        parent.dataset.photoVisible = photoPlane.visible ? 'shown' : 'hidden';
        const light = latest.current.lightBackground === true;
        if (light !== lightBackground) {
          lightBackground = light;
          renderer.setClearColor(light ? '#eef1f6' : '#0b1020', 1);
        }
        parent.dataset.background = light ? 'light' : 'dark';
        parent.dataset.cameras = showCameras ? 'shown' : 'hidden';
        const moving = fly ? false : controls.update();
        // The live view's clock, pulse and red marks (`docs/lab-live-view.md`).
        const now = clock();
        liveUniforms.uNow.value = now;
        const activity = latest.current.activity;
        const pulsing =
          latest.current.live === true &&
          !reducedMotion &&
          !!activity &&
          activity.target !== 'none' &&
          performance.now() - activity.at < PULSE_HOLD_MS;
        const wave = pulsing ? 0.5 - 0.5 * Math.cos(now * Math.PI * 2 * 0.8) : 0;
        liveUniforms.uPulseAll.value = pulsing && activity!.target === 'all' ? wave : 0;
        liveUniforms.uPulseLatest.value = pulsing && activity!.target === 'latest' ? wave : 0;
        parent.dataset.pulse = pulsing ? activity!.target : 'none';
        const ghostAge = now - ghostsAt;
        ghostMaterial.opacity = Math.max(0, 1 - ghostAge / GHOST_S);
        if (ghosts.visible && ghostAge >= GHOST_S) ghosts.visible = false;
        renderer.render(scene, camera);
        if (moving || now < animateUntil) requestRender();
        else if (pulsing) requestSlowRender();
      } catch (error) {
        fail(error);
      }
    };
    const requestRender = () => {
      if (disposed || pending) return;
      pending = true;
      animation = requestAnimationFrame(frame);
    };
    // A pulse alone needs no 60 Hz: it redraws every PULSE_FRAME_MS, and the GPU stays the run's.
    let slowTimer = 0;
    const requestSlowRender = () => {
      if (disposed || slowTimer) return;
      slowTimer = window.setTimeout(() => {
        slowTimer = 0;
        requestRender();
      }, PULSE_FRAME_MS);
    };
    releases.push(() => window.clearTimeout(slowTimer));
    refreshRef.current = requestRender;
    releases.push(() => {
      refreshRef.current = () => {};
    });
    controls.addEventListener('change', requestRender);
    const resize = new ResizeObserver(() => {
      if (!parent.clientWidth || !parent.clientHeight) return;
      camera.aspect = parent.clientWidth / parent.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(parent.clientWidth, parent.clientHeight);
      requestRender();
    });
    resize.observe(parent);
    releases.push(() => resize.disconnect());
    const ray = new THREE.Raycaster();
    /** The camera under a pointer position, if any. Points are not pickable. */
    const pick = (e: MouseEvent): { view?: SceneView } => {
      const model = latest.current.reconstruction;
      if (!model) return {};
      const rect = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          1 - ((e.clientY - rect.top) / rect.height) * 2,
        ),
        camera,
      );
      let nearest: SceneView | undefined,
        nearestDistance = Infinity;
      for (const view of model.views) {
        cameraFrame(view, center, axes);
        if (ray.ray.distanceToPoint(center) >= extent * 0.015) continue;
        const distance = ray.ray.origin.distanceTo(center);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = view;
        }
      }
      return nearest ? { view: nearest } : {};
    };
    let down = { x: 0, y: 0 };
    const pointerDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
      if (fly) renderer.domElement.focus();
    };
    const pointerUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      const { view } = pick(e);
      if (view) latest.current.onSelectView(view.id);
    };
    // A double click moves the orbit target to the camera under the cursor, so the
    // wheel and a drag then work about it, at its scale.
    /**
     * The drawn point (sparse or Dense) nearest the viewer along the pointer's ray, within a cone
     * of 1 % of the distance (at least 0.2 % of the extent) around it, in the map's coordinates.
     */
    const localRay = new THREE.Ray(),
      rootInverse = new THREE.Matrix4(),
      probe = new THREE.Vector3(),
      along = new THREE.Vector3();
    const pickPoint = (): THREE.Vector3 | undefined => {
      rootInverse.copy(root.matrixWorld).invert();
      localRay.copy(ray.ray).applyMatrix4(rootInverse);
      let best: THREE.Vector3 | undefined,
        bestDistance = Infinity;
      const scan = (array: ArrayLike<number>, count: number) => {
        for (let i = 0; i < count; i++) {
          probe.set(array[i * 3]!, array[i * 3 + 1]!, array[i * 3 + 2]!);
          const t = along.subVectors(probe, localRay.origin).dot(localRay.direction);
          if (t <= 0 || t >= bestDistance) continue;
          const tolerance = Math.max(t * 0.01, extent * 0.002);
          if (localRay.distanceSqToPoint(probe) > tolerance * tolerance) continue;
          bestDistance = t;
          best = probe.clone();
        }
      };
      if (cloud.visible) {
        const attribute = cloudGeometry.getAttribute('position');
        if (attribute)
          scan(
            attribute.array as ArrayLike<number>,
            Math.min(attribute.count, cloudGeometry.drawRange.count),
          );
      }
      if (denseCloud.visible) scan(dense.positions, dense.drawn);
      if (meshObject.visible && shownMeshData)
        scan(shownMeshData.positions, shownMeshData.positions.length / 3);
      return best;
    };
    const doubleClick = (e: MouseEvent) => {
      const { view } = pick(e);
      if (picking >= 0) {
        // Picking the ground: the point under the cursor is the next of three.
        const point = pickPoint();
        if (!point) return announce('miss');
        picked.push(root.localToWorld(point));
        picking = picked.length;
        showPicked();
        if (picked.length === 3) finishPick();
        else announce();
        requestRender();
        return;
      }
      if (view) {
        cameraFrame(view, center, axes);
        controls.target.copy(root.localToWorld(center.clone()));
      } else {
        // `pick` set the ray from the pointer: the pivot moves to the point under the cursor.
        const point = pickPoint();
        if (!point) return;
        controls.target.copy(root.localToWorld(point));
      }
      controls.update();
      requestRender();
    };
    // Fly mode: a drag turns the camera in place; W A S D and Q E move it along its own axes at a
    // speed of three tenths of the extent per second, four times that with Shift, a quarter with
    // Alt; the wheel scales the speed. The canvas takes keyboard focus on a pointer down.
    let flyFrame = 0;
    let flyLast = 0;
    const flyStep = (now: number) => {
      flyFrame = 0;
      if (disposed || !fly || !flyKeys.size) return;
      const dt = Math.min(0.1, (now - flyLast) / 1000);
      flyLast = now;
      const fast = flyKeys.has('ShiftLeft') || flyKeys.has('ShiftRight');
      const slow = flyKeys.has('AltLeft') || flyKeys.has('AltRight');
      const step = extent * 0.3 * flySpeed * (fast ? 4 : slow ? 0.25 : 1) * dt;
      camera.getWorldDirection(flyForward);
      flyRight.crossVectors(flyForward, camera.up).normalize();
      const move = (key: string, axis: THREE.Vector3, sign: number) => {
        if (flyKeys.has(key)) camera.position.addScaledVector(axis, sign * step);
      };
      move('KeyW', flyForward, 1);
      move('ArrowUp', flyForward, 1);
      move('KeyS', flyForward, -1);
      move('ArrowDown', flyForward, -1);
      move('KeyD', flyRight, 1);
      move('ArrowRight', flyRight, 1);
      move('KeyA', flyRight, -1);
      move('ArrowLeft', flyRight, -1);
      move('KeyE', camera.up, 1);
      move('KeyQ', camera.up, -1);
      requestRender();
      flyFrame = requestAnimationFrame(flyStep);
    };
    const FLY_KEYS = new Set([
      'KeyW',
      'KeyA',
      'KeyS',
      'KeyD',
      'KeyQ',
      'KeyE',
      'ArrowUp',
      'ArrowDown',
      'ArrowLeft',
      'ArrowRight',
      'ShiftLeft',
      'ShiftRight',
      'AltLeft',
      'AltRight',
    ]);
    const keyDown = (e: KeyboardEvent) => {
      if (!fly || !FLY_KEYS.has(e.code)) return;
      e.preventDefault();
      flyKeys.add(e.code);
      if (!flyFrame) {
        flyLast = performance.now();
        flyFrame = requestAnimationFrame(flyStep);
      }
    };
    const keyUp = (e: KeyboardEvent) => {
      flyKeys.delete(e.code);
    };
    const flyLook = (e: PointerEvent) => {
      if (!fly || !(e.buttons & 1)) return;
      // Yaw about the up axis, pitch about the camera's right axis, 0.3 degrees per pixel.
      camera.rotateOnWorldAxis(camera.up, -e.movementX * 0.005);
      camera.rotateX(-e.movementY * 0.005);
      requestRender();
    };
    const flyWheel = (e: WheelEvent) => {
      if (!fly) return;
      e.preventDefault();
      flySpeed = Math.min(64, Math.max(1 / 64, flySpeed * (e.deltaY < 0 ? 1.25 : 0.8)));
    };
    renderer.domElement.addEventListener('pointerdown', pointerDown);
    renderer.domElement.addEventListener('pointerup', pointerUp);
    renderer.domElement.addEventListener('dblclick', doubleClick);
    renderer.domElement.addEventListener('pointermove', flyLook);
    renderer.domElement.addEventListener('wheel', flyWheel, { passive: false });
    renderer.domElement.addEventListener('keydown', keyDown);
    renderer.domElement.addEventListener('keyup', keyUp);
    releases.push(() => {
      cancelAnimationFrame(flyFrame);
      renderer.domElement.removeEventListener('pointerdown', pointerDown);
      renderer.domElement.removeEventListener('pointerup', pointerUp);
      renderer.domElement.removeEventListener('dblclick', doubleClick);
      renderer.domElement.removeEventListener('pointermove', flyLook);
      renderer.domElement.removeEventListener('wheel', flyWheel);
      renderer.domElement.removeEventListener('keydown', keyDown);
      renderer.domElement.removeEventListener('keyup', keyUp);
    });
    // Every program compiles now, in parallel where the browser allows it
    // (KHR_parallel_shader_compile), and a warm-up frame with every object visible then uses each
    // program once: three.js waits for a program's link at its first use, so without this the
    // first points, lines and photo of a run blocked the page's main thread (3.1 s on SwiftShader,
    // measured 2026-10-20). The first real frame then runs, so a renderer that cannot draw fails
    // before the viewer is ready.
    const hidden = () => [forwardArrow, motionArrow, photoPlane, grid, ghosts, ...linkObjects];
    renderer.compileAsync(scene, camera).then(
      () => {
        if (disposed) return;
        try {
          for (const object of hidden()) object.visible = true;
          renderer.render(scene, camera);
        } catch (error) {
          fail(error);
          return;
        } finally {
          for (const object of hidden()) object.visible = gridFit ? object === grid : false;
        }
        frame();
        if (!disposed) report({ status: 'ready' });
      },
      (error: unknown) => fail(error),
    );
    return () => {
      dispose();
      report({ status: 'initializing' });
    };
  }, [report]);
  useEffect(() => {
    try {
      return initialize();
    } catch (error) {
      disposeRef.current();
      report({ status: 'error', message: '3D viewer failed: ' + String(error) });
      return () => report({ status: 'initializing' });
    }
  }, [initialize, report]);
  // New data or selection: one frame, not a loop.
  useEffect(() => {
    refreshRef.current();
  }, [
    props.reconstruction,
    props.selectedViewId,
    props.photo,
    props.cameraView,
    props.fitNonce,
    props.viewPreset,
    props.showCameras,
    props.showPhoto,
    props.lightBackground,
    props.navigation,
    props.dense,
    props.showDense,
    props.densePointBudget,
    props.mesh,
    props.showMesh,
    props.videoColors,
    props.transformMode,
    props.placementCommand,
    props.live,
    props.activity,
    props.features,
    props.showLinks,
  ]);
  return (
    <div
      className="viewer"
      data-backend="webgl"
      ref={host}
      role="img"
      aria-label="Growing sparse 3D reconstruction with camera trajectory"
    >
      <span className="viewer-error" />
    </div>
  );
}
