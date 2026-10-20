import { useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  cameraPosition,
  fitGrid,
  gridChanged,
  type GridFit,
  type Scene as SceneData,
  type SceneView,
} from '../lib/sfm-scene';

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
  onSelectView: (id: number) => void;
  onState: (state: ViewerState) => void;
};
const EMPTY_SCENE: SceneData = { views: [], landmarks: [] };
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
  view.sequence !== undefined
    ? previous.sequence! + 1 === view.sequence
    : previous.id + 1 === view.id;

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
    root.updateMatrixWorld(true);
    scene.add(root);
    const DEFAULT_FOV = 48;
    const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.001, 100000);
    const controls = own(new OrbitControls(camera, renderer.domElement));
    controls.enableDamping = true;
    // The wheel zooms toward the point under the cursor, so a visitor dives into one part of a
    // small-scale map instead of toward a far target; the distance bounds follow the extent.
    controls.zoomToCursor = true;
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
    const cloud = new THREE.Points(
      cloudGeometry,
      own(new THREE.PointsMaterial({ vertexColors: true, size: 3, sizeAttenuation: false })),
    );
    cloud.frustumCulled = false;
    root.add(cloud);
    const frustumGeometry = own(new THREE.BufferGeometry());
    const frusta = new GrowingAttribute(frustumGeometry, 'position', 3, 16 * 8 * 2);
    const lineMaterial = own(new THREE.LineBasicMaterial({ color: '#ffb454' }));
    const frustumLines = new THREE.LineSegments(frustumGeometry, lineMaterial);
    frustumLines.frustumCulled = false;
    root.add(frustumLines);
    const pathGeometry = own(new THREE.BufferGeometry());
    const path = new GrowingAttribute(pathGeometry, 'position', 3, 16 * 2);
    const pathLines = new THREE.LineSegments(pathGeometry, lineMaterial);
    pathLines.frustumCulled = false;
    root.add(pathLines);
    cloudGeometry.setDrawRange(0, 0);
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
    // The ground grid is hidden until the first points arrive, so its program compiles with the
    // others before the viewer is ready. It is rebuilt (and the old one disposed) when the fitted
    // extent of the points and cameras moves by more than 20 %.
    const makeGrid = (size: number, divisions: number) => {
      const made = new THREE.GridHelper(size, divisions, '#35415d', '#202b43');
      made.visible = false;
      root.add(made);
      return made;
    };
    const removeGrid = (old: THREE.GridHelper) => {
      root.remove(old);
      old.geometry.dispose();
      (old.material as THREE.Material).dispose();
    };
    let grid = makeGrid(1, 10);
    let gridFit: GridFit | undefined;
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
        extent = 1;
        grid.visible = false;
        gridFit = undefined;
        camera.position.set(0, 0, 5);
        controls.target.set(0, 0, 0);
        controls.update();
      }
      // Points and colors in one pass; bounds without per-point allocations.
      positions.ensure(model.landmarks.length);
      colors.ensure(model.landmarks.length);
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
      cloudGeometry.setDrawRange(0, count);
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
          ? [...model.views].sort((a, b) => viewOrder(a) - viewOrder(b))
          : model.views;
      frusta.ensure(ordered.length * 16);
      path.ensure(ordered.length * 2);
      const f = frusta.array,
        q = path.array;
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
        if (previous && consecutive(previous, view)) {
          q[pathVertices * 3] = previousCenter.x;
          q[pathVertices * 3 + 1] = previousCenter.y;
          q[pathVertices * 3 + 2] = previousCenter.z;
          q[pathVertices * 3 + 3] = center.x;
          q[pathVertices * 3 + 4] = center.y;
          q[pathVertices * 3 + 5] = center.z;
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
        frustumVertices += 16;
      }
      frusta.commit(frustumVertices);
      path.commit(pathVertices);
      frustumGeometry.setDrawRange(0, frustumVertices);
      pathGeometry.setDrawRange(0, pathVertices);
      parent.dataset.points = String(model.landmarks.length);
      parent.dataset.views = String(model.views.length);
      // Fit once, so later map updates preserve the user's orbit and zoom.
      if (!fitted && count) {
        fitted = true;
        fit();
      }
      // The grid follows the robust extent of the landmarks and the camera centers in its plane.
      const ground = new Float64Array(2 * (count + model.views.length));
      for (let i = 0; i < count; i++) {
        ground[2 * i] = p[3 * i];
        ground[2 * i + 1] = p[3 * i + 2];
      }
      model.views.forEach((view, i) => {
        const at = cameraPosition(view);
        ground[2 * (count + i)] = at[0];
        ground[2 * (count + i) + 1] = at[2];
      });
      const next = fitGrid(ground);
      if (next && gridChanged(gridFit, next)) {
        const old = grid;
        grid = makeGrid(next.size, next.divisions);
        grid.position.set(next.x, 0, next.z);
        grid.visible = true;
        gridFit = next;
        removeGrid(old);
      }
    };
    /** The orbit camera above and behind the map, looking at its middle, with the default lens. */
    const fit = () => {
      // The bounds are the mapper's coordinates; the orbit camera lives in the scene.
      const middle = root.localToWorld(bounds.getCenter(new THREE.Vector3()));
      camera.up.set(0, 1, 0);
      camera.fov = DEFAULT_FOV;
      camera.updateProjectionMatrix();
      controls.minDistance = extent * 0.002;
      controls.maxDistance = extent * 50;
      camera.position.copy(middle).add(new THREE.Vector3(extent * 0.9, extent * 0.7, extent * 1.8));
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
      }
    };

    // Render on demand: a change of data, selection, size or orbit schedules one frame, and the
    // frame schedules another only while the orbit damping is still moving.
    let shown: SceneData | undefined;
    let selectedProps: Props | undefined;
    let appliedCameraView: Props['cameraView'] | undefined;
    let appliedFit: number | undefined;
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
        if (
          !selectedProps ||
          selectedProps.selectedViewId !== latest.current.selectedViewId ||
          selectedProps.photo !== latest.current.photo ||
          selectedProps.showPhoto !== latest.current.showPhoto
        ) {
          selectedProps = latest.current;
          updateSelected(selectedProps);
        }
        if (latest.current.fitNonce !== appliedFit) {
          appliedFit = latest.current.fitNonce;
          if (appliedFit !== undefined && shown && (shown.views.length || shown.landmarks.length))
            fit();
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
        renderer.render(scene, camera);
        if (moving) requestRender();
      } catch (error) {
        fail(error);
      }
    };
    const requestRender = () => {
      if (disposed || pending) return;
      pending = true;
      animation = requestAnimationFrame(frame);
    };
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
    const doubleClick = (e: MouseEvent) => {
      const { view } = pick(e);
      if (view) {
        cameraFrame(view, center, axes);
        controls.target.copy(root.localToWorld(center.clone()));
      } else return;
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
    const hidden = () => [forwardArrow, motionArrow, photoPlane, grid];
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
    props.showCameras,
    props.showPhoto,
    props.lightBackground,
    props.navigation,
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
