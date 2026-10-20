import { useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Scene as SceneData, SceneView } from '../lib/sfm-scene';

export type ViewerState =
  { status: 'initializing' } | { status: 'ready' } | { status: 'error'; message: string };
type Props = {
  reconstruction?: SceneData;
  selectedViewId?: number;
  selectedPoint?: number;
  photo?: { id: number; image?: string; width: number; height: number };
  onSelectPoint: (id: number) => void;
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

/** World-space camera center and the rows of R^T, without Matrix4 allocations. */
function cameraFrame(view: SceneView, center: THREE.Vector3, axes: THREE.Vector3[]): void {
  const { R, t } = view.pose;
  center.set(
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  );
  // Column k of R is camera axis k expressed in the world frame.
  for (let k = 0; k < 3; k++) axes[k].set(R[k], R[3 + k], R[6 + k]);
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
    const camera = new THREE.PerspectiveCamera(48, 1, 0.001, 100000);
    const controls = own(new OrbitControls(camera, renderer.domElement));
    controls.enableDamping = true;

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
    /** Landmark index of each drawn point, for picking. */
    let pointTracks: number[] = [];

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
    const marker = new THREE.Mesh(
      own(new THREE.SphereGeometry(1, 12, 8)),
      own(new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true, depthTest: false })),
    );
    const photoMaterial = own(
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.85 }),
    );
    const photoPlane = new THREE.Mesh(own(new THREE.PlaneGeometry(1, 1)), photoMaterial);
    photoPlane.matrixAutoUpdate = false;
    for (const object of [forwardArrow, motionArrow, marker, photoPlane]) {
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
        const depth = extent * 0.07;
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
        .then((blob) => createImageBitmap(blob, { colorSpaceConversion: 'none' }))
        .then((bitmap) => {
          if (disposed || generation !== photoGeneration) {
            bitmap.close();
            return;
          }
          const texture = new THREE.Texture(bitmap);
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.flipY = true;
          texture.needsUpdate = true;
          textures.set(image, texture);
          if (textures.size > PHOTO_CACHE) dropTexture(textures.keys().next().value!);
          apply(texture);
        })
        .catch((error) => console.error(error));
    };

    let extent = 1;
    let fitted = false;
    let grid: THREE.GridHelper | undefined;
    const clearGrid = () => {
      if (!grid) return;
      root.remove(grid);
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      grid = undefined;
    };
    releases.push(clearGrid);
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
        clearGrid();
        camera.position.set(0, 0, 5);
        controls.target.set(0, 0, 0);
        controls.update();
      }
      // Points and colors in one pass; bounds without per-point allocations.
      positions.ensure(model.landmarks.length);
      colors.ensure(model.landmarks.length);
      pointTracks = [];
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
        pointTracks.push(i);
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
      const depth = extent * 0.05;
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
        // The bounds are the mapper's coordinates; the orbit camera lives in the scene.
        const middle = root.localToWorld(bounds.getCenter(new THREE.Vector3()));
        camera.position
          .copy(middle)
          .add(new THREE.Vector3(extent * 0.9, extent * 0.7, extent * 1.8));
        controls.target.copy(middle);
        grid = new THREE.GridHelper(extent, 10, '#35415d', '#202b43');
        root.add(grid);
      }
    };
    const updateSelected = (state: Props) => {
      const model = state.reconstruction ?? EMPTY_SCENE;
      const view = model.views.find((v) => v.id === state.selectedViewId);
      parent.dataset.selectedView = String(view?.id ?? '');
      const point = model.landmarks.find((p) => p.trackId === state.selectedPoint);
      parent.dataset.selectedPoint = String(point?.trackId ?? '');
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
      marker.visible = !!point;
      if (point) {
        marker.position.set(point.xyz[0], point.xyz[1], point.xyz[2]);
        marker.scale.setScalar(extent * 0.007);
      }
    };

    // Render on demand: a change of data, selection, size or orbit schedules one frame, and the
    // frame schedules another only while the orbit damping is still moving.
    let shown: SceneData | undefined;
    let selectedProps: Props | undefined;
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
          selectedProps.selectedPoint !== latest.current.selectedPoint ||
          selectedProps.selectedViewId !== latest.current.selectedViewId ||
          selectedProps.photo !== latest.current.photo
        ) {
          selectedProps = latest.current;
          updateSelected(selectedProps);
        }
        const moving = controls.update();
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
    let down = { x: 0, y: 0 };
    const pointerDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
    };
    const pointerUp = (e: PointerEvent) => {
      const model = latest.current.reconstruction;
      if (!model) return;
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          1 - ((e.clientY - rect.top) / rect.height) * 2,
        ),
        camera,
      );
      ray.params.Points = { threshold: extent * 0.008 };
      const hit = ray.intersectObject(cloud)[0];
      if (hit?.index !== undefined) {
        const point = model.landmarks[pointTracks[hit.index]];
        if (point) latest.current.onSelectPoint(point.trackId);
        return;
      }
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
      if (nearest) latest.current.onSelectView(nearest.id);
    };
    renderer.domElement.addEventListener('pointerdown', pointerDown);
    renderer.domElement.addEventListener('pointerup', pointerUp);
    releases.push(() => {
      renderer.domElement.removeEventListener('pointerdown', pointerDown);
      renderer.domElement.removeEventListener('pointerup', pointerUp);
    });
    // The first frame runs now, so a renderer that cannot draw fails before the viewer is ready.
    frame();
    if (!disposed) report({ status: 'ready' });
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
  }, [props.reconstruction, props.selectedViewId, props.selectedPoint, props.photo]);
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
