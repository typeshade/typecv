import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Scene as SceneData, SceneView } from '../lib/sfm-scene';

type Props = {
  reconstruction: SceneData;
  selectedViewId?: number;
  selectedPoint?: number;
  photo?: { id: number; image?: string; width: number; height: number };
  onSelectPoint: (id: number) => void;
  onSelectView: (id: number) => void;
};

function cameraMatrix(view: SceneView) {
  const { R, t } = view.pose;
  return new THREE.Matrix4()
    .set(R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1)
    .invert();
}

export default function SfmViewer(props: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  useEffect(() => {
    const parent = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      parent.dataset.error = 'WebGL is unavailable in this browser.';
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setClearColor('#0b1020', 1);
    parent.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.001, 100000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
      vertexColors: true,
      size: 3,
      sizeAttenuation: false,
    });
    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);
    const frustumGeometry = new THREE.BufferGeometry();
    const linesMaterial = new THREE.LineBasicMaterial({ color: '#ffb454' });
    scene.add(new THREE.LineSegments(frustumGeometry, linesMaterial));
    const pathGeometry = new THREE.BufferGeometry();
    const pathMaterial = new THREE.LineBasicMaterial({ color: '#ffb454' });
    scene.add(new THREE.LineSegments(pathGeometry, pathMaterial));
    const selected = new THREE.Group();
    scene.add(selected);
    const clearSelected = () => {
      selected.traverse((object) => {
        const drawable = object as THREE.Mesh;
        drawable.geometry?.dispose();
        const materials = Array.isArray(drawable.material)
          ? drawable.material
          : [drawable.material];
        for (const m of materials)
          if (m) {
            (m as THREE.MeshBasicMaterial).map?.dispose();
            m.dispose();
          }
      });
      selected.clear();
    };
    let extent = 1;
    let selectedProps: Props | undefined;
    let photoGeneration = 0;
    const updateSelected = (state: Props) => {
      clearSelected();
      const generation = ++photoGeneration;
      const view = state.reconstruction.views.find((v) => v.id === state.selectedViewId);
      parent.dataset.selectedView = String(view?.id ?? '');
      parent.dataset.selectedPoint = String(state.selectedPoint ?? '');
      if (view) {
        const inverse = cameraMatrix(view);
        const origin = new THREE.Vector3().applyMatrix4(inverse);
        const forward = new THREE.Vector3(0, 0, 1).transformDirection(inverse);
        selected.add(new THREE.ArrowHelper(forward, origin, extent * 0.12, '#68ffb1'));
        // Consecutive input frames only: never draw motion across a failed pose.
        const previous = state.reconstruction.views.find((v) =>
          view.sequence !== undefined ? v.sequence === view.sequence - 1 : v.id === view.id - 1,
        );
        if (previous) {
          const before = new THREE.Vector3().applyMatrix4(cameraMatrix(previous));
          const delta = origin.clone().sub(before);
          if (delta.length() > 1e-9)
            selected.add(
              new THREE.ArrowHelper(delta.clone().normalize(), before, delta.length(), '#69b1ff'),
            );
        }
        const photo = state.photo?.id === view.id ? state.photo : view;
        if (photo.image)
          new THREE.TextureLoader().load(photo.image, (texture) => {
            if (generation !== photoGeneration) {
              texture.dispose();
              return;
            }
            texture.colorSpace = THREE.SRGBColorSpace;
            const depth = extent * 0.07;
            const width = photo.width ?? Math.round(view.camera.cx * 2 + 1);
            const height = photo.height ?? Math.round(view.camera.cy * 2 + 1);
            const { fx, fy, cx, cy } = view.camera;
            const plane = new THREE.PlaneGeometry((depth * width) / fx, (depth * height) / fy);
            // Image y increases downward in the calibrated camera coordinate system.
            const mesh = new THREE.Mesh(
              plane,
              new THREE.MeshBasicMaterial({
                map: texture,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.85,
              }),
            );
            const local = new THREE.Matrix4().makeRotationX(Math.PI);
            local.setPosition(
              (depth * ((width - 1) / 2 - cx)) / fx,
              (depth * ((height - 1) / 2 - cy)) / fy,
              depth,
            );
            mesh.applyMatrix4(inverse.clone().multiply(local));
            selected.add(mesh);
          });
      }
      const point = state.reconstruction.landmarks.find((p) => p.trackId === state.selectedPoint);
      if (point) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(extent * 0.007, 12, 8),
          new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true, depthTest: false }),
        );
        marker.position.set(...(point.xyz as unknown as [number, number, number]));
        selected.add(marker);
      }
    };
    let grid: THREE.GridHelper | undefined;
    let shown: SceneData | undefined;
    let fitted = false;
    const update = (model: SceneData) => {
      const points = model.landmarks
        .filter((point) => [...point.xyz].every(Number.isFinite))
        .flatMap((point) => [...point.xyz]);
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      const colors = model.landmarks
        .filter((p) => [...p.xyz].every(Number.isFinite))
        .flatMap((point) => {
          const c = new THREE.Color().setRGB(
            ...(point.color ?? [0.7, 0.7, 0.7]),
            THREE.SRGBColorSpace,
          );
          return [c.r, c.g, c.b];
        });
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      geometry.computeBoundingSphere();
      const bounds = new THREE.Box3();
      for (let i = 0; i < points.length; i += 3)
        bounds.expandByPoint(new THREE.Vector3(points[i], points[i + 1], points[i + 2]));
      const center = bounds.isEmpty() ? new THREE.Vector3() : bounds.getCenter(new THREE.Vector3());
      const size = bounds.isEmpty()
        ? new THREE.Vector3(5, 5, 5)
        : bounds.getSize(new THREE.Vector3());
      extent = Math.max(size.x, size.y, size.z, 0.1);
      const frusta: number[] = [],
        trajectory: number[] = [];
      const ordered = [...model.views].sort(
        (a, b) => (a.sequence ?? a.time ?? a.id) - (b.sequence ?? b.time ?? b.id),
      );
      for (const [index, view] of ordered.entries()) {
        const inverse = cameraMatrix(view);
        const origin = new THREE.Vector3().applyMatrix4(inverse);
        const previous = ordered[index - 1];
        if (
          previous &&
          (view.sequence !== undefined
            ? previous.sequence! + 1 === view.sequence
            : previous.id + 1 === view.id)
        ) {
          const before = new THREE.Vector3().applyMatrix4(cameraMatrix(previous));
          trajectory.push(before.x, before.y, before.z, origin.x, origin.y, origin.z);
        }
        const depth = extent * 0.05;
        const nx = (depth * (view.camera.cx + 0.5)) / view.camera.fx;
        const ny = (depth * (view.camera.cy + 0.5)) / view.camera.fy;
        const corners = [
          [-nx, -ny],
          [nx, -ny],
          [nx, ny],
          [-nx, ny],
        ].map(([x, y]) => new THREE.Vector3(x, y, depth).applyMatrix4(inverse));
        for (let i = 0; i < 4; i++) {
          const a = corners[i],
            b = corners[(i + 1) % 4];
          frusta.push(origin.x, origin.y, origin.z, a.x, a.y, a.z, a.x, a.y, a.z, b.x, b.y, b.z);
        }
      }
      parent.dataset.points = String(model.landmarks.length);
      frustumGeometry.setAttribute('position', new THREE.Float32BufferAttribute(frusta, 3));
      frustumGeometry.computeBoundingSphere();
      pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(trajectory, 3));
      pathGeometry.computeBoundingSphere();
      // Fit once, so later map updates preserve the user's orbit and zoom.
      if (!fitted && points.length) {
        fitted = true;
        camera.position
          .copy(center)
          .add(new THREE.Vector3(extent * 0.9, extent * 0.7, extent * 1.8));
        controls.target.copy(center);
        grid = new THREE.GridHelper(extent, 10, '#35415d', '#202b43');
        scene.add(grid);
      }
    };
    const resize = new ResizeObserver(() => {
      if (!parent.clientWidth || !parent.clientHeight) return;
      camera.aspect = parent.clientWidth / parent.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(parent.clientWidth, parent.clientHeight);
    });
    resize.observe(parent);
    const ray = new THREE.Raycaster();
    let down = { x: 0, y: 0 };
    const pointerDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
    };
    const pointerUp = (e: PointerEvent) => {
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
        const point = latest.current.reconstruction.landmarks.filter((p) =>
          [...p.xyz].every(Number.isFinite),
        )[hit.index];
        if (point) latest.current.onSelectPoint(point.trackId);
      } else {
        const candidates = latest.current.reconstruction.views.map((view) => ({
          view,
          point: new THREE.Vector3().applyMatrix4(cameraMatrix(view)),
        }));
        const near = candidates
          .filter(({ point }) => ray.ray.distanceToPoint(point) < extent * 0.015)
          .sort(
            (a, b) => ray.ray.origin.distanceTo(a.point) - ray.ray.origin.distanceTo(b.point),
          )[0];
        if (near) latest.current.onSelectView(near.view.id);
      }
    };
    renderer.domElement.addEventListener('pointerdown', pointerDown);
    renderer.domElement.addEventListener('pointerup', pointerUp);
    let animation = 0;
    const frame = () => {
      animation = requestAnimationFrame(frame);
      if (shown !== latest.current.reconstruction) {
        shown = latest.current.reconstruction;
        update(shown);
      }
      if (
        selectedProps?.reconstruction !== latest.current.reconstruction ||
        selectedProps?.selectedPoint !== latest.current.selectedPoint ||
        selectedProps?.selectedViewId !== latest.current.selectedViewId ||
        selectedProps?.photo !== latest.current.photo
      ) {
        selectedProps = latest.current;
        updateSelected(selectedProps);
      }
      controls.update();
      renderer.render(scene, camera);
    };
    frame();
    return () => {
      cancelAnimationFrame(animation);
      photoGeneration++;
      clearSelected();
      renderer.domElement.removeEventListener('pointerdown', pointerDown);
      renderer.domElement.removeEventListener('pointerup', pointerUp);
      resize.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      frustumGeometry.dispose();
      linesMaterial.dispose();
      pathGeometry.dispose();
      pathMaterial.dispose();
      grid?.geometry.dispose();
      if (grid) (grid.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);
  return (
    <div
      className="viewer"
      ref={host}
      role="img"
      aria-label="Growing sparse 3D reconstruction with camera trajectory"
    >
      <span className="viewer-error" />
    </div>
  );
}
