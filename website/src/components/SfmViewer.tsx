import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

type View = {
  id: number;
  pose: { R: Float64Array; t: Float64Array };
  camera: { fx: number; fy: number; cx: number; cy: number };
};
type SceneData = { views: View[]; landmarks: { xyz: Float64Array }[] };

export default function SfmViewer({ reconstruction }: { reconstruction: SceneData }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const parent = host.current;
    if (!parent) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      parent.dataset.error = 'WebGL is unavailable in this browser.';
      return;
    }
    parent.dataset.error = '';
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(parent.clientWidth, parent.clientHeight);
    renderer.setClearColor('#0b1020', 1);
    parent.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      48,
      parent.clientWidth / parent.clientHeight,
      0.001,
      100000,
    );
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const valid = reconstruction.landmarks
      .filter((point) => point.xyz.length === 3 && [...point.xyz].every(Number.isFinite))
      .flatMap((point) => [...point.xyz]);
    const count = Math.floor(valid.length / 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(valid, 3));
    const cloud = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ color: '#69b1ff', size: 3, sizeAttenuation: false }),
    );
    scene.add(cloud);
    let min = new THREE.Vector3(Infinity, Infinity, Infinity),
      max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (let i = 0; i < count; i++) {
      const p = new THREE.Vector3(valid[i * 3], valid[i * 3 + 1], valid[i * 3 + 2]);
      min.min(p);
      max.max(p);
    }
    const center = count ? min.clone().add(max).multiplyScalar(0.5) : new THREE.Vector3();
    const extent = count ? Math.max(max.x - min.x, max.y - min.y, max.z - min.z, 0.1) : 5;
    const frustumPositions: number[] = [];
    for (const view of reconstruction.views) {
      const p = view.pose;
      const worldToCamera = new THREE.Matrix4().set(
        p.R[0],
        p.R[1],
        p.R[2],
        p.t[0],
        p.R[3],
        p.R[4],
        p.R[5],
        p.t[1],
        p.R[6],
        p.R[7],
        p.R[8],
        p.t[2],
        0,
        0,
        0,
        1,
      );
      const cameraToWorld = worldToCamera.invert();
      const depth = extent * 0.09;
      const nx = Math.min(0.75, 90 / Math.max(view.camera.fx, 1)),
        ny = Math.min(0.75, 70 / Math.max(view.camera.fy, 1));
      const corners = (
        [
          [-nx, -ny, depth],
          [nx, -ny, depth],
          [nx, ny, depth],
          [-nx, ny, depth],
        ] as [number, number, number][]
      ).map((v) => new THREE.Vector3(...v).applyMatrix4(cameraToWorld));
      const origin = new THREE.Vector3(0, 0, 0).applyMatrix4(cameraToWorld);
      for (let i = 0; i < 4; i++) {
        const a = corners[i],
          b = corners[(i + 1) % 4];
        frustumPositions.push(
          origin.x,
          origin.y,
          origin.z,
          a.x,
          a.y,
          a.z,
          a.x,
          a.y,
          a.z,
          b.x,
          b.y,
          b.z,
        );
      }
    }
    const frustumGeometry = new THREE.BufferGeometry();
    frustumGeometry.setAttribute('position', new THREE.Float32BufferAttribute(frustumPositions, 3));
    const frusta = new THREE.LineSegments(
      frustumGeometry,
      new THREE.LineBasicMaterial({ color: '#ffb454' }),
    );
    scene.add(frusta);
    scene.add(new THREE.GridHelper(extent, 10, '#35415d', '#202b43'));
    camera.position.copy(center).add(new THREE.Vector3(extent * 0.9, extent * 0.7, extent * 1.8));
    controls.target.copy(center);
    controls.update();
    const resize = new ResizeObserver(() => {
      const width = parent.clientWidth,
        height = parent.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    });
    resize.observe(parent);
    let animation = 0;
    const frame = () => {
      animation = requestAnimationFrame(frame);
      controls.update();
      renderer.render(scene, camera);
    };
    frame();
    return () => {
      cancelAnimationFrame(animation);
      resize.disconnect();
      controls.dispose();
      geometry.dispose();
      frustumGeometry.dispose();
      (cloud.material as THREE.Material).dispose();
      (frusta.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [reconstruction]);
  return (
    <div
      className="viewer"
      ref={host}
      role="img"
      aria-label="Sparse 3D reconstruction with camera positions"
    >
      <span className="viewer-error" />
    </div>
  );
}
