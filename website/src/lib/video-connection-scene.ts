import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { VideoActivityLink } from './multi-video-activity';

export interface VideoConnectionGraph {
  element: HTMLElement;
  nodes: {
    id: string;
    videoId?: string;
    label: string;
    image?: string;
    detail: string;
    active: boolean;
    color?: string;
  }[];
  links: readonly VideoActivityLink[];
  selected?: string;
  focusConnections: boolean;
  onSelect: (id: string) => void;
}

type Card = {
  sprite: THREE.Sprite;
  canvas: HTMLCanvasElement;
  texture: THREE.CanvasTexture;
  image?: HTMLImageElement;
  source?: string;
  signature?: string;
};
type Edge = {
  line: THREE.Line<THREE.BufferGeometry, THREE.LineDashedMaterial>;
  curve: THREE.QuadraticBezierCurve3;
  dot: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  active: boolean;
};

/** A schematic graph drawn by the reconstruction renderer, with its own viewing camera.
 * Node positions describe the diagram only; they never enter reconstructed poses or exports. */
export class VideoConnectionScene {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  private readonly cards = new Map<string, Card>();
  private readonly edges = new Map<string, Edge>();
  private readonly dotGeometry = new THREE.SphereGeometry(0.055, 6, 4);
  private readonly dotMaterial = new THREE.MeshBasicMaterial({ color: '#4096ff' });
  private readonly ray = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private controls?: OrbitControls;
  private data?: VideoConnectionGraph;
  private element?: HTMLElement;
  private resize?: ResizeObserver;
  private light = false;
  private layout = '';
  private updatedAt = 0;
  private down = { x: 0, y: 0 };
  private readonly pointerDown = (event: PointerEvent) => {
    this.down = { x: event.clientX, y: event.clientY };
  };
  private readonly pointerUp = (event: PointerEvent) => {
    if (
      !this.element ||
      !this.data ||
      Math.hypot(event.clientX - this.down.x, event.clientY - this.down.y) > 5
    )
      return;
    const rect = this.element.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      1 - ((event.clientY - rect.top) / rect.height) * 2,
    );
    this.ray.setFromCamera(this.pointer, this.camera);
    const hit = this.ray.intersectObjects([...this.cards.values()].map((card) => card.sprite))[0];
    if (hit) this.data.onSelect(String(hit.object.userData.id));
  };

  constructor(private readonly requestRender: () => void) {
    this.camera.position.set(0, 0, 12);
  }

  private disconnect(): void {
    this.resize?.disconnect();
    this.controls?.dispose();
    this.element?.removeEventListener('pointerdown', this.pointerDown);
    this.element?.removeEventListener('pointerup', this.pointerUp);
    this.element?.closest('.mv-stage-activity')?.removeEventListener('scroll', this.requestRender);
    this.element = undefined;
    this.controls = undefined;
  }

  hide(): void {
    this.disconnect();
    this.data = undefined;
    for (const card of this.cards.values()) this.releaseCard(card);
    this.cards.clear();
    for (const edge of this.edges.values()) this.releaseEdge(edge);
    this.edges.clear();
    this.layout = '';
  }

  dispose(): void {
    this.hide();
    this.dotGeometry.dispose();
    this.dotMaterial.dispose();
  }

  private releaseCard(card: Card): void {
    if (card.image) card.image.onload = card.image.onerror = null;
    this.scene.remove(card.sprite);
    card.sprite.material.dispose();
    card.texture.dispose();
  }

  private releaseEdge(edge: Edge): void {
    this.scene.remove(edge.line, edge.dot);
    edge.line.geometry.dispose();
    edge.line.material.dispose();
  }

  private drawCard(card: Card, node: VideoConnectionGraph['nodes'][number], index: number): void {
    const selected = this.data?.selected === node.id;
    const signature = JSON.stringify([
      node.label,
      node.detail,
      node.active,
      node.color,
      selected,
      this.light,
      index,
    ]);
    if (signature === card.signature) return;
    card.signature = signature;
    const ctx = card.canvas.getContext('2d')!;
    const ink = this.light ? '#172033' : '#eef3ff';
    ctx.clearRect(0, 0, 256, 168);
    ctx.fillStyle = this.light ? '#ffffff' : '#172234';
    ctx.strokeStyle =
      selected || node.active ? '#4096ff' : (node.color ?? (this.light ? '#b8c5d6' : '#42516a'));
    ctx.lineWidth = selected || node.active ? 5 : 2;
    ctx.beginPath();
    ctx.roundRect(3, 3, 250, 162, 12);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = this.light ? '#e5edf7' : '#27364c';
    ctx.fillRect(10, 10, 236, 108);
    if (card.image?.complete && card.image.naturalWidth) {
      const image = card.image;
      const scale = Math.max(236 / image.naturalWidth, 108 / image.naturalHeight);
      const width = 236 / scale,
        height = 108 / scale;
      ctx.drawImage(
        image,
        (image.naturalWidth - width) / 2,
        (image.naturalHeight - height) / 2,
        width,
        height,
        10,
        10,
        236,
        108,
      );
    } else {
      ctx.fillStyle = '#739bdb';
      ctx.beginPath();
      ctx.moveTo(116, 38);
      ctx.lineTo(116, 90);
      ctx.lineTo(154, 64);
      ctx.fill();
    }
    ctx.fillStyle = ink;
    ctx.font = '600 15px sans-serif';
    let label = `${index + 1} · ${node.label}`;
    while (ctx.measureText(label).width > 230 && label.length > 2) label = label.slice(0, -2) + '…';
    ctx.fillText(label, 12, 140);
    ctx.fillStyle = this.light ? '#536581' : '#b5c6e0';
    ctx.font = '13px sans-serif';
    ctx.fillText(node.detail, 12, 157);
    if (node.active) {
      ctx.fillStyle = '#4096ff';
      ctx.beginPath();
      ctx.arc(236, 151, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    card.texture.needsUpdate = true;
  }

  /** Returns true only while damping or a recently reported active connection needs a frame. */
  draw(
    renderer: THREE.WebGLRenderer,
    data: VideoConnectionGraph,
    light: boolean,
    reducedMotion: boolean,
  ): boolean {
    if (data.element !== this.element) {
      this.disconnect();
      this.element = data.element;
      const controls = (this.controls = new OrbitControls(this.camera, data.element));
      controls.enableDamping = !reducedMotion;
      controls.enablePan = false;
      controls.zoomSpeed = 0.6;
      controls.minPolarAngle = Math.PI * 0.32;
      controls.maxPolarAngle = Math.PI * 0.68;
      controls.minAzimuthAngle = -0.45;
      controls.maxAzimuthAngle = 0.45;
      controls.addEventListener('change', this.requestRender);
      this.resize = new ResizeObserver(this.requestRender);
      this.resize.observe(data.element);
      data.element.addEventListener('pointerdown', this.pointerDown);
      data.element.addEventListener('pointerup', this.pointerUp);
      data.element.closest('.mv-stage-activity')?.addEventListener('scroll', this.requestRender);
    }
    const rect = data.element.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return false;
    const aspect = rect.width / rect.height;
    const layout = `${data.nodes.map((node) => node.id).join(':')}/${rect.width}/${rect.height}`;
    const relayout = this.layout !== layout;
    const changed = this.data !== data || relayout || this.light !== light;
    if (this.data?.links !== data.links) this.updatedAt = performance.now();
    this.data = data;
    this.light = light;
    if (changed) {
      const ids = new Set(data.nodes.map((node) => node.id));
      for (const [id, card] of this.cards)
        if (!ids.has(id)) {
          this.releaseCard(card);
          this.cards.delete(id);
        }
      const columns = Math.max(
        1,
        Math.min(data.nodes.length, Math.ceil(Math.sqrt(data.nodes.length * aspect * 0.68))),
      );
      const rows = Math.ceil(data.nodes.length / columns);
      data.nodes.forEach((node, index) => {
        let card = this.cards.get(node.id);
        if (!card) {
          const canvas = document.createElement('canvas');
          canvas.width = 256;
          canvas.height = 168;
          const texture = new THREE.CanvasTexture(canvas);
          texture.colorSpace = THREE.SRGBColorSpace;
          const sprite = new THREE.Sprite(
            new THREE.SpriteMaterial({ map: texture, depthTest: false }),
          );
          sprite.userData.id = node.id;
          sprite.scale.set(2.8, (2.8 * 168) / 256, 1);
          sprite.renderOrder = 2;
          card = { canvas, texture, sprite };
          this.cards.set(node.id, card);
          this.scene.add(sprite);
        }
        card.sprite.position.set(
          ((index % columns) - (columns - 1) / 2) * 3.5,
          ((rows - 1) / 2 - Math.floor(index / columns)) * 2.5,
          0,
        );
        if (node.image !== card.source) {
          if (card.image) card.image.onload = card.image.onerror = null;
          card.source = node.image;
          card.image = undefined;
          card.signature = undefined;
          if (node.image) {
            const image = (card.image = new Image());
            image.crossOrigin = 'anonymous';
            image.onload = () => {
              card!.signature = undefined;
              this.requestRender();
            };
            image.src = node.image;
          }
        }
        this.drawCard(card, node, index);
      });
      const byVideo = new Map(
        data.nodes.filter((node) => node.videoId).map((node) => [node.videoId!, node.id]),
      );
      const keys = new Set<string>();
      for (const link of data.links) {
        const first = byVideo.get(link.first),
          second = byVideo.get(link.second);
        const a = first && this.cards.get(first),
          b = second && this.cards.get(second);
        if (!a || !b) continue;
        const key = link.first + ':' + link.second;
        keys.add(key);
        let edge = this.edges.get(key);
        if (!edge || relayout) {
          if (edge) this.releaseEdge(edge);
          const start = a.sprite.position.clone(),
            end = b.sprite.position.clone();
          const middle = start.clone().add(end).multiplyScalar(0.5);
          start.z = end.z = -0.1;
          middle.z = -1.2;
          const curve = new THREE.QuadraticBezierCurve3(start, middle, end);
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
            new THREE.LineDashedMaterial({ transparent: true, dashSize: 0.18, gapSize: 0.12 }),
          );
          line.computeLineDistances();
          const dot = new THREE.Mesh(this.dotGeometry, this.dotMaterial);
          edge = { curve, line, dot, active: false };
          this.edges.set(key, edge);
          this.scene.add(line, dot);
        }
        const selectedVideo = data.nodes.find((node) => node.id === data.selected)?.videoId;
        const related =
          !data.focusConnections ||
          !selectedVideo ||
          selectedVideo === link.first ||
          selectedVideo === link.second;
        const material = edge.line.material;
        material.color.set(
          link.active
            ? '#4096ff'
            : link.accepted > 0
              ? '#52c41a'
              : link.matches > 0
                ? '#fa8c16'
                : light
                  ? '#7285a0'
                  : '#8b9ebd',
        );
        material.opacity = related ? (link.accepted > 0 || link.active ? 1 : 0.65) : 0.16;
        material.gapSize = link.accepted > 0 ? 0 : 0.12;
        edge.active = link.active && related;
      }
      for (const [key, edge] of this.edges)
        if (!keys.has(key)) {
          this.releaseEdge(edge);
          this.edges.delete(key);
        }
      if (relayout) {
        this.layout = layout;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        const distance =
          Math.max((columns * 3.5) / aspect, rows * 2.5) / (2 * Math.tan(Math.PI / 8));
        this.camera.position.set(0, 0, Math.max(4, distance * 1.08));
        this.controls!.target.set(0, 0, 0);
        this.controls!.minDistance = this.camera.position.z * 0.55;
        this.controls!.maxDistance = this.camera.position.z * 2;
      }
    }
    // Image loads invalidate just their card, without recreating scene objects or a renderer.
    data.nodes.forEach((node, index) => this.drawCard(this.cards.get(node.id)!, node, index));
    this.controls!.enableDamping = !reducedMotion;
    const moving = this.controls!.update();
    let flowing = false;
    const now = performance.now();
    for (const edge of this.edges.values()) {
      edge.dot.visible = edge.active && !reducedMotion && now - this.updatedAt < 5000;
      if (edge.dot.visible) {
        flowing = true;
        edge.dot.position.copy(edge.curve.getPoint((now % 1800) / 1800));
      }
    }
    const canvas = renderer.domElement.getBoundingClientRect();
    const x = rect.left - canvas.left,
      y = canvas.bottom - rect.bottom;
    const left = Math.max(0, x),
      bottom = Math.max(0, y);
    const width = Math.max(0, Math.min(canvas.width, x + rect.width) - left);
    const height = Math.max(0, Math.min(canvas.height, y + rect.height) - bottom);
    renderer.setViewport(x, y, rect.width, rect.height);
    renderer.setScissor(left, bottom, width, height);
    renderer.setScissorTest(true);
    renderer.render(this.scene, this.camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, canvas.width, canvas.height);
    return moving || flowing;
  }
}
