/**
 * The clouds of `docs/dense-mvs.md`, section 3.13: a sphere and a plane of oriented points. The
 * browser and Node build the same clouds. Normals use the octahedral map of `GpuDensePoints`.
 */
const SNORM = 32767;

function encodeNormal(x, y, z) {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  let px = x / l1,
    py = y / l1;
  if (z < 0) {
    const ox = px;
    px = (1 - Math.abs(py)) * (ox >= 0 ? 1 : -1);
    py = (1 - Math.abs(ox)) * (py >= 0 ? 1 : -1);
  }
  const q = (v) => Math.max(-SNORM, Math.min(SNORM, Math.round(v * SNORM)));
  return [q(px), q(py)];
}

function cloudOf(points, normals, colorOf) {
  const n = points.length;
  const positions = new Float32Array(3 * n),
    colors = new Uint8Array(3 * n),
    packed = new Int16Array(2 * n);
  for (let i = 0; i < n; i++) {
    positions.set(points[i], 3 * i);
    colors.set(colorOf(points[i]), 3 * i);
    packed.set(encodeNormal(...normals[i]), 2 * i);
  }
  return { count: n, positions, colors, normals: packed };
}

/** `count` points on the unit sphere (a Fibonacci lattice) with outward normals, red by x. */
export function sphereCloud(count = 4000, radius = 1) {
  const points = [],
    normals = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const z = 1 - (2 * (i + 0.5)) / count,
      r = Math.sqrt(1 - z * z),
      a = i * golden;
    const n = [r * Math.cos(a), r * Math.sin(a), z];
    normals.push(n);
    points.push(n.map((v) => v * radius));
  }
  return cloudOf(points, normals, (p) => [Math.round(127.5 + 127.5 * p[0]), 80, 200]);
}

/** A plane z = 0 over [0, size]², `side` × `side` points on a jittered grid, normals +z. */
export function planeCloud(side = 40, size = 1) {
  let a = 7;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const points = [],
    normals = [];
  const step = size / (side - 1);
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) {
      points.push([
        x * step + (random() - 0.5) * 0.3 * step,
        y * step + (random() - 0.5) * 0.3 * step,
        0,
      ]);
      normals.push([0, 0, 1]);
    }
  return cloudOf(points, normals, () => [30, 160, 60]);
}
