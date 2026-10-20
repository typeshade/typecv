/**
 * The synthetic textured scene of `docs/dense-mvs.md`, section 7.1: a ground plane, a box and a
 * sphere under a value-noise texture, seen by 5 views at 128 × 96 px with known K, R, t and
 * radial distortion. A host ray caster in f64 renders the views (4 × 4 samples per pixel) and the
 * exact depth and normal maps of each view on its undistorted pixel grid.
 */
import { cameraProjection, unprojectPixel } from '../../dist/geometry/camera.js';

export const MVS_WIDTH = 128;
export const MVS_HEIGHT = 96;
const SUPERSAMPLES = 4;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (a) => {
  const n = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / n, a[1] / n, a[2] / n];
};

const BOX = { min: [-1.05, -0.25, 0], max: [-0.1, 0.7, 0.45] };
const SPHERE = { center: [0.6, -0.3, 0.45], radius: 0.45 };

/** The nearest hit of a world ray: distance t, the point and the unit world normal. */
export function castRay(origin, dir) {
  let best = { t: Infinity };
  if (dir[2] < 0) {
    const t = -origin[2] / dir[2];
    if (t > 1e-9) best = { t, normal: [0, 0, 1] };
  }
  // The box, by slabs.
  let near = -Infinity,
    far = Infinity,
    axis = -1,
    sign = 0;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dir[i]) < 1e-15) {
      if (origin[i] < BOX.min[i] || origin[i] > BOX.max[i]) near = Infinity;
      continue;
    }
    let t0 = (BOX.min[i] - origin[i]) / dir[i],
      t1 = (BOX.max[i] - origin[i]) / dir[i],
      s = -1;
    if (t0 > t1) {
      [t0, t1] = [t1, t0];
      s = 1;
    }
    if (t0 > near) {
      near = t0;
      axis = i;
      sign = s;
    }
    far = Math.min(far, t1);
  }
  if (near <= far && near > 1e-9 && near < best.t) {
    const normal = [0, 0, 0];
    normal[axis] = sign;
    best = { t: near, normal };
  }
  // The sphere.
  const oc = sub(origin, SPHERE.center);
  const a = dot(dir, dir),
    b = dot(oc, dir),
    c = dot(oc, oc) - SPHERE.radius ** 2,
    disc = b * b - a * c;
  if (disc >= 0) {
    const t = (-b - Math.sqrt(disc)) / a;
    if (t > 1e-9 && t < best.t) {
      const p = [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]];
      best = { t, normal: unit(sub(p, SPHERE.center)) };
    }
  }
  if (!Number.isFinite(best.t)) return undefined;
  return finish(origin, dir, best);
}
function finish(origin, dir, best) {
  const point = [
    origin[0] + best.t * dir[0],
    origin[1] + best.t * dir[1],
    origin[2] + best.t * dir[2],
  ];
  return { t: best.t, point, normal: best.normal };
}

/**
 * The nearest point of the scene's surfaces to world point `p`: its distance and the unit world
 * normal of the surface there. The surfaces are the ground plane z = 0, the six faces of the box
 * and the sphere.
 */
export function nearestSurface(p) {
  let best = { distance: Math.abs(p[2]), normal: [0, 0, 1] };
  for (let axis = 0; axis < 3; axis++)
    for (const side of [0, 1]) {
      const q = p.map((v, k) => Math.min(Math.max(v, BOX.min[k]), BOX.max[k]));
      q[axis] = side ? BOX.max[axis] : BOX.min[axis];
      const distance = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
      if (distance < best.distance) {
        const normal = [0, 0, 0];
        normal[axis] = side ? 1 : -1;
        best = { distance, normal };
      }
    }
  const toCenter = sub(p, SPHERE.center),
    distance = Math.abs(Math.hypot(...toCenter) - SPHERE.radius);
  if (distance < best.distance) best = { distance, normal: unit(toCenter) };
  return best;
}

/** Smooth 3D value noise over a lattice of `cell` world units, in [0, 1]. */
function valueNoise(p, cell, salt) {
  const lattice = (i, j, k) => {
    let h = Math.imul(i, 73856093) ^ Math.imul(j, 19349663) ^ Math.imul(k, 83492791) ^ salt;
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };
  const x = p[0] / cell,
    y = p[1] / cell,
    z = p[2] / cell;
  const i = Math.floor(x),
    j = Math.floor(y),
    k = Math.floor(z);
  const fade = (t) => t * t * (3 - 2 * t);
  const u = fade(x - i),
    v = fade(y - j),
    w = fade(z - k);
  const mix = (a, b, t) => a + (b - a) * t;
  return mix(
    mix(
      mix(lattice(i, j, k), lattice(i + 1, j, k), u),
      mix(lattice(i, j + 1, k), lattice(i + 1, j + 1, k), u),
      v,
    ),
    mix(
      mix(lattice(i, j, k + 1), lattice(i + 1, j, k + 1), u),
      mix(lattice(i, j + 1, k + 1), lattice(i + 1, j + 1, k + 1), u),
      v,
    ),
    w,
  );
}
/** The albedo at a world point: two octaves of value noise, in [0.1, 0.9]. */
export function albedo(p) {
  return (
    0.1 +
    0.8 *
      (0.4 * valueNoise(p, 0.3, 17) +
        0.35 * valueNoise(p, 0.15, 91) +
        0.25 * valueNoise(p, 0.075, 5))
  );
}

function lookAt(center, target) {
  const forward = unit(sub(target, center));
  const right = unit(cross(forward, [0, 0, 1]));
  const down = cross(forward, right);
  const R = [...right, ...down, ...forward];
  const t = [-dot(right, center), -dot(down, center), -dot(forward, center)];
  return { R, t };
}

/** World ray direction (not unit) of a camera-frame direction. */
function worldDirection(R, d) {
  return [
    R[0] * d[0] + R[3] * d[1] + R[6] * d[2],
    R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
    R[2] * d[0] + R[5] * d[1] + R[8] * d[2],
  ];
}
function cameraPoint(view, X) {
  const { R, t } = view.pose;
  return [0, 1, 2].map((i) => R[3 * i] * X[0] + R[3 * i + 1] * X[1] + R[3 * i + 2] * X[2] + t[i]);
}

/** The RGBA of a view: each pixel the mean albedo of 4 × 4 rays through its distorted area. */
function render(view) {
  const rgba = new Uint8Array(view.width * view.height * 4);
  for (let y = 0; y < view.height; y++)
    for (let x = 0; x < view.width; x++) {
      let sum = 0;
      for (let sy = 0; sy < SUPERSAMPLES; sy++)
        for (let sx = 0; sx < SUPERSAMPLES; sx++) {
          const pixel = [x - 0.5 + (sx + 0.5) / SUPERSAMPLES, y - 0.5 + (sy + 0.5) / SUPERSAMPLES];
          const xy = unprojectPixel(view.camera, pixel);
          const hit = castRay(view.center, worldDirection(view.pose.R, [xy[0], xy[1], 1]));
          sum += hit ? albedo(hit.point) : 0.5;
        }
      const byte = Math.round((sum / SUPERSAMPLES ** 2) * 255),
        i = (y * view.width + x) * 4;
      rgba[i] = rgba[i + 1] = rgba[i + 2] = byte;
      rgba[i + 3] = 255;
    }
  return rgba;
}

/** The exact optical-axis depth and camera-frame unit normal of each undistorted pixel. */
function truth(view) {
  const P = view.width * view.height,
    depth = new Float64Array(P),
    normal = new Float64Array(3 * P);
  const { fx, fy, cx, cy } = view.camera;
  for (let y = 0; y < view.height; y++)
    for (let x = 0; x < view.width; x++) {
      const i = y * view.width + x;
      const hit = castRay(
        view.center,
        worldDirection(view.pose.R, [(x - cx) / fx, (y - cy) / fy, 1]),
      );
      if (!hit) continue;
      depth[i] = hit.t;
      const { R } = view.pose;
      for (let k = 0; k < 3; k++)
        normal[3 * i + k] =
          R[3 * k] * hit.normal[0] + R[3 * k + 1] * hit.normal[1] + R[3 * k + 2] * hit.normal[2];
    }
  return { depth, normal };
}

function visible(view, X) {
  const q = cameraPoint(view, X);
  if (!(q[2] > 0)) return undefined;
  const projected = cameraProjection(view.camera, q);
  if (!projected) return undefined;
  const [u, v] = projected.pixel;
  if (!(u >= 0 && v >= 0 && u <= view.width - 1 && v <= view.height - 1)) return undefined;
  const toward = sub(X, view.center),
    distance = Math.hypot(...toward);
  const hit = castRay(view.center, unit(toward));
  if (!hit || hit.t < distance * (1 - 1e-6)) return undefined;
  return projected.pixel;
}

/**
 * The scene: 5 views around the target, `landmarks` points sampled on the surfaces through
 * random reference pixels, with an observation in each view that sees them. View 2 is the
 * central view and the reference of the tests.
 */
export function mvsScene({ landmarks = 30, seed = 7 } = {}) {
  const width = MVS_WIDTH,
    height = MVS_HEIGHT,
    f = 160;
  const target = [0, 0, 0.2];
  const placements = [
    { azimuth: -24, elevation: 54, k1: -0.06 },
    { azimuth: -12, elevation: 66, k1: 0 },
    { azimuth: 0, elevation: 60, k1: -0.04 },
    { azimuth: 12, elevation: 52, k1: 0.03 },
    { azimuth: 24, elevation: 64, k1: 0 },
  ];
  const views = placements.map(({ azimuth, elevation, k1 }, id) => {
    const a = (azimuth * Math.PI) / 180,
      e = (elevation * Math.PI) / 180,
      distance = 3.5;
    const center = [
      target[0] + distance * Math.cos(e) * Math.sin(a),
      target[1] - distance * Math.cos(e) * Math.cos(a),
      target[2] + distance * Math.sin(e),
    ];
    return {
      id,
      width,
      height,
      pose: lookAt(center, target),
      camera: { fx: f, fy: f, cx: (width - 1) / 2, cy: (height - 1) / 2, k1, k2: 0 },
      center,
    };
  });
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const reference = views[2];
  const points = [];
  for (let tries = 0; points.length < landmarks && tries < 100 * landmarks; tries++) {
    const x = 2 + next() * (width - 4),
      y = 2 + next() * (height - 4);
    const { fx, fy, cx, cy } = reference.camera;
    const hit = castRay(
      reference.center,
      worldDirection(reference.pose.R, [(x - cx) / fx, (y - cy) / fy, 1]),
    );
    if (!hit) continue;
    const observations = [];
    for (const view of views) {
      const pixel = visible(view, hit.point);
      if (pixel) observations.push({ viewId: view.id, pixel });
    }
    if (observations.length >= 3) points.push({ xyz: hit.point, observations });
  }
  return {
    views: views.map(({ center: _center, ...view }) => view),
    landmarks: points,
    images: new Map(views.map((view) => [view.id, { rgba: render(view), width, height }])),
    truth: views.map((view) => truth(view)),
  };
}

/** The undistorted gray image of a view on the host, in f64: the reference of `mvsUndistort`. */
export function undistortReference(rgba, camera) {
  const width = MVS_WIDTH,
    height = MVS_HEIGHT;
  const P = width * height,
    gray = new Float64Array(P),
    out = new Float64Array(P);
  for (let i = 0; i < P; i++)
    gray[i] = (0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]) / 255;
  const { fx, fy, cx, cy, k1 = 0, k2 = 0 } = camera;
  if (!k1 && !k2) return gray;
  for (let v = 0; v < height; v++)
    for (let u = 0; u < width; u++) {
      const x = (u - cx) / fx,
        y = (v - cy) / fy,
        r2 = x * x + y * y,
        d = 1 + k1 * r2 + k2 * r2 * r2;
      const sx = fx * x * d + cx,
        sy = fy * y * d + cy;
      let value = -1;
      if (sx >= 0 && sy >= 0 && sx <= width - 1 && sy <= height - 1) {
        const x0 = Math.floor(sx),
          y0 = Math.floor(sy),
          x1 = Math.min(x0 + 1, width - 1),
          y1 = Math.min(y0 + 1, height - 1),
          ax = sx - x0,
          ay = sy - y0;
        const at = (xx, yy) => gray[yy * width + xx];
        value =
          (at(x0, y0) * (1 - ax) + at(x1, y0) * ax) * (1 - ay) +
          (at(x0, y1) * (1 - ax) + at(x1, y1) * ax) * ay;
      }
      out[v * width + u] = value;
    }
  return out;
}

const quantile = (sorted, q) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))] : NaN;

/**
 * Depth and normal errors of a reference's photometric result against the truth. A pixel is
 * kept when its cost is at most 0.9 (NCC at least 0.1) and its depth lies in the view's range,
 * the photometric rules of section 3.9, or when `keep(i)` holds if it is given (the filter's
 * output: a depth above 0). `scale` is the plan's normalization scale.
 */
export function depthMetrics(result, truthMap, scale, range, decodeNormal, keep) {
  keep ??= (i) =>
    result.cost[i] <= 0.9 && result.depth[i] >= range[0] && result.depth[i] <= range[1];
  const relative = [],
    all = [],
    angles = [];
  let valid = 0;
  for (let i = 0; i < result.depth.length; i++) {
    const expected = truthMap.depth[i] / scale;
    if (!(expected > 0)) continue;
    valid++;
    const depth = result.depth[i],
      error = Math.abs(depth - expected) / expected;
    all.push(error);
    if (!keep(i)) continue;
    relative.push(error);
    const n = decodeNormal(result.normal[i]);
    const cosine =
      n[0] * truthMap.normal[3 * i] +
      n[1] * truthMap.normal[3 * i + 1] +
      n[2] * truthMap.normal[3 * i + 2];
    angles.push((Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI);
  }
  relative.sort((a, b) => a - b);
  all.sort((a, b) => a - b);
  angles.sort((a, b) => a - b);
  return {
    kept: relative.length / valid,
    median: quantile(relative, 0.5),
    p90: quantile(relative, 0.9),
    p95: quantile(relative, 0.95),
    within1pct: relative.filter((e) => e <= 0.01).length / valid,
    medianAll: quantile(all, 0.5),
    medianNormalDeg: quantile(angles, 0.5),
  };
}

/** The pixel (x, y) and the depth of a world point in a view's undistorted image. */
function projectPinhole(view, X) {
  const q = cameraPoint(view, X),
    { fx, fy, cx, cy } = view.camera;
  return [(fx * q[0]) / q[2] + cx, (fy * q[1]) / q[2] + cy, q[2]];
}

/**
 * The fused points of `reconstructDenseGpu` against the truth (section 7.1). Each point's
 * reference is the view whose slice of the cloud holds it. `scale` is the plan's normalization
 * scale, the scene radius. Returns the share of points within 1 % and 5 % of the radius of the
 * nearest true surface, the median and 95th percentile of that distance in radii, the median
 * normal error, the share of duplicates, and the share of colors that match the reference's
 * undistorted image. A point is a duplicate when it lies within 0.5 px of an earlier point in
 * that point's reference view, at a depth within 1 % of the earlier point's. A color matches when
 * a pixel of the 3 × 3 block around the point's projection into its reference has that color
 * within 1, in the host undistortion of the reference's image.
 */
export function fusionMetrics(scene, result, scale, decodeNormal) {
  const { count, positions, normals, colors } = result.points;
  const views = new Map(scene.views.map((view) => [view.id, view]));
  const reference = new Array(count);
  for (const view of result.views)
    for (let k = view.firstPoint; k < view.firstPoint + view.points; k++)
      reference[k] = views.get(view.id);
  const point = (k) => [positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]];
  const distances = [],
    angles = [];
  for (let k = 0; k < count; k++) {
    const surface = nearestSurface(point(k));
    distances.push(surface.distance / scale);
    const word = ((normals[2 * k] & 0xffff) | (normals[2 * k + 1] << 16)) >>> 0;
    const cosine = dot(decodeNormal(word), surface.normal);
    angles.push((Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI);
  }
  // Duplicates, view by view: every point projected into the view, binned by pixel.
  const duplicate = new Uint8Array(count);
  for (const view of scene.views) {
    const projected = Array.from({ length: count }, (_, k) => projectPinhole(view, point(k)));
    const bins = new Map();
    projected.forEach(([x, y], k) => {
      const key = `${Math.floor(x)},${Math.floor(y)}`;
      if (!bins.has(key)) bins.set(key, []);
      bins.get(key).push(k);
    });
    for (let a = 0; a < count; a++) {
      if (reference[a] !== view) continue;
      const [x, y, z] = projected[a];
      for (let bx = Math.floor(x) - 1; bx <= Math.floor(x) + 1; bx++)
        for (let by = Math.floor(y) - 1; by <= Math.floor(y) + 1; by++)
          for (const b of bins.get(`${bx},${by}`) ?? []) {
            const [u, v, w] = projected[b];
            if (b > a && Math.hypot(u - x, v - y) <= 0.5 && Math.abs(w - z) <= 0.01 * z)
              duplicate[b] = 1;
          }
    }
  }
  // Colors against the host undistortion of each reference's image.
  const gray = new Map();
  let matched = 0;
  for (let k = 0; k < count; k++) {
    const view = reference[k];
    if (!gray.has(view))
      gray.set(view, undistortReference(scene.images.get(view.id).rgba, view.camera));
    const image = gray.get(view);
    const [x, y] = projectPinhole(view, point(k)).map(Math.round);
    const [r, g, b] = [colors[3 * k], colors[3 * k + 1], colors[3 * k + 2]];
    let match = false;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const u = x + dx,
          v = y + dy;
        if (u < 0 || v < 0 || u >= MVS_WIDTH || v >= MVS_HEIGHT) continue;
        const expected = image[v * MVS_WIDTH + u] * 255;
        if (Math.abs(r - expected) <= 1 && g === r && b === r) match = true;
      }
    if (match) matched++;
  }
  const sorted = [...distances].sort((a, b) => a - b);
  angles.sort((a, b) => a - b);
  return {
    points: count,
    within1pct: distances.filter((d) => d <= 0.01).length / count,
    within5pct: distances.filter((d) => d <= 0.05).length / count,
    medianDistance: quantile(sorted, 0.5),
    p95Distance: quantile(sorted, 0.95),
    medianNormalDeg: quantile(angles, 0.5),
    p90NormalDeg: quantile(angles, 0.9),
    duplicates: duplicate.reduce((n, d) => n + d, 0) / count,
    colorMatch: matched / count,
  };
}

/**
 * The bytes of section 4.2 for a plan of N views with P pixels each: the formula of each stage,
 * and what the device holds at that stage with the small arrays (`mvsCameras`, `mvsSources` and
 * `mvsSeeds` in the depth stage, `mvsCameras`, the scan's block offsets and its totals in the
 * fusion stage).
 */
export function memoryFormula(plan, P) {
  const N = plan.packed.length,
    cameras = 4 * plan.cameras.length;
  const depthFormula = (16 * N + 4) * P,
    fusionFormula = (8 * N + 28) * P;
  return {
    depthFormula,
    depthStage:
      depthFormula + cameras + 4 * plan.sources.length + 4 * Math.max(3, plan.seeds.length),
    fusionFormula,
    fusionStage: fusionFormula + 8 * Math.max(1, Math.ceil(P / 64)) + 8 + cameras,
  };
}
