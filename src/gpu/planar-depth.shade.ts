'use typeshade';

// Local inverse-depth planes propose holes only; two frozen source maps must verify each fill.
class PlanarControl {
  width: u32;
  height: u32;
  tilesX: u32;
  tilesY: u32;
  fx: f32;
  fy: f32;
  cx: f32;
  cy: f32;
  k1: f32;
  k2: f32;
  radialLimit: f32;
  r0: f32;
  r1: f32;
  r2: f32;
  r3: f32;
  r4: f32;
  r5: f32;
  r6: f32;
  r7: f32;
  r8: f32;
  centerX: f32;
  centerY: f32;
  centerZ: f32;
}
class PlanarSource {
  group: u32;
  fx: f32;
  fy: f32;
  cx: f32;
  cy: f32;
  r0: f32;
  r1: f32;
  r2: f32;
  r3: f32;
  r4: f32;
  r5: f32;
  r6: f32;
  r7: f32;
  r8: f32;
  centerX: f32;
  centerY: f32;
  centerZ: f32;
}
class PlanarRange {
  first: u32;
  end: u32;
}
declare const planar: uniform<PlanarControl>;
declare const planarSource: uniform<PlanarSource>;
declare const planarRange: uniform<PlanarRange>;
declare const planarRgba: storage<array<u32>>;
declare const planarDepth: storage<array<f32>>;
declare const planarNormal: storage<array<u32>>;
declare const planarConfidence: storage<array<f32>>;
declare const planarPlanes: storage<array<f32>, 'read_write'>;
declare const planarCandidate: storage<array<f32>, 'read_write'>;
declare const planarCandidateNormal: storage<array<u32>, 'read_write'>;
declare const planarCandidateConfidence: storage<array<f32>, 'read_write'>;
declare const planarSupport: storage<array<u32>, 'read_write'>;
declare const planarSupportGroup: storage<array<u32>, 'read_write'>;
declare const planarSourceDepth: storage<array<f32>>;
declare const planarSourceConfidence: storage<array<f32>>;
const TILE: u32 = 16;
const WORDS: u32 = 14;

function rgb(i: u32): vec3 {
  const c: u32 = planarRgba[i];
  return vec3(f32(c & 255), f32((c >> 8) & 255), f32((c >> 16) & 255)) / 255;
}
/** Depth coordinates are undistorted; sample the original colour through its radial model. */
function colorAt(x: u32, y: u32): vec3 {
  const u: f32 = (f32(x) - planar.cx) / planar.fx;
  const v: f32 = (f32(y) - planar.cy) / planar.fy;
  const r2: f32 = u * u + v * v;
  if (!(r2 < planar.radialLimit)) return vec3(-1);
  const d: f32 = 1 + planar.k1 * r2 + planar.k2 * r2 * r2;
  const sx: f32 = planar.fx * u * d + planar.cx;
  const sy: f32 = planar.fy * v * d + planar.cy;
  if (!(sx >= 0 && sy >= 0 && sx <= f32(planar.width - 1) && sy <= f32(planar.height - 1)))
    return vec3(-1);
  const ix: u32 = u32(floor(sx));
  const iy: u32 = u32(floor(sy));
  const jx: u32 = min(ix + 1, planar.width - 1);
  const jy: u32 = min(iy + 1, planar.height - 1);
  return mix(
    mix(rgb(iy * planar.width + ix), rgb(iy * planar.width + jx), sx - f32(ix)),
    mix(rgb(jy * planar.width + ix), rgb(jy * planar.width + jx), sx - f32(ix)),
    sy - f32(iy),
  );
}
function packNormal(n: vec3): u32 {
  const s: f32 = abs(n.x) + abs(n.y) + abs(n.z);
  let p: vec2 = vec2(n.x, n.y) / s;
  if (n.z < 0) p = vec2((1 - abs(p.y)) * (p.x >= 0 ? 1 : -1), (1 - abs(p.x)) * (p.y >= 0 ? 1 : -1));
  return pack2x16snorm(p);
}

@compute([64])
export function planarFit(@builtin('global_invocation_id') gid: vec3u): void {
  const t: u32 = planarRange.first + gid.x;
  if (t >= planarRange.end) return;
  const base: u32 = t * WORDS;
  planarPlanes[base + 3] = 0;
  const x0: u32 = (t % planar.tilesX) * TILE;
  const y0: u32 = (t / planar.tilesX) * TILE;
  const x1: u32 = min(x0 + TILE, planar.width);
  const y1: u32 = min(y0 + TILE, planar.height);
  let n: f32 = 0;
  let su: f32 = 0;
  let sv: f32 = 0;
  let suu: f32 = 0;
  let suv: f32 = 0;
  let svv: f32 = 0;
  let sq: f32 = 0;
  let suq: f32 = 0;
  let svq: f32 = 0;
  let confidence: f32 = 0;
  let color: vec3 = vec3(0);
  let loX: f32 = f32(x1);
  let loY: f32 = f32(y1);
  let hiX: f32 = f32(x0);
  let hiY: f32 = f32(y0);
  let loD: f32 = 3e38;
  let hiD: f32 = 0;
  for (let y: u32 = y0; y < y1; y++)
    for (let x: u32 = x0; x < x1; x++) {
      const i: u32 = y * planar.width + x;
      const d: f32 = planarDepth[i];
      const c: vec3 = colorAt(x, y);
      if (!(d > 0 && planarConfidence[i] > 0 && c.x >= 0)) continue;
      const u: f32 = f32(x - x0) / 16 - 0.5;
      const v: f32 = f32(y - y0) / 16 - 0.5;
      const q: f32 = 1 / d;
      n += 1;
      su += u;
      sv += v;
      suu += u * u;
      suv += u * v;
      svv += v * v;
      sq += q;
      suq += u * q;
      svq += v * q;
      confidence += planarConfidence[i];
      color += c;
      loX = min(loX, f32(x));
      loY = min(loY, f32(y));
      hiX = max(hiX, f32(x));
      hiY = max(hiY, f32(y));
      loD = min(loD, d);
      hiD = max(hiD, d);
    }
  if (n < max(12, 0.3 * f32((x1 - x0) * (y1 - y0)))) return;
  const a00: f32 = svv * n - sv * sv;
  const a01: f32 = su * sv - suv * n;
  const a02: f32 = suv * sv - su * svv;
  const a11: f32 = suu * n - su * su;
  const a12: f32 = suv * su - suu * sv;
  const a22: f32 = suu * svv - suv * suv;
  const det: f32 = suu * a00 + suv * a01 + su * a02;
  if (!(det > 0.000001 * n * n * n)) return;
  const coeff: vec3 =
    vec3(
      a00 * suq + a01 * svq + a02 * sq,
      a01 * suq + a11 * svq + a12 * sq,
      a02 * suq + a12 * svq + a22 * sq,
    ) / det;
  for (let y: u32 = y0; y < y1; y++)
    for (let x: u32 = x0; x < x1; x++) {
      const i: u32 = y * planar.width + x;
      const d: f32 = planarDepth[i];
      if (!(d > 0 && planarConfidence[i] > 0)) continue;
      const q: f32 =
        coeff.x * (f32(x - x0) / 16 - 0.5) + coeff.y * (f32(y - y0) / 16 - 0.5) + coeff.z;
      if (!(q > 0) || abs(q * d - 1) > 0.02) return;
    }
  planarPlanes[base] = coeff.x;
  planarPlanes[base + 1] = coeff.y;
  planarPlanes[base + 2] = coeff.z;
  planarPlanes[base + 3] = 1;
  planarPlanes[base + 4] = min(0.25, confidence / n);
  planarPlanes[base + 5] = color.x / n;
  planarPlanes[base + 6] = color.y / n;
  planarPlanes[base + 7] = color.z / n;
  planarPlanes[base + 8] = loX;
  planarPlanes[base + 9] = loY;
  planarPlanes[base + 10] = hiX;
  planarPlanes[base + 11] = hiY;
  planarPlanes[base + 12] = loD;
  planarPlanes[base + 13] = hiD;
}

@compute([64])
export function planarPropose(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = planarRange.first + gid.x;
  if (i >= planarRange.end) return;
  planarCandidate[i] = 0;
  if (planarDepth[i] > 0) return;
  const x: u32 = i % planar.width;
  const y: u32 = i / planar.width;
  const t: u32 = (y / TILE) * planar.tilesX + x / TILE;
  const b: u32 = t * WORDS;
  if (planarPlanes[b + 3] === 0) return;
  if (
    f32(x) < planarPlanes[b + 8] ||
    f32(y) < planarPlanes[b + 9] ||
    f32(x) > planarPlanes[b + 10] ||
    f32(y) > planarPlanes[b + 11]
  )
    return;
  const c: vec3 =
    colorAt(x, y) - vec3(planarPlanes[b + 5], planarPlanes[b + 6], planarPlanes[b + 7]);
  if (max(abs(c.x), max(abs(c.y), abs(c.z))) > 0.1) return;
  const a: f32 = planarPlanes[b];
  const v: f32 = planarPlanes[b + 1];
  const z: f32 = planarPlanes[b + 2];
  const ox: f32 = f32((x / TILE) * TILE) + 8;
  const oy: f32 = f32((y / TILE) * TILE) + 8;
  const q: f32 = (a * (f32(x) - ox)) / 16 + (v * (f32(y) - oy)) / 16 + z;
  if (!(q > 0)) return;
  const d: f32 = 1 / q;
  if (d < 0.98 * planarPlanes[b + 12] || d > 1.02 * planarPlanes[b + 13]) return;
  const n: vec3 = -normalize(
    vec3(
      (a * planar.fx) / 16,
      (v * planar.fy) / 16,
      z + (a * (planar.cx - ox)) / 16 + (v * (planar.cy - oy)) / 16,
    ),
  );
  planarCandidate[i] = d;
  planarCandidateNormal[i] = packNormal(n);
  planarCandidateConfidence[i] = planarPlanes[b + 4];
}

function referenceCamera(X: vec3): vec3 {
  const p: vec3 = X - vec3(planar.centerX, planar.centerY, planar.centerZ);
  return vec3(
    dot(vec3(planar.r0, planar.r1, planar.r2), p),
    dot(vec3(planar.r3, planar.r4, planar.r5), p),
    dot(vec3(planar.r6, planar.r7, planar.r8), p),
  );
}
@compute([64])
export function planarVerify(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = planarRange.first + gid.x;
  if (i >= planarRange.end || !(planarCandidate[i] > 0)) return;
  const x: f32 = f32(i % planar.width);
  const y: f32 = f32(i / planar.width);
  const ray: vec3 =
    vec3((x - planar.cx) / planar.fx, (y - planar.cy) / planar.fy, 1) * planarCandidate[i];
  const X: vec3 =
    vec3(
      planar.r0 * ray.x + planar.r3 * ray.y + planar.r6 * ray.z,
      planar.r1 * ray.x + planar.r4 * ray.y + planar.r7 * ray.z,
      planar.r2 * ray.x + planar.r5 * ray.y + planar.r8 * ray.z,
    ) + vec3(planar.centerX, planar.centerY, planar.centerZ);
  const p: vec3 = X - vec3(planarSource.centerX, planarSource.centerY, planarSource.centerZ);
  const s: vec3 = vec3(
    dot(vec3(planarSource.r0, planarSource.r1, planarSource.r2), p),
    dot(vec3(planarSource.r3, planarSource.r4, planarSource.r5), p),
    dot(vec3(planarSource.r6, planarSource.r7, planarSource.r8), p),
  );
  if (!(s.z > 0)) return;
  const u: f32 = (planarSource.fx * s.x) / s.z + planarSource.cx;
  const v: f32 = (planarSource.fy * s.y) / s.z + planarSource.cy;
  if (!(u >= 0 && v >= 0 && u <= f32(planar.width - 1) && v <= f32(planar.height - 1))) return;
  const ix: u32 = u32(floor(u + 0.5));
  const iy: u32 = u32(floor(v + 0.5));
  const j: u32 = iy * planar.width + ix;
  const d: f32 = planarSourceDepth[j];
  if (!(d > 0 && planarSourceConfidence[j] > 0) || abs(d - s.z) > 0.01 * max(d, s.z)) return;
  const r: vec3 =
    vec3(
      (f32(ix) - planarSource.cx) / planarSource.fx,
      (f32(iy) - planarSource.cy) / planarSource.fy,
      1,
    ) * d;
  const Q: vec3 =
    vec3(
      planarSource.r0 * r.x + planarSource.r3 * r.y + planarSource.r6 * r.z,
      planarSource.r1 * r.x + planarSource.r4 * r.y + planarSource.r7 * r.z,
      planarSource.r2 * r.x + planarSource.r5 * r.y + planarSource.r8 * r.z,
    ) + vec3(planarSource.centerX, planarSource.centerY, planarSource.centerZ);
  const b: vec3 = referenceCamera(Q);
  if (!(b.z > 0)) return;
  const error: vec2 = vec2(
    (planar.fx * b.x) / b.z + planar.cx - x,
    (planar.fy * b.y) / b.z + planar.cy - y,
  );
  if (length(error) <= 1 && planarSupportGroup[i] !== planarSource.group) {
    planarSupport[i] += 1;
    planarSupportGroup[i] = planarSource.group;
  }
}
@compute([64])
export function planarFinish(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = planarRange.first + gid.x;
  if (i >= planarRange.end) return;
  if (planarDepth[i] > 0) {
    planarCandidate[i] = planarDepth[i];
    planarCandidateNormal[i] = planarNormal[i];
    planarCandidateConfidence[i] = planarConfidence[i];
  } else if (planarSupport[i] < 2) {
    planarCandidate[i] = 0;
    planarCandidateNormal[i] = 0;
    planarCandidateConfidence[i] = 0;
  }
}
