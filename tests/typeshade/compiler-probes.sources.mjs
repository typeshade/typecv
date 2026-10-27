// The "use typeshade" programs of `compiler-probes.test.mjs`, one per probe.
//
// Each P source is the audit harness's source for its shape, cut to the one function under test
// (docs/audits/2026-10-26-kernel-style.md, section 0.1). Each S source is a shape of this suite.
// Each K source is a shape of the loops of src/mvs/gpu/cloud-knn.shade.ts, cut to one entry or
// one kernel function. MARK ends the line a refusal is expected on: the loop for TS8070, the
// helper's parameter for TS8020, and the call (P13) or the labelled statement (K3b) for TS8099. A
// source with no refusal has no mark.

export const MARK = '// probe';

export const SOURCES = {
  p1: `"use typeshade";
export function rgbaToGrayK(offset: u32, count: u32, rgba: array<u32>, gray: array<f32>): void {
  for (let i: u32 = offset; i < offset + count; i++) {
    const pixel: u32 = rgba[i];
    const r: f32 = f32(pixel & 255);
    const g: f32 = f32((pixel >> 8) & 255);
    const b: f32 = f32((pixel >> 16) & 255);
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.;
  }
}
`,

  p2: `"use typeshade";
declare const lut: storage<array<f32>>;
function lookup(k: u32): f32 {
  return lut[k];
}
export function mapK(xs: array<u32>, out: array<f32>): void {
  for (let i: u32 = 0; i < xs.length; i++) {
    out[i] = lookup(xs[i]);
  }
}
`,

  p3: `"use typeshade";
export function blurK(w: u32, h: u32, src: array<f32>, out: array<f32>): void {
  const at = (x: i32, y: i32): f32 => {
    const xx: u32 = u32(clamp(x, 0, i32(w) - 1));
    const yy: u32 = u32(clamp(y, 0, i32(h) - 1));
    return src[yy * w + xx];
  };
  for (let i: u32 = 0; i < w * h; i++) {
    const x: i32 = i32(i % w);
    const y: i32 = i32(i / w);
    out[i] = (at(x - 1, y) + 2. * at(x, y) + at(x + 1, y)) / 4.;
  }
}
`,

  p4: `"use typeshade";
function at(src: array<f32>, k: u32): f32 { // probe
  return src[k];
}
export function copyK(src: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = at(src, i);
  }
}
`,

  p5: `"use typeshade";
export function acceptA(state: array<f32>, costs: array<f32>): void {
  for (let i: u32 = 0; i < 1; i++) { // probe
    if (costs[0] < state[0]) {
      state[0] = costs[0];
      state[1] = state[1] * 0.5;
    }
  }
}
`,

  p5b: `"use typeshade";
export function acceptB(state: array<f32>, costs: array<f32>): void {
  for (let i: u32 = 0; i < 1; i++) {
    if (costs[0] < state[i * 2]) {
      state[i * 2] = costs[0];
      state[i * 2 + 1] = state[i * 2 + 1] * 0.5;
    }
  }
}
`,

  p6: `"use typeshade";
export function boundFromArray(counts: array<u32>, xs: array<f32>, out: array<f32>): void {
  const n: u32 = counts[0];
  for (let i: u32 = 0; i < n; i++) {
    out[i] = xs[i] * 2.;
  }
}
`,

  p7: `"use typeshade";
export function guardInLoop(counts: array<u32>, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    if (i >= counts[0]) continue;
    out[i] = xs[i] * 2.;
  }
}
`,

  p8: `"use typeshade";
let tile: workgroup<array<f32, 64>>;
export function tiled(xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < xs.length; i++) { // probe
    tile[i % 64] = xs[i];
    workgroupBarrier();
    out[i] = tile[(i + 1) % 64];
  }
}
`,

  p9: `"use typeshade";
export function strong(xs: array<f32>, t: f32, out: array<u32>): u32 {
  for (let i: u32 = 0; i < xs.length; i++) {
    if (xs[i] > t) out.push(i);
  }
  return out.length;
}
`,

  p10: `"use typeshade";
export function grid(w: u32, h: u32, out: array<f32>): void {
  for (let y: u32 = 0; y < h; y++) {
    for (let x: u32 = 0; x < w; x++) {
      out[y * w + x] = f32(x + y);
    }
  }
}
`,

  p16: `"use typeshade";
export function records(n: u32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) {
    for (let k: u32 = 0; k < 4; k++) {
      out[i * 4 + k] = xs[i] + f32(k);
    }
  }
}
`,

  p9b: `"use typeshade";
export function records2(n: u32, slots: u32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) { // probe
    for (let s: u32 = 0; s < slots; s++) {
      const o: u32 = i * slots + s;
      out[o * 4] = xs[i];
      out[o * 4 + 1] = f32(s);
    }
  }
}
`,

  p11: `"use typeshade";
export function scaleD(k: f64, xs: array<f64>, out: array<f64>): void {
  for (let i: u32 = 0; i < xs.length; i++) {
    out[i] = xs[i] * k;
  }
}
`,

  p18: `"use typeshade";
export function minScatter(keys: array<u32>, vals: array<u32>, best: array<u32>): void {
  for (let i: u32 = 0; i < keys.length; i++) { // probe
    best[keys[i]] = min(best[keys[i]], vals[i]);
  }
}
`,

  p19: `"use typeshade";
export function addScatter(keys: array<u32>, vals: array<f32>, sums: array<f32>): void {
  for (let i: u32 = 0; i < keys.length; i++) { // probe
    sums[keys[i]] += vals[i];
  }
}
`,

  p13: `"use typeshade";
export function first(xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < xs.length; i++) {
    out[i] = xs[i] * 2.;
  }
}
export function second(xs: array<f32>, out: array<f32>): void {
  first(xs, out); // probe
}
`,

  p14: `"use typeshade";
@workgroup_size([16])
export function scaled(n: u32, gain: f32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) {
    out[i] = xs[i] * gain;
  }
}
`,

  p15: `"use typeshade";
class Control {
  width: u32;
  height: u32;
  gain: f32;
}
export function gainK(c: Control, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < c.width * c.height; i++) {
    out[i] = xs[i] * c.gain;
  }
}
`,

  p20a: `"use typeshade";
export function countScatter(keys: array<u32>, bins: array<u32>): void {
  for (let i: u32 = 0; i < keys.length; i++) {
    bins[keys[i]] += 1;
  }
}
`,

  p20b: `"use typeshade";
export function strided(n: u32, stride: u32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) {
    out[stride * i] = xs[i];
  }
}
`,

  p20c: `"use typeshade";
export function stridedPlus(n: u32, stride: u32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) { // probe
    out[stride * i] = xs[i];
    out[stride * i + 1] = xs[i] + 1.;
  }
}
`,

  p20d: `"use typeshade";
export function offsetWrite(n: u32, base: u32, xs: array<f32>, out: array<f32>): void {
  for (let i: u32 = 0; i < n; i++) { // probe
    out[base + i] = xs[i];
  }
}
`,

  p20e: `"use typeshade";
export function argBest(xs: array<f32>): i32 {
  return xs.findIndex((x) => x > 0.5);
}
`,

  p20f: `"use typeshade";
export function sumSq(xs: array<f32>): f32 {
  return xs.reduce((a, x) => a + x * x, 0.);
}
`,

  p20g: `"use typeshade";
export function minScatter2(keys: array<u32>, vals: array<u32>, best: array<u32>): void {
  for (let i: u32 = 0; i < keys.length; i++) { // probe
    const k: u32 = keys[i];
    best[k] = min(best[k], vals[i]);
  }
}
`,

  p30a: `"use typeshade";
class Measurement {
  a: f64;
  b: f64;
  rho: f64;
  fx: f64;
}
class Linearization {
  ok: f64;
  u: f64;
  v: f64;
}
export function linearizeK(ms: array<Measurement>, out: array<Linearization>): void {
  for (let i: u32 = 0; i < ms.length; i++) {
    const m: Measurement = ms[i];
    out[i] = { ok: 0., u: 0., v: 0. };
    if (!(m.rho > 0.)) continue;
    out[i] = { ok: 1., u: m.a * m.fx, v: m.b * m.fx };
  }
}
`,

  p30b: `"use typeshade";
export function recordWrites(n: u32, xs: array<f64>, out: array<f64>): void {
  for (let i: u32 = 0; i < n; i++) { // probe
    for (let k: u32 = 0; k < 23; k++) out[i * 23 + k] = 0.;
    for (let row: u32 = 0; row < 2; row++) {
      out[i * 23 + 3 + row * 3] = xs[i];
    }
  }
}
`,

  // S1: a 64-lane tree sum with workgroup memory and barriers, and its barrier-free sibling, as
  // TypeCV writes an entry and its `...Oracle` twin.
  s1: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const sums: storage<array<f32>, 'read_write'>;
let tile: workgroup<array<f32, 64>>;
@compute([64])
export function blockSum(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const i: u32 = wid.x * 64 + lid.x;
  tile[lid.x] = 0.;
  if (i < xs.length) tile[lid.x] = xs[i];
  workgroupBarrier();
  for (let stride: u32 = 32; stride > 0; stride /= 2) {
    if (lid.x < stride) tile[lid.x] += tile[lid.x + stride];
    workgroupBarrier();
  }
  if (lid.x == 0) sums[wid.x] = tile[0];
}
@compute([1])
export function blockSumOracle(@builtin('global_invocation_id') gid: vec3u): void {
  let s: f32 = 0.;
  for (let j: u32 = 0; j < 64; j++) {
    const i: u32 = gid.x * 64 + j;
    if (i < xs.length) s += xs[i];
  }
  sums[gid.x] = s;
}
`,

  s3: `"use typeshade";
export function maxScatter(keys: array<u32>, vals: array<u32>, best: array<u32>): void {
  for (let i: u32 = 0; i < keys.length; i++) { // probe
    const k: u32 = keys[i];
    best[k] = max(best[k], vals[i]);
  }
}
`,

  // S4: the constants that hough.ts (lines 108-110) and tsdf.ts (line 121) copy today.
  s4: `"use typeshade";
export const MAX_SEGMENTS: u32 = 16;
export const SEGMENT_STRIDE: u32 = 1 + 4 * MAX_SEGMENTS;
export const TIE: f32 = 0.00001;
export const EMPTY: u32 = 0xffffffff;
`,

  // S6: the KLT maximum (kltEigenMax with kltEigenMaxFinal) as one kernel reduction.
  s6: `"use typeshade";
export function maxReduce(xs: array<f32>): f32 {
  let best: f32 = 0.;
  for (let i: u32 = 0; i < xs.length; i++) {
    best = max(best, xs[i]);
  }
  return best;
}
`,

  // K1 to K5: entries over the points `xs` with the kernel's clamp-and-mask lane. K1, K2 and K4
  // sum each point's three smallest squared distances to the other points, with the insertion of
  // cloud-knn.shade.ts (lines 116 to 126). K3, K3b and K5 find the first point within the
  // smallest ring of 1 to 4.
  k1: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const out: storage<array<f32>, 'read_write'>;
function insertSorted(@inout best: array<f32, 32>, k: u32, d2: f32): void {
  let s: u32 = k - 1;
  for (let step: u32 = 0; step < 31; step++) {
    if (s > 0 && best[s - 1] > d2) {
      best[s] = best[s - 1];
      s = s - 1;
    }
  }
  best[s] = d2;
}
@compute([64])
export function insertByRef(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  let best: array<f32, 32> = fill<f32, 32>(1e30);
  for (let j: u32 = 0; j < xs.length; j++) {
    const d: f32 = xs[j] - xs[i];
    if (j !== i && d * d < best[2]) insertSorted(best, 3, d * d);
  }
  if (gid.x < xs.length) out[gid.x] = best[0] + best[1] + best[2];
}
`,

  k2: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const bucketStart: storage<array<u32>>;
declare const out: storage<array<f32>, 'read_write'>;
function insertSorted(@inout best: array<f32, 32>, k: u32, d2: f32): void {
  let s: u32 = k - 1;
  for (let step: u32 = 0; step < 31; step++) {
    if (s > 0 && best[s - 1] > d2) {
      best[s] = best[s - 1];
      s = s - 1;
    }
  }
  best[s] = d2;
}
function scanBucket(@inout best: array<f32, 32>, k: u32, i: u32, bucket: u32): void {
  const end: u32 = bucketStart[bucket + 1];
  for (let j: u32 = bucketStart[bucket]; j < end; j++) {
    const d: f32 = xs[j] - xs[i];
    if (j !== i && d * d < best[k - 1]) insertSorted(best, k, d * d);
  }
}
@compute([64])
export function scanByRef(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  let best: array<f32, 32> = fill<f32, 32>(1e30);
  for (let b: u32 = 0; b < bucketStart.length - 1; b++) scanBucket(best, 3, i, b);
  if (gid.x < xs.length) out[gid.x] = best[0] + best[1] + best[2];
}
`,

  k3: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const out: storage<array<u32>, 'read_write'>;
@compute([64])
export function ringBreak(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  let found: bool = false;
  let ring: u32 = 0;
  let at: u32 = 0;
  for (let r: u32 = 1; r <= 4; r++) {
    for (let j: u32 = 0; j < xs.length; j++) {
      if (j !== i && abs(xs[j] - xs[i]) <= f32(r)) {
        found = true;
        ring = r;
        at = j;
        break;
      }
    }
    if (found) break;
  }
  if (gid.x < xs.length) out[gid.x] = found ? ring * 100 + at : 0;
}
`,

  k3b: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const out: storage<array<u32>, 'read_write'>;
@compute([64])
export function ringLabel(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  let ring: u32 = 0;
  let at: u32 = 0;
  rings: for (let r: u32 = 1; r <= 4; r++) { // probe
    for (let j: u32 = 0; j < xs.length; j++) {
      if (j !== i && abs(xs[j] - xs[i]) <= f32(r)) {
        ring = r;
        at = j;
        break rings;
      }
    }
  }
  if (gid.x < xs.length) out[gid.x] = ring * 100 + at;
}
`,

  k4: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const out: storage<array<f32>, 'read_write'>;
@compute([64])
export function insertWhile(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  const k: u32 = 3;
  let best: array<f32, 32> = fill<f32, 32>(1e30);
  for (let j: u32 = 0; j < xs.length; j++) {
    const d: f32 = xs[j] - xs[i];
    const d2: f32 = d * d;
    if (j !== i && d2 < best[k - 1]) {
      let s: u32 = k - 1;
      while (s > 0 && best[s - 1] > d2) {
        best[s] = best[s - 1];
        s = s - 1;
      }
      best[s] = d2;
    }
  }
  if (gid.x < xs.length) out[gid.x] = best[0] + best[1] + best[2];
}
`,

  k5: `"use typeshade";
declare const xs: storage<array<f32>>;
declare const out: storage<array<u32>, 'read_write'>;
@compute([64])
export function ringReturn(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = min(gid.x, xs.length - 1);
  for (let r: u32 = 1; r <= 4; r++) {
    for (let j: u32 = 0; j < xs.length; j++) {
      if (j !== i && abs(xs[j] - xs[i]) <= f32(r)) {
        if (gid.x < xs.length) out[gid.x] = r * 100 + j;
        return;
      }
    }
  }
  if (gid.x < xs.length) out[gid.x] = 0;
}
`,

  // K6 and K6b: the sum of K1 as a kernel function over the lanes of a visit order, written
  // through the order (K6) or at the lane (K6b).
  k6: `"use typeshade";
export function knnPermuted(k: u32, n: u32, xs: array<f32>, visitOrder: array<u32>, sums: array<f32>): void {
  for (let lane: u32 = 0; lane < visitOrder.length; lane++) { // probe
    const i: u32 = visitOrder[lane];
    let best: array<f32, 32> = fill<f32, 32>(1e30);
    for (let j: u32 = 0; j < n; j++) {
      const d: f32 = xs[j] - xs[i];
      const d2: f32 = d * d;
      if (j !== i && d2 < best[k - 1]) {
        let s: u32 = k - 1;
        for (let step: u32 = 0; step < 31; step++) {
          if (s > 0 && best[s - 1] > d2) {
            best[s] = best[s - 1];
            s = s - 1;
          }
        }
        best[s] = d2;
      }
    }
    let sum: f32 = 0.;
    for (let m: u32 = 0; m < k; m++) sum += best[m];
    sums[i] = sum;
  }
}
`,

  k6b: `"use typeshade";
export function knnByLane(k: u32, n: u32, xs: array<f32>, visitOrder: array<u32>, sums: array<f32>): void {
  for (let lane: u32 = 0; lane < visitOrder.length; lane++) {
    const i: u32 = visitOrder[lane];
    let best: array<f32, 32> = fill<f32, 32>(1e30);
    for (let j: u32 = 0; j < n; j++) {
      const d: f32 = xs[j] - xs[i];
      const d2: f32 = d * d;
      if (j !== i && d2 < best[k - 1]) {
        let s: u32 = k - 1;
        for (let step: u32 = 0; step < 31; step++) {
          if (s > 0 && best[s - 1] > d2) {
            best[s] = best[s - 1];
            s = s - 1;
          }
        }
        best[s] = d2;
      }
    }
    let sum: f32 = 0.;
    for (let m: u32 = 0; m < k; m++) sum += best[m];
    sums[lane] = sum;
  }
}
`,
};
