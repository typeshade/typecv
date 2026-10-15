"use typeshade";

class Bracket {
  a0: f64;
  a1: f64;
  a2: f64;
  a3: f64;
  a4: f64;
  lo: f64;
  hi: f64;
}

declare const problems: storage<array<Bracket>>;
declare const roots: storage<array<f64>, "read_write">;
declare const valid: storage<array<u32>, "read_write">;

function polynomial(p: Bracket, x: f64): f64 {
  return (((p.a4 * x + p.a3) * x + p.a2) * x + p.a1) * x + p.a0;
}

// A bracketed-root probe, not a complete P3P solver or an all-roots algorithm.
@compute([64])
export function bisect(@builtin("global_invocation_id") gid: vec3u): void {
  const i = gid.x;
  if (i >= problems.length) return;
  const p = problems[i];
  let lo: f64 = p.lo;
  let hi: f64 = p.hi;
  let left: f64 = polynomial(p, lo);
  const right: f64 = polynomial(p, hi);
  roots[i] = 0.;
  valid[i] = 0;
  if (!(hi > lo) || left * right >= 0.) return;
  for (let iteration: u32 = 0; iteration < 80; iteration++) {
    const mid: f64 = (lo + hi) * 0.5;
    const value: f64 = polynomial(p, mid);
    if (value === 0.) {
      lo = mid;
      hi = mid;
      break;
    }
    if (left * value < 0.) hi = mid;
    else {
      lo = mid;
      left = value;
    }
    if (hi - lo < 1e-13) break;
  }
  roots[i] = (lo + hi) * 0.5;
  valid[i] = 1;
}
