/**
 * The clouds of `docs/dense-mvs.md`, section 3.12: a jittered plane with scattered outliers and
 * two far singletons (the rings and the brute-force fallback of the kNN kernel), plus a blob with
 * repeated points. All values come from a seeded generator, so the browser and Node build the
 * same cloud.
 */
function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function cleanCloud(seed = 17) {
  const r = random(seed);
  const points = [];
  const spacing = 0.01;
  for (let y = 0; y < 50; y++)
    for (let x = 0; x < 60; x++)
      points.push([
        x * spacing + (r() - 0.5) * 0.2 * spacing,
        y * spacing + (r() - 0.5) * 0.2 * spacing,
        (r() - 0.5) * 0.2 * spacing,
      ]);
  const planeCount = points.length;
  for (let o = 0; o < 40; o++) points.push([r() * 0.6, r() * 0.5, 0.1 + r() * 0.4]);
  points.push([20, 20, 20], [-15, 8, 3]);
  // A small separate blob of 30 points, 1.5 units away.
  for (let b = 0; b < 30; b++)
    points.push([1.5 + (r() - 0.5) * 0.03, 0.2 + (r() - 0.5) * 0.03, (r() - 0.5) * 0.03]);
  const positions = new Float32Array(points.length * 3);
  points.forEach((p, i) => positions.set(p, 3 * i));
  const colors = new Uint8Array(points.length * 3);
  for (let i = 0; i < colors.length; i++) colors[i] = (i * 37) % 256;
  return { cloud: { count: points.length, positions, colors }, planeCount };
}
