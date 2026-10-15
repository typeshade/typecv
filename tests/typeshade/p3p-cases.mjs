// Independent geometry and scalar validation shared by the CPU and browser tests.
export function p3pCases() {
  let state = 8642;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
  const angle = 0.09;
  const R = [Math.cos(angle), 0, Math.sin(angle), 0, 1, 0, -Math.sin(angle), 0, Math.cos(angle)];
  const apply = (p, t) =>
    t.map((v, i) => v + R[i * 3] * p[0] + R[i * 3 + 1] * p[1] + R[i * 3 + 2] * p[2]);
  const cases = [];
  for (let i = 0; i < 100; i++) {
    const scale = [0.01, 1, 100][i % 3];
    const world = Array.from({ length: 3 }, () => [
      (random() - 0.5) * 3 * scale,
      (random() - 0.5) * 2 * scale,
      (4 + random() * 5) * scale,
    ]);
    const t = [0.3 * scale, -0.02 * scale, 0.1 * scale];
    cases.push({
      name: `scene-${i}`,
      world,
      rays: world.map((p) => apply(p, t)),
      truth: [...R, ...t],
      status: 0,
    });
  }
  const symmetric = [
    [-1, 0, 5],
    [1, 0, 5],
    [0, Math.sqrt(3), 5],
  ];
  cases.push({
    name: 'symmetric',
    world: symmetric,
    rays: symmetric,
    truth: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    status: 0,
  });
  const base = cases[1];
  for (const scale of [1e-20, 1e8]) {
    cases.push({
      ...base,
      name: `scaled-bearings-${scale}`,
      rays: base.rays.map((p) => p.map((x) => x * scale)),
    });
  }
  const shift = [1e5, -2e4, 3e4];
  const shiftedTranslation = base.truth
    .slice(9)
    .map((t, i) => t - R[i * 3] * shift[0] - R[i * 3 + 1] * shift[1] - R[i * 3 + 2] * shift[2]);
  cases.push({
    ...base,
    name: 'translated-origin',
    world: base.world.map((p) => p.map((x, i) => x + shift[i])),
    truth: [...R, ...shiftedTranslation],
  });
  cases.push({
    name: 'collinear',
    world: [
      [0, 0, 5],
      [1, 0, 5],
      [2, 0, 5],
    ],
    rays: base.rays,
    status: 1,
  });
  cases.push({
    name: 'coincident-world',
    world: [base.world[0], base.world[0], base.world[2]],
    rays: base.rays,
    status: 1,
  });
  cases.push({
    name: 'repeated-rays',
    world: base.world,
    rays: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
    ],
    status: 2,
  });
  cases.push({
    name: 'zero-ray',
    world: base.world,
    rays: [[0, 0, 0], ...base.rays.slice(1)],
    status: 2,
  });
  const narrow = [
    [0.01, 0, 100],
    [-0.01, 0, 100],
    [0, 0.01, 100],
  ];
  cases.push({ name: 'narrow-cone', world: narrow, rays: narrow, status: 2 });
  cases.push({
    name: 'behind-camera',
    world: base.world,
    rays: base.rays.map((p) => p.map((x) => -x)),
    status: 4,
  });
  cases.push({
    name: 'extreme-side-ratio',
    world: [
      [0, 0, 5],
      [1e-9, 0, 5],
      [1, 1, 5],
    ],
    rays: base.rays,
    status: 6,
  });
  return cases;
}

export function packP3pCases(cases) {
  return Float64Array.from(cases.flatMap(({ world, rays }) => [...world.flat(), ...rays.flat()]));
}

export function p3pMetrics(cases, result) {
  const output = [];
  let maxRayError = 0,
    maxOrthogonalityError = 0,
    maxTruthError = 0;
  const failures = [];
  for (let h = 0; h < cases.length; h++) {
    const data = cases[h],
      count = result.counts[h],
      candidates = [];
    const sceneScale = Math.max(
      ...data.world.flatMap((p, i) =>
        data.world.slice(i + 1).map((q) => Math.hypot(...p.map((x, j) => x - q[j]))),
      ),
    );
    if (data.status !== undefined && result.status[h] !== data.status)
      failures.push(`${data.name}: status ${result.status[h]}, expected ${data.status}`);
    if (
      count > 4 ||
      (data.status === 0 && count < 1) ||
      (data.status !== undefined && data.status !== 0 && count !== 0)
    )
      failures.push(`${data.name}: count ${count}`);
    for (let k = 0; k < 4; k++) {
      const p = Array.from(result.poses.slice(h * 48 + k * 12, h * 48 + k * 12 + 12));
      if (result.valid[h * 4 + k] !== Number(k < count))
        failures.push(`${data.name}: validity ${k}`);
      if (k >= count) {
        if (p.some((v) => v !== 0)) failures.push(`${data.name}: stale slot ${k}`);
        continue;
      }
      if (!p.every(Number.isFinite)) failures.push(`${data.name}: nonfinite pose`);
      candidates.push(p);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const value =
            p[i * 3] * p[j * 3] +
            p[i * 3 + 1] * p[j * 3 + 1] +
            p[i * 3 + 2] * p[j * 3 + 2] -
            Number(i === j);
          maxOrthogonalityError = Math.max(maxOrthogonalityError, Math.abs(value));
        }
      }
      const determinant =
        p[0] * (p[4] * p[8] - p[5] * p[7]) -
        p[1] * (p[3] * p[8] - p[5] * p[6]) +
        p[2] * (p[3] * p[7] - p[4] * p[6]);
      if (Math.abs(determinant - 1) > 1e-7) failures.push(`${data.name}: improper rotation`);
      for (let i = 0; i < 3; i++) {
        const X = data.world[i],
          b = data.rays[i];
        const q = [0, 1, 2].map(
          (j) => p[j * 3] * X[0] + p[j * 3 + 1] * X[1] + p[j * 3 + 2] * X[2] + p[9 + j],
        );
        if (!(q[2] > 0)) failures.push(`${data.name}: negative depth`);
        const nq = Math.hypot(...q),
          nb = Math.hypot(...b);
        maxRayError = Math.max(maxRayError, Math.hypot(...q.map((x, j) => x / nq - b[j] / nb)));
      }
    }
    if (data.truth && candidates.length) {
      const best = Math.min(...candidates.map((p) => poseDifference(p, data.truth, sceneScale)));
      maxTruthError = Math.max(maxTruthError, best);
    }
    output.push({ name: data.name, sceneScale, candidates, status: result.status[h] });
  }
  return { output, failures, maxRayError, maxOrthogonalityError, maxTruthError };
}

export function poseDifference(a, b, scale) {
  return Math.max(
    Math.hypot(...a.slice(0, 9).map((v, j) => v - b[j])),
    Math.hypot(...a.slice(9).map((v, j) => (v - b[9 + j]) / scale)),
  );
}
