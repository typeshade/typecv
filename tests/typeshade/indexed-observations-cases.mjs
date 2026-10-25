// Deterministic fixed-scene observations shared by numerical checks and the bounded benchmark.
function offsetArray(values) {
  const storage = new Float64Array(values.length + 4).fill(12345);
  storage.set(values, 2);
  return storage.subarray(2, 2 + values.length);
}
export function observationModel({
  points = 43,
  views = 3,
  depth = 10,
  baseline = 0.35,
  radial = true,
} = {}) {
  const camera = {
    fx: 730,
    fy: 710,
    cx: 312.5,
    cy: 247.25,
    k1: radial ? 0.08 : 0,
    k2: radial ? -0.01 : 0,
  };
  const registered = Array.from({ length: views }, (_, i) => ({
    id: 10 + 7 * i,
    camera: { ...camera },
    pose: { R: offsetArray([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: offsetArray([-i * baseline, -0, 0]) },
  }));
  const landmarks = Array.from({ length: points }, (_, p) => {
    const X = [-0.7 + 0.04 * p, 0.2 * Math.sin(p), depth + 0.03 * p];
    return {
      trackId: 1000 + p,
      xyz: offsetArray(X),
      observations: registered.map((v, i) => {
        const x = (X[0] - i * baseline) / X[2],
          y = X[1] / X[2],
          r2 = x * x + y * y;
        const d = 1 + camera.k1 * r2 + camera.k2 * r2 * r2;
        return {
          viewId: v.id,
          featureId: p,
          pixel: [
            camera.fx * x * d + camera.cx + ((p + i) % 4) * 0.03,
            camera.fy * y * d + camera.cy,
          ],
        };
      }),
    };
  });
  return { views: registered, landmarks };
}

export function observationCases() {
  const mixed = observationModel({ points: 5 });
  mixed.landmarks[1].observations[2].pixel[0] += 40;
  mixed.landmarks[2].xyz[2] = -10;
  const unused = observationModel({ points: 2 });
  unused.views.push({
    ...unused.views[0],
    id: 9999,
    camera: { ...unused.views[0].camera, fx: 1e8 },
  });
  const shuffled = observationModel({ points: 5, views: 4 });
  shuffled.views.reverse();
  shuffled.landmarks.reverse();
  for (const p of shuffled.landmarks) p.observations = [2, 0, 3, 1].map((i) => p.observations[i]);
  const calibrated = observationModel({ points: 5, views: 4 });
  calibrated.views.forEach((v, i) => {
    const angle = i * 0.025,
      c = Math.cos(angle),
      s = Math.sin(angle);
    v.pose.R.set([c, 0, s, 0, 1, 0, -s, 0, c]);
    Object.assign(v.camera, {
      fx: 600 + 41 * i,
      fy: 640 + 27 * i,
      cx: 301 + 3 * i,
      k1: -0.03 + i * 0.025,
      k2: 0.001 * i,
    });
  });
  for (const p of calibrated.landmarks)
    p.observations.forEach((o, i) => {
      const v = calibrated.views[i],
        k = v.camera,
        X = p.xyz,
        R = v.pose.R,
        t = v.pose.t;
      const q = [0, 1, 2].map(
        (row) => R[3 * row] * X[0] + R[3 * row + 1] * X[1] + R[3 * row + 2] * X[2] + t[row],
      );
      const x = q[0] / q[2],
        y = q[1] / q[2],
        r2 = x * x + y * y,
        d = 1 + k.k1 * r2 + k.k2 * r2 * r2;
      o.pixel = [k.fx * x * d + k.cx + 0.02, k.fy * y * d + k.cy];
    });
  return [
    { name: 'empty', model: { views: [], landmarks: [] } },
    {
      name: 'empty-guarded',
      model: { views: [], landmarks: [] },
      options: { maxRelativeDepthStdDev: 0.1 },
    },
    { name: 'unused-view-outside-gpu-bounds', model: unused },
    { name: 'radial-tail-129', model: observationModel() },
    {
      name: 'shuffled-view-point-observation-indices',
      model: shuffled,
      options: { maxRelativeDepthStdDev: 0.1 },
    },
    {
      name: 'different-calibration-and-rotated-poses',
      model: calibrated,
      options: { maxRelativeDepthStdDev: 0.1 },
    },
    {
      name: 'radial-tail-129-guarded',
      model: observationModel(),
      options: { maxRelativeDepthStdDev: 0.1 },
    },
    { name: 'mixed-inliers-behind-camera', model: mixed, options: { maxRelativeDepthStdDev: 0.1 } },
    {
      name: 'far-depth-rejected',
      model: observationModel({ points: 3, baseline: 0.0001, depth: 2000 }),
      options: { maxRelativeDepthStdDev: 0.01 },
    },
    {
      name: 'long-track-65',
      model: observationModel({ points: 1, views: 65, baseline: 0.01 }),
      options: { maxRelativeDepthStdDev: 0.1 },
    },
  ];
}
