// Independent scalar image formation; no TypeShade or TypeCV numerical code.
export function poseFixture({
  count = 97,
  noise = 0.3,
  outliers = 0.25,
  planar = false,
  scale = 1,
  radial = true,
} = {}) {
  let seed = 731;
  const next = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 2 ** 32;
  };
  const a = 0.11,
    b = -0.03,
    ca = Math.cos(a),
    sa = Math.sin(a),
    cb = Math.cos(b),
    sb = Math.sin(b);
  const R = [ca, sa * sb, sa * cb, 0, cb, -sb, -sa, ca * sb, ca * cb],
    t = [0.3 * scale, -0.1 * scale, 0.2 * scale];
  const camera = {
    fx: 900,
    fy: 920,
    cx: 480,
    cy: 270,
    k1: radial ? -0.05 : 0,
    k2: radial ? 0.005 : 0,
  };
  const world = new Float64Array(count * 3),
    pixels = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    const X = [
      (next() - 0.5) * 4 * scale,
      (next() - 0.5) * 3 * scale,
      (planar ? 6 : 4 + next() * 5) * scale,
    ];
    const q = t.map((v, j) => v + R[j * 3] * X[0] + R[j * 3 + 1] * X[1] + R[j * 3 + 2] * X[2]);
    const x = q[0] / q[2],
      y = q[1] / q[2],
      r2 = x * x + y * y,
      d = 1 + camera.k1 * r2 + camera.k2 * r2 * r2;
    world.set(X, 3 * i);
    pixels.set(
      [
        camera.fx * x * d + camera.cx + (next() - 0.5) * noise * 2,
        camera.fy * y * d + camera.cy + (next() - 0.5) * noise * 2,
      ],
      2 * i,
    );
    if (i < Math.floor(count * outliers)) pixels.set([50 + i * 17, 100 + i * 7], 2 * i);
  }
  return {
    world,
    pixels,
    camera,
    truth: { R, t },
    scale,
    expectedInliers: count - Math.floor(count * outliers),
  };
}
export function poseErrors(result, data) {
  const dr = Math.hypot(...result.pose.R.map((v, i) => v - data.truth.R[i]));
  const dt = Math.hypot(...result.pose.t.map((v, i) => (v - data.truth.t[i]) / data.scale));
  return {
    rotationFrobenius: dr,
    translation: dt,
    inliers: result.inlierCount,
    errorPx: result.meanErrorPx,
    trials: result.iterations,
  };
}

export function triangulationFixture(count = 33) {
  const camera = { fx: 800, fy: 1000, cx: 400, cy: 300, k1: -0.08, k2: 0.01 };
  const poses = Array.from({ length: 5 }, (_, i) => ({
    R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    t: [-0.3 * i, -0.01 * i, 0],
  }));
  const points = Array.from({ length: count }, (_, i) => [
    ((i % 7) - 3) * 0.3,
    ((i % 5) - 2) * 0.2,
    5 + (i % 3) * 0.4,
  ]);
  const tracks = points.map((X, h) =>
    poses.map((pose, i) => {
      const q = X.map((v, j) => v + pose.t[j]),
        x = q[0] / q[2],
        y = q[1] / q[2],
        r2 = x * x + y * y,
        d = 1 + camera.k1 * r2 + camera.k2 * r2 * r2;
      let pixel = [camera.fx * x * d + camera.cx, camera.fy * y * d + camera.cy];
      if (h % 2 === 0 && i === 2) pixel = [pixel[0] + 80, pixel[1] - 50];
      return { camera, pose, pixel };
    }),
  );
  tracks.push([tracks[0][0]]);
  tracks.push([tracks[0][0], tracks[0][0]]);
  const low = tracks[1][0],
    X = points[1];
  const pose = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-1e-5, 0, 0] };
  const x = (X[0] - 1e-5) / X[2],
    y = X[1] / X[2],
    r2 = x * x + y * y,
    d = 1 + camera.k1 * r2 + camera.k2 * r2 * r2;
  tracks.push([
    low,
    { camera, pose, pixel: [camera.fx * x * d + camera.cx, camera.fy * y * d + camera.cy] },
  ]);
  return { tracks, points, count };
}

/**
 * The SHA-256 of a `triangulateTracksGpu` result, in hex: the statuses as `u32` bytes, then for
 * each track either `-` (not accepted) or the `f64` bytes of xyz, the angle and the error, and
 * the inlier bytes. It runs in Node and in the browser.
 */
export async function triangulationDigest(result) {
  const parts = [new Uint8Array(Uint32Array.from(result.status).buffer)];
  for (const track of result.tracks)
    if (!track) parts.push(new TextEncoder().encode('-'));
    else
      parts.push(
        new Uint8Array(Float64Array.of(...track.xyz, track.maxAngleDeg, track.meanErrorPx).buffer),
        Uint8Array.from(track.inliers),
      );
  const bytes = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let at = 0;
  for (const part of parts) {
    bytes.set(part, at);
    at += part.length;
  }
  const hash = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
  return [...hash].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function mappingFixture() {
  const data = poseFixture({ count: 40, noise: 0, outliers: 0, radial: false });
  const images = Array.from({ length: 4 }, (_, i) => ({
    id: 10 + i * 10,
    camera: data.camera,
    keypoints: Float64Array.from(
      Array.from({ length: 40 }, (_, j) => {
        const X = data.world.subarray(j * 3, j * 3 + 3);
        return [
          (data.camera.fx * (X[0] - 0.3 * i)) / X[2] + data.camera.cx,
          (data.camera.fy * X[1]) / X[2] + data.camera.cy,
        ];
      }).flat(),
    ),
  }));
  const pairs = [];
  for (let i = 0; i < 4; i++)
    for (let j = i + 1; j < 4; j++)
      pairs.push({
        first: images[i].id,
        second: images[j].id,
        matches: Uint32Array.from(Array.from({ length: 40 }, (_, k) => [k, k]).flat()),
      });
  return { images, pairs };
}

export function bundleFixture({ count = 40, views = 4, outlier = false, shift = [0, 0, 0] } = {}) {
  const data = poseFixture({ count, noise: 0, outliers: 0 });
  const model = { views: [], landmarks: [] },
    truth = [];
  for (let c = 0; c < views; c++) {
    const a = 0.02 * c,
      ca = Math.cos(a),
      sa = Math.sin(a),
      R = [ca, 0, sa, 0, 1, 0, -sa, 0, ca],
      t = [-0.32 * c, 0.02 * Math.sin(c), 0.015 * c];
    truth.push({ R, t });
    const perturbed = a + (c ? 0.004 * Math.sin(c) : 0),
      cp = Math.cos(perturbed),
      sp = Math.sin(perturbed),
      Rp = [cp, 0, sp, 0, 1, 0, -sp, 0, cp];
    const tp = t.map(
      (x, j) =>
        x +
        (c ? 0.015 * Math.sin(c + j) : 0) -
        Rp[j * 3] * shift[0] -
        Rp[j * 3 + 1] * shift[1] -
        Rp[j * 3 + 2] * shift[2],
    );
    model.views.push({
      id: 10 + 10 * c,
      camera: { ...data.camera },
      pose: { R: Float64Array.from(Rp), t: Float64Array.from(tp) },
    });
  }
  for (let p = 0; p < count; p++) {
    const X = Array.from(data.world.subarray(p * 3, p * 3 + 3));
    const observations = truth.map((pose, c) => {
      const q = pose.t.map(
          (v, j) => v + pose.R[j * 3] * X[0] + pose.R[j * 3 + 1] * X[1] + pose.R[j * 3 + 2] * X[2],
        ),
        x = q[0] / q[2],
        y = q[1] / q[2],
        r2 = x * x + y * y,
        d = 1 + data.camera.k1 * r2 + data.camera.k2 * r2 * r2;
      let pixel = [
        data.camera.fx * x * d + data.camera.cx + 0.1 * Math.sin(p * 3 + c),
        data.camera.fy * y * d + data.camera.cy + 0.1 * Math.cos(p + c * 3),
      ];
      if (outlier && p === 0 && c === 2) pixel = [pixel[0] + 40, pixel[1] - 20];
      return { viewId: model.views[c].id, featureId: p, pixel };
    });
    model.landmarks.push({
      trackId: p,
      xyz: Float64Array.from(X.map((x, j) => x + shift[j] + 0.035 * Math.sin(p + j))),
      observations,
    });
  }
  return model;
}
export function bundleMetrics(model, huber = 2) {
  const views = new Map(model.views.map((v) => [v.id, v]));
  let cost = 0,
    squares = 0,
    count = 0;
  for (const p of model.landmarks)
    for (const o of p.observations) {
      const { camera: k, pose } = views.get(o.viewId),
        X = p.xyz,
        q = [0, 1, 2].map(
          (j) =>
            pose.R[j * 3] * X[0] + pose.R[j * 3 + 1] * X[1] + pose.R[j * 3 + 2] * X[2] + pose.t[j],
        );
      if (!(q[2] > 0)) return { cost: Infinity, rmse: Infinity };
      const x = q[0] / q[2],
        y = q[1] / q[2],
        r2 = x * x + y * y,
        d = 1 + (k.k1 ?? 0) * r2 + (k.k2 ?? 0) * r2 * r2;
      const error = Math.hypot(k.fx * x * d + k.cx - o.pixel[0], k.fy * y * d + k.cy - o.pixel[1]);
      squares += error * error;
      cost += error <= huber ? error * error : 2 * huber * error - huber * huber;
      count++;
    }
  return { cost, rmse: Math.sqrt(squares / count), observations: count };
}
/**
 * A ring of `views` cameras around a cloud of `points` landmarks, each landmark seen by `track`
 * consecutive cameras, so that every camera holds about `points * track / views` observations and
 * the observation graph is one ring. For the camera product at a large camera count
 * (`docs/gpu-throughput.md`, item 5). Poses and points are perturbed deterministically.
 */
export function ringBundleFixture({ views = 320, points = 4800, track = 5 } = {}) {
  let seed = 977;
  const next = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 2 ** 32;
  };
  const camera = { fx: 900, fy: 920, cx: 480, cy: 270, k1: -0.05, k2: 0.005 };
  const rotation = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const truth = [],
    model = { views: [], landmarks: [] };
  for (let c = 0; c < views; c++) {
    const a = (2 * Math.PI * c) / views,
      R = rotation(a),
      t = [0, 0, 10];
    truth.push({ R, t });
    const Rp = rotation(a + (c ? 0.002 * Math.sin(c) : 0)),
      tp = t.map((x, j) => x + (c ? 0.01 * Math.sin(c + j) : 0));
    model.views.push({
      id: 1000 + c,
      camera: { ...camera },
      pose: { R: Float64Array.from(Rp), t: Float64Array.from(tp) },
    });
  }
  const features = new Array(views).fill(0);
  for (let p = 0; p < points; p++) {
    const X = [(next() - 0.5) * 3, (next() - 0.5) * 3, (next() - 0.5) * 3];
    const observations = [];
    for (let k = 0; k < track; k++) {
      const c = (p + k) % views,
        pose = truth[c],
        q = pose.t.map(
          (v, j) => v + pose.R[j * 3] * X[0] + pose.R[j * 3 + 1] * X[1] + pose.R[j * 3 + 2] * X[2],
        ),
        x = q[0] / q[2],
        y = q[1] / q[2],
        r2 = x * x + y * y,
        d = 1 + camera.k1 * r2 + camera.k2 * r2 * r2;
      observations.push({
        viewId: model.views[c].id,
        featureId: features[c]++,
        pixel: [
          camera.fx * x * d + camera.cx + 0.1 * Math.sin(p * 3 + c),
          camera.fy * y * d + camera.cy + 0.1 * Math.cos(p + c * 3),
        ],
      });
    }
    model.landmarks.push({
      trackId: p,
      xyz: Float64Array.from(X.map((x, j) => x + 0.02 * Math.sin(p + j))),
      observations,
    });
  }
  return model;
}
