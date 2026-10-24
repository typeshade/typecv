export const W = 64,
  H = 32,
  P = W * H,
  z = 4,
  f = 80,
  cx = 31.5,
  cy = 15.5;
// Parallel calibrated cameras. Finite objects move by exactly ten pixels; world directions do
// not move. No ground-truth depths are estimated from the implementation under test.
export function fixture(kind, boundaryAware = false, minFiniteGain = 0) {
  const centers = [0, 0.5, -0.5],
    cameras = new Float32Array(72),
    gray = new Float32Array(3 * P);
  const texture = (x, y) =>
    0.45 +
    0.15 * Math.sin(x * 0.51) +
    0.12 * Math.cos(y * 0.37) +
    0.1 * Math.sin(x * 0.23 + y * 0.31);
  for (let s = 0; s < 3; s++) {
    const c = centers[s];
    cameras.set(
      [1, 0, 0, 0, 1, 0, 0, 0, 1, -c, 0, 0, f, f, cx, cy, W, H, 1, 40, c, 0, 0, s],
      24 * s,
    );
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let value;
        if (kind === 'finite') value = texture(x + (f * c) / z, y);
        else if (kind === 'direction')
          value = 0.5 + 0.17 * Math.sin((x * 2 * Math.PI) / 10) + 0.13 * Math.cos(y * 0.37);
        else {
          const rod = 32 - (f * c) / z;
          value =
            x === rod
              ? 0.2 + 0.04 * Math.sin(y * 0.7)
              : 0.9 + 0.015 * Math.sin(x * 0.4) + 0.015 * Math.cos(y * 0.55);
        }
        gray[s * P + y * W + x] = value;
      }
  }
  const axis = boundaryAware ? 5 : 6;
  return {
    mvsControl: {
      width: W,
      height: H,
      reference: 0,
      state: 0,
      color: 0,
      iteration: 1,
      seed: 0,
      bestSources: 2,
      windowRadius: 5,
      windowStep: 2,
      axisSamples: axis,
      minSamples: Math.ceil((axis * axis) / 4),
      bilateral: Number(boundaryAware),
      boundaryAware: Number(boundaryAware),
      minFiniteGain,
      spatialFalloff: 1 / 50,
      colorFalloff: 50,
      minAngleCos: Math.cos(Math.PI / 180),
      perturbDepth: 0.2,
      perturbAngle: 0.2,
      geometric: 1,
      geometricWeight: 0.3,
      geometricMaxCost: 3,
      view: 0,
      relativePoseCache: 0,
    },
    mvsCameras: cameras,
    mvsGray: gray,
    mvsSources: Uint32Array.from([2, 1, 2, 0, 0, 0, 0]),
    mvsDepth: new Float32Array(3 * P).fill(z),
    mvsGeoDepth: new Float32Array(P).fill(z),
    mvsNormal: new Uint32Array(P).fill(0x7fff7fff),
    mvsCost: new Float32Array(P).fill(2),
    mvsPartial: new Float32Array(3),
    mvsRange: { first: 0, end: 1 },
    filterControl: {
      maxCost: 0.9,
      minConsistent: 2,
      maxError: 1,
      minAngleCos: Math.cos((3 * Math.PI) / 180),
      minConfidence: 0,
      writeConfidence: 1,
    },
    streamControl: { phase: 5, start: 1, first: 0, stride: 3, costs: 2, sources: 2 },
  };
}
export const bind = (m, b) => {
  for (const [k, v] of Object.entries(b)) m.setBinding(k, v);
};
