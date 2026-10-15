import { reconstructTwoView } from 'typecv';

const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const angle = 0.07, c = Math.cos(angle), s = Math.sin(angle);
const matches = new Float64Array(80 * 4);
const pixel = ([x, y, z]) => [K.fx * x / z + K.cx, K.fy * y / z + K.cy];
for (let i = 0; i < 80; i++) {
  const x = ((i * 17) % 37) / 10 - 1.8;
  const y = ((i * 11) % 29) / 14 - 1;
  const z = 4 + ((i * 7) % 31) / 8;
  matches.set([...pixel([x, y, z]), ...pixel([c*x+s*z+0.4, y+0.02, -s*x+c*z+0.03])], i*4);
}
const result = reconstructTwoView(matches, K, K, { seed: 42 });
console.log(JSON.stringify({
  acceptedPoints: result.positiveDepthCount,
  meanReprojectionErrorPx: result.meanReprojectionErrorPx,
  translationDirection: Array.from(result.t),
  diagnostics: result.diagnostics,
}, null, 2));
