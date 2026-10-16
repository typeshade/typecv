import { copy } from './content.mjs';

const lang = document.documentElement.lang === 'ko' ? 'ko' : 'en',
  t = copy[lang];
const form = document.querySelector('#pair-form'),
  button = document.querySelector('#run');
const status = document.querySelector('#status'),
  results = document.querySelector('#results');
let busy = false,
  latest,
  current,
  displayRows;

function configuration() {
  const values = new FormData(form);
  const result = {
    backend: values.get('backend'),
    count: Number(values.get('count')),
    noise: Number(values.get('noise')),
    outliers: Number(values.get('outliers')),
  };
  if (
    !['cpu', 'webgpu'].includes(result.backend) ||
    ![64, 128, 256].includes(result.count) ||
    ![0, 0.12, 0.5].includes(result.noise) ||
    ![0, 20, 30].includes(result.outliers)
  )
    throw new RangeError('Unsupported demonstration settings');
  return result;
}

// Independent image formation supplies inputs, not an implementation of the estimator.
function fixture({ count, noise, outliers }) {
  let state = 71;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
  const first = { fx: 820, fy: 840, cx: 480, cy: 300, k1: 0.04, k2: -0.002 };
  const second = { fx: 850, fy: 790, cx: 480, cy: 300, k1: -0.025, k2: 0.001 };
  const c = Math.cos(0.11),
    s = Math.sin(0.11),
    R = [c, 0, s, 0, 1, 0, -s, 0, c],
    tr = [0.45, -0.03, 0.05];
  const project = (k, X) => {
    const x = X[0] / X[2],
      y = X[1] / X[2],
      r = x * x + y * y,
      d = 1 + k.k1 * r + k.k2 * r * r;
    return [k.fx * x * d + k.cx, k.fy * y * d + k.cy];
  };
  const pixels = new Float64Array(count * 4),
    clean = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const X = [(random() - 0.5) * 4, (random() - 0.5) * 2.5, 4 + random() * 5];
    const Y = tr.map((v, j) => v + R[3 * j] * X[0] + R[3 * j + 1] * X[1] + R[3 * j + 2] * X[2]);
    const row = [...project(first, X), ...project(second, Y)].map(
      (v) => v + (random() - 0.5) * noise,
    );
    clean[i] = random() >= outliers / 100 ? 1 : 0;
    if (!clean[i]) {
      row[2] = 120 + random() * 720;
      row[3] = 80 + random() * 440;
    }
    pixels.set(row, i * 4);
  }
  return { pixels, first, second, clean };
}

function draw(data, accepted) {
  displayRows = accepted ? [...accepted] : undefined;
  const style = getComputedStyle(document.documentElement);
  for (const [camera, id] of ['first-image', 'second-image'].entries()) {
    const canvas = document.getElementById(id),
      ctx = canvas.getContext('2d');
    ctx.fillStyle = style.getPropertyValue('--canvas').trim();
    ctx.fillRect(0, 0, 960, 600);
    ctx.strokeStyle = style.getPropertyValue('--grid').trim();
    ctx.lineWidth = 1;
    for (let x = 0; x <= 960; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 600);
      ctx.stroke();
    }
    for (let y = 0; y <= 600; y += 60) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(960, y);
      ctx.stroke();
    }
    for (let i = 0; i < data.clean.length; i++) {
      const x = data.pixels[4 * i + 2 * camera],
        y = data.pixels[4 * i + 2 * camera + 1];
      ctx.fillStyle = accepted ? (accepted.has(i) ? '#1677ff' : '#d45454') : '#8696aa';
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      if (i % 16 === 0) {
        ctx.font = '16px monospace';
        ctx.fillText(String(i), x + 9, y - 8);
      }
    }
  }
}

function reset() {
  if (busy) return;
  current = fixture(configuration());
  latest = undefined;
  draw(current);
  results.hidden = true;
  status.textContent = t.ready;
  status.dataset.state = 'ready';
}

async function verify() {
  if (busy) throw new Error('A verification is already running');
  const config = configuration();
  busy = true;
  button.textContent = t.running;
  for (const element of form.elements) element.disabled = true;
  results.hidden = true;
  status.textContent = t.running;
  status.dataset.state = 'running';
  let device;
  try {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const data = fixture(config);
    current = data;
    draw(data);
    let execute;
    if (config.backend === 'webgpu') {
      if (!navigator.gpu) throw new Error(t.noGpu);
      const api = await import('/runtime/gpu.js');
      device = await api.ShadeDevice.open();
      execute = (...args) => api.verifyImagePairGpu(device, ...args);
    } else {
      const api = await import('/runtime/core.js');
      execute = api.verifyImagePair;
    }
    const options = { iterations: 256, threshold: 0.001, seed: 71, refinementIterations: 0 };
    const started = performance.now();
    const pair = await execute(data.pixels, data.first, data.second, options);
    const elapsed = performance.now() - started;
    const accepted = new Set(pair.inlierRows),
      geometry = pair.geometry;
    const cleanCount = data.clean.reduce((n, v) => n + v, 0);
    const acceptedClean = [...accepted].filter((i) => data.clean[i]).length;
    draw(data, accepted);
    const details = pair.execution
      ? {
          backend: pair.execution.backend,
          dispatches: pair.execution.report.dispatches,
          computeSubmissions: pair.execution.report.submissions,
          applicationReadbackBytes: pair.execution.applicationReadbackBytes,
          cpuStages: pair.execution.cpuStages,
        }
      : { backend: 'cpu', implementation: 'Independent TypeCV CPU reference' };
    const entries = [
      [lang === 'ko' ? '역변환 가능' : 'Invertible', pair.invertibleCount],
      ['E inliers', accepted.size],
      ['H inliers', pair.homographyInlierCount],
      [lang === 'ko' ? '유효 점' : 'Quality points', geometry?.positiveDepthCount ?? 0],
      [
        lang === 'ko' ? '평균 오차 (px)' : 'Mean error (px)',
        geometry ? geometry.meanReprojectionErrorPx.toFixed(4) : '—',
      ],
      [
        lang === 'ko' ? '참 대응점 회수' : 'Clean matches recovered',
        `${acceptedClean} / ${cleanCount}`,
      ],
    ];
    const summary = document.querySelector('#summary');
    summary.replaceChildren();
    for (const [label, value] of entries) {
      const group = document.createElement('div'),
        name = document.createElement('dt'),
        number = document.createElement('dd');
      name.textContent = label;
      number.textContent = String(value);
      group.append(name, number);
      summary.append(group);
    }
    document.querySelector('#pose').textContent = geometry
      ? JSON.stringify(
          {
            R: [...geometry.R].map((v) => +v.toFixed(8)),
            t: [...geometry.t].map((v) => +v.toFixed(8)),
          },
          null,
          2,
        )
      : pair.reason;
    document.querySelector('#execution').textContent = JSON.stringify(
      { ...details, elapsedMs: +elapsed.toFixed(2) },
      null,
      2,
    );
    results.hidden = false;
    status.textContent = geometry
      ? `${details.backend} · ${accepted.size} / ${config.count} · ${elapsed.toFixed(1)} ms`
      : pair.reason;
    status.dataset.state = geometry ? 'success' : 'rejected';
    latest = {
      ...details,
      count: config.count,
      inliers: accepted.size,
      quality: geometry?.positiveDepthCount ?? 0,
      errorPx: geometry?.meanReprojectionErrorPx,
      acceptedClean,
      cleanCount,
      reason: pair.reason,
    };
    return latest;
  } catch (error) {
    latest = undefined;
    status.textContent = error instanceof Error ? error.message : String(error);
    status.dataset.state = 'error';
    throw error;
  } finally {
    await device?.dispose();
    busy = false;
    button.textContent = t.run + ' ↗';
    for (const element of form.elements) element.disabled = false;
  }
}
form.addEventListener('submit', (event) => {
  event.preventDefault();
  void verify().catch(() => {});
});
form.addEventListener('change', reset);
new MutationObserver(() => {
  if (current) draw(current, displayRows ? new Set(displayRows) : undefined);
}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
window.__typecvPlayground = {
  run: async (options = {}) => {
    for (const [key, value] of Object.entries(options)) {
      const field = form.elements.namedItem(key);
      if (field instanceof HTMLSelectElement) field.value = String(value);
    }
    return verify();
  },
  get result() {
    return latest;
  },
};
reset();
