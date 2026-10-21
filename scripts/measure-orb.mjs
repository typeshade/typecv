// Run the lab on a sample in headless Chromium and print the run record's numbers: the front end
// (SIFT or `?features=orb`) is chosen by the query. Needs a built site (`npm run build`, then
// `npm --prefix website run build`). SwiftShader unless TYPECV_HARDWARE_GPU=1.
//
//   node scripts/measure-orb.mjs --sample="Pyramid benchmark" --query="features=orb&octave=0&guided=0"
//   node scripts/measure-orb.mjs --sample="south-building (12 photos)" --query="octave=0&guided=0" --output=run.json
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { chromium } from 'playwright';

const args = Object.fromEntries(
  process.argv.slice(2).map((arg) => {
    const [key, ...rest] = arg.replace(/^--/, '').split('=');
    return [key, rest.join('=')];
  }),
);
const sample = args.sample ?? 'Pyramid benchmark',
  query = args.query ?? '',
  timeoutMs = Number(args.timeout ?? 1800) * 1000,
  directory = path.resolve(import.meta.dirname, '../website/dist'),
  mime = {
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.json': 'application/json',
    '.svg': 'image/svg+xml',
    '.woff2': 'font/woff2',
    '.txt': 'text/plain',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
  };
const stats = (values) =>
  values.length
    ? {
        mean: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
        min: Math.min(...values),
        max: Math.max(...values),
      }
    : null;
const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost'),
    pathname = decodeURIComponent(url.pathname),
    file = path.join(directory, pathname.endsWith('/') ? pathname + 'index.html' : pathname);
  if (!file.startsWith(directory + path.sep)) {
    res.writeHead(403);
    res.end();
    return;
  }
  try {
    res.setHeader('Content-Type', mime[path.extname(file)] ?? 'application/octet-stream');
    res.end(await readFile(file));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({
    executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
    args:
      process.env.TYPECV_HARDWARE_GPU === '1'
        ? ['--enable-unsafe-webgpu']
        : [
            '--enable-unsafe-webgpu',
            '--enable-unsafe-swiftshader',
            '--use-angle=swiftshader',
            '--use-vulkan=swiftshader',
            '--enable-features=Vulkan',
          ],
  });
  const page = await browser.newPage();
  page.on('pageerror', (e) => process.stderr.write(`[page error] ${e}\n`));
  await page.addInitScript(() => {
    window.__results = [];
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (data.type === 'result')
            window.__results.push({
              pairs: data.diagnostics?.pairVerification?.map((p) => ({
                first: p.first,
                second: p.second,
                matches: p.matches,
                inliers: p.inliers,
              })),
            });
        });
      }
    };
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/lab/${query ? '?' + query : ''}`);
  await page.waitForFunction(
    () => document.querySelector('.sfm-map .viewer')?.dataset.state === 'ready',
  );
  await page.getByRole('button', { name: /Choose a sample or my video/ }).click();
  await page
    .getByRole('button', { name: new RegExp('Load ' + sample.replace(/[()]/g, '\\$&')) })
    .click();
  await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length >= 2);
  await page.getByRole('button', { name: /Run GPU SfM/ }).click();
  await page.waitForFunction(
    () =>
      /completed|Reconstruction failed/.test(
        document.querySelector('.lab-status')?.textContent ?? '',
      ),
    undefined,
    { timeout: timeoutMs },
  );
  const status = await page.locator('.lab-status').textContent();
  const record = JSON.parse(await page.locator('.run-record details pre').textContent());
  const stages = JSON.parse(await page.locator('.run-details details pre').textContent());
  const pairs = (await page.evaluate(() => window.__results.at(-1)?.pairs)) ?? [];
  const total = (operation) =>
    stages.filter((s) => s.operation === operation).reduce((sum, s) => sum + (s.elapsedMs ?? 0), 0);
  const summary = {
    sample,
    query,
    status,
    frontEnd: record.options.frontEnd,
    features: record.options.features ?? 'sift',
    frames: record.input.frames,
    registeredViews: record.result.registeredViews,
    landmarks: record.result.landmarks,
    meanReprojectionErrorPx: record.result.meanReprojectionErrorPx,
    featuresPerFrame: record.result.featureCounts ?? record.result.features,
    matchedPairs: record.result.matchedPairs,
    verifiedPairs: pairs.length,
    matchesPerPair: stats(pairs.map((p) => p.matches)),
    inliersPerPair: stats(pairs.map((p) => p.inliers)),
    featureMs: total('features'),
    matchMs: total('matches'),
    guidedMs: total('guidedMatches'),
    totalMs: total('runTotal'),
  };
  console.log(JSON.stringify(summary, null, 2));
  if (args.output)
    await writeFile(args.output, JSON.stringify({ summary, pairs, record, stages }, null, 1));
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
