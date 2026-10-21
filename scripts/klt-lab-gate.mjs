/**
 * The lab gate of `docs/klt-tracking.md` 7.4: run the lab on one video with `?klt=0` and with
 * `?klt=1` on the same build, and print the rows of the table. Exit code 1 when a gated row fails.
 *
 *   node scripts/klt-lab-gate.mjs video.webm [extra query, for example "&gate=legacy&kltGate=1"]
 *   node scripts/klt-lab-gate.mjs --records off.json on.json
 *
 * The first form needs `npm run build && npm run build:website`, Chromium (Playwright) and a
 * GPU or SwiftShader, and takes minutes. It serves `website/dist`, loads `/lab/`, uploads the
 * video, presses "Run GPU SfM" and reads the run record from the page. It writes
 * `klt-gate-off.json` and `klt-gate-on.json` to the current directory. The second form reads two
 * records that exist already.
 */
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { kltGatePasses, kltGateRows, kltGateRun } from '../website/src/lib/klt-gate.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..'),
  dist = join(root, 'website', 'dist'),
  types = {
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.css': 'text/css',
    '.html': 'text/html',
    '.json': 'application/json',
    '.svg': 'image/svg+xml',
    '.wasm': 'application/wasm',
    '.woff2': 'font/woff2',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
  };

async function labRecord(video, query) {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    let file = join(dist, decodeURIComponent(url.pathname));
    if (url.pathname.endsWith('/')) file = join(file, 'index.html');
    if (!file.startsWith(dist)) return res.writeHead(403).end();
    try {
      res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' });
      res.end(await readFile(file));
    } catch {
      res.writeHead(404).end();
    }
  }).listen(0, '127.0.0.1');
  await new Promise((done) => server.once('listening', done));
  const browser = await chromium.launch({
    args: [
      '--enable-unsafe-webgpu',
      '--enable-unsafe-swiftshader',
      '--use-angle=swiftshader',
      '--use-vulkan=swiftshader',
      '--enable-features=Vulkan',
    ],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.goto(`http://127.0.0.1:${server.address().port}/lab/${query}`);
    await page.locator('input[type=file]').setInputFiles(video);
    await page.waitForFunction(() => document.querySelector('video')?.videoWidth > 0, null, {
      timeout: 120000,
      polling: 500,
    });
    await page.getByRole('button', { name: /Run GPU SfM/ }).click();
    await page.waitForFunction(
      () =>
        /completed|Reconstruction failed/.test(
          document.querySelector('.lab-status')?.textContent ?? '',
        ),
      null,
      { timeout: 5_400_000, polling: 5000 },
    );
    return JSON.parse(await page.locator('.run-record details pre').textContent());
  } finally {
    await browser.close();
    server.close();
  }
}

const args = process.argv.slice(2);
let off, on;
if (args[0] === '--records') {
  [off, on] = await Promise.all(
    args.slice(1, 3).map(async (f) => JSON.parse(await readFile(f, 'utf8'))),
  );
} else {
  const [video, extra = ''] = args;
  if (!video) throw new Error('Usage: see the header of this file');
  off = await labRecord(video, '?klt=0' + extra);
  await writeFile('klt-gate-off.json', JSON.stringify(off, null, 1));
  on = await labRecord(video, '?klt=1' + extra);
  await writeFile('klt-gate-on.json', JSON.stringify(on, null, 1));
}
const rows = kltGateRows(kltGateRun(off), kltGateRun(on));
for (const row of rows)
  console.log(
    `${row.pass === null ? 'info' : row.pass ? 'pass' : 'FAIL'}  ${row.name}: klt=0 ${row.off}, klt=1 ${row.on} (${row.rule})`,
  );
process.exit(kltGatePasses(rows) ? 0 : 1);
