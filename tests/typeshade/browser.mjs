// Consumer test harness patterned after TypeNN's tests/gpu/_browser.mjs (Apache-2.0).
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { chromium } from 'playwright';
export async function inBrowser(body, argument) {
  const root = resolve(import.meta.dirname, '../..');
  const server = createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/') {
      res.setHeader('content-type', 'text/html');
      res.end('<!doctype html><title>TypeCV WebGPU</title>');
      return;
    }
    if (pathname === '/favicon.ico') {
      res.statusCode = 204;
      res.end();
      return;
    }
    const file = resolve(root, '.' + pathname);
    if (
      ![resolve(root, 'dist') + sep, resolve(root, 'tests/typeshade') + sep].some((prefix) =>
        file.startsWith(prefix),
      )
    ) {
      res.statusCode = 404;
      res.end();
      return;
    }
    try {
      res.setHeader('content-type', 'text/javascript');
      res.end(await readFile(file));
    } catch {
      res.statusCode = 404;
      res.end();
    }
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
      args: process.env.TYPECV_HARDWARE_GPU
        ? ['--enable-unsafe-webgpu']
        : [
            '--enable-unsafe-webgpu',
            '--enable-unsafe-swiftshader',
            '--use-angle=swiftshader',
            '--use-vulkan=swiftshader',
            '--enable-features=Vulkan',
          ],
    });
    const page = await browser.newPage(),
      errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const result = await page.evaluate(body, argument);
    return { result, errors };
  } finally {
    await browser?.close();
    await new Promise((r) => server.close(r));
  }
}
