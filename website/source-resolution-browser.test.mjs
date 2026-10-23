import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { fixtureVideo, imageFixture } from './video-fixture.mjs';

test('source decoding changes the actual analysis pixels from 640 to 1280 without fetching JPEG inputs', async () => {
  const clip = await fixtureVideo(
    await Promise.all([0, 3].map((i) => readFile(imageFixture('pyramid', i)))),
    1920,
    1080,
    2,
  );
  const bundle = await build({
    stdin: {
      contents: `import {decodeUploadedFrames} from './src/lib/sfm-video.ts';
    window.decode = async edge => {
      const frames=[]; const info={path:'main-thread'};
      for await(const frame of decodeUploadedFrames('/source.webm',new AbortController().signal,edge,1,info,'?decode=main'))
        frames.push({width:frame.width,height:frame.height,sourceWidth:frame.sourceWidth,sourceHeight:frame.sourceHeight,pixels:frame.pixels.length,time:frame.time});
      return {frames,path:info.path};
    };`,
      resolveDir: import.meta.dirname,
      loader: 'ts',
    },
    bundle: true,
    write: false,
    format: 'esm',
  });
  const requests = [];
  const server = createServer((req, res) => {
    requests.push(req.url);
    if (req.url === '/probe.js')
      return res
        .writeHead(200, { 'content-type': 'text/javascript' })
        .end(bundle.outputFiles[0].text);
    if (req.url === '/source.webm') {
      const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
      const start = range ? Number(range[1]) : 0,
        end = range?.[2] ? Math.min(Number(range[2]), clip.length - 1) : clip.length - 1;
      return res
        .writeHead(range ? 206 : 200, {
          'content-type': 'video/webm',
          'accept-ranges': 'bytes',
          ...(range ? { 'content-range': `bytes ${start}-${end}/${clip.length}` } : {}),
        })
        .end(clip.subarray(start, end + 1));
    }
    res.end('<script type="module" src="/probe.js"></script>');
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => !!window.decode);
    for (const edge of [640, 1280]) {
      const result = await page.evaluate((edge) => window.decode(edge), edge);
      assert.equal(result.frames.length, 2);
      assert.equal(result.path, 'main-thread');
      for (const frame of result.frames) {
        assert.equal(frame.width, edge);
        assert.equal(frame.height, (edge * 9) / 16);
        assert.equal(frame.pixels, ((edge * edge * 9) / 16) * 4);
        assert.equal(frame.sourceWidth, 1920);
        assert.equal(frame.sourceHeight, 1080);
      }
    }
    assert.ok(requests.some((path) => path === '/source.webm'));
    assert.ok(!requests.some((path) => /\.jpg/.test(path)));
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
