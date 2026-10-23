// Test-only image inputs stay outside the public site. Browser sample tests encode them as video.
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { videoIdOf } from '../src/reconstruction/identity.ts';

export const imageFixture = (name, frame) =>
  new URL(
    `../tests/fixtures/lab-images/${name}/frame-${String(frame).padStart(2, '0')}.jpg`,
    import.meta.url,
  );
export async function fixtureVideo(images, width = 640, height = 360, duration = images.length) {
  const work = await mkdtemp(join(tmpdir(), 'typecv-video-'));
  try {
    const parts = [];
    for (const [i, data] of images.entries()) {
      const file = join(work, `${i}.jpg`);
      await writeFile(file, data);
      parts.push(`file '${file}'`, `duration ${duration / images.length}`);
    }
    parts.push(`file '${join(work, `${images.length - 1}.jpg`)}'`);
    const list = join(work, 'frames.txt');
    await writeFile(list, parts.join('\n'));
    return execFileSync(
      'ffmpeg',
      [
        '-v',
        'error',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        list,
        '-t',
        String(duration),
        '-vf',
        `scale=${width}:${height},fps=1`,
        '-c:v',
        'libvpx-vp9',
        '-lossless',
        '1',
        '-g',
        '1',
        '-threads',
        '2',
        '-f',
        'webm',
        'pipe:1',
      ],
      { maxBuffer: 32 * 1024 ** 2 },
    );
  } finally {
    await rm(work, { recursive: true, force: true });
  }
}
export async function videoEntry(entry, clip, width = 640, height = 360) {
  return {
    ...entry,
    source: {
      file: 'source.mp4',
      bytes: clip.length,
      width,
      height,
      codec: 'vp9',
      videoId: await videoIdOf(new Blob([clip])),
    },
  };
}
export async function routeVideo(page, entry, clip, requests = []) {
  await page.route('**' + entry.dir + 'thumbnail.jpg', async (route) =>
    route.fulfill({ contentType: 'image/jpeg', body: await readFile(imageFixture('pyramid', 0)) }),
  );
  await page.route('**' + entry.dir + 'source.mp4', async (route) => {
    const header = route.request().headers().range;
    requests.push(header ?? null);
    const match = /^bytes=(\d+)-(\d*)$/.exec(header ?? '');
    const first = match ? Number(match[1]) : 0;
    const last = match?.[2] ? Math.min(Number(match[2]), clip.length - 1) : clip.length - 1;
    if (first >= clip.length)
      return route.fulfill({ status: 416, headers: { 'content-range': `bytes */${clip.length}` } });
    await route.fulfill({
      status: match ? 206 : 200,
      contentType: 'video/webm',
      headers: {
        'accept-ranges': 'bytes',
        ...(match ? { 'content-range': `bytes ${first}-${last}/${clip.length}` } : {}),
      },
      body: clip.subarray(first, last + 1),
    });
  });
}
export async function preparePyramid(page) {
  let frames = await Promise.all(
    Array.from({ length: 8 }, async (_, id) => ({
      id,
      time: [0, 1, 2, 4, 8, 12, 16, 20][id],
      width: 640,
      height: 360,
      image:
        'data:image/jpeg;base64,' +
        (await readFile(imageFixture('pyramid', id))).toString('base64'),
    })),
  );
  frames = await page.evaluate(async (frames) => {
    for (const frame of frames) {
      const bitmap = await createImageBitmap(await (await fetch(frame.image)).blob());
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
      frame.image = canvas.toDataURL('image/png');
    }
    return frames;
  }, frames);
  const bundle = {
    format: 'typecv-sfm-video-v1',
    source: { name: 'Pyramid regression fixture', width: 640, height: 360, duration: 21, fps: 1 },
    frames,
  };
  const button = page.locator('.lab-topbar .ant-btn');
  if ((await button.getAttribute('aria-expanded')) !== 'true') await button.click();
  await page.locator('input[name="file"]').setInputFiles({
    name: 'pyramid-fixture.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(bundle)),
  });
  await page.getByRole('button', { name: /Prepare analysis frames|프레임 JSON 불러오기/ }).click();
  await page.waitForFunction(() => document.querySelectorAll('.frame-grid img').length === 8);
}
