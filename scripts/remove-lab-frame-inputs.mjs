#!/usr/bin/env node
// Run after deploying the video-only catalog. Removes only historical frame-NN.jpg inputs.
// Original videos, display thumbnails, saved results and record attachments are outside this set.
import { readFileSync } from 'node:fs';
import { readManifest } from './lab-remote-samples.mjs';
import { accountId } from './export-run-records.mjs';
import { videoSampleTimes } from '../website/src/lib/sample-frames.ts';

export function legacyFrameKeys(manifest) {
  return [
    ...manifest.videos.flatMap((video) =>
      videoSampleTimes(video.duration).map(
        (_, i) => `sample/${video.id}/frame-${String(i).padStart(2, '0')}.jpg`,
      ),
    ),
    ...Array.from(
      { length: 8 },
      (_, i) => `sample/pyramid/frame-${String(i).padStart(2, '0')}.jpg`,
    ),
  ];
}
async function main() {
  const manifest = readManifest();
  const keys = legacyFrameKeys(manifest);
  if (!process.argv.includes('--delete')) {
    console.log(
      `${keys.length} legacy frame keys; pass --delete after deploying the video-only site.`,
    );
    return;
  }
  const account = accountId(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8'));
  const token = process.env.CLOUDFLARE_API_TOKEN;
  if (!account || !token) throw new Error('Cloudflare account and API token are required.');
  // A missing thumbnail must not be made harder to repair by removing its old frame file.
  for (const video of manifest.videos) {
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${account}/r2/buckets/${manifest.bucket}/objects/sample/${video.id}/thumbnail.jpg`,
      { headers: { authorization: `Bearer ${token}` } },
    );
    await response.arrayBuffer();
    if (!response.ok)
      throw new Error(`${video.id}: publish thumbnail.jpg before deleting old frame inputs.`);
  }
  let removed = 0;
  const queue = [...keys];
  await Promise.all(
    Array.from({ length: 8 }, async () => {
      for (;;) {
        const key = queue.pop();
        if (!key) return;
        const response = await fetch(
          `https://api.cloudflare.com/client/v4/accounts/${account}/r2/buckets/${manifest.bucket}/objects/${key}`,
          { method: 'DELETE', headers: { authorization: `Bearer ${token}` } },
        );
        if (!response.ok && response.status !== 404)
          throw new Error(`${key}: HTTP ${response.status}`);
        removed++;
      }
    }),
  );
  console.log(`Removed ${removed} legacy frame inputs.`);
}
if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) await main();
