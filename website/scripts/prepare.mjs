import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { filesWithExtension, markdownFiles } from './docs.mjs';

const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(website, '..');
const out = path.join(website, 'public/runtime');
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
await cp(path.join(root, 'dist/gpu.js'), path.join(out, 'gpu.js'));
await cp(path.join(root, 'dist/gpu'), path.join(out, 'gpu'), { recursive: true });
// The multi-video job of the lab's worker (docs/multi-video.md step 6) loads the internal entry,
// its complete module tree (including shared and lazy chunks), and the 32-word vocabulary
// of the global descriptor. Keep relative imports intact beside the internal entry.
await cp(path.join(root, 'dist/internal'), path.join(out, 'internal'), { recursive: true });
await mkdir(path.join(out, 'vocabulary'), { recursive: true });
await cp(
  path.join(root, 'src/gpu/vocabulary/rootsift-32-v1.bin'),
  path.join(out, 'vocabulary/rootsift-32-v1.bin'),
);
await cp(path.join(root, 'LICENSE'), path.join(out, 'LICENSE-TypeCV'));
await cp(path.join(root, 'node_modules/typeshade/LICENSE'), path.join(out, 'LICENSE-TypeShade'));
await build({
  entryPoints: [path.join(root, 'dist/index.js')],
  outfile: path.join(out, 'core.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  minify: true,
  legalComments: 'eof',
});
await build({
  entryPoints: [path.join(website, 'playground.mjs')],
  outfile: path.join(out, 'playground.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  external: ['/runtime/*'],
});
const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
const docFiles = await markdownFiles(path.join(root, 'docs'));
for (const file of await filesWithExtension(path.join(root, 'docs'), '.json')) {
  const destination = path.join(
    website,
    'public/docs',
    path.relative(path.join(root, 'docs'), file),
  );
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(file, destination);
}
const routes = [
  '/',
  '/ko/',
  '/playground/',
  '/ko/playground/',
  '/lab/',
  '/ko/lab/',
  '/lab/probe/',
  '/ko/lab/probe/',
  '/lab/check/',
  '/ko/lab/check/',
  '/404.html',
  ...docFiles.flatMap((file) => {
    const rel = path
      .relative(path.join(root, 'docs'), file)
      .replaceAll(path.sep, '/')
      .replace(/\/README\.md$/, '')
      .replace(/\.md$/, '');
    const route = rel === 'README' ? '/docs/' : `/docs/${rel}/`;
    return route === '/404.html' ? [route] : [route, `/ko${route}`];
  }),
];
await writeFile(
  path.join(website, 'public/version.json'),
  JSON.stringify(
    {
      project: 'TypeCV',
      repository: 'https://github.com/typeshade/typecv',
      commit,
      runtime: 'TypeShade compiled GPU bundle',
      routes,
    },
    null,
    2,
  ) + '\n',
);
await writeFile(
  path.join(website, 'public/robots.txt'),
  'User-agent: *\nAllow: /\nSitemap: https://cv.typeshade.dev/sitemap.xml\n',
);
await writeFile(
  path.join(website, 'public/sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${routes
    .filter((route) => route !== '/404.html')
    .map((route) => `<url><loc>https://cv.typeshade.dev${route}</loc></url>`)
    .join('')}</urlset>\n`,
);
// The TypeShade pin of the runtime: the commit of the archive that package.json installs. A device
// reports only TypeShade's version, and a source install has no revision (`ShadeDevice.info.build`),
// so the hardware check reads the pin from here (docs/lab-run-records.md, "Hardware check").
const rootPackage = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const typeshade =
  /\/archive\/([0-9a-f]{7,40})\.tar\.gz$/.exec(
    rootPackage.devDependencies?.typeshade ?? rootPackage.dependencies?.typeshade ?? '',
  )?.[1] ?? null;
await writeFile(
  path.join(out, 'version.json'),
  JSON.stringify(
    {
      project: 'TypeCV',
      commit,
      runtime: 'TypeShade compiled GPU bundle',
      typeshade,
      localProcessing: true,
    },
    null,
    2,
  ) + '\n',
);
const record = path.join(root, 'docs/experiments/pyramid-8134952/typecv-gpu-resident-model.json');
try {
  await cp(record, path.join(out, 'pyramid-model.json'));
} catch {
  /* optional recorded reconstruction */
}
console.log(`Prepared TypeCV browser runtime from ${commit.slice(0, 7)}`);
