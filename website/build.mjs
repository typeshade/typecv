import { readFile, writeFile, mkdir, readdir, cp, rm, access } from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Marked, Renderer } from 'marked';
import { build } from 'esbuild';
import { copy, pairCode, setupCode, origin, repository } from './content.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'website/dist');
const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
const escape = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const code = (s) => `<pre><code>${escape(s)}</code></pre>`;
const base = (lang) => (lang === 'ko' ? '/ko/' : '/');
const documentRoute = (file) =>
  file === 'README.md'
    ? '/docs/project/'
    : file === 'docs/README.md'
      ? '/docs/'
      : '/docs/' + file.slice(5, -3).replace(/\/README$/, '') + '/';
const pages = new Map();
async function save(route, html) {
  const file = path.join(output, route === '/404.html' ? '404.html' : route + 'index.html');
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, html);
  pages.set(route, html);
}
function shell(
  title,
  description,
  content,
  { lang = 'en', route = '/', section = 'overview', alternate } = {},
) {
  const t = copy[lang],
    p = base(lang);
  return `<!doctype html><html lang="${lang}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(title)} · TypeCV</title><meta name="description" content="${escape(description)}">
<link rel="canonical" href="${origin}${route}"><meta property="og:title" content="${escape(title)} · TypeCV"><meta property="og:description" content="${escape(description)}"><meta property="og:url" content="${origin}${route}"><meta property="og:type" content="website"><meta name="theme-color" content="#1677ff">
${alternate ? `<link rel="alternate" hreflang="en" href="${origin}${alternate.en}"><link rel="alternate" hreflang="ko" href="${origin}${alternate.ko}">` : ''}
<link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="/site.css"><script type="module" src="/site.js"></script></head><body>
<a class="skip" href="#main">${lang === 'ko' ? '본문으로' : 'Skip to content'}</a>
<header class="header"><div class="nav-shell"><a class="brand" href="${p}" aria-label="TypeCV ${t.overview}"><span class="brand-icon" aria-hidden="true">⌗</span>Type<span>CV</span></a><nav aria-label="${lang === 'ko' ? '주 메뉴' : 'Main navigation'}">${[
    ['overview', p],
    ['docs', '/docs/'],
    ['playground', p + 'playground/'],
  ]
    .map(
      ([key, href]) =>
        `<a href="${href}" ${section === key ? 'aria-current="page"' : ''}>${t[key]}</a>`,
    )
    .join(
      '',
    )}<a href="${repository}">GitHub ↗</a></nav><div class="nav-tools"><a lang="${lang === 'ko' ? 'en' : 'ko'}" href="${alternate ? alternate[lang === 'ko' ? 'en' : 'ko'] : '/ko/'}">${lang === 'ko' ? 'EN' : '한국어'}</a><button id="theme" type="button" aria-label="${lang === 'ko' ? '색상 테마 변경' : 'Toggle color theme'}">◐</button></div></div></header>
${content}
<footer><div><a class="brand" href="${p}">Type<span>CV</span></a><p>${t.footer}</p></div><div class="footer-links"><a href="https://typeshade.dev">TypeShade ↗</a><a href="https://nn.typeshade.dev">TypeNN ↗</a><a href="${repository}/tree/${commit}">Source ${commit.slice(0, 7)}</a><a href="${repository}/blob/main/LICENSE">Apache-2.0</a></div></footer>
</body></html>`;
}

function home(lang) {
  const t = copy[lang],
    p = base(lang);
  const content = `<main id="main"><section class="hero shell"><div><p class="eyebrow">TYPECV / COMPUTER VISION</p><h1>${t.hero.split('\n').map(escape).join('<br>')}</h1><p class="lead">${t.intro}</p><div class="actions"><a class="button primary" href="${p}playground/">${t.try} <span aria-hidden="true">↗</span></a><a class="button" href="/docs/sfm-quickstart/">${t.start}</a></div><p class="small muted">${t.eyebrow}</p></div><div class="code-panel"><div class="panel-heading"><span class="live-dot"></span><span>PAIR VERIFICATION</span><span class="mono">pair.ts</span></div>${code(pairCode)}<div class="pipeline" aria-label="${t.eyebrow}"><span>pixels</span><span>→</span><span>H / E</span><span>→</span><span>R, t</span></div><p class="code-caption">${t.caption}</p></div></section>
<section class="section shell"><div class="section-intro"><p class="eyebrow">01 / OPERATORS</p><h2>${t.current}</h2><p>${t.currentText}</p></div><div class="operator-list">${t.ops.map(([n, title, text, href, api]) => `<a class="operator" href="${href}"><span class="operator-n">${n}</span><div><h3>${title}</h3><p>${text}</p></div><span class="operator-api">${api}<span aria-hidden="true"> ↗</span></span></a>`).join('')}</div></section>
<section class="scope section"><div class="shell split"><div><p class="eyebrow">02 / FOUNDATION</p><h2>${t.scope}</h2></div><div><p>${t.scopeText}</p><a class="text-link" href="/docs/roadmap/">${t.roadmap} →</a><p class="small muted">${t.englishDocs}</p></div></div></section>
<section class="section shell split"><div><p class="eyebrow">03 / REPRODUCIBLE</p><h2>${t.evidence}</h2><p>${t.evidenceText}</p><a class="text-link" href="/docs/experiments/gpu/resident-pair-verification-software.json">${t.evidenceLink} →</a></div><div class="evidence-figure" aria-label="Recorded sparse reconstruction"><div><strong>8 / 8</strong><span>registered views</span></div><div><strong>1,723</strong><span>sparse points</span></div><p class="mono">SWIFTSHADER · ASSUMED CALIBRATION</p></div></section>
<section class="section shell split start-section"><div><p class="eyebrow">04 / GET STARTED</p><h2>${t.source}</h2><p>${t.sourceText}</p><a class="text-link" href="/docs/">${t.docs} →</a></div><div class="setup">${code(setupCode)}</div></section></main>`;
  return shell(t.tagline, t.intro, content, { lang, route: p, alternate: { en: '/', ko: '/ko/' } });
}

function playground(lang) {
  const t = copy[lang],
    p = base(lang);
  const content = `<main id="main" class="shell playground"><p class="eyebrow">TYPECV / PLAYGROUND</p><h1>${t.demoTitle}</h1><p class="lead">${t.demoIntro}</p><p class="small muted">${t.privacy}</p>
<form id="pair-form" class="controls"><label>${t.backend}<select name="backend"><option value="webgpu">${t.gpu}</option><option value="cpu">${t.cpu}</option></select></label><label>${t.count}<select name="count"><option>64</option><option selected>128</option><option>256</option></select></label><label>${t.noise}<select name="noise"><option value="0">0 px</option><option value="0.12" selected>0.12 px</option><option value="0.5">0.5 px</option></select></label><label>${t.outliers}<select name="outliers"><option value="0">0%</option><option value="20" selected>20%</option><option value="30">30%</option></select></label><button class="button primary" id="run" type="submit">${t.run} ↗</button></form>
<p id="status" class="status" role="status" aria-live="polite">${t.ready}</p><div class="image-pair"><figure><figcaption>${t.input}</figcaption><canvas id="first-image" width="960" height="600" aria-label="${t.input}"></canvas></figure><figure><figcaption>${t.output}</figcaption><canvas id="second-image" width="960" height="600" aria-label="${t.output}"></canvas></figure></div><p class="small muted">${t.legend}<br>${t.synthetic}</p>
<div id="results" hidden><h2>${t.status}</h2><dl id="summary" class="metrics"></dl><div class="result-grid"><div><h3>${t.pose}</h3><pre id="pose"></pre></div><div><h3>${t.metrics}</h3><pre id="execution"></pre></div></div></div><p class="demo-note">${t.note}</p><details><summary>TypeScript / WebGPU</summary>${code(pairCode)}</details><a class="text-link" href="/docs/gpu-initialization/">${t.docs} →</a>
<script type="module" src="/playground.js"></script></main>`;
  return shell(t.playground, t.demoIntro, content, {
    lang,
    route: p + 'playground/',
    section: 'playground',
    alternate: { en: '/playground/', ko: '/ko/playground/' },
  });
}

async function filesIn(directory) {
  const entries = await readdir(path.join(root, directory), { withFileTypes: true });
  const lists = await Promise.all(
    entries.map((e) =>
      e.isDirectory() ? filesIn(directory + '/' + e.name) : [directory + '/' + e.name],
    ),
  );
  return lists.flat();
}
await access(path.join(root, 'dist/gpu.js')).catch(() => {
  throw new Error('Build TypeCV first: npm run build');
});
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(path.join(root, 'website/public'), output, { recursive: true });
for (const file of ['site.css', 'site.mjs'])
  await cp(path.join(root, 'website', file), path.join(output, file.replace('.mjs', '.js')));
await build({
  entryPoints: [path.join(root, 'website/playground.mjs')],
  outfile: path.join(output, 'playground.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  external: ['/runtime/*'],
});
await mkdir(path.join(output, 'runtime'), { recursive: true });
await cp(path.join(root, 'dist/gpu.js'), path.join(output, 'runtime/gpu.js'));
await cp(path.join(root, 'dist/gpu'), path.join(output, 'runtime/gpu'), { recursive: true });
await cp(path.join(root, 'LICENSE'), path.join(output, 'runtime/LICENSE-TypeCV'));
await cp(
  path.join(root, 'node_modules/typeshade/LICENSE'),
  path.join(output, 'runtime/LICENSE-TypeShade'),
);
await build({
  entryPoints: [path.join(root, 'dist/index.js')],
  outfile: path.join(output, 'runtime/core.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  minify: true,
  legalComments: 'eof',
});
for (const lang of ['en', 'ko']) {
  await save(base(lang), home(lang));
  await save(base(lang) + 'playground/', playground(lang));
}
const documents = ['README.md', ...(await filesIn('docs')).filter((f) => f.endsWith('.md'))];
const nav = [
  ['Get started', 'docs/sfm-quickstart.md'],
  ['Architecture', 'docs/architecture.md'],
  ['GPU pair verification', 'docs/gpu-initialization.md'],
  ['GPU geometry', 'docs/gpu-geometry.md'],
  ['Incremental SfM', 'docs/incremental-sfm.md'],
  ['Two-view geometry', 'docs/sfm.md'],
  ['GPU execution', 'docs/gpu-sfm.md'],
  ['TypeShade capabilities', 'docs/typeshade-capabilities.md'],
  ['CV roadmap', 'docs/roadmap.md'],
  ['SfM roadmap', 'docs/sfm-roadmap.md'],
  ['Video / COLMAP experiment', 'docs/experiments/incremental-pyramid/README.md'],
];
for (const file of documents) {
  const markdown = await readFile(path.join(root, file), 'utf8');
  const title = markdown.match(/^# (.+)$/m)?.[1] ?? 'TypeCV documentation';
  const renderer = new Renderer(),
    slugs = new Map();
  renderer.heading = function ({ tokens, depth }) {
    const html = this.parser.parseInline(tokens);
    const text = html
      .replace(/<[^>]*>/g, '')
      .replace(/&[^;]+;/g, '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}_\s-]/gu, '')
      .trim()
      .replace(/\s/g, '-');
    const n = slugs.get(text) ?? 0;
    slugs.set(text, n + 1);
    return `<h${depth} id="${escape(text + (n ? '-' + n : ''))}">${html}</h${depth}>\n`;
  };
  renderer.link = function ({ href, title, tokens }) {
    let target = href ?? '';
    if (target && !/^(?:https?:|mailto:|#)/i.test(target)) {
      const [local, fragment] = target.split('#');
      const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(file), local));
      target = documents.includes(resolved)
        ? documentRoute(resolved)
        : resolved.startsWith('docs/') && resolved.endsWith('.json')
          ? '/' + resolved
          : repository + '/blob/main/' + resolved;
      if (fragment) target += '#' + fragment;
    }
    return `<a href="${escape(target)}"${title ? ' title="' + escape(title) + '"' : ''}>${this.parser.parseInline(tokens)}</a>`;
  };
  const article = new Marked({ renderer, gfm: true }).parse(markdown);
  const route = documentRoute(file);
  const sidebar = `<aside class="docs-sidebar"><a href="/docs/" class="eyebrow">DOCUMENTATION</a><nav aria-label="Documentation">${nav.map(([label, source]) => `<a href="${documentRoute(source)}" ${file === source ? 'aria-current="page"' : ''}>${label}</a>`).join('')}</nav><a class="small" href="${repository}/blob/main/${file}">View source ↗</a></aside>`;
  await save(
    route,
    shell(
      title,
      'TypeCV reference: ' + title,
      `<main id="main" class="docs-shell">${sidebar}<article class="prose">${article}</article></main>`,
      { route, section: 'docs' },
    ),
  );
}
for (const file of (await filesIn('docs')).filter((f) => f.endsWith('.json'))) {
  await mkdir(path.dirname(path.join(output, file)), { recursive: true });
  await cp(path.join(root, file), path.join(output, file));
}
await save(
  '/404.html',
  shell(
    'Page not found',
    'This page does not exist.',
    '<main id="main" class="shell not-found"><p class="eyebrow">404</p><h1>Outside the frame.</h1><p>The page you requested does not exist.</p><a class="button primary" href="/">Back to TypeCV →</a></main>',
    { route: '/404.html' },
  ),
);
await writeFile(
  path.join(output, 'robots.txt'),
  `User-agent: *\nAllow: /\nSitemap: ${origin}/sitemap.xml\n`,
);
await writeFile(
  path.join(output, 'sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${[
    ...pages.keys(),
  ]
    .filter((p) => p !== '/404.html')
    .map((p) => `<url><loc>${origin}${p}</loc></url>`)
    .join('')}</urlset>\n`,
);
await writeFile(
  path.join(output, 'version.json'),
  JSON.stringify(
    {
      project: 'TypeCV',
      repository,
      commit,
      runtime: 'Built from this repository; TypeShade compiler excluded',
      routes: [...pages.keys()],
    },
    null,
    2,
  ) + '\n',
);
console.log(
  `Built ${pages.size} pages, local runtimes and experiment records for ${origin} (${commit.slice(0, 7)}).`,
);
