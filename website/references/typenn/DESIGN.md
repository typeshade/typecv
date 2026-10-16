# TypeNN site design

The TypeNN site keeps the TypeShade site's Ant Design and IBM Plex visual language while
organizing content as a developer site with an overview, guides, API reference, runnable
Playground, and linked source examples. TensorFlow.js is an information-architecture reference:
clear learning paths and a visible try-it-now experience. Its layout, writing, graphics, and
brand are not copied.

## Sources and product truth

- Source design inspiration: `references/typeshade-site/DESIGN.md`
- Source product brief: `references/typeshade-site/PRODUCT.md`
- TypeNN implementation: `https://github.com/typeshade/typenn`
- TypeShade compiler and runtime: `https://github.com/typeshade/typeshade`
- Current user flow: overview → get started → guides/API → Playground or source examples.

Use current source, tests, README, and TODO as evidence. Distinguish TypeNN's Tensor graph from
TypeShade Shader IR. Each `data()` runs as one TypeShade `fuse` scope; only WebGPU fuses an elementwise chain into one dispatch. Do not invent benchmark values,
hardware details, generated compiler output, model results, or support guarantees.

## Stack

- Astro generates static English routes at `/` and Korean routes at `/ko/`.
- React hydrates only interactive islands: the theme control and the Playground.
- TailwindCSS v4 handles layout utilities and Ant Design v5 provides real interactive controls.
- IBM Plex Sans, IBM Plex Sans KR, and IBM Plex Mono are self-hosted under `public/fonts/`.
- Astro's Shiki `Code` component highlights static TypeScript examples at build time. Prism
  highlights the live Playground sample.
- The Playground loads the compiled TypeNN package built at the commit recorded in
  `src/vendor/typenn/README.md`; it does not reproduce Tensor math in a mock implementation.

## Information architecture

- `/` and `/ko/`: product overview, real Tensor code, compiler/execution path, entry points.
- `/guide/` and `/ko/guide/`: first operation and lifecycle.
- `/guide/execution/` and `/ko/guide/execution/`: deferred graph, kernel functions, the fuse scope of
  each `data()`, backend tiers, and `device.lastReport`.
- `/guide/training/` and `/ko/guide/training/`: Module, autograd, optimizer, resource lifetime,
  and verified GPU-resident behavior.
- `/api/` and `/ko/api/`: practical index of the current Device, Tensor, Module, and optimizer
  API; link source for the complete and versioned signatures.
- `/playground/` and `/ko/playground/`: editable 2×2 matrices, selectable Tensor operations and
  backend, actual TypeNN execution, result, backend, errors, and equivalent highlighted code.
- `/examples/` and `/ko/examples/`: MNIST MLP code, source repository, correctness tests, and
  implementation status.

Every English route has a separately authored Korean counterpart with canonical and alternate
language links. Shared top navigation provides a short product-level route; a documentation
sidebar and previous/next links support deeper reading. Mobile turns the sidebar into a
horizontal navigation row.

## Visual system

- Use TypeShade's restrained Ant Design palette: primary `#1677ff`, neutral ground, white
  content panels, neutral ink, and hairline borders.
- Use IBM Plex Sans for interface text, IBM Plex Sans KR for Korean glyph coverage, and IBM Plex
  Mono for code and identifiers.
- Cap the shared header, page frame, and footer at 1440px. Cap overview content at 1320px,
  reading articles at 940px, and the Playground at 1180px. Use 32px shell gutters on wide
  screens, 24px below 1050px, 18px below 820px, and 14px below 560px.
- Use 6px control corners and 8px code/workbench corners. Panels stay flat; reserve shadow for
  the main code sample and avoid decorative gradients.
- The overview begins with working TypeNN code and the actual result shape. The rest of the page
  leads from the execution path into guides and examples; do not rebuild the site as a hero plus
  generic feature-card grid.
- Reading pages use a stable docs shell: global header, left navigation, readable article measure,
  examples in dark highlighted code frames, and related-page navigation.
- The Playground uses standard form controls and explicit loading, empty, success, and error
  states. It is a small functional demonstration, not a benchmark.
- Keep light/dark theme selection remembered in local storage. Keep keyboard focus visible,
  controls named, text contrast accessible, and content usable with reduced motion.
- No stock images are required: verified code, source links, and real output are the visual proof.

## Content and accessibility

- Write in plain sentences for TypeScript developers. English is the source route; Korean is
  separately written, uses 합쇼체, and does not translate API identifiers.
- Explain the observable behavior before implementation detail. Link claims to code or project
  documentation.
- The Playground accepts only numeric matrix values and fixed operations. It does not evaluate
  user-provided source text.
- Mark software-rendered validation as such. Do not call it a hardware GPU benchmark.
- Keep one `h1`, ordered sections, landmarks, a keyboard-visible skip link, accessible names,
  visible focus, and horizontal scrolling only for code or genuinely wide tables.
