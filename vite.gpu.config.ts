import { defineConfig } from 'vite';
import { typeshade } from 'typeshade/vite';
/**
 * A stopgap for a cost of the TypeShade runtime (docs/multi-video.md, Amendment 8). Before each
 * dispatch, `checkEntryCall` swaps every `Resident` binding for its host array and dry-packs the
 * array into scratch memory, one number at a time for an f64 array. The Resident is already on
 * the device and the dry run reads every element of it, again for every dispatch: 56 % of the
 * host CPU of a pair verification, and more in a bundle adjustment, whose arrays grow with the
 * map. This plugin makes the dry run skip the bindings that are Residents. `byteSize` still
 * checks their type and length, and every other binding is checked as before. It rewrites the
 * module text at bundle time and touches no file in node_modules. It fails the build when the
 * text is not as expected, so a TypeShade pin that fixes this (or changes the code) shows at
 * once. Remove it then. Upstream: typeshade/typeshade#717. `TYPECV_CHECK_RESIDENTS=1` keeps the dry
 * run, for a comparison.
 */
function skipResidentDryRun() {
  const edits: Record<string, [from: string, to: string][]> = {
    '/typeshade/src/core/host-entry.ts': [
      [
        '  v: unknown,\n): Checked {',
        '  v: unknown,\n  skipPack?: ReadonlySet<string>,\n): Checked {',
      ],
      [
        'if (bulkOf(b.layout, o[b.name]) === undefined)\n        pack(',
        'if (bulkOf(b.layout, o[b.name]) === undefined && !skipPack?.has(b.name))\n        pack(',
      ],
    ],
    '/typeshade/src/core/host-compute.ts': [
      [
        'checked: checkBindings(e, view), wg:',
        'checked: checkBindings(e, view, new Set([...states].map(([b]) => b.name))), wg:',
      ],
    ],
  };
  return {
    name: 'typecv-skip-resident-dry-run',
    enforce: 'pre' as const,
    transform(this: { error(message: string): never }, code: string, id: string) {
      if (process.env['TYPECV_CHECK_RESIDENTS'] === '1') return null;
      const path = id.split('?')[0]!.replaceAll('\\', '/');
      const key = Object.keys(edits).find((k) => path.endsWith(k));
      if (!key) return null;
      let out = code;
      for (const [from, to] of edits[key]!) {
        if (!out.includes(from))
          this.error(
            `typecv-skip-resident-dry-run: ${key} no longer has the text this plugin rewrites ` +
              `(${JSON.stringify(from.slice(0, 60))}). TypeShade changed: read vite.gpu.config.ts and remove or update the plugin.`,
          );
        out = out.replace(from, to);
      }
      return { code: out, map: null };
    },
  };
}

export default defineConfig({
  plugins: [
    skipResidentDryRun(),
    typeshade({ fuse: true }),
    {
      name: 'typecv-runtime-boundary',
      generateBundle(_options, output) {
        for (const chunk of Object.values(output)) {
          if (chunk.type !== 'chunk') continue;
          const forbidden = Object.keys(chunk.modules).filter(
            (id) =>
              /\/node_modules\/typescript\//.test(id) ||
              /\/typeshade\/src\/(compiler|vite)(\/|\.)/.test(id),
          );
          if (forbidden.length)
            throw new Error(`Build-time compiler entered browser bundle: ${forbidden.join(', ')}`);
        }
      },
    },
  ],
  build: {
    outDir: 'dist/gpu',
    emptyOutDir: true,
    lib: { entry: 'src/gpu/index.ts', formats: ['es'], fileName: () => 'index.js' },
    minify: false,
  },
});
