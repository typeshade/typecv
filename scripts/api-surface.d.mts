/** Declarations for scripts/api-surface.mjs, so that its test typechecks. */
export declare const SURFACE_FILE: string;
export declare const ENTRIES: readonly {
  readonly name: string;
  readonly source: string;
  readonly config: string;
}[];
/** The `.d.ts` files that a configuration's build emits, in memory: absolute path → text. */
export declare function emitDeclarations(
  root: string,
  config: string,
  views: ReadonlyMap<string, string>,
): {
  readonly outputs: ReadonlyMap<string, string>;
  readonly outDir: string;
  readonly rootDir: string;
};
/** The host view of each kernel, at the path the Vite plugin writes it to. */
export declare function kernelViews(root?: string): Promise<Map<string, string>>;
/** The text of tests/api-surface.txt. */
export declare function apiSurface(
  root?: string,
  views?: ReadonlyMap<string, string>,
): Promise<string>;
