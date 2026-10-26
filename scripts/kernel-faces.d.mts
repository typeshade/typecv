/** Declarations for scripts/kernel-faces.mjs, so that the tests that import it typecheck. */

/** One export of a kernel module as TypeShade's host face describes it; only `kind` and `name` are read here. */
export interface KernelExport {
  readonly kind: string;
  readonly name: string;
  readonly [field: string]: unknown;
}
/** The build's face of one kernel module. */
export interface KernelFace {
  /** Relative to the repository root, with `/`. */
  readonly path: string;
  /** Absolute, as Vite names the module. */
  readonly fileName: string;
  /** The host view, `name.shade.typeshade.ts`. */
  readonly view: string;
  /** The generated module's JavaScript. */
  readonly code: string;
  readonly exports?: readonly KernelExport[];
}
export declare function kernelFiles(root?: string): string[];
export declare function kernelFaces(root?: string): Promise<KernelFace[]>;
