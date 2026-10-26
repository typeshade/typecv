/** Declarations for scripts/kernel-digests.mjs, so that its test typechecks. */
import type { KernelFace } from './kernel-faces.mjs';

export declare const DIGEST_FILE: string;
/** The digests of one kernel module. */
export interface KernelDigest {
  readonly wgsl: string;
  readonly wgslInteger?: string;
  readonly glsl?: string;
  readonly entries: number;
  readonly kernelFunctions: number;
  readonly glslPrograms: number;
  readonly withoutGlsl: number;
}
/** The contents of tests/digests/kernels.json. */
export interface KernelDigests {
  /** The command that writes the file. */
  readonly update: string;
  /** The Node and V8 versions, the platform and the commit that wrote the file. */
  readonly configuration: {
    readonly node: string;
    readonly v8: string;
    readonly platform: string;
    readonly commit: string;
  };
  readonly compiler: string;
  readonly modules: Readonly<Record<string, KernelDigest>>;
}
export declare function digestFace(face: KernelFace): KernelDigest;
/** The target of a kernel alias, relative to the repository root, or undefined. */
export declare function aliasTarget(path: string, text: string): string | undefined;
/** A kernel alias, its target and whether its digests equal the target's. */
export interface KernelAlias {
  readonly alias: string;
  readonly target: string;
  readonly equal: boolean;
}
export declare function kernelAliases(
  root?: string,
  faces?: readonly KernelFace[],
): Promise<KernelAlias[]>;
export declare function compilerPin(root?: string): string;
export declare function kernelDigests(
  root?: string,
  faces?: readonly KernelFace[],
): Promise<KernelDigests>;
