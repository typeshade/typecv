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
  readonly compiler: string;
  readonly modules: Readonly<Record<string, KernelDigest>>;
}
export declare function digestFace(face: KernelFace): KernelDigest;
export declare function compilerPin(root?: string): string;
export declare function kernelDigests(
  root?: string,
  faces?: readonly KernelFace[],
): Promise<KernelDigests>;
