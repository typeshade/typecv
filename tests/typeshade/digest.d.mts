/** Declarations for tests/typeshade/digest.mjs, so that the tests in TypeScript typecheck. */
export declare const MASKED: readonly string[];
export interface DigestOptions {
  /** The keys whose values the encoding leaves out. Default MASKED. */
  readonly masked?: readonly string[];
}
export declare function canonicalBytes(value: unknown, options?: DigestOptions): Uint8Array;
export declare function digest(value: unknown, options?: DigestOptions): Promise<string>;
export declare function digestParts(
  value: unknown,
  options?: DigestOptions,
): Promise<{ digest: string; parts: Record<string, string> }>;
