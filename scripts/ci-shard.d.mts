/** Declarations for scripts/ci-shard.mjs, so that its test typechecks. */
export declare function scriptFiles(script: string, packageJson?: unknown): string[];
export declare function shardFiles(
  script: string,
  shard: number,
  of: number,
  packageJson?: unknown,
): string[];
