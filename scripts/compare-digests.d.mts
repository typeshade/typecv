/** Declarations for scripts/compare-digests.mjs, so that its test typechecks. */
export declare function compareDigests(
  baseDirectory: string,
  headDirectory: string,
): {
  /** The digest files of the head. */
  readonly files: string[];
  /** The cases whose digests differ, with the fields and the output parts. */
  readonly differences: string[];
  /** The cases and files that the base did not run: reported, not failed. */
  readonly notInBase: string[];
  readonly onlyInBase: string[];
  /** The files that the base wrote and the head did not: a failure. */
  readonly missingInHead: string[];
  /** The configuration of the first file of each side that records one. */
  readonly configurations: { readonly head?: object; readonly base?: object };
  /** The Node versions, when the files ran on more than one: a failure. */
  readonly mismatch: string[];
};
