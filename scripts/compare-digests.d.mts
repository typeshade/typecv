/** Declarations for scripts/compare-digests.mjs, so that its test typechecks. */
export declare function readDeclaration(file: string):
  | {
      /** The pull request that declares the changes; the comparison honours it for that one. */
      readonly pullRequest: number;
      /** Why the digests change. */
      readonly reason: string;
      /** The case keys `<file>: <case>` whose digests the pull request changes, without repeats. */
      readonly cases: string[];
    }
  | undefined;
export declare function compareDigests(
  baseDirectory: string,
  headDirectory: string,
  /** The case keys `<file>: <case>` that the current pull request declares. */
  declared?: readonly string[],
): {
  /** The digest files of the head. */
  readonly files: string[];
  /** The cases whose digests differ, with the fields and the output parts. */
  readonly differences: string[];
  /** The differences in declared cases: reported, not failed. */
  readonly declared: string[];
  /** The differences in cases that are not declared: a failure. */
  readonly undeclared: string[];
  /** The declared cases whose digests are equal: reported. */
  readonly declaredEqual: string[];
  /** The declared cases that the head did not run: reported. */
  readonly declaredUnknown: string[];
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
