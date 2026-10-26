/** Declarations for tests/typeshade/digest-fixture.mjs, so that the tests in TypeScript typecheck. */
export interface DigestRecord {
  readonly output: string;
  readonly parts: Readonly<Record<string, string>>;
  readonly trace: string;
  readonly requests: Readonly<Record<string, number>>;
}
/** The configuration that recorded a fixture, or that runs a test. */
export interface RunConfiguration {
  readonly node: string;
  readonly v8: string;
  readonly platform: string;
  readonly commit: string;
  /** The git tree of src/, or `modified` when src/ differs from the commit. */
  readonly src: string;
}
export declare function runConfiguration(): RunConfiguration;
export declare function readFixtureFile(file: string): {
  configuration: RunConfiguration | undefined;
  cases: Record<string, DigestRecord>;
};
export declare function readFixture(file: string): Record<string, DigestRecord>;
export declare function configurationNote(
  recorded: RunConfiguration | undefined,
  current: RunConfiguration,
): string;
export declare function compareRecord(
  name: string,
  recorded: DigestRecord | undefined,
  current: DigestRecord,
  compared: readonly string[],
  note?: string,
): void;
export declare function digestRecorder(
  fixture: string,
  options?: { readonly compared?: readonly string[]; readonly update?: string },
): { check(name: string, record: DigestRecord): void; write(): void };
