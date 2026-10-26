/** Declarations for tests/typeshade/digest-fixture.mjs, so that the tests in TypeScript typecheck. */
export interface DigestRecord {
  readonly output: string;
  readonly parts: Readonly<Record<string, string>>;
  readonly trace: string;
  readonly requests: Readonly<Record<string, number>>;
}
export declare function readFixture(file: string): Record<string, DigestRecord>;
export declare function compareRecord(
  name: string,
  recorded: DigestRecord | undefined,
  current: DigestRecord,
  compared: readonly string[],
): void;
export declare function digestRecorder(
  fixture: string,
  options?: { readonly compared?: readonly string[]; readonly update?: string },
): { check(name: string, record: DigestRecord): void; write(): void };
