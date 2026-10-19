/** Declarations for the export script, so that its unit test typechecks. */
export declare const NAMESPACE_TITLES: readonly string[];
export declare const RECORD_PREFIX: string;
export declare function accountId(config: string): string | undefined;
export declare function namespaceId(listing: unknown): string | undefined;
export interface RecordKey {
  readonly name: string;
  readonly metadata?: Record<string, unknown> | null;
}
export declare function selectKeys(keys: readonly RecordKey[], limit: number): RecordKey[];
export declare function summaryLine(key: RecordKey): string;
export declare function printed(record: { failure?: unknown }, mode: string): boolean;
