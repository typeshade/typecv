/** Declarations for scripts/fetch-record-files.mjs, so that its unit test typechecks. */
export declare const FILE_PREFIX: string;
export interface PlannedFile {
  readonly name: string;
  readonly bytes: number;
  readonly sha256: string;
  /** The R2 key: records/<id>/<name>. */
  readonly key: string;
}
export interface FilePlan {
  readonly id: string;
  readonly files: readonly PlannedFile[];
  /** Selected files that the page did not upload, with its reason. */
  readonly skipped: readonly { readonly name: string; readonly reason: string }[];
}
export interface FileCheck {
  readonly status: 'ok' | 'missing' | 'mismatch';
  readonly bytes?: number;
  readonly sha256?: string;
  readonly reason?: string;
}
export interface FetchResult {
  readonly name: string;
  readonly status: 'ok' | 'missing' | 'mismatch' | 'present' | 'downloaded' | 'failed';
  readonly bytes?: number;
  readonly sha256?: string;
  readonly reason?: string;
  readonly replaced?: boolean;
}
export interface FetchOptions {
  readonly out: string;
  readonly bucket?: string;
  readonly verify?: boolean;
  readonly download?: (bucket: string | undefined, key: string, file: string) => Promise<void>;
  readonly onResult?: (result: FetchResult) => void;
}
export declare function recordFilesBucket(config: string): string | undefined;
export declare function plannedFiles(
  record: unknown,
  options?: { readonly only?: readonly string[] },
): FilePlan;
export declare function verifyFile(
  filePath: string,
  declared: { readonly bytes: number; readonly sha256: string },
): Promise<FileCheck>;
export declare function wranglerArgs(bucket: string, key: string, file: string): string[];
export declare function fetchRecordFiles(
  plan: FilePlan,
  options: FetchOptions,
): Promise<FetchResult[]>;
export declare function resultLine(result: FetchResult): string;
