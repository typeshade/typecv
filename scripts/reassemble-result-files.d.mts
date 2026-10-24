export interface MultipartPlan {
  name: string;
  totalBytes: number;
  parts: { name: string; bytes: number; sha256: string; offset: number }[];
}
export interface ReassemblyOptions {
  partsDir: string;
  outDir: string;
  original?: string;
}
export interface ReassembledFile {
  name: string;
  bytes: number;
  sha256: string;
}
export function multipartPlans(record: unknown, original?: string): MultipartPlan[];
export function reassembleResultFiles(
  record: unknown,
  options: ReassemblyOptions,
): Promise<ReassembledFile[]>;
