/** Declarations for scripts/import-graph.mjs, so that its test typechecks. */

/** An area of docs/library-structure.md, section 3.1. */
export type Area =
  | 'contracts'
  | 'device'
  | 'image'
  | 'geometry'
  | 'features'
  | 'analysis'
  | 'motion'
  | 'sfm'
  | 'mvs';
export declare const LEVELS: Readonly<Record<Area, number>>;
/** An import between two files of the trees, as the file writes it. */
export interface Import {
  readonly from: string;
  readonly to: string;
  readonly specifier: string;
  /** True when TypeScript and esbuild erase it from the emitted module. */
  readonly typeOnly: boolean;
  readonly form: 'import' | 'export' | 'dynamic' | 'type';
  readonly line: number;
}
export interface Unresolved {
  readonly from: string;
  readonly specifier: string;
  readonly line: number;
}
export interface ImportGraph {
  readonly files: readonly string[];
  readonly imports: readonly Import[];
  readonly unresolved: readonly Unresolved[];
}
export interface TypeOnlyCycle {
  readonly files: readonly string[];
  readonly typeOnly: readonly Import[];
}
export interface LayeringBreak extends Import {
  readonly fromArea: Area;
  readonly toArea: Area;
}
export interface ImportReport {
  readonly files: number;
  readonly imports: number;
  readonly runtimeImports: number;
  readonly unresolved: readonly Unresolved[];
  readonly runtimeCycles: readonly (readonly string[])[];
  readonly typeOnlyCycles: readonly TypeOnlyCycle[];
  readonly layeringBreaks: readonly LayeringBreak[];
  readonly unassigned: readonly string[];
}
export declare function areaOf(file: string): Area | 'entry' | undefined;
export declare function sourceFiles(root?: string): string[];
export declare function importsOf(file: string, text: string): Omit<Import, 'from' | 'to'>[];
export declare function importGraph(root?: string): ImportGraph;
export declare function runtimeCycles(graph: ImportGraph): string[][];
export declare function typeOnlyCycles(graph: ImportGraph): TypeOnlyCycle[];
export declare function layeringBreaks(graph: ImportGraph): LayeringBreak[];
export declare function unassignedFiles(graph: ImportGraph): string[];
export declare function report(root?: string): ImportReport;
export declare function formatReport(result: ImportReport): string[];
