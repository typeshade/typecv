/** Lossless, bounded raw staging of MVS diagnostics into ordinary tar.gz result files. */
import type { GpuDenseDepthDiagnostic } from '../../../src/gpu/mvs.ts';

const TARGET_BYTES = 32 * 1024 ** 2;
const SEGMENT_BYTES = 8 * 1024 ** 2;
const encoder = new TextEncoder();
const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** Minimal POSIX ustar header. All paths are generated here, never from input filenames. */
function header(path: string, bytes: number): Uint8Array {
  const h = new Uint8Array(512);
  const text = (offset: number, value: string) => h.set(encoder.encode(value), offset);
  const octal = (offset: number, size: number, value: number) =>
    text(offset, value.toString(8).padStart(size - 1, '0') + '\0');
  text(0, path);
  octal(100, 8, 0o644);
  octal(108, 8, 0);
  octal(116, 8, 0);
  octal(124, 12, bytes);
  octal(136, 12, 0);
  h.fill(32, 148, 156);
  text(156, '0');
  text(257, 'ustar\0');
  text(263, '00');
  text(
    148,
    h
      .reduce((sum, b) => sum + b, 0)
      .toString(8)
      .padStart(6, '0') + '\0 ',
  );
  return h;
}

type Part = { path: string; offset: number; bytes: number };
export type DepthArtifact = { name: string; data: Uint8Array | string };

/** Arrays must be consumed inside the awaited callback, before MVS reuses or releases them. */
export class DepthArchive {
  private chunks: Uint8Array[] = [];
  private rawBytes = 0;
  private parts: string[] = [];
  private maps: Record<string, unknown>[] = [];
  private inputs = new Set<number>();
  private error = '';
  private finished = false;
  private readonly prefix: string;

  private readonly emit: (file: DepthArtifact) => void;
  private readonly compress: (chunks: Uint8Array[]) => Promise<Uint8Array>;
  private readonly targetBytes: number;

  constructor(
    component: number,
    emit: (file: DepthArtifact) => void,
    compress = async (chunks: Uint8Array[]) =>
      new Uint8Array(
        await new Response(
          new Blob(chunks as BlobPart[]).stream().pipeThrough(new CompressionStream('gzip')),
        ).arrayBuffer(),
      ),
    targetBytes = TARGET_BYTES,
  ) {
    this.emit = emit;
    this.compress = compress;
    this.targetBytes = targetBytes;
    this.prefix = 'typecv-depths-c' + String(component).padStart(3, '0');
  }

  private async flush(): Promise<void> {
    if (!this.rawBytes) return;
    if (this.parts.length >= 9999) throw new RangeError('Too many depth archive parts');
    const data = await this.compress([...this.chunks, new Uint8Array(1024)]);
    const name = this.prefix + '-' + String(this.parts.length).padStart(4, '0') + '.tar.gz';
    this.emit({ name, data });
    this.parts.push(name);
    this.chunks = [];
    this.rawBytes = 0;
  }

  private async add(path: string, data: Uint8Array): Promise<void> {
    this.chunks.push(header(path, data.byteLength), data.slice());
    const padding = (512 - (data.byteLength % 512)) % 512;
    if (padding) this.chunks.push(new Uint8Array(padding));
    this.rawBytes += 512 + data.byteLength + padding;
    if (this.rawBytes >= this.targetBytes) await this.flush();
  }

  private async array(path: string, values: Float32Array | Uint8ClampedArray): Promise<Part[]> {
    let bytes: Uint8Array;
    if (values instanceof Float32Array && !littleEndian) {
      bytes = new Uint8Array(values.byteLength);
      const view = new DataView(bytes.buffer);
      values.forEach((z, i) => view.setFloat32(i * 4, z, true));
    } else bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const parts: Part[] = [];
    for (let at = 0; at < bytes.byteLength; at += SEGMENT_BYTES) {
      const part = bytes.subarray(at, at + SEGMENT_BYTES);
      const name = path + '.' + String(parts.length).padStart(4, '0');
      await this.add(name, part);
      parts.push({ path: name, offset: at, bytes: part.byteLength });
    }
    return parts;
  }

  async capture(
    event: GpuDenseDepthDiagnostic,
    input?: { rgba: Uint8ClampedArray; width: number; height: number; camera?: unknown },
  ): Promise<void> {
    if (this.error || this.finished) return;
    try {
      if (
        event.depth.length !== event.width * event.height ||
        (event.confidence && event.confidence.length !== event.depth.length)
      )
        throw new RangeError('Diagnostic map shape mismatch');
      const folder = 'view-' + event.viewId + '/' + event.stage;
      const { depth, confidence, ...metadata } = event;
      let valid = 0;
      for (const z of depth) if (Number.isFinite(z) && z > 0) valid++;
      const map = {
        ...metadata,
        validPixels: valid,
        invalidDepth: 0,
        dtype: 'float32-le',
        grid: 'undistorted-pinhole',
        depthMeaning: 'camera-axis-z',
        depth: await this.array(folder + '/depth.f32le', depth),
        ...(confidence
          ? { confidence: await this.array(folder + '/confidence.f32le', confidence) }
          : {}),
      };
      this.maps.push(map);
      await this.add(folder + '/map.json', encoder.encode(JSON.stringify(map)));
      if (input && !this.inputs.has(event.viewId)) {
        const source = {
          width: input.width,
          height: input.height,
          camera: input.camera,
          dtype: 'rgba8',
          grid: 'original-distorted',
          parts: await this.array('view-' + event.viewId + '/input.rgba8', input.rgba),
        };
        await this.add(
          'view-' + event.viewId + '/input.json',
          encoder.encode(JSON.stringify(source)),
        );
        this.inputs.add(event.viewId);
      }
    } catch (error) {
      this.error = String(error).slice(0, 500);
      this.chunks = [];
      this.rawBytes = 0;
    }
  }

  async finish(
    outcome: 'completed' | 'cancelled' | 'failed',
    expectedViews: readonly number[],
    planar: boolean,
  ): Promise<void> {
    if (this.finished) return;
    this.finished = true;
    try {
      await this.flush();
    } catch (error) {
      this.error = String(error).slice(0, 500);
    }
    const stages = [
      'photometric',
      'geometric-unfiltered',
      'geometric-filtered',
      ...(planar ? ['planar'] : []),
    ];
    const missing = expectedViews.flatMap((viewId) =>
      stages
        .filter((stage) => !this.maps.some((m) => m.viewId === viewId && m.stage === stage))
        .map((stage) => ({ viewId, stage })),
    );
    this.emit({
      name: this.prefix + '.json',
      data: JSON.stringify(
        {
          format: 'typecv-depth-archive-v1',
          outcome,
          complete: outcome === 'completed' && !this.error && !missing.length,
          error: this.error || null,
          expectedViews,
          expectedStages: stages,
          missing,
          archives: this.parts,
          maps: this.maps,
          instructions:
            'Extract all tar.gz parts into the same directory. Concatenate array segments by offset. Decode float32 little endian, row-major. Model camera Z = depth * depthScaleToModel. Confidence is not photometric cost. Input RGBA uses its original camera; depth uses its undistorted pinhole camera.',
        },
        null,
        2,
      ),
    });
  }
}
