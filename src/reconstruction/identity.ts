/**
 * Content identity for multi-video reconstruction (docs/multi-video.md, section 2.4).
 *
 * Every identifier that decides anything comes from content. Input order, file names and the
 * arrival order of decoded frames decide nothing.
 */

const HEX_DIGITS = 16;
const MIB = 1024 * 1024;

/** A source of bytes that can be cut, such as a `Blob` or a `File`. */
export interface ByteSource {
  readonly size: number;
  slice(start?: number, end?: number): { arrayBuffer(): Promise<ArrayBuffer> };
}

/** The key of a keyframe in the run: its content, its video and its time. */
export interface KeyframeIdentity {
  readonly frameKey: string;
  readonly videoId: string;
  /** Time in the video, in seconds. */
  readonly time: number;
}

async function sha256Hex16(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
  let hex = '';
  for (let i = 0; i < HEX_DIGITS / 2; i++) hex += digest[i]!.toString(16).padStart(2, '0');
  return hex;
}

/**
 * `videoId`: SHA-256 of the file size (8 bytes, big-endian), its first 1 MiB and its last 1 MiB,
 * cut to 16 hex digits. A file under 2 MiB hashes whole, once.
 */
export async function videoIdOf(file: ByteSource): Promise<string> {
  const size = file.size;
  if (!Number.isSafeInteger(size) || size < 0) throw new RangeError('file size must be a size');
  const headEnd = Math.min(size, MIB),
    tailStart = Math.max(headEnd, size - MIB);
  const head = new Uint8Array(await file.slice(0, headEnd).arrayBuffer()),
    tail = new Uint8Array(await file.slice(tailStart, size).arrayBuffer());
  const bytes = new Uint8Array(8 + head.length + tail.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(size));
  bytes.set(head, 8);
  bytes.set(tail, 8 + head.length);
  return sha256Hex16(bytes);
}

/**
 * `frameKey`: SHA-256 of the analysis frame's RGBA bytes (the pixels the worker receives), cut to
 * 16 hex digits. Equal frames in two videos get equal keys.
 */
export function frameKeyOf(rgba: Uint8Array | Uint8ClampedArray): Promise<string> {
  return sha256Hex16(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength));
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** The order of 2.4: by `frameKey`, then `videoId`, then `time`. Never by locale. */
export function compareKeyframes(a: KeyframeIdentity, b: KeyframeIdentity): number {
  return (
    compareText(a.frameKey, b.frameKey) || compareText(a.videoId, b.videoId) || a.time - b.time
  );
}

/**
 * `viewId`: the rank of each keyframe in the sorted list of all keyframes of the run, from 0. The
 * result at index `i` is the `viewId` of `keyframes[i]`, so it does not depend on the order of the
 * input. Two keyframes with one `(frameKey, videoId, time)` have no content order, so the call
 * throws: the caller drops a duplicate video before it samples frames.
 */
export function assignViewIds(keyframes: readonly KeyframeIdentity[]): number[] {
  for (const k of keyframes)
    if (!Number.isFinite(k.time)) throw new RangeError('keyframe time must be finite');
  const order = keyframes
    .map((_, i) => i)
    .sort((a, b) => compareKeyframes(keyframes[a]!, keyframes[b]!));
  const ids = new Array<number>(keyframes.length);
  order.forEach((index, rank) => {
    if (rank > 0 && compareKeyframes(keyframes[order[rank - 1]!]!, keyframes[index]!) === 0)
      throw new RangeError('two keyframes have the same frameKey, videoId and time');
    ids[index] = rank;
  });
  return ids;
}

/**
 * The RANSAC seed of a pair: a fixed 32-bit mix of the run seed and the two view ids. It does not
 * depend on the id order inside the pair, nor on the position of the pair in a list. A pair keeps
 * its seed when another pair is added or dropped.
 */
export function pairSeed(seed: number, first: number, second: number): number {
  const lo = Math.min(first, second),
    hi = Math.max(first, second);
  let h = (seed | 0) ^ 0x9e3779b9;
  for (const v of [lo, hi]) {
    h = Math.imul(h ^ (v | 0), 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
  }
  return h >>> 0;
}
