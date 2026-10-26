// Canonical digests of computed outputs: docs/library-structure.md, section 5, phase 0.
//
// A digest is the SHA-256 of a canonical encoding of the whole value. The encoding is a
// depth-first traversal that writes a type tag before each value and then:
//
//   - a number as its IEEE 754 binary64 bit pattern, little-endian; every NaN as 0x7ff8000000000000;
//   - a string as its UTF-8 bytes after their count; a bigint as its decimal digits;
//   - an array as its length and its elements (a hole as undefined);
//   - a typed array as its type name, its length and each element as a number;
//   - a Map as its size and its entries, a Set as its size and its members, in insertion order;
//   - a plain object as its own enumerable keys in UTF-16 code-unit order, each with its value;
//   - an object of a class as its constructor's name and then as a plain object;
//   - an Error as its name, its message, its cause and then as a plain object.
//
// A key in MASKED keeps its place but not its value: such keys hold wall-clock times. A cycle is
// an error. The module runs in Node and in a browser page, where the WebGPU tests load it as
// /tests/typeshade/digest.mjs.

/** The keys whose values are wall-clock measurements. */
export const MASKED = Object.freeze(['elapsedMs', 'gpuTime', 'longestMs', 'msPerUnit', 'timing']);

const TAG = {
  undefined: 0,
  null: 1,
  false: 2,
  true: 3,
  number: 4,
  bigint: 5,
  string: 6,
  symbol: 7,
  function: 8,
  array: 9,
  typed: 10,
  map: 11,
  set: 12,
  object: 13,
  instance: 14,
  error: 15,
  date: 16,
  buffer: 17,
  masked: 18,
};
const encoder = new TextEncoder();
const byCodeUnit = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

class Writer {
  constructor() {
    this.bytes = new Uint8Array(1 << 16);
    this.view = new DataView(this.bytes.buffer);
    this.length = 0;
  }
  reserve(count) {
    if (this.length + count <= this.bytes.length) return;
    let size = this.bytes.length * 2;
    while (size < this.length + count) size *= 2;
    const next = new Uint8Array(size);
    next.set(this.bytes.subarray(0, this.length));
    this.bytes = next;
    this.view = new DataView(next.buffer);
  }
  tag(tag) {
    this.reserve(1);
    this.bytes[this.length++] = tag;
  }
  count(n) {
    this.reserve(4);
    this.view.setUint32(this.length, n, true);
    this.length += 4;
  }
  number(x) {
    this.reserve(8);
    if (Number.isNaN(x)) {
      this.view.setUint32(this.length, 0, true);
      this.view.setUint32(this.length + 4, 0x7ff80000, true);
    } else this.view.setFloat64(this.length, x, true);
    this.length += 8;
  }
  text(s) {
    const bytes = encoder.encode(s);
    this.count(bytes.length);
    this.reserve(bytes.length);
    this.bytes.set(bytes, this.length);
    this.length += bytes.length;
  }
}

function encodeFields(w, value, masked, seen) {
  const keys = Object.keys(value).sort(byCodeUnit);
  w.count(keys.length);
  for (const key of keys) {
    w.text(key);
    if (masked.has(key)) w.tag(TAG.masked);
    else encode(w, value[key], masked, seen);
  }
}

function encode(w, value, masked, seen) {
  switch (typeof value) {
    case 'undefined':
      return w.tag(TAG.undefined);
    case 'boolean':
      return w.tag(value ? TAG.true : TAG.false);
    case 'number':
      w.tag(TAG.number);
      return w.number(value);
    case 'bigint':
      w.tag(TAG.bigint);
      return w.text(value.toString());
    case 'string':
      w.tag(TAG.string);
      return w.text(value);
    case 'symbol':
      w.tag(TAG.symbol);
      return w.text(value.description ?? '');
    case 'function':
      return w.tag(TAG.function);
  }
  if (value === null) return w.tag(TAG.null);
  if (seen.has(value)) throw new TypeError('A digest reads a tree, and the value holds a cycle');
  seen.add(value);
  try {
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      w.tag(TAG.typed);
      w.text(value.constructor.name);
      w.count(value.length);
      if (typeof value[0] === 'bigint') for (const x of value) w.text(x.toString());
      else for (let i = 0; i < value.length; i++) w.number(value[i]);
    } else if (value instanceof ArrayBuffer) {
      w.tag(TAG.buffer);
      const bytes = new Uint8Array(value);
      w.count(bytes.length);
      for (const b of bytes) w.number(b);
    } else if (Array.isArray(value)) {
      w.tag(TAG.array);
      w.count(value.length);
      for (let i = 0; i < value.length; i++) encode(w, value[i], masked, seen);
    } else if (value instanceof Map) {
      w.tag(TAG.map);
      w.count(value.size);
      for (const [k, v] of value) {
        encode(w, k, masked, seen);
        encode(w, v, masked, seen);
      }
    } else if (value instanceof Set) {
      w.tag(TAG.set);
      w.count(value.size);
      for (const v of value) encode(w, v, masked, seen);
    } else if (value instanceof Error) {
      w.tag(TAG.error);
      w.text(value.name);
      w.text(value.message);
      encode(w, value.cause, masked, seen);
      encodeFields(w, value, masked, seen);
    } else if (value instanceof Date) {
      w.tag(TAG.date);
      w.number(value.getTime());
    } else {
      const prototype = Object.getPrototypeOf(value);
      if (prototype === Object.prototype || prototype === null) w.tag(TAG.object);
      else {
        w.tag(TAG.instance);
        w.text(value.constructor?.name ?? '');
      }
      encodeFields(w, value, masked, seen);
    }
  } finally {
    seen.delete(value);
  }
}

/** The canonical encoding of `value`. */
export function canonicalBytes(value, { masked = MASKED } = {}) {
  const w = new Writer();
  encode(w, value, new Set(masked), new Set());
  return w.bytes.subarray(0, w.length);
}

/** The SHA-256 of the canonical encoding of `value`, in hexadecimal. */
export async function digest(value, options = {}) {
  const hash = await crypto.subtle.digest('SHA-256', canonicalBytes(value, options));
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * The digest of `value` and, for a plain object, of each of its fields that MASKED does not
 * name: `{ digest, parts }`. The parts name the field that changed when a digest changes.
 */
export async function digestParts(value, options = {}) {
  const masked = new Set(options.masked ?? MASKED);
  const parts = {};
  if (value !== null && typeof value === 'object' && !Array.isArray(value))
    for (const key of Object.keys(value).sort(byCodeUnit))
      if (!masked.has(key)) parts[key] = await digest(value[key], options);
  return { digest: await digest(value, options), parts };
}
