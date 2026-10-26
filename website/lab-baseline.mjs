// The baselines of phase 0 of the refactoring programme (`docs/library-structure.md`, section 5):
// a canonical text form for the lab's transcripts, golden run records and start payloads, and the
// comparison of that text with a fixture under `tests/fixtures/`.
//
// To write the fixtures again, run a test with `TYPECV_BAKE=1`. A structure-only phase never
// bakes: a difference from a fixture is a behaviour difference, and it gets its own record.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

/** The tag key of an encoded value that JSON cannot carry. */
const TAG = '$';
/** Typed arrays up to this length are written out; longer ones are written as a digest. */
export const ARRAY_VALUES_LIMIT = 1024;

const kind = (value) => Object.prototype.toString.call(value).slice(8, -1);

/** The SHA-256 of the bytes of an array view or buffer, in hex. */
export function sha256(bytes) {
  const view = ArrayBuffer.isView(bytes)
    ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    : new Uint8Array(bytes);
  return createHash('sha256').update(view).digest('hex');
}

/**
 * Mask what depends on the machine or the build: the frames of a V8 stack trace name files,
 * lines and columns. The first line of a stack (the error's name and message) stays.
 */
export function maskText(text) {
  return text.replace(/\n\s+at [^\n]*/g, '');
}

/**
 * The canonical form of a value: JSON with tags for what JSON drops or changes. `undefined`,
 * `-0`, `NaN` and the infinities keep their identity; a typed array keeps its type; a `Map` and a
 * `Set` keep their order; an object keeps the insertion order of its keys. The function reads
 * values of any realm, so it never uses `instanceof`.
 */
export function canonical(value, seen = new Set()) {
  if (value === undefined) return { [TAG]: 'undefined' };
  if (value === null) return null;
  switch (typeof value) {
    case 'number':
      if (Object.is(value, -0)) return { [TAG]: '-0' };
      if (Number.isNaN(value)) return { [TAG]: 'NaN' };
      if (value === Infinity) return { [TAG]: 'Infinity' };
      if (value === -Infinity) return { [TAG]: '-Infinity' };
      return value;
    case 'string':
      return maskText(value);
    case 'boolean':
      return value;
    case 'bigint':
      return { [TAG]: 'bigint', value: String(value) };
    case 'function':
      return { [TAG]: 'function' };
    case 'symbol':
      return { [TAG]: 'symbol', description: String(value.description) };
  }
  if (seen.has(value)) return { [TAG]: 'cycle' };
  seen.add(value);
  try {
    const type = kind(value);
    if (ArrayBuffer.isView(value)) {
      if (type === 'DataView')
        return { [TAG]: type, length: value.byteLength, sha256: sha256(value) };
      if (value.length > ARRAY_VALUES_LIMIT)
        return { [TAG]: type, length: value.length, sha256: sha256(value) };
      return {
        [TAG]: type,
        values: Array.from(value, (item) => canonical(item, seen)),
      };
    }
    if (type === 'ArrayBuffer' || type === 'SharedArrayBuffer') {
      // A buffer that a transfer detached has no bytes left.
      return { [TAG]: type, length: value.byteLength, sha256: sha256(value) };
    }
    if (Array.isArray(value)) return Array.from(value, (item) => canonical(item, seen));
    if (type === 'Map')
      return {
        [TAG]: 'Map',
        entries: [...value].map(([key, item]) => [canonical(key, seen), canonical(item, seen)]),
      };
    if (type === 'Set')
      return { [TAG]: 'Set', values: [...value].map((item) => canonical(item, seen)) };
    if (type === 'Error')
      return { [TAG]: 'Error', name: String(value.name), message: maskText(String(value.message)) };
    if (type === 'Promise') return { [TAG]: 'Promise' };
    if (type === 'Date') return { [TAG]: 'Date', value: value.toISOString() };
    const out = {};
    for (const key of Object.keys(value)) out[key] = canonical(value[key], seen);
    return out;
  } finally {
    seen.delete(value);
  }
}

/** One JSON line per item of a list: the text form of a transcript. */
export function jsonLines(items) {
  return items.map((item) => JSON.stringify(item)).join('\n') + '\n';
}

/** The first line where two texts differ, with a few lines around it, for a failure message. */
function firstDifference(expected, actual) {
  const a = expected.split('\n'),
    b = actual.split('\n');
  let line = 0;
  while (line < a.length && line < b.length && a[line] === b[line]) line++;
  const around = (lines) =>
    lines
      .slice(Math.max(0, line - 2), line + 3)
      .map((text) => (text.length > 400 ? text.slice(0, 400) + ' …' : text))
      .join('\n');
  return `first difference at line ${line + 1}\n--- fixture\n${around(a)}\n+++ actual\n${around(b)}`;
}

/**
 * Compare `text` with the fixture at `url` byte for byte. With `TYPECV_BAKE=1` the fixture is
 * written instead, and the test reports what it wrote.
 */
export async function checkBaseline(url, text) {
  const path = url instanceof URL ? url.pathname : url;
  if (process.env.TYPECV_BAKE === '1') {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, text);
    return;
  }
  let expected;
  try {
    expected = await readFile(path, 'utf8');
  } catch (error) {
    throw new Error(`Missing baseline ${path}; bake it with TYPECV_BAKE=1`, { cause: error });
  }
  if (expected !== text)
    assert.fail(`${path} differs from the baseline: ${firstDifference(expected, text)}`);
}
