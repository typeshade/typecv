/**
 * A reader for the PLY files the lab writes and the usual ones around them: `ascii 1.0`,
 * `binary_little_endian 1.0` and `binary_big_endian 1.0`. It reads the `vertex` element (x, y, z,
 * optional nx, ny, nz and red, green, blue) and the `face` element (a list of vertex indices,
 * triangulated as a fan). Every other element is skipped, and every property keeps its place in
 * the row, so a file with extra properties still reads.
 */

export type PlyFormat = 'ascii' | 'binary_little_endian' | 'binary_big_endian';

export type Ply = {
  format: PlyFormat;
  comments: string[];
  /** x, y, z per vertex. */
  positions: Float32Array;
  /** r, g, b per vertex, bytes; undefined when the file has no color. */
  colors?: Uint8Array;
  /** nx, ny, nz per vertex; undefined when the file has no normals. */
  normals?: Float32Array;
  /** Three vertex indices per triangle; undefined when the file has no `face` element. */
  indices?: Uint32Array;
};

/** The largest header the reader accepts. */
export const PLY_MAX_HEADER = 64 * 1024;

const TYPES: Record<string, { size: number; name: string }> = {
  char: { size: 1, name: 'Int8' },
  int8: { size: 1, name: 'Int8' },
  uchar: { size: 1, name: 'Uint8' },
  uint8: { size: 1, name: 'Uint8' },
  short: { size: 2, name: 'Int16' },
  int16: { size: 2, name: 'Int16' },
  ushort: { size: 2, name: 'Uint16' },
  uint16: { size: 2, name: 'Uint16' },
  int: { size: 4, name: 'Int32' },
  int32: { size: 4, name: 'Int32' },
  uint: { size: 4, name: 'Uint32' },
  uint32: { size: 4, name: 'Uint32' },
  float: { size: 4, name: 'Float32' },
  float32: { size: 4, name: 'Float32' },
  double: { size: 8, name: 'Float64' },
  float64: { size: 8, name: 'Float64' },
};

type Property = { name: string; type: string; list?: { count: string } };
type Element = { name: string; count: number; properties: Property[] };

function fail(message: string): never {
  throw new RangeError('PLY: ' + message);
}

function getValue(view: DataView, name: string, at: number, little: boolean): number {
  switch (name) {
    case 'Int8':
      return view.getInt8(at);
    case 'Uint8':
      return view.getUint8(at);
    case 'Int16':
      return view.getInt16(at, little);
    case 'Uint16':
      return view.getUint16(at, little);
    case 'Int32':
      return view.getInt32(at, little);
    case 'Uint32':
      return view.getUint32(at, little);
    case 'Float32':
      return view.getFloat32(at, little);
    default:
      return view.getFloat64(at, little);
  }
}

function typeOf(name: string) {
  return TYPES[name] ?? fail('unknown property type "' + name + '"');
}

function parseHeader(bytes: Uint8Array) {
  const limit = Math.min(bytes.length, PLY_MAX_HEADER);
  const text = new TextDecoder('latin1').decode(bytes.subarray(0, limit));
  const end = text.indexOf('end_header');
  if (!text.startsWith('ply') || end < 0) fail('not a PLY file or the header is missing');
  let body = end + 'end_header'.length;
  // The header ends at the first line break after `end_header`: "\n" or "\r\n".
  if (text[body] === '\r') body++;
  if (text[body] !== '\n') fail('the header does not end with a line break');
  body++;
  const lines = text.slice(0, end).split(/\r?\n/);
  let format: PlyFormat | undefined;
  const comments: string[] = [];
  const elements: Element[] = [];
  for (const line of lines.slice(1)) {
    const words = line.trim().split(/\s+/);
    switch (words[0]) {
      case 'format':
        if (
          words[1] !== 'ascii' &&
          words[1] !== 'binary_little_endian' &&
          words[1] !== 'binary_big_endian'
        )
          fail('unknown format "' + words[1] + '"');
        format = words[1];
        break;
      case 'comment':
      case 'obj_info':
        comments.push(line.trim().slice(words[0].length).trim());
        break;
      case 'element': {
        const count = Number(words[2]);
        if (!words[1] || !Number.isInteger(count) || count < 0) fail('bad element line');
        elements.push({ name: words[1], count, properties: [] });
        break;
      }
      case 'property': {
        const element = elements.at(-1) ?? fail('a property before any element');
        if (words[1] === 'list') {
          if (words.length < 5) fail('bad list property');
          typeOf(words[2]);
          typeOf(words[3]);
          element.properties.push({ name: words[4], type: words[3], list: { count: words[2] } });
        } else {
          if (words.length < 3) fail('bad property');
          typeOf(words[1]);
          element.properties.push({ name: words[2], type: words[1] });
        }
        break;
      }
      case '':
        break;
      default:
        fail('unknown header line "' + line.trim().slice(0, 40) + '"');
    }
  }
  if (!format) fail('no format line');
  return { format, comments, elements, body };
}

/**
 * Parse a PLY file. Throws a `RangeError` that starts with "PLY:" on a file it cannot read,
 * including a binary file that is shorter than its header says.
 */
export function parsePly(bytes: Uint8Array): Ply {
  const { format, comments, elements, body } = parseHeader(bytes);
  const vertexElement = elements.find((element) => element.name === 'vertex');
  if (!vertexElement) fail('no vertex element');
  const names = vertexElement.properties.map((property) => property.name);
  for (const axis of ['x', 'y', 'z']) if (!names.includes(axis)) fail('no "' + axis + '" property');
  const hasColor = ['red', 'green', 'blue'].every((name) => names.includes(name));
  const hasNormal = ['nx', 'ny', 'nz'].every((name) => names.includes(name));
  const faceElement = elements.find((element) => element.name === 'face');
  const n = vertexElement.count;
  // A binary body that cannot hold the vertices is refused before any allocation.
  const positions = new Float32Array(3 * n);
  const colors = hasColor ? new Uint8Array(3 * n) : undefined;
  const normals = hasNormal ? new Float32Array(3 * n) : undefined;
  const slot: Record<string, [Float32Array | Uint8Array, number] | undefined> = {
    x: [positions, 0],
    y: [positions, 1],
    z: [positions, 2],
    ...(colors && { red: [colors, 0], green: [colors, 1], blue: [colors, 2] }),
    ...(normals && { nx: [normals, 0], ny: [normals, 1], nz: [normals, 2] }),
  };
  const faces: number[] = [];
  const clampByte = (value: number) => (value < 0 ? 0 : value > 255 ? 255 : Math.round(value));
  const store = (element: Element, row: number, property: Property, value: number) => {
    if (element.name !== 'vertex') return;
    const target = slot[property.name];
    if (!target) return;
    target[0][3 * row + target[1]] = target[0] === colors ? clampByte(value) : value;
  };

  if (format === 'ascii') {
    const tokens = new TextDecoder('latin1')
      .decode(bytes.subarray(body))
      .split(/\s+/)
      .filter((word) => word.length > 0);
    let at = 0;
    const next = () => {
      if (at >= tokens.length) fail('the file ends before its elements do');
      const value = Number(tokens[at++]);
      if (Number.isNaN(value)) fail('bad number "' + tokens[at - 1] + '"');
      return value;
    };
    for (const element of elements) {
      for (let row = 0; row < element.count; row++) {
        for (const property of element.properties) {
          if (property.list) {
            const length = next();
            const items: number[] = [];
            for (let i = 0; i < length; i++) items.push(next());
            if (element === faceElement && /^vertex_?ind(ex|ices)$/.test(property.name))
              fan(items, faces);
          } else store(element, row, property, next());
        }
      }
    }
  } else {
    const little = format === 'binary_little_endian';
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let at = body;
    const read = (type: string) => {
      const { size, name } = typeOf(type);
      if (at + size > bytes.length) fail('the file ends before its elements do');
      const value = getValue(view, name, at, little);
      at += size;
      return value;
    };
    for (const element of elements) {
      const fixed =
        element.properties.every((property) => !property.list) &&
        element.properties.reduce((sum, property) => sum + typeOf(property.type).size, 0);
      if (fixed && at + fixed * element.count > bytes.length)
        fail('the file ends before its elements do');
      for (let row = 0; row < element.count; row++) {
        for (const property of element.properties) {
          if (property.list) {
            const length = read(property.list.count);
            const items: number[] = [];
            const wanted = element === faceElement && /^vertex_?ind(ex|ices)$/.test(property.name);
            for (let i = 0; i < length; i++) {
              const value = read(property.type);
              if (wanted) items.push(value);
            }
            if (wanted) fan(items, faces);
          } else store(element, row, property, read(property.type));
        }
      }
    }
  }
  let indices: Uint32Array | undefined;
  if (faceElement) {
    indices = Uint32Array.from(faces);
    for (const index of indices) if (index >= n) fail('a face names vertex ' + index + ' of ' + n);
  }
  return { format, comments, positions, colors, normals, indices };
}

/** Append the triangle fan of one polygon. */
function fan(polygon: number[], out: number[]) {
  for (let i = 1; i + 1 < polygon.length; i++) out.push(polygon[0], polygon[i], polygon[i + 1]);
}
