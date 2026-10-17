export interface PreparedVideo {
  format: 'typecv-sfm-video-v1';
  source: { name: string; width: number; height: number; duration: number; fps: number };
  frames: { id: number; width: number; height: number; time: number; image: string }[];
}

/** Invert a pixel-center resize, including independently rounded x/y analysis extents. */
export function cameraInSourcePixels<T extends { fx: number; fy: number; cx: number; cy: number }>(
  camera: T,
  analysis: { width: number; height: number },
  source: { width: number; height: number },
): T {
  const sx = source.width / analysis.width,
    sy = source.height / analysis.height;
  return {
    ...camera,
    fx: camera.fx * sx,
    fy: camera.fy * sy,
    cx: (camera.cx + 0.5) * sx - 0.5,
    cy: (camera.cy + 0.5) * sy - 0.5,
  };
}

/** Validate the bounded analysis images before allocating browser pixel buffers. */
export function parsePreparedVideo(text: string): PreparedVideo {
  if (text.length > 64 * 1024 * 1024) throw new RangeError('Frame bundle exceeds 64 MiB.');
  const data = JSON.parse(text) as PreparedVideo;
  const dimension = (value: number) => Number.isSafeInteger(value) && value > 0;
  if (
    data?.format !== 'typecv-sfm-video-v1' ||
    !data.source ||
    typeof data.source.name !== 'string' ||
    !dimension(data.source.width) ||
    !dimension(data.source.height) ||
    !Number.isFinite(data.source.duration) ||
    data.source.duration <= 0 ||
    !Number.isFinite(data.source.fps) ||
    data.source.fps <= 0 ||
    !Array.isArray(data.frames) ||
    data.frames.length < 2 ||
    data.frames.length > 256
  )
    throw new RangeError('Invalid TypeCV video frame bundle.');
  const ids = new Set<number>();
  let pixels = 0;
  let lastTime = -1;
  for (const frame of data.frames) {
    if (
      !frame ||
      !Number.isSafeInteger(frame.id) ||
      frame.id < 0 ||
      ids.has(frame.id) ||
      !dimension(frame.width) ||
      !dimension(frame.height) ||
      Math.max(frame.width, frame.height) > 4096 ||
      frame.width > data.source.width ||
      frame.height > data.source.height ||
      !Number.isFinite(frame.time) ||
      frame.time < 0 ||
      frame.time <= lastTime ||
      frame.time >= data.source.duration ||
      typeof frame.image !== 'string' ||
      !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(frame.image)
    )
      throw new RangeError('Invalid or unbounded analysis frame.');
    // Permit rounding by one analysis pixel while retaining the source aspect ratio.
    if (Math.abs(frame.height - (frame.width * data.source.height) / data.source.width) > 1)
      throw new RangeError('Analysis frame aspect ratio differs from the source video.');
    pixels += frame.width * frame.height;
    if (pixels > 64 * 1024 * 1024)
      throw new RangeError('Analysis frames exceed the 64-megapixel bundle budget.');
    ids.add(frame.id);
    lastTime = frame.time;
  }
  // Inspect the PNG header before any decoder can allocate the encoded image extent.
  for (const frame of data.frames) {
    let bytes: Uint8Array;
    try {
      bytes = Uint8Array.from(
        atob(
          frame.image.slice('data:image/png;base64,'.length, 'data:image/png;base64,'.length + 44),
        ),
        (character) => character.charCodeAt(0),
      );
    } catch {
      throw new RangeError('Invalid PNG header in frame bundle.');
    }
    const signature = [137, 80, 78, 71, 13, 10, 26, 10];
    if (
      bytes.length < 33 ||
      !signature.every((value, index) => bytes[index] === value) ||
      new DataView(bytes.buffer).getUint32(8) !== 13 ||
      String.fromCharCode(...bytes.subarray(12, 16)) !== 'IHDR' ||
      new DataView(bytes.buffer).getUint32(16) !== frame.width ||
      new DataView(bytes.buffer).getUint32(20) !== frame.height
    )
      throw new RangeError('Encoded PNG dimensions do not match the bounded frame declaration.');
  }
  return data;
}
