/**
 * Loading a video from a URL: the parsing and the checks that decide, before and after the
 * download, whether a URL and its response can be a video file. The page fetches the bytes
 * itself (`fetch` with CORS), so there is no proxy, and hands them on as a `File`.
 */

/** The largest download: the lab has no upload limit, so a URL load stops at 2 GiB. */
export const MAX_VIDEO_URL_BYTES = 2 * 1024 * 1024 * 1024;

/** Longest origin-and-path text a run record keeps (the record's name field allows 128). */
export const MAX_RECORDED_URL_LENGTH = 128;

export type VideoUrlErrorCode =
  'empty' | 'invalid' | 'scheme' | 'page' | 'cors' | 'status' | 'type' | 'size';

export class VideoUrlError extends Error {
  readonly code: VideoUrlErrorCode;
  constructor(code: VideoUrlErrorCode, message: string) {
    super(message);
    this.name = 'VideoUrlError';
    this.code = code;
  }
}

const MESSAGES: Record<VideoUrlErrorCode, [string, string]> = {
  empty: ['Enter a video URL.', '영상 URL을 입력하세요.'],
  invalid: ['This is not a valid URL.', '올바른 URL이 아닙니다.'],
  scheme: ['Only http and https URLs are accepted.', 'http와 https URL만 사용할 수 있습니다.'],
  page: ['This is a page, not a video file.', '영상 파일이 아니라 웹 페이지입니다.'],
  cors: [
    'The server does not allow this page to read the file (CORS). Use a direct file link from a host that sends Access-Control-Allow-Origin, or download the file and upload it.',
    '서버가 이 페이지에서 파일을 읽는 것을 허용하지 않습니다(CORS). Access-Control-Allow-Origin을 보내는 호스트의 직접 파일 링크를 쓰거나, 파일을 내려받아 업로드하세요.',
  ],
  status: ['The server answered with status {0}.', '서버가 상태 {0}으로 응답했습니다.'],
  type: [
    'The server sent "{0}", which is not a video. Use a direct link to an MP4 or WebM file.',
    '서버가 영상이 아닌 "{0}"을 보냈습니다. MP4나 WebM 파일의 직접 링크를 사용하세요.',
  ],
  size: ['The video is larger than the limit of 2 GiB.', '영상이 한도인 2 GiB보다 큽니다.'],
};

/** The message of an error code in one language; `detail` fills `{0}`. */
export function videoUrlMessage(code: VideoUrlErrorCode, ko: boolean, detail = ''): string {
  return MESSAGES[code][ko ? 1 : 0].replace('{0}', detail);
}

function fail(code: VideoUrlErrorCode, ko: boolean, detail?: string): never {
  throw new VideoUrlError(code, videoUrlMessage(code, ko, detail));
}

/** Hosts whose links are watch pages, never files. */
const PAGE_HOSTS = [
  'youtube.com',
  'youtu.be',
  'vimeo.com',
  'dailymotion.com',
  'twitch.tv',
  'tiktok.com',
  'instagram.com',
  'facebook.com',
  'naver.com',
  'bilibili.com',
];

function isPageHost(hostname: string): boolean {
  const host = hostname.toLowerCase();
  return PAGE_HOSTS.some((page) => host === page || host.endsWith('.' + page));
}

/** Parse and check a URL typed by the person. Returns it as a normal URL, or throws. */
export function parseVideoUrl(input: string, ko = false): URL {
  const raw = input.trim();
  if (!raw) fail('empty', ko);
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return fail('invalid', ko);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') fail('scheme', ko);
  if (isPageHost(url.hostname)) fail('page', ko);
  return url;
}

/** Check the response headers: status, content type and the declared size. */
export function checkVideoResponse(
  response: { ok: boolean; status: number; headers: { get(name: string): string | null } },
  ko = false,
): { total: number | undefined; type: string } {
  if (!response.ok) fail('status', ko, String(response.status));
  const type = (response.headers.get('content-type') ?? '').split(';')[0].trim().toLowerCase();
  // A page answers text/html; a generic binary type is accepted, since many hosts send it.
  if (type === 'text/html' || type === 'application/xhtml+xml') fail('page', ko);
  if (
    type &&
    !type.startsWith('video/') &&
    type !== 'application/octet-stream' &&
    type !== 'binary/octet-stream' &&
    type !== 'application/mp4' &&
    type !== 'application/x-matroska'
  )
    fail('type', ko, type);
  const length = Number(response.headers.get('content-length'));
  const total = Number.isFinite(length) && length > 0 ? length : undefined;
  if (total !== undefined && total > MAX_VIDEO_URL_BYTES) fail('size', ko);
  return { total, type };
}

/** The file name for a downloaded video: the last path segment, or `video`. */
export function videoUrlFileName(url: URL): string {
  const last = decodeURIComponentSafe(url.pathname.split('/').filter(Boolean).at(-1) ?? '');
  return last || 'video';
}

function decodeURIComponentSafe(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

/** What a run record keeps of the URL: origin and path only. The query string and the
 *  fragment may hold tokens, and the user name and password never leave the page. */
export function recordedVideoUrl(url: URL): string {
  return (url.origin + url.pathname).slice(0, MAX_RECORDED_URL_LENGTH);
}

export type VideoUrlProgress = { loaded: number; total: number | undefined };

/** The progress text, in bytes and (when the total is known) in percent. */
export function formatVideoUrlProgress(progress: VideoUrlProgress, ko: boolean): string {
  const loaded = formatBytes(progress.loaded);
  if (progress.total === undefined) return (ko ? '내려받는 중: ' : 'Downloading: ') + loaded;
  const percent = Math.min(100, Math.floor((progress.loaded / progress.total) * 100));
  return `${ko ? '내려받는 중: ' : 'Downloading: '}${loaded} / ${formatBytes(progress.total)} (${percent}%)`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KiB';
  if (bytes < 1024 ** 3) return (bytes / 1024 ** 2).toFixed(1) + ' MiB';
  return (bytes / 1024 ** 3).toFixed(2) + ' GiB';
}

/**
 * Download a video into a `File`. A network or CORS failure surfaces from `fetch` as a
 * `TypeError`, which the page cannot tell apart; both map to the CORS message.
 */
export async function fetchVideoFile(
  input: string,
  options: {
    ko?: boolean;
    signal?: AbortSignal;
    onProgress?: (progress: VideoUrlProgress) => void;
    fetchImpl?: typeof fetch;
  } = {},
): Promise<{ file: File; url: URL }> {
  const ko = options.ko ?? false;
  const url = parseVideoUrl(input, ko);
  let response: Response;
  try {
    response = await (options.fetchImpl ?? fetch)(url.href, {
      mode: 'cors',
      signal: options.signal,
    });
  } catch (error) {
    if ((error as Error)?.name === 'AbortError') throw error;
    return fail('cors', ko);
  }
  const { total, type } = checkVideoResponse(response, ko);
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  options.onProgress?.({ loaded, total });
  if (response.body) {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      if (loaded > MAX_VIDEO_URL_BYTES) {
        await reader.cancel();
        fail('size', ko);
      }
      chunks.push(value);
      options.onProgress?.({ loaded, total });
    }
  } else {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > MAX_VIDEO_URL_BYTES) fail('size', ko);
    chunks.push(bytes);
    loaded = bytes.byteLength;
    options.onProgress?.({ loaded, total: loaded });
  }
  const blob = new Blob(chunks as BlobPart[], { type: type.startsWith('video/') ? type : '' });
  return { file: new File([blob], videoUrlFileName(url), { type: blob.type }), url };
}
