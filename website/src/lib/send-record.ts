/**
 * Send one record to the site's collector (`website/worker.mjs`, `POST /api/records/`): a run
 * record of the SfM lab or a probe record of `/lab/probe/`. The page validates the record first;
 * this posts it and reads the collector's answer.
 */
import {
  RESULT_FILE_MAX_BYTES,
  RESULT_FILE_MAX_COUNT,
  RESULT_FILE_PART_BYTES,
  RESULT_FILES_MAX_BYTES,
  RUN_RECORD_MAX_BYTES,
  resultFileBaseNameValid,
  type ResultFileBaseName,
  type ResultFileName,
  type RunRecord,
  type RunRecordFile,
} from './run-record.ts';

export type ResultFileData = Uint8Array | string | Blob;
/** A producer regenerates a serialized export without retaining every export in the plan. */
export type ResultFilePayload = {
  name: ResultFileBaseName;
  data: ResultFileData | (() => ResultFileData | Promise<ResultFileData>);
};
type UploadBody = Uint8Array | Blob;
export type PlannedUpload = {
  name: ResultFileName;
  body: UploadBody | (() => Promise<UploadBody>);
  /** Release the one cached original after its final part, including a failed PUT. */
  release?: () => void;
};

/** The record's `files` entries, and the bytes of the files that the page uploads. */
export type FilePlan = {
  entries: RunRecordFile[];
  uploads: PlannedUpload[];
};

export type FileOutcome = { name: ResultFileName; reason: string };

export type SendState =
  | { status: 'idle' }
  | { status: 'sending'; progress?: { done: number; total: number } }
  | {
      status: 'sent';
      id: string;
      /** Present when the send had result files: the files not stored, with the reason. */
      files?: { stored: ResultFileName[]; failed: FileOutcome[] };
    }
  | { status: 'failed'; message: string };

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function prepared(data: ResultFileData): UploadBody {
  if (typeof data === 'string') return new TextEncoder().encode(data);
  if (data instanceof Blob || data instanceof Uint8Array) return data;
  throw new TypeError('Result producer must return text, bytes or a Blob');
}
const sizeOf = (body: UploadBody) => (body instanceof Blob ? body.size : body.byteLength);
const sliceOf = (body: UploadBody, offset: number, end: number): UploadBody =>
  body instanceof Blob ? body.slice(offset, end) : body.subarray(offset, end);
const reasonOf = (error: unknown) => String(error).slice(0, 200) || 'File preparation failed';

/** Keep planning's serialized bytes outside the closures retained for later upload. */
function regenerator(produce: () => ResultFileData | Promise<ResultFileData>, size: number) {
  let cached: Promise<UploadBody> | undefined;
  return {
    async read(offset: number, end: number): Promise<UploadBody> {
      cached ??= Promise.resolve().then(produce).then(prepared);
      const current = await cached;
      if (sizeOf(current) !== size) throw new Error('Result size changed after planning');
      return offset === 0 && end === size ? current : sliceOf(current, offset, end);
    },
    release() {
      cached = undefined;
    },
  };
}

/**
 * Read one original at a time. A failed producer/read is represented honestly as a skipped
 * entry; it never prevents later files or the run JSON from being sent. Large originals use
 * ordered <=16 MiB parts, each with its own digest and explicit reassembly metadata.
 */
export async function planFiles(files: Iterable<ResultFilePayload>): Promise<FilePlan> {
  const plan: FilePlan = { entries: [], uploads: [] };
  const names = new Set<string>();
  let totalBytes = 0;
  for (const file of files) {
    if (!resultFileBaseNameValid(file.name) || names.has(file.name))
      throw new RangeError('Invalid or duplicate result file name: ' + file.name);
    names.add(file.name);
    if (plan.entries.length >= RESULT_FILE_MAX_COUNT) {
      // Keep the manifest valid and persist an explicit overflow outcome rather than
      // dropping further attachments silently or preventing the run JSON from being sent.
      const last = plan.entries.at(-1)!;
      plan.entries[plan.entries.length - 1] = {
        ...last,
        skipped:
          'Attachment count limit (1024); further files beginning with ' +
          file.name +
          ' were not prepared',
      };
      const upload = plan.uploads.findIndex((entry) => entry.name === last.name);
      if (upload >= 0) {
        const [removed] = plan.uploads.splice(upload, 1);
        const previous = plan.uploads.at(-1);
        if (
          last.part &&
          removed!.release &&
          previous?.name.startsWith(last.part.original + '.part-')
        )
          previous.release = removed!.release;
      }
      break;
    }
    const lazy = typeof file.data === 'function';
    const produce = lazy
      ? (file.data as () => ResultFileData | Promise<ResultFileData>)
      : () => file.data as ResultFileData;
    let body: UploadBody;
    try {
      body = prepared(await produce());
    } catch (error) {
      plan.entries.push({ name: file.name, skipped: reasonOf(error) });
      continue;
    }
    const size = sizeOf(body);
    if (!Number.isSafeInteger(size) || size < 0) {
      plan.entries.push({ name: file.name, skipped: 'Invalid result file size' });
      continue;
    }
    const partCount = size > RESULT_FILE_MAX_BYTES ? Math.ceil(size / RESULT_FILE_PART_BYTES) : 1;
    if (
      size > RESULT_FILES_MAX_BYTES - totalBytes ||
      plan.entries.length + partCount > RESULT_FILE_MAX_COUNT
    ) {
      plan.entries.push({
        name: file.name,
        bytes: size,
        skipped:
          size > RESULT_FILES_MAX_BYTES - totalBytes
            ? 'Total attachment limit (4 GiB) would be exceeded'
            : 'Attachment count limit (1024) would be exceeded',
      });
      continue;
    }
    // A factory is called once more at PUT time, and only this original stays cached until
    // its last part. Planning's serialization is not captured by these loaders.
    const regenerate = lazy ? regenerator(produce, size) : undefined;
    for (let index = 0; index < partCount; index++) {
      const offset = partCount === 1 ? 0 : index * RESULT_FILE_PART_BYTES;
      const end = partCount === 1 ? size : Math.min(size, offset + RESULT_FILE_PART_BYTES);
      const name: ResultFileName =
        partCount === 1 ? file.name : `${file.name}.part-${String(index).padStart(4, '0')}`;
      const entry: RunRecordFile = {
        name,
        bytes: end - offset,
        ...(partCount > 1 ? { part: { original: file.name, totalBytes: size, offset } } : {}),
      };
      try {
        const part = partCount === 1 ? body : sliceOf(body, offset, end);
        const bytes = part instanceof Blob ? new Uint8Array(await part.arrayBuffer()) : part;
        const digest = await sha256Hex(bytes);
        plan.entries.push({ ...entry, sha256: digest });
        totalBytes += end - offset;
        plan.uploads.push({
          name,
          body: regenerate ? () => regenerate.read(offset, end) : part,
          ...(regenerate && index === partCount - 1 ? { release: regenerate.release } : {}),
        });
      } catch (error) {
        plan.entries.push({ ...entry, skipped: reasonOf(error) });
        // Even when the final part could not be prepared, the preceding loader must release
        // the cached original after its last actual PUT.
        if (
          regenerate &&
          index === partCount - 1 &&
          plan.uploads.at(-1)?.name.startsWith(file.name)
        )
          plan.uploads.at(-1)!.release = regenerate.release;
      }
    }
  }
  return plan;
}

/** Whether the collector has a store for result files (`GET /api/config`). */
export async function resultFilesAvailable(): Promise<boolean> {
  try {
    const response = await fetch('/api/config/');
    if (!response.ok) return false;
    return (await response.json()).files === true;
  } catch {
    return false;
  }
}

/** Upload the planned files one after the other; a failed file does not stop the next. */
export async function putFiles(
  id: string,
  plan: FilePlan,
  onProgress?: (done: number, total: number) => void,
): Promise<{ stored: ResultFileName[]; failed: FileOutcome[] }> {
  const stored: ResultFileName[] = [],
    failed: FileOutcome[] = [];
  for (const entry of plan.entries)
    if (entry.skipped) failed.push({ name: entry.name, reason: entry.skipped });
  for (const [index, upload] of plan.uploads.entries()) {
    onProgress?.(index, plan.uploads.length);
    try {
      const loaded = typeof upload.body === 'function' ? await upload.body() : upload.body;
      // Use fresh bounded bytes rather than asking fetch to reread a disk-backed Blob after
      // its successful hash read. A read failure belongs to this file, not to the record.
      const bytes = loaded instanceof Blob ? new Uint8Array(await loaded.arrayBuffer()) : loaded;
      const response = await fetch('/api/records/' + id + '/files/' + upload.name, {
        method: 'PUT',
        headers: { 'content-type': 'application/octet-stream' },
        body: bytes as Uint8Array<ArrayBuffer>,
      });
      if (response.status === 201) stored.push(upload.name);
      else {
        const body = await response.json().catch(() => ({}));
        // The immutable manifest authenticated this exact name/digest on its original PUT.
        // An existing stored file is a completed retry, never replaced by this request.
        if (response.status === 409 && body.error === 'already stored') stored.push(upload.name);
        else
          failed.push({
            name: upload.name,
            reason: 'HTTP ' + response.status + (body.error ? ': ' + body.error : ''),
          });
      }
    } catch (error) {
      failed.push({ name: upload.name, reason: String(error) });
    } finally {
      upload.release?.();
    }
  }
  onProgress?.(plan.uploads.length, plan.uploads.length);
  return { stored, failed };
}

/** Post a validated record same-origin and return what the page shows of the answer. */
export async function postRecord(
  record: unknown,
  ko: boolean,
  plan?: FilePlan,
  onProgress?: (done: number, total: number) => void,
): Promise<SendState> {
  try {
    let excluded: FileOutcome[] | undefined;
    if (plan)
      record = { ...(record as RunRecord), files: plan.entries, filePreparationFailure: undefined };
    let payload = JSON.stringify(record);
    if (plan && new TextEncoder().encode(payload).byteLength > RUN_RECORD_MAX_BYTES) {
      const reason =
        'Attachment manifest exceeds the 256 KiB record limit; attachments were not declared';
      excluded = plan.entries.map((entry) => ({
        name: entry.name,
        reason: entry.skipped || reason,
      }));
      const {
        files: _files,
        filePreparationFailure: _previousFailure,
        ...base
      } = record as RunRecord;
      record = { ...base, filePreparationFailure: { reason, requested: plan.entries.length } };
      payload = JSON.stringify(record);
      if (new TextEncoder().encode(payload).byteLength > RUN_RECORD_MAX_BYTES) {
        record = base;
        payload = JSON.stringify(base);
      }
    }
    if (new TextEncoder().encode(payload).byteLength > RUN_RECORD_MAX_BYTES)
      return {
        status: 'failed',
        message: ko
          ? '실행 기록이 256 KiB 한도를 넘습니다. JSON을 다운로드해 보관해 주세요.'
          : 'The run record exceeds 256 KiB. Download its JSON to preserve it.',
      };
    // With the trailing slash: the host redirects `/api/records` to `/api/records/` with a 301,
    // which a browser follows as a GET.
    const response = await fetch('/api/records/', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: payload,
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 201 && typeof body.id === 'string') {
      if (excluded) return { status: 'sent', id: body.id, files: { stored: [], failed: excluded } };
      if (!plan) return { status: 'sent', id: body.id };
      return { status: 'sent', id: body.id, files: await putFiles(body.id, plan, onProgress) };
    }
    if (response.status === 503)
      return {
        status: 'failed',
        message: ko
          ? '수집 서버가 아직 설정되지 않았습니다. 기록을 다운로드해 전달해 주세요.'
          : 'The collector is not configured yet. Download the record and share it instead.',
      };
    if (response.status === 429)
      return {
        status: 'failed',
        message: ko ? '전송 한도(시간당 30건)를 넘었습니다.' : 'Rate limit reached (30 per hour).',
      };
    return {
      status: 'failed',
      message: 'HTTP ' + response.status + (body.error ? ': ' + body.error : ''),
    };
  } catch (error) {
    return { status: 'failed', message: String(error) };
  }
}
