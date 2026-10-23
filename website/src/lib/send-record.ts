/**
 * Send one record to the site's collector (`website/worker.mjs`, `POST /api/records/`): a run
 * record of the SfM lab or a probe record of `/lab/probe/`. The page validates the record first;
 * this posts it and reads the collector's answer.
 */
import {
  RESULT_FILE_MAX_BYTES,
  type ResultFileName,
  type RunRecord,
  type RunRecordFile,
} from './run-record.ts';

/** A result file the page can send: the bytes of one of the lab's exports. */
export type ResultFilePayload = { name: ResultFileName; data: Uint8Array | string | Blob };

/** The record's `files` entries, and the bytes of the files that the page uploads. */
export type FilePlan = {
  entries: RunRecordFile[];
  uploads: { name: ResultFileName; body: Uint8Array | Blob }[];
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

/**
 * The upload plan of the result files: a `files` entry for each file with its size and SHA-256,
 * and the bytes of each file that fits `RESULT_FILE_MAX_BYTES`. A larger file gets a `skipped`
 * entry and no upload.
 */
export async function planFiles(files: readonly ResultFilePayload[]): Promise<FilePlan> {
  const plan: FilePlan = { entries: [], uploads: [] };
  for (const file of files) {
    const body = typeof file.data === 'string' ? new TextEncoder().encode(file.data) : file.data;
    const bytes = body instanceof Blob ? new Uint8Array(await body.arrayBuffer()) : body;
    const size = bytes.byteLength;
    const entry: RunRecordFile = {
      name: file.name,
      bytes: size,
      sha256: await sha256Hex(bytes),
      ...(size > RESULT_FILE_MAX_BYTES
        ? { skipped: 'larger than ' + RESULT_FILE_MAX_BYTES / 1024 ** 2 + ' MB' }
        : {}),
    };
    plan.entries.push(entry);
    if (!entry.skipped) plan.uploads.push({ name: file.name, body });
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
async function putFiles(
  id: string,
  plan: FilePlan,
  onProgress: ((done: number, total: number) => void) | undefined,
): Promise<{ stored: ResultFileName[]; failed: FileOutcome[] }> {
  const stored: ResultFileName[] = [],
    failed: FileOutcome[] = [];
  for (const entry of plan.entries)
    if (entry.skipped) failed.push({ name: entry.name, reason: entry.skipped });
  for (const [index, upload] of plan.uploads.entries()) {
    onProgress?.(index, plan.uploads.length);
    try {
      const response = await fetch('/api/records/' + id + '/files/' + upload.name, {
        method: 'PUT',
        headers: { 'content-type': 'application/octet-stream' },
        body: upload.body as Blob | Uint8Array<ArrayBuffer>,
      });
      if (response.status === 201) stored.push(upload.name);
      else {
        const body = await response.json().catch(() => ({}));
        failed.push({
          name: upload.name,
          reason: 'HTTP ' + response.status + (body.error ? ': ' + body.error : ''),
        });
      }
    } catch (error) {
      failed.push({ name: upload.name, reason: String(error) });
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
    if (plan) record = { ...(record as RunRecord), files: plan.entries };
    // With the trailing slash: the host redirects `/api/records` to `/api/records/` with a 301,
    // which a browser follows as a GET.
    const response = await fetch('/api/records/', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(record),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 201 && typeof body.id === 'string') {
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
