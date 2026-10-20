/**
 * Send one record to the site's collector (`website/worker.mjs`, `POST /api/records/`): a run
 * record of the SfM lab or a probe record of `/lab/probe/`. The page validates the record first;
 * this posts it and reads the collector's answer.
 */
export type SendState =
  | { status: 'idle' }
  | { status: 'sending' }
  | { status: 'sent'; id: string }
  | { status: 'failed'; message: string };

/** Post a validated record same-origin and return what the page shows of the answer. */
export async function postRecord(record: unknown, ko: boolean): Promise<SendState> {
  try {
    // With the trailing slash: the host redirects `/api/records` to `/api/records/` with a 301,
    // which a browser follows as a GET.
    const response = await fetch('/api/records/', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(record),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 201 && typeof body.id === 'string')
      return { status: 'sent', id: body.id };
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
