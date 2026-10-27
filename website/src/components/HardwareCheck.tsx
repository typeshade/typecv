import { useEffect, useRef, useState } from 'react';
import {
  failedCases,
  hardwareCheckProblem,
  HARDWARE_CHECK_MAX_NOTE,
  type HardwareCheckRecord,
} from '../lib/hardware-check/record';
import { postRecord, type SendState } from '../lib/send-record';

type State = 'idle' | 'running' | 'done' | 'error';

/** The page's `?cases=<name>,…`: the cases to run, as a record's `resume` names them. */
function chosenCases(): string[] {
  const value = new URLSearchParams(location.search).get('cases') ?? '';
  return /^[a-z0-9-]+(,[a-z0-9-]+)*$/.test(value) ? value.split(',') : [];
}

const TEXT = {
  en: {
    noteLabel:
      'Your device: the GPU model and the computer. The browser does not report the model.',
    notePlaceholder: 'For example: RTX 5060 Ti, Windows 11, office computer',
    run: 'Run check',
    copy: 'Copy JSON',
    copied: 'Copied',
    download: 'Download JSON',
    idle: 'Type your device, then select “Run check”. The check takes about a minute. Keep this tab open.',
    running: 'The check is running. Keep this tab open.',
    done: 'The check is complete. Send, copy or download the JSON.',
    error: 'The check stopped with an error.',
    passed: 'Every case that ran is within its bound.',
    failed: 'Cases outside their bound or not complete: ',
    resume: 'To run the cases that did not run, open this page with ',
    log: 'Log',
    result: 'Result',
    sendTitle: 'Send the result',
    sendHelp:
      'The result holds the GPU adapter strings, limits and features, the browser user agent, platform, language and core count, the page address, the site commit, the device you typed, and for each case its outcome, digests, times and any device loss. It holds no image, file name or IP address. The JSON below is everything that is sent.',
    consent:
      'I agree to send the check result to the TypeCV developers. Checked before a run, the result is sent as soon as the check ends.',
    send: 'Send result',
    sent: 'Sent: ',
    notSent: 'Not sent: ',
    sending: 'Sending…',
    unsent: 'Not sent yet.',
  },
  ko: {
    noteLabel: '기기: GPU 모델과 컴퓨터를 적어 주세요. 브라우저는 GPU 모델을 알려 주지 않습니다.',
    notePlaceholder: '예: RTX 5060 Ti, Windows 11, 회사 컴퓨터',
    run: '점검 실행',
    copy: 'JSON 복사',
    copied: '복사됨',
    download: 'JSON 내려받기',
    idle: '기기를 적은 뒤 “점검 실행”을 누르세요. 점검은 1분 정도 걸립니다. 이 탭을 열어 두세요.',
    running: '점검이 실행 중입니다. 이 탭을 열어 두세요.',
    done: '점검이 끝났습니다. JSON을 보내거나 복사하거나 내려받으세요.',
    error: '점검이 오류로 멈췄습니다.',
    passed: '실행한 모든 사례가 허용 범위 안에 있습니다.',
    failed: '허용 범위를 벗어났거나 끝나지 않은 사례: ',
    resume: '실행하지 못한 사례를 실행하려면 이 페이지를 다음 주소로 여세요: ',
    log: '로그',
    result: '결과',
    sendTitle: '결과 보내기',
    sendHelp:
      '결과에는 GPU 어댑터 문자열·한도·기능, 브라우저 User-Agent·플랫폼·언어·코어 수, 페이지 주소, 사이트 커밋, 적은 기기 이름, 그리고 사례마다 결과·다이제스트·시간·장치 손실이 들어갑니다. 이미지, 파일 이름, IP 주소는 들어가지 않습니다. 아래 JSON이 전송되는 전부입니다.',
    consent:
      '점검 결과를 TypeCV 개발자에게 보내는 데 동의합니다. 실행 전에 선택하면 점검이 끝나는 즉시 결과가 전송됩니다.',
    send: '결과 보내기',
    sent: '전송됨: ',
    notSent: '전송 실패: ',
    sending: '전송 중…',
    unsent: '아직 보내지 않았습니다.',
  },
} as const;

/** The hardware check of `/lab/check/` (`docs/lab-run-records.md`, "Hardware check"). */
export default function HardwareCheck({ locale }: { locale: 'en' | 'ko' }) {
  const text = TEXT[locale];
  const [state, setState] = useState<State>('idle');
  const [lines, setLines] = useState<string[]>([]);
  const [result, setResult] = useState<HardwareCheckRecord | null>(null);
  const [note, setNote] = useState('');
  const [copied, setCopied] = useState(false);
  const [consent, setConsent] = useState(false);
  const [sendState, setSendState] = useState<SendState>({ status: 'idle' });
  const consentRef = useRef(false);
  consentRef.current = consent;
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  useEffect(() => {
    (window as any).__typecvCheck = { state, result, lines, sendState };
  }, [state, result, lines, sendState]);
  const json = result ? JSON.stringify(result, null, 2) : '';
  const failed = result ? failedCases(result) : [];

  /** Validate the record as the collector does, then post it (`website/src/lib/send-record.ts`). */
  async function send(current: HardwareCheckRecord | null = result) {
    if (!current) return;
    const problem = hardwareCheckProblem(current);
    if (problem) {
      setSendState({ status: 'failed', message: 'Invalid record field: ' + problem });
      return;
    }
    setSendState({ status: 'sending' });
    setSendState(await postRecord(current, locale === 'ko'));
  }

  async function run() {
    setState('running');
    setLines([]);
    setResult(null);
    setCopied(false);
    setSendState({ status: 'idle' });
    const read = async (path: string, key: string) => {
      try {
        const value = (await (await fetch(path)).json())[key];
        return typeof value === 'string' ? value : null;
      } catch {
        return null;
      }
    };
    const [commit, typeshade] = await Promise.all([
      read('/version.json', 'commit'),
      read('/runtime/version.json', 'typeshade'),
    ]);
    worker.current?.terminate();
    const check = new Worker(new URL('../workers/hardware-check.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.current = check;
    const add = (line: string) => setLines((current) => [...current, line]);
    const cases = chosenCases();
    if (cases.length) add('Cases: ' + cases.join(', '));
    check.onmessage = ({ data }) => {
      if (data.type === 'log') add(data.line);
      else if (data.type === 'result') {
        setResult(data.result);
        setState('done');
        add('Done.');
        check.terminate();
        worker.current = null;
        // With consent given before the run, the result is sent at once.
        if (consentRef.current) void send(data.result);
      } else if (data.type === 'error') {
        add('Error: ' + data.message);
        setState('error');
      }
    };
    check.onerror = (event) => {
      add('Worker error: ' + (event.message || '(no message)'));
      setState('error');
    };
    check.postMessage({
      type: 'start',
      href: location.href,
      commit,
      typeshade,
      note: note.slice(0, HARDWARE_CHECK_MAX_NOTE),
      cases,
    });
  }

  function download() {
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-hardware-check.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="probe check">
      <div className="lab-panel">
        <label className="check-note">
          <span>{text.noteLabel}</span>
          <input
            type="text"
            maxLength={HARDWARE_CHECK_MAX_NOTE}
            placeholder={text.notePlaceholder}
            value={note}
            onChange={(event) => {
              setNote(event.target.value);
              // A note typed after the run goes into the record that is sent.
              if (result)
                setResult({
                  ...result,
                  note: event.target.value.slice(0, HARDWARE_CHECK_MAX_NOTE),
                });
            }}
          />
        </label>
        <div className="probe-controls">
          <button
            className="button-primary"
            type="button"
            onClick={() => void run()}
            disabled={state === 'running'}
          >
            {text.run}
          </button>
          <button
            className="button-secondary"
            type="button"
            disabled={!result}
            onClick={async () => {
              await navigator.clipboard?.writeText(json);
              setCopied(true);
            }}
          >
            {copied ? text.copied : text.copy}
          </button>
          <button className="button-secondary" type="button" disabled={!result} onClick={download}>
            {text.download}
          </button>
        </div>
      </div>
      <section className="lab-panel probe-send" aria-label={text.sendTitle}>
        <h2>{text.sendTitle}</h2>
        <p className="field-help">{text.sendHelp}</p>
        <label className="probe-consent">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
          />{' '}
          {text.consent}
        </label>
        <div className="probe-controls">
          <button
            className="button-primary"
            type="button"
            disabled={
              !result || !consent || sendState.status === 'sending' || sendState.status === 'sent'
            }
            onClick={() => void send()}
          >
            {text.send}
          </button>
        </div>
        <p className="probe-send-status" role="status">
          {sendState.status === 'sent'
            ? text.sent + sendState.id
            : sendState.status === 'failed'
              ? text.notSent + sendState.message
              : sendState.status === 'sending'
                ? text.sending
                : text.unsent}
        </p>
      </section>
      <p className="lab-status probe-status" role="status" aria-live="polite" data-state={state}>
        {text[state]}
        {result && (failed.length ? ' ' + text.failed + failed.join(', ') : ' ' + text.passed)}
        {result?.resume && ' ' + text.resume + result.resume}
      </p>
      <h2>{text.log}</h2>
      <pre className="probe-log">{lines.join('\n')}</pre>
      {result && (
        <>
          <h2>{text.result}</h2>
          <pre className="probe-result">{json}</pre>
        </>
      )}
    </div>
  );
}
