import { useEffect, useRef, useState } from 'react';
import { probeRecordProblem, type ProbeRecord } from '../lib/probe-record';
import { postRecord, type SendState } from '../lib/send-record';
import {
  PROBE_STORAGE_KEY,
  clearProbe,
  loadProbe,
  saveProbe,
  type ProbeState,
} from '../lib/probe-plan';

type State = 'idle' | 'running' | 'done' | 'error';

/** The page's `?bounded=1` and `?fp64=`, the lab's two switches (`SfmLab.tsx`). */
function switches() {
  const query = new URLSearchParams(location.search);
  const fp64 = query.get('fp64');
  // `?probe-fail=<track>` is a test hook: it simulates a loss for the chunk that holds the track.
  const fail = query.get('probe-fail');
  // `?tracks=<first>-<last>` runs every step on these tracks only (`docs/lab-run-records.md`).
  const range = /^(\d+)-(\d+)$/.exec(query.get('tracks') ?? '');
  return {
    bounded: query.get('bounded') === '1',
    fp64: fp64 === 'float' || fp64 === 'integer' ? fp64 : undefined,
    simulatedLossTrack: fail !== null && /^\d+$/.test(fail) ? Number(fail) : undefined,
    tracks: range ? { first: Number(range[1]), last: Number(range[2]) } : undefined,
  } as const;
}

const TEXT = {
  en: {
    run: 'Run probe',
    copy: 'Copy JSON',
    copied: 'Copied',
    download: 'Download JSON',
    idle: 'Select “Run probe”. The probe can take several minutes. Keep this tab open.',
    running:
      'The probe is running. After a device loss the page reloads itself and goes on. Do not close this tab.',
    done: 'The probe is complete. Send, copy or download the JSON.',
    error: 'The probe stopped with an error.',
    log: 'Log',
    result: 'Result',
    sendTitle: 'Send the result',
    sendHelp:
      'The result holds the GPU adapter strings, the browser user agent, the page address with its switches, the site commit, the fixture’s cameras and poses, and the outcome, timings and chunk measurements of each step. A hanging track also carries its two pixels. It holds no image, file name or IP address. The JSON below is everything that is sent.',
    consent:
      'I agree to send the probe result to the TypeCV developers. Checked before a run, the result is sent as soon as the probe completes or stops.',
    send: 'Send result',
    sent: 'Sent: ',
    failed: 'Not sent: ',
    sending: 'Sending…',
    unsent: 'Not sent yet.',
  },
  ko: {
    run: '프로브 실행',
    copy: 'JSON 복사',
    copied: '복사됨',
    download: 'JSON 내려받기',
    idle: '“프로브 실행”을 누르세요. 몇 분 걸릴 수 있습니다. 이 탭을 열어 두세요.',
    running:
      '프로브가 실행 중입니다. 장치를 잃으면 페이지가 스스로 다시 열리고 이어서 실행됩니다. 이 탭을 닫지 마세요.',
    done: '프로브가 끝났습니다. JSON을 보내거나 복사하거나 내려받으세요.',
    error: '프로브가 오류로 멈췄습니다.',
    log: '로그',
    result: '결과',
    sendTitle: '결과 보내기',
    sendHelp:
      '결과에는 GPU 어댑터 문자열, 브라우저 User-Agent, 스위치를 포함한 페이지 주소, 사이트 커밋, 픽스처의 카메라와 자세, 각 단계의 결과와 시간과 청크 측정값이 들어갑니다. 멈춘 트랙은 두 픽셀 좌표도 담습니다. 이미지, 파일 이름, IP 주소는 들어가지 않습니다. 아래 JSON이 전송되는 전부입니다.',
    consent:
      '프로브 결과를 TypeCV 개발자에게 보내는 데 동의합니다. 실행 전에 선택하면 프로브가 끝나거나 멈추는 즉시 결과가 전송됩니다.',
    send: '결과 보내기',
    sent: '전송됨: ',
    failed: '전송 실패: ',
    sending: '전송 중…',
    unsent: '아직 보내지 않았습니다.',
  },
} as const;

export default function TriangulationProbe({ locale }: { locale: 'en' | 'ko' }) {
  const text = TEXT[locale];
  const [state, setState] = useState<State>('idle');
  const [lines, setLines] = useState<string[]>([]);
  const [result, setResult] = useState<ProbeRecord | null>(null);
  const [copied, setCopied] = useState(false);
  const [consent, setConsent] = useState(false);
  const [sendState, setSendState] = useState<SendState>({ status: 'idle' });
  const consentRef = useRef(false);
  consentRef.current = consent;
  const worker = useRef<Worker | null>(null);
  /** The log and the worker's last state, saved after every step so that a reload goes on. */
  const linesRef = useRef<string[]>([]);
  const probeState = useRef<ProbeState | null>(null);
  const persist = () =>
    probeState.current !== null &&
    saveProbe({
      href: location.href,
      consent: consentRef.current,
      lines: linesRef.current,
      state: probeState.current,
    });
  useEffect(() => () => worker.current?.terminate(), []);
  // A probe that reloaded the page after a device loss goes on without a click.
  useEffect(() => {
    const saved = loadProbe();
    if (!saved) return;
    if (saved.href !== location.href) {
      clearProbe();
      return;
    }
    consentRef.current = saved.consent;
    setConsent(saved.consent);
    linesRef.current = [...saved.lines];
    void run(saved.state);
  }, []);
  useEffect(() => {
    (window as any).__typecvProbe = { state, result, lines, sendState };
  }, [state, result, lines, sendState]);
  const json = result ? JSON.stringify(result, null, 2) : '';

  /** Validate the result as the collector does, then post it (`website/src/lib/send-record.ts`). */
  async function send(current: ProbeRecord | null = result) {
    if (!current) return;
    const problem = probeRecordProblem(current);
    if (problem) {
      setSendState({ status: 'failed', message: 'Invalid record field: ' + problem });
      return;
    }
    setSendState({ status: 'sending' });
    setSendState(await postRecord(current, locale === 'ko'));
  }

  async function run(resumed: ProbeState | null = null) {
    if (!resumed) {
      clearProbe();
      linesRef.current = [];
    }
    probeState.current = resumed;
    setState('running');
    setLines(linesRef.current);
    setResult(null);
    setCopied(false);
    setSendState({ status: 'idle' });
    let commit: string | null = null;
    try {
      commit = (await (await fetch('/version.json')).json()).commit ?? null;
    } catch {
      commit = null;
    }
    const chosen = switches();
    worker.current?.terminate();
    const probe = new Worker(new URL('../workers/probe.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.current = probe;
    const add = (line: string) => {
      linesRef.current = [...linesRef.current, line];
      setLines(linesRef.current);
    };
    // The page can reload only when it can keep the state over the reload.
    let canReload = false;
    try {
      sessionStorage.setItem(PROBE_STORAGE_KEY + '-check', '1');
      sessionStorage.removeItem(PROBE_STORAGE_KEY + '-check');
      canReload = true;
    } catch {
      canReload = false;
    }
    if (resumed) add(`Resumed after reload ${resumed.reloads}`);
    else
      add(
        `Switches: bounded=${chosen.bounded ? 1 : 0}, fp64=${chosen.fp64 ?? 'auto'}` +
          (chosen.tracks ? `, tracks ${chosen.tracks.first}..${chosen.tracks.last}` : '') +
          (chosen.simulatedLossTrack !== undefined
            ? `, simulated loss at track ${chosen.simulatedLossTrack}`
            : ''),
      );
    probe.onmessage = ({ data }) => {
      if (data.type === 'log') add(data.line);
      else if (data.type === 'state') {
        probeState.current = data.state;
        persist();
      } else if (data.type === 'reload') {
        // No device opened after a loss: keep the state, and go on from it after the reload.
        probeState.current = data.state;
        probe.terminate();
        worker.current = null;
        if (persist()) location.reload();
        else {
          add('Error: the state could not be saved for the reload');
          setState('error');
        }
      } else if (data.type === 'result') {
        clearProbe();
        probeState.current = null;
        setResult(data.result);
        setState('done');
        add('Done.');
        probe.terminate();
        worker.current = null;
        // With consent given before the run, the result is sent at once, completed or aborted.
        if (consentRef.current) void send(data.result);
      } else if (data.type === 'error') {
        clearProbe();
        add('Error: ' + data.message);
        setState('error');
      }
    };
    probe.onerror = (event) => {
      clearProbe();
      add('Worker error: ' + (event.message || '(no message)'));
      setState('error');
    };
    probe.postMessage({
      type: 'start',
      ...chosen,
      href: location.href,
      commit,
      state: resumed,
      canReload,
    });
  }

  function download() {
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-triangulation-probe.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="probe">
      <div className="lab-panel probe-controls">
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
      <section className="lab-panel probe-send" aria-label={text.sendTitle}>
        <h2>{text.sendTitle}</h2>
        <p className="field-help">{text.sendHelp}</p>
        <label className="probe-consent">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => {
              consentRef.current = event.target.checked;
              setConsent(event.target.checked);
              persist();
            }}
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
              ? text.failed + sendState.message
              : sendState.status === 'sending'
                ? text.sending
                : text.unsent}
        </p>
      </section>
      <p className="lab-status probe-status" role="status" aria-live="polite" data-state={state}>
        {text[state]}
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
