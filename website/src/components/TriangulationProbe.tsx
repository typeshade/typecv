import { useEffect, useRef, useState } from 'react';

type State = 'idle' | 'running' | 'done' | 'error';

/** The page's `?bounded=1` and `?fp64=`, the lab's two switches (`SfmLab.tsx`). */
function switches() {
  const query = new URLSearchParams(location.search);
  const fp64 = query.get('fp64');
  // `?probe-fail=<track>` is a test hook: it simulates a loss for the chunk that holds the track.
  const fail = query.get('probe-fail');
  return {
    bounded: query.get('bounded') === '1',
    fp64: fp64 === 'float' || fp64 === 'integer' ? fp64 : undefined,
    simulatedLossTrack: fail !== null && /^\d+$/.test(fail) ? Number(fail) : undefined,
  } as const;
}

const TEXT = {
  en: {
    run: 'Run probe',
    copy: 'Copy JSON',
    copied: 'Copied',
    download: 'Download JSON',
    idle: 'Select “Run probe”. The probe can take several minutes. Keep this tab open.',
    running: 'The probe is running. Do not close this tab.',
    done: 'The probe is complete. Copy or download the JSON.',
    error: 'The probe stopped with an error.',
    log: 'Log',
    result: 'Result',
  },
  ko: {
    run: '프로브 실행',
    copy: 'JSON 복사',
    copied: '복사됨',
    download: 'JSON 내려받기',
    idle: '“프로브 실행”을 누르세요. 몇 분 걸릴 수 있습니다. 이 탭을 열어 두세요.',
    running: '프로브가 실행 중입니다. 이 탭을 닫지 마세요.',
    done: '프로브가 끝났습니다. JSON을 복사하거나 내려받으세요.',
    error: '프로브가 오류로 멈췄습니다.',
    log: '로그',
    result: '결과',
  },
} as const;

export default function TriangulationProbe({ locale }: { locale: 'en' | 'ko' }) {
  const text = TEXT[locale];
  const [state, setState] = useState<State>('idle');
  const [lines, setLines] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [copied, setCopied] = useState(false);
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  useEffect(() => {
    (window as any).__typecvProbe = { state, result, lines };
  }, [state, result, lines]);
  const json = result ? JSON.stringify(result, null, 2) : '';

  async function run() {
    setState('running');
    setLines([]);
    setResult(null);
    setCopied(false);
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
    const add = (line: string) => setLines((previous) => [...previous, line]);
    add(
      `Switches: bounded=${chosen.bounded ? 1 : 0}, fp64=${chosen.fp64 ?? 'auto'}` +
        (chosen.simulatedLossTrack !== undefined
          ? `, simulated loss at track ${chosen.simulatedLossTrack}`
          : ''),
    );
    probe.onmessage = ({ data }) => {
      if (data.type === 'log') add(data.line);
      else if (data.type === 'result') {
        setResult(data.result);
        setState('done');
        add('Done.');
        probe.terminate();
        worker.current = null;
      } else if (data.type === 'error') {
        add('Error: ' + data.message);
        setState('error');
      }
    };
    probe.onerror = (event) => {
      add('Worker error: ' + (event.message || '(no message)'));
      setState('error');
    };
    probe.postMessage({ type: 'start', ...chosen, href: location.href, commit });
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
          onClick={run}
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
