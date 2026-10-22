import { Component, useState, type ReactNode } from 'react';

export class ViewerBoundary extends Component<
  { children: ReactNode; onError: (message: string) => void },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: Error) {
    this.props.onError(error.message);
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function ViewerDiagnostics({ error, ko }: { error: string; ko: boolean }) {
  const [report, setReport] = useState('');
  const [copied, setCopied] = useState(false);
  async function copy() {
    const canvas = document.createElement('canvas');
    let webgl2 = false;
    try {
      const gl = canvas.getContext('webgl2');
      webgl2 = !!gl;
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    } catch {
      /* Report unsupported contexts as unavailable. */
    }
    const text = JSON.stringify(
      {
        time: new Date().toISOString(),
        userAgent: navigator.userAgent,
        webgl2,
        webgpu: 'gpu' in navigator,
        pixelRatio: window.devicePixelRatio,
        viewport: { width: innerWidth, height: innerHeight },
        error: error || null,
      },
      null,
      2,
    );
    setReport(text);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="viewer-diagnostics">
      {error && (
        <p role="alert">
          {ko
            ? '3D 화면을 표시하지 못했습니다. 페이지를 새로고침하거나 Chrome 하드웨어 가속 설정을 확인해 주세요. 진단 정보를 전달하면 원인 확인에 도움이 됩니다.'
            : 'The 3D view could not be displayed. Reload the page or check Chrome hardware acceleration. Share the diagnostics to help investigate.'}
        </p>
      )}
      <button type="button" className="button-secondary" onClick={copy}>
        {copied ? (ko ? '복사됨' : 'Copied') : ko ? '진단 정보 복사' : 'Copy diagnostics'}
      </button>
      {report && (
        <details open={!copied}>
          <summary>{ko ? '진단 정보' : 'Diagnostics'}</summary>
          <pre tabIndex={0}>{report}</pre>
        </details>
      )}
    </div>
  );
}
