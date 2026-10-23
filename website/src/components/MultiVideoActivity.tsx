import { CloseOutlined, LoadingOutlined, ShareAltOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import type { VideoRow } from './MultiVideoPanel';
import type { GpuPreparation } from '../lib/gpu-preparation';
import type { KeyframeThumbnail, MultiVideoActivity } from '../lib/multi-video-activity';
import { componentColor } from '../lib/multi-video';
import {
  STAGE_IDS,
  STAGE_WORDS,
  formatElapsed,
  stalled,
  stallWarning,
  type StageBoard,
} from '../lib/multi-video-stages';

/** A schematic input graph in the reconstruction viewport, before actual poses exist. */
export default function MultiVideoActivityPanel({
  rows,
  board,
  graph,
  thumbnails,
  activeVideo,
  preparation,
  mapReady,
  preview,
  ko,
}: {
  rows: readonly VideoRow[];
  board?: StageBoard;
  graph?: MultiVideoActivity;
  thumbnails: readonly KeyframeThumbnail[];
  activeVideo?: string;
  preparation?: GpuPreparation;
  mapReady: boolean;
  preview?: ReactNode;
  ko: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<string>();
  const [width, setWidth] = useState(600);
  const canvas = useRef<HTMLDivElement>(null);
  const clip = useId().replace(/:/g, '');
  const full = !mapReady || expanded;
  const now = performance.now();
  useEffect(() => {
    if (!full || !canvas.current) return;
    const element = canvas.current;
    const observer = new ResizeObserver(() => setWidth(Math.max(260, element.clientWidth)));
    setWidth(Math.max(260, element.clientWidth));
    observer.observe(element);
    return () => observer.disconnect();
  }, [full]);
  const videos = rows.filter((row) => !row.duplicate);
  const visible = videos.slice(0, 16);
  const focused =
    videos.find((row) => row.key === selected) ??
    videos.find((row) => row.videoId === activeVideo) ??
    videos[0];
  const current =
    board?.rows.find((row) => row.state === 'running') ??
    [...(board?.rows ?? [])].reverse().find((row) => row.state === 'done') ??
    board?.rows[0];
  const stage =
    current?.id === 'reconstruction' ? 'reconstruction' : (graph?.stage ?? current?.id ?? 'frames');
  const title = preparation
    ? ko
      ? 'GPU 준비 중'
      : 'Preparing the GPU'
    : STAGE_WORDS[stage][ko ? 1 : 0];
  const count = current?.total ? `${current.done} / ${current.total}` : String(current?.done ?? 0);
  const height = 300;
  // Keep adjacent cards apart on the short side of the ellipse, including 15-input runs.
  const spacing = Math.sin(Math.PI / Math.max(visible.length, 2));
  const verticalLimit = ((height - 28) * spacing - 8) / (0.8 + 0.9 * spacing);
  const size = Math.min(
    88,
    verticalLimit,
    Math.max(44, width / Math.max(7, Math.sqrt(visible.length) * 2.4)),
  );
  const position = (index: number) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / Math.max(visible.length, 1);
    return {
      x: width / 2 + Math.cos(angle) * (width / 2 - size / 2 - 12),
      y: height / 2 + Math.sin(angle) * (height / 2 - size * 0.45 - 14),
    };
  };
  const positions = new Map(visible.map((row, index) => [row.videoId ?? row.key, position(index)]));
  const links = graph?.links ?? [];
  const selectedLinks = links.filter(
    (link) => link.first === focused?.videoId || link.second === focused?.videoId,
  );
  const frames = thumbnails.filter((frame) => frame.videoId === focused?.videoId);
  return (
    <section
      className="mv-stage-activity"
      data-testid="multi-video-activity"
      data-mode={full ? 'full' : 'compact'}
      data-stage={preparation ? 'preparing' : stage}
      aria-label={ko ? '다중 영상 실행 시각화' : 'Multi-video stage visualization'}
    >
      <header>
        <LoadingOutlined spin aria-hidden />
        <strong>{title}</strong>
        {!preparation && <span className="mv-activity-count">{count}</span>}
        {(current || preparation) && (
          <span className="mv-activity-time">
            {formatElapsed(
              preparation ? now - preparation.startedAt : (current?.elapsedMs ?? 0),
              ko,
            )}
          </span>
        )}
        {mapReady && (
          <Button
            type="text"
            size="small"
            icon={expanded ? <CloseOutlined /> : <ShareAltOutlined />}
            aria-label={
              expanded
                ? ko
                  ? '3D 장면으로 돌아가기'
                  : 'Return to 3D scene'
                : ko
                  ? '영상 연결 그래프 보기'
                  : 'Show video connection graph'
            }
            aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}
          />
        )}
      </header>
      {!preparation && board && stalled(board, now) && (
        <p className="mv-activity-stall" role="status">
          {stallWarning(now - board.lastEventAt, ko)}
        </p>
      )}
      {full && (
        <>
          <ol className="mv-activity-stages" aria-label={ko ? '실행 단계' : 'Run stages'}>
            {STAGE_IDS.map((id, index) => (
              <li
                key={id}
                data-state={board?.rows[index]?.state ?? 'pending'}
                aria-current={!preparation && stage === id ? 'step' : undefined}
              >
                <span>{index + 1}</span>
                <span>{STAGE_WORDS[id][ko ? 1 : 0].replace(/…$/, '')}</span>
              </li>
            ))}
          </ol>
          <div className="mv-activity-body">
            <div ref={canvas} className="mv-activity-network">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                aria-label={ko ? '입력 영상 연결 관계' : 'Input video connections'}
              >
                {links.map((link) => {
                  const a = positions.get(link.first),
                    b = positions.get(link.second);
                  if (!a || !b) return null;
                  const related =
                    !selected ||
                    link.first === focused?.videoId ||
                    link.second === focused?.videoId;
                  return (
                    <line
                      key={link.first + ':' + link.second}
                      x1={a.x}
                      y1={a.y}
                      x2={b.x}
                      y2={b.y}
                      className="mv-activity-edge"
                      data-state={
                        link.accepted ? 'verified' : link.matches > 0 ? 'matched' : 'candidate'
                      }
                      data-active={link.active || undefined}
                      data-muted={!related || undefined}
                    >
                      <title>{`${link.candidates} ${ko ? '후보 쌍' : 'candidate pairs'} · ${link.accepted} ${ko ? '검증된 쌍' : 'verified pairs'} · ${link.inliers} ${ko ? '일치점' : 'inliers'}`}</title>
                    </line>
                  );
                })}
                {visible.map((row, index) => {
                  const pos = position(index);
                  const src =
                    row.thumbnail ?? (row.entry ? row.entry.dir + row.entry.thumbnail : undefined);
                  const groups =
                    graph?.components.filter(
                      (c) => row.videoId && c.videos.includes(row.videoId),
                    ) ?? [];
                  const active =
                    row.videoId === activeVideo ||
                    (!!row.videoId && graph?.activeVideos.includes(row.videoId));
                  return (
                    <g
                      key={row.key}
                      className="mv-activity-node"
                      transform={`translate(${pos.x - size / 2} ${pos.y - size * 0.4})`}
                      role="button"
                      tabIndex={0}
                      aria-label={`${row.label}, ${row.keyframes ?? 0} ${ko ? '키프레임' : 'keyframes'}`}
                      aria-pressed={focused?.key === row.key}
                      data-active={active || undefined}
                      onClick={() => setSelected(row.key)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          setSelected(row.key);
                        }
                      }}
                    >
                      <title>{row.label}</title>
                      <rect
                        width={size}
                        height={size * 0.8}
                        rx={7}
                        style={
                          groups.length ? { stroke: componentColor(groups[0]!.index) } : undefined
                        }
                      />
                      <clipPath id={`${clip}-${index}`}>
                        <rect x={4} y={4} width={size - 8} height={size * 0.48} rx={4} />
                      </clipPath>
                      {src && (
                        <image
                          href={src}
                          x={4}
                          y={4}
                          width={size - 8}
                          height={size * 0.48}
                          preserveAspectRatio="xMidYMid slice"
                          clipPath={`url(#${clip}-${index})`}
                        />
                      )}
                      <text x={size / 2} y={size * 0.7} textAnchor="middle">
                        {index + 1} · {row.keyframes ?? '—'}
                      </text>
                    </g>
                  );
                })}
              </svg>
              <p className="mv-activity-caption">
                {ko ? '입력 영상 연결도' : 'Input video connection diagram'}
              </p>
            </div>
            <aside className="mv-activity-detail">
              {focused && (
                <>
                  <strong title={focused.label}>{focused.label}</strong>
                  <p>
                    {focused.keyframes ?? 0} {ko ? '키프레임' : 'keyframes'} ·{' '}
                    {selectedLinks.reduce((n, link) => n + link.accepted, 0)}{' '}
                    {ko ? '검증된 연결' : 'verified pairs'}
                  </p>
                  <div
                    className="mv-activity-thumbnails"
                    aria-label={ko ? '최근 키프레임' : 'Recent keyframes'}
                  >
                    {frames.map((frame) => (
                      <figure key={frame.frameId}>
                        <img
                          src={frame.image}
                          alt={`${focused.label} · ${frame.time.toFixed(1)} s`}
                        />
                        <figcaption>
                          {frame.time.toFixed(1)} s · {frame.features}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </>
              )}
              {stage === 'frames' && <div className="mv-activity-preview">{preview}</div>}
              {!!graph?.components.length && (
                <ul className="mv-activity-components">
                  {graph.components.map((group) => (
                    <li key={group.index} style={{ borderColor: componentColor(group.index) }}>
                      {ko ? '묶음 ' : 'Group '}
                      {group.index + 1} · {group.videos.length} {ko ? '영상' : 'videos'} ·{' '}
                      {group.keyframes} {ko ? '키프레임' : 'keyframes'}
                    </li>
                  ))}
                </ul>
              )}
            </aside>
          </div>
          <footer className="mv-activity-legend">
            <span data-state="candidate">{ko ? '후보 연결' : 'Candidate'}</span>
            <span data-state="matched">{ko ? '특징 매칭' : 'Matched'}</span>
            <span data-state="verified">{ko ? '기하 검증 통과' : 'Geometry verified'}</span>
            {(videos.length > visible.length || graph?.limited) && (
              <span>
                {ko ? '일부 연결만 표시' : 'Showing a subset of connections'} ({visible.length}/
                {videos.length})
              </span>
            )}
          </footer>
        </>
      )}
    </section>
  );
}
