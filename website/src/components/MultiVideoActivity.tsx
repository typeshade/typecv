import { CloseOutlined, LoadingOutlined, ShareAltOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { VideoRow } from './MultiVideoPanel';
import type { GpuPreparation } from '../lib/gpu-preparation';
import type { KeyframeThumbnail, MultiVideoActivity } from '../lib/multi-video-activity';
import { componentColor } from '../lib/multi-video';
import type { VideoConnectionGraph } from '../lib/video-connection-scene';
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
  onGraphChange,
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
  onGraphChange: (graph: VideoConnectionGraph | undefined) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<string>();
  const canvas = useRef<HTMLDivElement>(null);
  const full = !mapReady || expanded;
  const now = performance.now();
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
  const links = graph?.links ?? [];
  const selectedLinks = links.filter(
    (link) => link.first === focused?.videoId || link.second === focused?.videoId,
  );
  const frames = thumbnails.filter((frame) => frame.videoId === focused?.videoId);
  useEffect(() => {
    if (!full || !canvas.current) {
      onGraphChange(undefined);
      return;
    }
    onGraphChange({
      element: canvas.current,
      nodes: rows
        .filter((row) => !row.duplicate)
        .slice(0, 16)
        .map((row) => {
          const group = graph?.components.find(
            (group) => row.videoId && group.videos.includes(row.videoId),
          );
          const latest = thumbnails.findLast((frame) => frame.videoId === row.videoId);
          return {
            id: row.key,
            videoId: row.videoId,
            label: row.label,
            image:
              latest?.image ??
              row.thumbnail ??
              (row.entry ? row.entry.dir + row.entry.thumbnail : undefined),
            detail:
              row.keyframes === undefined
                ? ko
                  ? '프레임 준비 중'
                  : 'Preparing frames'
                : `${row.keyframes} ${ko ? '키프레임' : 'keyframes'}`,
            active:
              row.videoId === activeVideo ||
              (!!row.videoId && !!graph?.activeVideos.includes(row.videoId)),
            color: group ? componentColor(group.index) : undefined,
          };
        }),
      links: graph?.links ?? [],
      selected: focused?.key,
      focusConnections: selected !== undefined,
      onSelect: setSelected,
    });
  }, [full, rows, graph, thumbnails, activeVideo, focused?.key, selected, ko, onGraphChange]);
  useEffect(() => () => onGraphChange(undefined), [onGraphChange]);
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
            <div className="mv-activity-network">
              <div
                ref={canvas}
                className="mv-activity-surface"
                role="group"
                aria-label={ko ? '입력 영상 연결 관계' : 'Input video connections'}
              />
              <ul
                className="sr-only"
                aria-label={ko ? '영상 연결 상태' : 'Video connection states'}
              >
                {links.map((link) => (
                  <li
                    key={link.first + ':' + link.second}
                    className="mv-activity-edge"
                    data-state={
                      link.accepted ? 'verified' : link.matches > 0 ? 'matched' : 'candidate'
                    }
                    data-active={link.active || undefined}
                  >
                    {videos.find((row) => row.videoId === link.first)?.label} ·{' '}
                    {videos.find((row) => row.videoId === link.second)?.label}: {link.candidates}{' '}
                    {ko ? '후보 쌍' : 'candidate pairs'} · {link.accepted}{' '}
                    {ko ? '검증된 쌍' : 'verified pairs'} · {link.inliers}{' '}
                    {ko ? '일치점' : 'inliers'}
                  </li>
                ))}
              </ul>
              <p className="mv-activity-caption">
                {ko
                  ? '입력 영상 연결도 · 드래그: 회전 · 휠: 확대 · 선택: 상세 보기'
                  : 'Input connection diagram · Drag: rotate · Wheel: zoom · Select: details'}
              </p>
            </div>
            <aside className="mv-activity-detail">
              {focused && (
                <>
                  <select
                    className="mv-activity-selection"
                    aria-label={ko ? '활동 영상 선택' : 'Select activity video'}
                    value={focused.key}
                    onChange={(event) => setSelected(event.target.value)}
                  >
                    {visible.map((row) => (
                      <option key={row.key} className="mv-activity-node" value={row.key}>
                        {row.label}
                      </option>
                    ))}
                  </select>
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
