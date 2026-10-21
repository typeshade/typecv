import { Button, InputNumber, Tag, Tooltip } from 'antd';
import type { SampleEntry, VideoSetEntry } from '../lib/sample-catalog';
import { componentColor, matrixShade, type VideoTag } from '../lib/multi-video';

/**
 * The pieces of the lab that a multi-video run adds (`docs/multi-video.md`, 6.1 and 6.3): the card
 * of a sample set, the rows of the upload list with their tags, the component switch and the
 * n × n matrix. They hold no state of a run. The text is in tags, marks and tooltips.
 */

/** One video of the upload list: an uploaded file or a catalog sample. */
export type VideoRow = {
  key: string;
  /** The text of the row: the file name or the sample name. It is never sent. */
  label: string;
  /** The catalog id of a sample; the record carries it as the label. */
  sampleId?: string;
  file?: File;
  entry?: SampleEntry;
  videoId?: string;
  focal: number;
  thumbnail?: string;
  durationSeconds?: number;
  /** After a run. */
  keyframes?: number;
  registeredViews?: number;
  component?: number | null;
  tags?: readonly VideoTag[];
  /** The cross-video inliers of the video with all the others. */
  crossInliers?: number;
  /** The row has the `videoId` of an earlier row (2.1). */
  duplicate?: boolean;
};

const TAG_WORDS: Record<VideoTag, [string, string]> = {
  component: ['묶음', 'component'],
  alone: ['단독', 'alone'],
  'not connected': ['이어지지 않음', 'not connected'],
  rotation: ['회전', 'rotation'],
  duplicate: ['중복', 'duplicate'],
};

/** The tags of a row as the page shows them, with the component number in `component N`. */
function rowTags(row: VideoRow): VideoTag[] {
  const tags = [...(row.tags ?? [])];
  if (row.duplicate) tags.push('duplicate');
  return tags;
}

export function VideoRows(props: {
  ko: boolean;
  rows: readonly VideoRow[];
  busy: boolean;
  onFocal: (key: string, focal: number) => void;
  onRemove: (key: string) => void;
}) {
  const { ko, rows, busy } = props;
  return (
    <ul className="mv-rows" aria-label={ko ? '영상 목록' : 'Videos'}>
      {rows.map((row) => {
        const tip =
          row.keyframes === undefined
            ? undefined
            : (ko ? '키프레임 ' : 'keyframes ') +
              row.keyframes +
              ' · ' +
              (ko ? '카메라 ' : 'views registered ') +
              (row.registeredViews ?? 0) +
              ' · ' +
              (ko ? '영상 사이 맞는 점 ' : 'cross-video inliers ') +
              (row.crossInliers ?? 0);
        return (
          <li
            key={row.key}
            className="mv-row"
            data-video={row.sampleId ?? row.key}
            data-duplicate={row.duplicate || undefined}
          >
            {row.thumbnail ? (
              <img src={row.thumbnail} alt="" loading="lazy" />
            ) : (
              <span className="mv-thumb-blank" aria-hidden />
            )}
            <span className="mv-label" title={row.label}>
              {row.label}
            </span>
            <label className="mv-focal">
              <span className="sr-only">
                {ko ? '초점거리 / 영상 너비' : 'Focal length / image width'}
              </span>
              <InputNumber
                size="small"
                disabled={busy || row.duplicate}
                min={0.2}
                max={4}
                step={0.1}
                value={row.focal}
                aria-label={(ko ? '초점거리: ' : 'Focal length: ') + row.label}
                onChange={(v) => props.onFocal(row.key, Number(v ?? row.focal))}
              />
            </label>
            <span className="mv-tags">
              {row.keyframes !== undefined && (
                <Tooltip title={tip}>
                  <Tag className="mv-count">
                    {row.keyframes}
                    {ko ? ' 키프레임' : ' kf'}
                  </Tag>
                </Tooltip>
              )}
              {rowTags(row).map((tag) => (
                <Tooltip key={tag} title={tip}>
                  <Tag
                    data-tag={tag}

                    style={
                      tag === 'component' && row.component != null
                        ? {
                            borderColor: componentColor(row.component),
                            color: componentColor(row.component),
                          }
                        : undefined
                    }
                  >
                    {TAG_WORDS[tag][ko ? 0 : 1]}
                    {tag === 'component' && row.component != null ? ' ' + (row.component + 1) : ''}
                  </Tag>
                </Tooltip>
              ))}
            </span>
            <Button
              size="small"
              type="text"
              disabled={busy}
              aria-label={(ko ? '삭제: ' : 'Remove ') + row.label}
              onClick={() => props.onRemove(row.key)}
            >
              ×
            </Button>
          </li>
        );
      })}
    </ul>
  );
}

/** The card of a `video-set` catalog entry: a stack of thumbnails, the count and a Load button. */
export function VideoSetCard(props: {
  ko: boolean;
  entry: VideoSetEntry;
  members: readonly SampleEntry[];
  loaded: boolean;
  busy: boolean;
  missing: boolean;
  onLoad: () => void;
}) {
  const { ko, entry, members } = props;
  const shown = members.slice(0, 3);
  return (
    <figure
      className="sample-card"
      role="listitem"
      data-sample={entry.id}
      data-kind="video-set"
      data-loaded={props.loaded || undefined}
    >
      <Tooltip title={entry.description[ko ? 'ko' : 'en']} placement="bottom">
        <div className="mv-stack" role="img" aria-label={entry.name}>
          {shown.map((m, i) => (
            <img
              key={m.id}
              src={m.dir + m.thumbnail}
              alt=""
              loading="lazy"
              style={{ left: i * 14 + '%', top: i * 6 + '%', zIndex: shown.length - i }}
            />
          ))}
        </div>
      </Tooltip>
      <figcaption>
        <strong>{entry.name}</strong>
        <div className="sample-card-tags">
          <Tag color="geekblue">
            {entry.members.length}
            {ko ? '개 영상' : ' videos'}
          </Tag>
        </div>
        <div className="sample-card-actions">
          <Button
            size="small"
            disabled={props.busy || props.missing}
            aria-label={'Load ' + entry.name}
            onClick={props.onLoad}
          >
            {ko ? '불러오기' : 'Load'}
          </Button>
        </div>
      </figcaption>
    </figure>
  );
}

/** The switch between the components of a result: one chip for each, in its color. */
export function ComponentSwitch(props: {
  ko: boolean;
  components: readonly { index: number; keyframes: number; videos: number }[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  const { ko } = props;
  return (
    <div className="mv-components" role="group" aria-label={ko ? '묶음' : 'Components'}>
      {props.components.map((c) => (
        <Tooltip
          key={c.index}
          title={`${c.keyframes} ${ko ? '키프레임' : 'keyframes'} · ${c.videos} ${ko ? '영상' : 'videos'}`}
        >
          <Button
            size="small"
            type={props.selected === c.index ? 'primary' : 'default'}
            data-component={c.index}
            style={{ borderColor: componentColor(c.index) }}
            onClick={() => props.onSelect(c.index)}
          >
            {(ko ? '구성요소 ' : 'component ') + (c.index + 1)}
          </Button>
        </Tooltip>
      ))}
    </div>
  );
}

/** The n × n matrix of 6.3: the cross-video inliers of each pair of videos, shaded. */
export function InlierMatrix(props: {
  ko: boolean;
  names: readonly string[];
  cells: readonly (readonly number[])[];
}) {
  const { ko, names, cells } = props;
  const max = Math.max(0, ...cells.flatMap((row, i) => row.filter((_, j) => i !== j)));
  return (
    <div
      className="mv-matrix"
      role="table"
      aria-label={ko ? '영상 사이 맞는 점 표' : 'Cross-video inlier matrix'}
      style={{ gridTemplateColumns: `repeat(${names.length}, 1fr)` }}
    >
      {cells.flatMap((row, i) =>
        row.map((value, j) => (
          <Tooltip
            key={i + ':' + j}
            title={i === j ? names[i] : `${names[i]} / ${names[j]}: ${value}`}
          >
            <span
              role="cell"
              className="mv-cell"
              data-value={value}
              style={{
                background:
                  i === j
                    ? 'var(--line)'
                    : `color-mix(in srgb, #1677ff ${Math.round(100 * matrixShade(value, max))}%, transparent)`,
              }}
            />
          </Tooltip>
        )),
      )}
    </div>
  );
}
