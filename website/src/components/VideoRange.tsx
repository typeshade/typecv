import { Button, InputNumber, Slider, Tooltip } from 'antd';
import { UndoOutlined } from '@ant-design/icons';
import { formatVideoTime, type VideoRange as Range } from '../lib/video-range';

/** Source-time selection shared by single uploads and each row of a multi-video input. */
export default function VideoRange({
  duration,
  value,
  disabled,
  ko,
  onChange,
  onPreview,
}: {
  duration: number;
  value?: Range;
  disabled: boolean;
  ko: boolean;
  onChange: (range: Range | undefined) => void;
  onPreview?: (time: number) => void;
}) {
  const range = value ?? { start: 0, end: duration };
  const minimum = Math.min(1, duration);
  const change = (start: number, end: number, preview: number) => {
    onChange(start === 0 && end === duration ? undefined : { start, end });
    onPreview?.(Math.min(preview, Math.max(0, duration - 0.001)));
  };
  return (
    <section className="lab-video-range" aria-label={ko ? '분석 구간' : 'Analysis range'}>
      <div className="lab-video-range-head">
        <strong>{ko ? '분석 구간' : 'Analysis range'}</strong>
        <span>
          {formatVideoTime(range.end - range.start)} / {formatVideoTime(duration)}
        </span>
        <Tooltip title={ko ? '영상 전체 선택' : 'Select the whole video'}>
          <Button
            size="small"
            type="text"
            icon={<UndoOutlined aria-hidden />}
            disabled={disabled || !value}
            aria-label={ko ? '영상 전체 선택' : 'Select the whole video'}
            onClick={() => change(0, duration, 0)}
          />
        </Tooltip>
      </div>
      <Slider
        range
        min={0}
        max={duration}
        step={0.01}
        disabled={disabled}
        value={[range.start, range.end]}
        tooltip={{ formatter: (time) => formatVideoTime(time ?? 0) }}
        ariaLabelForHandle={[
          ko ? '분석 시작 위치' : 'Analysis start position',
          ko ? '분석 종료 위치' : 'Analysis end position',
        ]}
        onChange={(times) => {
          const [start, end] = times as number[];
          if (end - start < minimum) return;
          change(start, end, start !== range.start ? start : end);
        }}
      />
      <div className="lab-video-range-times">
        <label>
          {ko ? '시작' : 'Start'}{' '}
          <InputNumber
            min={0}
            max={Math.max(0, range.end - minimum)}
            step={0.1}
            precision={2}
            value={range.start}
            disabled={disabled}
            suffix="s"
            aria-label={ko ? '분석 시작 시간 (초)' : 'Analysis start time (seconds)'}
            onChange={(time) => {
              if (time !== null) {
                const start = Math.max(0, Math.min(Number(time), range.end - minimum));
                change(start, range.end, start);
              }
            }}
          />
        </label>
        <label>
          {ko ? '종료' : 'End'}{' '}
          <InputNumber
            min={range.start + minimum}
            max={duration}
            step={0.1}
            precision={2}
            value={range.end}
            disabled={disabled}
            suffix="s"
            aria-label={ko ? '분석 종료 시간 (초)' : 'Analysis end time (seconds)'}
            onChange={(time) => {
              if (time !== null) {
                const end = Math.min(duration, Math.max(Number(time), range.start + minimum));
                change(range.start, end, end);
              }
            }}
          />
        </label>
      </div>
    </section>
  );
}
