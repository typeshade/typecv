import { Button, InputNumber, Popover, Radio, Tooltip } from 'antd';
import { useState } from 'react';
import type { MapPlacement, PlacementState } from './SfmViewer';

export type PlacementCommand = {
  kind: 'reset' | 'ground' | 'auto' | 'pick' | 'set';
  nonce: number;
  value?: MapPlacement;
};
export type TransformMode = 'off' | 'translate' | 'rotate' | 'scale';

export const UNPLACED: PlacementState = {
  position: [0, 0, 0],
  rotationDeg: [0, 0, 0],
  scale: 1,
  picking: -1,
};

/**
 * The 3D card's placement menu: put the map on the ground grid by hand (handles and numbers),
 * by three ground points, or by the plane that holds most points.
 */
export default function PlacementMenu({
  ko,
  disabled,
  mode,
  onMode,
  state,
  onCommand,
  getPopupContainer,
}: {
  ko: boolean;
  disabled: boolean;
  mode: TransformMode;
  onMode: (mode: TransformMode) => void;
  state: PlacementState;
  onCommand: (kind: PlacementCommand['kind'], value?: MapPlacement) => void;
  getPopupContainer: () => HTMLElement;
}) {
  const [open, setOpen] = useState(false);
  const t = (k: string, e: string) => (ko ? k : e);
  const set = (change: Partial<MapPlacement>) =>
    onCommand('set', {
      position: state.position,
      rotationDeg: state.rotationDeg,
      scale: state.scale,
      ...change,
    });
  const axisInputs = (
    label: string,
    values: [number, number, number],
    step: number,
    apply: (next: [number, number, number]) => void,
  ) => (
    <div className="placement-row">
      <span className="placement-label">{label}</span>
      {(['x', 'y', 'z'] as const).map((axis, i) => (
        <InputNumber
          key={axis}
          size="small"
          prefix={axis}
          step={step}
          value={values[i]}
          aria-label={label + ' ' + axis}
          onChange={(value) => {
            if (typeof value !== 'number' || !Number.isFinite(value)) return;
            const next = [...values] as [number, number, number];
            next[i] = value;
            apply(next);
          }}
        />
      ))}
    </div>
  );
  const content = (
    <div className="placement-menu" role="group" aria-label={t('배치', 'Placement')}>
      <div className="placement-row">
        <span className="placement-label">{t('조작', 'Handles')}</span>
        <Radio.Group
          size="small"
          value={mode}
          onChange={(event) => onMode(event.target.value)}
          aria-label={t('조작', 'Handles')}
        >
          <Radio.Button value="off">{t('끔', 'Off')}</Radio.Button>
          <Radio.Button value="translate">{t('이동', 'Move')}</Radio.Button>
          <Radio.Button value="rotate">{t('회전', 'Turn')}</Radio.Button>
          <Radio.Button value="scale">{t('크기', 'Scale')}</Radio.Button>
        </Radio.Group>
      </div>
      <div className="placement-actions">
        <Tooltip
          title={t(
            '바닥에 있는 점 3개를 차례로 더블클릭하면 그 면이 그리드에 놓입니다.',
            'Double-click three points on the ground in turn; their plane goes on the grid.',
          )}
          getPopupContainer={getPopupContainer}
        >
          <Button
            block
            onClick={() => {
              onCommand('pick');
              setOpen(false);
            }}
          >
            {t('바닥 점 3개 찍기', 'Pick three ground points')}
          </Button>
        </Tooltip>
        <Tooltip
          title={t(
            '점이 가장 많이 놓인 평면을 바닥으로 보고 그리드에 맞춥니다. 카메라가 있는 쪽을 위로 봅니다. 틀리면 ‘바닥 점 3개 찍기’나 조작 도구로 고치세요.',
            'Takes the plane that holds most points as the ground, with the cameras above it. If it is wrong, pick three points or use the handles.',
          )}
          getPopupContainer={getPopupContainer}
        >
          <Button block onClick={() => onCommand('auto')}>
            {t('자동으로 바닥 맞추기', 'Level automatically')}
          </Button>
        </Tooltip>
        <Tooltip
          title={t(
            '가장 낮은 점들을 그리드 높이에, 가운데를 원점 위에 놓습니다.',
            'Puts the lowest points at the grid and the middle above the origin.',
          )}
          getPopupContainer={getPopupContainer}
        >
          <Button block onClick={() => onCommand('ground')}>
            {t('그리드 위에 놓기', 'Put on the grid')}
          </Button>
        </Tooltip>
      </div>
      {axisInputs(t('위치', 'Position'), state.position, 0.1, (position) => set({ position }))}
      {axisInputs(t('회전 °', 'Turn °'), state.rotationDeg, 5, (rotationDeg) =>
        set({ rotationDeg }),
      )}
      <div className="placement-row">
        <span className="placement-label">{t('크기', 'Scale')}</span>
        <InputNumber
          size="small"
          min={0.0001}
          step={0.1}
          value={state.scale}
          aria-label={t('크기', 'Scale')}
          onChange={(value) => {
            if (typeof value === 'number' && value > 0) set({ scale: value });
          }}
        />
        <Button size="small" danger onClick={() => onCommand('reset')}>
          {t('초기화', 'Reset')}
        </Button>
      </div>
      {(state.message === 'line' || state.message === 'few') && (
        <p className="placement-message" role="status">
          {state.message === 'line'
            ? t(
                '세 점이 거의 한 줄에 있어 평면을 정할 수 없습니다. 떨어진 점을 다시 찍어 주세요.',
                'The three points are nearly on one line. Pick points farther apart.',
              )
            : t('점이 너무 적어 평면을 찾지 못했습니다.', 'Too few points to find a plane.')}
        </p>
      )}
    </div>
  );
  return (
    <Popover
      trigger="click"
      placement="bottomRight"
      open={open && !disabled}
      onOpenChange={setOpen}
      getPopupContainer={getPopupContainer}
      content={content}
    >
      <Button aria-haspopup="true" disabled={disabled}>
        {t('배치', 'Place')} ▾
      </Button>
    </Popover>
  );
}
