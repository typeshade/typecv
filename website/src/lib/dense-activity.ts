import type { DenseChunk, DenseCloud } from './dense-cloud';

/** Real worker work, rather than a time-based estimate of the whole MVS. */
export interface DenseActivity {
  phase: string;
  done: number;
  total: number;
  phaseDone?: number;
  phaseTotal?: number;
  pass?: number;
  passes?: number;
  view?: number;
  views?: number;
  viewId?: number;
  points?: number;
  elapsedMs: number;
  phaseMs: number;
  receivedAt: number;
  component?: number;
}

const LABELS: Record<string, [string, string]> = {
  waiting: ['다음 Dense 작업 준비', 'Preparing the next dense operation'],
  photometric: ['사진에서 깊이 추정', 'Estimating depths'],
  geometric: ['깊이 맵 검증', 'Verifying depth maps'],
  fusion: ['깊이 합치기', 'Combining depths'],
  allocate: ['융합 공간 준비', 'Preparing fusion space'],
  assign: ['구역 준비', 'Preparing region'],
  integrate: ['깊이 합치기', 'Combining depths'],
  carve: ['빈 공간 확인', 'Checking free space'],
  count: ['표면 찾기', 'Finding surfaces'],
  extract: ['포인트 추출', 'Extracting points'],
  readback: ['결과 가져오기', 'Reading results'],
  sort: ['최종 포인트 정리', 'Ordering final points'],
  clean: ['노이즈 정리', 'Cleaning points'],
  mesh: ['표면 연결', 'Building surface'],
};

/** An awaited readback or synchronous sort has no measurable intermediate percentage. */
export function denseActivityProgress(
  activity: DenseActivity,
): { done: number; total: number; percent: number; submitted: boolean } | undefined {
  if (
    ![
      'photometric',
      'geometric',
      'fusion',
      'allocate',
      'assign',
      'integrate',
      'carve',
      'count',
      'extract',
    ].includes(activity.phase)
  )
    return undefined;
  const done = activity.phaseDone ?? activity.done;
  const total = activity.phaseTotal ?? activity.total;
  if (!(total > 0)) return undefined;
  const bounded = Math.max(0, Math.min(done, total));
  return {
    done: bounded,
    total,
    percent: Math.floor((100 * bounded) / total),
    submitted: !['photometric', 'geometric', 'fusion'].includes(activity.phase),
  };
}

const DESCRIPTIONS: Record<string, [string, string]> = {
  waiting: ['다음 작업을 기다리고 있습니다.', 'Waiting for the next operation.'],
  photometric: [
    '사진을 비교해 카메라별 깊이를 구합니다.',
    'Comparing photos to estimate each camera’s depth map.',
  ],
  geometric: [
    '다른 카메라의 깊이와 비교해 잘못된 깊이를 걸러냅니다.',
    'Comparing depth maps across cameras to reject inconsistent depths.',
  ],
  carve: [
    '카메라별 깊이로 표면 앞의 빈 공간을 확인합니다.',
    'Checking space in front of surfaces against each camera’s depth.',
  ],
  count: [
    '융합 공간을 훑어 추출할 표면 포인트를 셉니다.',
    'Scanning the fused volume to count surface points.',
  ],
  extract: [
    '표면의 위치·색상·방향을 추출합니다.',
    'Extracting surface positions, colors and normals.',
  ],
  readback: [
    'GPU 작업이 끝나고 결과가 도착하기를 기다립니다.',
    'Waiting for GPU work to finish and results to arrive.',
  ],
  sort: [
    '추출한 포인트를 합치고 최종 순서로 정리합니다.',
    'Combining extracted points and putting them in final order.',
  ],
  clean: ['포인트의 노이즈와 중복을 정리합니다.', 'Removing point noise and duplicates.'],
  mesh: ['포인트를 연결해 표면을 만듭니다.', 'Connecting points into a surface.'],
};

export function denseActivityDescription(phase: string, ko: boolean): string {
  return (
    DESCRIPTIONS[phase]?.[ko ? 0 : 1] ??
    (ko ? '깊이를 3D 공간에 합치고 있습니다.' : 'Combining depths in 3D space.')
  );
}

export function denseActivityText(activity: DenseActivity, ko: boolean): string {
  const label = LABELS[activity.phase]?.[ko ? 0 : 1] ?? (ko ? 'Dense 처리 중' : 'Dense processing');
  const details: string[] = [];
  if (activity.component !== undefined)
    details.push(ko ? `묶음 ${activity.component + 1}` : `Component ${activity.component + 1}`);
  if (activity.pass && activity.passes)
    details.push(
      ko
        ? `구역 ${activity.pass}/${activity.passes}`
        : `Region ${activity.pass}/${activity.passes}`,
    );
  if (activity.viewId !== undefined)
    details.push(
      (ko ? '카메라 #' : 'Camera #') +
        activity.viewId +
        (activity.view !== undefined && activity.views
          ? ` (${activity.view + 1}/${activity.views})`
          : ''),
    );
  return [label, ...details].join(' · ');
}

/** Temporary fusion boxes never enter the final cloud/export and are replaced by final points. */
export function appendFusionPreview(
  previous: DenseCloud | undefined,
  run: number,
  chunk: DenseChunk,
): DenseCloud {
  const same = previous?.run === run;
  return {
    run,
    chunks: [...(same ? previous.chunks : []), chunk],
    count: (same ? previous.count : 0) + chunk.positions.length / 3,
  };
}
