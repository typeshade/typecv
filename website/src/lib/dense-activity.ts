import type { DenseChunk, DenseCloud } from './dense-cloud';

/** Real worker work, rather than a time-based estimate of the whole MVS. */
export interface DenseActivity {
  phase: string;
  done: number;
  total: number;
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
