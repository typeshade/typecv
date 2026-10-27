/** The lines the page shows of a run record: chunk rates, the KLT block and the dense MVS. */
import {
  RUN_RECORD_DENSE_STAGES,
  type RunRecordChunk,
  type RunRecordDense,
  type RunRecordKlt,
} from './schema.ts';

/** One short line per chunk key: `scorePairModels: 37 chunks, longest 212 ms, 0.41 ms/unit`. */
export function chunkLines(chunks: Readonly<Record<string, RunRecordChunk>> | undefined): string[] {
  return Object.entries(chunks ?? {}).map(
    ([key, rate]) =>
      key +
      ': ' +
      rate.chunks +
      (rate.chunks === 1 ? ' chunk' : ' chunks') +
      ', longest ' +
      Math.round(rate.longestMs) +
      ' ms, ' +
      Number(rate.msPerUnit.toPrecision(2)) +
      ' ms/unit',
  );
}

/** One line for the page: `Dense MVS completed: 123456 points from 12 depth maps in 812 s …`. */
/** One line for a record's KLT block: the matches it added and the mean length of a track. */
export function kltLine(klt: RunRecordKlt | undefined): string | undefined {
  if (!klt) return undefined;
  return (
    'KLT ' +
    klt.matches +
    ' matches (' +
    klt.pairMatches +
    ' new), mean track ' +
    klt.meanLived.toFixed(1) +
    ' frames, ' +
    klt.rejected +
    ' rejected'
  );
}

export function denseLine(dense: RunRecordDense | undefined): string | undefined {
  if (!dense) return undefined;
  const stages = RUN_RECORD_DENSE_STAGES.map(
    (stage) => stage + ' ' + (dense.stageMs[stage] / 1000).toFixed(1) + ' s',
  ).join(', ');
  const before = (dense.memory.liveBytesBefore / 1024 ** 2).toFixed(1);
  return (
    'Dense MVS ' +
    dense.status +
    (dense.status === 'skipped' ? ' (' + dense.skipped + ')' : '') +
    ': ' +
    dense.points +
    ' points from ' +
    dense.views +
    ' depth maps in ' +
    (dense.totalMs / 1000).toFixed(1) +
    ' s (' +
    stages +
    '), ' +
    before +
    ' MiB live before' +
    (dense.clean
      ? dense.clean.error
        ? ', cleanup failed: ' + dense.clean.error
        : ', cleaned to ' + dense.clean.after + ' points'
      : '') +
    (dense.mesh
      ? dense.mesh.error
        ? ', mesh failed: ' + dense.mesh.error
        : ', mesh of ' + dense.mesh.faces + ' faces'
      : '')
  );
}
