// The stage list of a multi-video run (docs/multi-video.md, Amendment 8).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  STAGE_IDS,
  STALL_MS,
  advance,
  currentRow,
  finish,
  formatElapsed,
  newBoard,
  stageCount,
  stalled,
  statusLine,
  tick,
  touch,
} from './src/lib/multi-video-stages.ts';

const rowOf = (board, id) => board.rows.find((r) => r.id === id);

test('an event marks the stages before it done and the stage itself running', () => {
  let b = newBoard(0);
  assert.ok(b.rows.every((r) => r.state === 'pending'));
  b = advance(b, { stage: 'frames', done: 5, total: 40, detail: '3 keyframes' }, 1000);
  b = advance(b, { stage: 'matching', done: 10, total: 100 }, 5000);
  assert.deepEqual(
    STAGE_IDS.map((id) => rowOf(b, id).state),
    ['done', 'done', 'running', 'pending', 'pending', 'pending'],
  );
  assert.equal(rowOf(b, 'frames').elapsedMs, 4000);
  // A skipped stage is done at once and shows its time as zero.
  assert.equal(rowOf(b, 'candidates').elapsedMs, 0);
  b = tick(b, 8000);
  assert.equal(rowOf(b, 'matching').elapsedMs, 3000);
  assert.equal(rowOf(b, 'frames').elapsedMs, 4000, 'a done stage keeps its time');
});

test('a final event closes its stage, and a late event for a done stage changes nothing', () => {
  let b = newBoard(0);
  b = advance(b, { stage: 'candidates', done: 90, total: 90, final: true }, 100);
  assert.equal(rowOf(b, 'candidates').state, 'done');
  const again = advance(b, { stage: 'candidates', done: 1, total: 2 }, 200);
  assert.equal(rowOf(again, 'candidates').done, 90);
  assert.equal(again.lastEventAt, 200);
});

test('finish closes every stage', () => {
  let b = advance(newBoard(0), { stage: 'reconstruction', done: 3, total: 9 }, 10);
  b = finish(b, 50);
  assert.ok(b.rows.every((r) => r.state === 'done'));
  assert.equal(rowOf(b, 'reconstruction').done, 9);
});

test('the status line names the stage, its count and both times, and warns on a stall', () => {
  let b = newBoard(0);
  b = advance(b, { stage: 'verification', done: 312, total: 908 }, 10000);
  b = touch(tick(b, 52000), 52000);
  const line = statusLine(b, 52000, false);
  assert.match(line, /Pair verification/);
  assert.match(line, /312 \/ 908 pairs/);
  assert.match(line, /42 s/);
  assert.match(line, /total 52 s/);
  assert.doesNotMatch(line, /No progress/);
  assert.match(statusLine(b, 52000, true), /맞는 점 확인하는 중….*312 \/ 908 쌍/);
  // Kernel progress without a stage change keeps the warning away. Silence for STALL_MS brings it.
  const quiet = 52000 + STALL_MS;
  assert.equal(stalled(b, quiet - 1), false);
  assert.equal(stalled(b, quiet), true);
  assert.match(statusLine(b, quiet), /No progress from the GPU for 30 s/);
  assert.equal(stalled(touch(b, quiet), quiet), false);
  assert.match(statusLine(b, quiet, true), /30초째 GPU 응답이 없습니다/);
});

test('the status line moves every tick, so it is never still', () => {
  let b = advance(newBoard(0), { stage: 'reconstruction', done: 4, total: 93 }, 0);
  const lines = new Set();
  for (let t = 0; t <= 3000; t += 500) lines.add(statusLine(tick(b, t), t));
  assert.ok(lines.size >= 6);
  assert.equal(currentRow(b).id, 'reconstruction');
});

test('counts and times read well in both languages', () => {
  const row = (extra) => ({
    id: 'frames',
    state: 'running',
    done: 12,
    total: 40,
    elapsedMs: 0,
    ...extra,
  });
  assert.equal(stageCount(row({ detail: '4 keyframes' })), '12 / 40 frames · 4 keyframes');
  assert.equal(stageCount(row({ total: 0 })), '12 frames');
  assert.equal(stageCount(row(), true), '12 / 40 프레임');
  assert.equal(formatElapsed(800), '0.8 s');
  assert.equal(formatElapsed(125000), '2 min 5 s');
  assert.equal(formatElapsed(125000, true), '2분 5초');
});
