import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GHOST_S,
  LiveGhosts,
  LiveVertices,
  OLD,
  TWEEN_S,
  sampleVertex,
} from './src/lib/live-vertices.ts';

const at = (vertices, index, time) => {
  const position = new Float32Array(3);
  sampleVertex(vertices.state, index, time, position);
  return [...position];
};

test('a 500 ms snapshot preserves a birth until its 800 ms growth finishes', () => {
  const vertices = new LiveVertices();
  vertices.update([7], Float32Array.of(0, 0, 1), 10, true);
  vertices.update([7], Float32Array.of(0, 0, 1), 10.5, true);
  assert.equal(vertices.state.born[0], 10);
  assert.ok(10.5 - vertices.state.born[0] < TWEEN_S);
  vertices.update([8, 7], Float32Array.of(1, 0, 1, 0, 0, 1), 10.6, true);
  assert.deepEqual([...vertices.state.born], [Math.fround(10.6), 10]);
});

test('an unchanged snapshot leaves an unfinished move on its original clock', () => {
  const vertices = new LiveVertices();
  vertices.update([7], Float32Array.of(0, 0, 1), 0, false);
  vertices.update([7], Float32Array.of(10, 0, 1), 10, true);
  const before = at(vertices, 0, 10.5);
  vertices.update([7], Float32Array.of(10, 0, 1), 10.5, true);
  assert.deepEqual(at(vertices, 0, 10.5), before);
  assert.deepEqual(at(vertices, 0, 10.8), [10, 0, 1]);
});

test('a changed target continues from the drawn position and survives reordering', () => {
  const vertices = new LiveVertices();
  vertices.update([7, 8], Float32Array.of(0, 0, 1, 20, 0, 1), 0, false);
  vertices.update([7, 8], Float32Array.of(10, 0, 1, 20, 0, 1), 10, true);
  const before = at(vertices, 0, 10.5);
  vertices.update([8, 7], Float32Array.of(20, 0, 1, 30, 0, 1), 10.5, true);
  assert.deepEqual(at(vertices, 1, 10.5), before);
  assert.deepEqual(at(vertices, 1, 11.3), [30, 0, 1]);
  assert.deepEqual(at(vertices, 0, 10.5), [20, 0, 1]);
});

test('a deletion starts at the drawn position and retains the unfinished birth', () => {
  const vertices = new LiveVertices();
  vertices.update([7], Float32Array.of(0, 0, 1), 10, true);
  vertices.update([7], Float32Array.of(10, 0, 1), 10.2, true);
  const before = at(vertices, 0, 10.5);
  vertices.update([], new Float32Array(), 10.5, true);
  assert.deepEqual([...vertices.removed.positions], before);
  assert.equal(vertices.removed.born[0], 10);
});

test('layers have independent clocks and reused caller buffers cannot change history', () => {
  const first = new LiveVertices(),
    second = new LiveVertices();
  const buffer = Float32Array.of(0, 0, 1);
  first.update([1], buffer, 0, false);
  buffer[0] = 10;
  first.update([1], buffer, 10, true);
  const before = at(first, 0, 10.5);
  second.update([1], Float32Array.of(1, 2, 3), 10.5, true);
  buffer[0] = 100;
  assert.deepEqual(at(first, 0, 10.5), before);
});

test('saved results and reduced motion use reported positions immediately', () => {
  const vertices = new LiveVertices();
  vertices.update([7], Float32Array.of(0, 0, 1), 10, false);
  assert.equal(vertices.state.born[0], OLD);
  vertices.update([7, 8], Float32Array.of(10, 0, 1, 20, 0, 1), 11, true, false);
  assert.deepEqual(at(vertices, 0, 11), [10, 0, 1]);
  assert.equal(vertices.state.born[1], 11); // Color still identifies new points.
  vertices.clear();
  vertices.update([7], Float32Array.of(0, 0, 1), 12, true);
  assert.equal(vertices.state.born[0], 12);
});

test('successive deletions retain their own shrink clocks and expire independently', () => {
  const ghosts = new LiveGhosts();
  ghosts.append(Float32Array.of(1, 0, 1), Float32Array.of(10), Float32Array.of(1, 0, 0), 10.4);
  const initialScale = ghosts.scales[0];
  assert.ok(initialScale > 0 && initialScale < 1);
  ghosts.append(Float32Array.of(2, 0, 1), Float32Array.of(OLD), Float32Array.of(0, 1, 0), 10.8);
  assert.deepEqual([...ghosts.positions], [1, 0, 1, 2, 0, 1]);
  assert.deepEqual([...ghosts.died], [Math.fround(10.4), Math.fround(10.8)]);
  assert.equal(ghosts.scales[0], initialScale);
  assert.equal(ghosts.scales[1], 1);
  ghosts.append(
    Float32Array.of(3, 0, 1),
    Float32Array.of(OLD),
    Float32Array.of(0, 0, 1),
    10.4 + GHOST_S + 0.1,
  );
  assert.deepEqual([...ghosts.positions], [2, 0, 1, 3, 0, 1]);
  ghosts.clear();
  assert.equal(ghosts.positions.length, 0);
});
