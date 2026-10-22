/** Drawing state between reported maps. Targets stay in the mapper's coordinates. */
export const TWEEN_S = 0.8;
export const GHOST_S = 1.2;
export const OLD = -1e6;

export function transitionProgress(age: number, duration = TWEEN_S): number {
  const t = Math.max(0, Math.min(1, age / duration));
  return t * t * (3 - 2 * t);
}

export interface VertexState {
  readonly positions: Float32Array;
  readonly from: Float32Array;
  readonly born: Float32Array;
  readonly started: Float32Array;
}

/** Sample the same smoothstep as the vertex shader, including an interrupted move. */
export function sampleVertex(
  state: VertexState,
  index: number,
  now: number,
  out: Float32Array,
  offset = 0,
): void {
  const t = transitionProgress(now - state.started[index]!);
  for (let k = 0; k < 3; k++) {
    const at = 3 * index + k;
    out[offset + k] = state.from[at]! + (state.positions[at]! - state.from[at]!) * t;
  }
}

/**
 * Keep birth times and unfinished moves by identity, even when vertices reorder. Each vertex
 * has its own clock: updating another layer cannot restart or finish this one's transition.
 */
export class LiveVertices<Key> {
  private indices = new Map<Key, number>();
  state: VertexState = {
    positions: new Float32Array(),
    from: new Float32Array(),
    born: new Float32Array(),
    started: new Float32Array(),
  };
  removed = { positions: new Float32Array(), born: new Float32Array(), indices: [] as number[] };

  update(
    keys: ArrayLike<Key>,
    positions: Float32Array,
    now: number,
    live: boolean,
    motion = live,
    epsilon = 0,
  ): VertexState {
    const previous = this.state;
    const next: VertexState = {
      // The caller may reuse its upload buffer on the next map.
      positions: new Float32Array(positions),
      from: new Float32Array(positions),
      born: new Float32Array(keys.length).fill(OLD),
      started: new Float32Array(keys.length).fill(OLD),
    };
    const indices = new Map<Key, number>();
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i]!;
      indices.set(key, i);
      const before = this.indices.get(key);
      if (before === undefined) {
        if (live) next.born[i] = now;
        continue;
      }
      next.born[i] = live ? previous.born[before]! : OLD;
      if (!live || !motion) continue;
      let changed = false;
      for (let k = 0; k < 3; k++)
        if (Math.abs(next.positions[3 * i + k]! - previous.positions[3 * before + k]!) > epsilon)
          changed = true;
      if (changed) {
        sampleVertex(previous, before, now, next.from, 3 * i);
        next.started[i] = now;
      } else {
        next.from.set(previous.from.subarray(3 * before, 3 * before + 3), 3 * i);
        next.started[i] = previous.started[before]!;
      }
    }
    const removed: number[] = [];
    for (const [key, index] of this.indices) if (!indices.has(key)) removed.push(index);
    this.removed = {
      positions: new Float32Array(removed.length * 3),
      born: new Float32Array(removed.length),
      indices: removed,
    };
    for (let i = 0; i < removed.length; i++) {
      sampleVertex(previous, removed[i]!, now, this.removed.positions, 3 * i);
      this.removed.born[i] = previous.born[removed[i]!]!;
    }
    this.indices = indices;
    this.state = next;
    return next;
  }

  clear(): void {
    this.update([], new Float32Array(), 0, false);
  }
}

/** Keep overlapping deletion batches on their own clocks instead of replacing the last batch. */
export class LiveGhosts {
  positions = new Float32Array();
  colors = new Float32Array();
  died = new Float32Array();
  scales = new Float32Array();
  until = OLD;

  append(positions: Float32Array, born: Float32Array, colors: Float32Array, now: number): void {
    const kept: number[] = [];
    for (let i = 0; i < this.died.length; i++) if (now - this.died[i]! < GHOST_S) kept.push(i);
    const n = kept.length + born.length;
    const nextPositions = new Float32Array(n * 3),
      nextColors = new Float32Array(n * 3);
    const died = new Float32Array(n),
      scales = new Float32Array(n);
    for (let i = 0; i < kept.length; i++) {
      const before = kept[i]!;
      nextPositions.set(this.positions.subarray(3 * before, 3 * before + 3), 3 * i);
      nextColors.set(this.colors.subarray(3 * before, 3 * before + 3), 3 * i);
      died[i] = this.died[before]!;
      scales[i] = this.scales[before]!;
    }
    nextPositions.set(positions, 3 * kept.length);
    nextColors.set(colors, 3 * kept.length);
    for (let i = 0; i < born.length; i++) {
      died[kept.length + i] = now;
      scales[kept.length + i] = transitionProgress(now - born[i]!);
    }
    this.positions = nextPositions;
    this.colors = nextColors;
    this.died = died;
    this.scales = scales;
    this.until = now + GHOST_S;
  }

  clear(): void {
    this.positions = new Float32Array();
    this.colors = new Float32Array();
    this.died = new Float32Array();
    this.scales = new Float32Array();
    this.until = OLD;
  }
}
