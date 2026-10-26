import { TSDF_HEADER, tsdfInBox, tsdfKeyBlock, type TsdfBox, type TsdfVolume } from '../tsdf.ts';

const EMPTY = 0xffffffff;

/** The same u32 avalanche and linear probing as the TSDF kernels. */
export function tsdfKeySlot(key: number, mask: number): number {
  let h = Math.imul(key, 0x9e3779b1);
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  return ((h ^ (h >>> 16)) & mask) >>> 0;
}

/**
 * A pass's unassigned table from a COMPLETE successful discovery. Allocation's ray walk does
 * not depend on the pass box: the box only gates insertion of each discovered key. Therefore
 * filtering the discovery's keys, with the identical positive-neighbour margin, gives precisely
 * the keys another allocation pass would claim. Slot/block order is private and may already
 * differ between concurrent GPU claims; integration and extraction address blocks by key.
 */
export function tsdfBoxTable(volume: TsdfVolume, keys: ArrayLike<number>, box: TsdfBox) {
  const selected: number[] = [];
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i]!;
    const [level, x, y, z] = tsdfKeyBlock(volume, key);
    if (tsdfInBox(volume, box, level, [x, y, z], true)) selected.push(key);
  }
  const slots = 2 ** Math.ceil(Math.log2(Math.max(64, 2 * selected.length)));
  const words = new Uint32Array(TSDF_HEADER + 3 * slots);
  words.fill(EMPTY, TSDF_HEADER, TSDF_HEADER + slots);
  for (const key of selected) {
    let slot = tsdfKeySlot(key, slots - 1);
    while (words[TSDF_HEADER + slot] !== EMPTY) slot = (slot + 1) & (slots - 1);
    words[TSDF_HEADER + slot] = key;
  }
  words[0] = selected.length;
  return { words, slots, blocks: selected.length };
}
