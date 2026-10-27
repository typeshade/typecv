/**
 * The lab's stage ledger (`docs/library-structure.md`, section 3.4, `ledger`): the bound of the
 * detailed stage list that the SfM worker keeps for the page and the run record. The totals of each
 * operation cover every stage. This module has no DOM and no worker globals, so a unit test runs it
 * in Node.
 */
import { LAB_OPTIONS } from './lab-options.ts';

/**
 * Bound the detailed stage list to `detailedStageLimit`: drop the oldest entries, keyframe entries
 * last. The focal search runs hundreds of operations at once, and the page still lists each
 * keyframe.
 */
export function trimStages(stages: { operation: string }[]) {
  let excess = stages.length - LAB_OPTIONS.detailedStageLimit;
  for (let i = 0; excess > 0 && i < stages.length;)
    if (stages[i]!.operation === 'keyframe') i++;
    else {
      stages.splice(i, 1);
      excess--;
    }
  if (excess > 0) stages.splice(0, excess);
}
