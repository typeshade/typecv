/**
 * The batches of a GPU search between two reads of the scheduling word: the PnP search
 * (`pnp.ts`) and the pair search (`pair-estimation.ts`). The GPU decides each batch's work
 * through the indirect workgroup counts (TypeShade change 0075), so the host reads nothing
 * between batches. It reads the adaptive budget after every fourth batch to leave the search
 * early: at most three batches past the budget are submitted, and each dispatches zero workgroups
 * but its one-workgroup control kernels (`planPnPBatch` of PnP, three of the pair search).
 */
export const PROGRESS_INTERVAL = 4;
