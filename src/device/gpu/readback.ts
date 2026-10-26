/**
 * Start independent reads together at the call site. On failure, let every read settle before
 * the caller releases their resident buffers; Promise.all alone rejects while siblings run.
 */
export async function readTogether<T extends readonly unknown[]>(reads: {
  readonly [K in keyof T]: Promise<T[K]>;
}): Promise<T> {
  try {
    return await Promise.all(reads);
  } catch (error) {
    await Promise.allSettled(reads);
    throw error;
  }
}
