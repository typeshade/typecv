/** Test-only: await actual browser worker closure; never terminate a worker for the caller. */
export function waitForWorkersClosed(page, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const watched = new Map();
    let finished = false;
    const finish = (error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      page.off('worker', watch);
      for (const [worker, closed] of watched) worker.off('close', closed);
      watched.clear();
      if (error) reject(error);
      else resolve();
    };
    const check = () => {
      if (finished) return;
      for (const worker of page.workers()) watch(worker);
      if (watched.size === 0 && page.workers().length === 0) finish();
    };
    const watch = (worker) => {
      if (finished || watched.has(worker)) return;
      const closed = () => {
        watched.delete(worker);
        queueMicrotask(check);
      };
      watched.set(worker, closed);
      worker.once('close', closed);
    };
    page.on('worker', watch);
    const timer = setTimeout(() => {
      const remaining = [...new Set([...watched.keys(), ...page.workers()])];
      finish(
        new Error('Workers did not close: ' + remaining.map((worker) => worker.url()).join(', ')),
      );
    }, timeoutMs);
    check();
  });
}
