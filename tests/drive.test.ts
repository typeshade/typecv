import test from 'node:test';
import assert from 'node:assert/strict';
import { drive, driveAsync } from '../src/sfm/protocol/drive.ts';
import { GeometryEstimationError } from '../src/geometry/errors.ts';
import { ReconstructionOperationError } from '../src/sfm/protocol/errors.ts';

// The error rules of the drivers (docs/library-structure.md, section 3.3, `sfm/protocol`). Each
// driver keeps the rule that it had before phase 4 wrote the rules in one place (section 4,
// rule 1). The CPU-path digests run 'pass' and 'end' (tests/characterization.ts); no other test
// gives the device driver an error that is not a GeometryEstimationError.

/** A mapper of two requests that records each error that comes back into it. */
function* mapper(seen: unknown[]): Generator<string, string, unknown> {
  for (const job of ['first', 'second']) {
    try {
      yield job;
    } catch (error) {
      seen.push(error);
    }
  }
  return 'done';
}
/** An executor whose first request throws `error`. */
const failing = (error: unknown) => (job: string) => {
  if (job === 'first') throw error;
  return job;
};

test("'pass' gives every error back to the mapper as it is (the CPU drivers)", () => {
  for (const error of [new GeometryEstimationError('No pose'), new RangeError('Invalid input')]) {
    const seen: unknown[] = [];
    assert.equal(drive(mapper(seen), failing(error), 'pass'), 'done');
    assert.equal(seen.length, 1);
    assert.equal(seen[0], error);
  }
});

test("'wrap' gives a GeometryEstimationError back as it is and wraps any other error (the device driver)", async () => {
  const rejection = new GeometryEstimationError('No pose'),
    failure = new RangeError('Device lost');
  const kept: unknown[] = [];
  assert.equal(
    await driveAsync(mapper(kept), async (job) => failing(rejection)(job), 'wrap'),
    'done',
  );
  assert.equal(kept.length, 1);
  assert.equal(kept[0], rejection);
  const wrapped: unknown[] = [];
  await driveAsync(mapper(wrapped), async (job) => failing(failure)(job), 'wrap');
  assert.equal(wrapped.length, 1);
  assert.ok(wrapped[0] instanceof ReconstructionOperationError);
  assert.equal((wrapped[0] as Error).cause, failure);
  assert.equal((wrapped[0] as Error).message, 'Reconstruction operation failed: Device lost');
});

test("'end' gives no error back, so the first one ends the run (the component driver)", () => {
  const error = new GeometryEstimationError('No pose'),
    seen: unknown[] = [];
  assert.throws(
    () => drive(mapper(seen), failing(error), 'end'),
    (thrown) => thrown === error,
  );
  assert.equal(seen.length, 0);
});
