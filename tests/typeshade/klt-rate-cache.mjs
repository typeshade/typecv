import assert from 'node:assert/strict';

/** Check cache reuse at the actual sizing boundary, independent of adaptive chunk counts. */
export function assertKltRateCache(passes, timedChunks, sampleLimit) {
  const [cold, warm] = passes;
  const coldRate = cold.rates.kltRoundtrip;
  const warmRate = warm.rates.kltRoundtrip;
  assert.equal(cold.rateBefore, null, 'the dedicated device starts without a roundtrip rate');
  assert.equal(cold.firstSizingRate, null, 'the first cold sizing has no cached rate');
  assert.ok(cold.sizingCalls > 0 && warm.sizingCalls > 0, 'both passes size real roundtrips');
  assert.ok(coldRate && Number.isFinite(coldRate.msPerUnit) && coldRate.msPerUnit > 0);
  assert.ok(coldRate.chunks > 0 && coldRate.chunks <= sampleLimit);
  assert.deepEqual(warm.rateBefore, coldRate, 'warm entry reuses the cold rate');
  assert.deepEqual(warm.firstSizingRate, coldRate, 'first warm sizing reuses the cold rate');
  const warmRanges = warm.trackChunks.length / 2;
  assert.ok(Number.isSafeInteger(warmRanges) && warmRanges > 0);
  assert.equal(
    warmRate.chunks,
    Math.min(sampleLimit, coldRate.chunks + warmRanges),
    'warm tracking keeps learning until the same rate reaches its sample limit',
  );
  assert.equal(cold.timedBefore, 0);
  assert.equal(cold.timedAfter, coldRate.chunks);
  assert.equal(warm.timedBefore, cold.timedAfter);
  assert.equal(warm.timedAfter, warmRate.chunks);
  assert.equal(timedChunks.length, warmRate.chunks);
  assert.equal(
    coldRate.longestMs,
    Math.max(...timedChunks.slice(0, cold.timedAfter).map((chunk) => chunk.elapsedMs)),
    'cold timing history matches its observations',
  );
  assert.equal(
    warmRate.longestMs,
    Math.max(...timedChunks.map((chunk) => chunk.elapsedMs)),
    'warm timing history retains the cold observations',
  );
  if (coldRate.chunks === sampleLimit)
    assert.deepEqual(warmRate, coldRate, 'an already-settled rate remains cached');
}
