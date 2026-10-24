import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('RGB-derived expansion grows structure while rotation and low parallax retain pending depth', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'observation-rgb-'));
  try {
    const output = join(dir, 'report.json');
    await promisify(execFile)(
      process.execPath,
      [
        'scripts/benchmark-observation-expansion.mjs',
        '--backend=webgpu',
        '--cases=render-translation,render-translation-omission,render-rotation,render-low-parallax',
        `--output=${output}`,
      ],
      { cwd: new URL('../../', import.meta.url), timeout: 180000, maxBuffer: 1 << 22 },
    );
    const result = JSON.parse(await readFile(output, 'utf8'));
    assert.equal(result.diagnostics.memory.liveBytes, 0);
    for (const row of result.cases) {
      const a = row.baseline,
        b = row.expanded;
      assert.ok(row.expansion.newCoordinates > 100);
      if (['render-rotation', 'render-low-parallax'].includes(row.name)) {
        assert.equal(b.landmarks, 0);
        assert.equal(b.registeredFrames, 0);
        assert.equal(b.initialization, 'deferred');
        assert.ok(b.pendingTracks > a.pendingTracks * 1.5);
        assert.ok(b.trackedObservations > a.trackedObservations * 1.5);
      } else {
        assert.equal(b.registeredFrames, 4);
        assert.ok(b.landmarks >= a.landmarks * 1.5);
        assert.ok(row.newLandmarksWithoutBaselineObservation >= 100);
        assert.deepEqual(b.initialPair, a.initialPair);
        assert.ok(b.poseTruth.cameraCenterRmse < 0.01);
        assert.ok(b.poseTruth.maxRotationErrorDeg < 0.2);
        assert.ok(b.depthTruth.medianRelativeError < 0.04);
        assert.ok(b.depthTruth.fractionAbove10Percent < 0.01);
      }
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
