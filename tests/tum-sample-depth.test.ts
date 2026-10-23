// Verifies: docs/dense-mvs.md step 8 (section 7.3): every frame of the TUM sample has a 16-bit
// grayscale depth PNG at the analysis size.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const folder = join(import.meta.dirname, './fixtures/lab-images/tum-fr3-structure-texture-far');

test('the TUM sample holds a 16-bit depth PNG of the analysis size for each frame', () => {
  const sample = JSON.parse(readFileSync(join(folder, 'sample.json'), 'utf8'));
  assert.equal(sample.depthUnitsPerMeter, 5000);
  for (const frame of sample.frames) {
    const png = readFileSync(join(folder, frame.depthFile));
    assert.equal(png.toString('latin1', 1, 4), 'PNG');
    // IHDR: width, height, bit depth, color type (0 = grayscale).
    assert.deepEqual(
      [png.readUInt32BE(16), png.readUInt32BE(20), png[24], png[25]],
      [frame.width, frame.height, 16, 0],
    );
  }
});
