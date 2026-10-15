import { resident } from 'typeshade/runtime';
import { fuse } from 'typeshade/fuse';
import { epipolarSampsonSquared, normalizeMatches } from './epipolar.shade.ts';
import type { CameraIntrinsics } from '../../src/index.ts';

/**
 * TypeShade call-layer integration, following TypeNN's resident + fuse scheme.
 * The caller configures TypeShade's CPU/WebGPU runtime once (configure/createRuntime).
 * Returns f32 Sampson errors, normalized-camera-coordinate units squared.
 * Does not manage the application's shared GPUDevice.
 */
export async function scoreEpipolarOnTypeShade(
  pixels: Float64Array,
  E: Float64Array,
  first: CameraIntrinsics,
  second: CameraIntrinsics,
): Promise<{ scores: Float32Array; report: Awaited<ReturnType<typeof fuse>> }> {
  if (pixels.length % 4 !== 0 || E.length !== 9) throw new RangeError('Invalid packed matches or Essential matrix');
  const input = resident(Float32Array.from(pixels));
  const essential = resident(Float32Array.from(E));
  const output = resident(new Float32Array(pixels.length / 4));
  try {
    const report = await fuse(scope => {
      const normalized = scope.scratch(pixels.length);
      normalizeMatches(
        normalized, input,
        first.fx, first.fy, first.cx, first.cy,
        second.fx, second.fy, second.cx, second.cy,
      );
      epipolarSampsonSquared(output, normalized, essential);
    });
    return { scores: await output.read(), report };
  } finally {
    output.destroy();
    essential.destroy();
    input.destroy();
  }
}
