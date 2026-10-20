import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import {
  ORB_PATTERN,
  ORB_UMAX,
  extractOrbCpu,
  fastAtan2,
  orbCorners,
  orbGray,
  orbHarrisKey,
  orbLevelLayout,
  orbSelectTop,
} from '../src/gpu/orb-reference.ts';
import { grayToRgba, hamming, polygonImage } from './orb-images.ts';

interface Fixture {
  opencv: string;
  width: number;
  height: number;
  gray: string;
  keypoints: [number, number, number, number, number, number][];
  descriptors: string;
}
const fixture: Fixture = JSON.parse(
  readFileSync(new URL('./fixtures/orb-opencv.json', import.meta.url), 'utf8'),
);
const unpack = (text: string) => new Uint8Array(inflateSync(Buffer.from(text, 'base64')));

test('the ORB reference reproduces OpenCV: keypoints per level and bit-identical descriptors', () => {
  const { width, height } = fixture,
    gray = unpack(fixture.gray),
    expected = unpack(fixture.descriptors),
    rgba = grayToRgba(gray, width, height),
    mine = extractOrbCpu(rgba, width, height, { maxFeatures: 300 }),
    layout = orbLevelLayout(width, height, {
      ...{ maxFeatures: 300, levels: 8, scaleFactor: 1.2, fastThreshold: 20 },
    });
  // OpenCV's keypoint is the level pixel times the nominal scale; ours is the pixel centre in the image.
  const reference = new Map<string, number>();
  fixture.keypoints.forEach((k, j) => {
    const level = layout[k[5]]!;
    reference.set(`${k[5]}:${Math.round(k[0] / level.scale)}:${Math.round(k[1] / level.scale)}`, j);
  });
  let found = 0;
  const perLevel = new Array(8).fill(0);
  for (let i = 0; i < mine.count; i++) {
    const level = layout[mine.details[i * 8 + 5]!]!;
    perLevel[level.index]++;
    const lx = Math.round((mine.keypoints[i * 2]! + 0.5) / level.scaleX - 0.5),
      ly = Math.round((mine.keypoints[i * 2 + 1]! + 0.5) / level.scaleY - 0.5),
      j = reference.get(`${level.index}:${lx}:${ly}`);
    if (j === undefined) continue;
    found++;
    let distance = 0;
    for (let w = 0; w < 8; w++) {
      let v =
        (mine.descriptors[i * 8 + w]! ^
          new DataView(expected.buffer).getUint32(j * 32 + w * 4, true)) >>>
        0;
      while (v) {
        distance += v & 1;
        v >>>= 1;
      }
    }
    assert.equal(
      distance,
      0,
      `descriptor of keypoint ${i} (level ${level.index}) differs from OpenCV ${fixture.opencv}`,
    );
  }
  const expectedPerLevel = new Array(8).fill(0);
  for (const k of fixture.keypoints) expectedPerLevel[k[5]]++;
  assert.deepEqual(perLevel, expectedPerLevel);
  // Ties at the boundary of a ranking may keep another corner; everything else is identical.
  assert.ok(
    found >= 0.97 * fixture.keypoints.length,
    `${found} of ${fixture.keypoints.length} keypoints match`,
  );
});

test('FAST-9 finds the corners of a square and nothing on its edges or in flat areas', () => {
  const size = 200,
    rgba = polygonImage(size, size, [
      [60, 60],
      [140, 60],
      [140, 140],
      [60, 140],
    ]),
    gray = orbGray(rgba, size, size),
    corners = orbCorners(gray, size, size, 20);
  assert.ok(corners.length >= 4);
  const vertices = [
    [60, 60],
    [139, 60],
    [139, 139],
    [60, 139],
  ];
  for (const c of corners)
    assert.ok(
      vertices.some(([x, y]) => Math.hypot(c.x - x!, c.y - y!) <= 2),
      `corner (${c.x}, ${c.y}) is away from every vertex`,
    );
  for (const [x, y] of vertices)
    assert.ok(
      corners.some((c) => Math.hypot(c.x - x!, c.y - y!) <= 2),
      `no corner at (${x}, ${y})`,
    );
  const features = extractOrbCpu(rgba, size, size, { maxFeatures: 50 });
  assert.ok(features.count >= 4);
});

test('the intensity centroid points from the vertex to the inside of a rotated square', () => {
  const size = 260;
  for (const degrees of [0, 30, 60, 100, 145, 200, 250, 310]) {
    const phi = (degrees * Math.PI) / 180,
      centre = [130, 130] as const,
      half = 45,
      vertices = [0, 1, 2, 3].map((k) => {
        const a = phi + (k * Math.PI) / 2 + Math.PI / 4;
        return [
          centre[0] + half * Math.SQRT2 * Math.cos(a),
          centre[1] + half * Math.SQRT2 * Math.sin(a),
        ] as [number, number];
      }),
      // Counter-clockwise in image coordinates is the order the polygon test expects.
      rgba = polygonImage(size, size, vertices),
      features = extractOrbCpu(rgba, size, size, { maxFeatures: 80, levels: 1 });
    let checked = 0;
    for (let i = 0; i < features.count; i++) {
      const x = features.keypoints[i * 2]!,
        y = features.keypoints[i * 2 + 1]!,
        vertex = vertices.find(([vx, vy]) => Math.hypot(vx - x, vy - y) <= 3);
      if (!vertex) continue;
      checked++;
      const expected = Math.atan2(centre[1] - vertex[1], centre[0] - vertex[0]),
        angle = features.details[i * 8 + 3]!;
      let error = Math.abs(angle - expected) % (2 * Math.PI);
      if (error > Math.PI) error = 2 * Math.PI - error;
      assert.ok(
        (error * 180) / Math.PI < 12,
        `square rotated ${degrees} degrees: angle ${((angle * 180) / Math.PI).toFixed(1)} against ${((expected * 180) / Math.PI).toFixed(1)}`,
      );
    }
    assert.ok(checked >= 2, `square rotated ${degrees} degrees: ${checked} corners checked`);
  }
});

test('the orientation circle, the rank key and the pattern table keep their shape', () => {
  assert.equal(ORB_PATTERN.length, 1024);
  assert.ok(ORB_PATTERN.every((v) => v >= -13 && v <= 12));
  assert.deepEqual(
    Array.from(ORB_UMAX),
    [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3, 0],
  );
  for (const [y, x] of [
    [3, 4],
    [-5, 2],
    [0, -1],
    [-2, -2],
    [1, 0],
  ] as const)
    assert.ok(
      Math.abs(((fastAtan2(y, x) - (Math.atan2(y, x) * 180) / Math.PI + 540) % 360) - 180) < 0.3,
    );
  assert.equal(orbHarrisKey(-1), 0);
  assert.equal(orbHarrisKey(0), 0);
  assert.ok(orbHarrisKey(1e-3) < orbHarrisKey(2e-3));
  // The top three of eight: the bin above the boundary, then the boundary bin in index order.
  assert.deepEqual(
    Array.from(orbSelectTop([5, 1, 5, 9, 5, 2, 7, 5], 3, 16)),
    [1, 0, 0, 1, 0, 0, 1, 0],
  );
  assert.deepEqual(Array.from(orbSelectTop([1, 2], 3, 16)), [1, 1]);
  assert.equal(hamming(new Uint32Array(8), 0, new Uint32Array(8).fill(0xffffffff), 0), 256);
});
