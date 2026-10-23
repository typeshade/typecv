import test from 'node:test';
import assert from 'node:assert/strict';
import {
  VideoActivityGraph,
  appendKeyframeThumbnail,
  MAX_ACTIVITY_LINKS,
  MAX_ACTIVITY_THUMBNAILS,
  MAX_ACTIVITY_THUMBNAILS_PER_VIDEO,
} from './src/lib/multi-video-activity.ts';

test('video graph distinguishes candidates, matches and accepted geometry and snapshots have no aliases', () => {
  const graph = new VideoActivityGraph();
  const videoOf = new Map([
    [1, 'a'],
    [2, 'a'],
    [3, 'b'],
    [4, 'c'],
  ]);
  const pairs = [
    { first: 1, second: 3 },
    { first: 2, second: 3 },
    { first: 1, second: 4 },
    { first: 1, second: 2 },
  ];
  graph.update({ stage: 'candidates', phase: 'complete', pairs }, videoOf);
  assert.deepEqual(
    graph.snapshot().links.map((link) => link.candidates),
    [2, 1],
  );
  graph.update({ stage: 'matching', phase: 'start', pairs: [{ first: 1, second: 2 }] }, videoOf);
  assert.deepEqual(
    graph.snapshot().activeVideos,
    ['a'],
    'within-video work still highlights its input',
  );
  graph.update(
    {
      stage: 'matching',
      phase: 'complete',
      pairs: pairs.map((pair) => ({ ...pair, matches: 40 })),
    },
    videoOf,
  );
  assert.ok(graph.snapshot().links.every((link) => link.accepted === 0));
  graph.update(
    {
      stage: 'verification',
      phase: 'complete',
      pairs: [
        { first: 1, second: 3, accepted: true, inliers: 32 },
        { first: 2, second: 3, accepted: false, inliers: 20 },
        { first: 1, second: 4, accepted: false, inliers: 0 },
      ],
    },
    videoOf,
  );
  graph.update(
    {
      stage: 'components',
      components: [
        { index: 0, videoIds: ['a', 'b'], viewIds: [1, 2, 3], reconstructed: true },
        { index: 1, videoIds: ['c'], viewIds: [4], reconstructed: false },
      ],
    },
    videoOf,
  );
  const snapshot = graph.snapshot();
  assert.deepEqual(
    [
      snapshot.links[0].matched,
      snapshot.links[0].verified,
      snapshot.links[0].accepted,
      snapshot.links[0].inliers,
    ],
    [2, 2, 1, 32],
  );
  assert.equal(snapshot.links[1].accepted, 0, 'rejected pairs never become a verified connection');
  snapshot.links[0].accepted = 100;
  snapshot.components[0].videos.push('wrong');
  assert.equal(graph.snapshot().links[0].accepted, 1);
  assert.deepEqual(graph.snapshot().components[0].videos, ['a', 'b']);
});

test('activity links and recent frame images are bounded independently of solver data', () => {
  const graph = new VideoActivityGraph();
  const videos = new Map(Array.from({ length: MAX_ACTIVITY_LINKS + 2 }, (_, i) => [i, String(i)]));
  graph.update(
    {
      stage: 'candidates',
      phase: 'complete',
      pairs: Array.from({ length: MAX_ACTIVITY_LINKS + 1 }, (_, i) => ({
        first: 0,
        second: i + 1,
      })),
    },
    videos,
  );
  assert.equal(graph.snapshot().links.length, MAX_ACTIVITY_LINKS);
  assert.equal(graph.snapshot().limited, true);
  let frames = [];
  for (let i = 0; i < MAX_ACTIVITY_THUMBNAILS * 2; i++)
    frames = appendKeyframeThumbnail(frames, {
      videoId: String(i % 20),
      frameId: i,
      time: i,
      image: 'fixture',
      features: 10,
    });
  assert.ok(frames.length <= MAX_ACTIVITY_THUMBNAILS);
  for (const id of new Set(frames.map((frame) => frame.videoId)))
    assert.ok(
      frames.filter((frame) => frame.videoId === id).length <= MAX_ACTIVITY_THUMBNAILS_PER_VIDEO,
    );
  frames = appendKeyframeThumbnail(frames, { ...frames.at(-1), features: 30 });
  assert.equal(frames.filter((frame) => frame.frameId === frames.at(-1).frameId).length, 1);
});
