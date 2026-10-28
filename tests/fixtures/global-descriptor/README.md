# Global descriptor sample metadata

`catalog.json` and `remote-videos.json` are frozen copies of
`website/public/sample/index.json` and `website/samples/remote-videos.json` from TypeCV revision
`db7216a4d0d763996b677bbf27deb0f3913c930c`. They define the research corpus used by
`scripts/global-descriptor-data.mjs`; they are data fixtures, not a copy of website implementation.

After moving the inputs to these fixtures, `listFrames()` serialized identically to the old script
against those two files: 248 frames, with matching ids, indices, times, sizes, groups and cameras.
