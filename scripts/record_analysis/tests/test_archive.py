"""Tests of the depth archive: segment concatenation, maps and inputs, safe extraction."""
from __future__ import annotations

import dataclasses
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_analysis.archive import (
    DepthArchive,
    Segment,
    UnsafeMemberError,
    extract_parts,
    inside,
    load_manifest,
    read_segments,
    verify_extraction,
)
from record_analysis.geometry import Pose
from record_analysis.tests import synthetic


class SegmentTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        (self.root / 'a').mkdir()
        (self.root / 'a/x.0000').write_bytes(b'0123')
        (self.root / 'a/x.0001').write_bytes(b'456')

    def tearDown(self):
        self.folder.cleanup()

    def test_concatenates_in_offset_order(self):
        segments = [Segment('a/x.0001', 4, 3), Segment('a/x.0000', 0, 4)]
        self.assertEqual(read_segments(self.root, segments), b'0123456')

    def test_refuses_a_gap_or_a_size_mismatch(self):
        with self.assertRaises(ValueError):
            read_segments(self.root, [Segment('a/x.0000', 0, 4), Segment('a/x.0001', 5, 3)])
        with self.assertRaises(ValueError):
            read_segments(self.root, [Segment('a/x.0000', 0, 5)])

    def test_refuses_a_path_outside_the_root(self):
        for path in ('../x', '/etc/passwd', 'a/../../x', 'a\\x', 'C:x', ''):
            with self.subTest(path=path), self.assertRaises(UnsafeMemberError):
                inside(self.root, path)

    def test_refuses_names_that_windows_reserves(self):
        # On Windows, view-1/CON opens the console and nul.txt the null device; a trailing dot or
        # space is dropped, so 'a.' would write to 'a'.
        for path in ('view-1/CON', 'view-1/nul.txt', 'com1', 'LPT9.json', 'aux .json', 'view-1/conin$', 'a.',
                     'view-1/a ', 'a|b', 'a?b', 'a*', 'a<b', 'a"b', 'a\x01b'):
            with self.subTest(path=path), self.assertRaises(UnsafeMemberError):
                inside(self.root, path)
        for path in ('view-1/console.json', 'view-1/CONx', 'view-1/geometric-filtered/depth.f32le.0000', 'a/./x'):
            with self.subTest(path=path):
                self.assertTrue(inside(self.root, path).is_relative_to(self.root.resolve()))


class DepthArchiveTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name) / 'extracted'
        synthetic.write_archive(self.root)
        self.archive = DepthArchive(self.root)

    def tearDown(self):
        self.folder.cleanup()

    def test_views_and_maps(self):
        self.assertEqual(self.archive.view_ids(), [0, 1])
        record = self.archive.map_record(1, 'geometric-filtered')
        self.assertEqual((record.width, record.height, record.depth_scale_to_model), (64, 48, 2.0))
        self.assertEqual(record.source_view_ids, (0,))
        self.assertEqual(len(record.arrays['depth']), 2)

    def test_depth_from_split_segments(self):
        record = self.archive.map_record(0, 'geometric-filtered')
        depth = self.archive.depth(record)
        expected = (synthetic.plane_depth(synthetic.poses()[0]) / synthetic.SCALE).astype(np.float32)
        expected[:, :synthetic.INVALID_COLUMNS] = 0
        np.testing.assert_array_equal(depth, expected)
        model = self.archive.model_depth(record)
        self.assertEqual(model.dtype, np.float32)
        np.testing.assert_array_equal(model, expected * np.float32(2.0))

    def test_stored_normals(self):
        normals = self.archive.normals(self.archive.map_record(1, 'photometric'))
        pose = synthetic.poses()[1]
        expected = pose.R @ synthetic.plane_normal()
        self.assertLess(np.abs(normals - expected).max(), 1e-4)

    def test_input_image(self):
        record = self.archive.input_record(0)
        rgba = self.archive.input_rgba(record)
        self.assertEqual(rgba.shape, (48, 64, 4))
        self.assertEqual(record.grid, 'original-distorted')
        self.assertEqual(record.camera.fx, 50.0)


def bad_member(kind: str, base: Path) -> tarfile.TarInfo:
    """A member that would write `base/escape.txt` or plant a link if it were extracted into a
    directory under `base`."""
    names = {'parent': '../escape.txt', 'absolute': str(base / 'escape.txt'), 'nested': 'view-0/../../escape.txt',
             'drive': 'C:escape.txt', 'backslash': 'view-0\\..\\..\\escape.txt'}
    if kind in names:
        info = tarfile.TarInfo(names[kind])
        info.size = 4
        return info
    info = tarfile.TarInfo('view-0/link')
    info.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE, 'fifo': tarfile.FIFOTYPE}[kind]
    info.linkname = '../escape.txt'
    return info


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_extracts_parts_and_verifies_them_against_the_manifest(self):
        manifest_path = synthetic.write_parts(self.base / 'source', self.base / 'parts')
        manifest = load_manifest(manifest_path)
        target = self.base / 'target'
        summary = extract_parts([manifest_path.parent / name for name in manifest.archives], target)
        self.assertEqual(summary.parts, 2)
        self.assertEqual(verify_extraction(manifest, target), [])
        for path in (self.base / 'source').rglob('*'):
            if path.is_file():
                self.assertEqual((target / path.relative_to(self.base / 'source')).read_bytes(), path.read_bytes())
        (target / manifest.maps[0].arrays['depth'][0].path).unlink()
        self.assertEqual(len(verify_extraction(manifest, target)), 1)

    def test_verification_compares_each_map_json_with_the_manifest(self):
        manifest_path = synthetic.write_parts(self.base / 'source', self.base / 'parts')
        manifest = load_manifest(manifest_path)
        target = self.base / 'target'
        extract_parts([manifest_path.parent / name for name in manifest.archives], target)
        # The manifest of another run of the same views: other poses and another scale, the same
        # segment sizes. Every map.json disagrees with it.
        other = dataclasses.replace(manifest, maps=tuple(
            dataclasses.replace(record, pose=Pose(record.pose.R[::-1].copy(), record.pose.t + 5.0),
                                depth_scale_to_model=record.depth_scale_to_model * 3)
            for record in manifest.maps))
        problems = verify_extraction(other, target)
        self.assertEqual(len(problems), len(manifest.maps))
        self.assertTrue(all(p.endswith('map.json differs from the manifest in pose, depthScaleToModel')
                            for p in problems), problems)
        # A map.json with another view id, camera and segment list; a map.json that is no record.
        first, second = manifest.maps[0], manifest.maps[1]
        path = target / f'view-{first.view_id}' / first.stage / 'map.json'
        changed = {**first.metadata, 'viewId': 7, 'camera': {**first.metadata['camera'], 'fx': 51.0},
                   'depth': first.metadata['depth'][:1]}
        path.write_text(json.dumps(changed), encoding='utf-8')
        (target / f'view-{second.view_id}' / second.stage / 'map.json').write_text('{"viewId": 0}', encoding='utf-8')
        problems = verify_extraction(manifest, target)
        self.assertEqual(problems, [
            f'view-{first.view_id}/{first.stage}/map.json differs from the manifest in viewId, camera, segments',
            f"view-{second.view_id}/{second.stage}/map.json is not a map record: KeyError('stage')",
        ])

    def test_refuses_unsafe_members(self):
        for kind in ('parent', 'absolute', 'nested', 'drive', 'backslash', 'symlink', 'hardlink', 'fifo'):
            with self.subTest(kind=kind):
                part = self.base / f'{kind}.tar.gz'
                with tarfile.open(part, 'w:gz') as tar:
                    info = bad_member(kind, self.base)
                    tar.addfile(info, io.BytesIO(b'evil') if info.isfile() else None)
                target = self.base / f'target-{kind}'
                with self.assertRaises(UnsafeMemberError):
                    extract_parts([part], target)
                self.assertFalse((self.base / 'escape.txt').exists())
                self.assertEqual([p for p in target.rglob('*') if not p.is_dir()], [])

    def test_refuses_to_write_through_a_link_in_the_target(self):
        target = self.base / 'target'
        (self.base / 'outside').mkdir()
        target.mkdir()
        try:
            os.symlink(self.base / 'outside', target / 'view-0', target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            # Windows makes a link only with Developer Mode or the privilege (WinError 1314).
            self.skipTest(f'cannot make a directory link here: {error}')
        part = self.base / 'part.tar.gz'
        synthetic.tar_gz(part, {'view-0/input.json': b'{}'})
        with self.assertRaises(UnsafeMemberError):
            extract_parts([part], target)
        self.assertEqual(list((self.base / 'outside').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
