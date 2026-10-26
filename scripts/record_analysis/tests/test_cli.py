"""Tests of the command line on the synthetic record: each subcommand that needs no pycolmap runs,
prints, and writes its JSON."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from record_analysis.cli import build_parser, main
from record_analysis.tests import synthetic


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.base = Path(cls.folder.name)
        cls.paths = synthetic.write_record(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_command(self, *argv: str) -> tuple[dict, str]:
        out = self.base / f'{argv[0]}.json'
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            status = main([*argv, '--json', str(out)])
        self.assertEqual(status, 0)
        document = json.loads(out.read_text())
        self.assertEqual(document['command'], argv[0])
        self.assertIn('numpy', document['versions'])
        return document['result'], printed.getvalue()

    def test_extract(self):
        result, text = self.run_command('extract', '--manifest', str(self.paths['manifest']),
                                        '--out', str(self.base / 'extract-target'))
        self.assertEqual(result['problems'], [])
        self.assertIn('every listed map', text)

    def test_frames(self):
        result, _ = self.run_command('frames', '--archive', str(self.paths['archive']), '--out',
                                     str(self.base / 'frames'), '--png-views', '1')
        self.assertEqual(len(result['jpeg']), 2)
        self.assertEqual([Path(p).name for p in result['png']], ['view-001.png'])

    def test_sparse(self):
        result, text = self.run_command('sparse', '--export', str(self.paths['export']))
        self.assertEqual((result['views'], result['landmarks']), (2, 130))
        self.assertIn('view direction change first to last: 4.0 deg', text)

    def test_seeds(self):
        result, text = self.run_command('seeds', '--archive', str(self.paths['archive']), '--export',
                                        str(self.paths['export']), '--views', '0,1')
        self.assertEqual([r['view_id'] for r in result], [0, 1])
        self.assertIn('within 2%', text)

    def test_normals(self):
        result, _ = self.run_command('normals', '--ply', str(self.paths['ply']), '--export', str(self.paths['export']))
        self.assertEqual(len(result['views']), 2)

    def test_depth_view(self):
        out = self.base / 'depth-view'
        result, _ = self.run_command('depth-view', '--archive', str(self.paths['archive']), '--view', '0',
                                     '--out', str(out))
        self.assertEqual([s['stage'] for s in result['stages']], ['photometric', 'geometric-filtered'])
        self.assertEqual(result['missing'], ['geometric-unfiltered', 'planar'])
        self.assertTrue((out / 'view-0-surface-normal-angle.png').is_file())

    def test_blocks(self):
        result, text = self.run_command('blocks', '--archive', str(self.paths['archive']), '--export',
                                        str(self.paths['export']), '--ply', str(self.paths['ply']), '--view', '0',
                                        '--grid', '2x2', '--pair', '0,0', '1,1', '--pair', '0,1', '1,0')
        self.assertEqual(len(result['blocks']), 4)
        self.assertEqual([p['first'] for p in result['pairs']], [[0, 0], [0, 1]])
        self.assertIn('angle between the planes of blocks (0, 0) and (1, 1)', text)

    def test_sweep_relief(self):
        result, text = self.run_command('sweep-relief', str(self.paths['export']), '--views', '0,5',
                                        '--image-size', '64x48')
        self.assertEqual(result[0]['unregistered'], [5])
        self.assertIn('not registered', text)

    def test_parser_defaults_are_the_diagnosis_values(self):
        args = build_parser().parse_args(['blocks', '--archive', 'a', '--export', 'e', '--ply', 'p'])
        self.assertEqual((args.view, args.grid, args.consistency, args.depth_stride), (120, (4, 3), 0.05, 7))
        args = build_parser().parse_args(['relief', '--archive', 'a', '--colmap', 'c'])
        self.assertEqual((args.views, args.sample, args.seed, args.min_dx, args.min_third),
                         ([0, 62, 104, 120, 137], 1500, 0, 0.05, 20))


if __name__ == '__main__':
    unittest.main()
