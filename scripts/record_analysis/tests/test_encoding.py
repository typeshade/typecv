"""Tests that the package reads and writes its JSON and text files as UTF-8.

Without an explicit encoding, Python decodes with the locale's code page: cp1252 or cp949 on the
owner's Windows machine. An export holds U+00B7 (`verifyPair #11 · prepareImagePair`), and a run
record can hold Korean text, which those code pages misread or refuse.
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from record_analysis.archive import DepthArchive, load_manifest
from record_analysis.scene import load_reconstruction
from record_analysis.tests import synthetic

PACKAGE = Path(__file__).resolve().parents[1]
TEXT = 'verifyPair #11 · prepareImagePair; 초점거리 직접 넣기'


def _text_opens_without_encoding(path: Path) -> list[str]:
    """The calls in the file that open a text file without `encoding`: `read_text`, `write_text`
    and the built-in `open` in a text mode."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node, ast.Call) or any(k.arg == 'encoding' for k in node.keywords):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in ('read_text', 'write_text'):
            found.append(f'{path.name}:{node.lineno} {function.attr}')
        elif isinstance(function, ast.Name) and function.id == 'open':
            mode = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == 'mode'), None)
            binary = isinstance(mode, ast.Constant) and 'b' in str(mode.value)
            if not binary:
                found.append(f'{path.name}:{node.lineno} open')
    return found


class EncodingTest(unittest.TestCase):
    def test_every_text_open_names_its_encoding(self):
        sources = sorted(PACKAGE.rglob('*.py'))
        self.assertGreater(len(sources), 10)
        self.assertEqual([call for path in sources for call in _text_opens_without_encoding(path)], [])

    def test_reads_non_ascii_text_as_utf8(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = synthetic.write_record(Path(folder))
            export = json.loads(paths['export'].read_text(encoding='utf-8'))
            export['source'] = TEXT
            paths['export'].write_bytes(json.dumps(export, ensure_ascii=False).encode('utf-8'))
            manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
            manifest['error'] = TEXT
            paths['manifest'].write_bytes(json.dumps(manifest, ensure_ascii=False).encode('utf-8'))
            map_path = paths['archive'] / 'view-0' / 'photometric' / 'map.json'
            record = json.loads(map_path.read_text(encoding='utf-8'))
            map_path.write_bytes(json.dumps({**record, 'note': TEXT}, ensure_ascii=False).encode('utf-8'))
            self.assertEqual(load_reconstruction(paths['export']).source, TEXT)
            self.assertEqual(load_manifest(paths['manifest']).error, TEXT)
            self.assertEqual(DepthArchive(paths['archive']).map_record(0, 'photometric').metadata['note'], TEXT)


if __name__ == '__main__':
    unittest.main()
