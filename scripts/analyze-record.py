"""Offline analysis of a lab run record's result files (docs/chimney-rock-58532c69.md).

Reads the files that a run sends with its record (docs/lab-run-records.md, Result files): the
export typecv-reconstruction.json, the dense cloud typecv-dense.ply and the depth archive
typecv-depths-cNNN.json with its tar.gz parts, and a pycolmap reference model for the COLMAP
commands. Each command prints a table; --json OUT.json also writes the numbers, the arguments and
the tool versions. The package is scripts/record_analysis/; the defaults are the values of the
58532c69 diagnosis. FILES is a record's folder as scripts/fetch-record-files.mjs writes it
(--parts-dir names another folder of the tar.gz parts), FRAMES a frames output folder, COLMAP a
colmap-run output folder and SWEEP the --models folder of scripts/measure-focal-sweep.mjs.

  pip install -r scripts/requirements-record-analysis.txt
  python3 scripts/analyze-record.py extract --manifest FILES/typecv-depths-c000.json --out FILES/extracted
  python3 scripts/analyze-record.py frames --archive FILES/extracted --out FRAMES --png-views 0,120,137
  python3 scripts/analyze-record.py sparse --export FILES/typecv-reconstruction.json
  python3 scripts/analyze-record.py seeds --archive FILES/extracted --export FILES/typecv-reconstruction.json --views 0,62,104,120,137
  python3 scripts/analyze-record.py normals --ply FILES/typecv-dense.ply --export FILES/typecv-reconstruction.json
  python3 scripts/analyze-record.py depth-view --archive FILES/extracted --view 120 --out OUT
  python3 scripts/analyze-record.py blocks --archive FILES/extracted --export FILES/typecv-reconstruction.json --ply FILES/typecv-dense.ply --view 120 --pair 1,1 3,1
  python3 scripts/analyze-record.py colmap-run --images FRAMES/jpg --out COLMAP --camera-model SIMPLE_RADIAL --focal-prior 1.2
  python3 scripts/analyze-record.py colmap-compare --export FILES/typecv-reconstruction.json --colmap COLMAP/sparse/0 --views 120,137
  python3 scripts/analyze-record.py relief --archive FILES/extracted --colmap COLMAP/sparse/0 --views 0,62,104,120,137
  python3 scripts/analyze-record.py sweep-relief FILES/typecv-reconstruction.json SWEEP/ratio-0.97.model.json --views 120,137
  python3 scripts/analyze-record.py figure --export FILES/typecv-reconstruction.json --colmap COLMAP/sparse/0 --out figure.png

  python3 -m unittest discover -s scripts -p 'test_*.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_analysis.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
