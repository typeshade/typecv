"""Offline analysis of the result files of a TypeCV lab run record (docs/chimney-rock-58532c69.md).

The inputs are the files a run sends with its record (docs/lab-run-records.md, Result files): the
reconstruction export `typecv-reconstruction.json`, the dense cloud `typecv-dense.ply`, the depth
archive `typecv-depths-cNNN.json` with its tar.gz parts, and optionally a pycolmap reference model.

Modules:
  archive    depth archive: manifest, maps, arrays, inputs, safe extraction of the parts
  ply        dense cloud PLY reader and writer
  scene      reconstruction export loader
  geometry   cameras, poses, projection, angles, robust planes, normal decode, block grid
  relief     depth-relief statistics and the relief of sparse models
  colmap     pycolmap reference run and model loading (pycolmap imported lazily)
  sparse, dense, reference, figure, frames   the analyses
  cli        the subcommands of scripts/analyze-record.py
"""
