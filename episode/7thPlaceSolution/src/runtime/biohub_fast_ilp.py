#!/usr/bin/env python3
"""Run any existing Biohub ILP entry with the reviewed automatic fast path.

Usage: python tools/biohub_fast_ilp.py path/to/original_ilp.py [original arguments]
Keep the experiment's existing Python environment and vendor PYTHONPATH.
"""
import os
from pathlib import Path
import runpy
import sys

runtime = Path(__file__).resolve().parent / 'biohub_solver_runtime'
sys.path.insert(0, str(runtime))
dependency_paths = [os.environ.get('BIOHUB_ORTOOLS_RUNTIME', ''),
    str(runtime.parent.parent / 'ortools_runtime'),
    '/biohub-gpu/shared_solver/ortools_runtime',
    '/home/ubuntu/biohub-gpu/shared_solver/ortools_runtime',
    '/workspace/biohub-gpu/shared_solver/ortools_runtime']
dependency_paths = [p + f'_py{sys.version_info.major}{sys.version_info.minor}' for p in dependency_paths if p] + dependency_paths
for dependency_path in dependency_paths:
    if dependency_path and (Path(dependency_path) / 'ortools').is_dir():
        sys.path.insert(0, dependency_path)
        break
from adapter import install

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(target.parent))
    install()
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')
else:
    # multiprocessing spawn imports the launcher as __mp_main__.
    install()
