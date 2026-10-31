"""Biohub cell tracking: test movies -> lineage-graph submission CSV (the stack of submission 060).

    python -m biohub_infer --test-dir <dir of <sid>.zarr> --weights <weights dataset> --out submission.csv

Upstream inference runs one worker process per visible GPU (movies split round-robin over the sorted ids);
the downstream (calibration, pooled LP, gap closing, affine refinement) runs on the CPU afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from biohub_infer import config as C


def check_md5(paths):
    import hashlib
    want = dict(line.split()[::-1] for line in paths[C.MD5SUMS].read_text().splitlines() if line.strip())
    for name, path in paths.items():
        if name == C.MD5SUMS:
            continue
        digest = hashlib.md5()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                digest.update(block)
        if want.get(name) != digest.hexdigest():
            raise ValueError(f'{path}: md5 {digest.hexdigest()} != {C.MD5SUMS} {want.get(name)}')


def preflight(weights):
    """Every weight file present exactly once, md5 as listed, the calibrator bundle intact -> {name: path}."""
    from biohub_infer.infer.clip import resolve_weights
    from biohub_infer.lineage.div846_lineage import check_bundle
    paths = resolve_weights(weights)
    check_md5(paths)
    if len({paths[name].parent for name in ('div797.json', 'div846.json', C.CALIBRATOR)}) != 1:
        raise ValueError(f'the calibrator files must sit beside {C.CALIBRATOR}')
    from biohub_infer.core.division_phase_wiring import PHASE_DIV_FEATURES, PHASE_EDGE_FEATURES
    cal = json.loads(paths[C.CALIBRATOR].read_text())
    if cal['edge_features'] != PHASE_EDGE_FEATURES or cal['division_features'] != PHASE_DIV_FEATURES \
            or cal.get('fold') != -1 or cal.get('candidate_radius_um') != 12. or cal.get('division_phase_expand') is not False:
        raise ValueError(f'{paths[C.CALIBRATOR]}: not the fit-full radius-12 calibrator of this feature set')
    check_bundle(paths[C.CALIBRATOR].parent)
    return paths


def run_workers(args, ids, upstream):
    import torch
    shards = min(args.shards or max(1, torch.cuda.device_count()), len(ids))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if shards > len(devices):
        raise SystemExit(f'{shards} shards but {len(devices)} visible GPUs')
    procs = []
    try:
        for i in range(shards):
            cmd = [sys.executable, '-m', 'biohub_infer', '--test-dir', str(args.test_dir), '--weights', str(args.weights),
                   '--work-dir', str(args.work_dir), '--out', str(args.out), '--shards', str(shards), '--shard', str(i)]
            procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[i])))
        codes = [proc.wait() for proc in procs]
        if any(codes):
            raise RuntimeError(f'inference worker exit codes: {codes}')
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    return shards


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-dir', type=Path, required=True, help='directory of the <sid>.zarr test movies')
    p.add_argument('--weights', type=Path, required=True, help='weights dataset root (files resolved by name)')
    p.add_argument('--out', type=Path, default=Path('submission.csv'))
    p.add_argument('--work-dir', type=Path, default=Path('work'), help='upstream caches and LP files (must be empty)')
    p.add_argument('--jobs', type=int, default=4, help='CPU workers of the downstream')
    p.add_argument('--shards', type=int, default=None, help='GPU workers; default every visible GPU')
    p.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    ids = sorted(f.stem for f in args.test_dir.glob('*.zarr'))
    if not ids:
        raise SystemExit(f'no .zarr movies in {args.test_dir}')
    upstream = args.work_dir / 'upstream'
    if args.shard is not None:                       # worker
        from biohub_infer.infer.clip import run
        run(ids[args.shard::args.shards], args.weights, args.test_dir, upstream, tag=f'[infer shard {args.shard}]')
        return

    started = time.perf_counter()
    paths = preflight(args.weights)
    print(f'[biohub_infer] {len(ids)} movies; detectors {C.DETECTORS} (TTA {C.DET_TTA}); div3f TTA {C.DIV_TTA}; '
          f'calibrator {paths[C.CALIBRATOR]}', flush=True)
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise SystemExit(f'{args.work_dir}: use an empty work directory')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shards = run_workers(args, ids, upstream)
    from biohub_infer.infer.clip import SUBDIRS
    for directory in SUBDIRS:
        if {f.stem for f in (upstream / directory).glob('*.npz')} != set(ids):
            raise RuntimeError(f'incomplete upstream caches in {upstream / directory}')
    (upstream / 'manifest.json').write_text(json.dumps(dict(
        datasets=ids, weights={k: str(v) for k, v in paths.items()}), indent=2) + '\n')
    inferred = time.perf_counter()

    from biohub_infer.core.dataset import CellStore
    from biohub_infer.lineage.build import build_submission
    from biohub_infer.submission import validate_submission
    import numpy as np
    rows = build_submission(upstream, paths[C.CALIBRATOR], args.test_dir, args.work_dir / 'lineage', args.out, args.jobs)
    shapes = CellStore(args.test_dir, ids, with_gt=False, downsample=np.ones(3, int)).shape
    stats = validate_submission(args.out, shapes, ids)
    if rows != stats['rows']:
        raise RuntimeError(f'written {rows} rows, read back {stats["rows"]}')
    print(f'[biohub_infer] {args.out}: {stats["rows"]} rows ({stats["nodes"]} nodes, {stats["edges"]} edges), '
          f'{len(ids)} datasets on {shards} GPU(s); upstream {inferred - started:.0f}s, '
          f'downstream {time.perf_counter() - inferred:.0f}s', flush=True)


if __name__ == '__main__':
    main()
