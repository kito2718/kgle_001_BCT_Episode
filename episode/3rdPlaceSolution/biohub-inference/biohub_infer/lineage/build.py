"""Downstream: upstream caches -> calibrated predictions -> pooled LP per test block -> gap closing / L6 ->
affine coordinate refinement -> submission CSV.

The calibration predict and the LP run as subprocesses of `biohub_infer.lineage.div846_lineage`, so the
single-thread BLAS environment that module sets applies to them only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from biohub_infer import config as C
from biohub_infer.lineage.affine_refinement import refine_final
from biohub_infer.submission import export_submission


def pool_ids(sorted_ids, splits):
    """Contiguous pools of the sorted ids, a new one starting at every split position."""
    bounds = [0, *[s for s in splits if s < len(sorted_ids)], len(sorted_ids)]
    return [sorted_ids[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def _lineage(*args):
    subprocess.run([sys.executable, '-m', 'biohub_infer.lineage.div846_lineage', *map(str, args)], check=True)


def build_submission(upstream, calibrator, test_dir, work_dir, output, jobs=4):
    """upstream: the dir with graphs / confidence / div_scores / division_graphs; returns the CSV row count."""
    upstream, work_dir, output, test_dir = map(Path, (upstream, work_dir, output, test_dir))
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f'{work_dir}: use an empty work directory')
    work_dir.mkdir(parents=True, exist_ok=True)
    graphs, div_scores = upstream / 'graphs', upstream / 'div_scores'
    _lineage('predict', '--graphs', graphs, '--confidence', upstream / 'confidence', '--div-scores', div_scores,
             '--model', calibrator, '--out', work_dir / 'calibration', '--jobs', jobs)
    predictions = work_dir / 'calibration/predictions'
    pools = pool_ids(sorted(p.stem for p in predictions.glob('*.npz')), C.POOL_SPLITS)
    final = work_dir / 'optimization/final'
    for k, members in enumerate(pools):
        pool_predictions, out = predictions, work_dir / 'optimization'
        if len(pools) > 1:
            pool_predictions = work_dir / f'calibration/predictions_pool{k}'
            pool_predictions.mkdir()
            for sid in members:
                (pool_predictions / f'{sid}.npz').symlink_to((predictions / f'{sid}.npz').resolve())
            out = work_dir / f'optimization_pool{k}'
        _lineage('optimize', '--graphs', graphs, '--predictions', pool_predictions, '--div-scores', div_scores,
                 '--out', out, *C.LP_ARGS, '--jobs', jobs)
        if len(pools) > 1:
            final.mkdir(parents=True, exist_ok=True)
            for sid in members:
                (final / f'{sid}.npz').symlink_to((out / 'final' / f'{sid}.npz').resolve())
    rows = export_submission(final, work_dir / 'baseline.csv')

    started = time.perf_counter()
    stats = []
    for path in sorted(final.glob('*.npz')):
        shape = json.loads((test_dir / f'{path.stem}.zarr/0/zarr.json').read_text())['shape'][1:]
        row = refine_final(predictions / path.name, path, shape, work_dir / 'affine/final' / path.name, gap_affine=True)
        stats.append(row)
        print(f"[affine] {path.stem}: moved {row['moved']}/{row['nodes']} real, "
              f"{row['gap']['moved']}/{row['gap']['synthetic']} gap", flush=True)
    refined_rows = export_submission(work_dir / 'affine/final', output)
    if refined_rows != rows:
        raise RuntimeError(f'affine refinement changed the row count: {rows} -> {refined_rows}')
    (work_dir / 'affine/report.json').write_text(json.dumps(dict(
        method='q-weighted affine strength 2 + affine gap bridge', samples=len(stats),
        wall_seconds=time.perf_counter() - started, per_sample=stats), indent=2) + '\n')
    return rows


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--calibrator', type=Path, required=True, help=f'the {C.CALIBRATOR} of the weights dataset')
    p.add_argument('--test-dir', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args(argv)
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    print(f'{build_submission(args.upstream, args.calibrator, args.test_dir, args.work_dir, args.out, args.jobs)} rows '
          f'-> {args.out}')


if __name__ == '__main__':
    main()
