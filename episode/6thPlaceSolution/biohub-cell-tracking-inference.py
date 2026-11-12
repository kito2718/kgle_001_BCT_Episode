#!/usr/bin/env python3
"""Biohub Cell Tracking — 6th Place Solution Inference Script.

Predicts lineage graphs for every test movie and writes `submission.csv`.
Runs on Kaggle (2xT4, internet off) and locally; paths come from `SETTINGS.json`.

Usage:
    .venv\\Scripts\\python.exe biohub-cell-tracking-inference.py                  # default: variant="fin"
    .venv\\Scripts\\python.exe biohub-cell-tracking-inference.py --variant best   # best-validation checkpoints
    .venv\\Scripts\\python.exe biohub-cell-tracking-inference.py --movies 44b6_0113de3b
"""

from __future__ import annotations

import argparse
import os

# Set before torch initialises CUDA: avoids fragmentation OOMs on 8-16 GB GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import subprocess
import sys
import zipfile
from pathlib import Path

IS_KAGGLE = Path("/kaggle/working").exists()
ROOT = Path(__file__).resolve().parent


def setup_environment(settings_override: Path | None = None) -> tuple[Path, Path | None]:
    """Ensure `biohub_tracking` is importable and return `(CODE_DIR, SETTINGS_PATH)`."""
    if IS_KAGGLE:
        candidates = [
            Path("/kaggle/input/datasets/jamalsaeedi/biohub-cell-tracking-kaggle"),
            Path("/kaggle/input/biohub-cell-tracking-kaggle"),
        ]
        code_dir = next((c for c in candidates if c.exists()), candidates[0])
        src = code_dir / "src"
        if not src.is_dir():  # a directory upload can arrive as <name>.zip
            src = Path("/kaggle/working/biohub_artifacts/src")
            if not src.exists():
                src.mkdir(parents=True)
                with zipfile.ZipFile(code_dir / "src.zip") as bundle:
                    bundle.extractall(src)
        wheel = sorted(src.glob("biohub_tracking-*.whl"))[-1]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
        )
        settings_path = settings_override or (code_dir / "SETTINGS.json")
    else:
        repo_candidates = [
            ROOT / "biohub_cell_tracking_kaggle",
            ROOT,
        ]
        repo_root = next(
            (p for p in repo_candidates if (p / "pyproject.toml").exists()),
            ROOT / "biohub_cell_tracking_kaggle",
        )
        src_dir = repo_root / "src"
        if src_dir.is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        code_dir = repo_root
        if settings_override is not None:
            settings_path = settings_override
        elif os.environ.get("BIOHUB_SETTINGS"):
            settings_path = Path(os.environ["BIOHUB_SETTINGS"])
        elif (ROOT / "SETTINGS.json").is_file():
            settings_path = ROOT / "SETTINGS.json"
        else:
            settings_path = None
    return code_dir, settings_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Biohub 6th Place Solution Inference (Detect, Link, Solve, Repeat)"
    )
    parser.add_argument(
        "--variant",
        choices=("fin", "best"),
        default=os.environ.get("BIOHUB_VARIANT", "fin"),
        help='Model set variant: "fin" (selected final submission, default) or "best" (best-val checkpoints)',
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to SETTINGS.json (default: BIOHUB_SETTINGS or ./SETTINGS.json)",
    )
    parser.add_argument(
        "--movies",
        nargs="+",
        default=None,
        metavar="STEM",
        help="Predict only these movie stems (default: all *.zarr in TEST_DATA_DIR)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="At most this many GPU worker processes (default: one per GPU)",
    )
    parser.add_argument(
        "--session-hours",
        type=float,
        default=12.0,
        help="Time budget the deadline guard plans against (default: 12.0)",
    )
    args = parser.parse_args(argv)

    code_dir, settings_path = setup_environment(args.settings)
    print(f"IS_KAGGLE={IS_KAGGLE}  CODE_DIR={code_dir}", flush=True)

    # 1. Bootstrap & Package Versions
    from biohub_tracking.isotropic.kaggle import bootstrap, versions

    environment = bootstrap(code_dir)
    print(environment.describe(), flush=True)
    print(versions(), flush=True)

    # 2. Configuration & SHA-256 Verification
    from biohub_tracking.recipe import VARIANTS, shipped_config
    from biohub_tracking.settings import load_settings

    variant = args.variant
    settings = load_settings(settings_path)
    print(settings.describe(), flush=True)
    config = shipped_config(variant, list(settings.model_dirs))
    spec = VARIANTS[variant]
    d, s = config.detection, config.solver
    print(f"variant      {variant}  ({spec.kaggle_handle}, sha256 verified)", flush=True)
    for run_name, views, rescorer in zip(spec.runs, d.member_tta_views, spec.rescorers):
        print(f"  {run_name:32s} {views} TTA views   re-scorer {rescorer}", flush=True)
    print(f"detection    threshold {d.threshold}", flush=True)
    print(
        f"solver       disappearance {s.disappearance_weight}, appearance bias {s.appearance_bias}, "
        f"association temperature {s.association_temperature}",
        flush=True,
    )
    print(
        f"smoothing    line fit, window {config.smoothing.window}, weight {config.smoothing.weight}, "
        f"drift-compensated {config.smooth_drift_compensated}",
        flush=True,
    )

    # 3. Run Pipeline
    from biohub_tracking.cli import run, summarize

    submission_path, timing = run(
        variant,
        settings,
        movies=args.movies,
        num_shards=args.num_shards,
        session_hours=args.session_hours,
        config=config,
    )
    print(f"\nSubmission written and validated at: {submission_path}", flush=True)
    print(
        f"  {timing['nodes']} nodes, {timing['edges']} edges, {timing['divisions']} divisions",
        flush=True,
    )
    print(
        f"  predict {timing['predict_seconds'] / 60:.1f} min, total {timing['total_seconds'] / 60:.1f} min",
        flush=True,
    )

    # 4. Summary Table
    print(summarize(submission_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
