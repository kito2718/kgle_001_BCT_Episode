#!/usr/bin/env python3
"""Biohub cell tracking: inference run script (Python port of biohub-inference-run.ipynb).

Runs the submission-060 stack on the test movies.
Supports both Local environment and Kaggle execution.
"""
from collections import Counter
import csv
import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# -----------------------------------------------------------------------------
# 1. Path Resolution (Auto-detects Local and Kaggle layouts)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

INPUT_CANDIDATES = [
    Path("/kaggle/input"),
    ROOT,
    ROOT / "input",
    ROOT.parent / "input",
    Path("c:/work/aaa/s9/input"),
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), ROOT)


def find_input(name, kind="datasets", owner="ren4yu"):
    # Direct check under workspace root
    for direct in (ROOT / name, ROOT / "ren4yu" / name, Path.cwd() / name):
        if direct.exists():
            return direct
    for root in [INPUT, *INPUT_CANDIDATES]:
        if not root.exists():
            continue
        for cand in (
            root / kind / owner / name,
            root / kind / name,
            root / owner / name,
            root / name,
        ):
            if cand.exists():
                return cand
        if (root / "test").exists() and name == "biohub-cell-tracking-during-development":
            return root
    raise SystemExit(f"{name} not found in candidate input paths")


def main():
    print("=" * 70)
    print("Biohub 3rd Place Solution - Inference Runner")
    print("=" * 70)

    CODE = find_input("biohub-inference")
    WEIGHTS = find_input("biohub-weights")
    COMP = find_input("biohub-cell-tracking-during-development", kind="competitions")
    TEST_DIR = COMP / "test" if (COMP / "test").exists() else ROOT / "input" / "test"
    WORK = Path("/kaggle/working/work") if Path("/kaggle/working").exists() else ROOT / "work"
    SUBMISSION = (
        Path("/kaggle/working/submission.csv")
        if Path("/kaggle/working").exists()
        else ROOT / "submission.csv"
    )

    print(f"Code path    : {CODE}")
    print(f"Weights path : {WEIGHTS} ({len(list(WEIGHTS.iterdir()))} files)")
    print(f"Test movies  : {TEST_DIR} ({len(list(TEST_DIR.glob('*.zarr')))} movies)")
    print(f"Work dir     : {WORK}")
    print(f"Output CSV   : {SUBMISSION}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 2. Dependency Check / Offline Install (for Kaggle environment)
    # -------------------------------------------------------------------------
    need = [
        package
        for module, package in [
            ("zarr", "zarr==3.3.0"),
            ("numcodecs", "numcodecs==0.16.5"),
            ("donfig", "donfig==0.8.1.post1"),
            ("monai", "monai==1.5.2"),
            ("narwhals", "narwhals==2.25.0"),
            ("tensorrt", "tensorrt-cu12==10.9.0.34"),
            ("tensorrt", "tensorrt-cu12-bindings==10.9.0.34"),
            ("tensorrt", "tensorrt-cu12-libs==10.9.0.34"),
        ]
        if importlib.util.find_spec(module) is None
    ]
    for package, pin in [("lightgbm", "4.7.0"), ("xgboost", "3.2.0")]:
        if importlib.util.find_spec(package) is None or version(package) != pin:
            need.append(f"{package}=={pin}")
    if (
        importlib.util.find_spec("zarr") is not None
        and int(version("zarr").split(".")[0]) < 3
    ):
        need += ["zarr==3.3.0", "numcodecs==0.16.5", "donfig==0.8.1.post1"]

    if need:
        wheels_dir = CODE / "wheels"
        if wheels_dir.exists():
            print(f"Installing missing packages from {wheels_dir}: {need}")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    f"--find-links={wheels_dir}",
                    *need,
                ],
                check=True,
            )
        else:
            print(f"[Warning] Missing packages {need}, but wheels dir not found; proceeding with current environment.")

    import numpy
    import scipy
    import torch
    import tensorrt
    import xgboost
    import lightgbm
    import zarr

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"
    print(
        f"numpy {numpy.__version__} | scipy {scipy.__version__} | "
        f"torch {torch.__version__} | tensorrt {tensorrt.__version__} | "
        f"xgboost {xgboost.__version__} | lightgbm {lightgbm.__version__} | "
        f"zarr {zarr.__version__} | GPUs: {gpu_count} ({gpu_name})"
    )
    print("-" * 70)

    # -------------------------------------------------------------------------
    # 3. Execute Inference
    # -------------------------------------------------------------------------
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE="1")

    # Set appropriate worker count (at least 1)
    jobs = max(1, gpu_count * 2) if gpu_count > 0 else 1

    cmd = [
        sys.executable,
        "-m",
        "biohub_infer",
        "--test-dir",
        str(TEST_DIR),
        "--weights",
        str(WEIGHTS),
        "--work-dir",
        str(WORK),
        "--out",
        str(SUBMISSION),
        "--jobs",
        str(jobs),
    ]

    log_path = WORK.parent / "inference.log"
    print(f"Running inference CLI: {' '.join(cmd)}")
    print(f"Streaming logs to {log_path} ...")

    with log_path.open("w", encoding="utf-8") as stream:
        code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode

    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        print("\n--- Last 50 lines of inference.log ---")
        print("\n".join(log_text.splitlines()[-50:]))
        print("--------------------------------------")

    if code != 0:
        raise SystemExit(f"inference failed with exit code ({code})")

    elapsed = time.time() - t0
    print(f"Inference completed successfully in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # -------------------------------------------------------------------------
    # 4. Submission Validation
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("Validating generated submission file...")
    with SUBMISSION.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    kinds = Counter(r[2] for r in rows[1:])
    datasets = sorted({r[1] for r in rows[1:]})
    with_edges = {r[1] for r in rows[1:] if r[2] == "edge"}
    test_datasets = sorted(p.stem for p in TEST_DIR.glob("*.zarr"))

    print(f"Rows: {len(rows) - 1} {dict(kinds)} | datasets: {len(datasets)}")
    assert datasets == test_datasets, f"Dataset mismatch: {datasets} != {test_datasets}"
    assert set(datasets) == with_edges, f"Missing edges in datasets: {set(datasets) - with_edges}"
    print("All validation checks passed!")

    # -------------------------------------------------------------------------
    # 5. Cleanup
    # -------------------------------------------------------------------------
    KEEP_WORK = False
    if not KEEP_WORK and WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    print("Work directory cleaned.")
    print("=" * 70)


if __name__ == "__main__":
    main()
