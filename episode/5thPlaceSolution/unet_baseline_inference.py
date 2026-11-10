# %% [markdown]
# # UNet + Transformer baseline — inference & submission (Local & Kaggle)
#
# Runs the trained baseline on the test volumes and writes `submission_baseline.csv` (or `submission.csv` on Kaggle).
#
# Reference:
# - https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission
# - https://github.com/royerlab/kaggle-cell-tracking-competition

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_KAGGLE = Path("/kaggle/working").exists()

COMP_DIR = (
    "/kaggle/input/competitions/biohub-cell-tracking-during-development"
    if IS_KAGGLE
    else str(ROOT / "input")
)
TEST_DIR = os.path.join(COMP_DIR, "test")

ARTIFACTS_CANDIDATES = [
    "/kaggle/input/datasets/thibautgoldsborough/cellmot-baseline-artifacts/cellmot-baseline-artifacts",
    str(ROOT / "cellmot-baseline-artifacts" / "cellmot-baseline-artifacts"),
    str(ROOT / "cellmot-baseline-artifacts"),
]
ARTIFACTS = next((p for p in ARTIFACTS_CANDIDATES if os.path.isdir(os.path.join(p, "repo"))), ARTIFACTS_CANDIDATES[1])

REPO_DIR = "/kaggle/working/repo" if IS_KAGGLE else str(ROOT / "working" / "baseline_repo")
OUT_CSV = "submission.csv" if IS_KAGGLE else str(ROOT / "working" / "submission_baseline.csv")
METHOD = "unet_transformer"

# --- Test-time options (tune these) ---------------------------------------
WEIGHTS = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
DET_THRESHOLD = 0.99
UNET_BATCH_SIZE = 4          # frame-pairs per UNet forward; lower if you OOM
SLICE = os.environ.get("BASELINE_SLICE", "")  # e.g. ":1" to predict only the first video; "" = all

USE_ILP = True
ILP_EDGE_WEIGHT = -1.0           # weight on edge probability
ILP_APPEARANCE_WEIGHT = 0.1      # cost of a track appearing
ILP_DISAPPEARANCE_WEIGHT = 0.1   # cost of a track disappearing
ILP_DIVISION_WEIGHT = 1.0        # cost of a division; lower (~0.2) allows more splits
# --------------------------------------------------------------------------


def main():
    try:
        import tracksdata  # noqa: F401
        import zarr  # noqa: F401
        import pyscipopt  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", f"{ARTIFACTS}/wheels",
             "tracksdata", "zarr>=3.0.10", "pyscipopt"],
            check=True,
        )

    os.makedirs(REPO_DIR, exist_ok=True)
    shutil.copytree(f"{ARTIFACTS}/repo", REPO_DIR, dirs_exist_ok=True)
    shutil.copytree(f"{ARTIFACTS}/weights", f"{REPO_DIR}/weights", dirs_exist_ok=True)
    sys.path.insert(0, f"{REPO_DIR}/src")

    print("Weights:", os.listdir(f"{REPO_DIR}/weights/{METHOD}/split_0"))

    test_stems = sorted(f[:-5] for f in os.listdir(TEST_DIR) if f.endswith(".zarr"))
    print(f"{len(test_stems)} test videos: {test_stems}")

    with open(f"{REPO_DIR}/kaggle_test_splits.json", "w") as f:
        json.dump([{"split": 0, "train": [], "test": test_stems}], f)

    cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", TEST_DIR, "--splits", "kaggle_test_splits.json", "--split", "0",
        "--weights", WEIGHTS, "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        cmd.append("--use-ilp")
    if SLICE:
        cmd += ["--slice", SLICE]

    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)

    import pandas as pd
    import tracksdata as td

    geffs = sorted(Path(REPO_DIR, "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
    print(f"{len(geffs)} prediction graphs")

    rows = []
    for g in geffs:
        name = g.stem
        graph = td.graph.IndexedRXGraph.from_geff(g)
        graph = graph[0] if isinstance(graph, tuple) else graph
        for r in graph.node_attrs().iter_rows(named=True):
            rows.append({
                "dataset": name, "row_type": "node", "node_id": int(r["node_id"]),
                "t": int(r["t"]), "z": int(round(r["z"])), "y": int(round(r["y"])),
                "x": int(round(r["x"])), "source_id": -1, "target_id": -1,
            })
        for r in graph.edge_attrs().iter_rows(named=True):
            rows.append({
                "dataset": name, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
            })

    submission = pd.DataFrame(
        rows,
        columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
    )
    submission.index.name = "id"
    os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
    submission.to_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV} with {len(submission)} rows")


if __name__ == "__main__":
    main()
