#!/usr/bin/env python3
"""Biohub 7th Place Solution - Local Environment Verification Script.

Verifies Python/CUDA environment, required packages, test videos, SHA-256 hashes of
all 7th-place checkpoints and division models (3D ResEncM nnU-Net pretrained & scratch,
SlidingT3Residual BiGRU division image classifiers, ExtraTrees & 5-seed LightGBM
daughter-pair rankers, 83-input SWA MLP), OR-Tools min-cost flow + SCIP ILP solver,
and postprocessing hidden-rerun safety regression checks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
WORK = ROOT / "working" / "m4000_pretrain_two_model_full_paired_v001_0923"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 76)
    print("Biohub 7th Place Solution - Local Environment Verification")
    print("=" * 76)

    all_ok = True

    # 1. Python & GPU
    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"[1] Python: {sys.version.split()[0]}")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA Available: {cuda_ok}")
    if cuda_ok:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Capability: {torch.cuda.get_device_capability(0)}")
        print(
            f"    Native bf16 Supported: {torch.cuda.is_bf16_supported(including_emulation=False)}"
        )
    else:
        print("    ERROR: CUDA GPU is required for 7th place 3D nnU-Net inference")
        all_ok = False

    # 2. Required Packages
    print("\n[2] Checking Required Packages:")
    pkgs = [
        ("torch", lambda: __import__("torch")),
        ("zarr", lambda: __import__("zarr")),
        ("polars", lambda: __import__("polars")),
        ("tracksdata", lambda: __import__("tracksdata")),
        ("geff", lambda: __import__("geff")),
        ("pyscipopt", lambda: __import__("pyscipopt")),
        ("ilpy", lambda: __import__("ilpy")),
        ("ortools", lambda: __import__("ortools.graph.python.min_cost_flow")),
        ("lightgbm", lambda: __import__("lightgbm")),
        ("timm", lambda: __import__("timm")),
        ("SimpleITK", lambda: __import__("SimpleITK")),
        ("nibabel", lambda: __import__("nibabel")),
        ("cc3d", lambda: __import__("cc3d")),
        ("blosc2", lambda: __import__("blosc2")),
        ("scipy", lambda: __import__("scipy")),
    ]
    for name, importer in pkgs:
        try:
            importer()
            print(f"    - {name}: OK")
        except Exception as e:
            print(f"    - {name}: FAILED ({e})")
            all_ok = False

    # 3. Competition Input & Test Videos
    test_dir = ROOT / "input" / "test"
    sample_sub = ROOT / "input" / "sample_submission.csv"
    movies = sorted(p.name for p in test_dir.glob("*.zarr")) if test_dir.is_dir() else []
    print(f"\n[3] Competition Input ({len(movies)} test movies in {test_dir}):")
    for m in movies:
        print(f"    - {m}")
    if len(movies) != 4 or not sample_sub.is_file():
        print(f"    ERROR: Expected 4 test movies and sample_submission.csv")
        all_ok = False
    else:
        print(f"    - sample_submission.csv: OK ({sample_sub.stat().st_size} bytes)")

    # 4. Materialize Runtime if Needed & Verify Model Assets (SHA-256)
    print("\n[4] Materializing Runtime & Verifying 7th Place Model Checkpoints (SHA-256):")
    if not (WORK / "portable_division" / "runtime" / "choice_worker.py").is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "biohub-7th-place-inference.py"), "--setup-only"],
            check=True,
        )

    artifact_dir = DATASETS / "biohub-motion-nnunet-ilp-artifacts-1500-v001"
    artifact_manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text())
    scratch_ckpt = artifact_dir / artifact_manifest["model_relative_dir"] / "fold_0" / "checkpoint_final.pth"
    pretrain_ckpt = DATASETS / "biohub-synthetic-native1500-0921" / "checkpoint_final.pth"
    addon_dir = DATASETS / "biohub-division-runtime-20260908-v001"
    swa_dir = DATASETS / "biohub-division-swa-image5-assets-0926"

    expected_hashes = [
        (
            "3D ResEncM Detector+Motion (Pretrained 1000ep + Real 1500ep)",
            pretrain_ckpt,
            "8e86ed5aaf05f60aee755f909c8236f3c8ab5eca7e29d246da1e7e6ba3be7afb",
        ),
        (
            "3D ResEncM Detector+Motion (Scratch Real 1500ep)",
            scratch_ckpt,
            "f23d91ff09ad30f6980159669933a3d2b36873940a4a75dbc45d6290f86ec421",
        ),
        (
            "Division 3D CNN + BiGRU (seed20260817)",
            addon_dir / "parent_models/seed20260817_hard10_heatmap/arms/sliding_t3_bigru/checkpoint_best.pt",
            "28bf1bd594ffac5ed4c535f4d446eaa746e24002f8298bb38d9e7c5bf516d98f",
        ),
        (
            "Division 3D CNN + BiGRU (seed20260818)",
            addon_dir / "parent_models/seed20260818_hard10_heatmap/arms/sliding_t3_bigru/checkpoint_best.pt",
            "0607b9867fcd6f1c5e52fde5af3527587f0daaac6501349d86e56daf9b5c72dd",
        ),
        (
            "Daughter-Pair ExtraTrees (73 features)",
            addon_dir / "models/extra_trees_fold2.npz",
            "a8bd40d0a98afc87c4b5a4a476e5f9aebc199f02c7ee9e981114a9db3f18ad2c",
        ),
        (
            "Division Event SWA MLP (nn_h.pt, 83 inputs -> 32 SiLU -> 1)",
            swa_dir / "model_swa_compatible.pt",
            "c54012e127ac0e2f47d648d1f54ffefeef992a651bcca69d3d70b2880d9987ae",
        ),
        (
            "Daughter-Image LightGBM Seed 1 (91 features)",
            swa_dir / "image_pair_fold2_seed20260925.txt",
            "148ba3c664fc7e5f2456091e088d6b06a5cd1d0699a5985fa0795290f60f227a",
        ),
        (
            "Daughter-Image LightGBM Seed 5 (91 features)",
            swa_dir / "image_pair_fold2_seed20264961.txt",
            "fbe6e95fab1a37155d8d6661ba1a469a2996e25a0080349f41ed16e86b44ef89",
        ),
    ]
    for label, path, expected in expected_hashes:
        if not path.is_file():
            print(f"    - {label}: MISSING ({path})")
            all_ok = False
            continue
        actual = sha256_file(path)
        if actual == expected:
            print(f"    - {label}: SHA-256 OK ({actual[:12]}...)")
        else:
            print(f"    - {label}: SHA-256 MISMATCH ({actual} != {expected})")
            all_ok = False

    # 5. Component Smoke Tests (Division Scorer, Forest, LightGBM, MLP, Min-Cost Flow + SCIP, Postprocess)
    print("\n[5] Component Smoke Tests (Division Models, Flow+SCIP ILP, Postprocess):")
    try:
        import os
        import numpy as np
        import lightgbm as lgb

        portable = WORK / "portable_division"
        sys.path[:0] = [
            str(portable / "runtime"),
            str(portable / "runtime" / "legacy"),
            str(portable / "runtime" / "biohub_solver_runtime"),
            str(artifact_dir / "runtime" / "nnUNet"),
            str(artifact_dir / "runtime" / "vendor"),
            str(artifact_dir / "runtime" / "tools"),
        ]
        os.environ["BIOHUB_ROOT"] = str(portable)

        # a) ParentScorer (2x SlidingT3Residual 3D CNN + BiGRU)
        from parent import ParentScorer

        scorer = ParentScorer("cuda" if cuda_ok else "cpu")
        n_params = sum(p.numel() for p in scorer.models[0].parameters())
        print(
            f"    - ParentScorer (2x SlidingT3Residual BiGRU, {n_params / 1e6:.2f}M params each): OK"
        )

        # b) PortableForest (ExtraTrees) + 5-seed LightGBM + 83-input MLP
        from portable_forest import PortableForest
        from choice_net import Net

        forest = PortableForest(portable / "models" / "extra_trees_fold2.npz")
        probs = forest.predict_proba(np.zeros((4, 73), dtype=np.float32))
        assert probs.shape == (4, 2)
        lgb_model = lgb.Booster(
            model_file=str(
                portable / "models" / "daughter_image_lgb5" / "image_pair_fold2_seed20260925.txt"
            )
        )
        lgb_pred = lgb_model.predict(np.zeros((4, 91), dtype=np.float32))
        assert lgb_pred.shape == (4,)
        cp = torch.load(portable / "models" / "nn_h.pt", map_location="cpu", weights_only=False)
        mlp = Net(len(cp["mean"]))
        mlp.load_state_dict(cp["state"])
        mlp.eval()
        with torch.no_grad():
            mlp_out = mlp(torch.zeros((4, 83), dtype=torch.float32))
        assert mlp_out.shape == (4, 1)
        print("    - ExtraTrees (73-dim), LightGBM (91-dim), SWA MLP (83-dim): OK")

        # c) OR-Tools Min-Cost Circulation + SCIP ILP Solver
        from flow_solver import optimize

        ns, es, flow_rec = optimize(
            [1, 2], [(10, 1, 2, 0.9)], timeout=10, gap=0, division_costs=[0.3, 0.3]
        )
        assert ns == {1, 2} and es == {10}
        print(f"    - OR-Tools Min-Cost Flow + SCIP Solver (status={flow_rec['status']}): OK")

        # d) Postprocess Hidden-Safety Regression
        post_test = DATASETS / "biohub-motion-postprocess-v001" / "test_biohub_postprocess_hidden_safety.py"
        res = subprocess.run(
            [sys.executable, str(post_test)],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path[:6])},
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"    - {res.stdout.strip()}: OK")
    except Exception as e:
        print(f"    - Component test FAILED: {e}")
        all_ok = False

    print("\n" + "=" * 76)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED! Ready for 7th place inference.")
    else:
        print("RESULT: SOME CHECKS FAILED. Please review the output above.")
    print("=" * 76)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
