# %% [code]
# Kaggle kernel entry for biohub-cell-tracking-during-development.
#
# The repo only knows local paths. This script points SubmitConfig at the Kaggle mounts,
# installs the offline dependencies, then calls the repo's own inference code —
# no inference logic is duplicated here.
#
# Mounted datasets:
#   hirotetsu/ctdd-final-v2-ilp1-rematch-adj-off-code  CTDD repo (branch final_v2, commit a83ca5a + ilp1 switch + group N
#                                                      parameters in SubmitConfig) .py + vendored tracking_cellmot
#   hirotetsu/ctdd-run53-run74-run79-ensemble-code   selected checkpoints/configs only (no code imports)
#   hirotetsu/biohub-tracking-support-pack  wheels/ (internet is off at submission time)

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent
WORK = str(Path(os.environ.get("BIOHUB_WORK", ROOT / "working" if not Path("/kaggle/working").exists() else "/kaggle/working")))
os.makedirs(WORK, exist_ok=True)

COMPETITION  = "biohub-cell-tracking-during-development"
CODE_DATASET = ("hirotetsu", "ctdd-final-v2-ilp1-rematch-adj-off-code")
SUPPORT_PACK = ("hirotetsu", "biohub-tracking-support-pack")   # offline wheels

# Only these 13. numpy/scipy/torch/pandas/dask ship with the Kaggle image at compatible
# versions — reinstalling them breaks the ABI.
OFFLINE_PACKAGES = [
    "tracksdata", "zarr==3.2.1", "numcodecs==0.15.1", "donfig==0.8.1.post1",
    "geff==1.2.0.1.1", "geff-spec==1.1.1", "pyscipopt==6.2.1", "ilpy==0.6.0",
    "rustworkx==0.18.0", "polars==1.42.0", "polars-runtime-32==1.42.0",
    "bidict==0.23.1", "imagecodecs==2026.6.26",
]
PURGE_ROOTS = ("tracksdata", "zarr", "numcodecs", "geff", "pyscipopt",
               "ilpy", "rustworkx", "polars", "imagecodecs")


def log(msg):
    print(msg, flush=True)


def mount(owner_slug):
    # Kaggle mounts an input at /kaggle/input/<slug>, but also nests it as
    # /kaggle/input/datasets/<owner>/<slug>; both layouts have been seen in one run.
    # Also support local workspace directories next to this script.
    owner, slug = owner_slug
    candidates = (
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{owner}/{slug}",
        str(ROOT / slug),
        str(ROOT / "input" / slug),
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"{slug} is not mounted (tried {candidates})")


def install_offline_deps():
    try:
        import zarr, geff, tracksdata  # noqa: F401
        return log("deps already present")
    except ImportError:
        pass

    wheels = os.path.join(mount(SUPPORT_PACK), "wheels")
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
         "--find-links", wheels, *OFFLINE_PACKAGES],
        text=True, capture_output=True)
    if r.returncode != 0:
        log(r.stdout[-4000:] + r.stderr[-4000:])
        raise RuntimeError("offline install failed")

    for name in [n for n in sys.modules
                 if any(n == root or n.startswith(root + ".") for root in PURGE_ROOTS)]:
        sys.modules.pop(name, None)
    log(f"offline install ok ({time.time() - T0:.0f}s)")


def competition_dir():
    candidates = (
        f"/kaggle/input/competitions/{COMPETITION}",
        f"/kaggle/input/{COMPETITION}",
        str(ROOT / "input"),
        str(ROOT.parent / "input"),
        r"c:\work\aaa\kgle_001_BCT_Episode\episode\input",
    )
    for path in candidates:
        if os.path.isdir(os.path.join(path, "test")):
            return path
    raise FileNotFoundError(f"{COMPETITION} data is not mounted")


def configure(cfg, code_dir):
    # Must run before importing dataset/utils/models — they do
    # `sys.path.append(BaseConfig.vendor_src)` at import time.
    cfg.vendor_src  = os.path.join(code_dir, "vendor", "repo", "src")
    cfg.weights_dir = os.path.join(code_dir, "models")
    cfg.parabolic_refinement = True
    cfg.data_dir    = competition_dir()

    cfg.train_dir = os.path.join(cfg.data_dir, "train")
    cfg.test_dir = os.path.join(cfg.data_dir, "test")
    cfg.split = "test"
    cfg.clip_ids = None

    # the code dataset is read-only, and Kaggle only collects /kaggle/working/submission.csv
    cfg.output_dir = WORK
    cfg.predict_dir = os.path.join(WORK, "predictions")
    cfg.results_dir = os.path.join(WORK, "results")
    cfg.submission_csv = os.path.join(WORK, "submission.csv")
    return cfg


CODE_DIR = mount(CODE_DATASET)
sys.path.insert(0, CODE_DIR)

from config import *  # noqa: E402,F403


def main():
    log(f"INPUT MOUNTS: {glob.glob('/kaggle/input/*') or [str(ROOT / 'input')]}")
    install_offline_deps()

    # Only the repo root goes on sys.path. tracking_cellmot is reached via BaseConfig.vendor_src,
    # which dataset/utils/models append themselves.
    cfg = configure(SubmitConfig, CODE_DIR)
    cfg.centroid = {**cfg.centroid, "enable": False}

    import hashlib
    expected = {"config.py": "4ced1fc851b506f5edc38684e78bea57a950c53a389a3983fd8d222a10ecf658", "dataset.py": "8ee403ec487bc4032b0e3910dab7cb8195a15b769eecb6e4b9e15af97f56341d", "inference.py": "276beffe71cf26b944797eccc07d2e142a1d9e20d58f8b19f541fac6795cbe8a", "inference_fast.py": "9c04c6710c65ea9a910ea38356404c04ad0fe5884c03773c39f8bf5eb584127c", "models.py": "8f4ed16253a80b5d1f2e2337c801592248ffdf09329a8e3d1730422918350998", "utils.py": "034ec5b6e0a1c6ab29df4fd8dc68a5c55cbaa99af455e1d3d80860e235426bef", "vendor/repo/scripts/augmentations.py": "a929e73a2d6f963f2d9dc7003d96cf7c8ff4476e33601cdc1ea92cbb6604d37b", "vendor/repo/scripts/dataspec.py": "a1454dc5feb9a794dff4e6b0a9f98a75ad69e0bb1d459911de6faa491d18c2f6", "vendor/repo/scripts/train_unet_transformer.py": "83e3f30a3313ea452b5072b118bfd06539302cf6886e317c542a602deb27d281", "vendor/repo/src/tracking_cellmot/__init__.py": "f9df2098dd85934d0b0d48d477420b1f9ed20a9a55ca6294a1874b964e9166e6", "vendor/repo/src/tracking_cellmot/division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9", "vendor/repo/src/tracking_cellmot/img_proc.py": "00e8ef0adc8b39f1aaaa547ea6197b906bf9e8c009e339d3e95f8f8dbf31be3f", "vendor/repo/src/tracking_cellmot/io.py": "efae135b088cecaab463d889f16c885ef6da3ad27b0747327d8ddc28d866b7bd", "vendor/repo/src/tracking_cellmot/metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444", "vendor/repo/src/tracking_cellmot/models/__init__.py": "0861b0bf0d5a574dd0d582c529b763e75c8d271f4df48df15fae3cf5b5fbb127", "vendor/repo/src/tracking_cellmot/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e", "vendor/repo/src/tracking_cellmot/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac"}
    for relative, digest in expected.items():
        with open(os.path.join(CODE_DIR, relative), "rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest, relative
    log(f"Code and checkpoint hashes verified ({len(expected)} files)")

    # Version 1 (ILP1 with both prunings, rematch and adjacent joins off) + the group N parameters in SubmitConfig.
    assert cfg.edge_threshold == 0.20
    assert not cfg.ilp1["enable"] and not cfg.rematch["enable"] and cfg.gap_close["enable"]
    assert not cfg.adjacent_join["enable"]
    assert tuple(cfg.gap_close["gaps"]) == (2, 10) and cfg.gap_fill["velocity_residual_um"] == 3.0
    assert cfg.ilp3["keep_lo"] == 0.25 and cfg.ilp3["gamma"] == 0.5 and cfg.ilp3["disappearance"] == 3.5
    assert cfg.ilp2["idsim"] == 0.5 and cfg.ilp_division_weight == 0.85
    assert cfg.smooth_weight == 0.825 and cfg.smooth_max_shift_um == 4.5
    cfg.weights_dir = os.path.join(mount(("hirotetsu", "ctdd-run53-run74-run79-ensemble-code")), "models")
    cfg.checkpoint_paths = [
        "run53/fold_0/epoch_035.pth",
        "run53/fold_1/epoch_025.pth",
        "run53/fold_2/epoch_020.pth",
        "run53/fold_3/epoch_028.pth",
        "run53/fold_4/epoch_028.pth",
        "run74/fold_0/epoch_027.pth",
        "run74/fold_1/epoch_030.pth",
        "run74/fold_2/epoch_024.pth",
        "run74/fold_3/epoch_025.pth",
        "run74/fold_4/epoch_025.pth",
        "run79/fold_0/epoch_024.pth",
    ]
    expected_weights = {"run53/fold_0/config.json":"4bda7f24b9fa191933b16cbdbe60e1a392344b950a8bfb725541c69e1c871b0d","run53/fold_0/epoch_035.pth":"f2f7a389db5e8176e79457d44fe4e5f257d2606095f84302c001e03b3f3ba049","run53/fold_1/config.json":"4bda7f24b9fa191933b16cbdbe60e1a392344b950a8bfb725541c69e1c871b0d","run53/fold_1/epoch_025.pth":"c483c14ec9b08dd80a300c068768eff08a8b68ebcedbd155c6d8a944b48d5a0c","run53/fold_2/config.json":"4bda7f24b9fa191933b16cbdbe60e1a392344b950a8bfb725541c69e1c871b0d","run53/fold_2/epoch_020.pth":"35bb3c9016c0cd65c743dc6e9c0240ab7e51c4ac5fd72a8e688de8658c0989c0","run53/fold_3/config.json":"4bda7f24b9fa191933b16cbdbe60e1a392344b950a8bfb725541c69e1c871b0d","run53/fold_3/epoch_028.pth":"45577ad81faaf4e011304121d53c07eefa8e7b71a9d3968b44b69dee0ea3a314","run53/fold_4/config.json":"4bda7f24b9fa191933b16cbdbe60e1a392344b950a8bfb725541c69e1c871b0d","run53/fold_4/epoch_028.pth":"32bba84732e76668d51a6b0ac56df8560fa073f01743d1cbc6c8c5ba712d28c6","run74/fold_0/config.json":"c12f27db934d78b8252ad841fc97340450677a566ef447064352d8988d8a39db","run74/fold_0/epoch_027.pth":"3ef6c81db42a3144d6bba0744291fc0581b18929ef3a5f99191cc2167e6a5995","run74/fold_1/config.json":"c12f27db934d78b8252ad841fc97340450677a566ef447064352d8988d8a39db","run74/fold_1/epoch_030.pth":"68ce29fdfc82a7df75638c0d7ceac481b47c767903cefa4cf249e1cdae28f9d5","run74/fold_2/config.json":"c12f27db934d78b8252ad841fc97340450677a566ef447064352d8988d8a39db","run74/fold_2/epoch_024.pth":"bdd822f4a1829aca673f27660b6453a1c7e3a23c5b1d51b53f776229c398be92","run74/fold_3/config.json":"c12f27db934d78b8252ad841fc97340450677a566ef447064352d8988d8a39db","run74/fold_3/epoch_025.pth":"bd6ef04419cce9db46796960020f064b86c3b9918cf37024e1e505bc2e4a71cb","run74/fold_4/config.json":"c12f27db934d78b8252ad841fc97340450677a566ef447064352d8988d8a39db","run74/fold_4/epoch_025.pth":"f08ea81b59174ee75872a272feb79e465c53eb5653093fd4f29f7512d0abfb71","run79/fold_0/config.json":"57d4e3f1d2c1295f9e06da8fba31f803e2ef7383954a7e5f3a1e7c8bfe418119","run79/fold_0/epoch_024.pth":"0989b24341c462fd777156ea8b7b7b280cd888584e158e55d3a7cf3dcc8e8a07"}
    for relative, digest in expected_weights.items():
        with open(os.path.join(cfg.weights_dir, relative), "rb") as checkpoint:
            assert hashlib.file_digest(checkpoint, "sha256").hexdigest() == digest, relative
    log(f"Selected {len(cfg.checkpoint_paths)} models, equal weights; checkpoint/config hashes verified")
    log("solve: repo inference.solve_graph (ILP2 -> ILP3 relink -> ILP4 gap close (2-10 frames) -> smoothing .825 cap 4.5 um "
        "+ gap fill; ILP1, rematch and adjacent joins off)")

    import torch
    from dataset import clip_ids
    from inference import run_clips
    from utils import refine_peaks_parabolic, write_submission

    assert callable(refine_peaks_parabolic)

    ids = clip_ids(cfg)
    n_gpu = torch.cuda.device_count()
    log(f"vendor_src : {cfg.vendor_src}")
    log(f"weights_dir: {cfg.weights_dir}")
    log(f"data_dir   : {cfg.data_dir} | clips: {len(ids)} | GPUs: {n_gpu}")
    log(f"inference  : tta={cfg.tta_mode if cfg.use_tta else 'none'} "
        f"tta_head={cfg.tta_head_fusion} tta_idsim={cfg.tta_idsim_fusion} "
        f"thr={cfg.point_threshold} postprocess={cfg.track_postprocess} "
        f"edge_thr={cfg.edge_threshold} ilp={cfg.ilp_solver} "
        f"temporal_fusion={cfg.temporal_detection_fusion} "
        f"parabolic={cfg.parabolic_refinement} "
        f"checkpoints={cfg.checkpoint_paths} "
        f"ilp1={cfg.ilp1} ilp2={cfg.ilp2} ilp3={cfg.ilp3} division_weight={cfg.ilp_division_weight} gap_close={cfg.gap_close} "
        f"rematch={cfg.rematch['enable']} gap_close={cfg.gap_close['enable']} adjacent_join={cfg.adjacent_join['enable']} "
        f"smooth={cfg.smooth_window}/{cfg.smooth_weight} cap={cfg.smooth_max_shift_um} "
        f"smooth_preserve_motion={cfg.smooth_preserve_motion} "
        f"gap_fill={cfg.gap_fill} "
        f"idsim={cfg.idsim} centroid={cfg.centroid} "
        f"tta_batch={cfg.inference_tta_batch_size} devices={cfg.inference_devices}")

    if n_gpu < 2:
        log("WARNING: single GPU — the scored rerun may exceed the 12h limit")
    # fast path: one persistent worker per GPU on a shared longest-first clip queue
    run_clips(cfg, ids)

    write_submission(cfg, ids)
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
