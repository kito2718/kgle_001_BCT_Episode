"""Biohub cell tracking submission: UnetCellPose detection, soon-to-divide scoring, edge linking."""

import glob
import os
import queue
import shutil
import tempfile
import threading
import time

T_BOOT = time.perf_counter()
import numpy as np
import torch
from cell_tracking.common import tracks as B
from cell_tracking.linker.cache import extract_frame_cache, write_frame_cache
from cell_tracking.linker.ensemble import EdgeEnsemble
from cell_tracking.linker.intensity import write_intensity_sidecar
from cell_tracking.linker.graph import graph_from_pair_links, postprocess_graph
from cell_tracking.linker.link import link_movie_from_cache, load_edge_checkpoint
from cell_tracking.linker.scoring import link_confidence
from cell_tracking.linker.occupancy import occupancy_frame_path, write_occupancy_from_maps
from cell_tracking.linker.soon_cache import soon_frame_path, write_soon_frame
from cell_tracking.detector.inference import load_unetcellpose_ensemble
from cell_tracking.common.io import normalize
from cell_tracking.soon.scoring import score_points_per_net
from cell_tracking.soon.checkpoints import deploy_context, load_soon_crop
from cell_tracking.soon.crops import native_crop_size
from cell_tracking.linker.tiered_gate import tiered_temporal_gate

DETECTORS = ["det_fold0.ckpt", "det_fold1.ckpt", "det_fold2.ckpt", "det_fold3.ckpt", "det_fold4.ckpt"]
SOONS = ["soon_fold0.pt", "soon_fold1.pt", "soon_fold2.pt", "soon_fold3.pt", "soon_fold4.pt"]
EDGES = ["edge_fold0/edge.pt", "edge_fold1/edge.pt", "edge_fold2/edge.pt", "edge_fold3/edge.pt", "edge_fold4/edge.pt"]
DET_TTA = "xy"
DET_AUTOCAST = "fp16"
SOON_AUTOCAST = "fp16"
SOON_BATCH = 64
EDGE_SHORTLIST = "consensus"
THRESHOLD = 0.1
MAX_CHILDREN = 2
MIN_PROB = 0.0
DECODE = "dual"
FORK_DECODE = "greedy"
DIV_REFRACTORY = True
DIV_REFRACTORY_ORDER = "prob"
FLOW_KIND = "local_c2f"
TIER_CONF = 0.3
TIER_MIN_LEN = 3
LINEFIT = True
MAX_GAP = 1
GAP_UM = 4.0
GAP_DRIFT_UM = 30.0


def weight_path(name):
    path = os.path.join(CODE, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("%s missing from the dataset" % path)
    return path


def load_stack(device):
    detector = load_unetcellpose_ensemble(
        [weight_path(p) for p in DETECTORS], device=device, tta=DET_TTA, autocast=DET_AUTOCAST, expected_pool=2
    )
    soon = []
    for p in SOONS:
        net, meta = load_soon_crop(weight_path(p), device=device)
        assert meta.get("features") == "intensity" and int(meta.get("num_classes", 1)) == 3, p
        assert getattr(net, "aux_class_aware", False), "%s has no class-aware aux head" % p
        crop_contract = net.n_frames == 3 and net.in_ch == 4 and (net.xy_pool == 1)
        if torch.device(device).type == "cuda" and (not crop_contract):
            raise SystemExit("%s is incompatible with the GPU crop path" % p)
        soon.append(
            {
                "net": net,
                "crop_size": tuple((int(v) for v in tuple(meta["crop_size"]))),
                "xy_pool": int(getattr(net, "xy_pool", 1)),
                "context": deploy_context(meta),
            }
        )
    if len({s["crop_size"] for s in soon}) != 1 or len({s["xy_pool"] for s in soon}) != 1:
        raise SystemExit("soon nets must share one crop size and xy_pool")
    models = [load_edge_checkpoint(weight_path(p), device=device) for p in EDGES]
    edge = EdgeEnsemble([m for m, _ in models], shortlist=EDGE_SHORTLIST)
    cfg = models[0][1]
    for _, edge_cfg in models[1:]:
        for key in (
            "feat_dim",
            "fork_kind_label_smoothing",
            "flow_kind",
            "occupancy_pair",
            "occupancy_fork",
            "occupancy_fork_daughter",
        ):
            if edge_cfg.get(key) != cfg.get(key):
                raise SystemExit("edge ensemble config mismatch for %s" % key)
    for key in (
        "fork_head",
        "shape_feats",
        "intensity_pair",
        "intensity_fork",
        "occupancy_pair",
        "occupancy_fork",
        "occupancy_fork_daughter",
    ):
        if not cfg.get(key):
            raise SystemExit("this kernel expects a linker with %s=True" % key)
    return (detector, soon, edge, cfg)


class NormWindow:
    """Normalised frames for one movie, keeping only the last couple in memory."""

    def __init__(self, raw):
        self.raw = raw
        self.cache = {}

    def get(self, t):
        if t not in self.cache:
            self.cache[t] = normalize(self.raw[t])
        return self.cache[t]

    def window(self, t, context):
        vol = self.get(t)
        n = len(self.raw)
        prev = (self.get(t - 1) if t >= 1 else vol) if context >= 1 else None
        nxt = (self.get(t + 1) if t + 1 < n else vol) if context >= 2 else None
        self.cache = {k: v for k, v in self.cache.items() if k >= t - 1}
        return (vol, prev, nxt)


def score_soon(soon, nvol, coords, prev, nxt, device):
    """Soon softmax averaged over nets, plus averaged occupancy and daughter maps."""
    if not len(coords):
        return (np.zeros((0, 3), np.float32), None, None)
    scores, occ, div = score_points_per_net(
        [s["net"] for s in soon],
        nvol,
        coords,
        crop_size=soon[0]["crop_size"],
        prev=prev,
        nxt=nxt,
        device=device,
        batch=SOON_BATCH,
        autocast=SOON_AUTOCAST,
    )
    mean = np.mean([s.astype(np.float64) for s in scores], axis=0)
    return (mean.astype(np.float32), occ, div)


def write_occupancy(root, movie, t, crop_size, coords, maps, dest_next, dest_prev):
    if not len(coords):
        maps = np.zeros((0, 3, 1, 1, 1), np.float32)
    write_occupancy_from_maps(
        occupancy_frame_path(root, movie, t), maps, coords, dest_next, dest_prev, crop_size, frame=t, store_maps=False
    )


def build_caches(detector, soon, raw, dirs, movie, device):
    """Detect every frame and write the edge cache plus soon/intensity/occupancy sidecars."""
    window = NormWindow(raw)
    context = soon[0]["context"]
    occ_crop = native_crop_size(soon[0]["crop_size"], soon[0]["xy_pool"])
    empty = np.zeros((0, 3), np.float64)
    coords_by_frame = []
    pending = None
    n_det = 0

    def flush(t, dest_next):
        dest_prev = coords_by_frame[t - 1] if t >= 1 else empty
        occ, div = pending
        write_occupancy(dirs["occ"], movie, t, occ_crop, coords_by_frame[t], occ, dest_next, dest_prev)
        write_occupancy(dirs["occ_div"], movie, t, occ_crop, coords_by_frame[t], div, dest_next, dest_prev)

    for t, vol in enumerate(raw):

        def frame_at(dt, t=t):
            return raw[t + dt] if 0 <= t + dt < len(raw) else None

        coords, feats, vols = extract_frame_cache(
            detector,
            vol,
            device=device,
            frame_at=frame_at,
            decode_device=device,
            include_appearance=False,
            include_geom=True,
            include_time=True,
        )
        coords = np.asarray(coords, np.float64)
        vols = np.asarray(vols, np.float64)
        write_frame_cache(
            os.path.join(dirs["cache"], movie, "frame_%04d" % t), coords, np.asarray(feats, np.float32), vols, t
        )
        write_intensity_sidecar(
            os.path.join(dirs["intensity"], movie, "frame_%04d_intensity.npz" % t), vol, coords, volumes=vols, frame=t
        )
        coords_by_frame.append(coords)
        n_det += len(coords)
        nvol, prev, nxt = window.window(t, context)
        scores, occ, div = score_soon(soon, nvol, coords, prev, nxt, device)
        write_soon_frame(soon_frame_path(dirs["soon"], movie, t), scores)
        if t >= 1:
            flush(t - 1, coords)
        pending = (occ, div)
    if raw:
        flush(len(raw) - 1, empty)
    return n_det


def link_and_post(edge, cfg, dirs, movie, device):
    frames, links, _, link_probs = link_movie_from_cache(
        edge,
        dirs["cache"],
        movie,
        device=device,
        threshold=THRESHOLD,
        max_children=MAX_CHILDREN,
        min_prob=MIN_PROB,
        support_radius_um=float(cfg.get("support_radius_um", 30.0)),
        support_k=int(cfg.get("support_k", 16)),
        soon_dir=dirs["soon"],
        flow_k=int(cfg.get("flow_k", 20)),
        intensity_dir=dirs["intensity"],
        occupancy_dir=dirs["occ"],
        occupancy_division_dir=dirs["occ_div"],
    )
    if len(frames) < 2:
        return None
    g = graph_from_pair_links(B, frames, links)
    g = tiered_temporal_gate(g, link_confidence(frames, links, link_probs), conf_thresh=TIER_CONF, min_len=TIER_MIN_LEN)
    return postprocess_graph(
        B, frames, g, linefit=LINEFIT, max_gap=MAX_GAP, gap_dist_um=GAP_UM, gap_drift_um=GAP_DRIFT_UM
    )


def movie_bytes(path):
    """On-disk size of a zarr movie: a cheap proxy for its detection count."""
    return sum((os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs))


def main():
    t_code = time.perf_counter() - T_BOOT
    test_dir = test_directory()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = ["cuda:%d" % i for i in range(n_gpu)] or ["cpu"]
    t0 = time.perf_counter()
    stacks = [load_stack(d) for d in devices]
    print("code", CODE, "(imports + code search %.0fs, model load %.0fs)" % (t_code, time.perf_counter() - t0))
    print("members detectors=%d soon=%d linkers=%d" % (len(DETECTORS), len(SOONS), len(EDGES)), flush=True)
    print("detectors", DETECTORS, "soon", SOONS, "edge", EDGES)
    print("devices", devices, flush=True)
    if all((torch.device(d).type == "cuda" for d in devices)):
        print("soon crop path: GPU (3 frames + Gaussian, xy_pool=1)", flush=True)
    names = sorted((os.path.basename(p)[:-5] for p in glob.glob(test_dir + "/*.zarr")))
    print("test movies:", len(names), flush=True)
    todo = queue.Queue()
    for i in sorted(range(len(names)), key=lambda i: -movie_bytes(test_dir + "/" + names[i] + ".zarr")):
        todo.put(i)
    tmp = tempfile.mkdtemp(prefix="biohub_", dir=WORK_DIR)
    rows_by_movie = [None] * len(names)
    errors = []
    t_start = time.perf_counter()

    def run_worker(k):
        detector, soon, edge, cfg = stacks[k]
        device = devices[k]
        dirs = {key: os.path.join(tmp, "%s_%d" % (key, k)) for key in ("cache", "soon", "intensity", "occ", "occ_div")}
        try:
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                name = names[i]
                t0 = time.perf_counter()
                movie = B.open_image(test_dir + "/" + name + ".zarr")
                raw = [movie.frame(t) for t in range(movie.n_t)]
                t1 = time.perf_counter()
                n_det = build_caches(detector, soon, raw, dirs, name, device)
                t2 = time.perf_counter()
                g = link_and_post(edge, cfg, dirs, name, device)
                rows_by_movie[i] = [] if g is None else B.graph_to_rows(name, g)
                t3 = time.perf_counter()
                print(
                    "  %s  %.1fs (read %.0f, detect+soon %.0f, link+post %.0f)  ndet=%d  nodes=%d  [%s]"
                    % (name, t3 - t0, t1 - t0, t2 - t1, t3 - t2, n_det, g.n_nodes if g else 0, device),
                    flush=True,
                )
                for d in dirs.values():
                    shutil.rmtree(os.path.join(d, name), ignore_errors=True)
        except Exception as exc:
            errors.append((device, exc))
            raise

    threads = [threading.Thread(target=run_worker, args=(k,), daemon=True) for k in range(len(devices))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    shutil.rmtree(tmp, ignore_errors=True)
    if errors:
        raise RuntimeError("worker on %s failed: %r" % errors[0])
    missing = [n for n, r in zip(names, rows_by_movie) if r is None]
    if missing:
        raise RuntimeError("no rows for %d movie(s): %r" % (len(missing), missing[:3]))
    rows = [r for part in rows_by_movie for r in part]
    B.write_submission(rows, OUTPUT)
    total = time.perf_counter() - t_start
    print("wrote submission.csv  %d rows  %.0fs" % (len(rows), total), flush=True)


def test_directory():
    return TEST_DIR


def cli():
    import argparse
    from pathlib import Path

    global CODE, TEST_DIR, WORK_DIR, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    CODE = str(Path(args.weights_dir).resolve())
    TEST_DIR = str(Path(args.data_dir).resolve())
    WORK_DIR = str(Path(args.work_dir).resolve())
    OUTPUT = str(Path(args.output).resolve())
    if not Path(TEST_DIR).is_dir() or not list(Path(TEST_DIR).glob("*.zarr")):
        raise FileNotFoundError(f"no movies in {TEST_DIR}")
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    main()
