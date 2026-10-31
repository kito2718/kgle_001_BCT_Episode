"""Donor bank — real cell material harvested per sample, once.

Two donor kinds:

  tracklet   maximal single-child GT chains cut to runs where EVERY node is
             isolated (no other GT node within 8 um in its frame), in-bounds,
             and reads as ONE clean blob (n_peaks_5um == 1 at the [1,4,4]
             grid). These become faint cells and manufactured second
             daughters. Isolation is what keeps a transplant from carrying a
             neighbor's shoulder into the new site.
  event      whole GT divisions: mother chain (<= 3 back) + both daughter
             chains (<= 6 forward). No isolation requirement — divisions are
             precious (151 total) — but crowding is recorded so B1 can prefer
             clean ones.

Patches are HARVEST-NORMALIZED raw voxels (exact cache path: per-frame
quantile lo 0.02 / hi 0.999), float16, (13, 49, 49) around the integer GT
voxel. Donor units are the training units by construction.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.metrics.gt import SCALE_ARR, load_gt
from biohub.pipeline.baseline import BaselineConfig, downsample, normalise
from biohub.synthesis.compose import (
    PATCH_SHAPE, PATCH_X, PATCH_Y, PATCH_Z,
    estimate_background, measure_prominence_grid,
)

# 6.5 um: above the 5 um NMS floor with margin, relaxed from 8.0 after the
# first harvest measured yields of 5-38 tracklets/sample (annotation sparsity,
# not filter tightness, is the binding constraint — but every tracklet counts).
ISO_UM = 6.5
MIN_SPAN = 5
CAP_TRACKLETS = 200
EVENT_BACK = 3
EVENT_FWD = 6
NORM = BaselineConfig(norm_lo_q=0.02, norm_hi_q=0.999)


def _chains(gt) -> tuple[dict[int, list[int]], dict[int, int]]:
    childs: dict[int, list[int]] = defaultdict(list)
    parent: dict[int, int] = {}
    for s, d in gt.edges:
        childs[int(s)].append(int(d))
        parent[int(d)] = int(s)
    return childs, parent


def _in_bounds(zyx: np.ndarray, shape_raw: tuple[int, int, int]) -> bool:
    z, y, x = zyx
    return (PATCH_Z <= z < shape_raw[0] - PATCH_Z
            and PATCH_Y <= y < shape_raw[1] - PATCH_Y
            and PATCH_X <= x < shape_raw[2] - PATCH_X)


def _isolated(gt, trees) -> np.ndarray:
    """Per-node: no other GT node within ISO_UM at its frame."""
    out = np.zeros(gt.n_nodes, dtype=bool)
    pos_um = gt.zyx * SCALE_ARR
    for i in range(gt.n_nodes):
        tree, rows = trees(int(gt.t[i]))
        if tree is None:
            out[i] = True
            continue
        hits = tree.query_ball_point(pos_um[i], ISO_UM)
        out[i] = len(hits) <= 1  # itself
    return out


def harvest_sample(name: str, data_root: Path, out_dir: Path,
                   cap_tracklets: int = CAP_TRACKLETS,
                   seed: int = 20260817) -> Path:
    """Build `{out_dir}/{name}.npz` + `{name}.parquet`; returns the npz path.

    Idempotent: returns immediately when both files exist.
    """
    import zarr
    from scipy.spatial import cKDTree

    npz_path = out_dir / f"{name}.npz"
    meta_path = out_dir / f"{name}.parquet"
    if npz_path.exists() and meta_path.exists():
        return npz_path
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    gt = load_gt(Path(data_root) / f"{name}.geff")
    arr = zarr.open_array(str(Path(data_root) / f"{name}.zarr" / "0"), mode="r")
    shape_raw = tuple(arr.shape[1:])
    childs, parent = _chains(gt)
    pos_um = gt.zyx * SCALE_ARR

    trees_cache: dict[int, tuple] = {}

    def trees(t: int):
        if t not in trees_cache:
            rows = np.flatnonzero(gt.t == t)
            trees_cache[t] = ((cKDTree(pos_um[rows]) if len(rows) else None),
                              rows)
        return trees_cache[t]

    iso = _isolated(gt, trees)
    inb = np.array([_in_bounds(gt.zyx[i], shape_raw)
                    for i in range(gt.n_nodes)])

    # ---- tracklet runs: maximal chains cut to isolated & in-bounds runs ----
    div_rows = set(int(p) for p in gt.divisions())
    starts = [i for i in range(gt.n_nodes)
              if i not in parent or int(parent[i]) in div_rows]
    runs: list[list[int]] = []
    for s in starts:
        chain = [s]
        cur = s
        while True:
            kids = childs.get(cur, [])
            if len(kids) != 1:
                break
            cur = kids[0]
            chain.append(cur)
        ok = [bool(iso[i] and inb[i]) for i in chain]
        run: list[int] = []
        for node, good in zip(chain, ok):
            if good:
                run.append(node)
            else:
                if len(run) >= MIN_SPAN:
                    runs.append(run)
                run = []
        if len(run) >= MIN_SPAN:
            runs.append(run)
    rng = np.random.default_rng(seed)
    if len(runs) > cap_tracklets:
        keep = rng.choice(len(runs), size=cap_tracklets, replace=False)
        runs = [runs[int(k)] for k in sorted(keep)]

    # ---- events: mother history + both daughter chains ----
    events: list[dict] = []
    for p in gt.divisions():
        kids = childs[int(p)]
        if len(kids) != 2:
            continue
        hist = [int(p)]
        cur = int(p)
        for _ in range(EVENT_BACK):
            if cur not in parent:
                break
            cur = parent[cur]
            hist.append(cur)
        hist = hist[::-1]  # oldest first, mother last
        d_chains = []
        for k in kids:
            ch = [int(k)]
            cur = int(k)
            for _ in range(EVENT_FWD - 1):
                kk = childs.get(cur, [])
                if len(kk) != 1:
                    break
                cur = kk[0]
                ch.append(cur)
            d_chains.append(ch)
        # nearer daughter first (matches division_bank convention)
        d0 = float(np.linalg.norm(pos_um[d_chains[0][0]] - pos_um[int(p)]))
        d1 = float(np.linalg.norm(pos_um[d_chains[1][0]] - pos_um[int(p)]))
        if d1 < d0:
            d_chains = [d_chains[1], d_chains[0]]
        all_nodes = hist + d_chains[0] + d_chains[1]
        if not all(inb[i] for i in all_nodes):
            continue
        events.append({"parent_row": int(p), "hist": hist,
                       "d1": d_chains[0], "d2": d_chains[1]})

    # ---- frames to normalize ----
    need: dict[int, list[tuple]] = defaultdict(list)  # t -> [(kind, ids...)]
    for ti, run in enumerate(runs):
        for si, node in enumerate(run):
            need[int(gt.t[node])].append(("tracklet", ti, si, node, "cell"))
    for ei, ev in enumerate(events):
        for si, node in enumerate(ev["hist"]):
            need[int(gt.t[node])].append(("event", ei, si, node, "mother"))
        for role, chain in (("d1", ev["d1"]), ("d2", ev["d2"])):
            for si, node in enumerate(chain):
                need[int(gt.t[node])].append(("event", ei, si, node, role))

    patches: list[np.ndarray] = []
    meta: list[dict] = []
    for t in sorted(need):
        frame = normalise(np.asarray(arr[t]), NORM)
        grid = downsample(frame)
        for kind, gid, si, node, role in need[t]:
            cz, cy, cx = (int(round(v)) for v in gt.zyx[node])
            if not _in_bounds((cz, cy, cx), shape_raw):
                continue
            patch = frame[cz - PATCH_Z:cz + PATCH_Z + 1,
                          cy - PATCH_Y:cy + PATCH_Y + 1,
                          cx - PATCH_X:cx + PATCH_X + 1]
            assert patch.shape == PATCH_SHAPE
            bg_med, bg_mad = estimate_background(patch)
            q = measure_prominence_grid(grid, gt.zyx[node])
            gz, gy, gx = (int(round(v / d)) for v, d in
                          zip(gt.zyx[node], (1, 4, 4)))
            gz = min(max(gz, 1), grid.shape[0] - 2)
            gy = min(max(gy, 1), grid.shape[1] - 2)
            gx = min(max(gx, 1), grid.shape[2] - 2)
            peak_grid = float(grid[gz - 1:gz + 2, gy - 1:gy + 2,
                                   gx - 1:gx + 2].max())
            meta.append({
                "idx": len(patches), "kind": kind, "gid": int(gid),
                "seq": int(si), "role": role, "node_row": int(node),
                "t": int(t), "cz": cz, "cy": cy, "cx": cx,
                "z_um": float(gt.zyx[node][0] * SCALE_ARR[0]),
                "bg_med": bg_med, "bg_mad": bg_mad,
                "prom44": (q or {}).get("prominence", np.nan),
                "n_peaks44": (q or {}).get("n_peaks_5um", -1),
                "peak_grid": peak_grid,
                "peak_sig_raw": float(patch.max() - bg_med),
            })
            patches.append(patch.astype(np.float16))

    np.savez_compressed(npz_path, patches=np.stack(patches)
                        if patches else np.zeros((0, *PATCH_SHAPE), np.float16))
    pd.DataFrame(meta).to_parquet(meta_path, index=False)
    print(f"{name}: {len(runs)} tracklets, {len(events)} events, "
          f"{len(patches)} patches, {time.perf_counter()-t0:.0f}s", flush=True)
    return npz_path


class DonorBank:
    """Loader over one sample's harvested bank."""

    def __init__(self, name: str, bank_dir: Path):
        self.name = name
        self.patches = np.load(bank_dir / f"{name}.npz")["patches"]
        self.meta = pd.read_parquet(bank_dir / f"{name}.parquet")
        m = self.meta
        self.tracklets: dict[int, pd.DataFrame] = {
            int(g): d.sort_values("seq")
            for g, d in m[m.kind == "tracklet"].groupby("gid")}
        # Clean tracklets: mostly <= 2 LoG maxima near center. ==1 at 80% was
        # measured too strict (5/34 donors on 6bba_57b7cc1e — noise ripple
        # makes shallow second maxima); the footprint mask confines any
        # fragment to the central component regardless.
        self.clean_ids = [g for g, d in self.tracklets.items()
                          if (d.n_peaks44 <= 2).mean() >= 0.7]
        self.events: dict[int, pd.DataFrame] = {
            int(g): d for g, d in m[m.kind == "event"].groupby("gid")}

    def patch(self, idx: int) -> np.ndarray:
        return self.patches[int(idx)].astype(np.float32)
