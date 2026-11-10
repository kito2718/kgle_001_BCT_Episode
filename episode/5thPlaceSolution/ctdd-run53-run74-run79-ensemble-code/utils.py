# Array ops, TTA, graph assembly, submission writing and metrics — pure functions.

import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import *

sys.path.append(BaseConfig.vendor_src)

import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import tracksdata as td  # noqa: E402
from ilpy import Relation  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]

def calculate_downsample(cfg, raw_shape=None):
    """Calculate integer ZYX downsampling from the configured input shape."""
    raw_shape = tuple(cfg.raw_volume_shape if raw_shape is None else raw_shape)
    volume_shape = tuple(cfg.volume_shape)
    if len(raw_shape) != 3 or len(volume_shape) != 3:
        raise ValueError("raw_volume_shape and volume_shape must have three dimensions")
    if any(
        target <= 0 or raw % target
        for raw, target in zip(raw_shape, volume_shape)
    ):
        raise ValueError(
            f"volume_shape {volume_shape} must divide raw shape {raw_shape} exactly"
        )
    return tuple(raw // target for raw, target in zip(raw_shape, volume_shape))


def downsample_volume(volume, downsample):
    """Average-pool an (N, Z, Y, X) tensor onto the network grid."""
    # PyTorch requires Python ints; only the container type changes, not the factors.
    downsample = tuple(int(value) for value in downsample)
    return F.avg_pool3d(
        volume[:, None], kernel_size=downsample, stride=downsample
    )[:, 0]


def image_to_network_coordinates(coords, downsample):
    """Map raw-image ZYX coordinates onto the downsampled network grid."""
    if torch.is_tensor(coords):
        dtype = coords.dtype if coords.is_floating_point() else torch.float32
        scale = torch.as_tensor(downsample, dtype=dtype, device=coords.device)
    else:
        scale = np.asarray(downsample, dtype=np.float32)
    return (coords + 0.5) / scale - 0.5


def network_to_image_coordinates(coords, downsample):
    """Map network-grid ZYX coordinates back onto the raw image."""
    if torch.is_tensor(coords):
        dtype = coords.dtype if coords.is_floating_point() else torch.float32
        scale = torch.as_tensor(downsample, dtype=dtype, device=coords.device)
    else:
        scale = np.asarray(downsample, dtype=np.float32)
    return (coords + 0.5) * scale - 0.5


def config_dict(cfg):
    # vars() alone only sees settings overridden by the leaf config class.
    settings = {}
    for klass in reversed(cfg.__mro__):
        for key, value in vars(klass).items():
            if key.startswith("_"):
                continue
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                settings[key] = value
    return settings


def write_provenance(cfg, out_dir, fold, model_config, settings):
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump({
            "fold_mode": cfg.fold_mode,
            "fold": cfg.fold,
            "n_folds": cfg.n_folds,
            "seed": cfg.seed,
            "val_in_train": fold["val_in_train"],
            "n_train": len(fold["train"]),
            "n_test": len(fold["test"]),
            "train": fold["train"],
            "test": fold["test"],
        }, f, indent=2)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(out_dir, "training_config.json"), "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)


def load_checkpoint(model, checkpoint_path, device="cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"loaded checkpoint: {checkpoint_path}")
    return model


def quiet_known_warnings():
    # Two tracksdata warnings fire on every clip and mean nothing here. Filter by message,
    # not by level: "Trivial solution" and "did not converge" come from the same logger and
    # would mean the ILP is broken.
    import logging
    import warnings

    # Only 0.0276% of ZebraHub cells collide after the configured Y/X downsampling.
    # The binary detection target already merges them; repeating one warning per
    # sample only obscures the training log.
    warnings.filterwarnings(
        "ignore",
        message=r"Sample .* GT nodes collapsed to duplicate voxels after downsampling.*",
    )

    known = (
        "Solver failed with Gurobi",   # no commercial licence; SCIP solves it exactly
        "No matching edges found",     # one per GT division the model failed to predict
        "No matching nodes found",
    )
    logging.getLogger("tracksdata.utils._logging").addFilter(
        lambda r: not any(s in r.getMessage() for s in known))


# ── detection helpers ────────────────────────────────────────────────────────────
def pool_kernel_from_um(um, voxel_size):
    # Physical radius -> odd voxel kernel. At 1.625 um/voxel both 3.0 and 5.0 give 3.
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def prob_to_zyx(prob, threshold=0.5, pool_kernel=(3, 3, 3)):
    # Non-maximum suppression: keep voxels that equal the max of their neighbourhood.
    # One cell activates a whole blob; this collapses each blob to a single centre.
    prob = prob.unsqueeze(0)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(prob, pool_kernel, stride=1, padding=pad)
    is_peak = (prob == pooled) & (prob > threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])
    if peak_idx.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.long, device=prob.device)
    return peak_idx


def refine_peaks_parabolic(prob, peaks):
    """Refine integer ZYX peaks by fitting a log-parabola along each axis."""
    refined = peaks.float()
    if not len(peaks):
        return refined

    heatmap = prob[0]
    centre = heatmap[peaks[:, 0], peaks[:, 1], peaks[:, 2]].clamp_min(1e-6).log()
    shape = heatmap.shape

    for axis in range(3):
        valid = (peaks[:, axis] > 0) & (peaks[:, axis] < shape[axis] - 1)
        before = peaks.clone()
        after = peaks.clone()
        before[:, axis] = (before[:, axis] - 1).clamp_min(0)
        after[:, axis] = (after[:, axis] + 1).clamp_max(shape[axis] - 1)

        left = heatmap[
            before[:, 0], before[:, 1], before[:, 2]
        ].clamp_min(1e-6).log()
        right = heatmap[
            after[:, 0], after[:, 1], after[:, 2]
        ].clamp_min(1e-6).log()
        curvature = left - 2 * centre + right
        good = valid & (curvature < -1e-6)
        offset = torch.zeros_like(centre)
        offset[good] = 0.5 * (left[good] - right[good]) / curvature[good]
        refined[:, axis] += offset.clamp(-0.5, 0.5)

    return refined


def refine_centroid(raw, peaks_net, downsample, rz=2, ryx=5):
    """Return raw-intensity centroids in full-resolution image coordinates."""
    out = np.empty((len(peaks_net), 3), np.float32)
    z_size, y_size, x_size = raw.shape
    dz, dy, dx = (int(d) for d in downsample)
    for n, (z, y, x) in enumerate(peaks_net):
        zc = int(round((float(z) + 0.5) * dz - 0.5))
        yc = int(round((float(y) + 0.5) * dy - 0.5))
        xc = int(round((float(x) + 0.5) * dx - 0.5))
        z0, z1 = max(0, zc - rz), min(z_size, zc + rz + 1)
        y0, y1 = max(0, yc - ryx), min(y_size, yc + ryx + 1)
        x0, x1 = max(0, xc - ryx), min(x_size, xc + ryx + 1)
        crop = raw[z0:z1, y0:y1, x0:x1].astype(np.float32)
        weight = crop - crop.min()
        total = float(weight.sum())
        if total <= 0:
            out[n] = (zc, yc, xc)
            continue
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        out[n] = (
            (zz * weight).sum() / total,
            (yy * weight).sum() / total,
            (xx * weight).sum() / total,
        )
    return out


def select_feature(feature, zyx):
    # Index the already-computed feature map at the detected centres (C,Z,Y,X) -> (N,C).
    _, Z, Y, X = feature.shape
    z = zyx[:, 0].long().clamp(0, Z - 1)
    y = zyx[:, 1].long().clamp(0, Y - 1)
    x = zyx[:, 2].long().clamp(0, X - 1)
    return feature[:, z, y, x].permute(1, 0).contiguous()


def embed_position(zyx, t, cfg):
    # Fourier features over (t, z, y, x). The transformer is permutation invariant and
    # linking depends critically on distance, so raw coordinates are not enough.
    zyx = zyx.float()
    z, y, x = zyx.unbind(dim=1)
    t_tensor = torch.as_tensor(t, dtype=zyx.dtype, device=zyx.device)
    tzyx = [
        torch.ones_like(z) * (t_tensor / cfg.time_length),
        z / cfg.volume_shape[0],
        y / cfg.volume_shape[1],
        x / cfg.volume_shape[2],
    ]

    def embed(values):
        freqs = 2.0 ** torch.arange(cfg.pos_per_dim // 2, dtype=values.dtype, device=values.device)
        angles = values[:, None] * freqs[None, :] * torch.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    return torch.cat([embed(v) for v in tzyx], dim=1)


# ── test-time augmentation ───────────────────────────────────────────────────────
# Only the YX plane is transformed. Z is physically asymmetric (light enters from one
# side, deeper slices are dimmer), so flipping it would produce impossible inputs.
def do_tta(im, mode="8fliprot"):
    d = (-2, -1)
    if mode == "none":
        return [im]
    images = [
        im,                                  # identity
        im.flip(dims=(-1,)),                 # X flip
        im.flip(dims=(-2,)),                 # Y flip
        im.flip(dims=(-2, -1)),              # 180 rotation
        torch.rot90(im, 1, dims=d),          # 90
        torch.rot90(im, 3, dims=d),          # 270
        im.transpose(-1, -2),                # main-diagonal reflection
    ]
    if mode == "8fliprot":
        images.append(torch.rot90(im, 1, dims=d).transpose(-1, -2))   # == flip_x (duplicate)
    elif mode == "8fixed":
        images.append(torch.rot90(im.transpose(-1, -2), 2, dims=d))   # true anti-diagonal
    else:
        raise ValueError(mode)
    return images


def undo_tta(x, mode="8fliprot"):
    d = (-2, -1)
    if mode == "none":
        return torch.stack([x[0]])
    restored = [
        x[0],
        x[1].flip(dims=(-1,)),
        x[2].flip(dims=(-2,)),
        x[3].flip(dims=(-2, -1)),
        torch.rot90(x[4], -1, dims=d),
        torch.rot90(x[5], -3, dims=d),
        x[6].transpose(-1, -2),
    ]
    if mode == "8fliprot":
        restored.append(torch.rot90(x[7].transpose(-1, -2), -1, dims=d))
    else:
        restored.append(torch.rot90(x[7], -2, dims=d).transpose(-1, -2))
    return torch.stack(restored)


# ── graph ────────────────────────────────────────────────────────────────────────
def build_graph(coord, edge, node_prob, appearance_prob, division_prob):
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x", "node_prob"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    graph.add_node_attr_key("appearance_prob", pl.Float64, 0.0)
    graph.add_node_attr_key("division_prob", pl.Float64, 0.0)

    node_ids = graph.bulk_add_nodes(
        [
            {
                "t": int(t), "z": float(z), "y": float(y), "x": float(x),
                "node_prob": float(q), "appearance_prob": float(a),
                "division_prob": float(d),
            }
            for (t, z, y, x), q, a, d in zip(
                coord, node_prob, appearance_prob, division_prob, strict=True,
            )
        ]
    )
    if edge:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.add_edge_attr_key("id_dissim", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {"source_id": node_ids[i], "target_id": node_ids[j],
             "edge_prob": p, "edge_dist": d, "id_dissim": s}
            for i, j, p, d, s in edge
        ])
    return graph


def local_flow(um, t, neighbours, search_um):
    # Collective motion of the detections at every frame, from detections alone.
    # Mutual nearest neighbours saturate once a drift exceeds the cell spacing
    # (an 8 um jump reads as 1 um), so the whole-frame shift is first found by
    # a grid search, refined by the matched-pair median; each detection then
    # takes the median displacement of its nearest matched pairs.
    from scipy.spatial import cKDTree

    def matched_pairs(a, b, tree_a, tree_b, shift):
        # mutual nearest neighbours between frame t shifted by `shift` and t+1;
        # the trees are built once per frame pair, the shift moves the queries
        _, ia = tree_b.query(um[a] + shift, k=1)
        _, ib = tree_a.query(um[b] - shift, k=1)
        m = np.flatnonzero(ib[ia] == np.arange(len(a)))
        return m, um[b[ia[m]]] - um[a[m]]

    flow = np.zeros_like(um)
    grid = np.arange(-search_um, search_um + 1, 2.0)
    for f in np.unique(t):
        a, b = np.flatnonzero(t == f), np.flatnonzero(t == f + 1)
        if len(a) < 5 or len(b) < 5:
            continue
        tree_a, tree_b = cKDTree(um[a]), cKDTree(um[b])
        best = np.zeros(3)
        m, disp = matched_pairs(a, b, tree_a, tree_b, best)
        best_n = int((np.linalg.norm(disp, axis=1) < 2.0).sum())
        if best_n < 0.8 * len(m):
            # not a still frame: search whole-frame shifts for the most consistent match
            for dz in (-3.0, 0.0, 3.0):
                for dy in grid:
                    for dx in grid:
                        shift = np.array([dz, dy, dx])
                        _, disp = matched_pairs(a, b, tree_a, tree_b, shift)
                        n = int((np.linalg.norm(disp - shift, axis=1) < 2.0).sum())
                        if n > best_n:
                            best, best_n = shift, n
        for _ in range(2):
            m, disp = matched_pairs(a, b, tree_a, tree_b, best)
            if len(m) < 5:
                break
            best = np.median(disp, axis=0)
        if len(m) < 5:
            continue
        k = min(neighbours, len(m))
        _, nn = cKDTree(um[a[m]]).query(um[a], k=k)
        flow[a] = np.median(disp[nn.reshape(len(a), k)], axis=1)
    return flow


def add_flow_cost(cfg, graph, scale):
    # Edge attr flow_cost = weight * max(0, |step - local flow| - knee); dividing
    # sources are exempt because daughters move against their neighbours.
    p = cfg.flow_cost
    if graph.num_edges() == 0:
        return graph
    nodes = graph.node_attrs()
    row = {int(v): i for i, v in enumerate(nodes["node_id"].to_numpy())}
    um = np.column_stack([nodes[a].to_numpy() for a in "zyx"]) * np.asarray(scale)
    flow = local_flow(um, nodes["t"].to_numpy(), p["neighbours"], p["drift_search_um"])
    dividing = nodes["division_prob"].to_numpy() > 0.5

    edges = graph.edge_attrs()
    src = np.array([row[int(v)] for v in edges["source_id"].to_numpy()])
    dst = np.array([row[int(v)] for v in edges["target_id"].to_numpy()])
    residual = np.linalg.norm((um[dst] - um[src]) - flow[src], axis=1)
    cost = p["weight"] * np.maximum(0.0, residual - p["knee_um"])
    cost[dividing[src]] = 0.0
    graph.add_edge_attr_key("flow_cost", pl.Float64, 0.0)
    graph.update_edge_attrs(edge_ids=edges["edge_id"].to_list(),
                            attrs={"flow_cost": cost.tolist()})
    return graph


class HighsILPSolver(td.solvers.ILPSolver):
    # The tracking ILP's LP relaxation is almost integral, so the exact solve is restricted
    # to the constraint neighbourhood of the fractional variables; the rest keeps its LP value.
    def __init__(self, *, hops, **kwargs):
        super().__init__(**kwargs)
        self.hops = hops

    def _solve(self):
        if self._count == 0 or len(self._edge_vars) == 0:
            raise ValueError("Empty ILPSolver model, there is nothing to solve.")
        n = self._count
        c = np.array(list(self._objective), dtype=float)
        rows, cols, vals, lo, hi = [], [], [], [], []
        for r, con in enumerate(self._constraints):
            for idx, coef in con.get_coefficients().items():
                rows.append(r); cols.append(idx); vals.append(coef)
            rel, val = con.get_relation(), con.get_value()
            lo.append(-np.inf if rel == Relation.LessEqual else val)
            hi.append(np.inf if rel == Relation.GreaterEqual else val)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(lo), n))
        cons = LinearConstraint(A, lo, hi)
        x = milp(c, constraints=cons, integrality=np.zeros(n), bounds=Bounds(0, 1)).x
        free = np.abs(x - np.round(x)) >= 1e-6
        if not free.any():
            return np.round(x)
        At = A.T.tocsr()
        for _ in range(self.hops):
            touched = np.unique(A[:, np.flatnonzero(free)].nonzero()[0])
            free[np.unique(At[:, touched].nonzero()[0])] = True
        bounds = Bounds(np.where(free, 0.0, np.round(x)), np.where(free, 1.0, np.round(x)))
        exact = {"mip_rel_gap": 0.0}  # scipy's default 1e-4 gap would accept suboptimal solutions
        result = milp(c, constraints=cons, integrality=np.ones(n), bounds=bounds, options=exact)
        if result.x is None:
            result = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1), options=exact)
        return np.round(result.x)


def make_ilp_solver(cfg, **weights):
    if cfg.ilp_solver == "highs":
        return HighsILPSolver(hops=cfg.ilp_relax_hops, num_threads=1, **weights)
    if cfg.ilp_solver == "scip":
        return td.solvers.ILPSolver(num_threads=1, **weights)
    raise ValueError(f"Unknown ilp_solver: {cfg.ilp_solver}")


def solve_ilp(cfg, graph):
    # Per-frame-pair scores are independent and conflict globally (two cells claiming one
    # target, tracks breaking and rejoining). The ILP resolves the whole video at once.
    if graph.num_edges() == 0:
        return graph
    appearance_weight = (
        cfg.ilp_appearance_weight
        + cfg.ilp_appearance_scale * cfg.ilp_edge_weight
        * td.NodeAttr("appearance_prob")
    )
    division_weight = (
        cfg.ilp_division_weight
        + cfg.ilp_division_scale * (0.5 - td.NodeAttr("division_prob"))
    )

    edge_weight = cfg.ilp_edge_weight * td.EdgeAttr("edge_prob")
    if cfg.idsim["enable"]:
        edge_weight = edge_weight + cfg.idsim["weight"] * td.EdgeAttr("id_dissim")
    if cfg.flow_cost["enable"]:
        edge_weight = edge_weight + 1.0 * td.EdgeAttr("flow_cost")

    solver = make_ilp_solver(
        cfg,
        edge_weight=edge_weight,
        node_weight=-cfg.ilp_node_weight * (
            td.NodeAttr("node_prob") - cfg.ilp_node_bias
        ),
        appearance_weight=appearance_weight,
        disappearance_weight=cfg.ilp_disappearance_weight,
        division_weight=division_weight,
    )
    # ILPSolver returns a GraphView. The official division metric copies the
    # prediction when forks are present, and GraphView.copy() is unsupported.
    # Detaching preserves the selected nodes/edges in an independent graph.
    solved = solver.solve(graph).detach()
    if cfg.flow_cost["enable"]:
        # solve-time attribute only; later steps add edges without it
        solved.remove_edge_attr_key("flow_cost")
    return solved


def filter_track_graph(cfg, reference, graph, reference_peaks, peaks):
    # Prune the low-threshold solution against the baseline one. Shared detector
    # peak IDs identify anchors across the two independently solved graphs; the
    # baseline is never repaired and model attributes are never changed.
    from collections import defaultdict

    graph = graph.copy()
    reference_nodes = reference.node_ids()
    anchors_by_peak = {int(reference_peaks[n]) for n in reference_nodes}
    reference_forks = {int(reference_peaks[n]) for n in reference_nodes if reference.out_degree(n) == 2}
    reference_edges = {(int(reference_peaks[a]), int(reference_peaks[b]))
                       for a, b in reference.edge_list()}
    children = defaultdict(list)
    edge_rows = {}
    for row in graph.edge_attrs().to_dicts():
        a, b = int(row["source_id"]), int(row["target_id"])
        children[a].append(b)
        edge_rows[a, b] = row
    idsim_weight = cfg.idsim["weight"] if cfg.idsim["enable"] else 0.0
    removed_edges = []
    for a, targets in children.items():
        if len(targets) < 2 or int(peaks[a]) in reference_forks:
            continue
        original = [b for b in targets if (int(peaks[a]), int(peaks[b])) in reference_edges]
        keep = original[0] if original else min(
            targets, key=lambda b: (cfg.ilp_edge_weight * edge_rows[a, b]["edge_prob"]
                                   + idsim_weight * edge_rows[a, b]["id_dissim"], int(peaks[b])))
        removed_edges.extend(edge_rows[a, b]["edge_id"] for b in targets if b != keep)
    if removed_edges:
        graph.bulk_remove_edges(removed_edges)

    if cfg.track_postprocess["mode"] == "anchor_paths":
        children, parents = {}, {}
        for a, b in graph.edge_list():
            children.setdefault(int(a), []).append(int(b))
            parents[int(b)] = int(a)
        anchors = {int(n) for n in graph.node_ids() if int(peaks[n]) in anchors_by_peak}
        extras = set(map(int, graph.node_ids())) - anchors
        removed_nodes = []
        for first in sorted(extras):
            if parents.get(first) in extras:
                continue
            path = [first]
            while children.get(path[-1]) and children[path[-1]][0] in extras:
                path.append(children[path[-1]][0])
            right = children.get(path[-1], [])
            if parents.get(first) not in anchors or not right or right[0] not in anchors:
                removed_nodes.extend(path)
        if removed_nodes:
            graph.bulk_remove_nodes(removed_nodes)
    return graph


def smooth_track_coords(cfg, graph):
    # Smooth each track's coordinates, optionally removing/restoring common
    # frame motion so real collective displacement is not smoothed away.
    # Only coordinates change; selected nodes and edges stay the same.
    if not cfg.smooth_window or graph.num_nodes() == 0:
        return graph

    ids = list(graph.node_ids())
    df = graph.node_attrs()
    pos = {int(v): k for k, v in enumerate(ids)}
    coord = np.empty((len(ids), 4), dtype=np.float64)
    for r in df.iter_rows(named=True):
        k = pos[int(r["node_id"])]
        coord[k] = (r["t"], r["z"], r["y"], r["x"])

    nxt, prv = {}, {}
    for a, b in graph.edge_list():
        nxt.setdefault(pos[int(a)], []).append(pos[int(b)])
        prv.setdefault(pos[int(b)], []).append(pos[int(a)])

    motion = np.zeros_like(coord[:, 1:4])
    if cfg.smooth_preserve_motion:
        displacements = {}
        for source, targets in nxt.items():
            for target in targets:
                if any(len(links.get(node, [])) != 1
                       for node in (source, target) for links in (prv, nxt)):
                    continue
                if coord[target, 0] - coord[source, 0] == 1:
                    frame = int(coord[target, 0])
                    displacements.setdefault(frame, []).append(
                        coord[target, 1:4] - coord[source, 1:4])
        if displacements:
            frames = sorted(displacements)
            velocity = np.array([np.median(displacements[t], axis=0) for t in frames])
            times = coord[:, 0].astype(int)
            first, last = times.min(), times.max()
            steps = np.arange(first + 1, last + 1)
            # Interpolate unsupported steps; extend nearest velocity at boundaries.
            velocity = np.column_stack([
                np.interp(steps, frames, velocity[:, axis]) for axis in range(3)])
            offsets = np.vstack([np.zeros(3), velocity.cumsum(axis=0)])
            motion = offsets[times - first]
            coord[:, 1:4] -= motion

    w, win = cfg.smooth_weight, cfg.smooth_window
    out = coord.copy()
    for i in range(len(ids)):
        chain, c = [i], i
        for _ in range(win):
            p = prv.get(c)
            if not p or len(p) != 1:
                break
            c = p[0]; chain.append(c)
        c = i
        for _ in range(win):
            n = nxt.get(c)
            if not n or len(n) != 1:
                break
            c = n[0]; chain.append(c)
        if len(chain) < 3:
            continue
        ts = coord[chain, 0] - coord[i, 0]
        tbar = ts.mean()
        dt = ts - tbar
        den = float(dt @ dt)
        if den == 0.0:
            continue
        Y = coord[chain, 1:4]
        ybar = Y.mean(axis=0)
        fit = ybar - (dt @ (Y - ybar)) / den * tbar     # least-squares line evaluated at dt=0
        out[i, 1:4] = (1 - w) * coord[i, 1:4] + w * fit

    out[:, 1:4] += motion
    graph.update_node_attrs(
        node_ids=ids,
        attrs={"z": out[:, 1].tolist(), "y": out[:, 2].tolist(), "x": out[:, 3].tolist()},
    )
    return graph


def fill_track_gaps(cfg, graph, scale):
    """Fill one missing frame: accept direction-consistent gaps, then motion-only gaps."""
    p = cfg.gap_fill
    if graph.num_edges() == 0:
        return graph
    from scipy.spatial import cKDTree

    nodes = graph.node_attrs()
    ids = nodes["node_id"].to_numpy()
    t = nodes["t"].to_numpy()
    scale = np.asarray(scale, dtype=np.float64)
    um = np.column_stack([nodes[a].to_numpy() for a in "zyx"]) * scale

    n = len(ids)
    row = {int(v): i for i, v in enumerate(ids)}
    parent = np.full(n, -1)
    child = np.full(n, -1)
    n_in = np.zeros(n, dtype=int)
    n_out = np.zeros(n, dtype=int)
    for a, b in graph.edge_list():
        i, j = row[int(a)], row[int(b)]
        n_out[i] += 1
        n_in[j] += 1
        parent[j] = i
        child[i] = j
    ends = np.flatnonzero((n_out == 0) & (n_in == 1))
    starts = np.flatnonzero((n_in == 0) & (n_out == 1))
    frame_um = {f: cKDTree(um[t == f]) for f in np.unique(t)}

    # Freeze candidates before modifying the graph. Distances are in micrometres.
    proposals = []
    for f in np.unique(t[ends]):
        e_at = ends[t[ends] == f]
        s_at = starts[t[starts] == f + 2]
        if not len(s_at):
            continue
        tree = cKDTree(um[s_at])
        for e in e_at:
            for j in tree.query_ball_point(um[e], p["gate_um"]):
                s = s_at[j]
                step = (um[s] - um[e]) / 2
                before = um[e] - um[parent[e]]
                after = um[child[s]] - um[s]
                residual = max(np.linalg.norm(step - before),
                               np.linalg.norm(step - after))
                if residual > p["velocity_residual_um"]:
                    continue
                mid = (um[e] + um[s]) / 2
                occupied = frame_um.get(f + 1)
                if occupied is not None and occupied.query(mid)[0] < p["occupied_um"]:
                    continue
                aligned = step @ before >= 0 and step @ after >= 0
                proposals.append((residual + 0.1 * np.linalg.norm(step),
                                  int(ids[e]), int(ids[s]), e, s, aligned))
    proposals.sort()

    used = set()
    added = {}
    for strict in (True, False):
        for _, source, target, e, s, aligned in proposals:
            if source in used or target in used or (strict and not aligned):
                continue
            f = int(t[e]) + 1
            mid_um = (um[e] + um[s]) / 2
            # Keep the validated first-pass choices unchanged. Only extensions
            # are checked against all synthesized points, including first-pass ones.
            if not strict and any(np.linalg.norm(mid_um - q) < p["occupied_um"]
                                  for q in added.get(f, [])):
                continue
            attrs = nodes.row(int(e), named=True)
            attrs.pop("node_id")
            mid = mid_um / scale
            new_id = graph.add_node({
                **attrs,
                "t": f,
                "z": float(mid[0]),
                "y": float(mid[1]),
                "x": float(mid[2]),
            })
            for a, b in ((source, new_id), (new_id, target)):
                graph.add_edge(a, b, {
                    "solution": True, "edge_prob": float("nan"),
                    "edge_dist": float(np.linalg.norm(um[s] - um[e]) / 2),
                    "id_dissim": float("nan"),
                })
            used.update((source, target))
            added.setdefault(f, []).append(mid_um)
    return graph


# ── submission ───────────────────────────────────────────────────────────────────
def write_submission(cfg, ids, path=None):
    # node rows carry coordinates, edge rows reference node_id via source/target.
    path = path or cfg.submission_csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_id = n_node = n_edge = 0

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for sample_id in ids:
            graph = td.graph.IndexedRXGraph.from_geff(
                os.path.join(cfg.predict_dir, f"{sample_id}.geff"))[0]
            node_row = list(graph.node_attrs().iter_rows(named=True))
            edge_row = list(graph.edge_attrs().iter_rows(named=True))
            node_id = {int(r["node_id"]) for r in node_row}
            assert node_id, f"{sample_id}: empty graph"

            for r in sorted(node_row, key=lambda x: int(x["node_id"])):
                writer.writerow({
                    "id": row_id, "dataset": sample_id, "row_type": "node",
                    # Float, not int: the metric matches nodes by Euclidean distance, and
                    # rounding to the full-resolution grid quantises z at 1.625 um, which
                    # gives back 0.0074 of the 0.0246 that smoothing wins on the holdout.
                    # 190 of the cached public kernels write floats here, 75 round.
                    "node_id": int(r["node_id"]), "t": int(r["t"]),
                    "z": max(0.0, round(float(r["z"]), 3)),
                    "y": max(0.0, round(float(r["y"]), 3)),
                    "x": max(0.0, round(float(r["x"]), 3)),
                    "source_id": -1, "target_id": -1,
                })
                row_id += 1

            for r in edge_row:
                s, t = int(r["source_id"]), int(r["target_id"])
                assert s in node_id and t in node_id, f"{sample_id}: dangling edge {s}->{t}"
                writer.writerow({
                    "id": row_id, "dataset": sample_id, "row_type": "edge", "node_id": -1,
                    "t": -1, "z": -1, "y": -1, "x": -1, "source_id": s, "target_id": t,
                })
                row_id += 1

            n_node += len(node_row)
            n_edge += len(edge_row)

    print(f"submission: {path} | nodes={n_node} edges={n_edge}")
    return n_node, n_edge


# ── training validation ────────────────────────────────────────────────────────
def _parent_top1_counts(logits, target, mask_t, mask_t1):
    """Count correct top-1 parents among targets whose GT edge is a candidate."""
    logits = logits.masked_fill(~mask_t[:, :, None], -1e4).transpose(1, 2)
    targets = target.transpose(1, 2)
    ask = (targets.sum(-1) == 1) & mask_t1
    if not bool(ask.any()):
        return 0, 0

    predicted = logits.argmax(-1)
    expected = targets.argmax(-1)
    parent_logit = logits.gather(-1, expected[..., None]).squeeze(-1)
    candidate = ask & (parent_logit > -1e3)
    return (
        int((predicted[candidate] == expected[candidate]).sum()),
        int(candidate.sum()),
    )


def _candidate_parent_counts(logits, target, mask_t1):
    """Count detected GT parent edges present in the linker's candidate set."""
    targets = target.transpose(1, 2)
    ask = (targets.sum(-1) == 1) & mask_t1
    if not bool(ask.any()):
        return 0, 0

    expected = targets.argmax(-1)
    parent_logit = logits.transpose(1, 2).gather(
        -1, expected[..., None]).squeeze(-1)
    present = parent_logit > -1e3
    return int(present[ask].sum()), int(ask.sum())


def _binary_f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else float("nan")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    loss_fn,
    training_ops,
    pool_kernel_um=5.0,
):
    """Evaluate detection, identity, linking and division."""
    model.eval()
    edge_loss_sum, n_pairs = 0.0, 0
    det_loss_sum, det_samples = 0.0, 0
    id_loss_sum, id_targets = 0.0, 0
    division_loss_sum = 0.0
    division_targets = 0
    division_tp = division_fp = division_fn = 0
    e2e_division_tp = e2e_division_fn = 0
    parent_correct = parent_total = 0
    candidate_parent_found = candidate_parent_total = 0
    gt_parent_total = 0
    gt_matched, gt_total = 0, 0

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)

        batch_size, n_frames = imgs.shape[:2]
        unet_out, det_logits = model.encode(imgs)
        frames = []
        for i in range(n_frames):
            det_loss = loss_fn.detection(
                det_logits[i], coords[:, i], masks[:, i]
            )
            det_loss_sum += float(det_loss) * batch_size
            det_samples += batch_size

            det_c, det_p, det_m, matches = training_ops.detect_and_match(
                det_logits[i], coords[:, i], masks[:, i], image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i,
                window_size=n_frames,
            )
            features = model._index_features(unet_out[:, i], det_c, det_m)
            frames.append((det_c, det_p, det_m, matches, features))

            for b in range(batch_size):
                gt_total += int(masks[b, i].sum().item())
                gt_matched += int((matches[b] >= 0).sum().item())

        for i in range(n_frames - 1):
            gt_parent_total += int(targets[:, i].sum().item())
            src, tgt = frames[i], frames[i + 1]
            edge_target = training_ops.build_matched_edge_targets(
                src[3], tgt[3], targets[:, i],
                src[0].shape[1], tgt[0].shape[1],
            )
            link = model.predict_links(
                src[4], tgt[4],
                network_to_image_coordinates(src[0], ds_scale),
                network_to_image_coordinates(tgt[0], ds_scale),
                src[1], tgt[1], src[2], tgt[2],
            )
            edge_logits = link.edge

            division_loss, n_div, tp, fp, fn = loss_fn.division(
                link.division, targets[:, i], src[3])
            division_loss_sum += float(division_loss) * n_div
            division_targets += n_div
            division_tp += tp
            division_fp += fp
            division_fn += fn

            # The head metric above is conditional on detecting the GT parent.
            # Count missed parents as false negatives for an end-to-end metric.
            for b, matches in enumerate(src[3]):
                n_detected = len(matches)
                detected_gt = matches.to(link.division.device).long()
                matched = detected_gt >= 0
                predicted = link.division[b, :n_detected].sigmoid() >= 0.5
                gt_division = (
                    targets[b, i].sum(-1) == 2
                ) & masks[b, i]

                matched_gt = detected_gt[matched]
                matched_predictions = predicted[matched]
                matched_labels = gt_division[matched_gt]
                true_positives = int(
                    (matched_predictions & matched_labels).sum())
                e2e_division_tp += true_positives
                e2e_division_fn += int(gt_division.sum()) - int(
                    true_positives)

            correct, total = _parent_top1_counts(
                edge_logits, edge_target, src[2], tgt[2])
            parent_correct += correct
            parent_total += total

            found, total = _candidate_parent_counts(
                edge_logits, edge_target, tgt[2])
            candidate_parent_found += found
            candidate_parent_total += total

            id_src = model.identity_embedding(src[4])
            id_tgt = model.identity_embedding(tgt[4])
            id_loss, n_id = loss_fn.identity(
                id_src, id_tgt, src[2], tgt[2], edge_target
            )
            id_loss_sum += float(id_loss) * n_id
            id_targets += n_id

            for b in range(batch_size):
                n_src = int(src[2][b].sum().item())
                n_tgt = int(tgt[2][b].sum().item())
                loss, correct, total = training_ops._evaluate_pair(
                    edge_logits[b, :n_src, :n_tgt],
                    edge_target[b, :n_src, :n_tgt],
                )
                edge_loss_sum += loss
                n_pairs += 1

    parent_top1 = (
        parent_correct / parent_total if parent_total else float("nan"))
    node_tp = gt_matched
    node_fn = max(gt_total - node_tp, 0)
    node_recall = node_tp / gt_total if gt_total else float("nan")
    parent_fail_node = max(gt_parent_total - candidate_parent_total, 0)
    parent_fail_candidate = max(candidate_parent_total - candidate_parent_found, 0)
    parent_fail_linker = max(candidate_parent_found - parent_correct, 0)
    return {
        "edge_loss": edge_loss_sum / max(n_pairs, 1),
        "det_loss": det_loss_sum / max(det_samples, 1),
        "id_loss": id_loss_sum / id_targets if id_targets else float("nan"),
        "division_loss": (
            division_loss_sum / division_targets if division_targets else float("nan")),
        "node_recall": node_recall,
        "node_tp": node_tp,
        "node_fn": node_fn,
        "node_gt": gt_total,
        "parent_top1": parent_top1,
        "candidate_parent_recall": (
            candidate_parent_found / candidate_parent_total
            if candidate_parent_total else float("nan")),
        "e2e_parent_recall": (
            parent_correct / gt_parent_total
            if gt_parent_total else float("nan")),
        "parent_correct": parent_correct,
        "parent_total": gt_parent_total,
        "parent_fail_node": parent_fail_node,
        "parent_fail_candidate": parent_fail_candidate,
        "parent_fail_linker": parent_fail_linker,
        "division_precision": (
            division_tp / (division_tp + division_fp)
            if division_tp + division_fp else float("nan")),
        "division_recall": (
            division_tp / (division_tp + division_fn)
            if division_tp + division_fn else float("nan")),
        "division_f1": _binary_f1(
            division_tp, division_fp, division_fn),
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "division_gt": division_tp + division_fn,
        "e2e_division_recall": (
            e2e_division_tp / (e2e_division_tp + e2e_division_fn)
            if e2e_division_tp + e2e_division_fn else float("nan")),
        "e2e_division_tp": e2e_division_tp,
        "e2e_division_fn": e2e_division_fn,
        "e2e_division_gt": e2e_division_tp + e2e_division_fn,
    }


# ── metrics ──────────────────────────────────────────────────────────────────────
def score_clip(cfg, pred_graph, truth_graph, scale, n_total):
    # Official metric: edge Jaccard with a node-count penalty, plus 0.1 * division Jaccard.
    import warnings

    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Predicted graph has no edges or no nodes.*"
        )
        er = evaluate(
            pred_graph, truth_graph,
            scale=scale, max_distance=cfg.max_distance,
        )
    if pred_graph.num_nodes() == 0:
        recall = 0.0
    else:
        # The official evaluator skips matching when the prediction has no edges.
        # Match its nodes separately so detection recall remains meaningful.
        if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID not in pred_graph.node_attr_keys():
            from tracksdata.metrics import DistanceMatching
            from tracksdata.options import get_options, set_options

            show_progress = get_options().show_progress
            set_options(show_progress=False)
            try:
                pred_graph.match(
                    truth_graph,
                    matching=DistanceMatching(
                        max_distance=cfg.max_distance, scale=scale,
                    ),
                )
            finally:
                set_options(show_progress=show_progress)
        recall = node_recall(pred_graph, truth_graph)
    row = per_sample_metrics(er=er, n_total=n_total, node_recall=recall)
    return er, row


def summarise_scores(rows):
    from tracking_cellmot.metrics import summarise
    return summarise(rows)


def summarise_competition(rows):
    summary = summarise_scores(rows)
    ratios = [
        row["total_node_ratio"] for row in rows
        if row["total_node_ratio"] == row["total_node_ratio"]
    ]
    summary["node_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
    valid = [row for row in rows if row["edge_tp"] == row["edge_tp"]]
    for kind in ("edge", "division"):
        tp = sum(int(row[f"{kind}_tp"]) for row in valid)
        fp = sum(int(row[f"{kind}_fp"]) for row in valid)
        fn = sum(int(row[f"{kind}_fn"]) for row in valid)
        summary.update({
            f"{kind}_tp": tp,
            f"{kind}_fp": fp,
            f"{kind}_fn": fn,
            f"{kind}_precision": tp / (tp + fp) if tp + fp else float("nan"),
            f"{kind}_recall": tp / (tp + fn) if tp + fn else float("nan"),
            f"{kind}_f1": (
                2 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn else float("nan")
            ),
        })
    return summary


def save_result(cfg, metrics):
    os.makedirs(cfg.results_dir, exist_ok=True)
    rec = {"run": cfg.run, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}}
    path = os.path.join(cfg.results_dir, "result.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"result: {path}")
    return rec
