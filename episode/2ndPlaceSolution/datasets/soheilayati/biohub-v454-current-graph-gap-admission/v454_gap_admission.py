"""Runtime-safe V454 current-graph one-frame gap admission."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.tracker.postpass33 import changed_hop_guard


FEATURES = [
    "agreement_um", "endpoint_distance_um", "parent_speed_um",
    "child_speed_um", "raw_center_contrast", "raw_peak_contrast",
    "raw_peak_snr", "raw_peak_offset",
]
RADIUS_UM = 3.0
PATCH_SIZE = 96
Z_RADIUS = 1


def load_assets(root: Path) -> dict:
    metadata = json.loads((Path(root) / "metadata.json").read_text(encoding="utf-8"))
    if metadata["version"] != "v454_current_graph_gap_admission_v1":
        raise RuntimeError("unexpected V454 metadata version")
    if metadata["features"] != FEATURES or float(metadata["threshold"]) != 0.65:
        raise RuntimeError("V454 feature/threshold contract drift")
    return metadata


def _supported(index: int, direction: np.ndarray, times: np.ndarray) -> bool:
    start = index
    for step in range(2):
        index = int(direction[index])
        if index < 0 or abs(int(times[index]) - int(times[start])) != step + 1:
            return False
    return True


def _proposal_rows(graph: DatasetGraph, motion_ids: np.ndarray,
                   backward: np.ndarray, forward: np.ndarray) -> list[dict]:
    ids = np.asarray(list(graph.nodes), np.int64)
    nodes = np.asarray(list(graph.nodes.values()), np.int64).reshape(-1, 4)
    remap = {int(node_id): i for i, node_id in enumerate(ids)}
    edge_rows = np.asarray([(remap[p], remap[c]) for p, c, _ in graph.edges], np.int64).reshape(-1, 2)
    pi, ci = edge_rows.T
    indegree = np.bincount(ci, minlength=len(ids))
    outdegree = np.bincount(pi, minlength=len(ids))
    prev = np.full(len(ids), -1, np.int64)
    nxt = np.full(len(ids), -1, np.int64)
    prev[ci], nxt[pi] = pi, ci
    guard = changed_hop_guard(len(ids), pi, ci, np.arange(len(edge_rows)), hops=2)

    motion_ids = np.asarray(motion_ids, np.int64)
    backward = np.asarray(backward, np.float64)
    forward = np.asarray(forward, np.float64)
    if backward.shape != (len(motion_ids), 3) or forward.shape != backward.shape:
        raise RuntimeError("V454 motion array shape mismatch")
    motion_row = {int(node_id): i for i, node_id in enumerate(motion_ids)}
    has_motion = np.asarray([int(node_id) in motion_row for node_id in ids])
    times = nodes[:, 0]
    positions = nodes[:, 1:] * SCALE_ARR
    terminals = [
        i for i in range(len(ids))
        if outdegree[i] == 0 and not guard[i] and has_motion[i] and _supported(i, prev, times)
    ]
    roots = [
        i for i in range(len(ids))
        if indegree[i] == 0 and not guard[i] and has_motion[i] and _supported(i, nxt, times)
    ]
    roots_by_time: dict[int, list[int]] = defaultdict(list)
    for row in roots:
        roots_by_time[int(times[row])].append(row)
    old_by_time = {int(t): cKDTree(positions[times == t]) for t in np.unique(times)}
    shape = np.asarray(graph.shape[1:], np.int64)
    proposals = []
    for parent in terminals:
        target_t = int(times[parent]) + 2
        children = roots_by_time.get(target_t, [])
        if not children:
            continue
        pm = motion_row[int(ids[parent])]
        predicted_forward = positions[parent] + forward[pm]
        child_previous = np.asarray([
            positions[child] + backward[motion_row[int(ids[child])]] for child in children
        ])
        tree = cKDTree(child_previous)
        for j in tree.query_ball_point(predicted_forward, RADIUS_UM):
            child = int(children[j])
            agreement = float(np.linalg.norm(predicted_forward - child_previous[j]))
            midpoint_um = 0.5 * (predicted_forward + child_previous[j])
            coord = np.rint(midpoint_um / SCALE_ARR).astype(np.int64)
            coord = np.clip(coord, 0, shape - 1)
            actual_um = coord * SCALE_ARR
            novelty = float(old_by_time[target_t - 1].query(actual_um)[0]) if target_t - 1 in old_by_time else float("inf")
            if novelty <= RADIUS_UM:
                continue
            cm = motion_row[int(ids[child])]
            proposals.append({
                "parent": int(ids[parent]), "child": int(ids[child]),
                "t": target_t - 1, "z": int(coord[0]), "y": int(coord[1]), "x": int(coord[2]),
                "agreement_um": agreement,
                "endpoint_distance_um": float(np.linalg.norm(positions[child] - positions[parent])),
                "parent_speed_um": float(np.linalg.norm(forward[pm])),
                "child_speed_um": float(np.linalg.norm(backward[cm])),
                "novelty_um": novelty,
            })
    proposals.sort(key=lambda r: (r["agreement_um"], r["parent"], r["child"], r["z"], r["y"], r["x"]))
    best_parent, best_child = {}, {}
    for i, row in enumerate(proposals):
        best_parent.setdefault(row["parent"], i)
        best_child.setdefault(row["child"], i)
    selected, added = [], []
    for i, row in enumerate(proposals):
        if best_parent[row["parent"]] != i or best_child[row["child"]] != i:
            continue
        point = np.asarray([row["z"], row["y"], row["x"]]) * SCALE_ARR
        if any(np.linalg.norm(point - other) <= RADIUS_UM for other in added):
            continue
        added.append(point)
        selected.append(row)
    return selected


def _patch(raw_volume, t: int, z: int, y: int, x: int) -> np.ndarray:
    z = int(np.clip(z, 0, raw_volume.shape[1] - 1))
    slab = np.asarray(raw_volume[t, max(0, z-Z_RADIUS):min(raw_volume.shape[1], z+Z_RADIUS+1)], dtype=np.float32).max(axis=0)
    half, pad = PATCH_SIZE // 2, PATCH_SIZE // 2 + 2
    slab = np.pad(slab, ((pad, pad), (pad, pad)), mode="reflect")
    y, x = int(y + pad), int(x + pad)
    return slab[y-half:y+half, x-half:x+half]


def _patch_stats(x: np.ndarray) -> dict[str, float]:
    h, w = x.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    core, near, shell = rr <= 3, rr <= 5, (rr >= 7) & (rr <= 13)
    smooth = gaussian_filter(np.asarray(x, np.float32), 1.0, mode="nearest")
    background = float(np.median(smooth[shell]))
    scale = max(float(np.percentile(smooth, 95) - np.percentile(smooth, 5)), 1e-5)
    peak_at = np.unravel_index(int(np.argmax(np.where(near, smooth, -np.inf))), smooth.shape)
    peak = float(smooth[peak_at])
    mad = float(np.median(np.abs(smooth[shell] - background))) * 1.4826
    return {
        "raw_center_contrast": (float(smooth[core].mean()) - background) / scale,
        "raw_peak_contrast": (peak - background) / scale,
        "raw_peak_snr": (peak - background) / max(mad, 1e-4),
        "raw_peak_offset": float(math.hypot(peak_at[0] - cy, peak_at[1] - cx)),
    }


def _probabilities(frame: pd.DataFrame, metadata: dict) -> np.ndarray:
    values = frame[FEATURES].to_numpy(np.float64, copy=True)
    missing = ~np.isfinite(values)
    medians = np.asarray(metadata["imputer_statistics"], np.float64)
    values[missing] = np.take(medians, np.nonzero(missing)[1])
    values = (values - np.asarray(metadata["scaler_mean"], np.float64)) / np.asarray(metadata["scaler_scale"], np.float64)
    logits = values @ np.asarray(metadata["logistic_coef"], np.float64) + float(metadata["logistic_intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def apply_gap_admission(graph: DatasetGraph, motion_ids: np.ndarray,
                        backward: np.ndarray, forward: np.ndarray,
                        zarr_path: Path, metadata: dict) -> tuple[DatasetGraph, dict]:
    proposals = _proposal_rows(graph, motion_ids, backward, forward)
    if not proposals:
        return graph, {"proposals": 0, "above_threshold": 0, "selected": 0}
    import zarr
    raw = zarr.open_array(str(Path(zarr_path) / "0"), mode="r")
    for row in proposals:
        row.update(_patch_stats(_patch(raw, row["t"], row["z"], row["y"], row["x"])))
    frame = pd.DataFrame(proposals)
    frame["selection_probability"] = _probabilities(frame, metadata)
    eligible = frame[frame.selection_probability.ge(float(metadata["threshold"]))]
    if eligible.empty:
        return graph, {"proposals": len(frame), "above_threshold": 0, "selected": 0}
    row = eligible.sort_values(
        ["selection_probability", "agreement_um", "parent", "child"],
        ascending=[False, True, True, True], kind="stable",
    ).iloc[0]
    nodes, edges = dict(graph.nodes), list(graph.edges)
    new_id = max(nodes, default=-1) + 1
    nodes[new_id] = (int(row.t), int(row.z), int(row.y), int(row.x))
    edges.extend([(int(row.parent), new_id, 0.30), (new_id, int(row.child), 0.30)])
    result = DatasetGraph(graph.dataset, nodes, edges, graph.shape)
    validate_graph(result)
    return result, {
        "proposals": len(frame), "above_threshold": len(eligible), "selected": 1,
        "selection_probability": float(row.selection_probability),
        "parent": int(row.parent), "child": int(row.child), "new_node_id": int(new_id),
        "coordinate_tzyx": [int(row.t), int(row.z), int(row.y), int(row.x)],
        "agreement_um": float(row.agreement_um), "endpoint_distance_um": float(row.endpoint_distance_um),
        "features": {name: float(row[name]) for name in FEATURES},
    }
