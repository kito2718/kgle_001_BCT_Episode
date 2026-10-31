"""Scale-normalized division geometry shared by external and local tracks.

Only lineage geometry is used here. Image intensities, optical flow, detector
scores, and embeddings are deliberately excluded because their scales and
semantics differ between the Linajea/ITEC packages and competition data.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HORIZONS = (1, 2, 3, 5, 8)
HISTORY_HORIZONS = (1, 2, 3, 5)


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def _cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    den = _norm(left) * _norm(right)
    return np.divide(
        (left * right).sum(axis=1), den,
        out=np.zeros(len(left), dtype=np.float64), where=den > 1e-12)


def _layers(first: np.ndarray, depth: int = 8) -> np.ndarray:
    out = np.full((len(first), depth), -1, dtype=np.int64)
    out[:, 0] = first
    for level in range(1, depth):
        prior = out[:, level - 1]
        valid = prior >= 0
        out[valid, level] = first[prior[valid]]
    return out


def shared_feature_columns() -> list[str]:
    """Return the order-invariant feature contract used for transfer."""
    columns = [
        "dist_min_norm", "dist_max_norm", "sep_norm", "sep_z_fraction",
        "mid_resid_norm", "parent_step_norm", "opp_cos", "disp_asym",
        "parent_hist_len8", "daughter_future_min8", "daughter_future_max8",
    ]
    for horizon in HISTORY_HORIZONS:
        prefix = f"ph{horizon}"
        columns.extend([
            f"{prefix}_ok", f"{prefix}_mid_res_norm",
            f"{prefix}_mid_cos", f"{prefix}_daughter_res_min_norm",
            f"{prefix}_daughter_res_max_norm",
        ])
    for horizon in HORIZONS:
        prefix = f"f{horizon}"
        columns.extend([
            f"{prefix}_both", f"{prefix}_sep_norm",
            f"{prefix}_sep_growth_norm", f"{prefix}_sep_slope_norm",
            f"{prefix}_sep_cos", f"{prefix}_step_cos",
            f"{prefix}_accel_min_norm", f"{prefix}_accel_max_norm",
            f"{prefix}_mid_hist_res_norm",
        ])
    return columns


FEATURE_COLUMNS = shared_feature_columns()


def _sequence_arrays(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    ordered = nodes.sort_values(["t", "node_id"], kind="stable").reset_index(drop=True)
    ids = ordered.node_id.to_numpy(np.int64)
    id_to_index = {int(node_id): index for index, node_id in enumerate(ids)}
    pos = ordered[["z", "y", "x"]].to_numpy(np.float64)
    prev = np.full(len(ordered), -1, dtype=np.int64)
    outgoing: list[list[int]] = [[] for _ in range(len(ordered))]
    edge_steps: list[float] = []
    for row in edges.itertuples(index=False):
        parent = id_to_index[int(row.source_id)]
        child = id_to_index[int(row.target_id)]
        prev[child] = parent
        outgoing[parent].append(child)
        edge_steps.append(float(np.linalg.norm(pos[child] - pos[parent])))
    nxt = np.full(len(ordered), -1, dtype=np.int64)
    for parent, children in enumerate(outgoing):
        if len(children) == 1:
            nxt[parent] = children[0]
    positive_steps = np.asarray(edge_steps, dtype=np.float64)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 1e-9)]
    if not len(positive_steps):
        raise ValueError("cannot normalize a lineage sequence without positive edge steps")
    return ordered, prev, nxt, np.asarray(outgoing, dtype=object), float(np.median(positive_steps))


def _features_from_indices(
    ordered: pd.DataFrame,
    prev: np.ndarray,
    nxt: np.ndarray,
    parent: np.ndarray,
    child1: np.ndarray,
    child2: np.ndarray,
    step_scale: float,
) -> pd.DataFrame:
    pos = ordered[["z", "y", "x"]].to_numpy(np.float64)
    ancestors = _layers(prev)
    descendants = _layers(nxt)
    p0, p1, p2 = pos[parent], pos[child1], pos[child2]
    v1, v2 = p1 - p0, p2 - p0
    d1, d2 = _norm(v1), _norm(v2)
    sep0 = p1 - p2
    sep0_norm = _norm(sep0)
    mid0 = 0.5 * (p1 + p2)
    parent_prev = prev[parent]
    parent_step = np.full(len(parent), np.nan, dtype=np.float64)
    has_parent = parent_prev >= 0
    parent_step[has_parent] = _norm(
        p0[has_parent] - pos[parent_prev[has_parent]])
    future1 = (descendants[child1] >= 0).sum(axis=1).astype(np.float64)
    future2 = (descendants[child2] >= 0).sum(axis=1).astype(np.float64)

    out: dict[str, np.ndarray] = {
        "dist_min_norm": np.minimum(d1, d2) / step_scale,
        "dist_max_norm": np.maximum(d1, d2) / step_scale,
        "sep_norm": sep0_norm / step_scale,
        "sep_z_fraction": np.divide(
            np.abs(sep0[:, 0]), sep0_norm,
            out=np.zeros(len(parent), dtype=np.float64), where=sep0_norm > 1e-12),
        "mid_resid_norm": _norm(mid0 - p0) / step_scale,
        "parent_step_norm": parent_step / step_scale,
        "opp_cos": _cos(v1, v2),
        "disp_asym": np.abs(d1 - d2) / np.maximum(d1 + d2, 1e-12),
        "parent_hist_len8": (ancestors[parent] >= 0).sum(axis=1).astype(np.float64),
        "daughter_future_min8": np.minimum(future1, future2),
        "daughter_future_max8": np.maximum(future1, future2),
    }

    for horizon in HISTORY_HORIZONS:
        ancestor = ancestors[parent, horizon - 1]
        valid = ancestor >= 0
        incoming = (p0 - pos[np.maximum(ancestor, 0)]) / float(horizon)
        r1, r2 = _norm(v1 - incoming), _norm(v2 - incoming)
        prefix = f"ph{horizon}"
        out[f"{prefix}_ok"] = valid.astype(np.float64)
        out[f"{prefix}_mid_res_norm"] = np.where(
            valid, _norm((mid0 - p0) - incoming) / step_scale, np.nan)
        out[f"{prefix}_mid_cos"] = np.where(
            valid, _cos(mid0 - p0, incoming), np.nan)
        out[f"{prefix}_daughter_res_min_norm"] = np.where(
            valid, np.minimum(r1, r2) / step_scale, np.nan)
        out[f"{prefix}_daughter_res_max_norm"] = np.where(
            valid, np.maximum(r1, r2) / step_scale, np.nan)

    for horizon in HORIZONS:
        g1 = descendants[child1, horizon - 1]
        g2 = descendants[child2, horizon - 1]
        both = (g1 >= 0) & (g2 >= 0)
        q1, q2 = pos[np.maximum(g1, 0)], pos[np.maximum(g2, 0)]
        future_v1 = (q1 - p1) / float(horizon)
        future_v2 = (q2 - p2) / float(horizon)
        sep = q1 - q2
        sep_norm = _norm(sep)
        midpoint_step = (0.5 * (q1 + q2) - p0) / float(horizon + 1)
        history_horizon = min(horizon, 5)
        ancestor = ancestors[parent, history_horizon - 1]
        has_history = ancestor >= 0
        incoming = (p0 - pos[np.maximum(ancestor, 0)]) / float(history_horizon)
        accel1, accel2 = _norm(future_v1 - v1), _norm(future_v2 - v2)
        prefix = f"f{horizon}"
        out[f"{prefix}_both"] = both.astype(np.float64)
        out[f"{prefix}_sep_norm"] = np.where(both, sep_norm / step_scale, np.nan)
        out[f"{prefix}_sep_growth_norm"] = np.where(
            both, (sep_norm - sep0_norm) / step_scale, np.nan)
        out[f"{prefix}_sep_slope_norm"] = np.where(
            both, (sep_norm - sep0_norm) / (horizon * step_scale), np.nan)
        out[f"{prefix}_sep_cos"] = np.where(both, _cos(sep0, sep), np.nan)
        out[f"{prefix}_step_cos"] = np.where(
            both, _cos(future_v1, future_v2), np.nan)
        out[f"{prefix}_accel_min_norm"] = np.where(
            both, np.minimum(accel1, accel2) / step_scale, np.nan)
        out[f"{prefix}_accel_max_norm"] = np.where(
            both, np.maximum(accel1, accel2) / step_scale, np.nan)
        out[f"{prefix}_mid_hist_res_norm"] = np.where(
            both & has_history, _norm(midpoint_step - incoming) / step_scale, np.nan)
    return pd.DataFrame(out, columns=FEATURE_COLUMNS, dtype=np.float32)


def features_for_hypotheses(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    hypotheses: pd.DataFrame,
    *,
    step_scale: float | None = None,
) -> pd.DataFrame:
    """Compute shared features for explicit parent/child ID triples.

    ``hypotheses`` must contain ``parent_id``, ``child1_id``, and
    ``child2_id``. All inputs describe one source sequence.
    """
    ordered, prev, nxt, _, inferred_scale = _sequence_arrays(nodes, edges)
    id_to_index = {
        int(node_id): index
        for index, node_id in enumerate(ordered.node_id.to_numpy(np.int64))
    }
    try:
        parent = np.asarray(
            [id_to_index[int(value)] for value in hypotheses.parent_id], dtype=np.int64)
        child1 = np.asarray(
            [id_to_index[int(value)] for value in hypotheses.child1_id], dtype=np.int64)
        child2 = np.asarray(
            [id_to_index[int(value)] for value in hypotheses.child2_id], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"hypothesis references a missing lineage node: {error}") from error
    return _features_from_indices(
        ordered, prev, nxt, parent, child1, child2,
        inferred_scale if step_scale is None else float(step_scale))


def build_manual_division_hypotheses(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    nearest_children: int = 8,
    radius_steps: float = 12.0,
    max_negative_pairs_per_parent: int = 12,
) -> pd.DataFrame:
    """Create hard division candidates from manually annotated lineages.

    Every gold two-child edge is retained. Negatives are the closest
    alternative pairs at the next frame for parents with a known outgoing
    edge. This avoids treating track endings as explicit non-divisions.
    """
    parts: list[pd.DataFrame] = []
    for (source, sequence), seq_nodes in nodes.groupby(["source", "sequence"], sort=True):
        seq_edges = edges[(edges.source == source) & (edges.sequence == sequence)]
        ordered, prev, nxt, outgoing, step_scale = _sequence_arrays(seq_nodes, seq_edges)
        positions = ordered[["z", "y", "x"]].to_numpy(np.float64)
        ids = ordered.node_id.to_numpy(np.int64)
        times = ordered.t.to_numpy(np.int32)
        frame_indices = {
            int(t): index.to_numpy(np.int64)
            for t, index in ordered.groupby("t", sort=False).groups.items()
        }
        trees = {t: cKDTree(positions[index]) for t, index in frame_indices.items()}
        candidates: list[tuple[int, int, int, int, int]] = []
        for parent, true_children_obj in enumerate(outgoing):
            true_children = [int(value) for value in true_children_obj]
            if not true_children:
                continue
            child_time = int(times[parent]) + 1
            pool = frame_indices.get(child_time)
            if pool is None or len(pool) < 2:
                continue
            k = min(int(nearest_children), len(pool))
            distance, local = trees[child_time].query(
                positions[parent], k=k,
                distance_upper_bound=radius_steps * step_scale)
            distance = np.atleast_1d(distance)
            local = np.atleast_1d(local)
            nearby = [int(pool[int(i)]) for d, i in zip(distance, local)
                      if np.isfinite(d) and int(i) < len(pool)]
            child_pool = sorted(set(nearby).union(true_children))
            if len(child_pool) < 2:
                continue
            positive = tuple(sorted(true_children)) if len(true_children) == 2 else None
            ranked: list[tuple[float, tuple[int, int], int]] = []
            for child1, child2 in combinations(child_pool, 2):
                label = int(positive is not None and (child1, child2) == positive)
                d1 = np.linalg.norm(positions[child1] - positions[parent])
                d2 = np.linalg.norm(positions[child2] - positions[parent])
                midpoint = 0.5 * (positions[child1] + positions[child2])
                difficulty = max(d1, d2) + 0.25 * np.linalg.norm(midpoint - positions[parent])
                ranked.append((float(difficulty), (child1, child2), label))
            ranked.sort(key=lambda item: (-item[2], item[0]))
            negative_count = 0
            for _, (child1, child2), label in ranked:
                if label == 0:
                    if negative_count >= max_negative_pairs_per_parent:
                        continue
                    negative_count += 1
                candidates.append((parent, child1, child2, label, len(true_children)))

        if not candidates:
            continue
        raw = np.asarray(candidates, dtype=np.int64)
        features = _features_from_indices(
            ordered, prev, nxt, raw[:, 0], raw[:, 1], raw[:, 2], step_scale)
        meta = pd.DataFrame({
            "source": source,
            "sequence": sequence,
            "parent_id": ids[raw[:, 0]],
            "child1_id": ids[raw[:, 1]],
            "child2_id": ids[raw[:, 2]],
            "t": times[raw[:, 0]],
            "label": raw[:, 3].astype(np.int8),
            "parent_out_degree": raw[:, 4].astype(np.int8),
            "step_scale": np.float32(step_scale),
            "label_provenance": "manual_lineage_gold",
        })
        parts.append(pd.concat([meta, features], axis=1))
    if not parts:
        return pd.DataFrame(columns=[
            "source", "sequence", "parent_id", "child1_id", "child2_id",
            "t", "label", "parent_out_degree", "step_scale",
            "label_provenance", *FEATURE_COLUMNS,
        ])
    return pd.concat(parts, ignore_index=True)


def local_claims_shared_features(
    pairs: pd.DataFrame,
    tracklets: pd.DataFrame,
) -> pd.DataFrame:
    """Map competition claims pairs onto the external feature contract."""
    key = ["dataset", "par", "c1", "c2"]
    joined = pairs.merge(tracklets, on=key, validate="one_to_one")
    scale_by_dataset = (joined.groupby("dataset", sort=False).par_step_um
                        .median().replace(0, np.nan))
    fallback = float(np.nanmedian(scale_by_dataset.to_numpy(np.float64)))
    scale = joined.dataset.map(scale_by_dataset).fillna(fallback).to_numpy(np.float64)
    d1 = joined.dist1_um.to_numpy(np.float64)
    d2 = joined.dist2_um.to_numpy(np.float64)
    out: dict[str, np.ndarray] = {
        "dist_min_norm": np.minimum(d1, d2) / scale,
        "dist_max_norm": np.maximum(d1, d2) / scale,
        "sep_norm": joined.sep_um.to_numpy(np.float64) / scale,
        "sep_z_fraction": np.abs(joined.sep_z_um.to_numpy(np.float64)) /
                          np.maximum(joined.sep_um.to_numpy(np.float64), 1e-12),
        "mid_resid_norm": joined.mid_resid_um.to_numpy(np.float64) / scale,
        "parent_step_norm": joined.par_step_um.to_numpy(np.float64) / scale,
        "opp_cos": joined.opp_cos.to_numpy(np.float64),
        "disp_asym": joined.disp_asym.to_numpy(np.float64),
        "parent_hist_len8": joined.lt_parent_hist_len8.to_numpy(np.float64),
        "daughter_future_min8": joined.lt_min_fut_len8.to_numpy(np.float64),
        "daughter_future_max8": np.maximum(
            joined.lt_d1_fut_len8.to_numpy(np.float64),
            joined.lt_d2_fut_len8.to_numpy(np.float64)),
    }
    for horizon in HISTORY_HORIZONS:
        prefix = f"ph{horizon}"
        r1 = joined[f"lt_ph{horizon}_d1_res"].to_numpy(np.float64)
        r2 = joined[f"lt_ph{horizon}_d2_res"].to_numpy(np.float64)
        out[f"{prefix}_ok"] = joined[f"lt_ph{horizon}_ok"].to_numpy(np.float64)
        out[f"{prefix}_mid_res_norm"] = (
            joined[f"lt_ph{horizon}_mid_res"].to_numpy(np.float64) / scale)
        out[f"{prefix}_mid_cos"] = joined[f"lt_ph{horizon}_mid_cos"].to_numpy(np.float64)
        out[f"{prefix}_daughter_res_min_norm"] = np.minimum(r1, r2) / scale
        out[f"{prefix}_daughter_res_max_norm"] = np.maximum(r1, r2) / scale
    for horizon in HORIZONS:
        prefix = f"f{horizon}"
        a1 = joined[f"lt_f{horizon}_d1_accel"].to_numpy(np.float64)
        a2 = joined[f"lt_f{horizon}_d2_accel"].to_numpy(np.float64)
        out[f"{prefix}_both"] = joined[f"lt_f{horizon}_both"].to_numpy(np.float64)
        out[f"{prefix}_sep_norm"] = joined[f"lt_f{horizon}_sep"].to_numpy(np.float64) / scale
        out[f"{prefix}_sep_growth_norm"] = (
            joined[f"lt_f{horizon}_sep_growth"].to_numpy(np.float64) / scale)
        out[f"{prefix}_sep_slope_norm"] = (
            joined[f"lt_f{horizon}_sep_slope"].to_numpy(np.float64) / scale)
        out[f"{prefix}_sep_cos"] = joined[f"lt_f{horizon}_sep_cos"].to_numpy(np.float64)
        out[f"{prefix}_step_cos"] = joined[f"lt_f{horizon}_step_cos"].to_numpy(np.float64)
        out[f"{prefix}_accel_min_norm"] = np.minimum(a1, a2) / scale
        out[f"{prefix}_accel_max_norm"] = np.maximum(a1, a2) / scale
        out[f"{prefix}_mid_hist_res_norm"] = (
            joined[f"lt_f{horizon}_mid_hist_res"].to_numpy(np.float64) / scale)
    feature_frame = pd.DataFrame(out, columns=FEATURE_COLUMNS, dtype=np.float32)
    return pd.concat([joined[key].reset_index(drop=True), feature_frame], axis=1)
