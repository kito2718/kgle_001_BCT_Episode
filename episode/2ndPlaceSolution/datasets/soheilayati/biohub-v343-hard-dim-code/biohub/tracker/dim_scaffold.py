"""Inference-only construction of V340 dim-specialist support scaffolds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from biohub.metrics.gt import SCALE_ARR
from biohub.tracker.dim_reranker import (
    DIM_MODEL_FEATURES,
    MODEL_FEATURES,
    bounded_selection_score,
    candidate_image_features,
    candidate_model_features,
)
from biohub.tracker.dim_union import (
    dim_union_superset,
    hard_gated_dim_union_superset,
)


def raw_flow_chain_state(candidate: pd.DataFrame) -> pd.DataFrame:
    """Recreate the frozen V248 label-free 3 um raw-candidate chain state."""
    times = candidate.t.to_numpy(np.int64)
    points_um = (
        candidate[["z", "y", "x"]].to_numpy(np.float64) * SCALE_ARR
    )
    flow_um = candidate[["flow_z", "flow_y", "flow_x"]].to_numpy(np.float64)
    score = candidate.score.to_numpy(np.float64)
    predecessor = np.full(len(times), -1, np.int32)
    successor = np.full(len(times), -1, np.int32)
    residual = np.full(len(times), np.inf)
    at = {int(frame): np.flatnonzero(times == frame) for frame in np.unique(times)}
    for frame, current in at.items():
        previous = at.get(frame - 1)
        if previous is None:
            continue
        projected = points_um[current] + flow_um[current]
        tree = cKDTree(points_um[previous])
        nominations = tree.query_ball_point(projected, r=3.0)
        parent = np.full(len(current), -1, np.int32)
        distance = np.full(len(current), np.inf)
        for index, choices in enumerate(nominations):
            if choices:
                choices = np.asarray(choices, dtype=np.int64)
                d = np.linalg.norm(
                    points_um[previous[choices]] - projected[index], axis=1
                )
                winner = np.lexsort((previous[choices], d))[0]
                parent[index] = previous[choices[winner]]
                distance[index] = d[winner]
        valid = np.flatnonzero(parent >= 0)
        order = valid[np.lexsort((current[valid], -score[current[valid]], distance[valid]))]
        for index in order:
            p, c = parent[index], current[index]
            if successor[p] < 0:
                predecessor[c], successor[p], residual[c] = p, c, distance[index]
    before = np.ones(len(times), np.int16)
    after = np.ones(len(times), np.int16)
    for frame in sorted(at):
        rows = at[frame]
        valid = rows[predecessor[rows] >= 0]
        before[valid] = before[predecessor[valid]] + 1
    for frame in sorted(at, reverse=True):
        rows = at[frame]
        valid = rows[successor[rows] >= 0]
        after[valid] = after[successor[valid]] + 1
    return pd.DataFrame({
        "predecessor": predecessor,
        "successor": successor,
        "residual_um": residual,
        "span": before + after - 1,
        "centered3": (before >= 2) & (after >= 2),
        "centered5": (before >= 3) & (after >= 3),
    })


def specialist_scaffold(
    volume: np.ndarray,
    candidate: pd.DataFrame,
    dim_model,
    cell_model,
    cap: int,
    max_rank_shift: int = 150,
    dim_region_threshold: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Attach V326 ranks and return V340's support-only candidate union."""
    state = raw_flow_chain_state(candidate)
    image = candidate_image_features(volume, candidate)
    features = candidate_model_features(candidate, state, image)
    dim_probability = np.asarray(
        dim_model.predict(features[list(DIM_MODEL_FEATURES)]), dtype=np.float32
    )
    cell_probability = np.asarray(
        cell_model.predict(features[list(MODEL_FEATURES)]), dtype=np.float32
    )
    dim_weight = dim_probability.copy()
    eligible = np.ones(len(candidate), dtype=bool)
    if dim_region_threshold is not None:
        if not 0.0 <= dim_region_threshold <= 1.0:
            raise ValueError("dim-region threshold must be in [0, 1]")
        eligible = dim_probability >= dim_region_threshold
        dim_weight[~eligible] = 0.0
    selection_score, soft_rank = bounded_selection_score(
        candidate, dim_weight, cell_probability, max_shift=max_rank_shift
    )
    out = candidate.copy()
    out["selection_score"] = selection_score
    out["solver_selection_score"] = out.score.to_numpy(np.float32)
    if dim_region_threshold is None:
        control, replacement, added, union = dim_union_superset(out, cap)
    else:
        control, replacement, added, union = hard_gated_dim_union_superset(
            out, cap, eligible
        )
    out.loc[added, "solver_selection_score"] = np.float32(-1.0e6)
    audit = {
        "raw_candidates": int(len(out)),
        "control_superset": int(len(control)),
        "replacement_superset": int(len(replacement)),
        "added_specialist_support": int(len(added)),
        "union_superset": int(len(union)),
        "added_solver_score": -1.0e6,
        "max_rank_shift": int(max_rank_shift),
        "dim_region_threshold": dim_region_threshold,
        "dim_region_candidates": int(eligible.sum()),
        "outside_region_rank_weight_exactly_zero": bool(
            dim_region_threshold is None or np.all(dim_weight[~eligible] == 0.0)
        ),
        "mean_dim_probability": float(dim_probability.mean()),
        "mean_cell_probability": float(cell_probability.mean()),
        "score_histogram_preserved": bool(all(
            np.array_equal(
                np.sort(part.score.to_numpy(np.float32)),
                np.sort(part.selection_score.to_numpy(np.float32)),
            )
            for _, part in out.groupby("t", sort=False)
        )),
    }
    return out, union, audit
