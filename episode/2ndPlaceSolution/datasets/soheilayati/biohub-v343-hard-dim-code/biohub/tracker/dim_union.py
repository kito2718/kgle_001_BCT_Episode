"""Baseline-retaining solver superset union for dim candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from biohub.metrics.gt import SCALE_ARR
from biohub.tracker.edges import NMS_UM, select_nodes


def dim_union_superset(
    candidate: pd.DataFrame,
    cap: int,
    nms_um: float = NMS_UM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the full detector superset and add spatially distinct dim picks."""
    control = select_nodes(
        candidate, cap, nms_um=nms_um,
        score_values=candidate.score.to_numpy(np.float64),
    )
    treatment = select_nodes(
        candidate, cap, nms_um=nms_um,
        score_values=candidate.selection_score.to_numpy(np.float64),
    )
    novel = np.setdiff1d(treatment, control, assume_unique=True)
    times = candidate.t.to_numpy(np.int64)
    points = (
        np.round(candidate[["z", "y", "x"]].to_numpy(np.float64)) * SCALE_ARR
    )
    eligible = []
    for frame in np.unique(times[novel]):
        query = novel[times[novel] == frame]
        incumbent = control[times[control] == frame]
        if len(incumbent):
            distance = cKDTree(points[incumbent]).query(points[query], k=1)[0]
            query = query[distance > nms_um]
        eligible.append(query)
    added = np.concatenate(eligible) if eligible else np.zeros(0, dtype=np.int64)
    union = np.concatenate([control, added])
    if len(np.unique(union)) != len(union):
        raise RuntimeError("dim union contains duplicate candidate rows")
    return control, treatment, added, union


def hard_gated_dim_union_superset(
    candidate: pd.DataFrame,
    cap: int,
    eligible_mask: np.ndarray,
    nms_um: float = NMS_UM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add specialist support only at candidates inside a detected dim region."""
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    if eligible_mask.shape != (len(candidate),):
        raise ValueError("dim-region mask length differs from candidate rows")
    control = select_nodes(
        candidate, cap, nms_um=nms_um,
        score_values=candidate.score.to_numpy(np.float64),
    )
    treatment = select_nodes(
        candidate, cap, nms_um=nms_um,
        score_values=candidate.selection_score.to_numpy(np.float64),
    )
    novel = np.setdiff1d(treatment, control, assume_unique=True)
    novel = novel[eligible_mask[novel]]
    times = candidate.t.to_numpy(np.int64)
    points = (
        np.round(candidate[["z", "y", "x"]].to_numpy(np.float64)) * SCALE_ARR
    )
    added = []
    for frame in np.unique(times[novel]):
        query = novel[times[novel] == frame]
        incumbent = control[times[control] == frame]
        if len(incumbent):
            distance = cKDTree(points[incumbent]).query(points[query], k=1)[0]
            query = query[distance > nms_um]
        added.append(query)
    added_rows = np.concatenate(added) if added else np.zeros(0, dtype=np.int64)
    union = np.concatenate([control, added_rows])
    if len(np.unique(union)) != len(union):
        raise RuntimeError("hard-gated dim union contains duplicate candidate rows")
    if len(added_rows) and not eligible_mask[added_rows].all():
        raise RuntimeError("specialist support escaped the dim-region mask")
    return control, treatment, added_rows, union
