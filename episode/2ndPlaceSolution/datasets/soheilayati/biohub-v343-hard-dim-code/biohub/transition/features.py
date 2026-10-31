"""Annotation-free candidate and transition features for V34B0.

Every feature in this module is available on hidden test data.  Ground truth is
used by the corpus builder only after these arrays have been constructed, to
choose listwise training groups and labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biohub.metrics.gt import SCALE_ARR
from biohub.motion.flow import estimate_global_translation, estimate_local_flow

SOURCE_SHAPE = np.array([64, 256, 256], dtype=np.float64)

FEATURE_COLUMNS = (
    "score_parent",
    "score_child",
    "score_min",
    "score_max",
    "score_child_minus_parent",
    "logit_parent",
    "logit_child",
    "rank_fraction_parent",
    "rank_fraction_child",
    "density_10um_parent",
    "density_10um_child",
    "nearest_um_parent",
    "nearest_um_child",
    "boundary_um_parent",
    "boundary_um_child",
    "backward_dz_um",
    "backward_dy_um",
    "backward_dx_um",
    "distance_um",
    "distance_z_um",
    "distance_xy_um",
    "backward_global_resid_um",
    "backward_local_resid_um",
    "backward_local_resid_z_um",
    "backward_local_resid_xy_um",
    "forward_global_resid_um",
    "forward_local_resid_um",
    "forward_local_resid_z_um",
    "forward_local_resid_xy_um",
    "reciprocal_flow_disagreement_um",
    "parent_hypotheses",
    "local_resid_rank",
    "local_resid_margin_um",
    "distance_rank",
    "distance_margin_um",
    "parent_score_rank",
    "parent_score_margin",
    "child_claims_parent",
    "forward_resid_rank_for_parent",
    "time_fraction",
)


@dataclass(frozen=True)
class CandidateContext:
    t: np.ndarray
    zyx: np.ndarray
    um: np.ndarray
    score: np.ndarray
    logit: np.ndarray
    rank_fraction: np.ndarray
    density_10um: np.ndarray
    nearest_um: np.ndarray
    boundary_um: np.ndarray
    forward_global_um: np.ndarray
    backward_global_um: np.ndarray
    forward_local_um: np.ndarray
    backward_local_um: np.ndarray


def _rank_fraction(t: np.ndarray, score: np.ndarray) -> np.ndarray:
    out = np.zeros(len(t), dtype=np.float32)
    for frame in np.unique(t):
        rows = np.flatnonzero(t == frame)
        order = np.argsort(-score[rows], kind="stable")
        rank = np.empty(len(rows), dtype=np.float64)
        rank[order] = np.arange(len(rows), dtype=np.float64)
        out[rows] = rank / max(len(rows) - 1, 1)
    return out


def _spatial_context(t: np.ndarray, um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    density = np.zeros(len(t), dtype=np.float32)
    nearest = np.full(len(t), 1e3, dtype=np.float32)
    for frame in np.unique(t):
        rows = np.flatnonzero(t == frame)
        if not len(rows):
            continue
        tree = cKDTree(um[rows])
        density[rows] = tree.query_ball_point(
            um[rows], r=10.0, return_length=True
        ).astype(np.float32) - 1.0
        if len(rows) > 1:
            distance, _ = tree.query(um[rows], k=2)
            nearest[rows] = distance[:, 1].astype(np.float32)
    return density, nearest


def build_candidate_context(
    t: np.ndarray,
    zyx: np.ndarray,
    score: np.ndarray,
) -> CandidateContext:
    """Candidate-side context and bidirectional tissue motion."""

    t = np.asarray(t, dtype=np.int64).reshape(-1)
    zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if not (len(t) == len(zyx) == len(score)):
        raise ValueError("t, zyx, and score must have the same length")
    if not np.all(np.isfinite(zyx)) or not np.all(np.isfinite(score)):
        raise ValueError("candidate coordinates and scores must be finite")

    clipped = np.clip(score, 1e-5, 1.0 - 1e-5)
    um = zyx * SCALE_ARR
    density, nearest = _spatial_context(t, um)
    lo = zyx * SCALE_ARR
    hi = (SOURCE_SHAPE - 1.0 - zyx) * SCALE_ARR
    boundary = np.maximum(np.minimum(lo, hi).min(axis=1), 0.0).astype(np.float32)

    forward_global = np.zeros((len(t), 3), dtype=np.float32)
    backward_global = np.zeros((len(t), 3), dtype=np.float32)
    forward_local = np.zeros((len(t), 3), dtype=np.float32)
    backward_local = np.zeros((len(t), 3), dtype=np.float32)
    at = {int(frame): np.flatnonzero(t == frame) for frame in np.unique(t)}
    for frame in sorted(at):
        source = at[frame]
        target = at.get(frame + 1)
        if target is None or not len(source) or not len(target):
            continue
        shift_forward = estimate_global_translation(um[source], um[target])
        shift_backward = estimate_global_translation(um[target], um[source])
        forward_global[source] = shift_forward
        backward_global[target] = shift_backward
        forward_local[source] = estimate_local_flow(
            um[source], um[target], shift=shift_forward
        ).astype(np.float32)
        backward_local[target] = estimate_local_flow(
            um[target], um[source], shift=shift_backward
        ).astype(np.float32)

    return CandidateContext(
        t=t,
        zyx=zyx,
        um=um,
        score=score.astype(np.float32),
        logit=np.log(clipped / (1.0 - clipped)).astype(np.float32),
        rank_fraction=_rank_fraction(t, score),
        density_10um=density,
        nearest_um=nearest,
        boundary_um=boundary,
        forward_global_um=forward_global,
        backward_global_um=backward_global,
        forward_local_um=forward_local,
        backward_local_um=backward_local,
    )


def _group_rank(group: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Stable zero-based ascending rank within integer groups."""

    order = np.lexsort((np.arange(len(group)), value, group))
    sorted_group = group[order]
    starts = np.r_[True, sorted_group[1:] != sorted_group[:-1]]
    index = np.arange(len(group), dtype=np.int64)
    start_index = np.maximum.accumulate(np.where(starts, index, 0))
    ranked = index - start_index
    out = np.empty(len(group), dtype=np.float32)
    out[order] = ranked.astype(np.float32)
    return out


def _group_min(group: np.ndarray, value: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, np.inf, dtype=np.float64)
    np.minimum.at(out, group, value)
    return out[group]


def _group_max(group: np.ndarray, value: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(out, group, value)
    return out[group]


def build_transition_features(
    context: CandidateContext,
    edges: np.ndarray,
) -> pd.DataFrame:
    """One finite, deploy-safe feature row per ``(parent, child)`` edge."""

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(context.t)
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edges contain an out-of-range candidate index")
    if len(edges) and np.any(
        context.t[edges[:, 1]] != context.t[edges[:, 0]] + 1
    ):
        raise ValueError("transition edges must connect consecutive frames")
    if not len(edges):
        return pd.DataFrame({name: pd.Series(dtype=np.float32) for name in FEATURE_COLUMNS})

    parent, child = edges[:, 0], edges[:, 1]
    backward = context.um[parent] - context.um[child]
    forward = -backward
    back_global_vec = backward - context.backward_global_um[child]
    back_local_vec = backward - context.backward_local_um[child]
    forward_global_vec = forward - context.forward_global_um[parent]
    forward_local_vec = forward - context.forward_local_um[parent]

    distance = np.linalg.norm(backward, axis=1)
    back_global = np.linalg.norm(back_global_vec, axis=1)
    back_local = np.linalg.norm(back_local_vec, axis=1)
    forward_global = np.linalg.norm(forward_global_vec, axis=1)
    forward_local = np.linalg.norm(forward_local_vec, axis=1)
    reciprocal = np.linalg.norm(
        context.forward_local_um[parent] + context.backward_local_um[child], axis=1
    )

    n_child_groups = n
    n_parent_groups = n
    parent_hypotheses = np.bincount(child, minlength=n_child_groups)[child]
    child_claims = np.bincount(parent, minlength=n_parent_groups)[parent]
    local_rank = _group_rank(child, back_local)
    distance_rank = _group_rank(child, distance)
    parent_score_rank = _group_rank(child, -context.score[parent])
    forward_rank = _group_rank(parent, forward_local)
    local_margin = back_local - _group_min(child, back_local, n_child_groups)
    distance_margin = distance - _group_min(child, distance, n_child_groups)
    score_margin = _group_max(
        child, context.score[parent], n_child_groups
    ) - context.score[parent]

    frame_denominator = max(int(context.t.max(initial=0)), 1)
    values = {
        "score_parent": context.score[parent],
        "score_child": context.score[child],
        "score_min": np.minimum(context.score[parent], context.score[child]),
        "score_max": np.maximum(context.score[parent], context.score[child]),
        "score_child_minus_parent": context.score[child] - context.score[parent],
        "logit_parent": context.logit[parent],
        "logit_child": context.logit[child],
        "rank_fraction_parent": context.rank_fraction[parent],
        "rank_fraction_child": context.rank_fraction[child],
        "density_10um_parent": context.density_10um[parent],
        "density_10um_child": context.density_10um[child],
        "nearest_um_parent": context.nearest_um[parent],
        "nearest_um_child": context.nearest_um[child],
        "boundary_um_parent": context.boundary_um[parent],
        "boundary_um_child": context.boundary_um[child],
        "backward_dz_um": backward[:, 0],
        "backward_dy_um": backward[:, 1],
        "backward_dx_um": backward[:, 2],
        "distance_um": distance,
        "distance_z_um": np.abs(backward[:, 0]),
        "distance_xy_um": np.hypot(backward[:, 1], backward[:, 2]),
        "backward_global_resid_um": back_global,
        "backward_local_resid_um": back_local,
        "backward_local_resid_z_um": np.abs(back_local_vec[:, 0]),
        "backward_local_resid_xy_um": np.hypot(back_local_vec[:, 1], back_local_vec[:, 2]),
        "forward_global_resid_um": forward_global,
        "forward_local_resid_um": forward_local,
        "forward_local_resid_z_um": np.abs(forward_local_vec[:, 0]),
        "forward_local_resid_xy_um": np.hypot(forward_local_vec[:, 1], forward_local_vec[:, 2]),
        "reciprocal_flow_disagreement_um": reciprocal,
        "parent_hypotheses": parent_hypotheses,
        "local_resid_rank": local_rank,
        "local_resid_margin_um": local_margin,
        "distance_rank": distance_rank,
        "distance_margin_um": distance_margin,
        "parent_score_rank": parent_score_rank,
        "parent_score_margin": score_margin,
        "child_claims_parent": child_claims,
        "forward_resid_rank_for_parent": forward_rank,
        "time_fraction": context.t[child] / frame_denominator,
    }
    frame = pd.DataFrame({
        name: np.asarray(values[name], dtype=np.float32) for name in FEATURE_COLUMNS
    })
    if not np.isfinite(frame.to_numpy(dtype=np.float32)).all():
        raise RuntimeError("transition feature construction produced a non-finite value")
    return frame
