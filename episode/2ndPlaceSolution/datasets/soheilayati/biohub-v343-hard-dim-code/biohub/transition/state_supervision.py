"""Sparse-label-safe candidate-state supervision for V34B1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metrics.gt import SCALE_ARR


STATE_OFFSETS = tuple(range(-3, 4))
STATE_FEATURE_COLUMNS = (
    "detector_score",
    "detector_logit",
    "rank_fraction",
    "density_10um",
    "nearest_um",
    "boundary_um",
    "incoming_local_resid_um",
    "outgoing_local_resid_um",
    "incoming_distance_um",
    "outgoing_distance_um",
    "trajectory_valid_fraction",
    "time_fraction",
)


@dataclass(frozen=True)
class StateRows:
    candidate_rows: np.ndarray
    target: np.ndarray
    kind: np.ndarray
    group_gt_row: np.ndarray


def choose_local_alternatives(
    candidate_t: np.ndarray,
    candidate_zyx: np.ndarray,
    candidate_score: np.ndarray,
    pred_to_gt: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    duplicate_radius_um: float = 5.0,
    gt_exclusion_radius_um: float = 7.0,
    per_positive: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Verified mutually exclusive alternatives around matched positives.

    An alternative must be near the positive hypothesis, unmatched, and within
    the GT exclusion radius of exactly that positive's annotated cell. This
    avoids declaring a nearby second annotated cell to be a negative state.
    """

    from scipy.spatial import cKDTree

    t = np.asarray(candidate_t, dtype=np.int64)
    zyx = np.asarray(candidate_zyx, dtype=np.float64)
    score = np.asarray(candidate_score, dtype=np.float64)
    pred_to_gt = np.asarray(pred_to_gt, dtype=np.int64)
    positive = np.flatnonzero(pred_to_gt >= 0)
    selected: list[int] = []
    groups: list[int] = []
    used: set[int] = set()
    for frame in np.unique(t[positive]):
        candidate_rows = np.flatnonzero(t == frame)
        positive_rows = positive[t[positive] == frame]
        gt_rows = np.flatnonzero(np.asarray(gt_t, dtype=np.int64) == frame)
        if not len(candidate_rows) or not len(gt_rows):
            continue
        candidate_um = zyx[candidate_rows] * SCALE_ARR
        gt_um = np.asarray(gt_zyx, dtype=np.float64)[gt_rows] * SCALE_ARR
        candidate_tree = cKDTree(candidate_um)
        gt_tree = cKDTree(gt_um)
        local_for_global = {int(row): index for index, row in enumerate(candidate_rows)}
        gt_local_for_global = {int(row): index for index, row in enumerate(gt_rows)}
        for positive_row in positive_rows:
            gt_row = int(pred_to_gt[positive_row])
            gt_local = gt_local_for_global.get(gt_row)
            if gt_local is None:
                continue
            neighbours = candidate_tree.query_ball_point(
                candidate_um[local_for_global[int(positive_row)]],
                r=duplicate_radius_um,
            )
            eligible = []
            for local in neighbours:
                row = int(candidate_rows[int(local)])
                if row == int(positive_row) or pred_to_gt[row] >= 0 or row in used:
                    continue
                nearby_gt = gt_tree.query_ball_point(candidate_um[int(local)], r=gt_exclusion_radius_um)
                if set(map(int, nearby_gt)) != {int(gt_local)}:
                    continue
                eligible.append(row)
            eligible.sort(key=lambda row: (-score[row], row))
            for row in eligible[:per_positive]:
                used.add(int(row))
                selected.append(int(row))
                groups.append(gt_row)
    return np.asarray(selected, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def choose_dark_background(
    candidate_t: np.ndarray,
    candidate_zyx: np.ndarray,
    candidate_score: np.ndarray,
    center_intensity: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    excluded_rows: np.ndarray,
    maximum: int,
    background_tau: float = 0.02,
    positive_radius_um: float = 6.0,
) -> np.ndarray:
    """Highest-scoring candidates in measured reliable dark background."""

    from scipy.spatial import cKDTree

    t = np.asarray(candidate_t, dtype=np.int64)
    zyx = np.asarray(candidate_zyx, dtype=np.float64)
    score = np.asarray(candidate_score, dtype=np.float64)
    intensity = np.asarray(center_intensity, dtype=np.float64)
    allowed = intensity < background_tau
    allowed[np.asarray(excluded_rows, dtype=np.int64)] = False
    for frame in np.unique(t[allowed]):
        rows = np.flatnonzero((t == frame) & allowed)
        gt_rows = np.flatnonzero(np.asarray(gt_t, dtype=np.int64) == frame)
        if not len(rows) or not len(gt_rows):
            continue
        distance = cKDTree(np.asarray(gt_zyx)[gt_rows] * SCALE_ARR).query(
            zyx[rows] * SCALE_ARR, k=1
        )[0]
        allowed[rows[distance <= positive_radius_um]] = False
    rows = np.flatnonzero(allowed)
    order = np.lexsort((rows, -score[rows]))
    return rows[order[: int(maximum)]].astype(np.int64)


def best_edge_by_group(
    groups: np.ndarray,
    value: np.ndarray,
    edge_rows: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Stable minimum-value edge row for every candidate group."""

    groups = np.asarray(groups, dtype=np.int64)
    value = np.asarray(value, dtype=np.float64)
    edge_rows = np.asarray(edge_rows, dtype=np.int64)
    order = np.lexsort((edge_rows, value, groups))
    output = np.full(n_nodes, -1, dtype=np.int64)
    for position in order:
        group = int(groups[position])
        if output[group] < 0:
            output[group] = int(edge_rows[position])
    return output


def trace_candidate_paths(
    candidate_rows: np.ndarray,
    t: np.ndarray,
    zyx: np.ndarray,
    edges: np.ndarray,
    best_incoming_edge: np.ndarray,
    best_outgoing_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Seven-frame annotation-free paths through deterministic local links."""

    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    zyx = np.asarray(zyx, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    path_rows = np.full((len(candidate_rows), 7), -1, dtype=np.int64)
    path_rows[:, 3] = candidate_rows
    for output_row, center in enumerate(candidate_rows):
        current = int(center)
        for slot in range(2, -1, -1):
            edge = int(best_incoming_edge[current])
            if edge < 0:
                break
            current = int(edges[edge, 0])
            path_rows[output_row, slot] = current
        current = int(center)
        for slot in range(4, 7):
            edge = int(best_outgoing_edge[current])
            if edge < 0:
                break
            current = int(edges[edge, 1])
            path_rows[output_row, slot] = current
    coordinates = np.full((len(candidate_rows), 7, 3), np.nan, dtype=np.float32)
    valid = path_rows >= 0
    coordinates[valid] = zyx[path_rows[valid]].astype(np.float32)
    expected_t = t[candidate_rows, None] + np.asarray(STATE_OFFSETS)[None, :]
    if np.any(valid & (t[np.maximum(path_rows, 0)] != expected_t)):
        raise RuntimeError("traced candidate path has an incorrect frame offset")
    return path_rows, coordinates


def build_state_feature_matrix(
    context,
    transition,
    best_incoming_edge: np.ndarray,
    best_outgoing_edge: np.ndarray,
    t: np.ndarray,
    path_rows: np.ndarray,
) -> np.ndarray:
    """Deploy-safe structural state features for traced candidate centres."""

    t = np.asarray(t, dtype=np.int64)
    path_rows = np.asarray(path_rows, dtype=np.int64)
    selected = path_rows[:, 3]
    if np.any(selected < 0):
        raise ValueError("every state path must have a valid centre candidate")
    best_incoming_edge = np.asarray(best_incoming_edge, dtype=np.int64)
    best_outgoing_edge = np.asarray(best_outgoing_edge, dtype=np.int64)
    incoming_resid = np.full(len(t), 50.0, dtype=np.float32)
    outgoing_resid = np.full(len(t), 50.0, dtype=np.float32)
    incoming_distance = np.full(len(t), 50.0, dtype=np.float32)
    outgoing_distance = np.full(len(t), 50.0, dtype=np.float32)
    has_incoming = best_incoming_edge >= 0
    has_outgoing = best_outgoing_edge >= 0
    backward_resid = transition.backward_local_resid_um.to_numpy(dtype=np.float32)
    forward_resid = transition.forward_local_resid_um.to_numpy(dtype=np.float32)
    distance = transition.distance_um.to_numpy(dtype=np.float32)
    incoming_resid[has_incoming] = backward_resid[best_incoming_edge[has_incoming]]
    outgoing_resid[has_outgoing] = forward_resid[best_outgoing_edge[has_outgoing]]
    incoming_distance[has_incoming] = distance[best_incoming_edge[has_incoming]]
    outgoing_distance[has_outgoing] = distance[best_outgoing_edge[has_outgoing]]
    valid_fraction = (path_rows >= 0).mean(axis=1).astype(np.float32)
    denominator = max(int(t.max(initial=0)), 1)
    values = {
        "detector_score": context.score[selected],
        "detector_logit": context.logit[selected],
        "rank_fraction": context.rank_fraction[selected],
        "density_10um": context.density_10um[selected],
        "nearest_um": context.nearest_um[selected],
        "boundary_um": context.boundary_um[selected],
        "incoming_local_resid_um": incoming_resid[selected],
        "outgoing_local_resid_um": outgoing_resid[selected],
        "incoming_distance_um": incoming_distance[selected],
        "outgoing_distance_um": outgoing_distance[selected],
        "trajectory_valid_fraction": valid_fraction,
        "time_fraction": t[selected] / denominator,
    }
    output = np.column_stack([
        np.asarray(values[column], dtype=np.float32) for column in STATE_FEATURE_COLUMNS
    ]).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite candidate-state structural feature")
    return output
