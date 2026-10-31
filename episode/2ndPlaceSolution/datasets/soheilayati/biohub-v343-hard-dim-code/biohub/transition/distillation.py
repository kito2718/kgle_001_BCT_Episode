"""Annotation-free trajectory-event distillation for the V34 global decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metrics.gt import MATCH_UM, SCALE_ARR
from biohub.metrics.matching import match_sequence


@dataclass(frozen=True)
class DistilledTrajectoryEvents:
    """Dense continuation labels projected from a frozen teacher trajectory."""

    teacher_to_candidate: np.ndarray
    match_distance_um: np.ndarray
    incoming_target: np.ndarray
    incoming_mask: np.ndarray
    incoming_has_teacher_parent: np.ndarray
    incoming_true_edge_available: np.ndarray
    outgoing_target: np.ndarray
    outgoing_mask: np.ndarray
    outgoing_has_teacher_child: np.ndarray
    outgoing_true_edge_available: np.ndarray


def edge_membership(
    edges: np.ndarray,
    query: np.ndarray,
    *,
    n_candidates: int,
) -> np.ndarray:
    """Return exact directed-edge membership without materializing a dense matrix."""

    edge_values = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    query_values = np.asarray(query, dtype=np.int64).reshape(-1, 2)
    if n_candidates < 0:
        raise ValueError("n_candidates must be nonnegative")
    for label, values in (("edge", edge_values), ("query", query_values)):
        if len(values) and ((values < 0) | (values >= n_candidates)).any():
            raise ValueError(f"{label} endpoint lies outside the candidate universe")
    if not len(query_values):
        return np.zeros(0, dtype=bool)
    encoded = np.unique(
        edge_values[:, 0] * int(n_candidates) + edge_values[:, 1]
    )
    requested = query_values[:, 0] * int(n_candidates) + query_values[:, 1]
    location = np.searchsorted(encoded, requested)
    valid = location < len(encoded)
    output = np.zeros(len(requested), dtype=bool)
    output[valid] = encoded[location[valid]] == requested[valid]
    return output


def distill_ordinary_trajectory_events(
    *,
    candidate_t: np.ndarray,
    candidate_zyx: np.ndarray,
    candidate_edges: np.ndarray,
    deploy_candidate_mask: np.ndarray,
    teacher_t: np.ndarray,
    teacher_zyx: np.ndarray,
    teacher_edges: np.ndarray,
    max_distance_um: float = MATCH_UM,
) -> DistilledTrajectoryEvents:
    """Project a division-free incumbent graph onto a deploy candidate lattice.

    A real continuation is positive only when both teacher endpoints map and the
    exact directed candidate edge exists. An unavailable continuation is unknown
    (masked), never a negative. Nodes with no teacher predecessor/successor are
    reliable teacher starts/ends and therefore receive negative event labels.
    """

    candidate_t = np.asarray(candidate_t, dtype=np.int64).reshape(-1)
    candidate_zyx = np.asarray(candidate_zyx, dtype=np.float64).reshape(-1, 3)
    candidate_edges = np.asarray(candidate_edges, dtype=np.int64).reshape(-1, 2)
    deploy = np.asarray(deploy_candidate_mask, dtype=bool).reshape(-1)
    teacher_t = np.asarray(teacher_t, dtype=np.int64).reshape(-1)
    teacher_zyx = np.asarray(teacher_zyx, dtype=np.float64).reshape(-1, 3)
    teacher_edges = np.asarray(teacher_edges, dtype=np.int64).reshape(-1, 2)
    if not (len(candidate_t) == len(candidate_zyx) == len(deploy)):
        raise ValueError("candidate node-array lengths differ")
    if len(candidate_edges) and (
        (candidate_edges < 0) | (candidate_edges >= len(candidate_t))
    ).any():
        raise ValueError("candidate edge lies outside the node universe")
    if len(teacher_t) != len(teacher_zyx):
        raise ValueError("teacher node-array lengths differ")
    if len(teacher_edges) and (
        (teacher_edges < 0) | (teacher_edges >= len(teacher_t))
    ).any():
        raise ValueError("teacher edge lies outside the node universe")
    if not np.isfinite(candidate_zyx).all() or not np.isfinite(teacher_zyx).all():
        raise ValueError("trajectory coordinates must be finite")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0:
        raise ValueError("max_distance_um must be finite and positive")

    indegree = np.bincount(teacher_edges[:, 1], minlength=len(teacher_t))
    outdegree = np.bincount(teacher_edges[:, 0], minlength=len(teacher_t))
    if np.any(indegree > 1) or np.any(outdegree > 1):
        raise ValueError("teacher trajectory must be division-free with unit degrees")
    parent = np.full(len(teacher_t), -1, dtype=np.int64)
    child = np.full(len(teacher_t), -1, dtype=np.int64)
    if len(teacher_edges):
        parent[teacher_edges[:, 1]] = teacher_edges[:, 0]
        child[teacher_edges[:, 0]] = teacher_edges[:, 1]

    deploy_rows = np.flatnonzero(deploy)
    rounded_candidate_zyx = np.rint(candidate_zyx).astype(np.int64)
    teacher_to_local = match_sequence(
        candidate_t[deploy_rows],
        rounded_candidate_zyx[deploy_rows],
        teacher_t,
        teacher_zyx,
        max_distance=float(max_distance_um),
    )
    teacher_to_candidate = np.full(len(teacher_t), -1, dtype=np.int64)
    matched = teacher_to_local >= 0
    distance = np.full(len(teacher_t), np.inf, dtype=np.float64)
    proposed_candidate = deploy_rows[teacher_to_local[matched]]
    proposed_distance = np.linalg.norm(
        (
            rounded_candidate_zyx[proposed_candidate]
            - teacher_zyx[matched]
        )
        * SCALE_ARR,
        axis=1,
    )
    # The scorer-compatible matcher deliberately mirrors a sparse-fallback quirk
    # that can very rarely emit a pair outside its nominal gate. Distillation has
    # a strict physical teacher contract, so reject those fallback artefacts.
    within_gate = proposed_distance <= float(max_distance_um) + 1e-9
    matched_rows = np.flatnonzero(matched)
    accepted_rows = matched_rows[within_gate]
    teacher_to_candidate[accepted_rows] = proposed_candidate[within_gate]
    distance[accepted_rows] = proposed_distance[within_gate]
    matched = teacher_to_candidate >= 0
    if len(np.unique(teacher_to_candidate[matched])) != int(matched.sum()):
        raise RuntimeError("teacher projection is not one-to-one")

    incoming_has = (parent >= 0) & (teacher_t[parent.clip(min=0)] == teacher_t - 1)
    outgoing_has = (child >= 0) & (teacher_t[child.clip(min=0)] == teacher_t + 1)
    incoming_available = np.zeros(len(teacher_t), dtype=bool)
    outgoing_available = np.zeros(len(teacher_t), dtype=bool)

    incoming_rows = np.flatnonzero(
        matched & incoming_has & (teacher_to_candidate[parent.clip(min=0)] >= 0)
    )
    if len(incoming_rows):
        query = np.column_stack(
            [
                teacher_to_candidate[parent[incoming_rows]],
                teacher_to_candidate[incoming_rows],
            ]
        )
        incoming_available[incoming_rows] = edge_membership(
            candidate_edges, query, n_candidates=len(candidate_t)
        )

    outgoing_rows = np.flatnonzero(
        matched & outgoing_has & (teacher_to_candidate[child.clip(min=0)] >= 0)
    )
    if len(outgoing_rows):
        query = np.column_stack(
            [
                teacher_to_candidate[outgoing_rows],
                teacher_to_candidate[child[outgoing_rows]],
            ]
        )
        outgoing_available[outgoing_rows] = edge_membership(
            candidate_edges, query, n_candidates=len(candidate_t)
        )

    incoming_mask = matched & ((parent < 0) | incoming_available)
    outgoing_mask = matched & ((child < 0) | outgoing_available)
    return DistilledTrajectoryEvents(
        teacher_to_candidate=teacher_to_candidate,
        match_distance_um=distance,
        incoming_target=incoming_available.astype(np.int8),
        incoming_mask=incoming_mask,
        incoming_has_teacher_parent=incoming_has,
        incoming_true_edge_available=incoming_available,
        outgoing_target=outgoing_available.astype(np.int8),
        outgoing_mask=outgoing_mask,
        outgoing_has_teacher_child=outgoing_has,
        outgoing_true_edge_available=outgoing_available,
    )
