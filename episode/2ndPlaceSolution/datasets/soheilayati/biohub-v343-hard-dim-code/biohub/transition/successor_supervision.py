"""Exact parent-group supervision for the V34B5 conditional successor model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SuccessorRows:
    """Retained edge rows and exact one-hot targets for ordinary successors."""

    edge_rows: np.ndarray
    label: np.ndarray
    eligible_parent_candidate: np.ndarray
    ordinary_gt_edges: int
    matched_ordinary_pairs: int
    available_ordinary_groups: int


def label_ordinary_successor_lists(
    *,
    edges: np.ndarray,
    candidate_t: np.ndarray,
    gt_t: np.ndarray,
    gt_edges: np.ndarray,
    gt_to_candidate: np.ndarray,
) -> SuccessorRows:
    """Label complete parent-side lists with one available ordinary successor.

    Division parents, deaths, unmatched endpoints, non-adjacent annotations,
    and true successor pairs absent from the fixed candidate-edge universe are
    excluded. For every retained parent, all of its available outgoing edges
    are returned and exactly one is positive.
    """

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    candidate_t = np.asarray(candidate_t, dtype=np.int64).reshape(-1)
    gt_t = np.asarray(gt_t, dtype=np.int64).reshape(-1)
    gt_edges = np.asarray(gt_edges, dtype=np.int64).reshape(-1, 2)
    gt_to_candidate = np.asarray(gt_to_candidate, dtype=np.int64).reshape(-1)
    if len(gt_to_candidate) != len(gt_t):
        raise ValueError("GT-to-candidate mapping length differs from GT nodes")
    if len(edges) and ((edges < 0) | (edges >= len(candidate_t))).any():
        raise ValueError("candidate edge endpoint lies outside candidate nodes")
    if len(gt_edges) and ((gt_edges < 0) | (gt_edges >= len(gt_t))).any():
        raise ValueError("GT edge endpoint lies outside GT nodes")
    if np.any((gt_to_candidate < -1) | (gt_to_candidate >= len(candidate_t))):
        raise ValueError("GT-to-candidate mapping lies outside candidate nodes")
    if len(gt_edges) == 0 or len(edges) == 0:
        return SuccessorRows(
            edge_rows=np.zeros(0, dtype=np.int64),
            label=np.zeros(0, dtype=np.int8),
            eligible_parent_candidate=np.zeros(0, dtype=np.int64),
            ordinary_gt_edges=0,
            matched_ordinary_pairs=0,
            available_ordinary_groups=0,
        )

    outdegree = np.bincount(gt_edges[:, 0], minlength=len(gt_t))
    ordinary = (
        (outdegree[gt_edges[:, 0]] == 1)
        & (gt_t[gt_edges[:, 1]] == gt_t[gt_edges[:, 0]] + 1)
    )
    ordinary_edges = gt_edges[ordinary]
    parent_candidate = gt_to_candidate[ordinary_edges[:, 0]]
    child_candidate = gt_to_candidate[ordinary_edges[:, 1]]
    matched = (parent_candidate >= 0) & (child_candidate >= 0)
    parent_candidate = parent_candidate[matched]
    child_candidate = child_candidate[matched]
    if len(parent_candidate):
        if len(np.unique(parent_candidate)) != len(parent_candidate):
            raise ValueError("matched ordinary GT parents are not one-to-one")
        if not np.array_equal(
            candidate_t[child_candidate], candidate_t[parent_candidate] + 1
        ):
            raise ValueError("matched ordinary pair is not candidate-time adjacent")

    expected_child = np.full(len(candidate_t), -1, dtype=np.int64)
    expected_child[parent_candidate] = child_candidate
    positive_in_universe = expected_child[edges[:, 0]] == edges[:, 1]
    available_parent = np.unique(edges[positive_in_universe, 0])
    eligible = np.zeros(len(candidate_t), dtype=bool)
    eligible[available_parent] = True
    edge_rows = np.flatnonzero(eligible[edges[:, 0]])
    labels = positive_in_universe[edge_rows].astype(np.int8)
    if len(edge_rows):
        positive_count = np.bincount(
            edges[edge_rows, 0], weights=labels, minlength=len(candidate_t)
        )
        if not np.all(positive_count[available_parent] == 1):
            raise ValueError("successor parent group is not exactly one-hot")
        present_parent = np.unique(edges[edge_rows, 0])
        if not np.array_equal(present_parent, available_parent):
            raise ValueError("eligible successor parent lost its candidate list")

    return SuccessorRows(
        edge_rows=edge_rows,
        label=labels,
        eligible_parent_candidate=available_parent,
        ordinary_gt_edges=int(len(ordinary_edges)),
        matched_ordinary_pairs=int(len(parent_candidate)),
        available_ordinary_groups=int(len(available_parent)),
    )
