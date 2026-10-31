"""Sparse-label-safe supervision for the V34 hypothesis graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metrics.gt import GroundTruth, MATCH_UM
from biohub.metrics.matching import match_sequence

# A negative value is deliberately distinct from zero.  Unknown examples must
# be masked out of the loss, not interpreted as a weak negative.
UNKNOWN = np.int8(-1)
NEGATIVE = np.int8(0)
POSITIVE = np.int8(1)


@dataclass(frozen=True)
class CandidateSupervision:
    candidate_to_gt: np.ndarray
    gt_to_candidate: np.ndarray
    node_target: np.ndarray
    edge_target: np.ndarray

    @property
    def covered_gt_nodes(self) -> int:
        return int(np.count_nonzero(self.gt_to_candidate >= 0))

    @property
    def covered_gt_edges(self) -> int:
        return int(np.count_nonzero(self.edge_target == POSITIVE))


def label_candidate_graph(
    candidate_t: np.ndarray,
    candidate_zyx: np.ndarray,
    edges: np.ndarray,
    gt: GroundTruth,
    max_distance_um: float = MATCH_UM,
) -> CandidateSupervision:
    """Create positive/negative/unknown targets without sparse-label leakage.

    Candidate nodes matched by the official assignment are positive.  Every
    other candidate node is *unknown*, because it may be an unlabeled real cell.

    A transition is positive when both endpoints match a ground-truth edge.  It
    is a verified negative only when both endpoints match annotated nodes that
    are not connected.  If either endpoint is unmatched, the edge is unknown.
    """

    candidate_t = np.asarray(candidate_t, dtype=np.int64).reshape(-1)
    candidate_zyx = np.asarray(candidate_zyx, dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(candidate_t)
    if len(candidate_zyx) != n:
        raise ValueError("candidate_t and candidate_zyx must have the same length")
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edges contain an out-of-range candidate index")

    gt_to_candidate = match_sequence(
        candidate_t, candidate_zyx, gt.t, gt.zyx, max_distance=max_distance_um
    )
    candidate_to_gt = np.full(n, -1, dtype=np.int64)
    matched_gt = np.flatnonzero(gt_to_candidate >= 0)
    candidate_to_gt[gt_to_candidate[matched_gt]] = matched_gt

    node_target = np.full(n, UNKNOWN, dtype=np.int8)
    node_target[candidate_to_gt >= 0] = POSITIVE

    edge_target = np.full(len(edges), UNKNOWN, dtype=np.int8)
    if len(edges):
        gs = candidate_to_gt[edges[:, 0]]
        gd = candidate_to_gt[edges[:, 1]]
        known = (gs >= 0) & (gd >= 0)
        edge_target[known] = NEGATIVE
        gt_edges = {tuple(map(int, edge)) for edge in gt.edges.tolist()}
        positive = np.fromiter(
            ((int(a), int(b)) in gt_edges for a, b in zip(gs, gd)),
            dtype=bool,
            count=len(edges),
        )
        edge_target[known & positive] = POSITIVE

    return CandidateSupervision(
        candidate_to_gt=candidate_to_gt,
        gt_to_candidate=gt_to_candidate,
        node_target=node_target,
        edge_target=edge_target,
    )
