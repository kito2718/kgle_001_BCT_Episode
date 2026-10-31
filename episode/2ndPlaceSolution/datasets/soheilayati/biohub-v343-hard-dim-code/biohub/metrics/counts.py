"""Edge TP/FP/FN from arrays, without building a graph.

The oracle ladder evaluates hundreds of operating points across 199 samples.
Going through `tracksdata` each time means constructing a graph object per
evaluation, which dominates the runtime and buys nothing, because the counting
rule itself is short:

    TP          predicted edge whose two matched endpoints are a ground-truth edge
    pred_valid  matched source has GT out-degree > 0, OR matched target has GT
                in-degree > 0                     <- the sparse-annotation gate
    FP          pred_valid - TP
    FN          gt_edges - TP                     <- never discounted

The asymmetry in `pred_valid` is the whole reason a wrong link in unannotated
territory is free while a missed link never is (`reports/metric_verification.md`
§3). `tests/test_counts.py` asserts these counts equal the official scorer's on
real predictions.

Two of the scorer's repairs are deliberately *not* reimplemented — collapsing
merges and capping out-degree — because node matching is one-to-one and the
callers here emit in-degree <= 1 and out-degree <= 2 by construction, so neither
can fire. `assert_wellformed` checks that rather than assuming it.
"""

from __future__ import annotations

import numpy as np

from biohub.metrics.gt import MATCH_UM, GroundTruth
from biohub.metrics.matching import match_sequence


def assert_wellformed(pred_t: np.ndarray, edges: np.ndarray) -> None:
    """Fail loudly on a graph the scorer would silently repair instead of score."""
    if len(edges) == 0:
        return
    dt = pred_t[edges[:, 1]] - pred_t[edges[:, 0]]
    if not (dt == 1).all():
        raise ValueError(f"{int((dt != 1).sum())} edges are not t -> t+1")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("duplicate (source, target) edges")
    _, in_deg = np.unique(edges[:, 1], return_counts=True)
    if in_deg.max(initial=0) > 1:
        raise ValueError("in-degree > 1 (a merge)")
    _, out_deg = np.unique(edges[:, 0], return_counts=True)
    if out_deg.max(initial=0) > 2:
        raise ValueError("out-degree > 2")


def edge_counts(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    pred_edges: np.ndarray,
    gt: GroundTruth,
    max_distance: float = MATCH_UM,
    gt_to_pred: np.ndarray | None = None,
) -> dict:
    """Edge TP/FP/FN plus the node-matching diagnostics, for one sample.

    `gt_to_pred` may be supplied when the same matching is reused across several
    edge sets over one node set, which is the common case in a sweep.
    """
    pred_edges = np.asarray(pred_edges, dtype=np.int64).reshape(-1, 2)
    assert_wellformed(pred_t, pred_edges)

    if gt_to_pred is None:
        gt_to_pred = match_sequence(pred_t, pred_zyx, gt.t, gt.zyx, max_distance)

    pred_to_gt = np.full(len(pred_t), -1, dtype=np.int64)
    matched = gt_to_pred >= 0
    pred_to_gt[gt_to_pred[matched]] = np.flatnonzero(matched)

    gt_out = np.zeros(gt.n_nodes, dtype=np.int64)
    gt_in = np.zeros(gt.n_nodes, dtype=np.int64)
    if gt.n_edges:
        np.add.at(gt_out, gt.edges[:, 0], 1)
        np.add.at(gt_in, gt.edges[:, 1], 1)

    if len(pred_edges) == 0:
        tp = valid = 0
    else:
        ms = pred_to_gt[pred_edges[:, 0]]
        mt = pred_to_gt[pred_edges[:, 1]]
        both = (ms >= 0) & (mt >= 0)

        gt_edge_set = set(map(tuple, gt.edges.tolist()))
        tp = sum(
            1
            for s, t in zip(ms[both].tolist(), mt[both].tolist())
            if (s, t) in gt_edge_set
        )
        out_valid = (ms >= 0) & (gt_out[np.maximum(ms, 0)] > 0)
        in_valid = (mt >= 0) & (gt_in[np.maximum(mt, 0)] > 0)
        valid = int((out_valid | in_valid).sum())

    return {
        "edge_tp": int(tp),
        "edge_fp": int(valid - tp),
        "edge_fn": int(gt.n_edges - tp),
        "num_pred_nodes": int(len(pred_t)),
        "n_matched_gt": int(matched.sum()),
        "node_recall": float(matched.sum() / gt.n_nodes) if gt.n_nodes else float("nan"),
    }


def to_metric_row(
    counts: dict, gt: GroundTruth, division: tuple[int, int, int] | None = None
) -> dict:
    """Turn counts into a row `score_run` can aggregate exactly.

    `division=None` means the system emitted no forks, so every ground-truth
    division is a false negative. Leaving it at zero instead would print
    `0/0/0` and hide that the division term was forfeited entirely — the score
    is the same either way, because a Jaccard of 0 contributes 0, but the
    reported counts would be wrong. There are only 151 divisions in the whole
    training set, so the counts are the measurement (CLAUDE.md).
    """
    from biohub.metrics.score import score_from_counts

    j, adj = score_from_counts(
        counts["edge_tp"], counts["edge_fp"], counts["edge_fn"],
        counts["num_pred_nodes"], gt.n_total,
    )
    div_tp, div_fp, div_fn = division if division is not None else (0, 0, len(gt.divisions()))
    return {
        "dataset": gt.dataset,
        "embryo": gt.embryo,
        "n_total": gt.n_total,
        "gt_nodes": gt.n_nodes,
        "gt_edges": gt.n_edges,
        "edge_tp": counts["edge_tp"],
        "edge_fp": counts["edge_fp"],
        "edge_fn": counts["edge_fn"],
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "num_pred_nodes": counts["num_pred_nodes"],
        "node_recall": counts["node_recall"],
        "total_node_ratio": (counts["num_pred_nodes"] - gt.n_total) / gt.n_total,
        "edge_jaccard": j,
        "adj_edge_jaccard": adj,
    }
