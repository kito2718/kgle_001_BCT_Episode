"""Sparse-label-safe edge labels — PLAN.md §4.3, made exact against the metric.

The label rule is not an approximation of "biologically correct"; it is the
counting rule of `biohub.metrics.counts.edge_counts` (verified equal to the
official scorer) applied edge by edge:

  positive   both endpoints' official stand-ins form an annotated GT edge
             -> emitting it is a TP
  negative   the matched source has GT out-degree > 0, OR the matched target
             has GT in-degree > 0, and the pair is not a GT edge
             -> emitting it is an FP, with certainty
  unknown    everything else -> emitting it moves nothing

The directional degree condition is the whole sparse-annotation story: an edge
out of an annotated track-END, or into an annotated track-START, is *free* —
annotation ends where a human stopped following the cell, not where the cell
died (`reports/measured_geometry.md` §5), and the scorer does not punish what
it cannot check. `Do not turn every unlabeled edge into a negative` (§4.3):
here that falls out of the metric itself rather than being a modelling choice.

Because the node set is fixed and the official matching is deterministic,
these labels are *sufficient statistics* for scoring any emitted edge subset:

  TP = emitted positives    FP = emitted negatives    FN = gt_edges - TP

so a linker or threshold sweep never re-matches anything.
`tests/test_edge_labels.py` asserts this equality against `edge_counts` on
real samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biohub.metrics.gt import MATCH_UM, GroundTruth
from biohub.metrics.matching import match_sequence

POSITIVE, NEGATIVE, UNKNOWN = 1, 0, -1


def node_gt_rows(
    sel_t: np.ndarray,
    sel_zyx_vox: np.ndarray,
    gt: GroundTruth,
    max_distance: float = MATCH_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """`(pred_to_gt, gt_to_pred)` node correspondences, via the official matching.

    Row indices on both sides, `-1` where unmatched. `sel_zyx_vox` must be the
    ROUNDED integer submission coordinates — the matching must see exactly what
    the scorer will see, not the refined floats the edge geometry uses.
    """
    gt_to_pred = match_sequence(sel_t, sel_zyx_vox, gt.t, gt.zyx, max_distance)
    pred_to_gt = np.full(len(sel_t), -1, dtype=np.int64)
    m = gt_to_pred >= 0
    pred_to_gt[gt_to_pred[m]] = np.flatnonzero(m)
    return pred_to_gt, gt_to_pred


def gt_degrees(gt: GroundTruth) -> tuple[np.ndarray, np.ndarray]:
    """(out_degree, in_degree) per GT node row."""
    gt_out = np.zeros(gt.n_nodes, dtype=np.int64)
    gt_in = np.zeros(gt.n_nodes, dtype=np.int64)
    if gt.n_edges:
        np.add.at(gt_out, gt.edges[:, 0], 1)
        np.add.at(gt_in, gt.edges[:, 1], 1)
    return gt_out, gt_in


def label_edges(
    pi: np.ndarray,
    ci: np.ndarray,
    pred_to_gt: np.ndarray,
    gt: GroundTruth,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(label, is_division, child_scored)` per edge, mirroring `edge_counts`.

    `is_division` marks positives whose GT source has out-degree 2 — the 151
    events every division report must count rather than ratio (CLAUDE.md).

    `child_scored` marks edges whose CHILD alone forces scoring (matched, with
    an annotated parent). Groups of such edges have known truth — the child's
    real parent either is in the group or is absent — which is what a
    per-child competition model can train on; parent-side-only negatives say
    nothing about which of the child's edges is right.
    """
    g_p = pred_to_gt[np.asarray(pi, dtype=np.int64)]
    g_c = pred_to_gt[np.asarray(ci, dtype=np.int64)]
    gt_out, gt_in = gt_degrees(gt)

    if gt.n_edges:
        # Pair membership via a sorted 1-D encoding of (src, dst) GT rows.
        enc = gt.edges[:, 0].astype(np.int64) * gt.n_nodes + gt.edges[:, 1]
        enc.sort()
        both = (g_p >= 0) & (g_c >= 0)
        q = g_p[both] * gt.n_nodes + g_c[both]
        pos_at = np.searchsorted(enc, q)
        pos_at[pos_at == len(enc)] = 0
        hit = enc[pos_at] == q
        positive = np.zeros(len(g_p), dtype=bool)
        positive[np.flatnonzero(both)[hit]] = True
    else:
        positive = np.zeros(len(g_p), dtype=bool)

    child_scored = (g_c >= 0) & (gt_in[np.maximum(g_c, 0)] > 0)
    scored = ((g_p >= 0) & (gt_out[np.maximum(g_p, 0)] > 0)) | child_scored
    labels = np.full(len(g_p), UNKNOWN, dtype=np.int8)
    labels[scored] = NEGATIVE
    labels[positive] = POSITIVE

    is_div = positive & (gt_out[np.maximum(g_p, 0)] == 2)
    return labels, is_div, child_scored


def counts_from_labels(labels: np.ndarray, gt: GroundTruth) -> dict:
    """Edge TP/FP/FN for an *emitted* edge set, from its labels alone.

    Valid only when the emitted set satisfies the scorer's degree constraints
    (in <= 1, out <= 2) over the same node set the labels were built on — the
    caller's linker guarantees that by construction.
    """
    tp = int((labels == POSITIVE).sum())
    fp = int((labels == NEGATIVE).sum())
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": int(gt.n_edges - tp)}
