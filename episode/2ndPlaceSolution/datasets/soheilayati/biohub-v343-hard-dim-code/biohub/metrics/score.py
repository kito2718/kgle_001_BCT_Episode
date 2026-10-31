"""Scoring wrapper and the per-sample sufficient-statistics cache — PLAN.md §8.

Two rules this module exists to enforce:

1. **Aggregate with `summarise`, never `evaluate_datasets`.** Both are in the
   official source; only `summarise` applies the node-count adjustment, and only
   `summarise` is what the organizers' own `scripts/evaluate.py` calls
   (`reports/metric_verification.md` §1).
2. **Never average per-sample scores.** The run-level score weight-averages by
   `TP+FP+FN`, and 6bba carries 84.6 % of that weight. Sweeps and bootstraps
   recompute the exact aggregate from cached counts instead.

A cached row is everything `summarise` needs plus the identity columns, so any
later resampling or subsetting is exact rather than approximate.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.io.submission import DatasetGraph
from biohub.metrics.gt import MATCH_UM, SCALE_UM, GroundTruth

# Identity columns carried alongside the official metric columns.
ID_COLUMNS = ("dataset", "embryo", "n_total", "gt_nodes", "gt_edges")


def graph_to_tracksdata(g: DatasetGraph):
    """Convert a predicted graph for the official scorer.

    Edges are inserted in **descending confidence**, matching
    `biohub.io.submission.build_submission`. The scorer truncates out-degree > 2
    by insertion order, so the scoring path and the submission path must agree
    on that order or a local score will not predict the leaderboard one
    (`reports/metric_verification.md` §5).
    """
    import polars as pl
    import tracksdata as td

    pred = td.graph.IndexedRXGraph()
    for axis in ("z", "y", "x"):
        pred.add_node_attr_key(axis, pl.Float64, 0.0)
    remap = {}
    for nid, (t, z, y, x) in g.nodes.items():
        remap[nid] = pred.add_node(
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        )
    for s, d, _ in sorted(g.edges, key=lambda e: -e[2]):
        pred.add_edge(remap[s], remap[d], {})
    return pred


def score_graph(pred: DatasetGraph, gt: GroundTruth, gt_graph=None) -> dict:
    """Score one predicted graph. Returns a cacheable sufficient-statistics row.

    `gt_graph` may be passed in when the same ground truth is scored repeatedly
    (a threshold sweep), since building it is the expensive part.
    """
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    pred_g = graph_to_tracksdata(pred)
    gt_g = gt.to_tracksdata() if gt_graph is None else gt_graph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = evaluate(pred_g, gt_g, scale=SCALE_UM, max_distance=MATCH_UM)
        recall = node_recall(pred_g, gt_g) if pred_g.num_edges() else 0.0

    row = per_sample_metrics(er, gt.n_total, recall)
    row.update(
        dataset=gt.dataset,
        embryo=gt.embryo,
        n_total=gt.n_total,
        gt_nodes=gt.n_nodes,
        gt_edges=gt.n_edges,
    )
    return row


def score_run(rows: list[dict] | pd.DataFrame) -> dict:
    """Exact run-level score. Thin wrapper so no caller reimplements it."""
    from tracking_cellmot.metrics import summarise

    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return summarise(list(rows))


# Mirrors `tracking_cellmot.metrics.ADJUSTMENT_ALPHA`. Duplicated as a plain
# constant so the counts-only path stays pure numpy — importing the official
# module pulls in tracksdata, which costs several seconds per process and would
# be paid inside every sweep worker. `tests/test_metrics_layer.py` asserts the
# two are equal, so a change upstream fails loudly instead of drifting.
ADJUSTMENT_ALPHA = 0.1


def score_from_counts(
    edge_tp: int,
    edge_fp: int,
    edge_fn: int,
    num_pred_nodes: int,
    n_total: float,
) -> tuple[float, float]:
    """`(edge_jaccard, adj_edge_jaccard)` for one sample, from counts alone.

    Lets a sweep evaluate thousands of operating points without rebuilding a
    graph. Mirrors `per_sample_metrics`; asserted equal to it in the tests.
    """
    denom = edge_tp + edge_fp + edge_fn
    if denom <= 0:
        return float("nan"), float("nan")
    j = edge_tp / denom
    if not n_total > 0:
        return j, float("nan")
    ratio = (num_pred_nodes - n_total) / n_total
    return j, max(0.0, j * (1 - ADJUSTMENT_ALPHA * ratio))


def save_rows(rows: list[dict] | pd.DataFrame, path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def by_stratum(rows: pd.DataFrame, column: str = "embryo") -> pd.DataFrame:
    """Exact run-level score computed separately within each stratum."""
    out = []
    for key, sub in rows.groupby(column, sort=True):
        s = score_run(sub)
        s[column] = key
        s["weight"] = float(
            (sub["edge_tp"] + sub["edge_fp"] + sub["edge_fn"]).sum()
        )
        out.append(s)
    df = pd.DataFrame(out)
    total = df["weight"].sum()
    df["weight_share"] = df["weight"] / total if total else np.nan
    cols = [column, "n", "score", "adj_edge_jaccard", "edge_jaccard", "node_recall",
            "division_tp", "division_fp", "division_fn", "weight", "weight_share"]
    return df[[c for c in cols if c in df.columns]]
