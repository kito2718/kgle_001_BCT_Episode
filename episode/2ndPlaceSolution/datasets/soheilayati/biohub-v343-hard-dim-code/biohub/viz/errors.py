"""Visual error diagnostics — PLAN.md §7, Wednesday 12 August.

The plan's own warning is the design constraint here: *do not diagnose 3D errors
from a 2D projection alone*. A maximum-intensity projection collapses 64 z
slices, so two cells 20 um apart in z land on top of each other and a perfectly
good link looks like a crossing.

So every figure pairs a projection with the z evidence:

  - the MIP panel is for orientation and density, never for adjudication;
  - the z-slice panel shows the actual neighbourhood of the node in question;
  - the z-vs-time panel shows what the projection hides.

Nodes and edges are drawn from the *matched* graphs, so a false positive here is
a false positive by the official rule rather than by eye.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub.metrics.gt import SCALE_ARR, GroundTruth

# Colour-blind-safe; also distinct in greyscale print.
COLOURS = {
    "tp": "#0173b2",
    "fp": "#de8f05",
    "fn": "#d55e00",
    "gt": "#029e73",
    "pred": "#cc78bc",
}


@dataclass
class EdgeVerdict:
    """Per-edge classification against the official counting rule."""

    tp: np.ndarray  # (n, 2) predicted node indices
    fp: np.ndarray
    fn: np.ndarray  # (n, 2) *ground-truth* node indices — nothing was predicted


def classify_edges(
    pred_t: np.ndarray,
    pred_edges: np.ndarray,
    gt: GroundTruth,
    gt_to_pred: np.ndarray,
) -> EdgeVerdict:
    """Split predicted and ground-truth edges into TP / FP / FN.

    Mirrors `biohub.metrics.counts.edge_counts`; the two are asserted consistent
    in `tests/test_viz.py` so a figure can never disagree with the score.
    """
    pred_to_gt = np.full(len(pred_t), -1, dtype=np.int64)
    matched = gt_to_pred >= 0
    pred_to_gt[gt_to_pred[matched]] = np.flatnonzero(matched)

    gt_out = np.zeros(gt.n_nodes, dtype=np.int64)
    gt_in = np.zeros(gt.n_nodes, dtype=np.int64)
    if gt.n_edges:
        np.add.at(gt_out, gt.edges[:, 0], 1)
        np.add.at(gt_in, gt.edges[:, 1], 1)

    gt_edge_set = set(map(tuple, gt.edges.tolist()))
    pred_edges = np.asarray(pred_edges).reshape(-1, 2)

    is_tp = np.zeros(len(pred_edges), dtype=bool)
    is_valid = np.zeros(len(pred_edges), dtype=bool)
    if len(pred_edges):
        ms, mt = pred_to_gt[pred_edges[:, 0]], pred_to_gt[pred_edges[:, 1]]
        is_tp = np.array(
            [(s >= 0 and t >= 0 and (s, t) in gt_edge_set) for s, t in zip(ms, mt)]
        )
        is_valid = ((ms >= 0) & (gt_out[np.maximum(ms, 0)] > 0)) | (
            (mt >= 0) & (gt_in[np.maximum(mt, 0)] > 0)
        )

    hit = {
        (int(pred_to_gt[s]), int(pred_to_gt[d]))
        for s, d in pred_edges[is_tp].tolist()
    }
    missed = np.array(
        [e for e in gt.edges.tolist() if tuple(e) not in hit], dtype=np.int64
    ).reshape(-1, 2)

    return EdgeVerdict(
        tp=pred_edges[is_tp],
        fp=pred_edges[is_valid & ~is_tp],
        fn=missed,
    )


def _mip(vol: np.ndarray, axis: int = 0) -> np.ndarray:
    return vol.max(axis=axis)


def plot_frame(
    ax_mip,
    ax_slice,
    vol: np.ndarray,
    t: int,
    gt: GroundTruth,
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    verdict: EdgeVerdict,
    focus_z: int | None = None,
    z_halfwidth: int = 3,
):
    """Draw one frame: a z-MIP for orientation and a z-slab for adjudication."""
    gsel = np.flatnonzero(gt.t == t)
    psel = np.flatnonzero(pred_t == t)

    ax_mip.imshow(_mip(vol), cmap="gray", interpolation="nearest")
    ax_mip.scatter(gt.zyx[gsel, 2], gt.zyx[gsel, 1], s=26, facecolors="none",
                   edgecolors=COLOURS["gt"], linewidths=1.1, label="annotated")
    ax_mip.scatter(pred_zyx[psel, 2], pred_zyx[psel, 1], s=5, c=COLOURS["pred"],
                   alpha=0.55, label="predicted")
    ax_mip.set_title(f"t={t}  z-MIP  (orientation only)", fontsize=8)
    ax_mip.set_xticks([])
    ax_mip.set_yticks([])

    if focus_z is None:
        focus_z = int(np.median(gt.zyx[gsel, 0])) if len(gsel) else vol.shape[0] // 2
    lo, hi = max(0, focus_z - z_halfwidth), min(vol.shape[0], focus_z + z_halfwidth + 1)

    ax_slice.imshow(vol[lo:hi].max(axis=0), cmap="gray", interpolation="nearest")
    in_slab_g = gsel[(gt.zyx[gsel, 0] >= lo) & (gt.zyx[gsel, 0] < hi)]
    in_slab_p = psel[(pred_zyx[psel, 0] >= lo) & (pred_zyx[psel, 0] < hi)]
    ax_slice.scatter(gt.zyx[in_slab_g, 2], gt.zyx[in_slab_g, 1], s=44,
                     facecolors="none", edgecolors=COLOURS["gt"], linewidths=1.4)
    ax_slice.scatter(pred_zyx[in_slab_p, 2], pred_zyx[in_slab_p, 1], s=9,
                     c=COLOURS["pred"], alpha=0.8)

    for name, edges, coords in (
        ("fn", verdict.fn, gt.zyx),
        ("fp", verdict.fp, pred_zyx),
        ("tp", verdict.tp, pred_zyx),
    ):
        times = gt.t if name == "fn" else pred_t
        sel = [
            k for k, (s, d) in enumerate(edges.tolist())
            if times[s] == t and lo <= coords[s, 0] < hi
        ]
        for k in sel:
            s, d = edges[k]
            ax_slice.plot(
                [coords[s, 2], coords[d, 2]], [coords[s, 1], coords[d, 1]],
                color=COLOURS[name], linewidth=1.4, alpha=0.9,
            )
    ax_slice.set_title(f"z in [{lo},{hi})  —  blue TP, orange FP, red FN", fontsize=8)
    ax_slice.set_xticks([])
    ax_slice.set_yticks([])


def plot_z_time(ax, gt: GroundTruth, verdict: EdgeVerdict, pred_t, pred_zyx):
    """z against t — the axis the projection destroys.

    Two tracks that appear to cross in a MIP are usually just separated in z;
    this panel is what distinguishes a real identity switch from that artifact.
    """
    for s, d in gt.edges.tolist():
        ax.plot([gt.t[s], gt.t[d]], [gt.zyx[s, 0], gt.zyx[d, 0]],
                color=COLOURS["gt"], linewidth=0.6, alpha=0.35)
    for s, d in verdict.fn.tolist():
        ax.plot([gt.t[s], gt.t[d]], [gt.zyx[s, 0], gt.zyx[d, 0]],
                color=COLOURS["fn"], linewidth=1.2, alpha=0.9)
    for s, d in verdict.fp.tolist():
        ax.plot([pred_t[s], pred_t[d]], [pred_zyx[s, 0], pred_zyx[d, 0]],
                color=COLOURS["fp"], linewidth=1.0, alpha=0.8)
    ax.set_xlabel("t", fontsize=8)
    ax.set_ylabel("z (voxels)", fontsize=8)
    ax.set_title("z vs t — grey GT, red missed, orange false", fontsize=8)
    ax.tick_params(labelsize=7)


def displacement_panel(ax, gt: GroundTruth):
    """Physical displacement of annotated edges. Sets the linking gate."""
    d = np.linalg.norm(
        (gt.zyx[gt.edges[:, 1]] - gt.zyx[gt.edges[:, 0]]) * SCALE_ARR, axis=1
    )
    ax.hist(d, bins=40, color=COLOURS["gt"], alpha=0.85)
    for q, style in ((0.99, "--"), (0.999, ":")):
        v = np.quantile(d, q)
        ax.axvline(v, color=COLOURS["fn"], linestyle=style, linewidth=1,
                   label=f"q{q:g} = {v:.1f} um")
    ax.set_xlabel("edge displacement (um)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)


def save_report(path: Path, fig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
