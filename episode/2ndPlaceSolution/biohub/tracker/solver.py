"""Solver v1.1 node selection: the deployable core of stage-04 §11 / facts 47–48.

Three pure functions shared by the measurement rig (`scripts/verify_solver.py`)
and the deploy path (`pipeline.learned.size_select_solve`) — the same
discipline as `edges.select_nodes`: one implementation, so the deployed
selection and the measured selection cannot drift apart.

The mechanism, as measured (official 0.8571 → 0.8748, fact 47/48):

  support   best incoming + best outgoing edge probability per node —
            evidence the node sits on a track. Masked, an edge only counts
            if its OTHER endpoint is currently selected (junk loses phantom
            support from unselected superset neighbours); `wta` further
            requires the parent to be the child's argmax among selected
            parents — the claims linker's own semantics, so support only
            counts links that could actually emit.
  cut       NMS per frame in rank order, then either the per-frame quota
            ("frame") or ONE ranking cut per sample at n_target ("global") —
            frames compete for slots through a single price. The per-frame
            quota was a binding constraint worth +0.0131 (fact 47a).
  solve_wta the confirmed composition: rank by score + support, anchor the
            node count at the per-frame control's total, then iterate the
            global cut twice with WTA-masked support at lam = 1.0.
"""

from __future__ import annotations

import numpy as np

NMS_UM = 5.0          # fact 15 — the deployed suppression radius
SUPERSET_MULT = 2.0   # fact 48 — 3x measured +0.0003, not worth the compute
SOLVER_LAM = 1.0      # re-peaks at 1.0 once support is WTA-honest (fact 48)
SOLVER_ITERS = 2      # 5 mildly overshoots (fact 47b)
CTRL_LAM = 0.5        # the fact-43 control ranking that anchors n_target


def support(n_sup, pi, ci, p, mask=None, wta=False):
    """best incoming + best outgoing edge p per superset node.

    mask=None is fact 43's unmasked support. With a mask, an edge
    contributes to its child's incoming term only if the parent is selected,
    and to its parent's outgoing term only if the child is selected — the
    value of a node GIVEN the rest of the current selection.

    wta additionally requires the parent to be the child's argmax among
    selected parents (p >= the child's best_in): a non-argmax edge is one
    the claims linker will never emit, so it is not support. For an
    UNSELECTED candidate parent the same test reads "would beat the child's
    current argmax if added". Requires a mask.
    """
    best_out = np.zeros(n_sup)
    best_in = np.zeros(n_sup)
    if len(pi):
        ok_in = slice(None) if mask is None else mask[pi]
        np.maximum.at(best_in, ci[ok_in], p[ok_in])
        if wta:
            ok_out = mask[ci] & (p >= best_in[ci] - 1e-12)
        else:
            ok_out = slice(None) if mask is None else mask[ci]
        np.maximum.at(best_out, pi[ok_out], p[ok_out])
    return best_in + best_out


def tracklet_terms(n_sup, pi, ci, p, mask):
    """Three- and five-frame through-path evidence for every candidate.

    ``through3`` requires one supported edge on each side of the node and
    uses twice the weaker side, so one excellent accidental edge cannot earn
    full track support. ``through5`` applies the same bottleneck idea to two
    edges of history and two of future; each side is the best mean edge
    probability over that two-hop path, keeping its scale comparable to the
    one-hop probabilities.

    Internal path nodes must be selected.  The candidate being scored need
    not be: this is the counterfactual evidence it would receive if admitted.
    Future edges obey the same WTA rule as ``support(..., wta=True)``.
    """
    mask = np.asarray(mask, dtype=bool)
    best_in = np.zeros(n_sup, dtype=np.float64)
    best_out = np.zeros(n_sup, dtype=np.float64)
    if len(pi) == 0:
        return best_in, best_out

    ok_in = mask[pi]
    np.maximum.at(best_in, ci[ok_in], p[ok_in])
    ok_out = mask[ci] & (p >= best_in[ci] - 1e-12)
    np.maximum.at(best_out, pi[ok_out], p[ok_out])
    through3 = 2.0 * np.minimum(best_in, best_out)

    hist2 = np.zeros(n_sup, dtype=np.float64)
    h_ok = mask[pi] & (best_in[pi] > 0)
    h_val = 0.5 * (p[h_ok] + best_in[pi[h_ok]])
    np.maximum.at(hist2, ci[h_ok], h_val)

    fut2 = np.zeros(n_sup, dtype=np.float64)
    f_ok = ok_out & (best_out[ci] > 0)
    f_val = 0.5 * (p[f_ok] + best_out[ci[f_ok]])
    np.maximum.at(fut2, pi[f_ok], f_val)
    through5 = 2.0 * np.minimum(hist2, fut2)
    return through3, through5


def cut(mode, rank_score, sup_t, zyx_int, budget, n_target, nms_um=NMS_UM):
    """Selected superset rows, frame-major and NMS-rank-ordered within each
    frame — the node ordering every measured row was scored with (matching
    tie-breaks can be order-sensitive). NMS per frame in rank order, then
    either the per-frame quota or one global cut at n_target."""
    from biohub.candidates.nms import select_frame

    survivors = []
    for t in np.unique(sup_t):
        r = np.flatnonzero(sup_t == t)
        kept = r[select_frame(zyx_int[r], rank_score[r], nms_um)]
        survivors.append(kept[:budget] if mode == "frame" else kept)
    rows = np.concatenate(survivors) if survivors else np.zeros(0, dtype=np.int64)
    if mode == "global" and len(rows) > n_target:
        top = np.argsort(-rank_score[rows], kind="stable")[:n_target]
        rows = rows[np.sort(top)]  # membership by rank, order frame-major
    return rows


def solve_wta(sup_score, sup_t, sup_zyx_int, pi, ci, p, budget,
              lam=SOLVER_LAM, iters=SOLVER_ITERS, ctrl_lam=CTRL_LAM,
              nms_um=NMS_UM):
    """The v1.1 operating selection over a superset: rows into the superset.

    n_target — the exact node count submitted — is the per-frame control
    cut's total at the fact-43 ranking (score + 0.5·unmasked support), which
    is the count every measured operating point carried; the global cuts
    then trade MEMBERSHIP freely at that fixed count (fact 46).
    """
    n = len(sup_score)
    base = support(n, pi, ci, p)
    n_target = len(cut("frame", sup_score + ctrl_lam * base,
                       sup_t, sup_zyx_int, budget, 0, nms_um))
    rows = cut("global", sup_score + lam * base,
               sup_t, sup_zyx_int, budget, n_target, nms_um)
    for _ in range(iters):
        mask = np.zeros(n, dtype=bool)
        mask[rows] = True
        supp = support(n, pi, ci, p, mask, wta=True)
        rows = cut("global", sup_score + lam * supp,
                   sup_t, sup_zyx_int, budget, n_target, nms_um)
    return rows
