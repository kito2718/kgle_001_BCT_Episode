"""Pass-1 track context for pass-2 edge scoring — the multi-frame edge model v1.

Stage-04 §6.4's diagnosis: the residual association errors are not per-edge
decidable — the 2,529 far-parent mispicks have p_med 0.03 because the child's
flow points at junk, and every static feature round (embeddings, grandparent
geometry, flow coherence) measured zero. What no per-edge feature can see is
the *decision structure* of a first linking pass: which parents sit on tracks,
what velocity those tracks predict, whether the child keeps moving coherently
at t+2 through THIS parent, and who else is claiming whom.

So: link once (the deployed claims chain), then describe every candidate edge
by the fragments that pass produced. These columns are computed from the
pass-1 emitted edge set + the dense pass-1 probabilities only — no ground
truth anywhere, so the builder is deploy-side safe.

Feature notes, tied to the measured failure classes:

  cv_resid_um          | child − (parent + parent's own last step) | — a
                       constant-velocity motion prior that does not use the
                       child's flow at all. The mispick class is exactly
                       "child flow wrong"; this is the alternative signal.
  child_cont_resid_um  | (grandchild − child) − (child − parent) | — the only
                       column that judges the child's t+2 continuation
                       *through this specific parent*: a wrong parent bends
                       the two-step path, a right one keeps it straight.
  rival_p /            competition made explicit: the best OTHER claim on
  par_out_other        this parent (winner-take-all p1_adj, self-excluded),
                       and the parent's best emitted edge to another child.
  sup_in_p / sup_out_c the reselection support halves (max raw p in/out) —
                       dense, threshold-free history/future evidence.

All pass-1 quantities are OOF with respect to the sample (fold-k GBM never
saw fold k), so a model trained on these columns keeps the rig's discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order is part of the pass-2 model contract, exactly like FEATURE_COLUMNS.
CONTEXT_COLUMNS = (
    "p1",                   # pass-1 OOF binary p for this edge
    "p1_adj",               # pass-1 winner-take-all calibrated p (0 unless argmax)
    "p1_emitted",           # 1.0 if pass 1 emitted this edge
    "par_in_p",             # parent's emitted incoming claim p (0 = no history)
    "par_up_len",           # emitted-chain length ending at the parent
    "cv_resid_um",          # |child − (parent + parent's last step)|; NaN: no history
    "cv_resid_z_um",
    "cv_resid_xy_um",
    "par_emb_step",         # emb_l2 of the parent's emitted incoming edge; NaN: none
    "child_out_p",          # child's best emitted outgoing p at t+1→t+2 (0 = none)
    "child_out_n",          # how many emitted continuations the child has (0/1/2)
    "child_cont_resid_um",  # |(grandchild−child) − (child−parent)|; NaN: no grandchild
    "sup_in_p",             # max raw p over edges INTO the parent (history support)
    "sup_out_c",            # max raw p over edges OUT of the child (future support)
    "rival_p",              # best OTHER child's claim on this parent (self-excluded)
    "par_out_other",        # parent's best emitted out-edge to ANOTHER child
)


def _group_top2(keys: np.ndarray, vals: np.ndarray, n_groups: int):
    """Per group: (top1 value, row index of top1, top2 value).

    Rows sorted by (key, -val) with a stable lexsort, so ties break
    deterministically by row order — the same convention as the linker.
    """
    top1_val = np.zeros(n_groups)
    top1_row = np.full(n_groups, -1, dtype=np.int64)
    top2_val = np.zeros(n_groups)
    if len(keys) == 0:
        return top1_val, top1_row, top2_val
    o = np.lexsort((-vals, keys))
    first = np.r_[True, keys[o][1:] != keys[o][:-1]]
    idx = np.arange(len(o))
    start = np.maximum.accumulate(np.where(first, idx, 0))
    rank = idx - start
    r1, r2 = rank == 0, rank == 1
    top1_val[keys[o][r1]] = vals[o][r1]
    top1_row[keys[o][r1]] = o[r1]
    top2_val[keys[o][r2]] = vals[o][r2]
    return top1_val, top1_row, top2_val


def build_context(
    edges: pd.DataFrame,
    node_t: np.ndarray,
    node_um: np.ndarray,
    p1: np.ndarray,
    p1_adj: np.ndarray,
    emitted_rows: np.ndarray,
) -> pd.DataFrame:
    """One row per edge, aligned with `edges`, columns = CONTEXT_COLUMNS.

    `edges` needs `pi`, `ci`, `emb_l2`; `node_um` is the UNROUNDED refined
    position in micrometres (the same geometry convention as `build_edges`);
    `emitted_rows` indexes into `edges` (the pass-1 claims output).
    """
    n_nodes = len(node_t)
    n_edges = len(edges)
    if n_edges == 0:
        return pd.DataFrame(
            {c: pd.Series(dtype=np.float32) for c in CONTEXT_COLUMNS})

    pi = edges["pi"].to_numpy(np.int64)
    ci = edges["ci"].to_numpy(np.int64)
    emb_l2 = edges["emb_l2"].to_numpy(np.float32)
    p1 = np.asarray(p1, dtype=np.float64)
    p1_adj = np.asarray(p1_adj, dtype=np.float64)

    emitted = np.zeros(n_edges, dtype=bool)
    emitted[emitted_rows] = True
    e_pi, e_ci = pi[emitted_rows], ci[emitted_rows]
    e_p = p1_adj[emitted_rows]

    # ---- per-node pass-1 structure ----------------------------------------
    # in-degree <= 1 by claims construction, so plain assignment is safe
    parent_of = np.full(n_nodes, -1, dtype=np.int64)
    in_p = np.zeros(n_nodes)
    in_row = np.full(n_nodes, -1, dtype=np.int64)
    parent_of[e_ci] = e_pi
    in_p[e_ci] = e_p
    in_row[e_ci] = np.asarray(emitted_rows, dtype=np.int64)

    has_par = parent_of >= 0
    vel = np.full((n_nodes, 3), np.nan)
    vel[has_par] = node_um[has_par] - node_um[parent_of[has_par]]

    up_len = np.zeros(n_nodes, dtype=np.int64)
    for t in np.unique(node_t):  # ascending, parents already resolved
        rows = np.flatnonzero((node_t == t) & has_par)
        up_len[rows] = up_len[parent_of[rows]] + 1

    # child side of each node: best emitted outgoing edge
    out_n = np.zeros(n_nodes, dtype=np.int64)
    out_p = np.zeros(n_nodes)
    if len(e_pi):
        np.add.at(out_n, e_pi, 1)
        np.maximum.at(out_p, e_pi, e_p)
    et1_val, et1_row_local, _et2 = _group_top2(e_pi, e_p, n_nodes)
    # _group_top2 indexes rows within the emitted subset; lift to edge rows
    e_rows = np.asarray(emitted_rows, dtype=np.int64)
    e_top1_row = np.full(n_nodes, -1, dtype=np.int64)
    m = et1_row_local >= 0
    e_top1_row[m] = e_rows[et1_row_local[m]]
    e_top2_val = _et2
    # best emitted child per node, read off the same top-1 rows
    best_child = np.full(n_nodes, -1, dtype=np.int64)
    if m.any():
        best_child[m] = ci[e_top1_row[m]]

    # dense support: max raw p into / out of every node (self-free per edge:
    # an edge's own row has the parent as pi and the child as ci, so it never
    # contributes to sup_in[parent] or sup_out[child])
    sup_in = np.zeros(n_nodes)
    sup_out = np.zeros(n_nodes)
    np.maximum.at(sup_in, ci, p1)
    np.maximum.at(sup_out, pi, p1)

    # rival claims: top-2 winner-take-all p1_adj per parent over ALL edges
    a_top1_val, a_top1_row, a_top2_val = _group_top2(pi, p1_adj, n_nodes)

    # ---- per-edge assembly --------------------------------------------------
    rows_idx = np.arange(n_edges)

    pred = node_um[pi] + vel[pi]              # NaN where the parent has no history
    dvec = node_um[ci] - pred
    cv = np.sqrt((dvec ** 2).sum(axis=1))
    cv_z = np.abs(dvec[:, 0])
    cv_xy = np.hypot(dvec[:, 1], dvec[:, 2])

    pes = np.full(n_edges, np.nan, dtype=np.float64)
    mrow = in_row[pi] >= 0
    pes[mrow] = emb_l2[in_row[pi][mrow]]

    h = best_child[ci]
    ccr = np.full(n_edges, np.nan)
    mh = h >= 0
    if mh.any():
        step_next = node_um[h[mh]] - node_um[ci[mh]]
        step_this = node_um[ci[mh]] - node_um[pi[mh]]
        ccr[mh] = np.sqrt(((step_next - step_this) ** 2).sum(axis=1))

    rival = np.where(rows_idx == a_top1_row[pi], a_top2_val[pi], a_top1_val[pi])
    par_out_other = np.where(rows_idx == e_top1_row[pi], e_top2_val[pi], out_p[pi])

    return pd.DataFrame({
        "p1": p1.astype(np.float32),
        "p1_adj": p1_adj.astype(np.float32),
        "p1_emitted": emitted.astype(np.float32),
        "par_in_p": in_p[pi].astype(np.float32),
        "par_up_len": up_len[pi].astype(np.float32),
        "cv_resid_um": cv.astype(np.float32),
        "cv_resid_z_um": cv_z.astype(np.float32),
        "cv_resid_xy_um": cv_xy.astype(np.float32),
        "par_emb_step": pes.astype(np.float32),
        "child_out_p": out_p[ci].astype(np.float32),
        "child_out_n": out_n[ci].astype(np.float32),
        "child_cont_resid_um": ccr.astype(np.float32),
        "sup_in_p": sup_in[pi].astype(np.float32),
        "sup_out_c": sup_out[ci].astype(np.float32),
        "rival_p": rival.astype(np.float32),
        "par_out_other": par_out_other.astype(np.float32),
    })
