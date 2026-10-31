"""Division pair hypotheses — PLAN §6.1 / §9.9, the E3' independent division score.

Fact 44 measured why the fork decision cannot stay a threshold on the
continuation calibration: retraining the continuation score shifts the
second-claim distribution and the division term pays ~0.0014 of coupling
loss. Fact 36 measured that the failure is selection, not representation —
141 of 151 divisions already have distinct pool candidates. So the division
decision gets its OWN model, scoring the triple (parent, d1, d2) jointly:
both daughters' evidence enters at once, which no per-edge score can see.

This module enumerates the triples and builds their features. It is
deploy-side: no ground truth anywhere (labels live in
`scripts/build_pair_dataset.py`), and every input is OOF with respect to the
sample — the raw edge p comes from the fold GBM that never saw it.

Universe: per parent, the top `TOP_M` in-gate children by the hybrid key
max(raw p, calibrated argmax p) — so winner-take-all claimants always enter —
all unordered pairs, kept when the weaker edge clears `P_FLOOR` (junk–junk
pairs are the overwhelming mass and carry no decision). Claims pairs — both
children's argmax is this parent — bypass the floor: they are exactly the
pairs the τ2 dial rules on today, and the pair model must extend that
decision's reach, never shrink it.

Feature families, per PLAN §6.1:
  backward convergence   flow-centred residuals of both edges
  separation/symmetry    daughter separation, parent-vs-midpoint residual,
                         opposition cosine, displacement asymmetry
  appearance             sister similarity, parent-daughter similarity
                         (pairwise only — fact 38)
  future                 each daughter's best outgoing raw p into t+2
  competition            each daughter's best alternative parent, the
                         parent's best third child, claim structure
  history                parent's best incoming raw p, grandparent residual
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biohub.tracker.context import _group_top2
from biohub.tracker.linker import emit_claims

TOP_M = 8        # children per parent entering pair enumeration
P_FLOOR = 0.005  # weaker-edge floor on the ranking key; claims pairs bypass it

# Order is part of the pair-model contract, exactly like FEATURE_COLUMNS.
PAIR_FEATURES = (
    # edge probabilities, daughter 1 = higher ranking key of the two
    "p1", "p2",             # raw fold-GBM edge p
    "padj1", "padj2",       # winner-take-all calibrated p (0 unless argmax)
    "argmax1", "argmax2",   # this parent is the daughter's argmax parent
    # backward convergence — flow-centred residuals (fact 26 geometry)
    "resid1_um", "resid2_um",
    # raw displacement of each daughter edge (division edges move ~3x, fact 14)
    "dist1_um", "dist2_um",
    "flow1_mag", "flow2_mag",
    # joint geometry
    "sep_um", "sep_z_um", "sep_xy_um",   # daughter separation (q10 6.36 um)
    "mid_resid_um",                      # |parent - daughters' midpoint|
    "mid_cv_um",                         # |extrapolated parent - midpoint| —
                                         # the external-lessons pre-registered
                                         # symmetry probe (parent + its own
                                         # last emitted step; NaN, no history)
    "par_step_um",                       # |parent's last emitted step|
    "opp_cos",                           # cos of the two parent->daughter vectors
    "disp_asym",                         # |d1-d2| / (d1+d2)
    # appearance (pairwise only, fact 38)
    "emb_cos_12", "emb_l2_12",           # sister similarity
    "emb_cos_p1", "emb_cos_p2",          # parent-daughter similarity
    # competition
    "alt1_p", "alt2_p",     # daughter's best OTHER parent's raw p
    "par_third_p",          # parent's best child outside this pair
    "n_kids",               # parent's in-gate children
    # future continuation of each daughter into t+2
    "cont1_p", "cont2_p",
    # parent history and scalars
    "par_in_p",             # best raw p into the parent (track behind it)
    "par_gp_um",            # parent's grandparent residual (NaN in frame 1)
    "score_par", "width_par", "near_par", "dens_par", "bnd_par",
    # daughter scalars
    "score1", "score2", "width1", "width2", "near1", "near2",
)

# Post-split evidence.  These columns follow each proposed daughter through
# the same winner-take-all edge graph into t+2 and t+3.  They are optional so
# the v0 pair model and all of its stored measurements remain byte-identical.
TEMPORAL_PAIR_FEATURES = (
    "fut1_p", "fut2_p", "fut1_margin", "fut2_margin",
    "fut1_emits", "fut2_emits", "fut_both",
    "fut_sep_um", "fut_sep_growth_um", "fut_sep_dir_cos",
    "fut_step1_um", "fut_step2_um", "fut_step_cos",
    "fut_accel1_um", "fut_accel2_um",
    "fut_mid_step_um", "fut_mid_accel_um", "fut_mid_history_resid_um",
    "deep1_p", "deep2_p", "deep_both",
    "deep_sep_um", "deep_sep_growth_um", "deep_step_cos",
    "deep_mid_accel_um",
)


def _temporal_pair_features(
    par: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    pi: np.ndarray,
    ci: np.ndarray,
    p_adj: np.ndarray,
    node_um: np.ndarray,
    vel: np.ndarray | None,
) -> pd.DataFrame:
    """Winner-take-all daughter continuations for pair hypotheses."""
    n_nodes = len(node_um)
    top_p, top_row, second_p = _group_top2(pi, p_adj, n_nodes)
    best_child = np.full(n_nodes, -1, dtype=np.int64)
    live = (top_row >= 0) & (top_p > 0)
    best_child[live] = ci[top_row[live]]

    g1, g2 = best_child[c1], best_child[c2]
    h1 = np.full(len(par), -1, dtype=np.int64)
    h2 = np.full(len(par), -1, dtype=np.int64)
    has1, has2 = g1 >= 0, g2 >= 0
    h1[has1] = best_child[g1[has1]]
    h2[has2] = best_child[g2[has2]]
    both = has1 & has2
    deep1, deep2 = h1 >= 0, h2 >= 0
    deep_both = deep1 & deep2

    def nan_vec():
        return np.full((len(par), 3), np.nan, dtype=np.float64)

    step1, step2 = nan_vec(), nan_vec()
    step1[has1] = node_um[g1[has1]] - node_um[c1[has1]]
    step2[has2] = node_um[g2[has2]] - node_um[c2[has2]]
    deep_step1, deep_step2 = nan_vec(), nan_vec()
    deep_step1[deep1] = node_um[h1[deep1]] - node_um[g1[deep1]]
    deep_step2[deep2] = node_um[h2[deep2]] - node_um[g2[deep2]]

    def norm(v):
        return np.sqrt(np.sum(v * v, axis=1))

    def cosine(a, b):
        return np.einsum("ij,ij->i", a, b) / np.maximum(norm(a) * norm(b), 1e-6)

    u0, u1, u2 = node_um[par], node_um[c1], node_um[c2]
    sep1 = u1 - u2
    sep1_n = norm(sep1)
    sep2 = nan_vec()
    sep2[both] = node_um[g1[both]] - node_um[g2[both]]
    sep2_n = norm(sep2)
    sep3 = nan_vec()
    sep3[deep_both] = node_um[h1[deep_both]] - node_um[h2[deep_both]]
    sep3_n = norm(sep3)

    mid1 = 0.5 * (u1 + u2)
    mid2 = nan_vec()
    mid2[both] = 0.5 * (node_um[g1[both]] + node_um[g2[both]])
    mid3 = nan_vec()
    mid3[deep_both] = 0.5 * (node_um[h1[deep_both]] + node_um[h2[deep_both]])
    mid_step1 = mid2 - mid1
    initial_mid_step = mid1 - u0
    mid_history = nan_vec()
    if vel is not None:
        mid_history = mid2 - (mid1 + vel[par])

    initial1, initial2 = u1 - u0, u2 - u0
    f32 = np.float32
    return pd.DataFrame({
        "fut1_p": top_p[c1].astype(f32),
        "fut2_p": top_p[c2].astype(f32),
        "fut1_margin": (top_p[c1] - second_p[c1]).astype(f32),
        "fut2_margin": (top_p[c2] - second_p[c2]).astype(f32),
        "fut1_emits": (top_p[c1] >= 0.47).astype(f32),
        "fut2_emits": (top_p[c2] >= 0.47).astype(f32),
        "fut_both": both.astype(f32),
        "fut_sep_um": sep2_n.astype(f32),
        "fut_sep_growth_um": (sep2_n - sep1_n).astype(f32),
        "fut_sep_dir_cos": cosine(sep1, sep2).astype(f32),
        "fut_step1_um": norm(step1).astype(f32),
        "fut_step2_um": norm(step2).astype(f32),
        "fut_step_cos": cosine(step1, step2).astype(f32),
        "fut_accel1_um": norm(step1 - initial1).astype(f32),
        "fut_accel2_um": norm(step2 - initial2).astype(f32),
        "fut_mid_step_um": norm(mid_step1).astype(f32),
        "fut_mid_accel_um": norm(mid_step1 - initial_mid_step).astype(f32),
        "fut_mid_history_resid_um": norm(mid_history).astype(f32),
        "deep1_p": np.where(deep1, top_p[np.maximum(g1, 0)], 0).astype(f32),
        "deep2_p": np.where(deep2, top_p[np.maximum(g2, 0)], 0).astype(f32),
        "deep_both": deep_both.astype(f32),
        "deep_sep_um": sep3_n.astype(f32),
        "deep_sep_growth_um": (sep3_n - sep2_n).astype(f32),
        "deep_step_cos": cosine(deep_step1, deep_step2).astype(f32),
        "deep_mid_accel_um": norm((mid3 - mid2) - mid_step1).astype(f32),
    })


def enumerate_pairs(
    edges: pd.DataFrame,
    p_raw: np.ndarray,
    p_adj: np.ndarray,
    node_t: np.ndarray,
    node_um: np.ndarray,
    node_scalars: dict[str, np.ndarray],
    node_emb: np.ndarray,
    vel: np.ndarray | None = None,
    top_m: int = TOP_M,
    p_floor: float = P_FLOOR,
    temporal: bool = False,
) -> pd.DataFrame:
    """One row per (parent, d1, d2) hypothesis with `PAIR_FEATURES`.

    `edges` is `build_edges` output over the selected set; `p_raw` the fold
    GBM's per-edge p; `p_adj` the winner-take-all calibrated p (nonzero only
    on each child's argmax edge). `node_scalars` needs score / peak_width_um /
    near_um / density_10um / boundary_um arrays over the selected set;
    `node_um` is the UNROUNDED refined position in micrometres. `vel` is the
    per-node last emitted step in micrometres from the CONTINUATION-ONLY
    claims pass (NaN rows where the node has no emitted in-edge) — computed
    from `emit_claims(..., tau2=2.0)` so it cannot depend on any fork
    decision this model is about to make.

    Returns bookkeeping columns (`t`, `par`, `c1`, `c2`, `e1`, `e2`,
    `is_claims_pair`) alongside the features; `par`/`c1`/`c2` index the
    selected-node array, `e1`/`e2` the edge table.
    """
    n_nodes = len(node_t)
    n_edges = len(edges)
    cols = ["t", "par", "c1", "c2", "e1", "e2", "is_claims_pair",
            *PAIR_FEATURES,
            *(TEMPORAL_PAIR_FEATURES if temporal else ())]
    if n_edges == 0:
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in cols})

    pi = edges["pi"].to_numpy(np.int64)
    ci = edges["ci"].to_numpy(np.int64)
    p_raw = np.asarray(p_raw, dtype=np.float64)
    p_adj = np.asarray(p_adj, dtype=np.float64)
    gp_um = edges["grandparent_um"].to_numpy(np.float32)

    # ---- per-node aggregates ----------------------------------------------
    # child's top-2 parents by raw p -> best-alternative per edge
    c_top1, c_top1_row, c_top2 = _group_top2(ci, p_raw, n_nodes)
    # future/history support: best raw p out of / into every node
    sup_out = np.zeros(n_nodes)
    sup_in = np.zeros(n_nodes)
    np.maximum.at(sup_out, pi, p_raw)
    np.maximum.at(sup_in, ci, p_raw)

    # ---- rank children within each parent ---------------------------------
    # The key is max(raw p, calibrated argmax p): an argmax claimant always
    # outranks junk however its raw p compares, so every pair the old τ2
    # dial could rule on is enumerable — the pair model must strictly extend
    # the fork decision's reach, never silently shrink it.
    key = np.maximum(p_raw, p_adj)
    o = np.lexsort((-key, pi))
    first = np.r_[True, pi[o][1:] != pi[o][:-1]]
    idx = np.arange(n_edges)
    start = np.maximum.accumulate(np.where(first, idx, 0))
    rank = idx - start
    n_kids = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(n_kids, pi, 1)

    # per-parent top-M edge rows and their key, NaN/-1 padded
    top_rows = np.full((n_nodes, top_m), -1, dtype=np.int64)
    top_p = np.full((n_nodes, top_m), np.nan)
    m_top = rank < top_m
    top_rows[pi[o][m_top], rank[m_top]] = o[m_top]
    top_p[pi[o][m_top], rank[m_top]] = key[o][m_top]

    # ---- pair assembly: rank-pair joins -----------------------------------
    e1_parts: list[np.ndarray] = []
    e2_parts: list[np.ndarray] = []
    third_parts: list[np.ndarray] = []
    for r1 in range(top_m - 1):
        for r2 in range(r1 + 1, top_m):
            par = np.flatnonzero(top_rows[:, r2] >= 0)  # implies r1 present
            if not len(par):
                continue
            a = top_rows[par, r1]
            b = top_rows[par, r2]
            keep_ranks = [r for r in range(min(3, top_m))
                          if r not in (r1, r2)]
            third = (np.nan_to_num(top_p[par][:, keep_ranks], nan=0.0)
                     .max(axis=1) if keep_ranks
                     else np.zeros(len(par)))
            e1_parts.append(a)
            e2_parts.append(b)
            third_parts.append(third)
    if not e1_parts:
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in cols})

    e1 = np.concatenate(e1_parts)
    e2 = np.concatenate(e2_parts)
    third = np.concatenate(third_parts)

    # daughter 1 = higher ranking key (rank order within a parent)
    is_claims = (p_adj[e1] > 0) & (p_adj[e2] > 0)
    keep = (key[e2] >= p_floor) | is_claims
    e1, e2, third, is_claims = e1[keep], e2[keep], third[keep], is_claims[keep]

    par = pi[e1]
    c1 = ci[e1]
    c2 = ci[e2]

    # ---- joint geometry ----------------------------------------------------
    pu, u1, u2 = node_um[par], node_um[c1], node_um[c2]
    sep = u1 - u2
    sep_um = np.linalg.norm(sep, axis=1)
    v1 = u1 - pu
    v2 = u2 - pu
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    opp_cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-6)
    mid = (u1 + u2) / 2.0
    mid_resid = np.linalg.norm(mid - pu, axis=1)
    if vel is None:
        mid_cv = np.full(len(par), np.nan)
        par_step = np.full(len(par), np.nan)
    else:
        mid_cv = np.linalg.norm(mid - (pu + vel[par]), axis=1)
        par_step = np.linalg.norm(vel[par], axis=1)
    d1 = edges["dist_um"].to_numpy(np.float32)
    disp_asym = np.abs(d1[e1] - d1[e2]) / np.maximum(d1[e1] + d1[e2], 1e-6)

    # ---- appearance --------------------------------------------------------
    ep, ea, eb = (node_emb[par].astype(np.float32),
                  node_emb[c1].astype(np.float32),
                  node_emb[c2].astype(np.float32))

    def _cos(x, y):
        nn = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        return np.einsum("ij,ij->i", x, y) / np.maximum(nn, 1e-6)

    # ---- competition -------------------------------------------------------
    # best alternative parent: the child's top-1 unless this edge IS it
    alt1 = np.where(e1 == c_top1_row[c1], c_top2[c1], c_top1[c1])
    alt2 = np.where(e2 == c_top1_row[c2], c_top2[c2], c_top1[c2])

    sc = node_scalars
    f32 = np.float32
    out = pd.DataFrame({
        "t": edges["t"].to_numpy(np.int32)[e1],
        "par": par.astype(np.int32),
        "c1": c1.astype(np.int32),
        "c2": c2.astype(np.int32),
        "e1": e1.astype(np.int32),
        "e2": e2.astype(np.int32),
        "is_claims_pair": is_claims,
        "p1": p_raw[e1].astype(f32),
        "p2": p_raw[e2].astype(f32),
        "padj1": p_adj[e1].astype(f32),
        "padj2": p_adj[e2].astype(f32),
        "argmax1": (p_adj[e1] > 0).astype(f32),
        "argmax2": (p_adj[e2] > 0).astype(f32),
        "resid1_um": edges["resid_um"].to_numpy(f32)[e1],
        "resid2_um": edges["resid_um"].to_numpy(f32)[e2],
        "dist1_um": d1[e1],
        "dist2_um": d1[e2],
        "flow1_mag": edges["flow_mag_um"].to_numpy(f32)[e1],
        "flow2_mag": edges["flow_mag_um"].to_numpy(f32)[e2],
        "sep_um": sep_um.astype(f32),
        "sep_z_um": np.abs(sep[:, 0]).astype(f32),
        "sep_xy_um": np.hypot(sep[:, 1], sep[:, 2]).astype(f32),
        "mid_resid_um": mid_resid.astype(f32),
        "mid_cv_um": mid_cv.astype(f32),
        "par_step_um": par_step.astype(f32),
        "opp_cos": opp_cos.astype(f32),
        "disp_asym": disp_asym.astype(f32),
        "emb_cos_12": _cos(ea, eb).astype(f32),
        "emb_l2_12": np.linalg.norm(ea - eb, axis=1).astype(f32),
        "emb_cos_p1": _cos(ep, ea).astype(f32),
        "emb_cos_p2": _cos(ep, eb).astype(f32),
        "alt1_p": alt1.astype(f32),
        "alt2_p": alt2.astype(f32),
        "par_third_p": third.astype(f32),
        "n_kids": n_kids[par].astype(f32),
        "cont1_p": sup_out[c1].astype(f32),
        "cont2_p": sup_out[c2].astype(f32),
        "par_in_p": sup_in[par].astype(f32),
        "par_gp_um": gp_um[e1],
        "score_par": sc["score"][par].astype(f32),
        "width_par": sc["peak_width_um"][par].astype(f32),
        "near_par": sc["near_um"][par].astype(f32),
        "dens_par": sc["density_10um"][par].astype(f32),
        "bnd_par": sc["boundary_um"][par].astype(f32),
        "score1": sc["score"][c1].astype(f32),
        "score2": sc["score"][c2].astype(f32),
        "width1": sc["peak_width_um"][c1].astype(f32),
        "width2": sc["peak_width_um"][c2].astype(f32),
        "near1": sc["near_um"][c1].astype(f32),
        "near2": sc["near_um"][c2].astype(f32),
    })
    if temporal:
        tf = _temporal_pair_features(
            par, c1, c2, pi, ci, p_adj, node_um, vel)
        out = pd.concat([out, tf], axis=1)
    return out


def emit_second_by_pairs(
    pi: np.ndarray,
    ci: np.ndarray,
    p_adj: np.ndarray,
    tau: float,
    pairs: pd.DataFrame,
    pair_p: np.ndarray,
    tau_div: float,
    recruit_orphans: bool = False,
    tau_cont: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Claims linking with the fork decision moved to the pair model.

    First claims are `emit_claims` rank-0, byte-identical to the deployed
    continuation decision (`tau2 > 1` disables its fork arm). A parent with an
    emitted continuation child c0 then gets AT MOST one second edge: the
    argmax-by-`pair_p` partner among enumerated pairs containing c0, emitted
    when `pair_p >= tau_div`. Candidate partners are

      claimants          the edge to this parent is the child's argmax
                         (`p_adj > 0`) — rule A, exactly the pairs the old
                         τ2 dial ruled on, plus claimants below τ it never saw
      orphan recruits    `recruit_orphans=True` extends to children left with
                         no emitted in-edge at all — rule B0, retraction-free
                         by construction (in-degree stays <= 1)

    Ties and multi-parent competition resolve greedily by descending
    `pair_p`, so a child is recruited at most once. Returns `(rows, conf)` —
    edge rows into `pi`/`ci`, and per-row confidence for submission ordering
    (the calibrated p where nonzero, else the pair score that justified the
    fork).
    """
    if tau_cont is None or tau_cont >= tau:
        base = emit_claims(pi, ci, p_adj, tau, 2.0)
    else:
        from biohub.tracker.linker import emit_termination_continuations

        base = emit_termination_continuations(
            pi, ci, p_adj, tau=tau, tau_cont=tau_cont)
    n_nodes = int(max(pi.max(), ci.max())) + 1 if len(pi) else 0
    conf = np.asarray(p_adj, dtype=np.float32).copy()
    if len(pairs) == 0 or n_nodes == 0:
        return base, conf[base]

    first_child = np.full(n_nodes, -1, dtype=np.int64)
    first_child[pi[base]] = ci[base]
    has_parent = np.zeros(n_nodes, dtype=bool)
    has_parent[ci[base]] = True

    par = pairs["par"].to_numpy(np.int64)
    c1 = pairs["c1"].to_numpy(np.int64)
    c2 = pairs["c2"].to_numpy(np.int64)
    e1 = pairs["e1"].to_numpy(np.int64)
    e2 = pairs["e2"].to_numpy(np.int64)
    pair_p = np.asarray(pair_p, dtype=np.float64)

    m1 = first_child[par] == c1
    m2 = first_child[par] == c2
    cand = (m1 | m2) & (pair_p >= tau_div)
    sec_e = np.where(m1, e2, e1)[cand]
    sec_c = np.where(m1, c2, c1)[cand]
    sec_par = par[cand]
    sec_p = pair_p[cand]

    claimant = p_adj[sec_e] > 0
    ok = claimant | (recruit_orphans & ~has_parent[sec_c])
    sec_e, sec_c, sec_par, sec_p = (sec_e[ok], sec_c[ok], sec_par[ok],
                                    sec_p[ok])

    taken_par = np.zeros(n_nodes, dtype=bool)
    second: list[int] = []
    for i in np.argsort(-sec_p, kind="stable"):
        P, c, e = int(sec_par[i]), int(sec_c[i]), int(sec_e[i])
        if taken_par[P] or has_parent[c]:
            continue
        taken_par[P] = True
        has_parent[c] = True
        second.append(e)
        if conf[e] == 0.0:
            conf[e] = np.float32(sec_p[i])

    rows = np.concatenate([base, np.asarray(second, dtype=base.dtype)])
    return rows, conf[rows]
