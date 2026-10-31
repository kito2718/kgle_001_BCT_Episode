"""Deploy-side post-pass of the 0.9208 checkpoint (stage-04 §23).

Three label-free actions compose on the finished v5 graph
(`Progress/checkpoint-v4c-092.md`): a PU/long-track division blend, whole
weak-component pruning, and greedy one-to-one continuation assignment on the
raw pass-2 context probability. This module holds the deploy-callable forms,
lifted VERBATIM from the measurement scripts —
`build_pair_tracklet_features.py`, `build_component_dataset.py`,
`verify_assignment_linker.py` — so the deployed features are the measured
features; `tests/test_postpass.py` pins the equality against the stored
training artifacts.

The one deliberate deviation is `is_6bba`: the component model was trained
with the embryo identity as a feature, which does not exist for a hidden
embryo. Deployment forces it to a constant, and the operating bar is chosen
from the OOF frontier measured WITH that constant — the neutralised sweep,
not the frozen stratum bars (checkpoint §"Honest limitation").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

TRACKLET_HORIZONS = (1, 2, 3, 5, 8)

# Aggregated per component, verbatim from build_component_dataset.py.
COMPONENT_EDGE_COLUMNS = [
    "resid_um", "resid_z_um", "resid_xy_um", "dist_um", "flow_mag_um",
    "resid_over_sigma", "emb_cos", "emb_l2", "resid_rank", "resid_margin",
    "grandparent_um", "flow_coherence_um", "p1", "p1_adj", "par_in_p",
    "par_up_len", "cv_resid_um", "child_out_p", "child_cont_resid_um",
    "sup_in_p", "sup_out_c", "rival_p", "par_out_other",
]
COMPONENT_NODE_COLUMNS = [
    "score", "rank_in_frame", "flow_sigma_um", "near_um", "density_10um",
    "peak_width_um", "boundary_um",
]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def blend_pair_scores(p_pu: np.ndarray, p_long: np.ndarray,
                      alpha: float = 0.4) -> np.ndarray:
    """`logit_extra_a4`: sigmoid((1-a)·logit(PU) + a·logit(long))."""
    return _sigmoid((1 - alpha) * _logit(p_pu)
                    + alpha * _logit(p_long)).astype(np.float32)


# ---- tracklet (long-horizon) pair features --------------------------------

def _layers(first: np.ndarray, depth: int = 8) -> np.ndarray:
    out = np.full((len(first), depth), -1, dtype=np.int32)
    out[:, 0] = first
    for k in range(1, depth):
        prior = out[:, k - 1]
        ok = prior >= 0
        out[ok, k] = first[prior[ok]]
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = _norm(a) * _norm(b)
    return np.divide((a * b).sum(axis=1), den,
                     out=np.zeros(len(a), dtype=np.float32), where=den > 1e-8)


def _mean(values: np.ndarray, nodes: np.ndarray):
    ok = nodes >= 0
    x = values[np.maximum(nodes, 0)]
    count = ok.sum(axis=1).astype(np.float32)
    avg = (x * ok[..., None]).sum(axis=1) / np.maximum(count[:, None], 1.0)
    dev = x - avg[:, None, :]
    rms = np.sqrt(((dev * dev).sum(axis=2) * ok).sum(axis=1)
                  / np.maximum(count, 1.0))
    return avg.astype(np.float32), rms.astype(np.float32)


def tracklet_features(
    node_t: np.ndarray,
    final_zyx: np.ndarray,
    pi: np.ndarray,
    ci: np.ndarray,
    take: np.ndarray,
    conf: np.ndarray,
    cand_sel: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """The 94 `lt_*` columns for each claims pair, on the v5 emitted graph.

    `final_zyx` is the SMOOTHED integer coordinate array (the cache's
    convention); `conf` is the v5 emission confidence aligned with `take`.
    `cand_sel` is the candidate pool sliced to the selected rows.
    """
    n = len(node_t)
    pos = final_zyx.astype(np.float32) * SCALE

    prev = np.full(n, -1, dtype=np.int32)
    prev[ci[take]] = pi[take]
    nxt = np.full(n, -1, dtype=np.int32)
    priority = np.zeros(len(pi), dtype=np.float32)
    priority[take] = conf
    for edge in take[np.argsort(priority[take], kind="stable")]:
        nxt[int(pi[edge])] = int(ci[edge])
    ancestors, descendants = _layers(prev), _layers(nxt)

    emb = cand_sel[[f"emb_{i}" for i in range(48)]].to_numpy(np.float32)
    emb = emb / np.maximum(_norm(emb)[:, None], 1e-6)
    score = cand_sel.score.to_numpy(np.float32)
    width = cand_sel.peak_width_um.to_numpy(np.float32)
    flow = cand_sel[["flow_z", "flow_y", "flow_x"]].to_numpy(np.float32) * SCALE

    par = pairs.par.to_numpy(np.int64)
    c1 = pairs.c1.to_numpy(np.int64)
    c2 = pairs.c2.to_numpy(np.int64)
    p0, p1, p2 = pos[par], pos[c1], pos[c2]
    v1, v2 = p1 - p0, p2 - p0
    sep0 = p1 - p2
    sep0_n = _norm(sep0)
    mid0 = 0.5 * (p1 + p2)

    out: dict[str, np.ndarray] = {
        "lt_time": node_t[par].astype(np.float32) / 99.0,
        "lt_parent_hist_len8": (ancestors[par] >= 0).sum(axis=1).astype(np.float32),
        "lt_d1_fut_len8": (descendants[c1] >= 0).sum(axis=1).astype(np.float32),
        "lt_d2_fut_len8": (descendants[c2] >= 0).sum(axis=1).astype(np.float32),
        "lt_min_fut_len8": np.minimum((descendants[c1] >= 0).sum(axis=1),
                                      (descendants[c2] >= 0).sum(axis=1)).astype(np.float32),
        "lt_branch_speed_cos": _cos(v1, v2),
        "lt_parent_flow_mid_res": _norm((mid0 - p0) - flow[par]),
        "lt_parent_flow_mid_cos": _cos(mid0 - p0, flow[par]),
    }

    for h in (1, 2, 3, 5):
        anc = ancestors[par, h - 1]
        ok = anc >= 0
        vin = (p0 - pos[np.maximum(anc, 0)]) / float(h)
        out[f"lt_ph{h}_ok"] = ok.astype(np.float32)
        out[f"lt_ph{h}_mid_res"] = np.where(
            ok, _norm((mid0 - p0) - vin), np.nan).astype(np.float32)
        out[f"lt_ph{h}_mid_cos"] = np.where(
            ok, _cos(mid0 - p0, vin), np.nan).astype(np.float32)
        out[f"lt_ph{h}_d1_res"] = np.where(ok, _norm(v1 - vin), np.nan).astype(np.float32)
        out[f"lt_ph{h}_d2_res"] = np.where(ok, _norm(v2 - vin), np.nan).astype(np.float32)

    for h in TRACKLET_HORIZONS:
        g1, g2 = descendants[c1, h - 1], descendants[c2, h - 1]
        both = (g1 >= 0) & (g2 >= 0)
        q1, q2 = pos[np.maximum(g1, 0)], pos[np.maximum(g2, 0)]
        fv1 = (q1 - p1) / float(h)
        fv2 = (q2 - p2) / float(h)
        sep = q1 - q2
        sep_n = _norm(sep)
        mid = 0.5 * (q1 + q2)
        out[f"lt_f{h}_both"] = both.astype(np.float32)
        out[f"lt_f{h}_sep"] = np.where(both, sep_n, np.nan).astype(np.float32)
        out[f"lt_f{h}_sep_growth"] = np.where(
            both, sep_n - sep0_n, np.nan).astype(np.float32)
        out[f"lt_f{h}_sep_slope"] = np.where(
            both, (sep_n - sep0_n) / h, np.nan).astype(np.float32)
        out[f"lt_f{h}_sep_cos"] = np.where(
            both, _cos(sep0, sep), np.nan).astype(np.float32)
        out[f"lt_f{h}_step_cos"] = np.where(
            both, _cos(fv1, fv2), np.nan).astype(np.float32)
        out[f"lt_f{h}_d1_accel"] = np.where(
            g1 >= 0, _norm(fv1 - v1), np.nan).astype(np.float32)
        out[f"lt_f{h}_d2_accel"] = np.where(
            g2 >= 0, _norm(fv2 - v2), np.nan).astype(np.float32)
        future_mid_step = (mid - p0) / float(h + 1)
        out[f"lt_f{h}_mid_flow_res"] = np.where(
            both, _norm(future_mid_step - flow[par]), np.nan).astype(np.float32)
        anc = ancestors[par, min(h, 5) - 1]
        has_hist = anc >= 0
        vin = (p0 - pos[np.maximum(anc, 0)]) / float(min(h, 5))
        out[f"lt_f{h}_mid_hist_res"] = np.where(
            both & has_hist, _norm(future_mid_step - vin), np.nan).astype(np.float32)

    hist_nodes = np.column_stack([par, ancestors[par, :5]])
    d1_nodes = np.column_stack([c1, descendants[c1, :5]])
    d2_nodes = np.column_stack([c2, descendants[c2, :5]])
    he, he_rms = _mean(emb, hist_nodes)
    e1, e1_rms = _mean(emb, d1_nodes)
    e2, e2_rms = _mean(emb, d2_nodes)
    hs, hs_rms = _mean(score[:, None], hist_nodes)
    s1, s1_rms = _mean(score[:, None], d1_nodes)
    s2, s2_rms = _mean(score[:, None], d2_nodes)
    hw, _ = _mean(width[:, None], hist_nodes)
    w1, _ = _mean(width[:, None], d1_nodes)
    w2, _ = _mean(width[:, None], d2_nodes)
    out.update({
        "lt_emb_hist_d1_cos": _cos(he, e1),
        "lt_emb_hist_d2_cos": _cos(he, e2),
        "lt_emb_daughters_cos": _cos(e1, e2),
        "lt_emb_hist_d1_l2": _norm(he - e1),
        "lt_emb_hist_d2_l2": _norm(he - e2),
        "lt_emb_daughters_l2": _norm(e1 - e2),
        "lt_emb_hist_rms": he_rms, "lt_emb_d1_rms": e1_rms,
        "lt_emb_d2_rms": e2_rms,
        "lt_score_parent_daughters": hs[:, 0] - 0.5 * (s1[:, 0] + s2[:, 0]),
        "lt_score_daughter_asym": np.abs(s1[:, 0] - s2[:, 0]),
        "lt_score_hist_rms": hs_rms, "lt_score_d1_rms": s1_rms,
        "lt_score_d2_rms": s2_rms,
        "lt_width_parent_daughters": hw[:, 0] - 0.5 * (w1[:, 0] + w2[:, 0]),
        "lt_width_daughter_asym": np.abs(w1[:, 0] - w2[:, 0]),
    })
    return pd.DataFrame(out)


# ---- whole-component features ---------------------------------------------

def graph_roots(node_t: np.ndarray, pi: np.ndarray, ci: np.ndarray,
                take: np.ndarray):
    """Weak-component root per node, frame-ordered union — verbatim `_roots`."""
    n = len(node_t)
    root = np.arange(n, dtype=np.int64)
    parent_t = node_t[pi[take]]
    for frame in np.unique(parent_t):
        rows = take[parent_t == frame]
        root[ci[rows]] = root[pi[rows]]
    linked = np.zeros(n, dtype=bool)
    linked[pi[take]] = True
    linked[ci[take]] = True
    nodes = np.flatnonzero(linked)
    roots = np.unique(root[nodes])
    return root, nodes, roots


def _stats(group, values, count, n):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    g, x = group[finite], values[finite]
    nn = np.bincount(g, minlength=n).astype(np.float64)
    sm = np.bincount(g, weights=x, minlength=n)
    sq = np.bincount(g, weights=x * x, minlength=n)
    mean = np.divide(sm, nn, out=np.zeros(n), where=nn > 0)
    var = np.divide(sq, nn, out=np.zeros(n), where=nn > 0) - mean * mean
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(lo, g, x)
    np.maximum.at(hi, g, x)
    lo[~np.isfinite(lo)], hi[~np.isfinite(hi)] = np.nan, np.nan
    return mean, np.sqrt(np.maximum(var, 0)), lo, hi, nn / np.maximum(count, 1)


def component_features(
    node_t: np.ndarray,
    base_zyx: np.ndarray,
    final_zyx: np.ndarray,
    pi: np.ndarray,
    ci: np.ndarray,
    take: np.ndarray,
    conf: np.ndarray,
    p_emit: np.ndarray,
    cand_sel: pd.DataFrame,
    ctx_frame: pd.DataFrame,
    is_6bba: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-component feature frame + (root-per-node, roots).

    Deploy-safe subset of `build_component_dataset.run_one`: identical
    aggregate features, no label columns, and `is_6bba` supplied by the
    caller (a hidden embryo carries no training identity).
    """
    root, nodes, roots = graph_roots(node_t, pi, ci, take)
    n = len(node_t)
    node_group = root[nodes]
    edge_group = root[pi[take]]
    node_count = np.bincount(node_group, minlength=n).astype(np.float64)
    edge_count = np.bincount(edge_group, minlength=n).astype(np.float64)

    indeg = np.zeros(n, dtype=np.int8)
    outdeg = np.zeros(n, dtype=np.int8)
    np.add.at(indeg, ci[take], 1)
    np.add.at(outdeg, pi[take], 1)
    forks = np.bincount(root[nodes], weights=(outdeg[nodes] == 2), minlength=n)
    leaves = np.bincount(root[nodes], weights=(outdeg[nodes] == 0), minlength=n)

    t = node_t.astype(np.float32)
    tmin = np.full(n, np.inf)
    tmax = np.full(n, -np.inf)
    np.minimum.at(tmin, node_group, t[nodes])
    np.maximum.at(tmax, node_group, t[nodes])
    span = tmax - tmin + 1
    data: dict[str, np.ndarray] = {
        "component_nodes": node_count[roots].astype(np.int16),
        "component_edges": edge_count[roots].astype(np.int16),
        "span": span[roots].astype(np.float32),
        "start_t": (tmin[roots] / 99.0).astype(np.float32),
        "end_t": (tmax[roots] / 99.0).astype(np.float32),
        "nodes_per_span": (node_count[roots] / span[roots]).astype(np.float32),
        "forks": forks[roots].astype(np.float32),
        "leaves": leaves[roots].astype(np.float32),
        "is_6bba": np.repeat(np.float32(is_6bba), len(roots)),
    }

    for col in COMPONENT_NODE_COLUMNS:
        vals = cand_sel[col].to_numpy(np.float32)[nodes]
        mean, std, lo, hi, finite_frac = _stats(node_group, vals, node_count, n)
        for suffix, arr in (("mean", mean), ("std", std), ("min", lo),
                            ("max", hi), ("finite", finite_frac)):
            data[f"node_{col}_{suffix}"] = arr[roots].astype(np.float32)

    pos = final_zyx.astype(np.float32) * SCALE
    base = base_zyx.astype(np.float32) * SCALE
    for j, axis in enumerate("zyx"):
        mean, std, lo, hi, _ = _stats(node_group, pos[nodes, j], node_count, n)
        data[f"pos_{axis}_mean"] = mean[roots].astype(np.float32)
        data[f"pos_{axis}_std"] = std[roots].astype(np.float32)
        data[f"pos_{axis}_range"] = (hi[roots] - lo[roots]).astype(np.float32)
    move = np.linalg.norm(pos - base, axis=1)
    mean, std, lo, hi, _ = _stats(node_group, move[nodes], node_count, n)
    data["smooth_move_mean"] = mean[roots].astype(np.float32)
    data["smooth_move_std"] = std[roots].astype(np.float32)
    data["smooth_move_max"] = hi[roots].astype(np.float32)

    emb = cand_sel[[f"emb_{i}" for i in range(48)]].to_numpy(np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
    emb_mean_sq = np.zeros(n, dtype=np.float64)
    mean_norm_sq = np.zeros(n, dtype=np.float64)
    for j in range(48):
        sm = np.bincount(node_group, weights=emb[nodes, j], minlength=n)
        mean = np.divide(sm, node_count, out=np.zeros(n), where=node_count > 0)
        data[f"emb_mean_{j}"] = mean[roots].astype(np.float32)
        mean_norm_sq += mean * mean
        emb_mean_sq += np.bincount(
            node_group, weights=emb[nodes, j] ** 2, minlength=n)
    emb_mean_sq = np.divide(
        emb_mean_sq, node_count, out=np.zeros(n), where=node_count > 0)
    data["emb_rms"] = np.sqrt(np.maximum(
        emb_mean_sq[roots] - mean_norm_sq[roots], 0)).astype(np.float32)

    for col in COMPONENT_EDGE_COLUMNS:
        vals = ctx_frame[col].to_numpy(np.float32)[take]
        mean, std, lo, hi, finite_frac = _stats(edge_group, vals, edge_count, n)
        data[f"edge_{col}_mean"] = mean[roots].astype(np.float32)
        data[f"edge_{col}_std"] = std[roots].astype(np.float32)
        data[f"edge_{col}_min"] = lo[roots].astype(np.float32)
        data[f"edge_{col}_max"] = hi[roots].astype(np.float32)
        data[f"edge_{col}_finite"] = finite_frac[roots].astype(np.float32)

    for col, vals in (("emit_conf", conf), ("emit_raw", p_emit[take])):
        mean, std, lo, hi, _ = _stats(edge_group, vals, edge_count, n)
        data[f"{col}_mean"] = mean[roots].astype(np.float32)
        data[f"{col}_std"] = std[roots].astype(np.float32)
        data[f"{col}_min"] = lo[roots].astype(np.float32)
        data[f"{col}_max"] = hi[roots].astype(np.float32)
    return pd.DataFrame(data), root, roots


# ---- greedy assignment + divisions ----------------------------------------

def greedy_assign(pi, ci, score, threshold):
    """One continuation per parent and per child, by descending raw score."""
    n = int(max(pi.max(initial=-1), ci.max(initial=-1)) + 1)
    used_out = np.zeros(n, dtype=bool)
    used_in = np.zeros(n, dtype=bool)
    picked = []
    for edge in np.argsort(-score, kind="stable"):
        if score[edge] < threshold:
            break
        p, c = int(pi[edge]), int(ci[edge])
        if used_out[p] or used_in[c]:
            continue
        used_out[p] = True
        used_in[c] = True
        picked.append(int(edge))
    return np.asarray(picked, dtype=np.int64)


def add_divisions(base, pi, ci, p_emit, pairs, pair_score, threshold):
    """One second child per parent from the blended pair score — verbatim."""
    n = int(max(pi.max(initial=-1), ci.max(initial=-1)) + 1)
    first = np.full(n, -1, dtype=np.int64)
    first[pi[base]] = ci[base]
    has_parent = np.zeros(n, dtype=bool)
    has_parent[ci[base]] = True
    par = pairs.par.to_numpy(np.int64)
    c1, c2 = pairs.c1.to_numpy(np.int64), pairs.c2.to_numpy(np.int64)
    e1, e2 = pairs.e1.to_numpy(np.int64), pairs.e2.to_numpy(np.int64)
    m1, m2 = first[par] == c1, first[par] == c2
    eligible = (m1 | m2) & (pair_score >= threshold)
    sec_e = np.where(m1, e2, e1)[eligible]
    sec_c = np.where(m1, c2, c1)[eligible]
    sec_p = par[eligible]
    sec_score = pair_score[eligible]
    ok = (p_emit[sec_e] > 0) & ~has_parent[sec_c]
    sec_e, sec_c, sec_p, sec_score = (
        sec_e[ok], sec_c[ok], sec_p[ok], sec_score[ok])
    forked = np.zeros(n, dtype=bool)
    extra = []
    for i in np.argsort(-sec_score, kind="stable"):
        p, c = int(sec_p[i]), int(sec_c[i])
        if forked[p] or has_parent[c]:
            continue
        forked[p] = True
        has_parent[c] = True
        extra.append(int(sec_e[i]))
    return np.concatenate([base, np.asarray(extra, dtype=np.int64)])


def joint_division_assign(pi, ci, continuation_score, continuation_threshold,
                          pairs, pair_score, capacity):
    """Reserve complete forks before assigning ordinary continuations.

    ``add_divisions`` can add only an unparented second daughter after the
    continuation graph is fixed.  That makes the wrong incumbent continuation
    irreversible.  This allocator instead ranks complete mother/two-daughter
    hypotheses, reserves up to ``capacity`` mutually compatible forks, and
    only then greedily fills the remaining one-to-one continuations.

    Capacity is a structural regularizer, not a fitted probability threshold:
    V28 supplies the number of forks emitted by frozen V21 for that movie.
    A reserved daughter cannot be stolen by another parent, while a daughter
    may itself be a later reserved mother (successive divisions remain legal).
    Returns ``(edge_rows, chosen_pair_rows)``.
    """
    pi = np.asarray(pi, dtype=np.int64)
    ci = np.asarray(ci, dtype=np.int64)
    continuation_score = np.asarray(continuation_score, dtype=np.float64)
    pair_score = np.asarray(pair_score, dtype=np.float64)
    capacity = max(0, int(capacity))
    n = int(max(pi.max(initial=-1), ci.max(initial=-1)) + 1)
    used_out = np.zeros(n, dtype=bool)
    used_in = np.zeros(n, dtype=bool)
    par = pairs.par.to_numpy(np.int64)
    c1 = pairs.c1.to_numpy(np.int64)
    c2 = pairs.c2.to_numpy(np.int64)
    e1 = pairs.e1.to_numpy(np.int64)
    e2 = pairs.e2.to_numpy(np.int64)

    reserved_edges = []
    chosen_pairs = []
    for row in np.argsort(-pair_score, kind="stable"):
        if len(chosen_pairs) >= capacity:
            break
        p, a, b = int(par[row]), int(c1[row]), int(c2[row])
        ea, eb = int(e1[row]), int(e2[row])
        if not np.isfinite(pair_score[row]) or a == b:
            continue
        if not (0 <= ea < len(pi) and 0 <= eb < len(pi)):
            continue
        if not (pi[ea] == p and ci[ea] == a and pi[eb] == p and ci[eb] == b):
            raise ValueError("pair bookkeeping does not match the edge universe")
        if used_out[p] or used_in[a] or used_in[b]:
            continue
        used_out[p] = True
        used_in[a] = True
        used_in[b] = True
        reserved_edges.extend((ea, eb))
        chosen_pairs.append(int(row))

    continuations = []
    for edge in np.argsort(-continuation_score, kind="stable"):
        if continuation_score[edge] < continuation_threshold:
            break
        p, child = int(pi[edge]), int(ci[edge])
        if used_out[p] or used_in[child]:
            continue
        used_out[p] = True
        used_in[child] = True
        continuations.append(int(edge))
    return (
        np.asarray([*reserved_edges, *continuations], dtype=np.int64),
        np.asarray(chosen_pairs, dtype=np.int64),
    )


# ===========================================================================
# The 0.9289 checkpoint's additional stages (checkpoint-v4c-0929), verbatim
# lifts from build_pair_image_features / build_pair_embedding_features /
# build_pair_image_highres_features / build_stub_pruner_dataset /
# verify_tail_pruning — pinned by tests/test_postpass.py.
# ===========================================================================

IMG_ALPHAS = np.linspace(0.0, 1.0, 9, dtype=np.float32)
HR_ALPHAS = np.linspace(0, 1, 17, dtype=np.float32)

STUB_EDGE_FEATURES = [
    "parent_rank", "child_rank", "resid_um", "resid_z_um", "resid_xy_um",
    "dist_um", "dz_um", "dxy_um", "flow_mag_um", "resid_over_sigma",
    "score_p", "score_c", "rank_p", "rank_c", "sigma_p", "sigma_c",
    "near_p", "near_c", "dens_p", "dens_c", "width_p", "width_c",
    "bnd_p", "bnd_c", "emb_cos", "emb_l2", "k_in_gate", "resid_rank",
    "resid_margin", "grandparent_um", "parent_flow_mag", "flow_coherence_um",
    "n_claims_p", "claim_rank", "claim_margin", "p1", "p1_adj",
    "p1_emitted", "par_in_p", "par_up_len", "cv_resid_um", "cv_resid_z_um",
    "cv_resid_xy_um", "par_emb_step", "child_out_p", "child_out_n",
    "child_cont_resid_um", "sup_in_p", "sup_out_c", "rival_p",
    "par_out_other",
]


def next_map(n: int, pi, ci, take, conf) -> np.ndarray:
    """Strongest emitted continuation per parent (conf-priority overwrite)."""
    nxt = np.full(n, -1, dtype=np.int32)
    priority = np.zeros(len(pi), dtype=np.float32)
    priority[take] = conf
    for edge in take[np.argsort(priority[take], kind="stable")]:
        nxt[int(pi[edge])] = int(ci[edge])
    return nxt


def checkpoint_base_blend(p_allimg, p_full, p_hard) -> np.ndarray:
    """`embcore_top5_c0p01_w1`: 0.72 / 0.18 / 0.10 in logit space."""
    core = 0.8 * _logit(p_allimg) + 0.2 * _logit(p_full)
    return _sigmoid(np.clip(0.9 * core + 0.1 * _logit(p_hard),
                            -30, 30)).astype(np.float32)


def refine_with_highres(base, hr_p, in_gate, weight: float = 0.4) -> np.ndarray:
    """Blend the source-resolution arm inside the >= 0.50 gate only."""
    refined = _sigmoid(np.clip((1 - weight) * _logit(base)
                               + weight * _logit(hr_p), -30, 30))
    return np.where(in_gate, refined, base).astype(np.float32)


# ---- low-resolution (detector grid) image features -------------------------

def _img_at(vol, t, points):
    shape = np.asarray(vol.shape[1:], dtype=np.int64)
    q = np.clip(np.rint(points).astype(np.int64), 0, shape - 1)
    tt = np.clip(np.asarray(t, dtype=np.int64), 0, vol.shape[0] - 1)
    return np.asarray(vol[tt, q[:, 0], q[:, 1], q[:, 2]], dtype=np.float32)


def _img_local(vol, t, points):
    shape = np.asarray(vol.shape[1:], dtype=np.int64)
    q = np.clip(np.rint(points).astype(np.int64), 0, shape - 1)
    tt = np.clip(np.asarray(t, dtype=np.int64), 0, vol.shape[0] - 1)
    sm = np.zeros(len(q), dtype=np.float32)
    sq = np.zeros(len(q), dtype=np.float32)
    mx = np.full(len(q), -np.inf, dtype=np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                r = np.clip(q + np.asarray([dz, dy, dx]), 0, shape - 1)
                v = np.asarray(vol[tt, r[:, 0], r[:, 1], r[:, 2]],
                               dtype=np.float32)
                sm += v
                sq += v * v
                mx = np.maximum(mx, v)
    mean = sm / 27.0
    std = np.sqrt(np.maximum(sq / 27.0 - mean * mean, 0))
    shell = np.zeros(len(q), dtype=np.float32)
    for delta in ((-2, 0, 0), (2, 0, 0), (0, -2, 0),
                  (0, 2, 0), (0, 0, -2), (0, 0, 2)):
        r = np.clip(q + np.asarray(delta), 0, shape - 1)
        shell += np.asarray(vol[tt, r[:, 0], r[:, 1], r[:, 2]],
                            dtype=np.float32)
    shell /= 6.0
    center = _img_at(vol, tt, points)
    return np.column_stack([center, mx, mean, std, shell, mx - shell])


def _img_pair_ops(out, prefix, a, b):
    eps = np.float32(0.02)
    out[f"{prefix}_mean"] = 0.5 * (a + b)
    out[f"{prefix}_min"] = np.minimum(a, b)
    out[f"{prefix}_max"] = np.maximum(a, b)
    out[f"{prefix}_absdiff"] = np.abs(a - b)
    out[f"{prefix}_reldiff"] = np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps)


def pair_image_features(vol, node_t, node_grid, nxt, pairs) -> pd.DataFrame:
    """The `img_*` block: detector-grid morphology around the proposed split.

    `vol` is the detector-input volume (grid resolution), `node_grid` the
    selected candidates' positions in that grid (`src_to_grid`), `nxt` the
    conf-priority continuation map of the v5 graph.
    """
    par = pairs.par.to_numpy(np.int64)
    c1, c2 = pairs.c1.to_numpy(np.int64), pairs.c2.to_numpy(np.int64)
    t = node_t[par]
    p, d1, d2 = node_grid[par], node_grid[c1], node_grid[c2]
    mid = 0.5 * (d1 + d2)
    lp = _img_local(vol, t, p)
    l1 = _img_local(vol, t + 1, d1)
    l2 = _img_local(vol, t + 1, d2)
    names = ("center", "max3", "mean3", "std3", "shell2", "contrast")
    out: dict[str, np.ndarray] = {}
    for j, feature in enumerate(names):
        out[f"img_parent_{feature}"] = lp[:, j]
        _img_pair_ops(out, f"img_daughters_{feature}", l1[:, j], l2[:, j])
        child_mean = 0.5 * (l1[:, j] + l2[:, j])
        out[f"img_{feature}_child_minus_parent"] = child_mean - lp[:, j]
        out[f"img_{feature}_child_parent_ratio"] = child_mean / np.maximum(
            np.abs(lp[:, j]), 0.02)

    p_prev = _img_at(vol, t - 1, p)
    p_next = _img_at(vol, t + 1, p)
    d1_prev, d2_prev = _img_at(vol, t, d1), _img_at(vol, t, d2)
    out.update({
        "img_parent_prev": p_prev,
        "img_parent_next": p_next,
        "img_parent_temporal_drop": lp[:, 0] - p_next,
        "img_d1_at_parent_frame": d1_prev,
        "img_d2_at_parent_frame": d2_prev,
        "img_daughters_prior_mean": 0.5 * (d1_prev + d2_prev),
        "img_daughters_prior_absdiff": np.abs(d1_prev - d2_prev),
        "img_mid_parent_frame": _img_at(vol, t, mid),
        "img_mid_child_frame": _img_at(vol, t + 1, mid),
    })

    line = (d1[:, None, :] * (1 - IMG_ALPHAS[None, :, None])
            + d2[:, None, :] * IMG_ALPHAS[None, :, None])
    flat = line.reshape(-1, 3)
    pt = np.repeat(t, len(IMG_ALPHAS))
    profile0 = _img_at(vol, pt, flat).reshape(len(pairs), -1)
    profile1 = _img_at(vol, pt + 1, flat).reshape(len(pairs), -1)
    for k in range(len(IMG_ALPHAS)):
        out[f"img_line_t0_{k}"] = profile0[:, k]
        out[f"img_line_t1_{k}"] = profile1[:, k]
    for tag, profile in (("t0", profile0), ("t1", profile1)):
        endpoints = 0.5 * (profile[:, 0] + profile[:, -1])
        centre = profile[:, 4]
        out[f"img_line_{tag}_mean"] = profile.mean(axis=1)
        out[f"img_line_{tag}_std"] = profile.std(axis=1)
        out[f"img_line_{tag}_min"] = profile.min(axis=1)
        out[f"img_line_{tag}_max"] = profile.max(axis=1)
        out[f"img_line_{tag}_endpoints"] = endpoints
        out[f"img_line_{tag}_bimodal"] = endpoints - centre
        out[f"img_line_{tag}_centre_peak"] = centre - 0.5 * (
            profile[:, 2] + profile[:, 6])
    bimodal0 = 0.5 * (profile0[:, 0] + profile0[:, -1]) - profile0[:, 4]
    bimodal1 = 0.5 * (profile1[:, 0] + profile1[:, -1]) - profile1[:, 4]
    out["img_line_bimodal_growth"] = bimodal1 - bimodal0
    out["img_line_change_l1"] = np.mean(np.abs(profile1 - profile0), axis=1)
    a0 = profile0 - profile0.mean(axis=1, keepdims=True)
    a1 = profile1 - profile1.mean(axis=1, keepdims=True)
    out["img_line_temporal_corr"] = np.sum(a0 * a1, axis=1) / np.maximum(
        np.linalg.norm(a0, axis=1) * np.linalg.norm(a1, axis=1), 1e-6)

    g1, g2 = c1.copy(), c2.copy()
    for horizon in (1, 2, 3):
        old1, old2 = g1, g2
        g1 = np.where(old1 >= 0, nxt[np.maximum(old1, 0)], -1)
        g2 = np.where(old2 >= 0, nxt[np.maximum(old2, 0)], -1)
        both = (g1 >= 0) & (g2 >= 0)
        q1, q2 = node_grid[np.maximum(g1, 0)], node_grid[np.maximum(g2, 0)]
        lt = t + 1 + horizon
        f1 = _img_local(vol, lt, q1)
        f2 = _img_local(vol, lt, q2)
        f1[~(g1 >= 0)] = np.nan
        f2[~(g2 >= 0)] = np.nan
        out[f"img_f{horizon}_both"] = both.astype(np.float32)
        for j, feature in enumerate(names):
            _img_pair_ops(out, f"img_f{horizon}_{feature}", f1[:, j], f2[:, j])

        future_line = (q1[:, None, :] * (1 - IMG_ALPHAS[None, :, None])
                       + q2[:, None, :] * IMG_ALPHAS[None, :, None])
        future_profile = _img_at(
            vol, np.repeat(lt, len(IMG_ALPHAS)), future_line.reshape(-1, 3)
        ).reshape(len(pairs), -1)
        endpoints = 0.5 * (future_profile[:, 0] + future_profile[:, -1])
        future_stats = {
            "line_mean": future_profile.mean(axis=1),
            "line_std": future_profile.std(axis=1),
            "line_min": future_profile.min(axis=1),
            "line_max": future_profile.max(axis=1),
            "line_bimodal": endpoints - future_profile[:, 4],
            "line_change_l1": np.mean(np.abs(future_profile - profile1), axis=1),
        }
        for feature, values in future_stats.items():
            values[~both] = np.nan
            out[f"img_f{horizon}_{feature}"] = values
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32)
                         for k, v in out.items()})


# ---- symmetric embedding features ------------------------------------------

def pair_embedding_features(node_t, node_emb, pairs) -> pd.DataFrame:
    """The `ev_*` block: per-frame z-scored full embedding vectors."""
    t = np.asarray(node_t, dtype=np.int64)
    e = np.asarray(node_emb, dtype=np.float32)
    mean = np.zeros_like(e)
    scale = np.zeros_like(e)
    for frame in np.unique(t):
        ids = t == frame
        m = e[ids].mean(axis=0)
        s = e[ids].std(axis=0)
        mean[ids] = m
        scale[ids] = np.maximum(s, 1e-3)
    ez = (e - mean) / scale

    par = pairs.par.to_numpy(np.int64)
    c1, c2 = pairs.c1.to_numpy(np.int64), pairs.c2.to_numpy(np.int64)
    ep, ed = ez[par], .5 * (ez[c1] + ez[c2])
    raw_delta = e[par] - .5 * (e[c1] + e[c2])
    sister_abs = np.abs(e[c1] - e[c2])
    data = {}
    for i in range(48):
        data[f"ev_parent_z_{i}"] = ep[:, i]
        data[f"ev_daughters_zmean_{i}"] = ed[:, i]
        data[f"ev_delta_z_{i}"] = ep[:, i] - ed[:, i]
        data[f"ev_delta_raw_{i}"] = raw_delta[:, i]
        data[f"ev_sister_abs_{i}"] = sister_abs[:, i]
    return pd.DataFrame(data, dtype=np.float32)


# ---- source-resolution (high-res) features ---------------------------------

def _hr_patch_stats(frame, points, patch_radius, z_radius):
    from scipy.ndimage import maximum_filter

    axis = np.arange(-patch_radius, patch_radius + 1, dtype=np.int64)
    dz_axis = np.arange(-z_radius, z_radius + 1, dtype=np.int64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    r2 = yy * yy + xx * xx
    shell2 = max(patch_radius - 1, 1) ** 2
    shape = np.asarray(frame.shape, dtype=np.int64)
    q = np.rint(points).astype(np.int64)
    q = np.clip(q, 0, shape - 1)
    z = np.clip(q[:, 0, None, None, None] + dz_axis[None, :, None, None], 0, shape[0] - 1)
    y = np.clip(q[:, 1, None, None, None] + yy[None, None, :, :], 0, shape[1] - 1)
    x = np.clip(q[:, 2, None, None, None] + xx[None, None, :, :], 0, shape[2] - 1)
    patch = frame[z, y, x]
    mip = patch.max(axis=1)
    flat = mip.reshape(len(points), -1)
    shell = mip[:, r2 >= shell2].mean(axis=1)
    peak = flat.max(axis=1)
    signal = np.maximum(mip - shell[:, None, None], 0)
    mass = np.maximum(signal.sum(axis=(1, 2)), 1e-6)
    cy = (signal * yy).sum(axis=(1, 2)) / mass
    cx = (signal * xx).sum(axis=(1, 2)) / mass
    dy = yy[None] - cy[:, None, None]
    dx = xx[None] - cx[:, None, None]
    cxx = (signal * dx * dx).sum(axis=(1, 2)) / mass
    cyy = (signal * dy * dy).sum(axis=(1, 2)) / mass
    cxy = (signal * dx * dy).sum(axis=(1, 2)) / mass
    trace = cxx + cyy
    disc = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4 * cxy * cxy, 0))
    local_max = mip >= maximum_filter(mip, size=(1, 3, 3), mode="nearest") - 1e-7
    threshold = shell + 0.35 * np.maximum(peak - shell, 0)
    n_peaks = (local_max & (mip >= threshold[:, None, None])).sum(axis=(1, 2))
    dynamic = np.maximum(peak - shell, 1e-4)
    return {
        "center": frame[q[:, 0], q[:, 1], q[:, 2]].astype(np.float32),
        "max": peak.astype(np.float32),
        "mean": flat.mean(axis=1).astype(np.float32),
        "std": flat.std(axis=1).astype(np.float32),
        "q90": np.quantile(flat, .9, axis=1).astype(np.float32),
        "core_max": mip[:, r2 <= 4].max(axis=1).astype(np.float32),
        "core_mean": mip[:, r2 <= 4].mean(axis=1).astype(np.float32),
        "shell": shell.astype(np.float32),
        "contrast": (peak - shell).astype(np.float32),
        "centroid_r": np.hypot(cy, cx).astype(np.float32),
        "elongation": (disc / np.maximum(trace, 1e-5)).astype(np.float32),
        "area25": (mip >= (shell + .25 * dynamic)[:, None, None]).mean(axis=(1, 2)).astype(np.float32),
        "area50": (mip >= (shell + .50 * dynamic)[:, None, None]).mean(axis=(1, 2)).astype(np.float32),
        "area75": (mip >= (shell + .75 * dynamic)[:, None, None]).mean(axis=(1, 2)).astype(np.float32),
        "n_peaks": n_peaks.astype(np.float32),
    }


def _hr_pair_ops(out, prefix, a, b):
    out[f"{prefix}_mean"] = .5 * (a + b)
    out[f"{prefix}_min"] = np.minimum(a, b)
    out[f"{prefix}_max"] = np.maximum(a, b)
    out[f"{prefix}_absdiff"] = np.abs(a - b)
    out[f"{prefix}_reldiff"] = np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), .02)


def _hr_line(frame, d1, d2):
    from scipy.ndimage import map_coordinates

    line = (d1[:, None, :] * (1 - HR_ALPHAS[None, :, None])
            + d2[:, None, :] * HR_ALPHAS[None, :, None])
    return map_coordinates(frame, line.reshape(-1, 3).T, order=1,
                           mode="nearest").reshape(len(d1), -1).astype(np.float32)


def highres_pair_features(arr, node_t, node_zyx, pairs,
                          patch_radius: int = 6,
                          z_radius: int = 1) -> pd.DataFrame:
    """The `hr_*` block on normalized SOURCE frames (frozen 13x13x3 geometry).

    `arr` is the source-resolution 4D array (zarr array or ndarray);
    normalization is the deterministic 1/128-voxel-sample quantile scale the
    measurement used. Call with the GATED pair subset only — extraction is
    deliberately sparse.
    """
    node_t = np.asarray(node_t, dtype=np.int64)
    par = pairs.par.to_numpy(np.int64)
    c1, c2 = pairs.c1.to_numpy(np.int64), pairs.c2.to_numpy(np.int64)
    event_t = node_t[par]
    p, d1, d2 = node_zyx[par], node_zyx[c1], node_zyx[c2]
    out: dict[str, np.ndarray] = {}
    frames: dict[int, np.ndarray] = {}

    def frame(tt):
        tt = int(np.clip(tt, 0, arr.shape[0] - 1))
        if tt not in frames:
            raw = np.asarray(arr[tt])
            sample = raw[::2, ::8, ::8]
            lo, hi = np.quantile(sample, (.02, .999))
            frames[tt] = np.clip(
                (raw.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 2)
        return frames[tt]

    stat_names = None
    for t in np.unique(event_t):
        ids = np.flatnonzero(event_t == t)
        local: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for label, pts, dt in (
            ("p", p[ids], -1), ("p", p[ids], 0), ("p", p[ids], 1),
            ("d1", d1[ids], 0), ("d2", d2[ids], 0),
            ("d1", d1[ids], 1), ("d2", d2[ids], 1),
            ("d1", d1[ids], 2), ("d2", d2[ids], 2),
        ):
            local[(label, dt)] = _hr_patch_stats(
                frame(t + dt), pts, patch_radius, z_radius)
        if stat_names is None:
            stat_names = list(local[("p", 0)])
            for dt in (-1, 0, 1):
                for stat in stat_names:
                    out[f"hr_p_t{dt:+d}_{stat}"] = np.empty(len(pairs), np.float32)
            for dt in (0, 1, 2):
                for stat in stat_names:
                    for op in ("mean", "min", "max", "absdiff", "reldiff"):
                        out[f"hr_d_t{dt:+d}_{stat}_{op}"] = np.empty(len(pairs), np.float32)
            for stat in stat_names:
                out[f"hr_change_{stat}"] = np.empty(len(pairs), np.float32)
                out[f"hr_ratio_{stat}"] = np.empty(len(pairs), np.float32)
            for dt in (0, 1):
                for k in range(len(HR_ALPHAS)):
                    out[f"hr_line_t{dt}_{k}"] = np.empty(len(pairs), np.float32)
                for stat in ("mean", "std", "min", "max", "bimodal", "centre_peak"):
                    out[f"hr_line_t{dt}_{stat}"] = np.empty(len(pairs), np.float32)
            for stat in ("bimodal_growth", "change_l1", "temporal_corr"):
                out[f"hr_line_{stat}"] = np.empty(len(pairs), np.float32)
        for dt in (-1, 0, 1):
            for stat in stat_names:
                out[f"hr_p_t{dt:+d}_{stat}"][ids] = local[("p", dt)][stat]
        for dt in (0, 1, 2):
            for stat in stat_names:
                tmp = {}
                _hr_pair_ops(tmp, "x", local[("d1", dt)][stat], local[("d2", dt)][stat])
                for op in ("mean", "min", "max", "absdiff", "reldiff"):
                    out[f"hr_d_t{dt:+d}_{stat}_{op}"][ids] = tmp[f"x_{op}"]
        for stat in stat_names:
            child = .5 * (local[("d1", 1)][stat] + local[("d2", 1)][stat])
            parent = local[("p", 0)][stat]
            out[f"hr_change_{stat}"][ids] = child - parent
            out[f"hr_ratio_{stat}"][ids] = child / np.maximum(np.abs(parent), .02)
        profiles = [_hr_line(frame(t + dt), d1[ids], d2[ids]) for dt in (0, 1)]
        for dt, profile in enumerate(profiles):
            for k in range(len(HR_ALPHAS)):
                out[f"hr_line_t{dt}_{k}"][ids] = profile[:, k]
            ends = .5 * (profile[:, 0] + profile[:, -1])
            vals = {
                "mean": profile.mean(axis=1), "std": profile.std(axis=1),
                "min": profile.min(axis=1), "max": profile.max(axis=1),
                "bimodal": ends - profile[:, 8],
                "centre_peak": profile[:, 8] - .5 * (profile[:, 4] + profile[:, 12]),
            }
            for stat, values in vals.items():
                out[f"hr_line_t{dt}_{stat}"][ids] = values
        b0 = .5 * (profiles[0][:, 0] + profiles[0][:, -1]) - profiles[0][:, 8]
        b1 = .5 * (profiles[1][:, 0] + profiles[1][:, -1]) - profiles[1][:, 8]
        out["hr_line_bimodal_growth"][ids] = b1 - b0
        out["hr_line_change_l1"][ids] = np.mean(np.abs(profiles[1] - profiles[0]), axis=1)
        a = profiles[0] - profiles[0].mean(axis=1, keepdims=True)
        b = profiles[1] - profiles[1].mean(axis=1, keepdims=True)
        out["hr_line_temporal_corr"][ids] = np.sum(a * b, axis=1) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6)
        for old in [q for q in frames if q < t - 1]:
            del frames[old]
    return pd.DataFrame(out)


# ---- stub (premature terminal) features and the pruning pass ---------------

def stub_features(node_t, pi, ci, take, raw, p_emit, ctx_frame):
    """76 deployable features for premature terminal edges of `take`.

    Returns `(frame, chosen)` — the feature frame and the edge rows it
    describes. Verbatim from `build_stub_pruner_dataset.run_one`, minus the
    training label.
    """
    node_t = np.asarray(node_t, dtype=np.int32)
    n = len(node_t)
    out_degree = np.bincount(pi[take], minlength=n)
    in_degree = np.bincount(ci[take], minlength=n)
    last_t, first_t = int(node_t.max()), int(node_t.min())
    is_tail = ((out_degree[ci[take]] == 0) & (node_t[ci[take]] < last_t)
               & (out_degree[pi[take]] == 1))
    is_head = ((in_degree[pi[take]] == 0) & (node_t[pi[take]] > first_t)
               & (out_degree[pi[take]] == 1))
    chosen = take[is_tail | is_head]
    if not len(chosen):
        return pd.DataFrame(), chosen

    incoming = np.full(n, -1, np.int64)
    incoming[ci[take]] = take
    outgoing = np.full(n, -1, np.int64)
    single = out_degree[pi[take]] == 1
    outgoing[pi[take[single]]] = take[single]

    tail_set = set(take[is_tail].tolist())
    head_set = set(take[is_head].tolist())
    inner_raw = np.full((len(chosen), 5), np.nan, np.float32)
    inner_emit = np.full((len(chosen), 5), np.nan, np.float32)
    for row, edge_idx in enumerate(chosen):
        edge_idx = int(edge_idx)
        if edge_idx in tail_set:
            node = int(pi[edge_idx])
            step = int(incoming[node])
            direction = "tail"
        else:
            node = int(ci[edge_idx])
            step = int(outgoing[node])
            direction = "head"
        for depth in range(5):
            if step < 0:
                break
            inner_raw[row, depth] = raw[step]
            inner_emit[row, depth] = p_emit[step]
            if direction == "tail":
                node = int(pi[step])
                step = int(incoming[node])
            else:
                node = int(ci[step])
                step = int(outgoing[node])

    root = np.arange(n, dtype=np.int64)
    pt = node_t[pi[take]]
    for frame_t in np.unique(pt):
        rows = take[pt == frame_t]
        root[ci[rows]] = root[pi[rows]]
    linked = np.unique(np.concatenate([pi[take], ci[take]]))
    comp_n = np.bincount(root[linked], minlength=n)
    comp_min_t = np.full(n, int(node_t.max()) + 1, np.int32)
    comp_max_t = np.full(n, -1, np.int32)
    np.minimum.at(comp_min_t, root[linked], node_t[linked])
    np.maximum.at(comp_max_t, root[linked], node_t[linked])

    frame = ctx_frame[STUB_EDGE_FEATURES].iloc[chosen].reset_index(drop=True).copy()
    frame["raw_p"] = raw[chosen]
    frame["emit_p"] = p_emit[chosen]
    frame["is_tail"] = np.fromiter((int(e) in tail_set for e in chosen), bool)
    frame["is_head"] = np.fromiter((int(e) in head_set for e in chosen), bool)
    frame["t_frac"] = ((node_t[pi[chosen]] - first_t)
                       / max(last_t - first_t, 1)).astype(np.float32)
    frame["from_first"] = node_t[pi[chosen]] - first_t
    frame["to_last"] = last_t - node_t[ci[chosen]]
    r = root[pi[chosen]]
    frame["component_nodes"] = comp_n[r]
    frame["component_span"] = comp_max_t[r] - comp_min_t[r]
    frame["edge_from_component_start"] = node_t[pi[chosen]] - comp_min_t[r]
    frame["edge_to_component_end"] = comp_max_t[r] - node_t[ci[chosen]]
    for depth in range(5):
        frame[f"inner_raw_{depth + 1}"] = inner_raw[:, depth]
        frame[f"inner_emit_{depth + 1}"] = inner_emit[:, depth]
    inner_count = np.isfinite(inner_raw).sum(axis=1)
    frame["inner_count"] = inner_count
    frame["inner_raw_mean"] = np.divide(
        np.nansum(inner_raw, axis=1), inner_count,
        out=np.full(len(chosen), np.nan, np.float32), where=inner_count > 0)
    frame["inner_raw_min"] = np.min(
        np.where(np.isfinite(inner_raw), inner_raw, np.inf), axis=1)
    frame.loc[inner_count == 0, "inner_raw_min"] = np.nan
    frame["inner_emit_mean"] = np.divide(
        np.nansum(inner_emit, axis=1), inner_count,
        out=np.full(len(chosen), np.nan, np.float32), where=inner_count > 0)
    return frame, chosen


def tail_prune(take, node_t, pi, ci, leaf_conf, head_conf,
               leaf_bar: float, head_bar: float) -> np.ndarray:
    """One pruning pass (depth 1, mode 'both') — verbatim semantics."""
    node_t = np.asarray(node_t, dtype=np.int64)
    n = len(node_t)
    last_t = int(node_t.max())
    first_t = int(node_t.min())
    out_degree = np.bincount(pi[take], minlength=n)
    in_degree = np.bincount(ci[take], minlength=n)
    premature_leaf = (out_degree[ci[take]] == 0) & (node_t[ci[take]] < last_t)
    single_parent = out_degree[pi[take]] == 1
    premature_root = (in_degree[pi[take]] == 0) & (node_t[pi[take]] > first_t)
    tail_structural = premature_leaf & single_parent
    head_structural = premature_root & single_parent
    remove = ((tail_structural & (leaf_conf[take] < leaf_bar))
              | (head_structural & (head_conf[take] < head_bar)))
    return take[~remove]
