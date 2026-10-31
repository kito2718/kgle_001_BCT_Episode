"""Deploy-side stages of the 0.933 checkpoint (checkpoint-v4c-0933).

Verbatim lifts from the measurement scripts — `build_edge_tracklet_features`,
`build_edge_postselect_features`, `build_component_intensity_features`,
`build_component_checkpoint_features`, `verify_checkpoint_smoothing` — with
file IO replaced by arrays. The neutralised deployment semantics
(is_6bba = 1.0, ep02-only persistence, global component bars) are measured in
`reports/v4c_0933_neutral_official_rows.parquet`; every threshold here is the
frozen one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCALE = np.asarray([1.625, .40625, .40625], np.float32)

TRACKLET_EDGE_FLOOR = 0.30


def _cos(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), den,
                     out=np.zeros(len(a), np.float32), where=den > 1e-8)


def _track_summary(prev, values, depth):
    n = len(prev)
    cur = np.arange(n, dtype=np.int64)
    count = np.ones(n, np.float32)
    total = values.astype(np.float32).copy()
    end = np.arange(n, dtype=np.int64)
    alive = np.ones(n, bool)
    for _ in range(depth):
        nxt = np.full(n, -1, np.int64)
        nxt[alive] = prev[cur[alive]]
        alive &= nxt >= 0
        if not alive.any():
            break
        cur[alive] = nxt[alive]
        end[alive] = cur[alive]
        total[alive] += values[cur[alive]]
        count[alive] += 1
    return total / count[:, None], (count - 1).astype(np.float32), end


def edge_tracklet_features(node_t, base_zyx, pi, ci, raw, p_emit, take,
                           cand_sel, floor: float = TRACKLET_EDGE_FLOOR):
    """The 24-feature five-frame support table over raw >= floor edges.

    `take` is the greedy assignment (no divisions) — the topology the
    training builder used. Returns (frame, rows) with `rows` the edge ids.
    """
    n = len(node_t)
    prev = np.full(n, -1, np.int64)
    nxt = np.full(n, -1, np.int64)
    prev[ci[take]] = pi[take]
    nxt[pi[take]] = ci[take]

    emb_cols = [f"emb_{i}" for i in range(48)]
    emb = cand_sel[emb_cols].to_numpy(np.float32)
    node_score = cand_sel.score.to_numpy(np.float32)
    node_rank = cand_sel.rank_in_frame.to_numpy(np.float32)
    pos = base_zyx.astype(np.float32) * SCALE

    h_emb, h_len, h_end = _track_summary(prev, emb, 5)
    f_emb, f_len, f_end = _track_summary(nxt, emb, 5)
    h_score, _, _ = _track_summary(prev, node_score[:, None], 5)
    f_score, _, _ = _track_summary(nxt, node_score[:, None], 5)
    h_vel = np.divide(pos - pos[h_end], np.maximum(h_len[:, None], 1),
                      out=np.zeros_like(pos), where=h_len[:, None] > 0)
    f_vel = np.divide(pos[f_end] - pos, np.maximum(f_len[:, None], 1),
                      out=np.zeros_like(pos), where=f_len[:, None] > 0)

    rows = np.flatnonzero(raw >= floor)
    p, c = pi[rows], ci[rows]
    link = pos[c] - pos[p]
    hv, fv = h_vel[p], f_vel[c]
    he, fe = h_emb[p], f_emb[c]
    frame = pd.DataFrame({
        "edge_idx": rows.astype(np.int32),
        "raw_p": raw[rows],
        "emit_p": p_emit[rows].astype(np.float32),
        "hist_len5": h_len[p], "fut_len5": f_len[c],
        "hist_speed": np.linalg.norm(hv, axis=1),
        "fut_speed": np.linalg.norm(fv, axis=1),
        "link_speed": np.linalg.norm(link, axis=1),
        "link_hist_res": np.linalg.norm(link - hv, axis=1),
        "link_fut_res": np.linalg.norm(link - fv, axis=1),
        "hist_fut_res": np.linalg.norm(hv - fv, axis=1),
        "link_hist_cos": _cos(link, hv),
        "link_fut_cos": _cos(link, fv),
        "hist_fut_cos": _cos(hv, fv),
        "hist_fut_emb_cos": _cos(he, fe),
        "hist_fut_emb_l2": np.linalg.norm(he - fe, axis=1),
        "parent_hist_emb_cos": _cos(emb[p], he),
        "child_fut_emb_cos": _cos(emb[c], fe),
        "hist_score_mean": h_score[p, 0],
        "fut_score_mean": f_score[c, 0],
        "parent_score": node_score[p], "child_score": node_score[c],
        "parent_rank": node_rank[p], "child_rank": node_rank[c],
    })
    return frame, rows


def _other_max(index, values, selected, n):
    top = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(top, index, values)
    below = values < (top[index] - 1e-7)
    second = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(second, index[below], values[below])
    group = index[selected]
    own = values[selected]
    return np.where(own >= top[group] - 1e-7, second[group], top[group])


def postselect_features(node_t, pi, ci, raw, take, root, comp_score,
                        tracklet, tracklet_rows, track_scores, ctx_frame):
    """The postselect feature frame for the SELECTED edges of `take`.

    Mirrors `build_edge_postselect_features.run_one` (with base edges): the
    tracklet table restricted to selected edges + post_* topology columns +
    `edge_`-prefixed ctx columns + the track model scores. Returns
    (frame, selected_edge_ids).
    """
    n = len(node_t)
    chosen = np.zeros(len(pi), dtype=bool)
    chosen[take] = True
    in_track = chosen[tracklet_rows]
    feature = tracklet[in_track].copy().reset_index(drop=True)
    for col, vals in track_scores.items():
        feature[col] = vals[in_track]
    selected = feature.edge_idx.to_numpy(np.int64)
    p, c = pi[selected], ci[selected]

    out = np.bincount(pi[take], minlength=n)
    incoming = np.bincount(ci[take], minlength=n)
    prev = np.full(n, -1, dtype=np.int64)
    prev_edge = np.full(n, -1, dtype=np.int64)
    next_ = np.full(n, -1, dtype=np.int64)
    next_edge = np.full(n, -1, dtype=np.int64)
    simple_edges = take[(out[pi[take]] == 1) & (incoming[ci[take]] == 1)]
    prev[ci[simple_edges]] = pi[simple_edges]
    prev_edge[ci[simple_edges]] = simple_edges
    next_[pi[simple_edges]] = ci[simple_edges]
    next_edge[pi[simple_edges]] = simple_edges
    hist = np.zeros(n, dtype=np.int16)
    for node in np.argsort(node_t, kind="stable"):
        q = prev[node]
        if q >= 0:
            hist[node] = min(int(hist[q]) + 1, 100)
    future = np.zeros(n, dtype=np.int16)
    for node in np.argsort(-node_t, kind="stable"):
        q = next_[node]
        if q >= 0:
            future[node] = min(int(future[q]) + 1, 100)

    comp_nodes = np.bincount(root, minlength=n)
    other_out = _other_max(pi, raw, selected, n)
    other_in = _other_max(ci, raw, selected, n)
    prior_edge = prev_edge[p]
    following_edge = next_edge[c]
    feature["post_out_degree"] = out[p].astype(np.float32)
    feature["post_in_degree"] = incoming[c].astype(np.float32)
    feature["post_parent_hist"] = hist[p].astype(np.float32)
    feature["post_child_future"] = future[c].astype(np.float32)
    feature["post_min_support"] = np.minimum(hist[p], future[c]).astype(np.float32)
    feature["post_total_support"] = (hist[p] + future[c]).astype(np.float32)
    feature["post_is_head"] = (incoming[p] == 0).astype(np.float32)
    feature["post_is_tail"] = (out[c] == 0).astype(np.float32)
    feature["post_component_nodes"] = comp_nodes[root[p]].astype(np.float32)
    feature["post_component_score"] = comp_score[root[p]].astype(np.float32)
    feature["post_other_out"] = other_out
    feature["post_other_in"] = other_in
    feature["post_out_margin"] = raw[selected] - other_out
    feature["post_in_margin"] = raw[selected] - other_in
    feature["post_prior_raw"] = np.where(
        prior_edge >= 0, raw[np.maximum(prior_edge, 0)], np.nan)
    feature["post_following_raw"] = np.where(
        following_edge >= 0, raw[np.maximum(following_edge, 0)], np.nan)
    feature["post_neighbor_min_raw"] = np.fmin(
        feature.post_prior_raw, feature.post_following_raw)
    feature["post_time_frac"] = node_t[p].astype(np.float32) / max(node_t.max(), 1)

    omit = {"t", "pi", "ci", "label", "is_div", "child_scored"}
    edge_cols = [c_ for c_ in ctx_frame.columns if c_ not in omit]
    edge = (ctx_frame.iloc[selected][edge_cols].reset_index(drop=True)
            .add_prefix("edge_"))
    return pd.concat([feature.reset_index(drop=True), edge], axis=1), selected


def intensity_component_features(vol, node_t, cand_sel, root, nodes, roots):
    """`int_*` component summaries from the detector-grid volume."""
    n = len(node_t)
    group = root[nodes]
    count = np.bincount(group, minlength=n).astype(np.float64)
    tt = cand_sel.t.to_numpy(np.int64)[nodes]
    zz = np.clip(np.rint(cand_sel.z.to_numpy()[nodes]).astype(np.int64), 0, 63)
    yy = np.clip(np.rint(cand_sel.y.to_numpy()[nodes] / 4).astype(np.int64), 0, 63)
    xx = np.clip(np.rint(cand_sel.x.to_numpy()[nodes] / 4).astype(np.int64), 0, 63)
    center = np.asarray(vol[tt, zz, yy, xx], dtype=np.float32)
    sm = np.zeros(len(nodes), dtype=np.float32)
    sq = np.zeros(len(nodes), dtype=np.float32)
    mx = np.full(len(nodes), -np.inf, dtype=np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = np.asarray(vol[tt, np.clip(zz + dz, 0, 63),
                                   np.clip(yy + dy, 0, 63),
                                   np.clip(xx + dx, 0, 63)], dtype=np.float32)
                sm += v
                sq += v * v
                mx = np.maximum(mx, v)
    mean3 = sm / 27.0
    std3 = np.sqrt(np.maximum(sq / 27.0 - mean3 * mean3, 0))
    shell = np.zeros(len(nodes), dtype=np.float32)
    for dz, dy, dx in ((-2, 0, 0), (2, 0, 0), (0, -2, 0),
                       (0, 2, 0), (0, 0, -2), (0, 0, 2)):
        shell += np.asarray(vol[tt, np.clip(zz + dz, 0, 63),
                                np.clip(yy + dy, 0, 63),
                                np.clip(xx + dx, 0, 63)], dtype=np.float32)
    shell /= 6.0
    values = {
        "int_center": center,
        "int_max3": mx,
        "int_mean3": mean3,
        "int_std3": std3,
        "int_center_contrast": center - shell,
        "int_peak_contrast": mx - shell,
        "int_peak_snr": (mx - shell) / np.maximum(std3, 0.02),
    }
    out = {}
    for feature, vals in values.items():
        x = np.asarray(vals, dtype=np.float64)
        smean = np.bincount(group, weights=x, minlength=n)
        ssq = np.bincount(group, weights=x * x, minlength=n)
        mean = np.divide(smean, count, out=np.zeros(n), where=count > 0)
        std = np.sqrt(np.maximum(np.divide(
            ssq, count, out=np.zeros(n), where=count > 0) - mean * mean, 0))
        lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
        np.minimum.at(lo, group, x)
        np.maximum.at(hi, group, x)
        out[f"{feature}_mean"] = mean[roots].astype(np.float32)
        out[f"{feature}_std"] = std[roots].astype(np.float32)
        out[f"{feature}_min"] = lo[roots].astype(np.float32)
        out[f"{feature}_max"] = hi[roots].astype(np.float32)
    return pd.DataFrame(out)


def persistence_component_features(node_t, base_zyx, checkpoint_pool,
                                   root, nodes, roots, tag: str = "ep02"):
    """ep02-only detector-persistence component summaries — the deployable
    view (`checkpoint_pool` is a second detection pass's candidate frame)."""
    from scipy.spatial import cKDTree

    n = len(node_t)
    nt = node_t[nodes].astype(np.int64)
    nzyx = base_zyx[nodes].astype(np.float32)
    distance = np.full(len(nodes), np.inf, dtype=np.float32)
    score = np.full(len(nodes), np.nan, dtype=np.float32)
    rank = np.full(len(nodes), np.nan, dtype=np.float32)
    cand_t = checkpoint_pool.t.to_numpy(np.int64)
    cand_zyx = checkpoint_pool[["z", "y", "x"]].to_numpy(np.float32) * SCALE
    cand_score = checkpoint_pool.score.to_numpy(np.float32)
    cand_rank = checkpoint_pool.rank_in_frame.to_numpy(np.float32)
    for frame in np.intersect1d(np.unique(nt), np.unique(cand_t)):
        ni = np.flatnonzero(nt == frame)
        ai = np.flatnonzero(cand_t == frame)
        tree = cKDTree(cand_zyx[ai])
        d, j = tree.query(nzyx[ni] * SCALE, k=1, workers=1)
        distance[ni] = d
        score[ni] = cand_score[ai[j]]
        rank[ni] = cand_rank[ai[j]]

    frame = pd.DataFrame({"root": root[nodes]})
    frame[f"d_{tag}"] = distance
    frame[f"score_{tag}"] = score
    frame[f"rank_{tag}"] = rank
    for radius in (0.5, 1.0, 2.0, 3.0):
        frame[f"hit{radius:g}_{tag}"] = (distance <= radius).astype(np.float32)
    dmat = distance[:, None]
    smat = score[:, None]
    frame["ckpt_dist_mean_node"] = np.mean(dmat, axis=1)
    frame["ckpt_dist_max_node"] = np.max(dmat, axis=1)
    frame["ckpt_score_mean_node"] = np.nanmean(smat, axis=1)
    for radius in (0.5, 1.0, 2.0, 3.0):
        frame[f"ckpt_hit{radius:g}_frac_node"] = np.mean(dmat <= radius, axis=1)

    value_columns = [c for c in frame.columns if c != "root"]
    grouped = frame.groupby("root", sort=False)
    pieces = [grouped[value_columns].mean().add_suffix("_mean")]
    node_summary = [c for c in value_columns if c.endswith("_node")]
    pieces.extend([
        grouped[node_summary].min().add_suffix("_min"),
        grouped[node_summary].max().add_suffix("_max"),
        grouped[node_summary].std().fillna(0).add_suffix("_std"),
    ])
    out = pd.concat(pieces, axis=1)
    return out.reindex(roots).reset_index(drop=True)


def persistence_component_features_multi(node_t, base_zyx, checkpoint_pools,
                                         root, nodes, roots):
    """Cross-checkpoint component persistence for the research deployment.

    This is the array-backed equivalent of
    ``build_component_checkpoint_features.py``.  ``checkpoint_pools`` is an
    insertion-ordered ``tag -> candidate frame`` mapping.  The two acquisition
    domains legitimately expose different checkpoint sets; model features for
    unavailable views are filled with NaN by the caller, exactly as they were
    in the combined training table.
    """
    from scipy.spatial import cKDTree

    if not checkpoint_pools:
        raise ValueError("at least one checkpoint candidate pool is required")
    n = len(node_t)
    nt = node_t[nodes].astype(np.int64)
    nzyx = base_zyx[nodes].astype(np.float32)
    frame = pd.DataFrame({"root": root[nodes]})
    distances = []
    match_scores = []
    for tag, checkpoint_pool in checkpoint_pools.items():
        distance = np.full(len(nodes), np.inf, dtype=np.float32)
        score = np.full(len(nodes), np.nan, dtype=np.float32)
        rank = np.full(len(nodes), np.nan, dtype=np.float32)
        cand_t = checkpoint_pool.t.to_numpy(np.int64)
        cand_zyx = (checkpoint_pool[["z", "y", "x"]]
                    .to_numpy(np.float32) * SCALE)
        cand_score = checkpoint_pool.score.to_numpy(np.float32)
        cand_rank = checkpoint_pool.rank_in_frame.to_numpy(np.float32)
        for timepoint in np.intersect1d(np.unique(nt), np.unique(cand_t)):
            ni = np.flatnonzero(nt == timepoint)
            ai = np.flatnonzero(cand_t == timepoint)
            tree = cKDTree(cand_zyx[ai])
            distance_i, nearest = tree.query(
                nzyx[ni] * SCALE, k=1, workers=1)
            distance[ni] = distance_i
            score[ni] = cand_score[ai[nearest]]
            rank[ni] = cand_rank[ai[nearest]]
        distances.append(distance)
        match_scores.append(score)
        frame[f"d_{tag}"] = distance
        frame[f"score_{tag}"] = score
        frame[f"rank_{tag}"] = rank
        for radius in (0.5, 1.0, 2.0, 3.0):
            frame[f"hit{radius:g}_{tag}"] = (
                distance <= radius).astype(np.float32)

    dmat = np.column_stack(distances)
    smat = np.column_stack(match_scores)
    frame["ckpt_dist_mean_node"] = np.mean(dmat, axis=1)
    frame["ckpt_dist_max_node"] = np.max(dmat, axis=1)
    frame["ckpt_score_mean_node"] = np.nanmean(smat, axis=1)
    for radius in (0.5, 1.0, 2.0, 3.0):
        frame[f"ckpt_hit{radius:g}_frac_node"] = np.mean(
            dmat <= radius, axis=1)

    value_columns = [column for column in frame if column != "root"]
    grouped = frame.groupby("root", sort=False)
    pieces = [grouped[value_columns].mean().add_suffix("_mean")]
    node_summary = [column for column in value_columns
                    if column.endswith("_node")]
    pieces.extend([
        grouped[node_summary].min().add_suffix("_min"),
        grouped[node_summary].max().add_suffix("_max"),
        grouped[node_summary].std().fillna(0).add_suffix("_std"),
    ])
    out = pd.concat(pieces, axis=1)
    return out.reindex(roots).reset_index(drop=True)


def changed_hop_guard(n, pi, ci, take, hops: int = 2):
    """Fork nodes + their <=`hops`-hop neighbourhood on the final topology."""
    final_out = np.bincount(pi[take], minlength=n)
    guard = final_out >= 2
    up, down = pi[take], ci[take]
    for _ in range(hops):
        touched = guard[up] | guard[down]
        expanded = guard.copy()
        expanded[up[touched]] = True
        expanded[down[touched]] = True
        guard = expanded
    return guard
