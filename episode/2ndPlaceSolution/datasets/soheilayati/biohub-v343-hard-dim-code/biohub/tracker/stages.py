"""Deploy-side graph-action stages of the exact-v14 composed lineage.

The 0.938912 checkpoint (`Progress/checkpoint-v14-postedge-093891.md`) is a
chain of graph rewrites applied to the deployed pipeline-v8 graph. Each stage
is `build features -> average five fold models -> apply a conflict policy`.
This module holds the POLICY half as pure functions over graph arrays, lifted
verbatim from the frozen verifiers so the deployed rewrite and the verified
rewrite cannot diverge:

  fork_graph_veto      verify_v14_partialcomposed_093446_checkpoint.py
  edge_neighborhood    same file, `verify_edge_neighborhood`
  child swap / reparent  verify_current_childswap_topk.apply_actions /
                         verify_current_reparent_all.apply_actions

Scores are supplied by the caller: fold-held-out locally (which must
reproduce the stored graph digests exactly) and the five-fold mean at
deployment, the same convention every shipped kernel uses.

Every function takes and returns `take` — the active-edge row indices into
the frozen `(pi, ci)` edge universe — so stages compose without rebuilding
the edge table.
"""

from __future__ import annotations

import numpy as np

# Frozen thresholds. Each is the measured operating point of its stage; they
# are not tunable here.
FORK_VETO_TAU = 0.3341144621372223   # joint_symscore_k40_c0p01
EDGE_NB_TAU = 0.06                   # edgenb_post_l15, non-fork edges only
EDGE_NB_SHORTLIST_TAU = 0.20         # post_lgb_l31 shortlist for the rebuild


def fork_graph_veto(pi, ci, take, fork_pairs, post_l31, tau=FORK_VETO_TAU):
    """Drop the weaker arm of a fork whose joint symmetry score is below tau.

    `fork_pairs` is an iterable of (par, c1, c2, joint_symscore) over the
    current graph's evaluable forks; `post_l31` is the per-edge selected-edge
    score used only to choose WHICH arm dies (NaN reads as -1, i.e. weakest).
    Verbatim from `verify_fork_graph_veto`.
    """
    n_edges = len(pi)
    active = np.zeros(n_edges, dtype=bool)
    active[take] = True
    post = np.asarray(post_l31, dtype=np.float32)
    accepted = 0
    for par, c1, c2, score in fork_pairs:
        if float(score) >= tau:
            continue
        event = np.flatnonzero(active & (pi == int(par))
                               & np.isin(ci, [int(c1), int(c2)]))
        if len(event) < 2:
            continue
        active[int(event[np.argmin(np.nan_to_num(post[event], nan=-1.0))])] = False
        accepted += 1
    return np.flatnonzero(active).astype(np.int64), accepted


def edge_neighborhood_prune(pi, take, n_nodes, nb_score, tau=EDGE_NB_TAU):
    """Remove non-fork edges whose neighbourhood score is below tau.

    Fork arms are always protected. `nb_score` is per-edge, +inf where the
    stage abstains (an edge outside the shortlist). Verbatim from
    `verify_edge_neighborhood`.
    """
    out_degree = np.bincount(pi[take], minlength=n_nodes)
    fork = out_degree[pi[take]] >= 2
    values = np.asarray(nb_score, dtype=np.float32)
    kept = take[(values[take] >= tau) | fork]
    return kept.astype(np.int64), int(len(take) - len(kept))


def apply_child_swap(pi, ci, take, n_nodes, actions, values, threshold, cap=0):
    """Sequential child-swap policy — VERBATIM from
    `verify_current_childswap_topk.apply_actions`.

    The displaced edge is resolved DYNAMICALLY from `current_in`, not read
    from a stored column, and the action is gated on the parent's current
    single out-edge still being the incumbent. `actions` is the action frame
    (needs `parent`, `old_edge`, `new_edge`); `values` its score array.
    """
    pi = np.asarray(pi, dtype=np.int64)
    ci = np.asarray(ci, dtype=np.int64)
    take = np.asarray(take, dtype=np.int64)
    active = np.zeros(len(pi), bool)
    active[take] = True
    current_in = np.full(n_nodes, -1, np.int64)
    current_in[ci[take]] = take
    current_out = np.full(n_nodes, -1, np.int64)
    degree = np.bincount(pi[take], minlength=n_nodes)
    simple = take[degree[pi[take]] == 1]
    current_out[pi[simple]] = simple
    values = np.asarray(values, dtype=np.float32)
    accepted = 0
    for index in np.argsort(-values, kind="stable"):
        value = float(values[index])
        if value < threshold or (cap > 0 and accepted >= cap):
            break
        row = actions.iloc[int(index)]
        parent = int(row.parent)
        old, new = int(row.old_edge), int(row.new_edge)
        if current_out[parent] != old or not active[old] or active[new]:
            continue
        new_child = int(ci[new])
        drop = int(current_in[new_child])
        # Remove the old junk-child edge and any stolen incoming edge at the
        # replacement child, then install the coordinated alternative.
        active[old] = False
        current_in[int(ci[old])] = -1
        if drop >= 0:
            active[drop] = False
            current_in[new_child] = -1
            drop_parent = int(pi[drop])
            if current_out[drop_parent] == drop:
                current_out[drop_parent] = -1
        active[new] = True
        current_in[new_child] = new
        current_out[parent] = new
        accepted += 1
    return np.flatnonzero(active), accepted


def apply_reparent(pi, ci, take, n_nodes, conf, actions, values, threshold,
                   cap=0):
    """Sequential reparent policy — VERBATIM from
    `verify_current_reparent_all.apply_actions`.

    Preserves the receiving parent's ORIGINAL outgoing capacity: when full,
    its weakest current out-edge (by emission priority, seeded from `conf`)
    is displaced. `actions` needs `child`, `old_edge`, `new_edge`.
    """
    import pandas as pd

    pi = np.asarray(pi, dtype=np.int64)
    ci = np.asarray(ci, dtype=np.int64)
    take = np.asarray(take, dtype=np.int64)
    selected = np.zeros(len(pi), bool)
    selected[take] = True
    current = np.full(n_nodes, -1, np.int64)
    current[ci[take]] = take
    original_degree = np.bincount(pi[take], minlength=n_nodes)
    priority = np.zeros(len(pi), np.float32)
    priority[take] = conf
    outgoing = {int(parent): set(map(int, edges))
                for parent, edges in pd.Series(take).groupby(pi[take])}
    values = np.asarray(values, dtype=np.float32)
    accepted = 0
    for index in np.argsort(-values, kind="stable"):
        value = float(values[index])
        if value < threshold or (cap > 0 and accepted >= cap):
            break
        row = actions.iloc[int(index)]
        child, old, new = int(row.child), int(row.old_edge), int(row.new_edge)
        if current[child] != old or not selected[old] or selected[new]:
            continue
        new_parent = int(pi[new])
        capacity = max(1, int(original_degree[new_parent]))
        target = outgoing.get(new_parent, set())
        # Accept first, then clear the old endpoint and the weakest capacity
        # conflict.  This matches the degree-preserving oracle's action space.
        selected[old] = False
        current[child] = -1
        old_parent = int(pi[old])
        outgoing[old_parent].remove(old)
        if len(target) >= capacity:
            drop = min(target, key=priority.__getitem__)
            selected[drop] = False
            current[int(ci[drop])] = -1
            target.remove(drop)
        selected[new] = True
        current[child] = new
        priority[new] = value
        target.add(new)
        outgoing[new_parent] = target
        accepted += 1
    return np.flatnonzero(selected), accepted


def _veto_pairs(pi, ci, active, pairs, model_getter, threshold, post, raw=None):
    """Shared fork-veto loop: drop the weaker arm of a sub-threshold fork.

    `post` chooses which arm dies (NaN -> -1). When every candidate arm is
    unscored, the residual/lowgate stages fall back to `raw`. Verbatim from
    `materialize_current_residual_fork_graph` / `..._neighborhood_lowgate`.
    """
    removed = 0
    for row in pairs:
        value = float(model_getter(row))
        if not np.isfinite(value) or value >= threshold:
            continue
        event = np.flatnonzero(active & (pi == int(row.par))
                               & np.isin(ci, [int(row.c1), int(row.c2)]))
        if len(event) < 2:
            continue
        strength = np.nan_to_num(post[event], nan=-1.0)
        if raw is not None and np.all(strength == -1.0):
            strength = raw[event]
        active[int(event[np.argmin(strength)])] = False
        removed += 1
    return removed


def component_meta_prune(pi, ci, take, root, comp_score, threshold,
                         fork_pairs, fork_getter, fork_threshold, post):
    """Component threshold on the PARENT's root, then a fork veto.

    Verbatim from `materialize_current_component_meta_graph`: the component
    mask keys on `root[pi[take]]` only, and the veto runs on the masked graph.
    """
    keep = comp_score[root[pi[take]]] >= threshold
    active = np.zeros(len(pi), bool)
    active[take[keep]] = True
    vetoes = _veto_pairs(pi, ci, active, fork_pairs, fork_getter,
                         fork_threshold, post)
    return np.flatnonzero(active).astype(np.int64), vetoes


def fragment_prune(pi, ci, take, node_t, raw, comp_score_by_root,
                   fragment_threshold, threshold):
    """Rebuild roots over STRONG edges only, then require BOTH endpoints'
    components to clear the bar — `materialize_current_fragment_graph`."""
    n_nodes = len(node_t)
    out_degree = np.bincount(pi[take], minlength=n_nodes)
    strong = take[(raw[take] >= fragment_threshold)
                  | (out_degree[pi[take]] >= 2)]
    root = np.arange(n_nodes, dtype=np.int64)
    pt = node_t[pi[strong]]
    for frame in np.unique(pt):
        rows = strong[pt == frame]
        root[ci[rows]] = root[pi[rows]]
    value = np.asarray(comp_score_by_root, dtype=np.float32)
    keep = ((value[root[pi[take]]] >= threshold)
            & (value[root[ci[take]]] >= threshold))
    return take[keep].astype(np.int64), int(keep.sum())


def residual_fork_veto(pi, ci, take, raw, post, raw_pairs, raw_getter,
                       raw_threshold, meta_pairs, meta_getter,
                       meta_threshold):
    """Two residual fork-veto cascades — raw feature first, then the meta
    model. `materialize_current_residual_fork_graph`."""
    active = np.zeros(len(pi), bool)
    active[take] = True
    raw_vetoes = _veto_pairs(pi, ci, active, raw_pairs, raw_getter,
                             raw_threshold, post, raw)
    meta_vetoes = _veto_pairs(pi, ci, active, meta_pairs, meta_getter,
                              meta_threshold, post, raw)
    return np.flatnonzero(active).astype(np.int64), raw_vetoes, meta_vetoes


def graph_digest_arrays(take):
    """Stable per-sample content key for a rewritten graph (validation aid)."""
    import hashlib

    return hashlib.sha256(
        np.asarray(take, dtype=np.int32).tobytes()).hexdigest()
