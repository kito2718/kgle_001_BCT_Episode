"""Track post-processing: two-pass linker, gap close, short-track filter, line-fit."""

from __future__ import annotations
import numpy as np
from cell_tracking.common.tracks import SCALE, TrackGraph


def filter_short_tracks(g, min_len: int, keep_division_components: bool = False):
    """Drop whole tracks (connected components) shorter than ``min_len`` nodes."""
    if g.n_edges == 0 or min_len <= 1:
        return g
    parent = {int(n): int(n) for n in g.node_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in g.edges.reshape(-1, 2):
        ra, rb = (find(int(s)), find(int(t)))
        if ra != rb:
            parent[ra] = rb
    comp = {}
    for n in g.node_ids:
        comp.setdefault(find(int(n)), []).append(int(n))
    dividing = set()
    if keep_division_components:
        n_children = {}
        for s, _t in g.edges.reshape(-1, 2):
            n_children[int(s)] = n_children.get(int(s), 0) + 1
        dividing = {find(n) for n, c in n_children.items() if c >= 2}
    keep = set()
    for root, members in comp.items():
        if len(members) >= min_len or root in dividing:
            keep.update(members)
    idx = [i for i, n in enumerate(g.node_ids) if int(n) in keep]
    keepset = {int(g.node_ids[i]) for i in idx}
    edges = np.array(
        [(int(s), int(t)) for s, t in g.edges.reshape(-1, 2) if int(s) in keepset and int(t) in keepset], dtype=np.int64
    ).reshape(-1, 2)
    return TrackGraph(
        node_t=g.node_t[idx],
        node_z=g.node_z[idx],
        node_y=g.node_y[idx],
        node_x=g.node_x[idx],
        node_ids=g.node_ids[idx],
        edges=edges,
        meta=g.meta,
    )


def linefit_smooth(g, window: int = 2, weight: float = 0.8, stop_at_forks: bool = True):
    """Smooth track coordinates while preserving topology."""
    if g.n_edges == 0 or weight <= 0 or window <= 0:
        return g
    ids = [int(n) for n in g.node_ids]
    idx_of = {n: i for i, n in enumerate(ids)}
    pos = {n: np.array([float(g.node_z[i]), float(g.node_y[i]), float(g.node_x[i])]) for n, i in idx_of.items()}
    t_of = {n: int(g.node_t[i]) for n, i in idx_of.items()}
    preds, succs = ({}, {})
    for s, tt in g.edges.reshape(-1, 2):
        s, tt = (int(s), int(tt))
        if t_of.get(tt) != t_of.get(s, -99) + 1:
            continue
        succs.setdefault(s, []).append(tt)
        preds.setdefault(tt, []).append(s)
    w = float(np.clip(weight, 0.0, 1.0))
    new_z = g.node_z.astype(float).copy()
    new_y = g.node_y.astype(float).copy()
    new_x = g.node_x.astype(float).copy()
    for n in ids:
        hood = [(0, n)]
        cur = n
        for step in range(1, window + 1):
            p = preds.get(cur, [])
            if len(p) != 1 or p[0] not in pos or (stop_at_forks and len(succs.get(p[0], [])) != 1):
                break
            cur = p[0]
            hood.append((-step, cur))
        cur = n
        for step in range(1, window + 1):
            s = succs.get(cur, [])
            if len(s) != 1 or s[0] not in pos or (stop_at_forks and len(preds.get(s[0], [])) != 1):
                break
            cur = s[0]
            hood.append((step, cur))
        if len(hood) < 3:
            continue
        dts = np.array([d for d, _ in hood], float)
        coords = np.stack([pos[m] for _, m in hood])
        fit = np.array([np.polyval(np.polyfit(dts, coords[:, ax], 1), 0.0) for ax in range(3)])
        if not np.isfinite(fit).all():
            continue
        i = idx_of[n]
        new_z[i], new_y[i], new_x[i] = (1 - w) * pos[n] + w * fit
    return TrackGraph(
        node_t=g.node_t, node_z=new_z, node_y=new_y, node_x=new_x, node_ids=g.node_ids, edges=g.edges, meta=g.meta
    )


def add_safe_divisions(
    g,
    parent_max_um: float = 4.7,
    sister_max_um: float = 7.2,
    existing_child_max_um: float = 7.8,
    frame_frac_cap: float = 0.008,
    global_frac_cap: float = 0.004,
):
    """Recover divisions under spatial and frequency constraints."""
    if g.n_edges == 0:
        return g
    scale = np.asarray(SCALE)
    ids = [int(n) for n in g.node_ids]
    idx_of = {n: i for i, n in enumerate(ids)}
    t_of = {n: int(g.node_t[i]) for n, i in idx_of.items()}
    pos = {n: np.array([float(g.node_z[i]), float(g.node_y[i]), float(g.node_x[i])]) for n, i in idx_of.items()}

    def d_um(a, b):
        return float(np.linalg.norm((pos[a] - pos[b]) * scale))

    edges = [(int(s), int(t)) for s, t in g.edges.reshape(-1, 2)]
    out_by_src, incoming = ({}, set())
    for s, t in edges:
        out_by_src.setdefault(s, []).append(t)
        incoming.add(t)
    ids_by_t = {}
    for n in ids:
        ids_by_t.setdefault(t_of[n], []).append(n)
    existing = set(edges)
    global_cap = max(1, int(round(max(1, len(edges)) * global_frac_cap)))
    added, used_targets = ([], set())
    for t in sorted(ids_by_t):
        child_ids = ids_by_t.get(t + 1, [])
        if not child_ids:
            continue
        src_ids = [n for n in ids_by_t[t] if len(out_by_src.get(n, [])) == 1]
        cand_ids = [n for n in child_ids if n not in incoming and n not in used_targets]
        if not src_ids or not cand_ids:
            continue
        frame_cap = max(1, int(round(len(src_ids) * frame_frac_cap)))
        proposals = []
        for s in src_ids:
            child = out_by_src[s][0]
            if t_of.get(child) != t + 1:
                continue
            if d_um(s, child) > existing_child_max_um:
                continue
            for c in cand_ids:
                if (s, c) in existing:
                    continue
                pd = d_um(s, c)
                if pd > parent_max_um:
                    continue
                sd = d_um(child, c)
                if sd > sister_max_um:
                    continue
                proposals.append((pd + 0.15 * sd, s, c))
        if not proposals:
            continue
        proposals.sort(key=lambda p: p[0])
        cnt = 0
        for _, s, c in proposals:
            if len(added) >= global_cap or cnt >= frame_cap:
                break
            if c in used_targets or c in incoming:
                continue
            added.append((s, c))
            used_targets.add(c)
            cnt += 1
    if not added:
        return g
    return TrackGraph(
        node_t=g.node_t,
        node_z=g.node_z,
        node_y=g.node_y,
        node_x=g.node_x,
        node_ids=g.node_ids,
        edges=np.array(edges + added, np.int64).reshape(-1, 2),
        meta=g.meta,
    )
