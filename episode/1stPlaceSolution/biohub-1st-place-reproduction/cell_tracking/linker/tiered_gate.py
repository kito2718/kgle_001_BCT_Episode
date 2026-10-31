"""Confidence-tiered temporal gating: faint detections must earn their place in a track."""

from __future__ import annotations
import numpy as np
from cell_tracking.common.tracks import TrackGraph


def track_components(g) -> dict[int, list[int]]:
    """``{root_node_id: [member node ids]}`` over the graph's undirected connectivity."""
    parent = {int(n): int(n) for n in g.node_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in np.asarray(g.edges).reshape(-1, 2):
        ra, rb = (find(int(s)), find(int(t)))
        if ra != rb:
            parent[ra] = rb
    comp: dict[int, list[int]] = {}
    for n in g.node_ids:
        comp.setdefault(find(int(n)), []).append(int(n))
    return comp


def dividing_roots(g, comp: dict[int, list[int]]) -> set[int]:
    """Component roots holding a node with two or more children."""
    root = {n: r for r, members in comp.items() for n in members}
    n_children: dict[int, int] = {}
    for s, _t in np.asarray(g.edges).reshape(-1, 2):
        n_children[int(s)] = n_children.get(int(s), 0) + 1
    return {root[n] for n, c in n_children.items() if c >= 2 and n in root}


def align_conf(g, conf_by_id: dict, default: float = np.inf) -> np.ndarray:
    """Confidences in *g*'s current node order, from a ``{node_id: conf}`` mapping."""
    return np.array([float(conf_by_id.get(int(n), default)) for n in g.node_ids], dtype=np.float64)


def tiered_temporal_gate(g, conf, *, conf_thresh: float, min_len: int = 3, keep_division_components: bool = True):
    """Drop detections below *conf_thresh* unless a track of *min_len* nodes carries them."""
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    if len(conf) != len(g.node_ids):
        raise ValueError(
            f"conf has {len(conf)} entries but the graph has {len(g.node_ids)} nodes; they must be row-aligned (see detect_unetcellpose_conf)"
        )
    if min_len <= 1:
        return g
    comp = track_components(g)
    dividing = dividing_roots(g, comp) if keep_division_components else set()
    supported = {n for root, members in comp.items() for n in members if len(members) >= min_len or root in dividing}
    keep_idx = [i for i, n in enumerate(g.node_ids) if conf[i] >= conf_thresh or int(n) in supported]
    if len(keep_idx) == len(g.node_ids):
        return g
    keepset = {int(g.node_ids[i]) for i in keep_idx}
    edges = np.array(
        [(int(s), int(t)) for s, t in np.asarray(g.edges).reshape(-1, 2) if int(s) in keepset and int(t) in keepset],
        dtype=np.int64,
    ).reshape(-1, 2)
    return TrackGraph(
        node_t=g.node_t[keep_idx],
        node_z=g.node_z[keep_idx],
        node_y=g.node_y[keep_idx],
        node_x=g.node_x[keep_idx],
        node_ids=g.node_ids[keep_idx],
        edges=edges,
        meta=g.meta,
    )
