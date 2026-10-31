"""Small deployment-safe refinements for an already constructed graph.

These operations deliberately avoid annotations and training-only tables.
They consume either the submitted graph alone or an explicitly supplied
inference-time detector view, which makes the same code usable on Kaggle's
hidden test set.
"""

from __future__ import annotations

import numpy as np

from biohub.io.submission import DatasetGraph
from biohub.metrics.gt import SCALE_ARR
from biohub.tracker.postpass33 import changed_hop_guard
from biohub.tracker.smoothing import smooth_tracks


def refine_chain_coordinates(
    graph: DatasetGraph,
    *,
    weight: float = 0.05,
    radius: int = 4,
    guard_hops: int = 2,
) -> DatasetGraph:
    """Apply a small extra local-line correction to simple-chain nodes.

    The v14 graph has already been smoothed from fractional detector centres.
    This function is intentionally only a residual correction on the emitted
    integer coordinates.  Forks and their graph neighbourhood remain fixed,
    and topology/confidence order is preserved byte-for-byte.

    A negative ``weight`` is supported as a diagnostic for over-smoothing; it
    moves coordinates away from the local fit.  ``weight=0`` is an exact copy.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if guard_hops < 0:
        raise ValueError("guard_hops must be >= 0")
    if not graph.nodes or not graph.edges or weight == 0:
        return DatasetGraph(
            dataset=graph.dataset,
            nodes=dict(graph.nodes),
            edges=list(graph.edges),
            shape=graph.shape,
        )

    node_ids = np.array(sorted(graph.nodes), dtype=np.int64)
    row_for_id = {int(node_id): i for i, node_id in enumerate(node_ids)}
    node_t = np.array([graph.nodes[int(i)][0] for i in node_ids], dtype=np.int64)
    node_zyx = np.array(
        [graph.nodes[int(i)][1:] for i in node_ids], dtype=np.float64
    )
    pi = np.array([row_for_id[int(s)] for s, _, _ in graph.edges], dtype=np.int64)
    ci = np.array([row_for_id[int(d)] for _, d, _ in graph.edges], dtype=np.int64)
    take = np.arange(len(pi), dtype=np.int64)

    base_um = node_zyx * SCALE_ARR
    fit_um = smooth_tracks(
        node_t, base_um, pi, ci, weight=1.0, radius=int(radius)
    )
    refined_um = base_um + float(weight) * (fit_um - base_um)
    guarded = changed_hop_guard(len(node_ids), pi, ci, take, hops=int(guard_hops))
    refined_um[guarded] = base_um[guarded]
    refined_zyx = np.rint(refined_um / SCALE_ARR).astype(np.int64)

    if graph.shape is not None:
        for axis, size in enumerate(graph.shape[1:]):
            refined_zyx[:, axis] = np.clip(refined_zyx[:, axis], 0, size - 1)

    return DatasetGraph(
        dataset=graph.dataset,
        nodes={
            int(node_id): (int(node_t[i]), *(int(v) for v in refined_zyx[i]))
            for i, node_id in enumerate(node_ids)
        },
        edges=list(graph.edges),
        shape=graph.shape,
    )


def blend_detector_coordinates(
    graph: DatasetGraph,
    candidates,
    *,
    gate_um: float = 3.0,
    alpha: float = 0.5,
    guard_hops: int = 2,
    sigma_gate_um: float = 1.5,
) -> DatasetGraph:
    """Blend graph nodes with a second detector's fractional centres.

    A graph node is eligible only when it and a candidate are mutual nearest
    neighbours in the same frame, their physical separation is at most
    ``gate_um``, and the detector's predicted flow uncertainty is at most
    ``sigma_gate_um``.  Forks and their ``guard_hops`` neighbourhood stay
    byte-identical.  Topology, confidence order, node ids, and node count are
    never changed.

    The defaults are the fixed full-OOF winner from the four live scoring
    samples: 0.928189672 -> 0.933838043 adjusted edge Jaccard.  Ground truth is
    not an input to this function.
    """
    from scipy.spatial import cKDTree

    if gate_um <= 0:
        raise ValueError("gate_um must be > 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if guard_hops < 0:
        raise ValueError("guard_hops must be >= 0")
    if sigma_gate_um <= 0:
        raise ValueError("sigma_gate_um must be > 0")
    required = {"t", "z", "y", "x", "flow_sigma_um"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"detector candidates missing columns: {missing}")
    if not graph.nodes or alpha == 0 or len(candidates) == 0:
        return DatasetGraph(
            dataset=graph.dataset,
            nodes=dict(graph.nodes),
            edges=list(graph.edges),
            shape=graph.shape,
        )

    node_ids = np.asarray(sorted(graph.nodes), np.int64)
    row_for_id = {int(node_id): i for i, node_id in enumerate(node_ids)}
    node_t = np.asarray([graph.nodes[int(i)][0] for i in node_ids], np.int64)
    node_zyx = np.asarray([graph.nodes[int(i)][1:] for i in node_ids], np.float64)
    node_um = node_zyx * SCALE_ARR
    pi = np.asarray([row_for_id[int(s)] for s, _, _ in graph.edges], np.int64)
    ci = np.asarray([row_for_id[int(d)] for _, d, _ in graph.edges], np.int64)

    cand_t = candidates["t"].to_numpy(np.int64)
    cand_zyx = candidates[["z", "y", "x"]].to_numpy(np.float64)
    cand_um = cand_zyx * SCALE_ARR
    cand_sigma = candidates["flow_sigma_um"].to_numpy(np.float64)
    mapped = np.full(len(node_ids), -1, np.int64)
    map_um = np.full(len(node_ids), np.inf, np.float64)
    for t in np.unique(node_t):
        node_rows = np.flatnonzero(node_t == t)
        cand_rows = np.flatnonzero(cand_t == t)
        if not len(node_rows) or not len(cand_rows):
            continue
        distance, nearest_cand = cKDTree(cand_um[cand_rows]).query(
            node_um[node_rows], k=1
        )
        _, nearest_node = cKDTree(node_um[node_rows]).query(
            cand_um[cand_rows], k=1
        )
        mutual = nearest_node[nearest_cand] == np.arange(len(node_rows))
        mapped[node_rows[mutual]] = cand_rows[nearest_cand[mutual]]
        map_um[node_rows[mutual]] = distance[mutual]

    guard = changed_hop_guard(
        len(node_ids), pi, ci, np.arange(len(pi), dtype=np.int64),
        hops=int(guard_hops),
    )
    safe_mapped = np.maximum(mapped, 0)
    eligible = (
        (mapped >= 0)
        & (map_um <= float(gate_um))
        & (cand_sigma[safe_mapped] <= float(sigma_gate_um))
        & ~guard
    )
    fresh = node_zyx.copy()
    fresh[eligible] += float(alpha) * (
        cand_zyx[mapped[eligible]] - fresh[eligible]
    )
    rounded = np.rint(fresh).astype(np.int64)
    if graph.shape is not None:
        rounded = np.clip(
            rounded, np.zeros(3, np.int64), np.asarray(graph.shape[1:], np.int64) - 1
        )

    return DatasetGraph(
        dataset=graph.dataset,
        nodes={
            int(node_id): (int(node_t[i]), *(int(v) for v in rounded[i]))
            for i, node_id in enumerate(node_ids)
        },
        edges=list(graph.edges),
        shape=graph.shape,
    )


def flow_consistent_coordinates(
    graph: DatasetGraph,
    candidates,
    *,
    beta: float = 0.0,
    map_gate_um: float = 3.0,
    residual_gate_um: float = 3.0,
    sigma_gate_um: float = 1.5,
    guard_hops: int = 2,
) -> DatasetGraph:
    """Move simple-chain parents toward child-detector backward-flow targets."""
    from scipy.spatial import cKDTree

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if map_gate_um <= 0 or residual_gate_um <= 0 or sigma_gate_um <= 0:
        raise ValueError("all physical gates must be positive")
    if guard_hops < 0:
        raise ValueError("guard_hops must be >= 0")
    required = {
        "t", "z", "y", "x", "flow_z", "flow_y", "flow_x",
        "flow_sigma_um",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"detector candidates missing columns: {missing}")
    if not graph.nodes or not graph.edges or beta == 0.0 or len(candidates) == 0:
        return DatasetGraph(
            dataset=graph.dataset, nodes=dict(graph.nodes),
            edges=list(graph.edges), shape=graph.shape,
        )

    node_ids = np.asarray(sorted(graph.nodes), np.int64)
    row_for_id = {int(node_id): i for i, node_id in enumerate(node_ids)}
    node_t = np.asarray([graph.nodes[int(i)][0] for i in node_ids], np.int64)
    node_zyx = np.asarray([graph.nodes[int(i)][1:] for i in node_ids], np.float64)
    node_um = node_zyx * SCALE_ARR
    pi = np.asarray([row_for_id[int(s)] for s, _, _ in graph.edges], np.int64)
    ci = np.asarray([row_for_id[int(d)] for _, d, _ in graph.edges], np.int64)

    cand_t = candidates["t"].to_numpy(np.int64)
    cand_zyx = candidates[["z", "y", "x"]].to_numpy(np.float64)
    cand_um = cand_zyx * SCALE_ARR
    cand_flow = candidates[
        ["flow_z", "flow_y", "flow_x"]
    ].to_numpy(np.float64)
    cand_sigma = candidates["flow_sigma_um"].to_numpy(np.float64)

    mapped = np.full(len(node_ids), -1, np.int64)
    map_um = np.full(len(node_ids), np.inf, np.float64)
    for t in np.unique(node_t):
        ni = np.flatnonzero(node_t == t)
        cj = np.flatnonzero(cand_t == t)
        if not len(ni) or not len(cj):
            continue
        distance, nearest_cand = cKDTree(cand_um[cj]).query(node_um[ni], k=1)
        _, nearest_node = cKDTree(node_um[ni]).query(cand_um[cj], k=1)
        mutual = nearest_node[nearest_cand] == np.arange(len(ni))
        mapped[ni[mutual]] = cj[nearest_cand[mutual]]
        map_um[ni[mutual]] = distance[mutual]

    indeg = np.bincount(ci, minlength=len(node_ids))
    outdeg = np.bincount(pi, minlength=len(node_ids))
    guard = changed_hop_guard(
        len(node_ids), pi, ci, np.arange(len(pi), dtype=np.int64),
        hops=int(guard_hops),
    )
    child_map = mapped[ci]
    safe_map = np.maximum(child_map, 0)
    proposed_parent = cand_um[safe_map] + cand_flow[safe_map]
    residual = np.linalg.norm(proposed_parent - node_um[pi], axis=1)
    eligible = (
        (outdeg[pi] == 1)
        & (indeg[ci] == 1)
        & (child_map >= 0)
        & (map_um[ci] <= float(map_gate_um))
        & (cand_sigma[safe_map] <= float(sigma_gate_um))
        & (residual <= float(residual_gate_um))
        & ~guard[pi]
        & ~guard[ci]
    )

    fresh_um = node_um.copy()
    parents = pi[eligible]
    fresh_um[parents] += float(beta) * (
        proposed_parent[eligible] - fresh_um[parents]
    )
    rounded = np.rint(fresh_um / SCALE_ARR).astype(np.int64)
    if graph.shape is not None:
        rounded = np.clip(
            rounded, np.zeros(3, np.int64),
            np.asarray(graph.shape[1:], np.int64) - 1,
        )
    return DatasetGraph(
        dataset=graph.dataset,
        nodes={
            int(node_id): (int(node_t[i]), *(int(v) for v in rounded[i]))
            for i, node_id in enumerate(node_ids)
        },
        edges=list(graph.edges),
        shape=graph.shape,
    )
