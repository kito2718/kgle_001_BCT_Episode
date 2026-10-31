"""Deterministic graph control built only from held-out detector candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, drop_edgeless_nodes, validate_graph
from biohub.metrics.gt import SCALE_ARR


REQUIRED_COLUMNS = {
    "t",
    "z",
    "y",
    "x",
    "rank_in_frame",
    "flow_z",
    "flow_y",
    "flow_x",
}


def _integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or not (values == np.rint(values)).all():
        raise ValueError(f"candidate column {column!r} must contain finite integers")
    return values.astype(np.int64)


def build_backward_flow_graph(
    dataset: str,
    candidates: pd.DataFrame,
    *,
    rank_limit: int = 250,
    gate_um: float = 7.0,
    max_out_degree: int = 1,
) -> tuple[DatasetGraph, dict[str, int | float]]:
    """Build a conservative control graph without labels or trained association.

    The top ``rank_limit`` detector candidates in each frame become the node
    pool. Each child independently claims the nearest parent to its predicted
    backward-flow endpoint. Claims are accepted in increasing residual order
    under a fixed parent out-degree cap. Nodes that participate in no accepted
    edge are removed because they can only hurt the official metric.
    """
    missing = sorted(REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"{dataset}: candidate table misses columns {missing}")
    if rank_limit <= 0:
        raise ValueError("rank_limit must be positive")
    if not gate_um > 0:
        raise ValueError("gate_um must be positive")
    if max_out_degree not in (1, 2):
        raise ValueError("max_out_degree must be 1 or 2")

    frame = candidates.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    frame["t"] = _integer_column(frame, "t")
    frame["rank_in_frame"] = _integer_column(frame, "rank_in_frame")
    frame = frame[frame["rank_in_frame"] < rank_limit]
    frame = frame.sort_values(
        ["t", "rank_in_frame", "z", "y", "x"], kind="stable"
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{dataset}: no candidates survive rank_limit={rank_limit}")
    if frame.duplicated(["t", "rank_in_frame"]).any():
        raise ValueError(f"{dataset}: duplicate (t, rank_in_frame) candidate key")

    zyx = frame[["z", "y", "x"]].to_numpy(np.float64)
    flow = frame[["flow_z", "flow_y", "flow_x"]].to_numpy(np.float64)
    if not np.isfinite(zyx).all() or not np.isfinite(flow).all():
        raise ValueError(f"{dataset}: non-finite candidate coordinate or flow")
    times = frame["t"].to_numpy(np.int64)
    rounded = np.rint(zyx).astype(np.int64)
    node_ids = np.arange(len(frame), dtype=np.int64)
    nodes = {
        int(node_id): (int(t), int(z), int(y), int(x))
        for node_id, t, (z, y, x) in zip(node_ids, times, rounded, strict=True)
    }

    edges: list[tuple[int, int, float]] = []
    out_degree = np.zeros(len(frame), dtype=np.int8)
    for timepoint in np.unique(times):
        parents = np.flatnonzero(times == timepoint)
        children = np.flatnonzero(times == timepoint + 1)
        if not len(parents) or not len(children):
            continue
        parent_um = zyx[parents] * SCALE_ARR
        predicted_parent_um = zyx[children] * SCALE_ARR + flow[children]
        distance, local_parent = cKDTree(parent_um).query(
            predicted_parent_um, k=1, distance_upper_bound=gate_um
        )
        valid = np.isfinite(distance) & (local_parent < len(parents))
        claim_parent = parents[local_parent[valid]]
        claim_child = children[valid]
        claim_distance = distance[valid]
        # residual is primary; integer endpoints make every tie deterministic.
        order = np.lexsort((claim_child, claim_parent, claim_distance))
        for row in order:
            parent = int(claim_parent[row])
            if out_degree[parent] >= max_out_degree:
                continue
            child = int(claim_child[row])
            out_degree[parent] += 1
            edges.append((parent, child, float(gate_um - claim_distance[row])))

    raw_graph = DatasetGraph(dataset=dataset, nodes=nodes, edges=edges)
    graph = drop_edgeless_nodes(raw_graph)
    topology = validate_graph(graph)
    diagnostics: dict[str, int | float] = {
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(frame)),
        "dropped_edgeless": int(len(raw_graph.nodes) - len(graph.nodes)),
        "nodes": int(topology["nodes"]),
        "edges": int(topology["edges"]),
        "forks": int(topology["forks"]),
        "births": int(topology["births"]),
        "deaths": int(topology["deaths"]),
        "mean_disp_um": float(topology["mean_disp_um"]),
        "p99_disp_um": float(topology["p99_disp_um"]),
    }
    return graph, diagnostics
