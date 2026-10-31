"""Build and validate `submission.csv` — PLAN.md §15.

Two responsibilities:

1. **Ordering.** The official scorer truncates out-degree > 2 by keeping the two
   *lowest edge ids*, i.e. by insertion order, not by confidence
   (`tests/test_metric_semantics.py::test_out_degree_above_two_is_truncated_by_edge_id_order`).
   Edges are therefore written in descending confidence so that anything the
   scorer silently drops is the part we believed least.

2. **Validation.** Every structural assumption the scorer makes is asserted here
   and fails loudly. The scorer *repairs* invalid graphs rather than rejecting
   them, so a malformed submission scores quietly wrong instead of erroring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
UNUSED = -1


@dataclass
class DatasetGraph:
    """One dataset's predicted graph, in voxel coordinates."""

    dataset: str
    # node_id -> (t, z, y, x); ids must be unique within the dataset
    nodes: dict[int, tuple[int, int, int, int]]
    # (source_id, target_id, confidence); confidence orders the output
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    shape: tuple[int, int, int, int] | None = None  # (T, Z, Y, X) for bounds checks


class SubmissionError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SubmissionError(msg)


def validate_graph(g: DatasetGraph) -> dict:
    """Assert PLAN.md §15 invariants; return a per-dataset summary."""
    ds = g.dataset
    node_ids = np.fromiter(g.nodes.keys(), dtype=np.int64, count=len(g.nodes))
    _check(len(node_ids) == len(set(node_ids.tolist())), f"{ds}: duplicate node ids")

    coords = np.array(list(g.nodes.values()), dtype=np.float64) if g.nodes else np.zeros((0, 4))
    _check(np.isfinite(coords).all(), f"{ds}: non-finite coordinates")
    _check((coords == coords.astype(np.int64)).all(), f"{ds}: non-integer coordinates")
    if g.shape is not None and len(coords):
        for axis, (name, size) in enumerate(zip("tzyx", g.shape)):
            _check(
                (coords[:, axis] >= 0).all() and (coords[:, axis] < size).all(),
                f"{ds}: {name} out of bounds [0,{size})",
            )

    known = set(g.nodes)
    seen_pairs: set[tuple[int, int]] = set()
    in_deg: dict[int, int] = {}
    out_deg: dict[int, int] = {}
    for s, d, conf in g.edges:
        _check(s in known and d in known, f"{ds}: edge references unknown node ({s}->{d})")
        _check(s != d, f"{ds}: self edge at {s}")
        _check((s, d) not in seen_pairs, f"{ds}: duplicate edge {s}->{d}")
        _check(np.isfinite(conf), f"{ds}: non-finite confidence on {s}->{d}")
        seen_pairs.add((s, d))
        ts, td = g.nodes[s][0], g.nodes[d][0]
        _check(td == ts + 1, f"{ds}: edge {s}->{d} spans t={ts}->{td}, must be +1")
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[d] = in_deg.get(d, 0) + 1

    for n, k in in_deg.items():
        _check(k <= 1, f"{ds}: node {n} has in-degree {k} (merge)")
    for n, k in out_deg.items():
        _check(k <= 2, f"{ds}: node {n} has out-degree {k} (>2 children)")

    n_edge_endpoints = len(set(s for s, _, _ in g.edges) | set(d for _, d, _ in g.edges))
    disp = []
    for s, d, _ in g.edges:
        a, b = np.array(g.nodes[s][1:], float), np.array(g.nodes[d][1:], float)
        disp.append(np.linalg.norm((b - a) * np.array([1.625, 0.40625, 0.40625])))
    disp = np.array(disp) if disp else np.zeros(1)

    return {
        "dataset": ds,
        "nodes": len(g.nodes),
        "edges": len(g.edges),
        "forks": sum(1 for k in out_deg.values() if k == 2),
        "births": len(g.nodes) - len(in_deg),
        "deaths": len(g.nodes) - len(out_deg),
        "edgeless_nodes": len(g.nodes) - n_edge_endpoints,
        "mean_disp_um": float(disp.mean()),
        "p99_disp_um": float(np.percentile(disp, 99)),
    }


def drop_edgeless_nodes(g: DatasetGraph) -> DatasetGraph:
    """Remove nodes that participate in no edge.

    Nodes score only as edge endpoints, as division-component members, and
    through the count multiplier, so an edge-less node is pure count cost with
    no upside (`test_edgeless_nodes_are_pure_cost`). Dropping them is strictly
    non-negative under this metric.
    """
    keep = set(s for s, _, _ in g.edges) | set(d for _, d, _ in g.edges)
    return DatasetGraph(
        dataset=g.dataset,
        nodes={k: v for k, v in g.nodes.items() if k in keep},
        edges=list(g.edges),
        shape=g.shape,
    )


def build_submission(graphs: list[DatasetGraph], validate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (submission dataframe, per-dataset summary)."""
    _check(len(graphs) > 0, "no datasets in submission")
    names = [g.dataset for g in graphs]
    _check(len(names) == len(set(names)), "duplicate dataset names")

    rows: list[tuple] = []
    summaries = []
    for g in graphs:
        if validate:
            summaries.append(validate_graph(g))

        for nid, (t, z, y, x) in sorted(g.nodes.items()):
            rows.append((g.dataset, "node", nid, int(t), int(z), int(y), int(x), UNUSED, UNUSED))

        # Descending confidence: if the scorer truncates, it keeps what we trust.
        for s, d, _ in sorted(g.edges, key=lambda e: -e[2]):
            rows.append((g.dataset, "edge", UNUSED, UNUSED, UNUSED, UNUSED, UNUSED, s, d))

    df = pd.DataFrame(rows, columns=COLUMNS[1:])
    df.insert(0, "id", np.arange(len(df), dtype=np.int64))
    return df, pd.DataFrame(summaries)


def write_submission(df: pd.DataFrame, path) -> None:
    _check(list(df.columns) == COLUMNS, f"column mismatch: {list(df.columns)}")
    _check((df["id"].to_numpy() == np.arange(len(df))).all(), "ids must be consecutive from 0")
    df.to_csv(path, index=False)
