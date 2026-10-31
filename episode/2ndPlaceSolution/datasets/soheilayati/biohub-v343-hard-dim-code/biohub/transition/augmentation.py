"""Incumbent-preserving application of globally decoded recovery paths."""

from __future__ import annotations

import json
from typing import Any, Iterable

from biohub.io.submission import DatasetGraph, validate_graph


def apply_anchored_actions(
    incumbent: DatasetGraph,
    witness: DatasetGraph,
    actions: Iterable[dict[str, Any]],
) -> tuple[DatasetGraph, dict[str, Any]]:
    """Append a fixed set of anchored paths without changing incumbent evidence."""

    if incumbent.dataset != witness.dataset:
        raise ValueError("incumbent and witness dataset names differ")
    records = list(actions)
    nodes = dict(incumbent.nodes)
    edges = list(incumbent.edges)
    original_nodes = dict(nodes)
    original_edges = list(edges)
    used_anchors: set[int] = set()
    used_witness_nodes: set[int] = set()
    next_id = max(nodes, default=-1) + 1
    by_kind = {"bridge": 0, "tail": 0, "head": 0}

    for action in records:
        kind = str(action["kind"])
        if kind not in by_kind:
            raise ValueError(f"unknown anchored action kind: {kind}")
        path = [int(node) for node in json.loads(str(action["path_v24_nodes"]))]
        if not path:
            raise ValueError("anchored action path is empty")
        if any(node not in witness.nodes for node in path):
            raise ValueError("anchored action references an unknown witness node")
        start = int(action["mapped_start"])
        end = int(action["mapped_end"])
        anchors = {node for node in (start, end) if node >= 0}
        if any(node not in incumbent.nodes for node in anchors):
            raise ValueError("anchored action references an unknown incumbent anchor")
        if anchors & used_anchors or set(path) & used_witness_nodes:
            raise ValueError("anchored actions overlap in anchors or witness nodes")
        used_anchors.update(anchors)
        used_witness_nodes.update(path)

        new_ids = list(range(next_id, next_id + len(path)))
        next_id += len(path)
        for new_id, witness_id in zip(new_ids, path):
            coordinate = list(map(int, witness.nodes[witness_id]))
            if incumbent.shape is not None:
                coordinate = [
                    min(max(value, 0), size - 1)
                    for value, size in zip(coordinate, incumbent.shape)
                ]
            nodes[new_id] = tuple(coordinate)
        if start >= 0:
            edges.append((start, new_ids[0], 1.0))
        edges.extend((left, right, 1.0) for left, right in zip(new_ids[:-1], new_ids[1:]))
        if end >= 0:
            edges.append((new_ids[-1], end, 1.0))
        by_kind[kind] += 1

    graph = DatasetGraph(
        dataset=incumbent.dataset,
        nodes=nodes,
        edges=edges,
        shape=incumbent.shape,
    )
    incumbent_summary = validate_graph(incumbent)
    output_summary = validate_graph(graph)
    if graph.edges[: len(original_edges)] != original_edges:
        raise RuntimeError("anchored recovery changed an incumbent edge")
    if any(graph.nodes[node] != coordinate for node, coordinate in original_nodes.items()):
        raise RuntimeError("anchored recovery changed an incumbent node")
    if output_summary["forks"] != incumbent_summary["forks"]:
        raise RuntimeError("anchored recovery changed incumbent division topology")
    return graph, {
        "dataset": incumbent.dataset,
        "segments": int(len(records)),
        "bridges": int(by_kind["bridge"]),
        "tails": int(by_kind["tail"]),
        "heads": int(by_kind["head"]),
        "added_nodes": int(len(nodes) - len(original_nodes)),
        "added_edges": int(len(edges) - len(original_edges)),
        "original_nodes_preserved": True,
        "original_edges_preserved": True,
        "division_forks_preserved": True,
    }
