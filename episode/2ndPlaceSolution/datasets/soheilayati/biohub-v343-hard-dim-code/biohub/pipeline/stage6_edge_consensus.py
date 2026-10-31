"""Runtime-safe Stage-6/V24 ordinary-edge consensus for V37c."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.matching import match_sequence


MAP_UM = 3.0


def _mapped_ordinary_edges(
    base: DatasetGraph,
    witness: DatasetGraph,
) -> set[tuple[int, int]]:
    base_ids = np.fromiter(base.nodes, dtype=np.int64, count=len(base.nodes))
    witness_ids = np.fromiter(
        witness.nodes, dtype=np.int64, count=len(witness.nodes))
    base_t = np.asarray([base.nodes[int(node)][0] for node in base_ids], np.int64)
    witness_t = np.asarray(
        [witness.nodes[int(node)][0] for node in witness_ids], np.int64)
    base_pos = np.asarray(
        [base.nodes[int(node)][1:] for node in base_ids], np.float64)
    witness_pos = np.asarray(
        [witness.nodes[int(node)][1:] for node in witness_ids], np.float64)
    match = match_sequence(
        base_t, base_pos, witness_t, witness_pos, MAP_UM)
    mapped = np.full(int(witness_ids.max(initial=-1)) + 1, -1, np.int64)
    valid = match >= 0
    mapped[witness_ids[valid]] = base_ids[match[valid]]

    size = len(mapped)
    source = np.fromiter(
        (int(edge[0]) for edge in witness.edges),
        np.int64,
        count=len(witness.edges),
    )
    target = np.fromiter(
        (int(edge[1]) for edge in witness.edges),
        np.int64,
        count=len(witness.edges),
    )
    outdegree = np.bincount(source, minlength=size)
    indegree = np.bincount(target, minlength=size)
    result: set[tuple[int, int]] = set()
    for left, right in zip(source, target):
        if outdegree[left] != 1 or indegree[right] != 1:
            continue
        mapped_left, mapped_right = int(mapped[left]), int(mapped[right])
        if mapped_left < 0 or mapped_right < 0 or mapped_left == mapped_right:
            continue
        if base.nodes[mapped_right][0] != base.nodes[mapped_left][0] + 1:
            continue
        result.add((mapped_left, mapped_right))
    return result


def add_stage6_v24_consensus_edges(
    base: DatasetGraph,
    stage6: DatasetGraph,
    v24: DatasetGraph,
    *,
    output_base: DatasetGraph | None = None,
) -> tuple[DatasetGraph, dict[str, Any], list[dict[str, Any]]]:
    """Append mutually unique terminal-root links shared by Stage-6 and V24."""
    if stage6.dataset != base.dataset or v24.dataset != base.dataset:
        raise ValueError("base and witness dataset names differ")
    output_base = base if output_base is None else output_base
    if (
        set(output_base.nodes) != set(base.nodes)
        or output_base.edges != base.edges
        or output_base.dataset != base.dataset
    ):
        raise ValueError("output base changed the raw graph topology")

    size = max(base.nodes, default=-1) + 1
    source = np.fromiter(
        (int(edge[0]) for edge in base.edges), np.int64, count=len(base.edges))
    target = np.fromiter(
        (int(edge[1]) for edge in base.edges), np.int64, count=len(base.edges))
    outdegree = np.bincount(source, minlength=size)
    indegree = np.bincount(target, minlength=size)
    original = {(int(edge[0]), int(edge[1])) for edge in base.edges}

    stage6_edges = _mapped_ordinary_edges(base, stage6)
    v24_edges = _mapped_ordinary_edges(base, v24)
    agreed = stage6_edges & v24_edges
    qualifying = [
        (left, right) for left, right in agreed
        if (left, right) not in original
        and outdegree[left] == 0
        and indegree[right] == 0
        and base.nodes[right][0] == base.nodes[left][0] + 1
    ]
    by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in qualifying:
        by_source[edge[0]].append(edge)
        by_target[edge[1]].append(edge)
    selected = sorted(
        edge for edge in qualifying
        if len(by_source[edge[0]]) == 1 and len(by_target[edge[1]]) == 1
    )

    candidate = DatasetGraph(
        dataset=output_base.dataset,
        nodes=dict(output_base.nodes),
        edges=list(output_base.edges) + [
            (left, right, 1.0) for left, right in selected],
        shape=output_base.shape,
    )
    before, after = validate_graph(output_base), validate_graph(candidate)
    if candidate.nodes != output_base.nodes:
        raise RuntimeError("consensus pass changed incumbent nodes")
    if candidate.edges[:len(output_base.edges)] != output_base.edges:
        raise RuntimeError("consensus pass changed incumbent edges")
    if before["forks"] != after["forks"]:
        raise RuntimeError("consensus pass changed division topology")
    actions = [{
        "dataset": base.dataset,
        "source": left,
        "target": right,
        "source_t": int(base.nodes[left][0]),
    } for left, right in selected]
    audit = {
        "dataset": base.dataset,
        "stage6_mapped_edges": len(stage6_edges),
        "v24_mapped_edges": len(v24_edges),
        "raw_agreed_edges": len(agreed),
        "terminal_root_qualifying": len(qualifying),
        "selected_edges": len(selected),
        "ambiguous_rejected": len(qualifying) - len(selected),
        "original_nodes_preserved": True,
        "original_edges_preserved": True,
        "division_forks_preserved": True,
    }
    return candidate, audit, actions
