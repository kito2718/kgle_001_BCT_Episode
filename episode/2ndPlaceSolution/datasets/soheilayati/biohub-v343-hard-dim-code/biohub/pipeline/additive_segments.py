"""Add Phase-19 V24-only ordinary segments without changing V21 evidence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.metrics.matching import match_sequence


MAP_UM = 3.0
NOVEL_UM = 7.0
MIN_SUPPORT = 3
MAX_SEGMENT_NODES = 100


def _arrays(graph: DatasetGraph) -> dict[str, np.ndarray]:
    ids = np.fromiter(graph.nodes, dtype=np.int64, count=len(graph.nodes))
    size = int(ids.max(initial=-1)) + 1
    t = np.full(size, -1, np.int16)
    pos = np.full((size, 3), np.nan, np.float64)
    for node, coordinate in graph.nodes.items():
        t[int(node)] = int(coordinate[0])
        pos[int(node)] = coordinate[1:]
    source = np.fromiter(
        (int(edge[0]) for edge in graph.edges), np.int64, count=len(graph.edges),
    )
    target = np.fromiter(
        (int(edge[1]) for edge in graph.edges), np.int64, count=len(graph.edges),
    )
    indegree = np.bincount(target, minlength=size)
    outdegree = np.bincount(source, minlength=size)
    parent = np.full(size, -1, np.int64)
    child = np.full(size, -1, np.int64)
    simple = (indegree[target] == 1) & (outdegree[source] == 1)
    parent[target[simple]] = source[simple]
    child[source[simple]] = target[simple]
    history = np.ones(size, np.int16)
    future = np.ones(size, np.int16)
    for node in ids[np.argsort(t[ids], kind="stable")]:
        previous = int(parent[node])
        if previous >= 0:
            history[node] = min(32767, int(history[previous]) + 1)
    for node in ids[np.argsort(-t[ids], kind="stable")]:
        following = int(child[node])
        if following >= 0:
            future[node] = min(32767, int(future[following]) + 1)
    return {
        "ids": ids,
        "t": t,
        "pos": pos,
        "source": source,
        "target": target,
        "indegree": indegree,
        "outdegree": outdegree,
        "parent": parent,
        "child": child,
        "history": history,
        "future": future,
    }


def _nearest_distance_by_frame(
    base: DatasetGraph,
    witness: DatasetGraph,
    witness_ids: np.ndarray,
) -> np.ndarray:
    result = np.full(int(witness_ids.max(initial=-1)) + 1, np.inf, np.float32)
    base_by_time: dict[int, list[np.ndarray]] = {}
    for coordinate in base.nodes.values():
        base_by_time.setdefault(int(coordinate[0]), []).append(
            np.asarray(coordinate[1:], np.float64) * SCALE_ARR,
        )
    witness_by_time: dict[int, list[int]] = {}
    for node, coordinate in witness.nodes.items():
        witness_by_time.setdefault(int(coordinate[0]), []).append(int(node))
    for time, nodes in witness_by_time.items():
        reference = base_by_time.get(time)
        if not reference:
            continue
        positions = np.asarray(
            [witness.nodes[node][1:] for node in nodes], np.float64,
        ) * SCALE_ARR
        distance = cKDTree(np.asarray(reference)).query(positions, k=1)[0]
        result[np.asarray(nodes, np.int64)] = distance.astype(np.float32)
    return result


def _encode_path(witness: DatasetGraph, path: list[int]) -> dict[str, Any]:
    coordinates = np.asarray([witness.nodes[node] for node in path], np.int64)
    return {
        "path_v24_nodes": json.dumps(path),
        "path_t": json.dumps(coordinates[:, 0].astype(int).tolist()),
        "path_z": json.dumps(coordinates[:, 1].astype(int).tolist()),
        "path_y": json.dumps(coordinates[:, 2].astype(int).tolist()),
        "path_x": json.dumps(coordinates[:, 3].astype(int).tolist()),
    }


def full_length_segment_actions(
    base: DatasetGraph,
    witness: DatasetGraph,
) -> list[dict[str, Any]]:
    """Return the exact frozen Phase-19 additive action set."""
    if base.dataset != witness.dataset:
        raise ValueError("base and witness dataset names differ")
    a21, a24 = _arrays(base), _arrays(witness)
    base_ids, witness_ids = a21["ids"], a24["ids"]
    base_coordinates = np.asarray(
        [base.nodes[int(node)][1:] for node in base_ids], np.float64,
    )
    witness_coordinates = np.asarray(
        [witness.nodes[int(node)][1:] for node in witness_ids], np.float64,
    )
    match = match_sequence(
        a21["t"][base_ids], base_coordinates,
        a24["t"][witness_ids], witness_coordinates, MAP_UM,
    )
    mapped = np.full(len(a24["t"]), -1, np.int64)
    valid = match >= 0
    mapped[witness_ids[valid]] = base_ids[match[valid]]
    nearest = _nearest_distance_by_frame(base, witness, witness_ids)
    novel = (mapped < 0) & (nearest > NOVEL_UM) & (a24["t"] >= 0)

    records: list[dict[str, Any]] = []
    for anchor24 in witness_ids:
        anchor24 = int(anchor24)
        start21 = int(mapped[anchor24])
        if start21 < 0 or a24["outdegree"][anchor24] != 1:
            continue
        path: list[int] = []
        node = int(a24["child"][anchor24])
        while (
            len(path) < MAX_SEGMENT_NODES
            and node >= 0
            and novel[node]
            and a24["indegree"][node] == 1
            and a24["outdegree"][node] == 1
        ):
            path.append(node)
            node = int(a24["child"][node])
        if not path or a21["outdegree"][start21] != 0:
            continue
        end21 = int(mapped[node]) if node >= 0 else -1
        if end21 >= 0 and a21["indegree"][end21] == 0:
            kind = "bridge"
            support = min(
                int(a21["history"][start21]),
                int(a21["future"][end21]),
                int(a24["history"][path[0]]),
                int(a24["future"][path[-1]]),
            )
        elif (
            a21["history"][start21] >= MIN_SUPPORT
            and a24["future"][path[0]] >= MIN_SUPPORT
        ):
            kind = "tail"
            end21 = -1
            support = min(
                int(a21["history"][start21]),
                int(a24["future"][path[0]]),
            )
        else:
            continue
        records.append({
            "dataset": base.dataset,
            "kind": kind,
            "mapped_start": start21,
            "mapped_end": end21,
            "nodes": len(path),
            "support": support,
            **_encode_path(witness, path),
        })

    for anchor24 in witness_ids:
        anchor24 = int(anchor24)
        end21 = int(mapped[anchor24])
        if end21 < 0 or a24["indegree"][anchor24] != 1:
            continue
        backward: list[int] = []
        node = int(a24["parent"][anchor24])
        while (
            len(backward) < MAX_SEGMENT_NODES
            and node >= 0
            and novel[node]
            and a24["indegree"][node] == 1
            and a24["outdegree"][node] == 1
        ):
            backward.append(node)
            node = int(a24["parent"][node])
        if not backward or a21["indegree"][end21] != 0:
            continue
        if node >= 0 and mapped[node] >= 0:
            continue
        path = list(reversed(backward))
        if (
            a21["future"][end21] < MIN_SUPPORT
            or a24["history"][path[-1]] < MIN_SUPPORT
        ):
            continue
        support = min(
            int(a21["future"][end21]),
            int(a24["history"][path[-1]]),
        )
        records.append({
            "dataset": base.dataset,
            "kind": "head",
            "mapped_start": -1,
            "mapped_end": end21,
            "nodes": len(path),
            "support": support,
            **_encode_path(witness, path),
        })

    records.sort(key=lambda row: (
        {"bridge": 0, "tail": 1, "head": 2}[str(row["kind"])],
        -int(row["nodes"]),
        -int(row["support"]),
        str(row["path_v24_nodes"]),
    ))
    selected: list[dict[str, Any]] = []
    used_anchors: set[int] = set()
    used_nodes: set[int] = set()
    for row in records:
        anchors = {
            int(value) for value in (row["mapped_start"], row["mapped_end"])
            if int(value) >= 0
        }
        nodes = set(json.loads(str(row["path_v24_nodes"])))
        if anchors & used_anchors or nodes & used_nodes:
            continue
        selected.append(row)
        used_anchors.update(anchors)
        used_nodes.update(nodes)
    return selected


def add_full_length_segments(
    base: DatasetGraph,
    witness: DatasetGraph,
    *,
    output_base: DatasetGraph | None = None,
    output_witness: DatasetGraph | None = None,
) -> tuple[DatasetGraph, dict[str, Any]]:
    """Append Phase-19 segments while retaining all base evidence exactly.

    Action selection uses ``base`` and ``witness``. Optional output graphs let
    deployment retain the exact V21 coordinate post-pass while assigning added
    nodes their corresponding V24 post-pass coordinates.
    """
    output_base = base if output_base is None else output_base
    output_witness = witness if output_witness is None else output_witness
    if set(output_base.nodes) != set(base.nodes):
        raise ValueError("output base changed the V21 node identity set")
    if set(output_witness.nodes) != set(witness.nodes):
        raise ValueError("output witness changed the V24 node identity set")
    if output_base.edges != base.edges or output_witness.edges != witness.edges:
        raise ValueError("coordinate post-pass changed graph topology")

    actions = full_length_segment_actions(base, witness)
    nodes = dict(output_base.nodes)
    original_nodes = dict(nodes)
    edges = list(output_base.edges)
    original_edges = list(edges)
    next_id = max(nodes, default=-1) + 1
    added_by_kind = {"bridge": 0, "tail": 0, "head": 0}
    for row in actions:
        path = json.loads(str(row["path_v24_nodes"]))
        new_ids = list(range(next_id, next_id + len(path)))
        next_id += len(path)
        for new_id, witness_id in zip(new_ids, path):
            t, z, y, x = output_witness.nodes[int(witness_id)]
            if output_base.shape is not None:
                t = min(max(int(t), 0), output_base.shape[0] - 1)
                z = min(max(int(z), 0), output_base.shape[1] - 1)
                y = min(max(int(y), 0), output_base.shape[2] - 1)
                x = min(max(int(x), 0), output_base.shape[3] - 1)
            nodes[new_id] = (int(t), int(z), int(y), int(x))
        if int(row["mapped_start"]) >= 0:
            edges.append((int(row["mapped_start"]), new_ids[0], 1.0))
        edges.extend((left, right, 1.0) for left, right in zip(new_ids[:-1], new_ids[1:]))
        if int(row["mapped_end"]) >= 0:
            edges.append((new_ids[-1], int(row["mapped_end"]), 1.0))
        added_by_kind[str(row["kind"])] += 1

    candidate = DatasetGraph(
        dataset=output_base.dataset,
        nodes=nodes,
        edges=edges,
        shape=output_base.shape,
    )
    base_summary = validate_graph(output_base)
    candidate_summary = validate_graph(candidate)
    if any(candidate.nodes[node] != coordinate for node, coordinate in original_nodes.items()):
        raise RuntimeError("Phase 19 changed an existing V21 coordinate")
    if candidate.edges[:len(original_edges)] != original_edges:
        raise RuntimeError("Phase 19 changed an existing V21 edge")
    if candidate_summary["forks"] != base_summary["forks"]:
        raise RuntimeError("Phase 19 changed V21 division topology")
    audit = {
        "dataset": base.dataset,
        "segments": len(actions),
        "bridges": added_by_kind["bridge"],
        "tails": added_by_kind["tail"],
        "heads": added_by_kind["head"],
        "added_nodes": len(nodes) - len(original_nodes),
        "added_edges": len(edges) - len(original_edges),
        "original_nodes_preserved": True,
        "original_edges_preserved": True,
        "division_forks_preserved": True,
    }
    return candidate, audit
