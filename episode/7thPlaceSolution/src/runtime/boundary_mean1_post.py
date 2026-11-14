"""Prediction-only boundary1 mean-ratio restore after selective size8 postprocess."""
from __future__ import annotations

import copy
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.spatial import cKDTree

SCALE = np.asarray([1.625, .40625, .40625])
MIN_TRACK_LEN = 9
MIN_TRACK_LEN_BY_DATASET: dict[str, int] = {}


def base_short_track_filter(
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    dataset: str,
    stats: defaultdict[str, int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    minimum = MIN_TRACK_LEN_BY_DATASET.get(dataset, MIN_TRACK_LEN)
    stats["short_track_min_len_effective"] = minimum
    parent = {node_id: node_id for node_id in nodes}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    out_count: defaultdict[int, int] = defaultdict(int)
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] += 1
    components: dict[int, list[int]] = defaultdict(list)
    for node_id in nodes:
        components[find(node_id)].append(node_id)
    keep: set[int] = set()
    for members in components.values():
        has_division = any(out_count[node_id] >= 2 for node_id in members)
        if len(members) >= minimum or has_division:
            keep.update(members)
    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes, edges
    kept_nodes = {node_id: node for node_id, node in nodes.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(not (set(members) & keep) for members in components.values())
    stats["short_track_nodes_removed"] = len(nodes) - len(kept_nodes)
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def edge_ids(edges: list[dict]) -> set[tuple[int, int]]:
    return {(int(edge['source_id']), int(edge['target_id'])) for edge in edges}


def groups_of_extra(extra: set[int], edge_pairs: set[tuple[int, int]]) -> list[list[int]]:
    adjacency = {node_id: set() for node_id in extra}
    for source_id, target_id in edge_pairs:
        if source_id in extra or target_id in extra:
            assert source_id in extra and target_id in extra, 'extra component attaches to control'
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
    groups = []
    unseen = set(extra)
    while unseen:
        stack = [min(unseen)]
        group = set()
        while stack:
            node_id = stack.pop()
            if node_id in group:
                continue
            group.add(node_id)
            stack.extend(adjacency[node_id] - group)
        unseen -= group
        groups.append(sorted(group))
    return groups


def apply(raw: dict, raw_edges: list[dict], final: dict, final_edges: list[dict],
          stem: str, data: Path, candidates: Path, b, stats: dict) -> tuple[dict, list[dict]]:
    """Return size8 graph plus boundary components satisfying both relative-score gates."""
    rows = pickle.loads((candidates / f'{stem}.pkl').read_bytes())['node_rows']
    assert all(
        node_id < len(rows) and int(node['t']) == int(rows[node_id]['t'])
        and all(abs(float(node[axis]) - float(rows[node_id][axis])) < 1e-5 for axis in ('z', 'y', 'x'))
        for node_id, node in raw.items()
    ), stem
    current_filter, current_min = b.short_track_filter, b.MIN_TRACK_LEN
    assert current_min == 9 and b.MIN_TRACK_LEN_BY_DATASET == {}, stem
    try:
        b.short_track_filter = base_short_track_filter
        control, control_edges, _ = b.apply_arm('gap_short_track_linefit', raw, raw_edges, stem, data)
        b.short_track_filter = current_filter
        b.MIN_TRACK_LEN = 3
        short, short_edges, _ = b.apply_arm('gap_short_track_linefit', raw, raw_edges, stem, data)
    finally:
        b.short_track_filter = current_filter
        b.MIN_TRACK_LEN = current_min
    assert set(control) <= set(final) <= set(short), stem
    control_pairs, final_pairs, short_pairs = map(edge_ids, (control_edges, final_edges, short_edges))
    assert control_pairs <= final_pairs <= short_pairs, stem
    extra = set(short) - set(control)
    groups = groups_of_extra(extra, short_pairs)
    bytime: dict[int, list[int]] = defaultdict(list)
    for node_id in control:
        if node_id in raw:
            bytime[int(rows[node_id]['t'])].append(node_id)
    trees = {
        t: cKDTree(np.asarray([[rows[k][axis] for axis in ('z', 'y', 'x')] for k in ids]) * SCALE)
        for t, ids in bytime.items()
    }
    frames = int(zarr.open_group(str(data / f'{stem}.zarr'), mode='r')['0'].shape[0])
    selected = []
    for ids in groups:
        times = [int(short[k]['t']) for k in ids]
        assert 3 <= len(ids) <= 8 and len(times) == len(set(times)), (stem, len(ids))
        ratios = []
        missing = 0
        for node_id in ids:
            t = int(rows[node_id]['t']) if node_id in raw else -1
            if node_id not in raw or t not in bytime:
                missing += 1
                continue
            pool = bytime[t]
            loc = np.asarray([rows[node_id][axis] for axis in ('z', 'y', 'x')]) * SCALE
            near = trees[t].query_ball_point(loc, 7.)
            score = (max(rows[pool[j]]['node_score'] for j in near) if near
                     else float(np.median([rows[j]['node_score'] for j in pool])))
            ratios.append(float(rows[node_id]['node_score']) / max(float(score), 1e-12))
        boundary = min(times) == 0 or max(times) == frames - 1
        if boundary and missing == 0 and sum(q >= 1.0 for q in ratios) > len(ids) / 2 and float(np.mean(ratios)) >= 1.0:
            selected.append(ids)
    selected_ids = {k for ids in selected for k in ids}
    trial_nodes = copy.deepcopy(final)
    trial_nodes.update({k: copy.deepcopy(short[k]) for k in selected_ids if k not in trial_nodes})
    pairs = final_pairs | {(u, v) for u, v in short_pairs if u in selected_ids and v in selected_ids}
    assert max(abs(float(trial_nodes[k][axis]) - float(final[k][axis]))
               for k in final for axis in ('z', 'y', 'x')) < 1e-6, stem
    trial_edges = [dict(source_id=u, target_id=v) for u, v in sorted(pairs)]
    b.validate_and_clip_graph(trial_nodes, trial_edges, stem, data, defaultdict(int))
    stats['boundary_mean1_components_kept'] = len(selected)
    stats['boundary_mean1_added_nodes_vs_size8'] = len(trial_nodes) - len(final)
    stats['boundary_mean1_added_edges_vs_size8'] = len(pairs) - len(final_pairs)
    return trial_nodes, trial_edges
