"""Deployment-only V398 terminal-tail cleanup validated by strict CV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR
from biohub.metrics.matching import match_sequence


OFF = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)


def load_assets(root: str | Path):
    from lightgbm import Booster

    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    model = Booster(model_file=str(root / "tail_protector.txt"))
    return model, metadata


def _patches(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    out = np.empty((len(coords), len(OFF)), np.float32)
    for frame in np.unique(coords[:, 0]):
        take = np.flatnonzero(coords[:, 0] == frame)
        centre = coords[take, 1:].astype(float)
        centre[:, 1:] /= 4
        grid = centre[:, None] + OFF[None]
        out[take] = map_coordinates(
            np.asarray(volume[int(frame)], np.float32),
            grid.reshape(-1, 3).T,
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(len(take), -1)
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return np.sum(a * b, 1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8
    )


def _simple_paths(
    pi: np.ndarray, ci: np.ndarray, n_nodes: int, minimum_nodes: int
) -> list[list[int]]:
    uf = np.arange(n_nodes)

    def find(value: int) -> int:
        while uf[value] != value:
            uf[value] = uf[uf[value]]
            value = int(uf[value])
        return value

    for a, b in zip(pi, ci):
        a, b = find(int(a)), find(int(b))
        if a != b:
            uf[b] = a
    roots = np.asarray([find(i) for i in range(n_nodes)])
    outdegree = np.bincount(pi, minlength=n_nodes)
    indegree = np.bincount(ci, minlength=n_nodes)
    successor = np.full(n_nodes, -1, np.int64)
    successor[pi[outdegree[pi] == 1]] = ci[outdegree[pi] == 1]
    # Group once. Repeated ``flatnonzero(roots == root)`` is quadratic in the
    # number of components and dominated the deployment CPU time.
    root_order = np.argsort(roots, kind="stable")
    sorted_roots = roots[root_order]
    starts = np.flatnonzero(
        np.r_[True, sorted_roots[1:] != sorted_roots[:-1]]
    )
    components = np.split(root_order, starts[1:])
    result = []
    for component in components:
        if len(component) < minimum_nodes:
            continue
        if np.any(outdegree[component] > 1) or np.any(indegree[component] > 1):
            continue
        starts = component[indegree[component] == 0]
        ends = component[outdegree[component] == 0]
        if len(starts) != 1 or len(ends) != 1:
            continue
        order = [int(starts[0])]
        while order[-1] != int(ends[0]):
            nxt = int(successor[order[-1]])
            if nxt < 0 or nxt in order:
                order = []
                break
            order.append(nxt)
        if len(order) == len(component):
            result.append(order)
    return result


def trim_terminal_tails(
    graph: DatasetGraph,
    volume: np.ndarray,
    witnesses: dict[str, DatasetGraph],
    model,
    metadata: dict,
    cluster: str,
) -> tuple[DatasetGraph, dict]:
    """Remove only a contiguous low-protection prefix from eligible path ends."""

    ids = np.asarray(list(graph.nodes), np.int64)
    coords = np.asarray(list(graph.nodes.values()), np.int64)
    lookup = {int(node): row for row, node in enumerate(ids)}
    pi = np.asarray([lookup[int(a)] for a, _, _ in graph.edges], np.int64)
    ci = np.asarray([lookup[int(b)] for _, b, _ in graph.edges], np.int64)
    pair_to_edge = {
        (int(a), int(b)): edge for edge, (a, b) in enumerate(zip(pi, ci))
    }
    paths = _simple_paths(
        pi, ci, len(ids), int(metadata["minimum_component_nodes"])
    )

    actions = []
    maximum_depth = int(metadata["maximum_tail_depth"])
    for component, order in enumerate(paths):
        allowed = min(maximum_depth, (len(order) - 2) // 2)
        for depth in range(1, allowed + 1):
            actions.append(
                (component, 0, depth, order[depth - 1], order[depth],
                 pair_to_edge[(order[depth - 1], order[depth])], order)
            )
            actions.append(
                (component, 1, depth, order[-depth], order[-depth - 1],
                 pair_to_edge[(order[-depth - 1], order[-depth])], order)
            )

    if not actions:
        audit = validate_graph(graph)
        audit.update(
            acquisition_cluster=cluster,
            threshold=float(metadata["protect_thresholds"][cluster]),
            candidate_components=0,
            candidate_actions=0,
            removed_actions=0,
            removed_nodes=0,
            removed_edges=0,
            new_forward_passes=0,
        )
        return graph, audit

    action_rows = np.asarray([row for _, _, _, row, _, _, _ in actions], np.int64)
    neighbour_rows = np.asarray([row for _, _, _, _, row, _, _ in actions], np.int64)
    needed_rows = np.unique(np.r_[action_rows, neighbour_rows])
    patch = np.full((len(ids), len(OFF)), np.nan, np.float32)
    patch[needed_rows] = _patches(volume, coords[needed_rows])
    endpoint_patches = patch[action_rows]
    neighbour_patches = patch[neighbour_rows]
    endpoint_patch_mean = endpoint_patches.mean(1)
    endpoint_patch_std = endpoint_patches.std(1)
    endpoint_patch_max = endpoint_patches.max(1)
    neighbour_patch_mean = neighbour_patches.mean(1)
    neighbour_patch_std = neighbour_patches.std(1)
    edge_ncc = _ncc(endpoint_patches, neighbour_patches)

    xyz = coords[:, 1:] * SCALE_ARR
    nearest = np.full(len(ids), np.nan)
    for frame in np.unique(coords[action_rows, 0]):
        all_at = np.flatnonzero(coords[:, 0] == frame)
        endpoints = action_rows[coords[action_rows, 0] == frame]
        if len(all_at) > 1:
            nearest[endpoints] = cKDTree(xyz[all_at]).query(
                xyz[endpoints], k=2
            )[0][:, 1]

    edge_um = np.linalg.norm(xyz[ci] - xyz[pi], axis=1)
    support = np.zeros(len(graph.edges), np.int8)
    witness_features = {
        "edge_witnesses", "path_witnesses_mean", "path_witnesses_std",
        "path_witnesses_min", "path_witnesses_max",
    }
    needs_witnesses = bool(
        witness_features.intersection(metadata["feature_columns"])
    )
    if needs_witnesses:
        base_pairs = {
            (int(ids[a]), int(ids[b])): edge
            for edge, (a, b) in enumerate(zip(pi, ci))
        }
        for witness in witnesses.values():
            witness_ids = np.asarray(list(witness.nodes), np.int64)
            witness_coords = np.asarray(list(witness.nodes.values()), np.float64)
            mapping = match_sequence(
                coords[:, 0], coords[:, 1:],
                witness_coords[:, 0].astype(int), witness_coords[:, 1:],
                max_distance=3.0,
            )
            donor_to_base = {
                int(witness_ids[i]): int(ids[j])
                for i, j in enumerate(mapping) if j >= 0
            }
            seen = set()
            for source, target, _ in witness.edges:
                if source in donor_to_base and target in donor_to_base:
                    edge = base_pairs.get(
                        (donor_to_base[source], donor_to_base[target])
                    )
                    if edge is not None:
                        seen.add(edge)
            if seen:
                support[list(seen)] += 1

    max_t = max(int(coords[:, 0].max()), 1)
    maxima = np.maximum(coords[:, 1:].max(0), 1)
    first_a = np.full((len(actions), 3), np.nan, np.float64)
    first_b = np.full((len(actions), 3), np.nan, np.float64)
    for index, (_, side, depth, _, _, _, order) in enumerate(actions):
        if side == 0 and depth + 1 < len(order):
            first_a[index] = xyz[order[depth]] - xyz[order[depth - 1]]
            first_b[index] = xyz[order[depth + 1]] - xyz[order[depth]]
        elif side == 1 and depth + 1 < len(order):
            first_a[index] = xyz[order[-depth - 1]] - xyz[order[-depth]]
            first_b[index] = xyz[order[-depth - 2]] - xyz[order[-depth - 1]]
    first_turn = np.sum(first_a * first_b, axis=1) / np.maximum(
        np.linalg.norm(first_a, axis=1) * np.linalg.norm(first_b, axis=1),
        1e-8,
    )
    first_acceleration = np.linalg.norm(first_b - first_a, axis=1)
    path_cache = []
    for order in paths:
        path_edges = np.asarray(
            [pair_to_edge[(a, b)] for a, b in zip(order[:-1], order[1:])],
            np.int64,
        )
        path_coords = coords[order]
        path_cache.append({
            "path_edges": path_edges,
            "component_nodes": len(order),
            "component_span": int(
                path_coords[:, 0].max() - path_coords[:, 0].min() + 1
            ),
            "component_start_fraction": float(path_coords[:, 0].min() / max_t),
            "component_end_fraction": float(path_coords[:, 0].max() / max_t),
            "component_z_fraction": float(path_coords[:, 1].mean() / maxima[0]),
            "component_y_fraction": float(path_coords[:, 2].mean() / maxima[1]),
            "component_x_fraction": float(path_coords[:, 3].mean() / maxima[2]),
            "path_edge_um_mean": float(edge_um[path_edges].mean()),
            "path_edge_um_std": float(edge_um[path_edges].std()),
            "path_edge_um_max": float(edge_um[path_edges].max()),
            "path_witnesses_mean": float(support[path_edges].mean()),
            "path_witnesses_std": float(support[path_edges].std()),
            "path_witnesses_min": int(support[path_edges].min()),
            "path_witnesses_max": int(support[path_edges].max()),
        })
    records = []
    for action_index, (component, side, depth, terminal, neighbour, edge, order) in enumerate(actions):
        cached = path_cache[component]
        records.append({
            "tail_depth": depth,
            "component_nodes": cached["component_nodes"],
            "component_span": cached["component_span"],
            "component_start_fraction": cached["component_start_fraction"],
            "component_end_fraction": cached["component_end_fraction"],
            "component_z_fraction": cached["component_z_fraction"],
            "component_y_fraction": cached["component_y_fraction"],
            "component_x_fraction": cached["component_x_fraction"],
            "side_end": float(side),
            "terminal_time_fraction": float(coords[terminal, 0] / max_t),
            "temporal_boundary_fraction": float(
                min(coords[terminal, 0], max_t - coords[terminal, 0]) / max_t
            ),
            "terminal_z_fraction": float(coords[terminal, 1] / maxima[0]),
            "terminal_y_fraction": float(coords[terminal, 2] / maxima[1]),
            "terminal_x_fraction": float(coords[terminal, 3] / maxima[2]),
            "terminal_nearest_um": float(nearest[terminal]),
            "terminal_patch_mean": float(endpoint_patch_mean[action_index]),
            "terminal_patch_std": float(endpoint_patch_std[action_index]),
            "terminal_patch_max": float(endpoint_patch_max[action_index]),
            "neighbour_patch_mean": float(neighbour_patch_mean[action_index]),
            "neighbour_patch_std": float(neighbour_patch_std[action_index]),
            "patch_mean_ratio": float(
                (endpoint_patch_mean[action_index] + 1e-4)
                / (neighbour_patch_mean[action_index] + 1e-4)
            ),
            "edge_um": float(edge_um[edge]),
            "edge_ncc": float(edge_ncc[action_index]),
            "edge_witnesses": int(support[edge]),
            "path_edge_um_mean": cached["path_edge_um_mean"],
            "path_edge_um_std": cached["path_edge_um_std"],
            "path_edge_um_max": cached["path_edge_um_max"],
            "path_witnesses_mean": cached["path_witnesses_mean"],
            "path_witnesses_std": cached["path_witnesses_std"],
            "path_witnesses_min": cached["path_witnesses_min"],
            "path_witnesses_max": cached["path_witnesses_max"],
            "first_turn_cosine": float(first_turn[action_index]),
            "first_acceleration_um": float(first_acceleration[action_index]),
            "is_acq01": float(cluster == "acq_01"),
        })

    x = np.asarray(
        [
            [row.get(name, np.nan) for name in metadata["feature_columns"]]
            for row in records
        ],
        np.float32,
    )
    probability = np.asarray(model.predict(x, num_threads=1), np.float64)
    threshold = float(metadata["protect_thresholds"][cluster])
    remove_rows = []
    action_groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for index, (component, side, depth, terminal, _, _, _) in enumerate(actions):
        action_groups.setdefault((component, side), []).append(
            (depth, index, terminal)
        )
    for take in action_groups.values():
        for _, index, terminal in take:
            if probability[index] < threshold:
                remove_rows.append(terminal)
            else:
                break

    remove_ids = set(ids[np.asarray(remove_rows, np.int64)].tolist())
    result = DatasetGraph(
        graph.dataset,
        {node: value for node, value in graph.nodes.items() if node not in remove_ids},
        [
            edge for edge in graph.edges
            if edge[0] not in remove_ids and edge[1] not in remove_ids
        ],
        graph.shape,
    )
    audit = validate_graph(result)
    audit.update(
        acquisition_cluster=cluster,
        threshold=threshold,
        candidate_components=len(paths),
        candidate_actions=len(actions),
        removed_actions=len(remove_rows),
        removed_nodes=len(graph.nodes) - len(result.nodes),
        removed_edges=len(graph.edges) - len(result.edges),
        probability_min=float(probability.min()),
        probability_max=float(probability.max()),
        witness_matching_used=needs_witnesses,
        new_forward_passes=0,
    )
    return result, audit
