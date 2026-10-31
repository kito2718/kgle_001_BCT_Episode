"""Inference-only V507 gated-localization and fork policies.

V507 is V506 without the second terminal-tail pass: the actions prepared
on the V454 graph are applied to that same graph.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback is exercised only on stripped images
    njit = None

from biohub.io.submission import DatasetGraph, validate_graph
from biohub.metrics.gt import SCALE_ARR


DETECTOR_SCALARS = [
    "score", "rank_in_frame", "flow_sigma_um", "near_um", "density_10um",
    "peak_width_um", "boundary_um",
]


def _load_numeric_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_assets(root: str | Path) -> dict:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    models = {
        "localization": [_load_numeric_model(root / f"localization_axis{axis}.npz") for axis in range(3)],
        "gate": _load_numeric_model(root / "localization_gate.npz"),
        "fork": _load_numeric_model(root / "fork_pruner.npz"),
    }
    return {"metadata": metadata, "models": models}


def _transform(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(values, np.float64)
    missing = np.isnan(values)
    result = np.where(missing, model["imputer_statistics"][None, :], values)
    indicator = model["indicator_features"]
    if len(indicator):
        result = np.column_stack((result, missing[:, indicator].astype(np.float64)))
    if len(model["scaler_mean"]):
        result = (result - model["scaler_mean"]) / model["scaler_scale"]
    return result


def _tree_sum_numpy(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Portable fallback for sklearn HGB's numeric trees."""
    result = np.full(len(values), float(model["baseline"][0]), np.float64)
    offsets = model["tree_offsets"]
    for tree in range(len(offsets) - 1):
        start = int(offsets[tree])
        current = np.full(len(values), start, np.int64)
        active = np.ones(len(values), bool)
        while np.any(active):
            rows = np.flatnonzero(active)
            node = current[rows]
            leaf = model["is_leaf"][node].astype(bool)
            if np.any(leaf):
                leaf_rows = rows[leaf]
                result[leaf_rows] += model["value"][node[leaf]]
                active[leaf_rows] = False
            branch_rows = rows[~leaf]
            if len(branch_rows):
                branch_node = node[~leaf]
                feature = model["feature"][branch_node]
                x = values[branch_rows, feature]
                go_left = np.where(
                    np.isnan(x), model["missing_left"][branch_node].astype(bool),
                    x <= model["threshold"][branch_node],
                )
                local_next = np.where(
                    go_left, model["left"][branch_node], model["right"][branch_node],
                )
                current[branch_rows] = start + local_next
    return result


if njit is not None:
    @njit
    def _tree_sum_compiled(
        values, offsets, node_value, feature, threshold, missing_left,
        left, right, is_leaf, baseline,
    ):
        result = np.empty(values.shape[0], np.float64)
        for row in range(values.shape[0]):
            total = baseline
            for tree in range(len(offsets) - 1):
                start = offsets[tree]
                node = start
                while is_leaf[node] == 0:
                    x = values[row, feature[node]]
                    if (np.isnan(x) and missing_left[node] != 0) or (
                        not np.isnan(x) and x <= threshold[node]
                    ):
                        node = start + left[node]
                    else:
                        node = start + right[node]
                total += node_value[node]
            result[row] = total
        return result


def _tree_sum(values: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate sealed sklearn HGB trees, using a compiled numeric hot loop."""
    if njit is None:
        return _tree_sum_numpy(values, model)
    return _tree_sum_compiled(
        values, model["tree_offsets"], model["value"], model["feature"],
        model["threshold"], model["missing_left"], model["left"], model["right"],
        model["is_leaf"], float(model["baseline"][0]),
    )


def _predict(frame: pd.DataFrame, columns: list[str], model: dict[str, np.ndarray]) -> np.ndarray:
    return _tree_sum(_transform(frame[columns].to_numpy(float), model), model)


def _predict_probability(frame: pd.DataFrame, columns: list[str], model: dict[str, np.ndarray]) -> np.ndarray:
    raw = _predict(frame, columns, model)
    return 1.0 / (1.0 + np.exp(-raw))


def _tables(graph: DatasetGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame([
        {"node_id": int(node_id), "t": int(point[0]), "z": int(point[1]),
         "y": int(point[2]), "x": int(point[3])}
        for node_id, point in graph.nodes.items()
    ]).sort_values("node_id").reset_index(drop=True)
    edges = pd.DataFrame([
        {"source": int(source), "target": int(target), "conf": float(confidence)}
        for source, target, confidence in graph.edges
    ], columns=["source", "target", "conf"])
    return nodes, edges


def _localization_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, cluster: str,
) -> pd.DataFrame:
    xyz = nodes[["z", "y", "x"]].to_numpy(float)
    times = nodes.t.to_numpy(np.int64)
    ct = candidates.t.to_numpy(np.int64)
    cp = candidates[["z", "y", "x"]].to_numpy(float)
    nearest = np.full(len(nodes), -1, np.int64)
    distance = np.full(len(nodes), np.inf)
    for t in np.unique(times):
        ni = np.flatnonzero(times == t)
        ci = np.flatnonzero(ct == t)
        if len(ci):
            d, local = cKDTree(cp[ci] * SCALE_ARR).query(xyz[ni] * SCALE_ARR, k=1)
            nearest[ni] = ci[np.asarray(local, np.int64)]
            distance[ni] = d
    valid = nearest >= 0
    detector = np.full_like(xyz, np.nan)
    detector[valid] = cp[nearest[valid]]
    detector = detector.astype(np.float32).astype(float) * SCALE_ARR
    row = {int(node): i for i, node in enumerate(nodes.node_id)}
    pi = np.asarray([row[int(node)] for node in edges.source], np.int64)
    ci = np.asarray([row[int(node)] for node in edges.target], np.int64)
    incoming = np.bincount(ci, minlength=len(nodes))
    outgoing = np.bincount(pi, minlength=len(nodes))
    previous = np.full(len(nodes), -1, int)
    following = np.full(len(nodes), -1, int)
    previous[ci[incoming[ci] == 1]] = pi[incoming[ci] == 1]
    following[pi[outgoing[pi] == 1]] = ci[outgoing[pi] == 1]
    pos = xyz * SCALE_ARR
    before = np.full_like(pos, np.nan)
    after = np.full_like(pos, np.nan)
    valid_prev = previous >= 0
    valid_next = following >= 0
    before[valid_prev] = pos[valid_prev] - pos[previous[valid_prev]]
    after[valid_next] = pos[following[valid_next]] - pos[valid_next]
    frame = nodes[["node_id", "t", "z", "y", "x"]].copy()
    frame["time_fraction"] = times / max(1, int(times.max()))
    frame["acquisition_01"] = int(cluster == "acq_01")
    for axis in range(3):
        frame[f"position_{axis}"] = pos[:, axis]
        frame[f"detector_offset_{axis}"] = detector[:, axis] - pos[:, axis]
        frame[f"incoming_velocity_{axis}"] = before[:, axis]
        frame[f"outgoing_velocity_{axis}"] = after[:, axis]
        frame[f"midpoint_residual_{axis}"] = (after[:, axis] - before[:, axis]) / 2
    frame["detector_distance_um"] = distance
    for column in DETECTOR_SCALARS:
        values = np.full(len(nodes), np.nan, np.float32)
        values[valid] = candidates[column].to_numpy(np.float32)[nearest[valid]]
        frame[f"detector_{column}"] = values
    frame["indegree"] = incoming
    frame["outdegree"] = outgoing
    return frame.replace([np.inf, -np.inf], np.nan)


def _localization_actions(
    nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, cluster: str,
    assets: dict,
) -> tuple[dict[int, tuple[int, int, int]], dict]:
    metadata = assets["metadata"]
    frame = _localization_features(nodes, edges, candidates, cluster)
    columns = metadata["localization_columns"]
    raw = np.column_stack([
        _predict(frame, columns, model) for model in assets["models"]["localization"]
    ])
    stacked = frame[columns].copy()
    for axis in range(3):
        stacked[f"base_residual_{axis}"] = raw[:, axis]
    stacked["base_residual_norm"] = np.linalg.norm(raw, axis=1)
    stacked["base_alignment_detector"] = np.sum(
        raw * frame[[f"detector_offset_{axis}" for axis in range(3)]].to_numpy(float), axis=1,
    ) / np.maximum(np.linalg.norm(raw, axis=1), 1e-6)
    gate = _predict(
        stacked, metadata["localization_gate_columns"], assets["models"]["gate"],
    )
    residual = raw * float(metadata["localization_strength"])
    cap = float(metadata["localization_norm_cap_um"])
    residual *= np.minimum(1.0, cap / np.maximum(np.linalg.norm(residual, axis=1), 1e-12))[:, None]
    proposed = np.rint(nodes[["z", "y", "x"]].to_numpy(float) + residual / SCALE_ARR)
    proposed = np.clip(proposed, [0, 0, 0], [63, 255, 255]).astype(np.int64)
    selected = gate >= float(metadata["localization_gate_threshold_um"])
    changed = selected & np.any(proposed != nodes[["z", "y", "x"]].to_numpy(np.int64), axis=1)
    actions = {
        int(node): tuple(map(int, point))
        for node, point in zip(nodes.loc[selected, "node_id"], proposed[selected], strict=True)
    }
    audit = {
        "gate_threshold_um": float(metadata["localization_gate_threshold_um"]),
        "selected": int(selected.sum()), "moved": int(changed.sum()),
        "gate_min": float(gate.min()), "gate_max": float(gate.max()),
    }
    return actions, audit


def _detector_features(nodes: pd.DataFrame, candidates: pd.DataFrame) -> dict[int, dict[str, float]]:
    result = {}
    by_t = {int(t): part for t, part in candidates.groupby("t", sort=False)}
    for t, part in nodes.groupby("t", sort=False):
        candidate = by_t.get(int(t))
        if candidate is None or candidate.empty:
            for node in part.node_id:
                result[int(node)] = {"detector_distance_um": np.nan, **{
                    f"detector_{key}": np.nan for key in DETECTOR_SCALARS}}
            continue
        cp = candidate[["z", "y", "x"]].to_numpy(float)
        distance, index = cKDTree(cp * SCALE_ARR).query(
            part[["z", "y", "x"]].to_numpy(float) * SCALE_ARR, k=1,
        )
        for node, d, j in zip(part.node_id, distance, np.asarray(index, int), strict=True):
            row = {"detector_distance_um": float(d)}
            for key in DETECTOR_SCALARS:
                row[f"detector_{key}"] = float(candidate.iloc[j][key])
            result[int(node)] = row
    return result


def _exact_motion(
    nodes: pd.DataFrame, ids: np.ndarray, sampled_nodes: np.ndarray,
    backward: np.ndarray, forward: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lookup = {int(row.node_id): (int(row.t), int(row.z), int(row.y), int(row.x))
              for row in nodes.itertuples(index=False)}
    result = {}
    for node, point, back, ahead in zip(ids, sampled_nodes, backward, forward, strict=True):
        node = int(node)
        if node in lookup and lookup[node] == tuple(map(int, point)):
            result[node] = (np.asarray(back, float), np.asarray(ahead, float))
    return result


def _chain_depth(seed: int, direction: dict[int, list[int]], limit: int = 6) -> int:
    depth, current = 0, seed
    while depth < limit:
        choices = direction.get(current, [])
        if len(choices) != 1:
            break
        current = choices[0]
        depth += 1
    return depth


def _pair_summary(row: dict, prefix: str, a: float, b: float) -> None:
    row[f"{prefix}_min"] = float(min(a, b))
    row[f"{prefix}_max"] = float(max(a, b))
    row[f"{prefix}_absdiff"] = float(abs(a - b))


def _fork_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, cluster: str,
    ids: np.ndarray, sampled_nodes: np.ndarray, backward: np.ndarray, forward: np.ndarray,
) -> pd.DataFrame:
    node_ids = nodes.node_id.to_numpy(np.int64)
    node_xyz = nodes[["z", "y", "x"]].to_numpy(float) * SCALE_ARR
    positions = {int(node_id): node_xyz[index] for index, node_id in enumerate(node_ids)}
    times = {int(node_id): int(t) for node_id, t in zip(node_ids, nodes.t.to_numpy(np.int64), strict=True)}
    previous, following = defaultdict(list), defaultdict(list)
    for edge in edges.itertuples(index=False):
        following[int(edge.source)].append(int(edge.target))
        previous[int(edge.target)].append(int(edge.source))
    forks = {parent: sorted(children) for parent, children in following.items() if len(children) == 2}
    if not forks:
        return pd.DataFrame()
    # Fork inference only consumes parents and their two children.  Restricting
    # nearest-detector lookup to those nodes avoids a full-graph pandas loop.
    needed = {node for parent, children in forks.items() for node in [parent, *children]}
    fork_nodes = nodes[nodes.node_id.isin(needed)].copy()
    detector = _detector_features(fork_nodes, candidates)
    motion = _exact_motion(fork_nodes, ids, sampled_nodes, backward, forward)
    records = []
    for parent, children in forks.items():
        child0, child1 = children
        p = positions[parent]
        c = [positions[child0], positions[child1]]
        incoming = previous.get(parent, [])
        incoming_velocity = p - positions[incoming[0]] if len(incoming) == 1 else np.full(3, np.nan)
        arms = []
        for child in children:
            displacement = positions[child] - p
            successor = following.get(child, [])
            outgoing_velocity = positions[successor[0]] - positions[child] if len(successor) == 1 else np.full(3, np.nan)
            back, _ = motion.get(child, (np.full(3, np.nan), np.full(3, np.nan)))
            _, parent_forward = motion.get(parent, (np.full(3, np.nan), np.full(3, np.nan)))
            arms.append({
                "distance_um": float(np.linalg.norm(displacement)),
                "incoming_residual_um": float(np.linalg.norm(displacement - incoming_velocity)),
                "outgoing_residual_um": float(np.linalg.norm(outgoing_velocity - displacement)),
                "parent_forward_residual_um": float(np.linalg.norm(p + parent_forward - positions[child])),
                "child_backward_residual_um": float(np.linalg.norm(positions[child] + back - p)),
                "forward_depth": _chain_depth(child, following),
                **detector[child],
            })
        boundary = np.minimum(p, np.asarray([63, 255, 255], float) * SCALE_ARR - p)
        record = {
            "parent": parent, "child0": child0, "child1": child1,
            "time_fraction": float(times[parent] / max(1, nodes.t.max())),
            "position_z_um": float(p[0]), "position_y_um": float(p[1]),
            "position_x_um": float(p[2]), "position_boundary_um": float(boundary.min()),
            "backward_depth": _chain_depth(parent, previous),
            "child_separation_um": float(np.linalg.norm(c[0] - c[1])),
            "acquisition_01": int(cluster == "acq_01"),
        }
        for key, value in detector[parent].items():
            record[f"parent_{key}"] = value
        for key in [
            "distance_um", "incoming_residual_um", "outgoing_residual_um",
            "parent_forward_residual_um", "child_backward_residual_um", "forward_depth",
            "detector_distance_um", *[f"detector_{x}" for x in DETECTOR_SCALARS],
        ]:
            _pair_summary(record, key, arms[0][key], arms[1][key])
        record["arm0_parent_forward_residual_um"] = arms[0]["parent_forward_residual_um"]
        record["arm1_parent_forward_residual_um"] = arms[1]["parent_forward_residual_um"]
        records.append(record)
    return pd.DataFrame(records)


def _fork_action(
    nodes: pd.DataFrame, edges: pd.DataFrame, candidates: pd.DataFrame, cluster: str,
    ids: np.ndarray, sampled_nodes: np.ndarray, backward: np.ndarray, forward: np.ndarray,
    assets: dict,
) -> tuple[dict | None, dict]:
    frame = _fork_features(nodes, edges, candidates, cluster, ids, sampled_nodes, backward, forward)
    if frame.empty:
        return None, {"candidates": 0, "above_threshold": 0, "selected": 0}
    metadata = assets["metadata"]
    probability = _predict_probability(frame, metadata["fork_columns"], assets["models"]["fork"])
    threshold = float(metadata["fork_probability_threshold"])
    eligible = np.flatnonzero(probability >= threshold)
    if not len(eligible):
        return None, {"candidates": len(frame), "above_threshold": 0, "selected": 0,
                      "probability_max": float(probability.max())}
    order = sorted(eligible, key=lambda index: (-float(probability[index]), int(frame.iloc[index].parent)))
    index = int(order[0])
    row = frame.iloc[index]
    child = int(row.child0) if float(row.arm0_parent_forward_residual_um) > float(row.arm1_parent_forward_residual_um) else int(row.child1)
    action = {"parent": int(row.parent), "child": child, "probability": float(probability[index])}
    return action, {"candidates": len(frame), "above_threshold": int(len(eligible)),
                    "selected": 1, **action}


def prepare_actions(
    graph: DatasetGraph, candidates: pd.DataFrame, cluster: str,
    motion_ids: np.ndarray, motion_nodes: np.ndarray,
    motion_backward: np.ndarray, motion_forward: np.ndarray, assets: dict,
) -> tuple[dict, dict]:
    """Select all graph-dependent actions on the unmodified V454 graph."""
    nodes, edges = _tables(graph)
    localization, localization_audit = _localization_actions(nodes, edges, candidates, cluster, assets)
    fork, fork_audit = _fork_action(
        nodes, edges, candidates, cluster, motion_ids, motion_nodes,
        motion_backward, motion_forward, assets,
    )
    prepared = {"localization": localization, "fork": fork}
    return prepared, {"localization": localization_audit, "fork": fork_audit}


def apply_prepared(graph: DatasetGraph, prepared: dict) -> tuple[DatasetGraph, dict]:
    """Apply locked fork/localization actions to the prepared graph."""
    nodes = dict(graph.nodes)
    moved, missing_localization = 0, 0
    for node, point in prepared["localization"].items():
        if node not in nodes:
            missing_localization += 1
            continue
        old = nodes[node]
        new = (int(old[0]), int(point[0]), int(point[1]), int(point[2]))
        moved += int(new != old)
        nodes[node] = new
    edges = list(graph.edges)
    fork_applied = 0
    fork = prepared.get("fork")
    if fork is not None:
        matches = [i for i, (source, target, _) in enumerate(edges)
                   if int(source) == int(fork["parent"]) and int(target) == int(fork["child"])]
        if len(matches) == 1:
            del edges[matches[0]]
            fork_applied = 1
    result = DatasetGraph(graph.dataset, nodes, edges, graph.shape)
    valid = validate_graph(result)
    audit = {
        "localization_actions": len(prepared["localization"]),
        "localization_moved": moved,
        "localization_nodes_removed_by_tail": missing_localization,
        "fork_selected": int(fork is not None), "fork_applied": fork_applied,
        "fork_skipped": int(fork is not None and not fork_applied),
        "nodes": len(nodes), "edges": len(edges), "components": len(nodes) - len(edges),
        "forks": int(valid["forks"]),
    }
    return result, audit
