#!/usr/bin/env python3
"""Apply fixed LB0.897-style graph repairs to existing Biohub GEFF outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


VOXEL_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)

MOTION_TIGHT_UM = 6.0
MOTION_RELAXED_UM = 10.0
MOTION_VELOCITY_WEIGHT = 0.5
MOTION_MAX_FRAME_NODES = 2600

GAP_CLOSE_UM = 6.0
GAP_REUSE_UM = 3.2
GAP_MAX_ADDED_FRAC = 0.05
GAP_MAX_ADDED_ABS = 2000
GAP_REFINE_WIN_Z = 1
GAP_REFINE_WIN_YX = 3
GAP_REFINE_MAX_SHIFT_UM = 3.2

MIN_TRACK_LEN = 7
MIN_TRACK_LEN_BY_DATASET = {"6bba_05b6850b": 6}
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
POSTPROCESS_MAX_NODES = 200_000
POSTPROCESS_MAX_EDGES = 200_000

SAFE_DIV_MAX_UM = 4.7
SAFE_DIV_SISTER_MAX_UM = 7.2
SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.8
SAFE_DIV_FRAME_FRAC_CAP = 0.008
SAFE_DIV_GLOBAL_FRAC_CAP = 0.004

ARMS = (
    "gap_close",
    "short_track",
    "linefit",
    "motion_relink",
    "safe_division",
    "full_stack",
    "short_track_linefit",
    "gap_short_track",
    "gap_short_track_linefit",
)


def read_graph(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(row["node_id"]): {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges: list[dict[str, Any]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        prob = row.get("edge_prob")
        edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": None if prob is None else float(prob),
                "distance_um": edge_distance_um(nodes[source_id], nodes[target_id]),
            }
        )
    return nodes, edges


def write_graph(
    path: Path,
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    graph = td.graph.IndexedRXGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, dtype=pl.Float64)
    ordered_ids = sorted(nodes)
    graph.bulk_add_nodes(
        [
            {
                "t": int(nodes[node_id]["t"]),
                "z": float(nodes[node_id]["z"]),
                "y": float(nodes[node_id]["y"]),
                "x": float(nodes[node_id]["x"]),
            }
            for node_id in ordered_ids
        ],
        indices=ordered_ids,
    )
    if edges:
        graph.bulk_add_edges(
            [
                {
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                }
                for edge in edges
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_geff(path, overwrite=True)


def edge_distance_um(source: dict[str, Any], target: dict[str, Any]) -> float:
    a = np.asarray((source["z"], source["y"], source["x"]), dtype=np.float64)
    b = np.asarray((target["z"], target["y"], target["x"]), dtype=np.float64)
    return float(np.linalg.norm((a - b) * VOXEL_SCALE_UM))


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * VOXEL_SCALE_UM))


def node_point(node: dict[str, Any]) -> tuple[float, float, float]:
    return float(node["z"]), float(node["y"]), float(node["x"])


def position_um(node: dict[str, Any]) -> np.ndarray:
    return np.asarray(node_point(node), dtype=np.float64) * VOXEL_SCALE_UM


def motion_relink(
    nodes: dict[int, dict[str, Any]], stats: defaultdict[str, int]
) -> list[dict[str, Any]]:
    ids_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        ids_by_t[int(node["t"])].append(node_id)
    for ids in ids_by_t.values():
        ids.sort()
    if ids_by_t and max(map(len, ids_by_t.values())) > MOTION_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    pos = {node_id: position_um(node) for node_id, node in nodes.items()}
    predecessor_pos: dict[int, np.ndarray] = {}
    selected: list[dict[str, Any]] = []

    def assign(source_ids: list[int], target_ids: list[int], gate: float):
        if not source_ids or not target_ids:
            return []
        big = gate * 1000.0 + 1.0
        source_pos = np.stack([pos[node_id] for node_id in source_ids])
        target_pos = np.stack([pos[node_id] for node_id in target_ids])
        predicted_pos = np.stack(
            [
                current
                if (previous := predecessor_pos.get(node_id)) is None
                else current + MOTION_VELOCITY_WEIGHT * (current - previous)
                for node_id, current in zip(source_ids, source_pos)
            ]
        )
        raw_distance = cdist(source_pos, target_pos)
        motion_distance = cdist(predicted_pos, target_pos)
        cost = np.where(raw_distance <= gate, motion_distance + 0.05 * raw_distance, big)
        rows, cols = linear_sum_assignment(cost)
        return [
            (
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_distance[r, c]),
                float(motion_distance[r, c]),
            )
            for r, c in zip(rows, cols)
            if cost[r, c] < big
        ]

    for t in sorted(ids_by_t):
        source_ids = ids_by_t[t]
        target_ids = ids_by_t.get(t + 1, [])
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        for pass_name, gate in (("tight", MOTION_TIGHT_UM), ("relaxed", MOTION_RELAXED_UM)):
            matches = assign(
                [x for x in source_ids if x in unmatched_sources],
                [x for x in target_ids if x in unmatched_targets],
                gate,
            )
            for source_id, target_id, raw, motion in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                selected.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_prob": None,
                        "distance_um": raw,
                        "motion_distance_um": motion,
                        "motion_pass": pass_name,
                    }
                )
                predecessor_pos[target_id] = pos[source_id]
                stats[f"motion_relink_{pass_name}_edges"] += 1
        stats["motion_relink_frames"] += 1
    stats["motion_relink_edges"] = len(selected)
    return selected


def refine_midpoint(
    data_dir: Path,
    dataset: str,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: defaultdict[str, int],
) -> tuple[float, float, float]:
    try:
        if t not in frame_cache:
            frame_cache[t] = np.asarray(
                zarr.open_group(str(data_dir / f"{dataset}.zarr"), mode="r")["0"][t]
            )
        frame = frame_cache[t]
        z, y, x = (int(round(value)) for value in midpoint)
        z0, z1 = max(0, z - GAP_REFINE_WIN_Z), min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0, y1 = max(0, y - GAP_REFINE_WIN_YX), min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0, x1 = max(0, x - GAP_REFINE_WIN_YX), min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if patch.size == 0 or total <= 0:
            raise ValueError("empty refinement weights")
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint


def gap_close(
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    dataset: str,
    data_dir: Path,
    stats: defaultdict[str, int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if not edges:
        return nodes, edges
    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming
    ends_by_t: dict[int, list[int]] = defaultdict(list)
    starts_by_t: dict[int, list[int]] = defaultdict(list)
    isolated_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t[t].append(node_id)
        if node_id not in incoming:
            starts_by_t[t].append(node_id)
        if node_id not in incident:
            isolated_by_t[t].append(node_id)
    max_synthetic = min(GAP_MAX_ADDED_ABS, max(1, round(len(nodes) * GAP_MAX_ADDED_FRAC)))
    next_id = max(nodes, default=0) + 1
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, Any]] = []
    synthetic_added = 0
    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = [node_id for node_id in starts_by_t.get(t + 2, []) if node_id not in used_starts]
        if not end_ids or not start_ids:
            continue
        distances = np.asarray(
            [[point_distance_um(node_point(nodes[a]), node_point(nodes[b])) for b in start_ids] for a in end_ids]
        )
        threshold = GAP_CLOSE_UM * 2.0
        stats["gap_candidates"] += int((distances <= threshold).sum())
        big = threshold * 1000.0 + 1.0
        rows, cols = linear_sum_assignment(np.where(distances <= threshold, distances, big))
        for r, c in zip(rows, cols):
            if distances[r, c] > threshold:
                continue
            source_id, target_id = end_ids[int(r)], start_ids[int(c)]
            if source_id in outgoing or target_id in used_starts:
                continue
            source, target = nodes[source_id], nodes[target_id]
            mid_t = int(source["t"]) + 1
            midpoint = tuple((a + b) / 2.0 for a, b in zip(node_point(source), node_point(target)))
            middle_id: int | None = None
            candidates = [x for x in isolated_by_t.get(mid_t, []) if x not in used_isolated]
            if candidates:
                candidate_distances = [point_distance_um(node_point(nodes[x]), midpoint) for x in candidates]
                best = int(np.argmin(candidate_distances))
                if candidate_distances[best] <= GAP_REUSE_UM:
                    middle_id = candidates[best]
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1
            if middle_id is None:
                if synthetic_added >= max_synthetic:
                    stats["gap_skipped_node_cap"] += 1
                    continue
                middle_id = next_id
                next_id += 1
                refined = refine_midpoint(data_dir, dataset, mid_t, midpoint, frame_cache, stats)
                nodes[middle_id] = {
                    "node_id": middle_id,
                    "t": mid_t,
                    "z": refined[0],
                    "y": refined[1],
                    "x": refined[2],
                }
                synthetic_added += 1
                stats["gap_inserted_synthetic"] += 1
            middle = nodes[middle_id]
            new_edges.extend(
                [
                    {"source_id": source_id, "target_id": middle_id, "edge_prob": None, "distance_um": edge_distance_um(source, middle)},
                    {"source_id": middle_id, "target_id": target_id, "edge_prob": None, "distance_um": edge_distance_um(middle, target)},
                ]
            )
            outgoing.update((source_id, middle_id))
            incoming.update((middle_id, target_id))
            used_starts.add(target_id)
            stats["gap_pairs_selected"] += 1
            stats["gap_added_edges"] += 2
    stats["gap_added_nodes"] = synthetic_added
    return nodes, [*edges, *new_edges]


def safe_division(
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    stats: defaultdict[str, int],
) -> list[dict[str, Any]]:
    out_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    incoming: set[int] = set()
    for edge in edges:
        out_by_source[int(edge["source_id"])].append(edge)
        incoming.add(int(edge["target_id"]))
    ids_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        ids_by_t[int(node["t"])].append(node_id)
    existing = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP))
    added: list[dict[str, Any]] = []
    used_targets: set[int] = set()
    for t in sorted(ids_by_t):
        sources = [x for x in ids_by_t[t] if len(out_by_source.get(x, [])) == 1]
        candidates = [x for x in ids_by_t.get(t + 1, []) if x not in incoming and x not in used_targets]
        frame_cap = max(1, round(len(sources) * SAFE_DIV_FRAME_FRAC_CAP))
        proposals: list[tuple[float, int, int, float]] = []
        for source_id in sources:
            child_id = int(out_by_source[source_id][0]["target_id"])
            child = nodes[child_id]
            if edge_distance_um(nodes[source_id], child) > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            for candidate_id in candidates:
                if (source_id, candidate_id) in existing:
                    continue
                parent_distance = edge_distance_um(nodes[source_id], nodes[candidate_id])
                sister_distance = edge_distance_um(child, nodes[candidate_id])
                if parent_distance <= SAFE_DIV_MAX_UM and sister_distance <= SAFE_DIV_SISTER_MAX_UM:
                    proposals.append((parent_distance + 0.15 * sister_distance, source_id, candidate_id, parent_distance))
        stats["safe_division_candidates"] += len(proposals)
        added_this_frame = 0
        for _, source_id, candidate_id, parent_distance in sorted(proposals):
            if len(added) >= global_cap or added_this_frame >= frame_cap:
                break
            if candidate_id in incoming or candidate_id in used_targets:
                continue
            added.append(
                {"source_id": source_id, "target_id": candidate_id, "edge_prob": None, "distance_um": parent_distance}
            )
            used_targets.add(candidate_id)
            added_this_frame += 1
    stats["safe_divisions_added"] = len(added)
    return [*edges, *added]


def short_track_filter(
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


def linefit(
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    stats: defaultdict[str, int],
) -> dict[int, dict[str, Any]]:
    predecessor: dict[int, list[int]] = defaultdict(list)
    successor: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        if int(nodes[target_id]["t"]) == int(nodes[source_id]["t"]) + 1:
            successor[source_id].append(target_id)
            predecessor[target_id].append(source_id)
    original = {node_id: np.asarray(node_point(node), dtype=np.float64) for node_id, node in nodes.items()}
    updated: dict[int, np.ndarray] = {}
    for node_id in sorted(nodes):
        neighbourhood = [(0, node_id)]
        current = node_id
        for step in range(1, LINEFIT_WINDOW + 1):
            previous = predecessor.get(current, [])
            if len(previous) != 1:
                break
            current = previous[0]
            neighbourhood.append((-step, current))
        current = node_id
        for step in range(1, LINEFIT_WINDOW + 1):
            following = successor.get(current, [])
            if len(following) != 1:
                break
            current = following[0]
            neighbourhood.append((step, current))
        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue
        dts = np.asarray([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original[x] for _, x in neighbourhood])
        fitted = np.asarray([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)])
        if np.isfinite(fitted).all():
            updated[node_id] = (1.0 - LINEFIT_WEIGHT) * original[node_id] + LINEFIT_WEIGHT * fitted
    for node_id, point in updated.items():
        nodes[node_id].update(z=float(point[0]), y=float(point[1]), x=float(point[2]))
    stats["linefit_smoothed_nodes"] = len(updated)
    return nodes


def validate_and_clip_graph(
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    dataset: str,
    data_dir: Path,
    stats: defaultdict[str, int],
) -> None:
    """Enforce submission-safe time/coordinate/edge invariants per video."""
    array = zarr.open_group(str(data_dir / f"{dataset}.zarr"), mode="r")["0"]
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f"{dataset}: unexpected image shape {shape}")
    if not nodes:
        raise ValueError(f"{dataset}: postprocess removed every node")

    clipped_nodes = 0
    clipped_coordinates = 0
    upper = np.asarray(shape[1:], dtype=np.float64) - 1.0
    for node_id, node in nodes.items():
        t = int(node["t"])
        if not 0 <= t < shape[0]:
            raise ValueError(f"{dataset}: node {node_id} has out-of-range t={t}, T={shape[0]}")
        point = np.asarray(node_point(node), dtype=np.float64)
        if not np.isfinite(point).all():
            raise ValueError(f"{dataset}: node {node_id} has non-finite coordinate {point.tolist()}")
        clipped = np.clip(point, 0.0, upper)
        changed = ~np.isclose(point, clipped, rtol=0.0, atol=1e-9)
        if changed.any():
            clipped_nodes += 1
            clipped_coordinates += int(changed.sum())
            node.update(z=float(clipped[0]), y=float(clipped[1]), x=float(clipped[2]))

    seen_edges: set[tuple[int, int]] = set()
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(f"{dataset}: dangling edge {source_id}->{target_id}")
        pair = (source_id, target_id)
        if pair in seen_edges:
            raise ValueError(f"{dataset}: duplicate edge {source_id}->{target_id}")
        seen_edges.add(pair)
        source_t = int(nodes[source_id]["t"])
        target_t = int(nodes[target_id]["t"])
        if target_t != source_t + 1:
            raise ValueError(
                f"{dataset}: non-consecutive edge {source_id}->{target_id}, "
                f"times {source_t}->{target_t}"
            )
    stats["bounds_clipped_nodes"] += clipped_nodes
    stats["bounds_clipped_coordinates"] += clipped_coordinates
    stats["validated_nodes"] = len(nodes)
    stats["validated_edges"] = len(edges)


def copy_graph_fallback(source: Path, destination: Path) -> None:
    """Replace a partial postprocess graph with the original valid GEFF."""
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def prune_isolated(
    nodes: dict[int, dict[str, Any]], edges: list[dict[str, Any]], stats: defaultdict[str, int]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
    if not incident:
        return nodes, edges
    kept = {node_id: node for node_id, node in nodes.items() if node_id in incident}
    stats["pruned_isolated_nodes"] += len(nodes) - len(kept)
    return kept, edges


def apply_arm(
    arm: str,
    source_nodes: dict[int, dict[str, Any]],
    source_edges: list[dict[str, Any]],
    dataset: str,
    data_dir: Path,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes = {node_id: dict(node) for node_id, node in source_nodes.items()}
    edges = [dict(edge) for edge in source_edges]
    stats: defaultdict[str, int] = defaultdict(int)
    if arm in {"motion_relink", "full_stack"}:
        replacement = motion_relink(nodes, stats)
        if replacement:
            edges = replacement
    if arm in {"gap_close", "full_stack", "gap_short_track", "gap_short_track_linefit"}:
        nodes, edges = gap_close(nodes, edges, dataset, data_dir, stats)
    if arm in {"safe_division", "full_stack"}:
        edges = safe_division(nodes, edges, stats)
    if arm == "full_stack":
        nodes, edges = prune_isolated(nodes, edges, stats)
    if arm in {
        "short_track", "full_stack", "short_track_linefit",
        "gap_short_track", "gap_short_track_linefit",
    }:
        nodes, edges = short_track_filter(nodes, edges, dataset, stats)
    if arm in {"linefit", "full_stack", "short_track_linefit", "gap_short_track_linefit"}:
        nodes = linefit(nodes, edges, stats)
    validate_and_clip_graph(nodes, edges, dataset, data_dir, stats)
    return nodes, edges, dict(stats)


def main() -> None:
    global GAP_CLOSE_UM, GAP_REUSE_UM, GAP_MAX_ADDED_FRAC, GAP_MAX_ADDED_ABS, MIN_TRACK_LEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--gap-close-um", type=float, default=GAP_CLOSE_UM)
    parser.add_argument("--gap-reuse-um", type=float, default=GAP_REUSE_UM)
    parser.add_argument("--gap-max-added-frac", type=float, default=GAP_MAX_ADDED_FRAC)
    parser.add_argument("--gap-max-added-abs", type=int, default=GAP_MAX_ADDED_ABS)
    parser.add_argument("--min-track-len", type=int, default=MIN_TRACK_LEN)
    parser.add_argument("--disable-dataset-min-track-overrides", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.gap_close_um) or args.gap_close_um <= 0:
        raise ValueError("gap-close-um must be finite and positive")
    if not math.isfinite(args.gap_reuse_um) or args.gap_reuse_um < 0:
        raise ValueError("gap-reuse-um must be finite and non-negative")
    if not math.isfinite(args.gap_max_added_frac) or not 0 <= args.gap_max_added_frac <= 1:
        raise ValueError("gap-max-added-frac must be in [0, 1]")
    if args.gap_max_added_abs < 0 or args.min_track_len < 1:
        raise ValueError("gap-max-added-abs must be non-negative and min-track-len positive")
    GAP_CLOSE_UM = args.gap_close_um
    GAP_REUSE_UM = args.gap_reuse_um
    GAP_MAX_ADDED_FRAC = args.gap_max_added_frac
    GAP_MAX_ADDED_ABS = args.gap_max_added_abs
    MIN_TRACK_LEN = args.min_track_len
    if args.disable_dataset_min_track_overrides:
        MIN_TRACK_LEN_BY_DATASET.clear()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    sources = sorted(args.source_dir.glob("*.geff"))
    if args.limit_videos is not None:
        sources = sources[: args.limit_videos]
    if not sources:
        raise FileNotFoundError(f"no GEFF graphs under {args.source_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        dataset = source.stem
        source_nodes, source_edges = read_graph(source)
        for arm in arms:
            destination = args.output_root / "predictions" / arm / f"{dataset}.geff"
            resource_guard = (
                len(source_nodes) > POSTPROCESS_MAX_NODES
                or len(source_edges) > POSTPROCESS_MAX_EDGES
            )
            if resource_guard:
                copy_graph_fallback(source, destination)
                nodes, edges = source_nodes, source_edges
                stats = {
                    "postprocess_fallback": 1,
                    "postprocess_fallback_type": "ResourceGuard",
                }
                fallback_reason = (
                    f"graph exceeds safe postprocess size: nodes={len(source_nodes)} "
                    f"edges={len(source_edges)} limits={POSTPROCESS_MAX_NODES}/"
                    f"{POSTPROCESS_MAX_EDGES}"
                )
                print(
                    f"[{index}/{len(sources)}] {dataset} {arm} "
                    f"fallback=ResourceGuard: {fallback_reason}",
                    flush=True,
                )
            else:
                try:
                    nodes, edges, stats = apply_arm(
                        arm, source_nodes, source_edges, dataset, args.data_dir
                    )
                    write_graph(destination, nodes, edges)
                    fallback_reason = ""
                except Exception as exc:
                    traceback.print_exc()
                    copy_graph_fallback(source, destination)
                    nodes, edges = source_nodes, source_edges
                    stats = {
                        "postprocess_fallback": 1,
                        "postprocess_fallback_type": type(exc).__name__,
                    }
                    fallback_reason = str(exc).replace("\n", " ")[:1000]
                    print(
                        f"[{index}/{len(sources)}] {dataset} {arm} "
                        f"fallback={type(exc).__name__}: {fallback_reason}",
                        flush=True,
                    )
            row = {
                "dataset": dataset,
                "arm": arm,
                "source_nodes": len(source_nodes),
                "source_edges": len(source_edges),
                "nodes": len(nodes),
                "edges": len(edges),
                "postprocess_fallback_reason": fallback_reason,
                **stats,
            }
            rows.append(row)
            print(
                f"[{index}/{len(sources)}] {dataset} {arm} "
                f"nodes={len(source_nodes)}->{len(nodes)} edges={len(source_edges)}->{len(edges)}",
                flush=True,
            )
    columns = sorted({key for row in rows for key in row})
    with (args.output_root / "postprocess_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source_dir": str(args.source_dir),
        "data_dir": str(args.data_dir),
        "videos": len(sources),
        "arms": list(arms),
        "voxel_scale_um": VOXEL_SCALE_UM.tolist(),
        "gap_close_um_per_step": GAP_CLOSE_UM,
        "gap_endpoint_gate_um": 2.0 * GAP_CLOSE_UM,
        "gap_reuse_um": GAP_REUSE_UM,
        "gap_max_added_frac": GAP_MAX_ADDED_FRAC,
        "gap_max_added_abs": GAP_MAX_ADDED_ABS,
        "min_track_len": MIN_TRACK_LEN,
        "dataset_min_track_overrides": dict(MIN_TRACK_LEN_BY_DATASET),
        "note": "motion_relink is geometry-only because rejected raw learned edge candidates are unavailable",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
