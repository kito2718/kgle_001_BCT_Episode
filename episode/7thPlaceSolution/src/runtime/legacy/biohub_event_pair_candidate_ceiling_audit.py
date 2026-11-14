#!/usr/bin/env python3
"""Audit the candidate ceiling for Parent -> two-Daughter event hypotheses.

The legacy metric labels any two t+1 detector peaks matched to Daughter roles.
The stricter metric follows each Parent's two GEFF child branches and requires
one distinct detector peak for each branch. GT is used only for this audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import geff
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_division_crop_classifier import DAUGHTER_ROLES, SPACING_ZYX, atomic_json, write_csv
from biohub_division_detected_geometry_classifier import load_peaks
from biohub_division_physical_pair_cpu import (
    Candidate,
    PhysicalFeatureBuilder,
    PointClouds,
    _detector_point_roles,
    make_case_id,
)


COORDINATE_RE = re.compile(r"__z(-?[0-9.]+)_y(-?[0-9.]+)_x(-?[0-9.]+)_")


def parse_numbers(value: str, cast: type) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--role-lookup", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radii-um", default="8,10,12,15,20")
    parser.add_argument("--top-ks", default="4,6,8,12")
    parser.add_argument("--time-offsets", default="1,2,3")
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--heuristic-top-k", type=int, default=3)
    parser.add_argument("--baseline-radius-um", type=float, default=15.0)
    parser.add_argument("--baseline-top-k", type=int, default=4)
    parser.add_argument(
        "--folds", default="",
        help="Optional comma-separated candidate folds to audit (for example 0,1,2).",
    )
    parser.add_argument(
        "--refresh-parent-centers-from-detector", action="store_true",
        help=(
            "Match each exact GT Parent to the current detector peaks at t and use the "
            "matched detector centre for candidate generation. Missing Parents remain in "
            "the audit and reduce the end-to-end ceiling."
        ),
    )
    parser.add_argument("--limit-parents", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_parent_candidates(
    path: Path, limit: int | None, folds: set[int] | None = None,
) -> list[Candidate]:
    output: list[Candidate] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 1:
                continue
            fold = int(row["fold"])
            if folds and fold not in folds:
                continue
            match = COORDINATE_RE.search(row["sample_id"])
            if match is None:
                raise ValueError(f"cannot parse Parent coordinate: {row['sample_id']}")
            z, y, x = (float(value) for value in match.groups())
            case_id = row["case_id"]
            stem, frame_text = case_id.rsplit("_t", 1)
            output.append(Candidate(
                sample_id=row["sample_id"], source="detector", split=str(fold),
                case_id=case_id, stem=stem, frame=int(frame_text), frame_width=len(frame_text),
                center_um=np.asarray((z, y, x), dtype=np.float64) * SPACING_ZYX,
                label=1, category=row["category"], negative_subtype=row["negative_subtype"],
            ))
            if limit is not None and len(output) >= limit:
                break
    return output


def required_cases(candidates: list[Candidate]) -> set[str]:
    output: set[str] = set()
    for candidate in candidates:
        for dt in range(4):
            case_id = make_case_id(candidate.stem, candidate.frame + dt, candidate.frame_width)
            if case_id is not None:
                output.add(case_id)
    return output


def build_clouds(
    candidates: list[Candidate],
    roles: dict[str, list[list[int]]],
    peak_paths: list[Path],
    radius_um: float,
) -> PointClouds:
    peaks = load_peaks(peak_paths, required_cases(candidates))
    coordinates = {
        case_id: np.asarray([peak.coordinate for peak in rows], dtype=np.float64).reshape(-1, 3)
        * SPACING_ZYX
        for case_id, rows in peaks.items()
    }
    scores = {
        case_id: np.asarray([peak.score for peak in rows], dtype=np.float64)
        for case_id, rows in peaks.items()
    }
    role_values = _detector_point_roles(roles, peaks, radius_um)
    return PointClouds(coordinates=coordinates, scores=scores, roles=role_values)


def load_graph(path: Path) -> nx.DiGraph:
    graph, _ = geff.read(
        str(path), node_props=["t", "z", "y", "x"], structure_validation=False
    )
    return graph


def node_position_um(graph: nx.DiGraph, node: Any) -> np.ndarray:
    attrs = graph.nodes[node]
    return np.asarray([attrs["z"], attrs["y"], attrs["x"]], dtype=np.float64) * SPACING_ZYX


def resolve_parent_node(graph: nx.DiGraph, candidate: Candidate) -> tuple[Any, float]:
    nodes = [
        node for node in graph.nodes
        if int(graph.nodes[node]["t"]) == candidate.frame and graph.out_degree(node) == 2
    ]
    if not nodes:
        raise RuntimeError(f"no two-child Parent node for {candidate.sample_id}")
    distances = np.asarray([
        np.linalg.norm(node_position_um(graph, node) - candidate.center_um) for node in nodes
    ])
    best = int(np.argmin(distances))
    return nodes[best], float(distances[best])


def descendant_branches(graph: nx.DiGraph, parent: Any, dt: int) -> list[list[Any]]:
    children = list(graph.successors(parent))
    if len(children) != 2:
        raise RuntimeError(f"Parent {parent} has {len(children)} children, expected 2")
    branches: list[list[Any]] = []
    for child in children:
        frontier = [child]
        for _ in range(dt - 1):
            frontier = [successor for node in frontier for successor in graph.successors(node)]
            if not frontier:
                break
        branches.append(frontier)
    return branches


def match_lineage_branches(
    branch_points: list[np.ndarray], detector_points: np.ndarray, radius_um: float,
) -> tuple[list[int], list[float]]:
    """Maximum-cardinality two-branch matching with radius-aware dummies."""
    if len(branch_points) != 2 or len(detector_points) == 0:
        return [], []
    distances = np.full((2, len(detector_points)), 1e6, dtype=np.float64)
    for branch_index, points in enumerate(branch_points):
        if len(points):
            distances[branch_index] = np.linalg.norm(
                detector_points[:, None, :] - points[None, :, :], axis=2
            ).min(axis=1)
    costs = np.full((2, len(detector_points) + 2), radius_um + 1e-3, dtype=np.float64)
    costs[:, : len(detector_points)] = np.where(distances <= radius_um, distances, 1e6)
    rows, columns = linear_sum_assignment(costs)
    matched: list[tuple[int, int, float]] = []
    for row, column in zip(rows, columns):
        if column < len(detector_points) and distances[row, column] <= radius_um:
            matched.append((int(row), int(column), float(distances[row, column])))
    matched.sort()
    return [item[1] for item in matched], [item[2] for item in matched]


def pool_and_rank(
    points: np.ndarray,
    predicted_center: np.ndarray,
    radius_um: float,
    top_k: int,
    heuristic_top_k: int,
) -> dict[str, Any]:
    distances = np.linalg.norm(points - predicted_center[None, :], axis=1)
    order = np.argsort(distances, kind="stable")[:top_k]
    within = order[distances[order] <= radius_um]
    strict_pool = within
    fallback_pool = within if len(within) >= 2 else order

    def rank_pairs(pool: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
        rows: list[tuple[float, int, int]] = []
        for first, second in combinations(pool.tolist(), 2):
            q1, q2 = points[first], points[second]
            v1, v2 = q1 - predicted_center, q2 - predicted_center
            d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            sister = float(np.linalg.norm(q1 - q2))
            midpoint = float(np.linalg.norm((q1 + q2) / 2.0 - predicted_center))
            score = midpoint + 0.05 * (d1 + d2) + 0.02 * abs(sister - 10.5)
            rows.append((score, min(first, second), max(first, second)))
        rows.sort()
        pairs = [(first, second) for _, first, second in rows]
        return pairs, {pair: index + 1 for index, pair in enumerate(pairs)}

    strict_pairs, strict_ranks = rank_pairs(strict_pool)
    fallback_pairs, fallback_ranks = rank_pairs(fallback_pool)
    return {
        "distances": distances,
        "nearest_order": order,
        "strict_pool": strict_pool,
        "fallback_pool": fallback_pool,
        "strict_pairs": strict_pairs,
        "strict_ranks": strict_ranks,
        "fallback_pairs": fallback_pairs,
        "fallback_ranks": fallback_ranks,
        "heuristic_top_k": heuristic_top_k,
    }


def pair_stage(
    pair_indices: list[int], pool: dict[str, Any], role_values: np.ndarray,
) -> dict[str, Any]:
    exact_pair = tuple(sorted(pair_indices)) if len(pair_indices) == 2 else None
    nearest = set(map(int, pool["nearest_order"]))
    strict = set(map(int, pool["strict_pool"]))
    fallback = set(map(int, pool["fallback_pool"]))
    exact_nearest = bool(exact_pair and set(exact_pair) <= nearest)
    exact_strict = bool(exact_pair and set(exact_pair) <= strict)
    exact_fallback = bool(exact_pair and set(exact_pair) <= fallback)
    strict_rank = pool["strict_ranks"].get(exact_pair) if exact_pair else None
    fallback_rank = pool["fallback_ranks"].get(exact_pair) if exact_pair else None
    legacy_ranks = [
        rank for pair, rank in pool["fallback_ranks"].items()
        if pair[0] < len(role_values) and pair[1] < len(role_values)
        and int(role_values[pair[0]]) in DAUGHTER_ROLES
        and int(role_values[pair[1]]) in DAUGHTER_ROLES
    ]
    legacy_rank = min(legacy_ranks) if legacy_ranks else None
    return {
        "both_lineages_detected": int(exact_pair is not None),
        "lineage_pair_in_nearest_topk": int(exact_nearest),
        "lineage_pair_in_strict_pool": int(exact_strict),
        "lineage_pair_in_fallback_pool": int(exact_fallback),
        "lineage_strict_heuristic_rank": strict_rank if strict_rank is not None else "",
        "lineage_fallback_heuristic_rank": fallback_rank if fallback_rank is not None else "",
        "lineage_strict_top3": int(strict_rank is not None and strict_rank <= pool["heuristic_top_k"]),
        "lineage_fallback_top3": int(fallback_rank is not None and fallback_rank <= pool["heuristic_top_k"]),
        "legacy_role_pair_rank": legacy_rank if legacy_rank is not None else "",
        "legacy_role_pair_top3": int(legacy_rank is not None and legacy_rank <= pool["heuristic_top_k"]),
        "strict_pair_count": len(pool["strict_pairs"]),
        "fallback_pair_count": len(pool["fallback_pairs"]),
        "fallback_used": int(len(pool["strict_pool"]) < 2),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def main() -> int:
    args = parse_args()
    radii = parse_numbers(args.radii_um, float)
    top_ks = parse_numbers(args.top_ks, int)
    time_offsets = parse_numbers(args.time_offsets, int)
    requested_folds = set(parse_numbers(args.folds, int)) if args.folds else None
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    candidates = read_parent_candidates(
        args.candidate_scores, args.limit_parents, requested_folds
    )
    if not candidates:
        raise RuntimeError("no Parent candidates")
    roles_payload = json.loads(args.role_lookup.read_text())
    roles: dict[str, list[list[int]]] = roles_payload.get("cases", roles_payload)
    clouds = build_clouds(candidates, roles, args.detector_peaks, args.match_radius_um)
    builder = PhysicalFeatureBuilder(
        clouds, max(radii), max(top_ks)
    )
    graphs = {stem: load_graph(args.train_dir / f"{stem}.geff") for stem in sorted({c.stem for c in candidates})}

    event_rows: list[dict[str, Any]] = []
    event_meta: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        graph = graphs[candidate.stem]
        parent_node, parent_match_um = resolve_parent_node(graph, candidate)
        if parent_match_um > args.match_radius_um:
            raise RuntimeError(
                f"Parent lineage match exceeds {args.match_radius_um} um: "
                f"{candidate.sample_id} distance={parent_match_um:.3f}"
            )
        parent_gt_um = node_position_um(graph, parent_node)
        parent_detector_detected = 1
        parent_detector_match_um: float | str = parent_match_um
        candidate_center_um = candidate.center_um
        candidate_center_source = "candidate_scores"
        if args.refresh_parent_centers_from_detector:
            source_points = builder.cloud(
                candidate.stem, candidate.frame, candidate.frame_width
            )
            if len(source_points):
                source_distances = np.linalg.norm(
                    source_points - parent_gt_um[None, :], axis=1
                )
                nearest_source = int(np.argmin(source_distances))
                parent_detector_match_um = float(source_distances[nearest_source])
                parent_detector_detected = int(
                    parent_detector_match_um <= args.match_radius_um
                )
                candidate_center_um = (
                    source_points[nearest_source]
                    if parent_detector_detected else parent_gt_um
                )
            else:
                parent_detector_detected = 0
                parent_detector_match_um = ""
                candidate_center_um = parent_gt_um
            candidate_center_source = (
                "current_detector" if parent_detector_detected else "gt_diagnostic_fallback"
            )
        meta = {
            "candidate_index": candidate_index, "sample_id": candidate.sample_id,
            "case_id": candidate.case_id, "fold": int(candidate.split), "stem": candidate.stem,
            "parent_node": int(parent_node), "parent_match_um": parent_match_um,
            "parent_detector_detected": parent_detector_detected,
            "parent_detector_match_um": parent_detector_match_um,
            "candidate_center_source": candidate_center_source,
        }
        event_meta[candidate.sample_id] = meta
        for dt in time_offsets:
            target_case = make_case_id(candidate.stem, candidate.frame + dt, candidate.frame_width)
            points = builder.cloud(candidate.stem, candidate.frame + dt, candidate.frame_width)
            role_values = builder.roles(candidate.stem, candidate.frame + dt, candidate.frame_width)
            branches = descendant_branches(graph, parent_node, dt)
            branch_points = [
                np.asarray([node_position_um(graph, node) for node in branch], dtype=np.float64).reshape(-1, 3)
                for branch in branches
            ]
            matched_indices, matched_distances = match_lineage_branches(
                branch_points, points, args.match_radius_um
            )
            shift, shift_available = builder.cumulative_shift(
                candidate.stem, candidate.frame, candidate.frame + dt, candidate.frame_width
            )
            predicted = candidate_center_um + shift
            target_distances = np.linalg.norm(points - predicted[None, :], axis=1) if len(points) else np.empty(0)
            max_matched_distance_from_parent = (
                max(float(target_distances[index]) for index in matched_indices)
                if matched_indices else 0.0
            )
            distractors_closer = (
                int(np.count_nonzero(target_distances < max_matched_distance_from_parent)) - len(matched_indices)
                if len(matched_indices) == 2 else 0
            )
            for radius in radii:
                for top_k in top_ks:
                    pool = pool_and_rank(points, predicted, radius, top_k, args.heuristic_top_k)
                    stages = pair_stage(matched_indices, pool, role_values)
                    event_rows.append({
                        **meta, "target_case_id": target_case, "dt": dt,
                        "radius_um": radius, "top_k_points": top_k,
                        "detector_point_count": len(points),
                        "lineage_branches_present": sum(bool(branch) for branch in branches),
                        "matched_lineage_count": len(matched_indices),
                        "lineage_match_distance_max_um": max(matched_distances) if matched_distances else "",
                        "matched_max_parent_distance_um": max_matched_distance_from_parent,
                        "distractors_closer_than_far_daughter": max(0, distractors_closer),
                        "global_shift_available": int(shift_available),
                        **stages,
                    })
        if (candidate_index + 1) % 25 == 0:
            print(f"audited {candidate_index + 1}/{len(candidates)} Parent events", flush=True)

    sweep_rows: list[dict[str, Any]] = []
    for dt in time_offsets:
        for radius in radii:
            for top_k in top_ks:
                rows = [
                    row for row in event_rows
                    if row["dt"] == dt and math.isclose(float(row["radius_um"]), radius)
                    and row["top_k_points"] == top_k
                ]
                pair_counts = [float(row["fallback_pair_count"]) for row in rows]
                strict_counts = [float(row["strict_pair_count"]) for row in rows]
                sweep_rows.append({
                    "dt": dt, "radius_um": radius, "top_k_points": top_k,
                    "parent_events": len(rows),
                    "parent_detector_availability": np.mean([
                        row["parent_detector_detected"] for row in rows
                    ]),
                    "both_lineages_detected": sum(int(row["both_lineages_detected"]) for row in rows),
                    "lineage_detection_availability": np.mean([row["both_lineages_detected"] for row in rows]),
                    "lineage_nearest_topk_availability": np.mean([row["lineage_pair_in_nearest_topk"] for row in rows]),
                    "lineage_strict_pool_availability": np.mean([row["lineage_pair_in_strict_pool"] for row in rows]),
                    "lineage_fallback_pool_availability": np.mean([row["lineage_pair_in_fallback_pool"] for row in rows]),
                    "lineage_strict_top3_availability": np.mean([row["lineage_strict_top3"] for row in rows]),
                    "lineage_fallback_top3_availability": np.mean([row["lineage_fallback_top3"] for row in rows]),
                    "end_to_end_lineage_detection_availability": np.mean([
                        int(row["parent_detector_detected"])
                        * int(row["both_lineages_detected"])
                        for row in rows
                    ]),
                    "end_to_end_fallback_pool_availability": np.mean([
                        int(row["parent_detector_detected"])
                        * int(row["lineage_pair_in_fallback_pool"])
                        for row in rows
                    ]),
                    "end_to_end_fallback_top3_availability": np.mean([
                        int(row["parent_detector_detected"])
                        * int(row["lineage_fallback_top3"])
                        for row in rows
                    ]),
                    "legacy_role_top3_availability": np.mean([row["legacy_role_pair_top3"] for row in rows]),
                    "mean_strict_pair_count": float(np.mean(strict_counts)),
                    "mean_fallback_pair_count": float(np.mean(pair_counts)),
                    "median_fallback_pair_count": float(np.median(pair_counts)),
                    "p95_fallback_pair_count": percentile(pair_counts, 0.95),
                    "mean_distractors_closer": float(np.mean([
                        row["distractors_closer_than_far_daughter"] for row in rows
                    ])),
                })

    union_rows: list[dict[str, Any]] = []
    for radius in radii:
        for top_k in top_ks:
            rows = [
                row for row in event_rows
                if math.isclose(float(row["radius_um"]), radius)
                and row["top_k_points"] == top_k
            ]
            by_sample = {
                candidate.sample_id: [row for row in rows if row["sample_id"] == candidate.sample_id]
                for candidate in candidates
            }
            detection_union = [
                int(any(int(row["both_lineages_detected"]) for row in sample_rows))
                for sample_rows in by_sample.values()
            ]
            parent_detected = [
                int(sample_rows[0]["parent_detector_detected"])
                for sample_rows in by_sample.values()
            ]
            strict_union = [
                int(any(int(row["lineage_pair_in_strict_pool"]) for row in sample_rows))
                for sample_rows in by_sample.values()
            ]
            fallback_union = [
                int(any(int(row["lineage_pair_in_fallback_pool"]) for row in sample_rows))
                for sample_rows in by_sample.values()
            ]
            strict_top3_union = [
                int(any(int(row["lineage_strict_top3"]) for row in sample_rows))
                for sample_rows in by_sample.values()
            ]
            fallback_top3_union = [
                int(any(int(row["lineage_fallback_top3"]) for row in sample_rows))
                for sample_rows in by_sample.values()
            ]
            total_pair_counts = [
                sum(int(row["fallback_pair_count"]) for row in sample_rows)
                for sample_rows in by_sample.values()
            ]
            total_top3_counts = [
                sum(min(args.heuristic_top_k, int(row["fallback_pair_count"])) for row in sample_rows)
                for sample_rows in by_sample.values()
            ]
            union_rows.append({
                "time_offsets": "|".join(map(str, time_offsets)),
                "radius_um": radius, "top_k_points": top_k, "parent_events": len(candidates),
                "parent_detector_availability": float(np.mean(parent_detected)),
                "lineage_detection_union_availability": float(np.mean(detection_union)),
                "lineage_strict_pool_union_availability": float(np.mean(strict_union)),
                "lineage_fallback_pool_union_availability": float(np.mean(fallback_union)),
                "lineage_strict_top3_union_availability": float(np.mean(strict_top3_union)),
                "lineage_fallback_top3_union_availability": float(np.mean(fallback_top3_union)),
                "end_to_end_detection_union_availability": float(np.mean([
                    parent * available
                    for parent, available in zip(parent_detected, detection_union)
                ])),
                "end_to_end_fallback_pool_union_availability": float(np.mean([
                    parent * available
                    for parent, available in zip(parent_detected, fallback_union)
                ])),
                "end_to_end_fallback_top3_union_availability": float(np.mean([
                    parent * available
                    for parent, available in zip(parent_detected, fallback_top3_union)
                ])),
                "mean_total_fallback_pair_count": float(np.mean(total_pair_counts)),
                "median_total_fallback_pair_count": float(np.median(total_pair_counts)),
                "p95_total_fallback_pair_count": percentile(total_pair_counts, 0.95),
                "mean_total_retained_top3_count": float(np.mean(total_top3_counts)),
            })

    def config_row(dt: int, radius: float, top_k: int, sample_id: str) -> dict[str, Any]:
        return next(
            row for row in event_rows
            if row["dt"] == dt and math.isclose(float(row["radius_um"]), radius)
            and row["top_k_points"] == top_k and row["sample_id"] == sample_id
        )

    baseline_rows = [
        row for row in event_rows
        if row["dt"] == 1 and math.isclose(float(row["radius_um"]), args.baseline_radius_um)
        and row["top_k_points"] == args.baseline_top_k
    ]
    failures: list[dict[str, Any]] = []
    time_recovery: list[dict[str, Any]] = []
    for row in baseline_rows:
        if int(row["legacy_role_pair_top3"]):
            continue
        if not int(row["parent_detector_detected"]):
            cause = "parent_missed"
        elif int(row["matched_lineage_count"]) == 0:
            cause = "both_daughters_missed"
        elif int(row["matched_lineage_count"]) == 1:
            cause = "one_daughter_missed"
        elif not int(row["lineage_pair_in_nearest_topk"]):
            cause = "point_topk_truncation"
        elif not int(row["lineage_pair_in_strict_pool"]):
            cause = "radius_exclusion"
        elif not int(row["lineage_fallback_top3"]):
            cause = "heuristic_pair_top3_truncation"
        else:
            cause = "legacy_role_assignment_mismatch"
        later: dict[str, Any] = {}
        for dt in [value for value in time_offsets if value > 1]:
            later_row = config_row(dt, args.baseline_radius_um, args.baseline_top_k, row["sample_id"])
            later[f"tplus{dt}_both_lineages_detected"] = later_row["both_lineages_detected"]
            later[f"tplus{dt}_fallback_pool"] = later_row["lineage_pair_in_fallback_pool"]
            later[f"tplus{dt}_fallback_top3"] = later_row["lineage_fallback_top3"]
        failure = {**row, "failure_cause": cause, **later}
        failures.append(failure)
        time_recovery.append({
            "sample_id": row["sample_id"], "case_id": row["case_id"], "fold": row["fold"],
            "baseline_failure_cause": cause, "tplus1_detected": row["both_lineages_detected"],
            **later,
        })

    baseline_sweep = next(
        row for row in sweep_rows
        if row["dt"] == 1 and math.isclose(float(row["radius_um"]), args.baseline_radius_um)
        and row["top_k_points"] == args.baseline_top_k
    )
    pareto_candidates = [
        row for row in sweep_rows
        if float(row["lineage_fallback_pool_availability"]) >= 0.95
        and float(row["mean_fallback_pair_count"]) <= 28.0
    ]
    pareto_candidates.sort(key=lambda row: (
        -float(row["lineage_fallback_top3_availability"]),
        float(row["mean_fallback_pair_count"]), int(row["dt"]),
    ))
    best_practical = pareto_candidates[0] if pareto_candidates else max(
        sweep_rows, key=lambda row: (
            float(row["lineage_fallback_pool_availability"]),
            -float(row["mean_fallback_pair_count"]),
        )
    )
    max_pool = max(float(row["lineage_fallback_pool_availability"]) for row in sweep_rows)
    max_top3 = max(float(row["lineage_fallback_top3_availability"]) for row in sweep_rows)
    max_union_detection = max(
        float(row["lineage_detection_union_availability"]) for row in union_rows
    )
    max_union_pool = max(
        float(row["lineage_fallback_pool_union_availability"]) for row in union_rows
    )
    max_union_top3 = max(
        float(row["lineage_fallback_top3_union_availability"]) for row in union_rows
    )
    best_union = max(union_rows, key=lambda row: (
        float(row["lineage_fallback_top3_union_availability"]),
        float(row["lineage_fallback_pool_union_availability"]),
        -float(row["mean_total_fallback_pair_count"]),
    ))
    if max_union_detection < 0.95:
        next_decision = "improve detector or multi-time union before pair reranking"
    elif max_union_pool < 0.95:
        next_decision = "broaden multi-time candidate generation before pair reranking"
    elif max_union_top3 < 0.95:
        next_decision = "train a multi-time full-pool pre-ranker before top-3 continuity tie-break"
    else:
        next_decision = "multi-time candidate ceiling is sufficient; rerank the per-time top-3 union"

    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_events": len(candidates),
        "requested_folds": sorted(requested_folds) if requested_folds else "all",
        "refresh_parent_centers_from_detector": args.refresh_parent_centers_from_detector,
        "parent_detector_availability": float(np.mean([
            row["parent_detector_detected"] for row in baseline_rows
        ])),
        "baseline": baseline_sweep,
        "legacy_baseline_expected": (
            121 / 150
            if len(candidates) == 150 and not args.refresh_parent_centers_from_detector
            else None
        ),
        "legacy_baseline_reproduced": (
            math.isclose(float(baseline_sweep["legacy_role_top3_availability"]), 121 / 150)
            if len(candidates) == 150 and not args.refresh_parent_centers_from_detector
            else None
        ),
        "baseline_legacy_missing": len(failures),
        "baseline_failure_counts": dict(Counter(row["failure_cause"] for row in failures)),
        "max_lineage_fallback_pool_availability": max_pool,
        "max_lineage_fallback_top3_availability": max_top3,
        "max_lineage_detection_union_availability": max_union_detection,
        "max_lineage_fallback_pool_union_availability": max_union_pool,
        "max_lineage_fallback_top3_union_availability": max_union_top3,
        "best_practical_config": best_practical,
        "best_time_union_config": best_union,
        "next_decision": next_decision,
        "lineage_contract": (
            "Each detector Parent is matched to its exact two-child GEFF node; "
            "t+2/t+3 follow the two child branches and require distinct detector peaks."
        ),
        "legacy_contract": "Any two detector peaks assigned a Daughter role, matching the prior event implementation.",
        "runtime_seconds": time.perf_counter() - started,
    }
    write_csv(args.output_dir / "event_config_rows.csv", event_rows)
    write_csv(args.output_dir / "sweep_summary.csv", sweep_rows)
    write_csv(args.output_dir / "time_union_summary.csv", union_rows)
    write_csv(args.output_dir / "baseline_failure_audit.csv", failures)
    write_csv(args.output_dir / "time_recovery.csv", time_recovery)
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
