#!/usr/bin/env python3
"""OOF pre-ranker for multi-time Division daughter-pair hypotheses.

The model sees detector coordinates, detector confidence, global-shift geometry,
and detector-only temporal assignment features.  GEFF lineage is used only to
label the exact daughter pair and to evaluate recall@K.  Parent events with no
current detector node remain in the denominator and are excluded from fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from biohub_division_crop_classifier import atomic_json, write_csv
from biohub_division_physical_pair_cpu import PhysicalFeatureBuilder, finite, make_case_id
from biohub_event_pair_candidate_ceiling_audit import (
    build_clouds,
    descendant_branches,
    load_graph,
    match_lineage_branches,
    node_position_um,
    parse_numbers,
    read_parent_candidates,
    resolve_parent_node,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--role-lookup", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--time-offsets", default="1,2,3")
    parser.add_argument("--pair-radius-um", type=float, default=15.0)
    parser.add_argument("--max-pair-points", type=int, default=6)
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--rank-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--limit-parents-per-fold", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_cases(candidates: list[Any]) -> set[str]:
    output: set[str] = set()
    for candidate in candidates:
        for offset in range(-1, 5):
            case_id = make_case_id(candidate.stem, candidate.frame + offset, candidate.frame_width)
            if case_id is not None:
                output.add(case_id)
    return output


def detector_score(builder: PhysicalFeatureBuilder, case_id: str | None, index: int) -> float:
    if case_id is None:
        return 0.0
    values = builder.clouds.scores.get(case_id, np.empty(0, dtype=np.float64))
    return float(values[index]) if index < len(values) else 0.0


def enumerate_rows(
    candidates: list[Any], builder: PhysicalFeatureBuilder, graphs: dict[str, Any], args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        graph = graphs[candidate.stem]
        parent_node, parent_match_um = resolve_parent_node(graph, candidate)
        if parent_match_um > args.match_radius_um:
            raise RuntimeError(
                f"Parent lineage match exceeds {args.match_radius_um} um: "
                f"{candidate.sample_id} distance={parent_match_um:.3f}"
            )
        gt_center = node_position_um(graph, parent_node)
        source = builder.cloud(candidate.stem, candidate.frame, candidate.frame_width)
        parent_detected = 0
        parent_detector_match_um: float | str = ""
        center = gt_center
        if len(source):
            distances = np.linalg.norm(source - gt_center[None, :], axis=1)
            source_index = int(np.argmin(distances))
            parent_detector_match_um = float(distances[source_index])
            parent_detected = int(parent_detector_match_um <= args.match_radius_um)
            if parent_detected:
                center = source[source_index]
        parent_pair_count = 0
        parent_positive_count = 0
        for dt in parse_numbers(args.time_offsets, int):
            target_frame = candidate.frame + dt
            target_case = make_case_id(candidate.stem, target_frame, candidate.frame_width)
            points = builder.cloud(candidate.stem, target_frame, candidate.frame_width)
            if len(points) < 2:
                continue
            shift, shift_available = builder.cumulative_shift(
                candidate.stem, candidate.frame, target_frame, candidate.frame_width
            )
            predicted = center + shift
            distances = np.linalg.norm(points - predicted[None, :], axis=1)
            order = np.argsort(distances, kind="stable")[: args.max_pair_points]
            within = order[distances[order] <= args.pair_radius_um]
            pool = within if len(within) >= 2 else order
            branches = descendant_branches(graph, parent_node, dt)
            branch_points = [
                np.asarray(
                    [node_position_um(graph, node) for node in branch], dtype=np.float64
                ).reshape(-1, 3)
                for branch in branches
            ]
            matched, _ = match_lineage_branches(branch_points, points, args.match_radius_um)
            exact_pair = tuple(sorted(matched)) if len(matched) == 2 else None
            local_counts = {
                radius: float(np.count_nonzero(distances <= radius))
                for radius in (5.0, 7.0, 10.0, 15.0, 20.0)
            }
            for first, second in combinations(pool.tolist(), 2):
                q1, q2 = points[first], points[second]
                v1, v2 = q1 - predicted, q2 - predicted
                d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                sister = float(np.linalg.norm(q1 - q2))
                midpoint = float(np.linalg.norm((q1 + q2) / 2.0 - predicted))
                denominator = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8)
                cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
                score1 = detector_score(builder, target_case, first)
                score2 = detector_score(builder, target_case, second)
                heuristic = midpoint + 0.05 * (d1 + d2) + 0.02 * abs(sister - 10.5)
                features: dict[str, float] = {
                    "geom_dt": float(dt),
                    "geom_dt1": float(dt == 1),
                    "geom_dt2": float(dt == 2),
                    "geom_dt3": float(dt == 3),
                    "geom_shift_available": float(shift_available),
                    "geom_shift_norm": float(np.linalg.norm(shift)),
                    "geom_d1": min(d1, d2),
                    "geom_d2": max(d1, d2),
                    "geom_distance_sum": d1 + d2,
                    "geom_distance_absdiff": abs(d1 - d2),
                    "geom_sister": sister,
                    "geom_midpoint": midpoint,
                    "geom_cosine": cosine,
                    "geom_heuristic": heuristic,
                    "geom_score_min": min(score1, score2),
                    "geom_score_max": max(score1, score2),
                    "geom_score_mean": (score1 + score2) / 2.0,
                    "geom_score_absdiff": abs(score1 - score2),
                    "geom_point_rank_max": float(max(np.flatnonzero(order == first)[0], np.flatnonzero(order == second)[0]) + 1),
                    "geom_pair_count_dt": float(math.comb(len(pool), 2)),
                }
                for radius, count in local_counts.items():
                    features[f"geom_count_r{int(radius)}"] = count
                pair = points[[first, second]]
                for target_offset in (-1, 0, 1, 2, 3, 4):
                    features.update(builder.assignment_features(
                        candidate,
                        pair,
                        target_frame,
                        candidate.frame + target_offset,
                        f"traj_t{target_offset:+d}".replace("+", "p").replace("-", "m"),
                    ))
                label = int(exact_pair == tuple(sorted((first, second))))
                parent_pair_count += 1
                parent_positive_count += label
                pair_rows.append({
                    "candidate_index": candidate_index,
                    "sample_id": candidate.sample_id,
                    "case_id": candidate.case_id,
                    "stem": candidate.stem,
                    "fold": int(candidate.split),
                    "parent_detector_detected": parent_detected,
                    "dt": dt,
                    "first": first,
                    "second": second,
                    "label": label,
                    **{name: finite(value) for name, value in features.items()},
                })
        parent_rows.append({
            "candidate_index": candidate_index,
            "sample_id": candidate.sample_id,
            "case_id": candidate.case_id,
            "stem": candidate.stem,
            "fold": int(candidate.split),
            "parent_node": int(parent_node),
            "parent_detector_detected": parent_detected,
            "parent_detector_match_um": parent_detector_match_um,
            "pair_count": parent_pair_count,
            "positive_pair_count": parent_positive_count,
            "positive_pair_available": int(parent_positive_count > 0),
        })
        if (candidate_index + 1) % 20 == 0:
            print(f"features {candidate_index + 1}/{len(candidates)} pairs={len(pair_rows)}", flush=True)
    return pair_rows, parent_rows


def model_factories(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000, class_weight="balanced", random_state=seed
            ),
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    }


def rank_metrics(
    condition: str,
    scores: np.ndarray,
    pair_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    rank_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[int, list[int]] = {}
    for row_index, row in enumerate(pair_rows):
        grouped.setdefault(int(row["candidate_index"]), []).append(row_index)
    audit: list[dict[str, Any]] = []
    top1_all = topk_all = top1_eligible = topk_eligible = 0
    eligible = parent_available = 0
    reciprocal: list[float] = []
    for parent in parent_rows:
        candidate_index = int(parent["candidate_index"])
        indices = grouped.get(candidate_index, [])
        parent_detected = int(parent["parent_detector_detected"])
        parent_available += parent_detected
        order = sorted(indices, key=lambda i: (-float(scores[i]), int(pair_rows[i]["dt"]), i))
        labels = [int(pair_rows[i]["label"]) for i in order]
        has_positive = bool(sum(labels)) and bool(parent_detected)
        eligible += int(has_positive)
        positive_rank = next((rank + 1 for rank, value in enumerate(labels) if value), None)
        hit1 = int(parent_detected and positive_rank == 1)
        hitk = int(parent_detected and positive_rank is not None and positive_rank <= rank_k)
        top1_all += hit1
        topk_all += hitk
        if has_positive:
            top1_eligible += hit1
            topk_eligible += hitk
            reciprocal.append(1.0 / positive_rank)
        audit.append({
            **parent,
            "condition": condition,
            "positive_rank": positive_rank if positive_rank is not None else "",
            "hit_top1": hit1,
            f"hit_top{rank_k}": hitk,
            "selected_dt": pair_rows[order[0]]["dt"] if order else "",
            "selected_label": pair_rows[order[0]]["label"] if order else "",
            "selected_score": float(scores[order[0]]) if order else "",
        })
    total = len(parent_rows)
    return {
        "condition": condition,
        "parent_events": total,
        "parent_detector_available": parent_available,
        "eligible_parent_events": eligible,
        "pair_rows": len(pair_rows),
        "positive_pair_rows": sum(int(row["label"]) for row in pair_rows),
        "pair_auprc": float(average_precision_score(
            np.asarray([int(row["label"]) for row in pair_rows]), scores
        )),
        "top1_all": top1_all / total if total else 0.0,
        f"top{rank_k}_all": topk_all / total if total else 0.0,
        "top1_eligible": top1_eligible / eligible if eligible else 0.0,
        f"top{rank_k}_eligible": topk_eligible / eligible if eligible else 0.0,
        "mrr_eligible": float(np.mean(reciprocal)) if reciprocal else 0.0,
    }, audit


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    requested_folds = set(parse_numbers(args.folds, int))
    candidates = read_parent_candidates(args.candidate_scores, None, requested_folds)
    if args.limit_parents_per_fold:
        counts: Counter[int] = Counter()
        selected = []
        for candidate in candidates:
            fold = int(candidate.split)
            if counts[fold] < args.limit_parents_per_fold:
                selected.append(candidate)
                counts[fold] += 1
        candidates = selected
    roles_payload = json.loads(args.role_lookup.read_text())
    roles = roles_payload.get("cases", roles_payload)
    # Build a slightly wider cloud window than the ceiling audit so temporal
    # assignment remains detector-only at t-1..t+4.
    from biohub_division_detected_geometry_classifier import load_peaks
    from biohub_division_physical_pair_cpu import PointClouds, _detector_point_roles
    peaks = load_peaks(args.detector_peaks, required_cases(candidates))
    from biohub_division_crop_classifier import SPACING_ZYX
    coordinates = {
        case_id: np.asarray([peak.coordinate for peak in rows], dtype=np.float64).reshape(-1, 3)
        * SPACING_ZYX
        for case_id, rows in peaks.items()
    }
    cloud_scores = {
        case_id: np.asarray([peak.score for peak in rows], dtype=np.float64)
        for case_id, rows in peaks.items()
    }
    role_values = _detector_point_roles(roles, peaks, args.match_radius_um)
    builder = PhysicalFeatureBuilder(
        PointClouds(coordinates=coordinates, scores=cloud_scores, roles=role_values),
        args.pair_radius_um,
        args.max_pair_points,
    )
    graphs = {
        stem: load_graph(args.train_dir / f"{stem}.geff")
        for stem in sorted({candidate.stem for candidate in candidates})
    }
    pair_rows, parent_rows = enumerate_rows(candidates, builder, graphs, args)
    if not pair_rows or not any(int(row["label"]) for row in pair_rows):
        raise RuntimeError("no labelled pair hypotheses")
    feature_names = sorted(
        name for name in pair_rows[0] if name.startswith(("geom_", "traj_"))
    )
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in pair_rows], dtype=np.float64
    )
    labels = np.asarray([int(row["label"]) for row in pair_rows], dtype=np.int8)
    folds = np.asarray([int(row["fold"]) for row in pair_rows], dtype=np.int8)
    parent_detected = np.asarray(
        [int(row["parent_detector_detected"]) for row in pair_rows], dtype=bool
    )
    sets = {
        "geometry": np.asarray([
            index for index, name in enumerate(feature_names) if name.startswith("geom_")
        ], dtype=np.int32),
        "geometry_temporal": np.arange(len(feature_names), dtype=np.int32),
    }
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    heuristic_scores = -np.asarray([float(row["geom_heuristic"]) for row in pair_rows])
    metrics, audit = rank_metrics(
        "heuristic", heuristic_scores, pair_rows, parent_rows, args.rank_k
    )
    results.append(metrics)
    audits.extend(audit)
    score_columns: dict[str, np.ndarray] = {"score_heuristic": heuristic_scores}
    for set_name, columns in sets.items():
        for model_name, factory in model_factories(args.seed, args.n_jobs).items():
            condition = f"{set_name}_{model_name}"
            oof = np.full(len(pair_rows), np.nan, dtype=np.float64)
            for validation_fold in sorted(requested_folds):
                train = (folds != validation_fold) & parent_detected
                validation = folds == validation_fold
                if not np.any(validation) or len(np.unique(labels[train])) < 2:
                    raise RuntimeError(
                        f"invalid OOF partition fold={validation_fold}: "
                        f"train={train.sum()} validation={validation.sum()}"
                    )
                model = factory()
                model.fit(matrix[train][:, columns], labels[train])
                oof[validation] = model.predict_proba(matrix[validation][:, columns])[:, 1]
                if model_name == "extra_trees":
                    values = model.feature_importances_
                    for name, value in zip([feature_names[i] for i in columns], values):
                        importance_rows.append({
                            "condition": condition,
                            "validation_fold": validation_fold,
                            "feature": name,
                            "importance": float(value),
                        })
            if np.isnan(oof).any():
                raise RuntimeError(f"missing OOF scores for {condition}")
            score_columns[f"score_{condition}"] = oof
            metrics, audit = rank_metrics(condition, oof, pair_rows, parent_rows, args.rank_k)
            results.append(metrics)
            audits.extend(audit)
    write_csv(args.output_dir / "pair_features_oof_scores.csv", [
        {**row, **{name: float(values[index]) for name, values in score_columns.items()}}
        for index, row in enumerate(pair_rows)
    ])
    write_csv(args.output_dir / "parent_rank_audit.csv", audits)
    write_csv(args.output_dir / "metrics.csv", results)
    write_csv(args.output_dir / "feature_importance.csv", importance_rows)
    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "folds": sorted(requested_folds),
        "parent_events": len(parent_rows),
        "parent_detector_available": sum(int(row["parent_detector_detected"]) for row in parent_rows),
        "positive_pair_pool_available": sum(
            int(row["parent_detector_detected"]) * int(row["positive_pair_available"])
            for row in parent_rows
        ),
        "pair_radius_um": args.pair_radius_um,
        "max_pair_points": args.max_pair_points,
        "time_offsets": parse_numbers(args.time_offsets, int),
        "rank_k": args.rank_k,
        "feature_policy": "detector-only geometry and temporal assignment; GEFF only labels/evaluation",
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
