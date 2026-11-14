#!/usr/bin/env python3
"""Infer daughter-pair scores for every high Parent-prior candidate in one fold.

Training rows contain labelled true-Parent daughter pairs from the other folds.
The target fold is selected only by its OOF Parent score; target-fold Parent and
daughter labels are never read.  Output coordinates are sufficient to attach a
true two-edge hyperedge term to the tracking ILP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from biohub_division_crop_classifier import SPACING_ZYX, atomic_json, write_csv
from biohub_division_detected_geometry_classifier import load_peaks
from biohub_division_multitime_pair_preranker import model_factories
from biohub_division_physical_pair_cpu import (
    Candidate,
    PhysicalFeatureBuilder,
    PointClouds,
    finite,
    make_case_id,
)


COORDINATE_RE = re.compile(r"__z(-?[0-9.]+)_y(-?[0-9.]+)_x(-?[0-9.]+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--training-pair-scores", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--parent-score-field", default="fusion_history_image_w0p75")
    parser.add_argument("--min-parent-prior", type=float, default=0.995)
    parser.add_argument("--parent-match-um", type=float, default=0.25)
    parser.add_argument("--pair-radius-um", type=float, default=15.0)
    parser.add_argument("--max-pair-points", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_target_candidates(args: argparse.Namespace) -> list[tuple[Candidate, float]]:
    output: list[tuple[Candidate, float]] = []
    with args.candidate_scores.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"fold", "sample_id", "case_id", args.parent_score_field}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"candidate score file missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["fold"]) != args.fold:
                continue
            prior = float(row[args.parent_score_field])
            if not np.isfinite(prior) or prior < args.min_parent_prior:
                continue
            match = COORDINATE_RE.search(row["sample_id"])
            if match is None:
                raise ValueError(f"cannot parse candidate coordinate: {row['sample_id']}")
            center = np.asarray([float(value) for value in match.groups()]) * SPACING_ZYX
            stem, frame_text = row["case_id"].rsplit("_t", 1)
            output.append((Candidate(
                sample_id=row["sample_id"],
                source="detector",
                split=str(args.fold),
                case_id=row["case_id"],
                stem=stem,
                frame=int(frame_text),
                frame_width=len(frame_text),
                center_um=center,
                label=0,
                category="unknown",
                negative_subtype="unknown",
            ), prior))
    return output


def required_cases(candidates: list[tuple[Candidate, float]]) -> set[str]:
    output: set[str] = set()
    for candidate, _ in candidates:
        for offset in range(-1, 5):
            case_id = make_case_id(
                candidate.stem, candidate.frame + offset, candidate.frame_width
            )
            if case_id is not None:
                output.add(case_id)
    return output


def train_ranker(
    path: Path,
    target_fold: int,
    seed: int,
    n_jobs: int,
) -> tuple[Any, list[str], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feature_names = sorted(
        name for name in rows[0] if name.startswith(("geom_", "traj_"))
    )
    training = [
        row for row in rows
        if int(row["fold"]) != target_fold and int(row["parent_detector_detected"])
    ]
    labels = np.asarray([int(row["label"]) for row in training], dtype=np.int8)
    if len(np.unique(labels)) != 2:
        raise RuntimeError("pair training rows do not contain both labels")
    matrix = np.asarray([
        [float(row[name]) for name in feature_names] for row in training
    ], dtype=np.float64)
    model = model_factories(seed, n_jobs)["extra_trees"]()
    model.fit(matrix, labels)
    return model, feature_names, {
        "training_rows": len(training),
        "training_positive_rows": int(labels.sum()),
    }


def enumerate_candidate_pairs(
    candidate: Candidate,
    parent_prior: float,
    builder: PhysicalFeatureBuilder,
    voxel_points: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool]:
    source = builder.cloud(candidate.stem, candidate.frame, candidate.frame_width)
    if not len(source):
        return [], False
    distances = np.linalg.norm(source - candidate.center_um[None, :], axis=1)
    parent_index = int(np.argmin(distances))
    if float(distances[parent_index]) > args.parent_match_um:
        return [], False
    center = source[parent_index]
    target_frame = candidate.frame + 1
    target_case = make_case_id(candidate.stem, target_frame, candidate.frame_width)
    if target_case is None:
        return [], True
    points = builder.cloud(candidate.stem, target_frame, candidate.frame_width)
    if len(points) < 2:
        return [], True
    shift, shift_available = builder.cumulative_shift(
        candidate.stem, candidate.frame, target_frame, candidate.frame_width
    )
    predicted = center + shift
    target_distances = np.linalg.norm(points - predicted[None, :], axis=1)
    order = np.argsort(target_distances, kind="stable")[: args.max_pair_points]
    within = order[target_distances[order] <= args.pair_radius_um]
    pool = within if len(within) >= 2 else order
    local_counts = {
        radius: float(np.count_nonzero(target_distances <= radius))
        for radius in (5.0, 7.0, 10.0, 15.0, 20.0)
    }
    target_voxels = voxel_points[target_case]
    source_voxels = voxel_points[candidate.case_id]
    output: list[dict[str, Any]] = []
    for first, second in combinations(pool.tolist(), 2):
        q1, q2 = points[first], points[second]
        v1, v2 = q1 - predicted, q2 - predicted
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        sister = float(np.linalg.norm(q1 - q2))
        midpoint = float(np.linalg.norm((q1 + q2) / 2.0 - predicted))
        denominator = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8)
        cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
        scores = builder.clouds.scores[target_case]
        score1, score2 = float(scores[first]), float(scores[second])
        heuristic = midpoint + 0.05 * (d1 + d2) + 0.02 * abs(sister - 10.5)
        features: dict[str, float] = {
            "geom_dt": 1.0,
            "geom_dt1": 1.0,
            "geom_dt2": 0.0,
            "geom_dt3": 0.0,
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
            "geom_point_rank_max": float(
                max(np.flatnonzero(order == first)[0], np.flatnonzero(order == second)[0]) + 1
            ),
            "geom_pair_count_dt": float(math.comb(len(pool), 2)),
        }
        for radius, count in local_counts.items():
            features[f"geom_count_r{int(radius)}"] = count
        pair = points[[first, second]]
        scoring_candidate = Candidate(
            **{**candidate.__dict__, "center_um": center}
        )
        for target_offset in (-1, 0, 1, 2, 3, 4):
            prefix = f"traj_t{target_offset:+d}".replace("+", "p").replace("-", "m")
            features.update(builder.assignment_features(
                scoring_candidate,
                pair,
                target_frame,
                candidate.frame + target_offset,
                prefix,
            ))
        parent_voxel = source_voxels[parent_index]
        first_voxel, second_voxel = target_voxels[[first, second]]
        output.append({
            "fold": args.fold,
            "sample_id": candidate.sample_id,
            "case_id": candidate.case_id,
            "target_case_id": target_case,
            "stem": candidate.stem,
            "parent_prior": parent_prior,
            "first": int(first),
            "second": int(second),
            "parent_z": float(parent_voxel[0]),
            "parent_y": float(parent_voxel[1]),
            "parent_x": float(parent_voxel[2]),
            "daughter1_z": float(first_voxel[0]),
            "daughter1_y": float(first_voxel[1]),
            "daughter1_x": float(first_voxel[2]),
            "daughter2_z": float(second_voxel[0]),
            "daughter2_y": float(second_voxel[1]),
            "daughter2_x": float(second_voxel[2]),
            **{name: finite(value) for name, value in features.items()},
        })
    return output, True


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    candidates = read_target_candidates(args)
    model, feature_names, training_summary = train_ranker(
        args.training_pair_scores, args.fold, args.seed, args.n_jobs
    )
    peaks = load_peaks([args.detector_peaks], required_cases(candidates))
    voxel_points = {
        case_id: np.asarray([peak.coordinate for peak in rows], dtype=np.float64).reshape(-1, 3)
        for case_id, rows in peaks.items()
    }
    coordinates = {
        case_id: values * SPACING_ZYX for case_id, values in voxel_points.items()
    }
    scores = {
        case_id: np.asarray([peak.score for peak in rows], dtype=np.float64)
        for case_id, rows in peaks.items()
    }
    roles = {case_id: np.zeros(len(values), dtype=np.int8) for case_id, values in coordinates.items()}
    builder = PhysicalFeatureBuilder(
        PointClouds(coordinates=coordinates, scores=scores, roles=roles),
        args.pair_radius_um,
        args.max_pair_points,
    )
    rows: list[dict[str, Any]] = []
    matched_candidates = 0
    affected_stems: set[str] = set()
    for candidate, prior in candidates:
        candidate_rows, matched = enumerate_candidate_pairs(
            candidate, prior, builder, voxel_points, args
        )
        matched_candidates += int(matched)
        if matched:
            affected_stems.add(candidate.stem)
        rows.extend(candidate_rows)
    if rows:
        matrix = np.asarray([
            [float(row[name]) for name in feature_names] for row in rows
        ], dtype=np.float64)
        probabilities = model.predict_proba(matrix)[:, 1]
        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row["sample_id"]), []).append(index)
        for indices in grouped.values():
            order = sorted(indices, key=lambda index: (-float(probabilities[index]), index))
            denominator = max(1, len(order) - 1)
            for rank, index in enumerate(order):
                rows[index]["pair_score"] = float(probabilities[index])
                rows[index]["pair_rank_norm"] = 1.0 - rank / denominator
                rows[index]["pair_rank"] = rank + 1
    write_csv(args.output_dir / "division_pair_scores.csv", rows)
    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold,
        "parent_score_field": args.parent_score_field,
        "min_parent_prior": args.min_parent_prior,
        "target_candidates": len(candidates),
        "matched_target_candidates": matched_candidates,
        "pair_rows": len(rows),
        "affected_stems": sorted(affected_stems),
        "feature_count": len(feature_names),
        "training_policy": "other-fold true-Parent pair rows only; target-fold labels unread",
        **training_summary,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

