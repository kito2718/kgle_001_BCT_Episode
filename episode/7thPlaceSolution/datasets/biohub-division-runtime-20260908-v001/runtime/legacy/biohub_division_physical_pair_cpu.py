#!/usr/bin/env python3
"""CPU-only physical Division classifier on GT centres and detector peaks.

The model never reads image crops. It ranks candidate parent nodes from point-
cloud geometry, backward pair convergence, and forward pair persistence. GT
roles are used only for targets and audit labels, never as input features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from biohub_division_crop_classifier import (
    DAUGHTER_ROLES,
    SPACING_ZYX,
    Sample,
    atomic_json,
    build_samples,
    write_csv,
)
from biohub_division_detected_geometry_classifier import (
    CandidateProvenance,
    Peak,
    build_labelable_candidates,
    detector_geometry_raw,
    load_peaks,
    match_peaks_to_roles,
    select_negatives,
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    source: str
    split: str
    case_id: str
    stem: str
    frame: int
    frame_width: int
    center_um: np.ndarray
    label: int
    category: str
    negative_subtype: str


@dataclass
class PointClouds:
    coordinates: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    roles: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dataset-dir", type=Path, required=True)
    parser.add_argument("--role-lookup", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--pair-radius-um", type=float, default=15.0)
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--max-pair-points", type=int, default=8)
    parser.add_argument("--train-normal-per-parent", type=int, default=8)
    parser.add_argument("--val-normal-per-parent", type=int, default=16)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--sources", default="gt,detector")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--limit-stems-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_parts(case_id: str) -> tuple[str, int, int]:
    stem, frame_text = case_id.rsplit("_t", 1)
    return stem, int(frame_text), len(frame_text)


def make_case_id(stem: str, frame: int, width: int) -> str | None:
    if frame < 0:
        return None
    return f"{stem}_t{frame:0{width}d}"


def required_case_ids(candidates: Sequence[Candidate]) -> set[str]:
    output = set()
    for candidate in candidates:
        for offset in range(-2, 4):
            case_id = make_case_id(candidate.stem, candidate.frame + offset, candidate.frame_width)
            if case_id is not None:
                output.add(case_id)
    return output


def finite(value: float, fill: float = 30.0) -> float:
    return float(value) if math.isfinite(float(value)) else fill


def gt_candidates(
    roles: dict[str, list[list[int]]],
    train_cases: set[str],
    val_cases: set[str],
    args: argparse.Namespace,
) -> tuple[list[Candidate], PointClouds, dict[str, Any]]:
    train = build_samples(roles, train_cases, "train", args.train_normal_per_parent, False)
    val = build_samples(roles, val_cases, "val", args.val_normal_per_parent, False)
    samples = train + val
    candidates: list[Candidate] = []
    for sample in samples:
        stem, frame, width = case_parts(sample.case_id)
        candidates.append(Candidate(
            sample_id=sample.sample_id,
            source="gt",
            split=sample.split,
            case_id=sample.case_id,
            stem=stem,
            frame=frame,
            frame_width=width,
            center_um=np.asarray((sample.z, sample.y, sample.x), dtype=np.float64) * SPACING_ZYX,
            label=sample.label,
            category=sample.category,
            negative_subtype="positive" if sample.label else sample.category,
        ))
    coordinates, scores, point_roles = {}, {}, {}
    needed = required_case_ids(candidates)
    for case_id in needed:
        rows = roles.get(case_id, [])
        coordinates[case_id] = np.asarray([row[:3] for row in rows], dtype=np.float64).reshape(-1, 3) * SPACING_ZYX
        scores[case_id] = np.ones(len(rows), dtype=np.float64)
        point_roles[case_id] = np.asarray([int(row[3]) for row in rows], dtype=np.int16)
    return candidates, PointClouds(coordinates, scores, point_roles), {
        "train_samples": len(train), "val_samples": len(val),
        "train_categories": dict(Counter(x.category for x in candidates if x.split == "train")),
        "val_categories": dict(Counter(x.category for x in candidates if x.split == "val")),
    }


def _detector_point_roles(
    roles: dict[str, list[list[int]]],
    peaks_by_case: dict[str, list[Peak]],
    radius_um: float,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for case_id, peaks in peaks_by_case.items():
        values = np.zeros(len(peaks), dtype=np.int16)
        pairs, _ = match_peaks_to_roles(peaks, roles.get(case_id, []), radius_um)
        rows = roles.get(case_id, [])
        for pred_index, gt_index, _ in pairs:
            values[pred_index] = int(rows[gt_index][3])
        output[case_id] = values
    return output


def detector_candidates(
    roles: dict[str, list[list[int]]],
    train_cases: set[str],
    val_cases: set[str],
    peaks_by_case: dict[str, list[Peak]],
    args: argparse.Namespace,
) -> tuple[list[Candidate], PointClouds, dict[str, Any]]:
    train_all, train_prov, train_avail = build_labelable_candidates(
        roles, peaks_by_case, train_cases, "train", args.match_radius_um, args.pair_radius_um
    )
    val_all, val_prov, val_avail = build_labelable_candidates(
        roles, peaks_by_case, val_cases, "val", args.match_radius_um, args.pair_radius_um
    )
    all_samples = train_all + val_all
    all_prov = {**train_prov, **val_prov}
    raw, names = detector_geometry_raw(all_samples, all_prov, peaks_by_case)
    selected_train, train_prov = select_negatives(
        train_all, train_prov, raw[: len(train_all)], names,
        args.train_normal_per_parent, 2, args.hard_negative_fraction, args.seed,
    )
    selected_val_local, val_prov = select_negatives(
        val_all, val_prov, raw[len(train_all):], names,
        args.val_normal_per_parent, 4, args.hard_negative_fraction, args.seed + 1,
    )
    all_prov = {**train_prov, **val_prov}
    selected = selected_train + [len(train_all) + i for i in selected_val_local]
    candidates: list[Candidate] = []
    for index in selected:
        sample = all_samples[index]
        prov: CandidateProvenance = all_prov[sample.sample_id]
        stem, frame, width = case_parts(sample.case_id)
        _, z_key, y_key, x_key = prov.peak_key
        candidates.append(Candidate(
            sample_id=sample.sample_id,
            source="detector",
            split=sample.split,
            case_id=sample.case_id,
            stem=stem,
            frame=frame,
            frame_width=width,
            center_um=np.asarray((z_key, y_key, x_key), dtype=np.float64) / 1000.0 * SPACING_ZYX,
            label=sample.label,
            category=sample.category,
            negative_subtype=all_prov[sample.sample_id].negative_subtype,
        ))
    needed = required_case_ids(candidates)
    selected_clouds = {case_id: peaks_by_case.get(case_id, []) for case_id in needed}
    coordinates = {
        case_id: np.asarray([p.coordinate for p in peaks], dtype=np.float64).reshape(-1, 3) * SPACING_ZYX
        for case_id, peaks in selected_clouds.items()
    }
    scores = {case_id: np.asarray([p.score for p in peaks], dtype=np.float64) for case_id, peaks in selected_clouds.items()}
    point_roles = _detector_point_roles(roles, selected_clouds, args.match_radius_um)
    return candidates, PointClouds(coordinates, scores, point_roles), {
        "train_availability": train_avail,
        "val_availability": val_avail,
        "train_categories": dict(Counter(x.category for x in candidates if x.split == "train")),
        "val_categories": dict(Counter(x.category for x in candidates if x.split == "val")),
        "negative_subtypes": dict(Counter(x.negative_subtype for x in candidates)),
        "caveat": "detector provenance depends on the supplied peak files; inspect detector_peaks in the run summary",
    }


class PhysicalFeatureBuilder:
    def __init__(self, clouds: PointClouds, pair_radius_um: float, max_pair_points: int):
        self.clouds = clouds
        self.pair_radius_um = pair_radius_um
        self.max_pair_points = max_pair_points
        self.shift_cache: dict[tuple[str, int, int], tuple[np.ndarray, int]] = {}

    def cloud(self, stem: str, frame: int, width: int) -> np.ndarray:
        case_id = make_case_id(stem, frame, width)
        if case_id is None:
            return np.empty((0, 3), dtype=np.float64)
        return self.clouds.coordinates.get(case_id, np.empty((0, 3), dtype=np.float64))

    def roles(self, stem: str, frame: int, width: int) -> np.ndarray:
        case_id = make_case_id(stem, frame, width)
        if case_id is None:
            return np.empty(0, dtype=np.int16)
        return self.clouds.roles.get(case_id, np.empty(0, dtype=np.int16))

    def adjacent_shift(self, stem: str, frame: int, width: int) -> tuple[np.ndarray, int]:
        key = (stem, frame, width)
        if key in self.shift_cache:
            return self.shift_cache[key]
        source, target = self.cloud(stem, frame, width), self.cloud(stem, frame + 1, width)
        if len(source) < 3 or len(target) < 3:
            result = (np.zeros(3, dtype=np.float64), 0)
            self.shift_cache[key] = result
            return result
        shift = np.zeros(3, dtype=np.float64)
        used = 0
        target_tree = cKDTree(target)
        for _ in range(3):
            distances, indices = target_tree.query(source + shift[None, :], k=1)
            valid = distances <= 20.0
            if np.count_nonzero(valid) < 3:
                break
            differences = target[indices[valid]] - source[valid]
            median = np.median(differences, axis=0)
            residual = np.linalg.norm(differences - median[None, :], axis=1)
            cutoff = max(2.0, float(np.quantile(residual, 0.70)))
            keep = residual <= cutoff
            if np.count_nonzero(keep) < 3:
                break
            shift = np.median(differences[keep], axis=0)
            used = int(np.count_nonzero(keep))
        result = (shift.astype(np.float64), used)
        self.shift_cache[key] = result
        return result

    def cumulative_shift(self, stem: str, source_frame: int, target_frame: int, width: int) -> tuple[np.ndarray, int]:
        shift = np.zeros(3, dtype=np.float64)
        availability = 1
        if target_frame > source_frame:
            for frame in range(source_frame, target_frame):
                item, available = self.adjacent_shift(stem, frame, width)
                shift += item
                availability &= int(available > 0)
        elif target_frame < source_frame:
            for frame in range(target_frame, source_frame):
                item, available = self.adjacent_shift(stem, frame, width)
                shift -= item
                availability &= int(available > 0)
        return shift, availability

    @staticmethod
    def distances(points: np.ndarray, center: np.ndarray, exclude_center: bool = False) -> np.ndarray:
        if not len(points):
            return np.empty(0, dtype=np.float64)
        distances = np.linalg.norm(points - center[None, :], axis=1)
        if exclude_center:
            distances = distances[distances > 0.25]
        return np.sort(distances)

    def select_pair(self, candidate: Candidate) -> tuple[np.ndarray | None, tuple[int, int] | None, dict[str, float]]:
        shift, shift_available = self.cumulative_shift(
            candidate.stem, candidate.frame, candidate.frame + 1, candidate.frame_width
        )
        predicted = candidate.center_um + shift
        points = self.cloud(candidate.stem, candidate.frame + 1, candidate.frame_width)
        if len(points) < 2:
            return None, None, {
                "t1_pair_available": 0.0, "t1_pair_candidate_count": 0.0,
                "t1_global_available": float(shift_available),
                "t1_global_dz": shift[0], "t1_global_dy": shift[1], "t1_global_dx": shift[2],
                "t1_global_norm": float(np.linalg.norm(shift)),
                "t1_pair_d1": 30.0, "t1_pair_d2": 30.0, "t1_pair_sister": 30.0,
                "t1_pair_midpoint": 30.0, "t1_pair_cosine": 0.0, "t1_pair_selection_score": 60.0,
            }
        distances = np.linalg.norm(points - predicted[None, :], axis=1)
        order = np.argsort(distances, kind="stable")[: self.max_pair_points]
        within = order[distances[order] <= self.pair_radius_um]
        pool = within if len(within) >= 2 else order
        pair_rows = []
        for first, second in combinations(pool.tolist(), 2):
            q1, q2 = points[first], points[second]
            v1, v2 = q1 - predicted, q2 - predicted
            d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            sister = float(np.linalg.norm(q1 - q2))
            midpoint = float(np.linalg.norm((q1 + q2) / 2.0 - predicted))
            denominator = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8)
            cosine = float(np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0))
            score = midpoint + 0.05 * (d1 + d2) + 0.02 * abs(sister - 10.5)
            pair_rows.append((score, midpoint, max(d1, d2), first, second, d1, d2, sister, cosine))
        selected = min(pair_rows)
        score, midpoint, _, first, second, d1, d2, sister, cosine = selected
        return points[[first, second]], (first, second), {
            "t1_pair_available": float(len(within) >= 2),
            "t1_pair_candidate_count": float(len(pair_rows)),
            "t1_global_available": float(shift_available),
            "t1_global_dz": shift[0], "t1_global_dy": shift[1], "t1_global_dx": shift[2],
            "t1_global_norm": float(np.linalg.norm(shift)),
            "t1_pair_d1": min(d1, d2), "t1_pair_d2": max(d1, d2),
            "t1_pair_sister": sister, "t1_pair_midpoint": midpoint,
            "t1_pair_cosine": cosine, "t1_pair_selection_score": score,
        }

    def assignment_features(
        self,
        candidate: Candidate,
        pair: np.ndarray | None,
        pair_frame: int,
        target_frame: int,
        prefix: str,
    ) -> dict[str, float]:
        defaults = {
            f"{prefix}_available": 0.0, f"{prefix}_common_mean": 30.0,
            f"{prefix}_distinct_mean": 30.0, f"{prefix}_distinct_max": 30.0,
            f"{prefix}_distinct_count7": 0.0, f"{prefix}_distinct_minus_common": 0.0,
            f"{prefix}_same_nearest": 0.0, f"{prefix}_assigned_sister": 30.0,
        }
        target = self.cloud(candidate.stem, target_frame, candidate.frame_width)
        if pair is None or not len(target):
            return defaults
        shift, shift_available = self.cumulative_shift(
            candidate.stem, pair_frame, target_frame, candidate.frame_width
        )
        projected = pair + shift[None, :]
        matrix = np.linalg.norm(projected[:, None, :] - target[None, :, :], axis=2)
        nearest = np.argmin(matrix, axis=1)
        common_costs = matrix.mean(axis=0)
        common_mean = float(common_costs.min())
        if len(target) >= 2:
            rows, columns = linear_sum_assignment(matrix)
            assigned = matrix[rows, columns]
            distinct_mean = float(assigned.mean())
            distinct_max = float(assigned.max())
            distinct_count = float(np.count_nonzero(assigned <= 7.0))
            assigned_sister = float(np.linalg.norm(target[columns[0]] - target[columns[1]]))
        else:
            distinct_mean = distinct_max = 30.0
            distinct_count = 0.0
            assigned_sister = 30.0
        return {
            f"{prefix}_available": float(shift_available),
            f"{prefix}_common_mean": common_mean,
            f"{prefix}_distinct_mean": distinct_mean,
            f"{prefix}_distinct_max": distinct_max,
            f"{prefix}_distinct_count7": distinct_count,
            f"{prefix}_distinct_minus_common": distinct_mean - common_mean,
            f"{prefix}_same_nearest": float(nearest[0] == nearest[1]),
            f"{prefix}_assigned_sister": assigned_sister,
        }

    def build(self, candidate: Candidate) -> tuple[dict[str, float], dict[str, Any]]:
        features: dict[str, float] = {}
        current = self.cloud(candidate.stem, candidate.frame, candidate.frame_width)
        next_points = self.cloud(candidate.stem, candidate.frame + 1, candidate.frame_width)
        shift, _ = self.cumulative_shift(candidate.stem, candidate.frame, candidate.frame + 1, candidate.frame_width)
        next_center = candidate.center_um + shift
        current_dist = self.distances(current, candidate.center_um, exclude_center=True)
        next_dist = self.distances(next_points, next_center)
        radii = (5.0, 7.0, 10.0, 15.0)
        for index in range(3):
            features[f"t1_current_knn{index+1}"] = float(current_dist[index]) if index < len(current_dist) else 30.0
        for index in range(4):
            features[f"t1_next_knn{index+1}"] = float(next_dist[index]) if index < len(next_dist) else 30.0
        for radius in radii:
            current_count = float(np.count_nonzero(current_dist <= radius))
            next_count = float(np.count_nonzero(next_dist <= radius))
            features[f"t1_current_count_r{int(radius)}"] = current_count
            features[f"t1_next_count_r{int(radius)}"] = next_count
            features[f"t1_count_delta_r{int(radius)}"] = next_count - current_count

        pair, pair_indices, pair_features = self.select_pair(candidate)
        features.update(pair_features)
        for offset in (0, -1, -2):
            features.update(self.assignment_features(
                candidate, pair, candidate.frame + 1, candidate.frame + offset, f"hist_t{offset}"
            ))
        for offset in (2, 3):
            features.update(self.assignment_features(
                candidate, pair, candidate.frame + 1, candidate.frame + offset, f"future_tplus{offset}"
            ))
        features["future_sister_delta_t2"] = features["future_tplus2_assigned_sister"] - features["t1_pair_sister"]
        features["future_sister_delta_t3"] = features["future_tplus3_assigned_sister"] - features["t1_pair_sister"]

        selected_roles = []
        if pair_indices is not None:
            roles = self.roles(candidate.stem, candidate.frame + 1, candidate.frame_width)
            if len(roles) > max(pair_indices):
                selected_roles = [int(roles[pair_indices[0]]), int(roles[pair_indices[1]])]
        audit = {
            "selected_pair_roles": "|".join(str(x) for x in selected_roles),
            "selected_pair_both_daughter": int(
                len(selected_roles) == 2 and all(role in DAUGHTER_ROLES for role in selected_roles)
            ),
        }
        return {name: finite(value) for name, value in features.items()}, audit


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    threshold = float(thresholds[min(best, len(thresholds) - 1)]) if len(thresholds) else 0.5
    output = {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "best_f1": float(f1[best]), "best_precision": float(precision[best]),
        "best_recall": float(recall[best]), "best_threshold": threshold,
    }
    for target in (0.5, 0.8):
        eligible = np.flatnonzero(precision >= target)
        output[f"recall_at_precision_{target}"] = float(recall[eligible].max()) if len(eligible) else 0.0
    return output


def fit_models(
    source: str,
    candidates: Sequence[Candidate],
    feature_rows: Sequence[dict[str, float]],
    output_dir: Path,
    seed: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    feature_names = sorted(feature_rows[0])
    matrix = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float64)
    labels = np.asarray([candidate.label for candidate in candidates], dtype=np.int64)
    train = np.asarray([candidate.split == "train" for candidate in candidates])
    val = ~train
    sets = {
        "t1_pair": [i for i, name in enumerate(feature_names) if name.startswith("t1_")],
        "t1_pair_history": [i for i, name in enumerate(feature_names) if name.startswith(("t1_", "hist_"))],
        "t1_pair_history_future": list(range(len(feature_names))),
    }
    model_factories = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs,
        ),
    }
    results: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for feature_set, columns in sets.items():
        names = [feature_names[i] for i in columns]
        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(matrix[train][:, columns], labels[train])
            scores = model.predict_proba(matrix[val][:, columns])[:, 1]
            overall = binary_metrics(labels[val], scores)
            slices = {}
            val_candidates = [candidate for candidate in candidates if candidate.split == "val"]
            categories = np.asarray([candidate.category for candidate in val_candidates])
            subtypes = np.asarray([candidate.negative_subtype for candidate in val_candidates])
            for slice_name, mask in {
                "parent_vs_daughter": (labels[val] == 1) | (categories == "daughter"),
                "parent_vs_normal": (labels[val] == 1) | (categories == "normal"),
                "parent_vs_hard_pair_normal": (labels[val] == 1) | np.char.startswith(subtypes.astype(str), "hard_pair_normal"),
            }.items():
                if np.count_nonzero(mask) and len(np.unique(labels[val][mask])) == 2:
                    slices[slice_name] = binary_metrics(labels[val][mask], scores[mask])
            results.append({
                "source": source, "feature_set": feature_set, "model": model_name,
                "metrics": overall, "slices": slices, "feature_count": len(columns),
            })
            for candidate, score in zip(val_candidates, scores):
                score_rows.append({
                    "source": source, "feature_set": feature_set, "model": model_name,
                    "sample_id": candidate.sample_id, "case_id": candidate.case_id,
                    "category": candidate.category, "negative_subtype": candidate.negative_subtype,
                    "label": candidate.label, "score": float(score),
                })
            if model_name == "extra_trees":
                values = model.feature_importances_
            else:
                values = np.abs(model[-1].coef_[0])
            for name, value in sorted(zip(names, values), key=lambda item: -item[1]):
                importance_rows.append({
                    "source": source, "feature_set": feature_set, "model": model_name,
                    "feature": name, "importance": float(value),
                })
    write_csv(output_dir / source / "validation_scores.csv", score_rows)
    write_csv(output_dir / source / "feature_importance.csv", importance_rows)
    return results


def main() -> int:
    args = parse_args()
    if args.overwrite and args.output_dir.exists():
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.role_lookup.read_text())
    roles: dict[str, list[list[int]]] = payload.get("cases", payload)
    splits = json.loads((args.preprocessed_dataset_dir / "splits_final.json").read_text())
    split = splits[args.fold]
    train_cases, val_cases = set(split["train"]), set(split["val"])
    if args.limit_stems_per_split:
        def select_stems(case_ids: set[str]) -> set[str]:
            parent_stems = []
            for case_id in sorted(case_ids):
                if any(int(row[3]) in {2, 4} for row in roles.get(case_id, [])):
                    stem, _, _ = case_parts(case_id)
                    if stem not in parent_stems:
                        parent_stems.append(stem)
            chosen = set(parent_stems[: args.limit_stems_per_split])
            return {case_id for case_id in case_ids if case_parts(case_id)[0] in chosen}
        train_cases, val_cases = select_stems(train_cases), select_stems(val_cases)
    all_cases = train_cases | val_cases
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = set(sources) - {"gt", "detector"}
    if invalid:
        raise ValueError(f"unsupported sources: {sorted(invalid)}")
    peaks_by_case = load_peaks(args.detector_peaks, all_cases) if "detector" in sources else {}
    all_results, source_metadata = [], {}
    started = time.perf_counter()
    for source in sources:
        if source == "gt":
            candidates, clouds, metadata = gt_candidates(roles, train_cases, val_cases, args)
        else:
            candidates, clouds, metadata = detector_candidates(
                roles, train_cases, val_cases, peaks_by_case, args
            )
        if args.limit_per_split:
            limited = []
            for split_name in ("train", "val"):
                partition = [x for x in candidates if x.split == split_name]
                for category in ("parent", "daughter", "normal"):
                    limited.extend([x for x in partition if x.category == category][: args.limit_per_split])
            candidates = limited
        builder = PhysicalFeatureBuilder(clouds, args.pair_radius_um, args.max_pair_points)
        feature_rows, audit_rows = [], []
        for index, candidate in enumerate(candidates, start=1):
            features, audit = builder.build(candidate)
            feature_rows.append(features)
            audit_rows.append({
                "source": source, "sample_id": candidate.sample_id, "split": candidate.split,
                "case_id": candidate.case_id, "label": candidate.label,
                "category": candidate.category, "negative_subtype": candidate.negative_subtype,
                **audit, **features,
            })
            if index % 250 == 0:
                print(f"[{source}] features {index}/{len(candidates)}", flush=True)
        (args.output_dir / source).mkdir(exist_ok=True)
        write_csv(args.output_dir / source / "physical_features.csv", audit_rows)
        results = fit_models(source, candidates, feature_rows, args.output_dir, args.seed, args.n_jobs)
        all_results.extend(results)
        metadata["selected_pair_both_daughter_parent_rate"] = (
            sum(row["selected_pair_both_daughter"] for row in audit_rows if row["label"])
            / max(1, sum(row["label"] for row in audit_rows))
        )
        metadata["global_shift_cache_entries"] = len(builder.shift_cache)
        source_metadata[source] = metadata
    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold, "sources": sources, "seed": args.seed,
        "pair_radius_um": args.pair_radius_um,
        "feature_policy": "point coordinates and point-cloud registration only; roles used only as targets/audit",
        "results": all_results, "source_metadata": source_metadata,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "metrics.csv", [
        {
            "source": row["source"], "feature_set": row["feature_set"], "model": row["model"],
            "feature_count": row["feature_count"], **row["metrics"],
        }
        for row in all_results
    ])
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
