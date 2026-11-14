#!/usr/bin/env python3
"""Train a division-parent classifier with detector-derived 1-to-2 geometry.

GT annotations are used only to label detector peaks.  Every geometry feature
is computed from detector peaks, including UNKNOWN-region peaks.  UNKNOWN peaks
are never assigned a negative target because Biohub supervision is sparse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_division_crop_classifier import (
    DAUGHTER_ROLES,
    PARENT_ROLES,
    SPACING_ZYX,
    Sample,
    atomic_json,
    binary_metrics,
    build_crop_cache,
    plot_curves,
    save_reviews,
    train_mode,
    write_csv,
)


@dataclass(frozen=True)
class Peak:
    case_id: str
    z: float
    y: float
    x: float
    score: float
    status: str
    source: str
    row_id: int

    @property
    def coordinate(self) -> np.ndarray:
        return np.asarray((self.z, self.y, self.x), dtype=np.float64)

    @property
    def key(self) -> tuple[str, int, int, int]:
        return self.case_id, round(self.z * 1000), round(self.y * 1000), round(self.x * 1000)


@dataclass(frozen=True)
class CandidateProvenance:
    sample_id: str
    peak_key: tuple[str, int, int, int]
    source_category: str
    negative_subtype: str
    detector_score: float
    matched_role: int | None
    matched_distance_um: float | None
    hard_pair_selected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dataset-dir", type=Path, required=True)
    parser.add_argument("--data-identifier", default="nnUNetPlans_3d_fullres")
    parser.add_argument("--role-lookup", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--crop-shape", default="12,48,48")
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--pair-radius-um", type=float, default=15.0)
    parser.add_argument("--train-normal-per-parent", type=int, default=8)
    parser.add_argument("--val-normal-per-parent", type=int, default=16)
    parser.add_argument("--train-fp-per-parent", type=int, default=2)
    parser.add_argument("--val-fp-per-parent", type=int, default=4)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--image-modes", default="t3,t3_geometry")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_triplet(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"invalid crop shape: {text}")
    return values


def case_parts(case_id: str) -> tuple[str, int]:
    stem, frame = case_id.rsplit("_t", 1)
    return stem, int(frame)


def neighbor_case_id(case_id: str, offset: int) -> str:
    stem, frame_text = case_id.rsplit("_t", 1)
    frame = max(0, int(frame_text) + offset)
    return f"{stem}_t{frame:0{len(frame_text)}d}"


def _status_rank(status: str) -> int:
    return {"FP": 3, "TP": 2, "UNKNOWN": 1}.get(status.upper(), 0)


def load_peaks(paths: Sequence[Path], allowed_cases: set[str]) -> dict[str, list[Peak]]:
    """Stream one or more peak CSVs and de-duplicate exact coordinates."""

    merged: dict[tuple[str, int, int, int], Peak] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"case_id", "z", "y", "x", "score"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            for row_id, row in enumerate(reader):
                case_id = row["case_id"]
                if case_id not in allowed_cases:
                    continue
                peak = Peak(
                    case_id=case_id,
                    z=float(row["z"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                    score=float(row["score"]),
                    status=(row.get("status") or "UNKNOWN").upper(),
                    source=str(path),
                    row_id=row_id,
                )
                previous = merged.get(peak.key)
                if previous is None or (_status_rank(peak.status), peak.score) > (
                    _status_rank(previous.status), previous.score
                ):
                    merged[peak.key] = peak
    by_case: dict[str, list[Peak]] = defaultdict(list)
    for peak in merged.values():
        by_case[peak.case_id].append(peak)
    for case_id in by_case:
        by_case[case_id].sort(key=lambda peak: (-peak.score, peak.z, peak.y, peak.x))
    return dict(by_case)


def match_peaks_to_roles(
    peaks: Sequence[Peak], rows: Sequence[Sequence[int]], radius_um: float
) -> tuple[list[tuple[int, int, float]], set[int]]:
    """Maximum-cardinality-like GT-to-peak assignment with per-GT dummies."""

    if not peaks or not rows:
        return [], set()
    predicted = np.asarray([(p.z, p.y, p.x) for p in peaks], dtype=np.float64)
    expected = np.asarray([row[:3] for row in rows], dtype=np.float64)
    distances = np.linalg.norm(
        (expected[:, None, :] - predicted[None, :, :]) * SPACING_ZYX[None, None, :], axis=2
    )
    n_gt, n_pred = distances.shape
    invalid = max(radius_um * 1000.0, 1e6)
    cost = np.full((n_gt, n_pred + n_gt), invalid, dtype=np.float64)
    cost[:, :n_pred] = np.where(distances <= radius_um, distances, invalid)
    for gt_index in range(n_gt):
        cost[gt_index, n_pred + gt_index] = radius_um + 1e-3
    gt_indices, columns = linear_sum_assignment(cost)
    pairs = [
        (int(column), int(gt_index), float(distances[gt_index, column]))
        for gt_index, column in zip(gt_indices, columns)
        if column < n_pred and distances[gt_index, column] <= radius_um
    ]
    return pairs, {pred_index for pred_index, _, _ in pairs}


def make_detector_sample(split: str, peak: Peak, role: int | None, source_category: str) -> Sample:
    stem, frame = case_parts(peak.case_id)
    is_parent = role in PARENT_ROLES if role is not None else False
    category = "parent" if is_parent else "daughter" if source_category == "daughter" else "normal"
    role_value = int(role) if role is not None else 0
    return Sample(
        sample_id=(
            f"{split}__det__{peak.case_id}__z{peak.z:.3f}_y{peak.y:.3f}_x{peak.x:.3f}"
            f"_r{role_value}_{source_category}"
        ),
        split=split,
        case_id=peak.case_id,
        stem=stem,
        frame=frame,
        z=int(round(peak.z)),
        y=int(round(peak.y)),
        x=int(round(peak.x)),
        role=role_value,
        label=int(is_parent),
        category=category,
        matched_parent_id="",
    )


def build_labelable_candidates(
    roles: dict[str, list[list[int]]],
    peaks_by_case: dict[str, list[Peak]],
    cases: Iterable[str],
    split: str,
    match_radius_um: float,
    pair_radius_um: float,
) -> tuple[list[Sample], dict[str, CandidateProvenance], dict[str, Any]]:
    samples: list[Sample] = []
    provenance: dict[str, CandidateProvenance] = {}
    total_roles = Counter()
    matched_roles = Counter()
    parent_pair_available = 0
    matched_parent_pair_available = 0

    for case_id in sorted(cases):
        role_rows = roles.get(case_id, [])
        peaks = peaks_by_case.get(case_id, [])
        pairs, matched_peak_indices = match_peaks_to_roles(peaks, role_rows, match_radius_um)
        match_by_peak = {pred: (gt, distance) for pred, gt, distance in pairs}
        matched_gt_indices = {gt for _, gt, _ in pairs}
        for row in role_rows:
            role = int(row[3])
            total_roles["parent" if role in PARENT_ROLES else "daughter" if role in DAUGHTER_ROLES else "normal"] += 1

        for pred_index, (gt_index, distance) in match_by_peak.items():
            peak = peaks[pred_index]
            role = int(role_rows[gt_index][3])
            source_category = (
                "parent" if role in PARENT_ROLES else "daughter" if role in DAUGHTER_ROLES else "normal_continuation"
            )
            sample = make_detector_sample(split, peak, role, source_category)
            samples.append(sample)
            matched_roles[sample.category] += 1
            provenance[sample.sample_id] = CandidateProvenance(
                sample_id=sample.sample_id,
                peak_key=peak.key,
                source_category=source_category,
                negative_subtype="positive" if sample.label else source_category,
                detector_score=peak.score,
                matched_role=role,
                matched_distance_um=distance,
                hard_pair_selected=False,
            )

        for pred_index, peak in enumerate(peaks):
            if pred_index in matched_peak_indices or peak.status != "FP":
                continue
            sample = make_detector_sample(split, peak, None, "known_region_fp")
            samples.append(sample)
            provenance[sample.sample_id] = CandidateProvenance(
                sample_id=sample.sample_id,
                peak_key=peak.key,
                source_category="known_region_fp",
                negative_subtype="known_region_fp",
                detector_score=peak.score,
                matched_role=None,
                matched_distance_um=None,
                hard_pair_selected=False,
            )

        next_peaks = peaks_by_case.get(neighbor_case_id(case_id, 1), [])
        if next_peaks:
            next_coords = np.asarray([(p.z, p.y, p.x) for p in next_peaks], dtype=np.float64)
            for gt_index, row in enumerate(role_rows):
                if int(row[3]) not in PARENT_ROLES:
                    continue
                center = np.asarray(row[:3], dtype=np.float64)
                distances = np.linalg.norm((next_coords - center) * SPACING_ZYX[None, :], axis=1)
                pair_available = int(np.count_nonzero(distances <= pair_radius_um)) >= 2
                parent_pair_available += int(pair_available)
                if gt_index in matched_gt_indices:
                    matched_parent_pair_available += int(pair_available)

    total_parent = total_roles["parent"]
    availability = {
        "total_roles": dict(total_roles),
        "matched_roles": dict(matched_roles),
        "parent_peak_recall": matched_roles["parent"] / total_parent if total_parent else None,
        "pair_radius_um": pair_radius_um,
        "parent_two_next_peaks_within_pair_radius": parent_pair_available,
        "parent_two_next_peaks_within_pair_radius_rate": parent_pair_available / total_parent if total_parent else None,
        "matched_parent_and_two_next_peaks_within_pair_radius": matched_parent_pair_available,
        "known_region_fp_candidates": sum(p.source_category == "known_region_fp" for p in provenance.values()),
    }
    return samples, provenance, availability


def detector_geometry_raw(
    samples: Sequence[Sample],
    provenance: dict[str, CandidateProvenance],
    peaks_by_case: dict[str, list[Peak]],
) -> tuple[np.ndarray, list[str]]:
    radii = (5.0, 7.0, 10.0, 15.0)
    names = [
        *(f"prev_knn_{i}_um" for i in range(1, 4)),
        *(f"current_knn_{i}_um" for i in range(1, 4)),
        *(f"next_knn_{i}_um" for i in range(1, 5)),
        *(f"{time}_count_r{radius:g}" for time in ("prev", "current", "next") for radius in radii),
        "next_d1_um", "next_d2_um", "next_d2_minus_d1_um",
        "next_pair_sister_um", "next_pair_midpoint_residual_um", "next_pair_angle_cosine",
        *(f"next_minus_current_count_r{radius:g}" for radius in radii),
        "center_detector_score", "prev_nearest_score", "current_nearest_other_score",
        "next_nearest_score", "next_second_score", "next_mean_score_r15",
        "next_pair_available", "prev_available",
    ]
    output: list[list[float]] = []
    for sample in samples:
        center = np.asarray((sample.z, sample.y, sample.x), dtype=np.float64)
        peak_key = provenance[sample.sample_id].peak_key
        temporal: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for offset in (-1, 0, 1):
            case_id = neighbor_case_id(sample.case_id, offset)
            peaks = peaks_by_case.get(case_id, [])
            coords = np.asarray([(p.z, p.y, p.x) for p in peaks], dtype=np.float64) if peaks else np.empty((0, 3))
            scores = np.asarray([p.score for p in peaks], dtype=np.float64)
            vectors = (coords - center[None, :]) * SPACING_ZYX[None, :] if len(coords) else np.empty((0, 3))
            distances = np.linalg.norm(vectors, axis=1) if len(coords) else np.empty(0)
            if offset == 0 and len(peaks):
                keep = np.asarray([peak.key != peak_key for peak in peaks], dtype=bool)
                distances, vectors, scores = distances[keep], vectors[keep], scores[keep]
            order = np.argsort(distances, kind="stable")
            temporal.append((distances[order], vectors[order], scores[order]))

        def pad(values: np.ndarray, count: int, fill: float) -> list[float]:
            result = values[:count].tolist()
            return [float(value) for value in result] + [fill] * (count - len(result))

        prev, current, nxt = temporal
        features = [*pad(prev[0], 3, 30.0), *pad(current[0], 3, 30.0), *pad(nxt[0], 4, 30.0)]
        counts = [[float(np.count_nonzero(item[0] <= radius)) for radius in radii] for item in temporal]
        features.extend(value for group in counts for value in group)
        if len(nxt[0]) >= 2:
            first, second = nxt[1][0], nxt[1][1]
            d1, d2 = float(nxt[0][0]), float(nxt[0][1])
            sister = float(np.linalg.norm(first - second))
            midpoint = float(np.linalg.norm((first + second) / 2.0))
            denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        elif len(nxt[0]) == 1:
            d1, d2, sister, midpoint, cosine = float(nxt[0][0]), 30.0, 30.0, 30.0, 0.0
        else:
            d1, d2, sister, midpoint, cosine = 30.0, 30.0, 30.0, 30.0, 0.0
        features.extend((d1, d2, d2 - d1, sister, midpoint, cosine))
        features.extend(n - c for n, c in zip(counts[2], counts[1]))
        center_score = provenance[sample.sample_id].detector_score
        features.extend((
            center_score,
            float(prev[2][0]) if len(prev[2]) else 0.0,
            float(current[2][0]) if len(current[2]) else 0.0,
            float(nxt[2][0]) if len(nxt[2]) else 0.0,
            float(nxt[2][1]) if len(nxt[2]) >= 2 else 0.0,
            float(nxt[2][nxt[0] <= 15.0].mean()) if np.any(nxt[0] <= 15.0) else 0.0,
            float(len(nxt[0]) >= 2),
            float(len(prev[0]) >= 1),
        ))
        output.append(features)
    raw = np.asarray(output, dtype=np.float32)
    if raw.shape != (len(samples), len(names)):
        raise RuntimeError(f"detector geometry shape mismatch: {raw.shape}, expected {(len(samples), len(names))}")
    return raw, names


def select_negatives(
    samples: Sequence[Sample],
    provenance: dict[str, CandidateProvenance],
    raw: np.ndarray,
    names: Sequence[str],
    normal_per_parent: int,
    fp_per_parent: int,
    hard_fraction: float,
    seed: int,
) -> tuple[list[int], dict[str, CandidateProvenance]]:
    parent_indices = [i for i, sample in enumerate(samples) if sample.label]
    daughter_indices = [i for i, sample in enumerate(samples) if sample.category == "daughter"]
    normal_indices = [i for i, sample in enumerate(samples) if provenance[sample.sample_id].source_category == "normal_continuation"]
    fp_indices = [i for i, sample in enumerate(samples) if provenance[sample.sample_id].source_category == "known_region_fp"]
    name_to_index = {name: index for index, name in enumerate(names)}

    def rank(indices: Sequence[int]) -> list[int]:
        midpoint = name_to_index["next_pair_midpoint_residual_um"]
        d2 = name_to_index["next_d2_um"]
        delta = name_to_index["next_minus_current_count_r10"]
        return sorted(
            indices,
            key=lambda i: (
                raw[i, d2] > 15.0,
                raw[i, midpoint],
                raw[i, d2],
                -raw[i, delta],
                -provenance[samples[i].sample_id].detector_score,
            ),
        )

    rng = random.Random(seed)
    selected: list[int] = list(parent_indices) + list(daughter_indices)
    updated = dict(provenance)
    for pool, quota_per_parent in ((normal_indices, normal_per_parent), (fp_indices, fp_per_parent)):
        quota = min(len(pool), max(1, len(parent_indices)) * quota_per_parent)
        hard_count = min(quota, int(round(quota * hard_fraction)))
        ranked = rank(pool)
        chosen_hard = ranked[:hard_count]
        remainder = ranked[hard_count:]
        rng.shuffle(remainder)
        chosen = chosen_hard + remainder[: quota - hard_count]
        selected.extend(chosen)
        for index in chosen_hard:
            item = updated[samples[index].sample_id]
            updated[samples[index].sample_id] = CandidateProvenance(
                **{**asdict(item), "negative_subtype": f"hard_pair_{item.source_category}", "hard_pair_selected": True}
            )
    return sorted(set(selected)), updated


def normalize_geometry(raw: np.ndarray, train_indices: Sequence[int], output_dir: Path, names: Sequence[str]) -> np.ndarray:
    train = raw[np.asarray(train_indices, dtype=np.int64)]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = ((raw - mean) / std).astype(np.float32)
    np.save(output_dir / "cache" / "detector_geometry_raw.float32.npy", raw)
    np.save(output_dir / "cache" / "detector_geometry_normalized.float32.npy", normalized)
    atomic_json(output_dir / "cache" / "detector_geometry_config.json", {
        "feature_names": list(names), "train_mean": mean.tolist(), "train_std": std.tolist(),
        "source": "detector peaks only; GT used only for candidate labels",
    })
    return normalized


def subtype_metrics(
    result: dict[str, Any], samples: Sequence[Sample], provenance: dict[str, CandidateProvenance]
) -> dict[str, Any]:
    indices = np.asarray(result["final_indices"], dtype=np.int64)
    scores = np.asarray(result["final_scores"], dtype=np.float64)
    labels = np.asarray([samples[i].label for i in indices], dtype=bool)
    subtypes = np.asarray([provenance[samples[i].sample_id].negative_subtype for i in indices])
    output: dict[str, Any] = {}
    for subtype in sorted(set(subtypes) - {"positive"}):
        mask = labels | (subtypes == subtype)
        output[f"parent_vs_{subtype}"] = binary_metrics(labels[mask], scores[mask])
    return output


def main() -> int:
    args = parse_args()
    args.crop_shape = parse_triplet(args.crop_shape)
    allowed_modes = {"t3", "t3_geometry", "t3_diff_signed", "t3_diff_signed_abs"}
    modes = tuple(mode.strip() for mode in args.image_modes.split(",") if mode.strip())
    if not modes or any(mode not in allowed_modes for mode in modes):
        raise ValueError(f"unsupported --image-modes={args.image_modes!r}; allowed={sorted(allowed_modes)}")
    if not 0.0 <= args.hard_negative_fraction <= 1.0:
        raise ValueError("--hard-negative-fraction must be in [0,1]")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 128)
        args.num_workers = min(args.num_workers, 2)
        args.patience = args.epochs
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("config", "cache", "plots", "sample_losses", "sample_scores", "sample_reviews"):
        (args.output_dir / directory).mkdir(exist_ok=True)

    lookup = json.loads(args.role_lookup.read_text())
    roles: dict[str, list[list[int]]] = lookup.get("cases", lookup)
    splits = json.loads((args.preprocessed_dataset_dir / "splits_final.json").read_text())
    split = splits[args.fold]
    train_cases, val_cases = set(split["train"]), set(split["val"])
    all_cases = train_cases | val_cases
    peaks_by_case = load_peaks(args.detector_peaks, all_cases)
    if not peaks_by_case:
        raise RuntimeError("no detector peaks overlap the requested fold cases")

    train_all, train_provenance, train_availability = build_labelable_candidates(
        roles, peaks_by_case, train_cases, "train", args.match_radius_um, args.pair_radius_um
    )
    val_all, val_provenance, val_availability = build_labelable_candidates(
        roles, peaks_by_case, val_cases, "val", args.match_radius_um, args.pair_radius_um
    )
    all_samples = train_all + val_all
    all_provenance = {**train_provenance, **val_provenance}
    if not all_samples:
        raise RuntimeError("no labelable detector candidates")
    raw_all, geometry_names = detector_geometry_raw(all_samples, all_provenance, peaks_by_case)
    train_raw_indices = list(range(len(train_all)))
    val_raw_indices = list(range(len(train_all), len(all_samples)))
    selected_train, train_provenance = select_negatives(
        all_samples[: len(train_all)], train_provenance, raw_all[: len(train_all)], geometry_names,
        args.train_normal_per_parent, args.train_fp_per_parent, args.hard_negative_fraction, args.seed,
    )
    selected_val_local, val_provenance = select_negatives(
        all_samples[len(train_all):], val_provenance, raw_all[len(train_all):], geometry_names,
        args.val_normal_per_parent, args.val_fp_per_parent, args.hard_negative_fraction, args.seed + 1,
    )
    selected_val = [len(train_all) + index for index in selected_val_local]
    all_provenance = {**train_provenance, **val_provenance}
    selected_global = selected_train + selected_val
    samples = [all_samples[index] for index in selected_global]
    raw = raw_all[np.asarray(selected_global, dtype=np.int64)]
    train_count = len(selected_train)
    train_indices = list(range(train_count))
    val_indices = list(range(train_count, len(samples)))
    if not any(samples[i].label for i in train_indices):
        raise RuntimeError(
            "no detected Parent positives in classifier train split; provide detector peaks for train cases "
            "(prefer OOF peaks, not only fold-0 validation predictions)"
        )
    if not any(samples[i].label for i in val_indices):
        raise RuntimeError("no detected Parent positives in classifier validation split")
    provenance = {sample.sample_id: all_provenance[sample.sample_id] for sample in samples}
    geometry = normalize_geometry(raw, train_indices, args.output_dir, geometry_names)

    dataset_dir = args.preprocessed_dataset_dir / args.data_identifier
    crop_path, _ = build_crop_cache(samples, dataset_dir, args.output_dir / "cache", args.crop_shape, (-1, 0, 1))
    provenance_rows = []
    for sample in samples:
        item = provenance[sample.sample_id]
        row = {**asdict(sample), **asdict(item)}
        row["peak_key"] = "|".join(str(value) for value in item.peak_key)
        provenance_rows.append(row)
    write_csv(args.output_dir / "cache" / "candidate_provenance.csv", provenance_rows)

    args.modes = modes
    config = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "detector_peaks": [str(path) for path in args.detector_peaks],
        "role_lookup": str(args.role_lookup),
        "fold": args.fold,
        "crop_shape_zyx": list(args.crop_shape),
        "match_radius_um": args.match_radius_um,
        "pair_radius_um": args.pair_radius_um,
        "hard_negative_fraction": args.hard_negative_fraction,
        "negative_policy": "matched Daughter + matched normal + parent-like hard subset + status=FP; UNKNOWN excluded",
        "feature_policy": "all geometry from detector peaks; GT only labels matched candidates",
        "train_availability": train_availability,
        "val_availability": val_availability,
        "sample_counts": {
            partition: dict(Counter(samples[i].category for i in indices))
            for partition, indices in (("train", train_indices), ("val", val_indices))
        },
        "negative_subtype_counts": {
            partition: dict(Counter(provenance[samples[i].sample_id].negative_subtype for i in indices))
            for partition, indices in (("train", train_indices), ("val", val_indices))
        },
        "geometry_feature_names": geometry_names,
        "image_modes": list(modes),
        "difference_channels": {
            "t3_diff_signed": ["(t - t-1) / sqrt(2)", "(t+1 - t) / sqrt(2)"],
            "t3_diff_signed_abs": ["signed channels", "abs(signed channels)"],
            "constructed_after_shared_augmentation": True,
        },
        "smoke": args.smoke,
    }
    atomic_json(args.output_dir / "config" / "run_config.json", config)
    results = [train_mode(mode, crop_path, samples, train_indices, val_indices, geometry, args) for mode in modes]
    write_csv(args.output_dir / "sample_losses" / "train_sample_loss_by_epoch.csv", [r for x in results for r in x["train_loss_rows"]])
    write_csv(args.output_dir / "sample_losses" / "valid_sample_loss_by_epoch.csv", [r for x in results for r in x["val_loss_rows"]])
    write_csv(args.output_dir / "sample_scores" / "oof_sample_scores.csv", [r for x in results for r in x["score_rows"]])
    plot_curves(args.output_dir, results)
    save_reviews(crop_path, samples, max(results, key=lambda x: x["metrics"]["parent_vs_all"]["auprc"]), args.output_dir)
    summary_modes = {}
    for result in results:
        summary_modes[result["mode"]] = {
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "runtime_seconds": result["runtime_seconds"],
            "metrics": result["metrics"],
            "negative_subtype_metrics": subtype_metrics(result, samples, provenance),
        }
    t3_ap = summary_modes.get("t3", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    geometry_ap = summary_modes.get("t3_geometry", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    signed_ap = summary_modes.get("t3_diff_signed", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    signed_abs_ap = summary_modes.get("t3_diff_signed_abs", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    summary = {
        "modes": summary_modes,
        "t3_geometry_minus_t3_auprc": None if geometry_ap is None or t3_ap is None else geometry_ap - t3_ap,
        "t3_diff_signed_minus_t3_auprc": None if signed_ap is None or t3_ap is None else signed_ap - t3_ap,
        "t3_diff_signed_abs_minus_t3_auprc": None if signed_abs_ap is None or t3_ap is None else signed_abs_ap - t3_ap,
        "train_availability": train_availability,
        "val_availability": val_availability,
        "config": config,
        "caveat": "conditional on detected/labeled candidates; missing Parent peaks remain availability FN",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
