#!/usr/bin/env python3
"""Train Division classifiers on global/local-flow-aligned T3 difference crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.ndimage import shift as image_shift
from scipy.spatial import cKDTree

from biohub_division_crop_classifier import (
    SPACING_ZYX,
    Sample,
    atomic_json,
    build_crop_cache,
    plot_curves,
    train_mode,
    write_csv,
)
from biohub_division_detected_geometry_classifier import Peak, load_peaks
from biohub_division_human_sidecar import read_human_event_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--human-sidecar", type=Path)
    parser.add_argument("--preprocessed-dataset-dir", type=Path)
    parser.add_argument("--human-loss-weight", type=float, default=0.0)
    parser.add_argument("--official-hard-negative-pool", type=Path)
    parser.add_argument(
        "--official-hard-negative-fraction",
        type=float,
        default=0.0,
        help="Fraction of negative-class draws allocated to OOF-mined hard negatives.",
    )
    parser.add_argument("--official-targeted-negative-pool", type=Path)
    parser.add_argument(
        "--official-targeted-negative-fraction",
        type=float,
        default=0.0,
        help="Additional negative-class draw fraction for a disjoint targeted subset.",
    )
    parser.add_argument(
        "--official-targeted-sampling",
        choices=("replacement", "cycle_without_replacement"),
        default="replacement",
        help="Optionally cycle targeted rows once before any targeted row repeats.",
    )
    parser.add_argument(
        "--official-targeted-min-cycles",
        type=int,
        default=0,
        help="Minimum complete targeted-pool cycles before early-stop patience can expire.",
    )
    parser.add_argument(
        "--human-draws-per-epoch",
        type=int,
        default=0,
        help="Human-sidecar draws per epoch; 0 matches official draws.",
    )
    parser.add_argument("--human-sampling-without-replacement", action="store_true")
    parser.add_argument(
        "--human-epoch-weight-matched",
        action="store_true",
        help="Scale sparse human batches so their epoch-integrated coefficient matches --human-loss-weight.",
    )
    parser.add_argument(
        "--official-train-negative-policy",
        choices=("all", "normal_only"),
        default="all",
        help="Optionally exclude Daughter negatives from official training while retaining full validation.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best_auprc", "fixed_final"),
        default="best_auprc",
        help="Select the outer-validation best checkpoint or the preregistered final epoch.",
    )
    parser.add_argument(
        "--allow-single-class-human-sidecar",
        action="store_true",
        help="Explicitly allow a positive-only or negative-only human auxiliary sidecar.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "t3_global_diff_signed_abs",
            "t3_local_diff_signed",
            "t3_local_diff_signed_abs",
        ),
        default=(
            "t3_global_diff_signed_abs",
            "t3_local_diff_signed",
            "t3_local_diff_signed_abs",
        ),
    )
    parser.add_argument("--sigma-um", type=float, default=20.0)
    parser.add_argument(
        "--alignment-method",
        choices=(
            "gaussian_translation",
            "weighted_geomedian_translation",
            "huber_irls_translation",
        ),
        default="gaussian_translation",
        help="Local translation estimator; the default preserves the historical Gaussian control.",
    )
    parser.add_argument("--huber-delta-um", type=float, default=2.0)
    parser.add_argument("--local-radius-um", type=float, default=50.0)
    parser.add_argument("--exclude-radius-um", type=float, default=3.0)
    parser.add_argument("--residual-gate-um", type=float, default=7.0)
    parser.add_argument("--min-local-support", type=int, default=3)
    parser.add_argument("--max-local-support", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-global-cache",
        action="store_true",
        help="Do not materialize the unused global-alignment cache for local-only runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_samples(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(Sample(
                sample_id=row["sample_id"],
                split=row["split"],
                case_id=row["case_id"],
                stem=row["stem"],
                frame=int(row["frame"]),
                z=int(row["z"]),
                y=int(row["y"]),
                x=int(row["x"]),
                role=int(row["role"]),
                label=int(row["label"]),
                category=row["category"],
                matched_parent_id=row.get("matched_parent_id", ""),
            ))
    return rows


def read_human_samples(
    path: Path,
    evaluation_fold: int,
    *,
    allow_single_class: bool = False,
) -> tuple[list[Sample], list[dict[str, str]]]:
    rows = read_human_event_rows(
        path,
        evaluation_fold,
        require_both_classes=not allow_single_class,
    )
    samples: list[Sample] = []
    for row in rows:
        sample_id = row["sample_id"]
        target = int(row["target"])
        frame = int(row["frame"])
        case_id = row["case_id"]
        samples.append(Sample(
            sample_id=sample_id,
            split="human_train",
            case_id=case_id,
            stem=row["stem"],
            frame=frame,
            z=int(round(float(row["z"]))),
            y=int(round(float(row["y"]))),
            x=int(round(float(row["x"]))),
            role=0,
            label=target,
            category=row["category"],
            matched_parent_id=row["candidate_id"],
        ))
    return samples, rows


def neighbor_case_id(sample: Sample, offset: int) -> str | None:
    frame_text = sample.case_id.rsplit("_t", 1)[1]
    frame = sample.frame + offset
    if frame < 0:
        return None
    return f"{sample.stem}_t{frame:0{len(frame_text)}d}"


def physical_points(peaks: Sequence[Peak]) -> np.ndarray:
    return np.asarray([peak.coordinate for peak in peaks], dtype=np.float64).reshape(-1, 3) * SPACING_ZYX


def global_shift(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3 or len(target) < 3:
        return np.zeros(3, dtype=np.float64)
    shift = np.zeros(3, dtype=np.float64)
    tree = cKDTree(target)
    for _ in range(4):
        distances, indices = tree.query(source + shift[None], k=1)
        valid = distances <= 25.0
        if np.count_nonzero(valid) < 3:
            break
        vectors = target[indices[valid]] - source[valid]
        median = np.median(vectors, axis=0)
        residual = np.linalg.norm(vectors - median[None], axis=1)
        cutoff = max(2.0, float(np.quantile(residual, 0.70)))
        keep = residual <= cutoff
        if np.count_nonzero(keep) < 3:
            break
        shift = np.median(vectors[keep], axis=0)
    return shift


def mutual_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    shift: np.ndarray,
    residual_gate_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(source) or not len(target):
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, empty
    shifted = source + shift[None]
    target_tree = cKDTree(target)
    distances, target_indices = target_tree.query(shifted, k=1)
    source_tree = cKDTree(shifted)
    _, source_indices = source_tree.query(target, k=1)
    source_ids = np.arange(len(source))
    mutual = source_indices[target_indices] == source_ids
    valid = mutual & (distances <= residual_gate_um)
    chosen_source = source[valid]
    chosen_target = target[target_indices[valid]]
    return chosen_source, chosen_target, chosen_target - chosen_source


def _weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        return np.median(vectors, axis=0)
    return np.sum(vectors * weights[:, None], axis=0) / total


def weighted_geometric_median(
    vectors: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float = 1.0e-6,
    max_iterations: int = 100,
) -> np.ndarray:
    point = _weighted_mean(vectors, weights)
    for _ in range(max_iterations):
        distances = np.linalg.norm(vectors - point[None], axis=1)
        at_point = distances <= tolerance
        if np.any(at_point):
            exact_weight = float(np.sum(weights[at_point]))
            other_weight = float(np.sum(weights[~at_point]))
            if exact_weight >= other_weight:
                return vectors[np.flatnonzero(at_point)[0]].copy()
        updated = _weighted_mean(vectors, weights / np.maximum(distances, tolerance))
        if float(np.linalg.norm(updated - point)) <= tolerance:
            return updated
        point = updated
    return point


def huber_translation(
    vectors: np.ndarray,
    weights: np.ndarray,
    delta_um: float,
    *,
    iterations: int = 20,
) -> np.ndarray:
    point = _weighted_mean(vectors, weights)
    for _ in range(iterations):
        residual = np.linalg.norm(vectors - point[None], axis=1)
        robust = np.minimum(1.0, delta_um / np.maximum(residual, 1.0e-8))
        updated = _weighted_mean(vectors, weights * robust)
        if float(np.linalg.norm(updated - point)) <= 1.0e-6:
            return updated
        point = updated
    return point


class SparseLocalFlow:
    def __init__(self, peaks: dict[str, list[Peak]], args: argparse.Namespace):
        self.peaks = peaks
        self.args = args
        self.cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def pair(self, source_case: str | None, target_case: str | None):
        if source_case is None or target_case is None:
            empty = np.empty((0, 3), dtype=np.float64)
            return np.zeros(3, dtype=np.float64), empty, empty, empty
        key = (source_case, target_case)
        if key not in self.cache:
            source = physical_points(self.peaks.get(source_case, []))
            target = physical_points(self.peaks.get(target_case, []))
            shift = global_shift(source, target)
            anchors_source, anchors_target, vectors = mutual_correspondences(
                source, target, shift, self.args.residual_gate_um
            )
            self.cache[key] = (shift, anchors_source, anchors_target, vectors)
        return self.cache[key]

    def estimate(
        self,
        source_case: str | None,
        target_case: str | None,
        query_um: np.ndarray,
        query_side: str,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        global_vector, source, target, vectors = self.pair(source_case, target_case)
        if not len(vectors):
            return global_vector, global_vector, 0, 1
        anchors = source if query_side == "source" else target
        distances = np.linalg.norm(anchors - query_um[None], axis=1)
        valid = (distances >= self.args.exclude_radius_um) & (distances <= self.args.local_radius_um)
        indices = np.flatnonzero(valid)
        if len(indices) > self.args.max_local_support:
            indices = indices[np.argsort(distances[indices], kind="stable")[: self.args.max_local_support]]
        if len(indices) < self.args.min_local_support:
            return global_vector, global_vector, len(indices), 1
        local_vectors = vectors[indices]
        local_distances = distances[indices]
        preliminary = np.median(local_vectors, axis=0)
        deviation = np.linalg.norm(local_vectors - preliminary[None], axis=1)
        cutoff = max(3.0, float(np.quantile(deviation, 0.80)))
        keep = deviation <= cutoff
        local_vectors = local_vectors[keep]
        local_distances = local_distances[keep]
        if len(local_vectors) < self.args.min_local_support:
            return global_vector, global_vector, len(local_vectors), 1
        weights = np.exp(-0.5 * np.square(local_distances / self.args.sigma_um))
        alignment_method = getattr(self.args, "alignment_method", "gaussian_translation")
        if alignment_method == "gaussian_translation":
            local = _weighted_mean(local_vectors, weights)
        elif alignment_method == "weighted_geomedian_translation":
            local = weighted_geometric_median(local_vectors, weights)
        elif alignment_method == "huber_irls_translation":
            local = huber_translation(
                local_vectors, weights, getattr(self.args, "huber_delta_um", 2.0)
            )
        else:
            raise ValueError(f"unknown alignment method: {alignment_method}")
        return global_vector, local, len(local_vectors), 0


def build_aligned_caches(
    raw_crop_path: Path,
    samples: Sequence[Sample],
    flow: SparseLocalFlow,
    output_dir: Path,
    *,
    write_global: bool = True,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    raw = np.load(raw_crop_path, mmap_mode="r")
    if raw.shape[1] != 3:
        raise ValueError(f"expected T3 cache with 3 channels, got {raw.shape}")
    global_path = output_dir / "cache/global_aligned_crops.float16.npy"
    local_path = output_dir / "cache/local_aligned_crops.float16.npy"
    global_tmp = global_path.with_suffix(global_path.suffix + ".tmp")
    local_tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    temporary_paths = (global_tmp, local_tmp) if write_global else (local_tmp,)
    for path in temporary_paths:
        if path.exists():
            path.unlink()
    global_crops = (
        np.lib.format.open_memmap(global_tmp, mode="w+", dtype=np.float16, shape=raw.shape)
        if write_global
        else None
    )
    local_crops = np.lib.format.open_memmap(local_tmp, mode="w+", dtype=np.float16, shape=raw.shape)
    rows: list[dict[str, Any]] = []
    spatial_shape = np.asarray(raw.shape[2:], dtype=np.float64)

    def padding_fraction(vector_um: np.ndarray) -> float:
        shift_voxels = np.abs(vector_um / SPACING_ZYX)
        retained = np.maximum(spatial_shape - shift_voxels, 0.0) / spatial_shape
        return float(1.0 - np.prod(retained))

    for index, sample in enumerate(samples):
        center_um = np.asarray((sample.z, sample.y, sample.x), dtype=np.float64) * SPACING_ZYX
        current_case = sample.case_id
        prev_case = neighbor_case_id(sample, -1)
        next_case = neighbor_case_id(sample, 1)
        prev_global, prev_local, prev_support, prev_fallback = flow.estimate(
            prev_case, current_case, center_um, "target"
        )
        next_global, next_local, next_support, next_fallback = flow.estimate(
            current_case, next_case, center_um, "source"
        )
        destinations = [(local_crops, prev_local, next_local)]
        if global_crops is not None:
            destinations.insert(0, (global_crops, prev_global, next_global))
        for destination, prev_vector, next_vector in destinations:
            destination[index, 0] = image_shift(
                np.asarray(raw[index, 0], dtype=np.float32),
                shift=prev_vector / SPACING_ZYX,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ).astype(np.float16)
            destination[index, 1] = raw[index, 1]
            destination[index, 2] = image_shift(
                np.asarray(raw[index, 2], dtype=np.float32),
                shift=-next_vector / SPACING_ZYX,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ).astype(np.float16)
        rows.append({
            "sample_id": sample.sample_id,
            "split": sample.split,
            "case_id": sample.case_id,
            "label": sample.label,
            "category": sample.category,
            "prev_global_dz_um": prev_global[0], "prev_global_dy_um": prev_global[1], "prev_global_dx_um": prev_global[2],
            "prev_local_dz_um": prev_local[0], "prev_local_dy_um": prev_local[1], "prev_local_dx_um": prev_local[2],
            "next_global_dz_um": next_global[0], "next_global_dy_um": next_global[1], "next_global_dx_um": next_global[2],
            "next_local_dz_um": next_local[0], "next_local_dy_um": next_local[1], "next_local_dx_um": next_local[2],
            "prev_support": prev_support, "next_support": next_support,
            "prev_fallback": prev_fallback, "next_fallback": next_fallback,
            "prev_global_padding_fraction": padding_fraction(prev_global),
            "prev_local_padding_fraction": padding_fraction(prev_local),
            "next_global_padding_fraction": padding_fraction(next_global),
            "next_local_padding_fraction": padding_fraction(next_local),
        })
        if (index + 1) % 200 == 0 or index + 1 == len(samples):
            print(f"[aligned-cache] {index + 1}/{len(samples)}", flush=True)
    if global_crops is not None:
        global_crops.flush()
    local_crops.flush()
    del global_crops, local_crops
    if write_global:
        os.replace(global_tmp, global_path)
    os.replace(local_tmp, local_path)
    return global_path, local_path, rows


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 128)
        args.patience = args.epochs
        args.num_workers = min(args.num_workers, 2)
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for directory in ("config", "cache", "plots", "sample_losses", "sample_scores"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    official_samples_path = args.control_dir / "cache/samples.csv"
    official_crops_path = args.control_dir / "cache/crops.float16.npy"
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    official_fingerprint_before = {
        "samples_sha256": sha256_file(official_samples_path),
        "samples_size": official_samples_path.stat().st_size,
        "samples_mtime_ns": official_samples_path.stat().st_mtime_ns,
        "crops_sha256": sha256_file(official_crops_path),
        "crops_size": official_crops_path.stat().st_size,
        "crops_mtime_ns": official_crops_path.stat().st_mtime_ns,
    }
    if bool(args.human_sidecar) != bool(args.human_loss_weight > 0):
        raise ValueError(
            "--human-sidecar and positive --human-loss-weight must be supplied together: "
            f"sidecar={args.human_sidecar} weight={args.human_loss_weight}"
        )
    if args.human_sidecar is not None:
        if args.preprocessed_dataset_dir is None:
            raise ValueError("--preprocessed-dataset-dir is required with --human-sidecar")
        if not (args.human_sidecar.parent / "PREFLIGHT_PASSED").exists():
            raise ValueError(f"human sidecar has no PREFLIGHT_PASSED marker: {args.human_sidecar}")

    all_official_samples = read_samples(official_samples_path)
    all_official_index = {
        sample.sample_id: index for index, sample in enumerate(all_official_samples)
    }
    official_samples = list(all_official_samples)
    official_samples_before_policy = len(official_samples)
    if args.official_train_negative_policy == "normal_only":
        official_samples = [
            sample
            for sample in official_samples
            if sample.split != "train" or sample.label or sample.category == "normal"
        ]
    if args.smoke:
        # Preserve both classes and both splits while keeping cache validation meaningful.
        smoke_hard_ids: set[str] = set()
        smoke_targeted_ids: set[str] = set()
        for pool_path, destination in (
            (args.official_hard_negative_pool, smoke_hard_ids),
            (args.official_targeted_negative_pool, smoke_targeted_ids),
        ):
            if pool_path is not None:
                with pool_path.open(newline="", encoding="utf-8") as handle:
                    destination.update(
                        row["canonical_sample_id"]
                        for row in csv.DictReader(handle)
                        if row.get("usage") == "train"
                    )
        keep: list[Sample] = []
        for split in ("train", "val"):
            partition = [sample for sample in official_samples if sample.split == split]
            positives = [sample for sample in partition if sample.label][:6]
            negatives_all = [sample for sample in partition if not sample.label]
            hard_negatives = [
                sample for sample in negatives_all
                if sample.sample_id.partition("__")[2] in smoke_hard_ids
            ][:8] if split == "train" else []
            targeted_negatives = [
                sample for sample in negatives_all
                if sample.sample_id.partition("__")[2] in smoke_targeted_ids
            ][:8] if split == "train" else []
            hard_sample_ids = {
                sample.sample_id for sample in hard_negatives + targeted_negatives
            }
            regular_negatives = [
                sample for sample in negatives_all if sample.sample_id not in hard_sample_ids
            ][:48 - len(hard_negatives) - len(targeted_negatives)]
            negatives = hard_negatives + targeted_negatives + regular_negatives
            keep.extend(positives + negatives)
        official_samples = keep

    human_samples: list[Sample] = []
    human_sidecar_rows: list[dict[str, str]] = []
    if args.human_sidecar is not None:
        human_samples, human_sidecar_rows = read_human_samples(
            args.human_sidecar,
            args.fold,
            allow_single_class=args.allow_single_class_human_sidecar,
        )
        shutil.copy2(args.human_sidecar, args.output_dir / "config/human_event_sidecar.csv")
        official_val_stems = {sample.stem for sample in official_samples if sample.split == "val"}
        leaked_stems = sorted({sample.stem for sample in human_samples} & official_val_stems)
        if leaked_stems:
            raise ValueError(f"human sidecar overlaps official validation stems: {leaked_stems[:10]}")
        if args.smoke:
            human_samples = (
                [sample for sample in human_samples if sample.label][:4]
                + [sample for sample in human_samples if not sample.label][:4]
            )

    raw_all = np.load(official_crops_path, mmap_mode="r")
    selected_indices = [all_official_index[sample.sample_id] for sample in official_samples]
    human_raw_path: Path | None = None
    if human_samples:
        human_raw_path, _ = build_crop_cache(
            human_samples,
            args.preprocessed_dataset_dir,
            args.output_dir / "cache/human_raw",
            tuple(int(value) for value in raw_all.shape[2:]),
            (-1, 0, 1),
        )
        human_raw = np.load(human_raw_path, mmap_mode="r")
        if tuple(human_raw.shape[1:]) != tuple(raw_all.shape[1:]):
            raise ValueError(
                f"human crop shape {human_raw.shape} does not match official crop shape {raw_all.shape}"
            )
    else:
        human_raw = None
    samples = official_samples + human_samples
    raw_selected_path = args.output_dir / "cache/raw_selected.float16.npy"
    selected = np.lib.format.open_memmap(raw_selected_path, mode="w+", dtype=np.float16, shape=(len(samples), *raw_all.shape[1:]))
    for output_index, input_index in enumerate(selected_indices):
        selected[output_index] = raw_all[input_index]
    if human_raw is not None:
        for human_index in range(len(human_samples)):
            selected[len(official_samples) + human_index] = human_raw[human_index]
    selected.flush()
    del selected, raw_all, human_raw

    allowed_cases = set()
    for sample in samples:
        allowed_cases.update(case_id for case_id in (neighbor_case_id(sample, -1), sample.case_id, neighbor_case_id(sample, 1)) if case_id)
    peaks = load_peaks(args.detector_peaks, allowed_cases)
    flow = SparseLocalFlow(peaks, args)
    global_crop_path, local_crop_path, flow_rows = build_aligned_caches(
        raw_selected_path,
        samples,
        flow,
        args.output_dir,
        write_global=not args.skip_global_cache,
    )
    raw_selected_path.unlink()
    write_csv(args.output_dir / "cache/flow_alignment.csv", flow_rows)
    write_csv(args.output_dir / "cache/samples.csv", [asdict(sample) for sample in samples])

    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    val_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    human_train_indices = [index for index, sample in enumerate(samples) if sample.split == "human_train"]
    geometry = np.zeros((len(samples), 0), dtype=np.float32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    args.modes = tuple(args.modes)
    results = []
    for mode in args.modes:
        crop_path = global_crop_path if mode.startswith("t3_global") else local_crop_path
        results.append(train_mode(
            mode,
            crop_path,
            samples,
            train_indices,
            val_indices,
            geometry,
            args,
            human_train_indices=human_train_indices,
            human_loss_weight=args.human_loss_weight,
        ))
    write_csv(args.output_dir / "sample_losses/train_sample_loss_by_epoch.csv", [row for result in results for row in result["train_loss_rows"]])
    write_csv(args.output_dir / "sample_losses/valid_sample_loss_by_epoch.csv", [row for result in results for row in result["val_loss_rows"]])
    write_csv(args.output_dir / "sample_scores/oof_sample_scores.csv", [row for result in results for row in result["score_rows"]])
    plot_curves(args.output_dir, results)
    fallback_prev = float(np.mean([int(row["prev_fallback"]) for row in flow_rows]))
    fallback_next = float(np.mean([int(row["next_fallback"]) for row in flow_rows]))
    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold,
        "control_dir": str(args.control_dir),
        "detector_peaks": [str(path) for path in args.detector_peaks],
        "alignment": {
            "source": "detector sparse mutual-nearest flow; GT used only via inherited labels",
            "method": args.alignment_method,
            "sigma_um": args.sigma_um,
            "huber_delta_um": args.huber_delta_um,
            "local_radius_um": args.local_radius_um,
            "exclude_radius_um": args.exclude_radius_um,
            "residual_gate_um": args.residual_gate_um,
            "min_local_support": args.min_local_support,
            "prev_fallback_rate": fallback_prev,
            "next_fallback_rate": fallback_next,
        },
        "sample_count": len(samples),
        "official_sample_count": len(official_samples),
        "official_sample_count_before_train_negative_policy": official_samples_before_policy,
        "official_train_negative_policy": args.official_train_negative_policy,
        "official_train_sample_count": len(train_indices),
        "official_validation_sample_count": len(val_indices),
        "human_sidecar": str(args.human_sidecar) if args.human_sidecar else None,
        "human_sidecar_sha256": (
            hashlib.sha256(args.human_sidecar.read_bytes()).hexdigest() if args.human_sidecar else None
        ),
        "human_train_sample_count": len(human_train_indices),
        "human_positive_count": sum(sample.label for sample in human_samples),
        "human_negative_count": len(human_samples) - sum(sample.label for sample in human_samples),
        "human_loss_weight": args.human_loss_weight,
        "official_hard_negative_pool": str(args.official_hard_negative_pool) if args.official_hard_negative_pool else None,
        "official_hard_negative_fraction": args.official_hard_negative_fraction,
        "official_targeted_negative_pool": str(args.official_targeted_negative_pool) if args.official_targeted_negative_pool else None,
        "official_targeted_negative_fraction": args.official_targeted_negative_fraction,
        "human_draws_per_epoch": args.human_draws_per_epoch,
        "human_sampling_without_replacement": args.human_sampling_without_replacement,
        "human_epoch_weight_matched": args.human_epoch_weight_matched,
        "checkpoint_selection": args.checkpoint_selection,
        "human_sidecar_input_rows": len(human_sidecar_rows),
        "official_artifact_fingerprint_before": official_fingerprint_before,
        "mask_contract": {
            "used_by_this_event_only_trainer": ["event_mask", "event_weight"],
            "ignored_auxiliary_masks": ["edge_mask", "point_mask", "timing_mask", "detection_mask"],
            "validation": "official GT only",
        },
        "modes": {
            result["mode"]: {
                "best_epoch": result["best_epoch"],
                "epochs_completed": result["epochs_completed"],
                "runtime_seconds": result["runtime_seconds"],
                "metrics": result["metrics"],
            }
            for result in results
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    official_fingerprint_after = {
        "samples_sha256": sha256_file(official_samples_path),
        "samples_size": official_samples_path.stat().st_size,
        "samples_mtime_ns": official_samples_path.stat().st_mtime_ns,
        "crops_sha256": sha256_file(official_crops_path),
        "crops_size": official_crops_path.stat().st_size,
        "crops_mtime_ns": official_crops_path.stat().st_mtime_ns,
    }
    if official_fingerprint_after != official_fingerprint_before:
        raise RuntimeError(
            "official control artifacts changed during sidecar training: "
            f"before={official_fingerprint_before} after={official_fingerprint_after}"
        )
    summary["official_artifact_fingerprint_after"] = official_fingerprint_after
    summary["official_artifacts_unchanged"] = True
    atomic_json(args.output_dir / "config/run_config.json", {
        "fold": args.fold,
        "control_dir": str(args.control_dir),
        "detector_peaks": [str(path) for path in args.detector_peaks],
        "human_sidecar": str(args.human_sidecar) if args.human_sidecar else None,
        "human_loss_weight": args.human_loss_weight,
        "official_hard_negative_pool": str(args.official_hard_negative_pool) if args.official_hard_negative_pool else None,
        "official_hard_negative_fraction": args.official_hard_negative_fraction,
        "official_targeted_negative_pool": str(args.official_targeted_negative_pool) if args.official_targeted_negative_pool else None,
        "official_targeted_negative_fraction": args.official_targeted_negative_fraction,
        "human_draws_per_epoch": args.human_draws_per_epoch,
        "human_sampling_without_replacement": args.human_sampling_without_replacement,
        "human_epoch_weight_matched": args.human_epoch_weight_matched,
        "official_train_negative_policy": args.official_train_negative_policy,
        "checkpoint_selection": args.checkpoint_selection,
        "seed": args.seed,
        "alignment_method": args.alignment_method,
        "huber_delta_um": args.huber_delta_um,
        "skip_global_cache": args.skip_global_cache,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "modes": list(args.modes),
        "validation_contract": "official validation only",
    })
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
