#!/usr/bin/env python3
"""Audit Division successor-feedback sensitivity to local-flow registration error."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.ndimage import shift as image_shift

from biohub_division_crop_classifier import (
    SPACING_ZYX,
    Sample,
    atomic_json,
    binary_metrics,
    metrics_with_slices,
    write_csv,
)
from biohub_division_flow_aligned_diff_classifier import read_samples
from biohub_division_light_unet_multitask import amp_context, input_channels
from biohub_division_small_cnn_multitask import SmallDivisionSuccessorCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-crops", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--flow-alignment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--magnitudes-um", default="0,0.5,1,2,4")
    parser.add_argument("--families", nargs="+", choices=("common", "independent"), default=("common", "independent"))
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--max-clean-score-delta", type=float, default=0.005)
    parser.add_argument("--skip-reference-replay", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_magnitudes(text: str) -> list[float]:
    values = [float(value) for value in text.split(",")]
    if not values or values[0] != 0.0 or any(value < 0 for value in values):
        raise ValueError("magnitudes must begin with 0 and be nonnegative")
    if values != sorted(set(values)):
        raise ValueError("magnitudes must be unique and sorted")
    return values


def unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("random unit-vector construction produced zero norm")
    return vectors / norms


def registration_errors(
    count: int,
    magnitude_um: float,
    family: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if magnitude_um < 0:
        raise ValueError("magnitude must be nonnegative")
    if family not in {"common", "independent"}:
        raise ValueError(f"unknown error family: {family}")
    if magnitude_um == 0:
        zeros = np.zeros((count, 3), dtype=np.float64)
        return zeros, zeros.copy()
    rng = np.random.default_rng(seed)
    previous = unit_vectors(rng, count) * magnitude_um
    following = previous.copy() if family == "common" else unit_vectors(rng, count) * magnitude_um
    return previous, following


def aligned_crop(
    raw: np.ndarray,
    prev_vector_um: np.ndarray,
    next_vector_um: np.ndarray,
) -> np.ndarray:
    if raw.shape[0] != 3:
        raise ValueError(f"expected raw T3 crop, got {raw.shape}")
    output = np.empty_like(raw, dtype=np.float32)
    output[0] = image_shift(
        np.asarray(raw[0], dtype=np.float32),
        shift=prev_vector_um / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    output[1] = np.asarray(raw[1], dtype=np.float32)
    output[2] = image_shift(
        np.asarray(raw[2], dtype=np.float32),
        shift=-next_vector_um / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    # Match the original cache contract: aligned raw frames were persisted as float16
    # before the seven signed/absolute channels were derived.
    return output.astype(np.float16).astype(np.float32)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_vector(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_d{axis}_um"]) for axis in "zyx"], dtype=np.float64)


def infer_condition(
    model: torch.nn.Module,
    raw_crops: np.ndarray,
    sample_indices: Sequence[int],
    flow_rows: Sequence[dict[str, str]],
    prev_errors: np.ndarray,
    next_errors: np.ndarray,
    batch_size: int,
    amp: str,
) -> np.ndarray:
    device = torch.device("cuda")
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sample_indices), batch_size):
            stop = min(start + batch_size, len(sample_indices))
            channels = []
            for local_index in range(start, stop):
                sample_index = int(sample_indices[local_index])
                row = flow_rows[sample_index]
                crop = aligned_crop(
                    np.asarray(raw_crops[sample_index]),
                    row_vector(row, "prev_local") + prev_errors[local_index],
                    row_vector(row, "next_local") + next_errors[local_index],
                )
                channels.append(input_channels(crop))
            images = torch.stack(channels).to(device, non_blocking=True)
            with amp_context(amp):
                outputs = model(images)
            probabilities.append(torch.sigmoid(outputs["binary"].float()).cpu().numpy())
    return np.concatenate(probabilities)


def safe_slice_metrics(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected_labels = labels[mask]
    if len(selected_labels) < 2 or len(np.unique(selected_labels)) < 2:
        return {
            "status": "unavailable_single_class_or_empty",
            "n": int(len(selected_labels)),
            "positives": int(selected_labels.sum()),
        }
    result = binary_metrics(selected_labels.astype(bool), scores[mask])
    result["status"] = "available"
    return result


def condition_seed(base: int, magnitude: float, family: str, replicate: int) -> int:
    payload = f"{base}|{magnitude:.6f}|{family}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def main() -> int:
    args = parse_args()
    magnitudes = parse_magnitudes(args.magnitudes_um)
    if args.replicates <= 0 or args.batch_size <= 0:
        raise ValueError("replicates and batch size must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    required = (args.raw_crops, args.samples, args.flow_alignment, args.checkpoint, args.reference_scores)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    samples = read_samples(args.samples)
    flow_rows = read_rows(args.flow_alignment)
    raw_crops = np.load(args.raw_crops, mmap_mode="r")
    if len(samples) != len(flow_rows) or len(samples) != len(raw_crops):
        raise RuntimeError(
            f"input length mismatch samples={len(samples)} flow={len(flow_rows)} crops={len(raw_crops)}"
        )
    if [sample.sample_id for sample in samples] != [row["sample_id"] for row in flow_rows]:
        raise RuntimeError("flow-alignment rows do not match sample order")

    validation_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    if args.smoke:
        positives = [index for index in validation_indices if samples[index].label]
        negatives = [index for index in validation_indices if not samples[index].label]
        validation_indices = positives[:6] + negatives[:24]
        magnitudes = [0.0, 1.0]
        args.replicates = 1
    validation_samples = [samples[index] for index in validation_indices]
    validation_rows = [flow_rows[index] for index in validation_indices]
    labels = np.asarray([sample.label for sample in validation_samples], dtype=bool)
    categories = np.asarray([sample.category for sample in validation_samples])

    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = SmallDivisionSuccessorCNN("feedback").cuda()
    model.load_state_dict(checkpoint["model"])

    reference_by_id = {
        row["sample_id"]: float(row["prediction"])
        for row in read_rows(args.reference_scores)
        if row.get("arm") == "feedback"
    }
    if not args.smoke and set(sample.sample_id for sample in validation_samples) != set(reference_by_id):
        raise RuntimeError("reference scores do not exactly cover the validation cohort")

    supports = np.asarray([
        min(int(row["prev_support"]), int(row["next_support"])) for row in validation_rows
    ])
    fallbacks = np.asarray([
        bool(int(row["prev_fallback"]) or int(row["next_fallback"])) for row in validation_rows
    ])
    paddings = np.asarray([
        max(float(row["prev_local_padding_fraction"]), float(row["next_local_padding_fraction"]))
        for row in validation_rows
    ])
    movements = np.asarray([
        max(np.linalg.norm(row_vector(row, "prev_local")), np.linalg.norm(row_vector(row, "next_local")))
        for row in validation_rows
    ])
    slice_thresholds = {
        "support_low_q25": float(np.quantile(supports, 0.25)),
        "support_high_q75": float(np.quantile(supports, 0.75)),
        "padding_high_q75": float(np.quantile(paddings, 0.75)),
        "large_movement_q75_um": float(np.quantile(movements, 0.75)),
    }
    slice_masks = {
        "support_low": supports <= slice_thresholds["support_low_q25"],
        "support_high_dense_proxy": supports >= slice_thresholds["support_high_q75"],
        "padding_high": paddings >= slice_thresholds["padding_high_q75"],
        "large_movement": movements >= slice_thresholds["large_movement_q75_um"],
        "fallback": fallbacks,
        "nonfallback": ~fallbacks,
    }

    conditions: list[tuple[float, str, int]] = [(0.0, "exact", 0)]
    for magnitude in magnitudes:
        if magnitude == 0:
            continue
        for family in args.families:
            for replicate in range(args.replicates):
                conditions.append((magnitude, family, replicate))

    score_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    condition_details: list[dict[str, Any]] = []
    clean_scores: np.ndarray | None = None
    clean_metrics: dict[str, Any] | None = None
    clean_slice_metrics: dict[str, dict[str, Any]] = {}
    clean_threshold: float | None = None

    for condition_index, (magnitude, family, replicate) in enumerate(conditions, start=1):
        seed = condition_seed(args.seed, magnitude, family, replicate)
        previous_error, next_error = registration_errors(
            len(validation_indices), magnitude, "common" if family == "exact" else family, seed
        )
        scores = infer_condition(
            model,
            raw_crops,
            validation_indices,
            flow_rows,
            previous_error,
            next_error,
            args.batch_size,
            args.amp,
        )
        metrics = metrics_with_slices(validation_samples, range(len(validation_samples)), scores)
        slices = {
            name: safe_slice_metrics(labels, scores, mask) for name, mask in slice_masks.items()
        }
        if clean_scores is None:
            clean_scores = scores.copy()
            clean_metrics = metrics
            clean_slice_metrics = slices
            clean_threshold = float(metrics["parent_vs_all"]["best_threshold"])
            if not args.smoke and not args.skip_reference_replay:
                reference = np.asarray([reference_by_id[sample.sample_id] for sample in validation_samples])
                max_delta = float(np.max(np.abs(clean_scores - reference)))
                if max_delta > args.max_clean_score_delta:
                    raise RuntimeError(
                        f"clean reconstruction score delta {max_delta:.8f} exceeds {args.max_clean_score_delta}"
                    )
        assert clean_scores is not None and clean_metrics is not None and clean_threshold is not None
        flips = (scores >= clean_threshold) != (clean_scores >= clean_threshold)
        delta_scores = scores - clean_scores
        condition_id = f"m{magnitude:g}_{family}_r{replicate}"
        detail = {
            "condition_id": condition_id,
            "magnitude_um": magnitude,
            "family": family,
            "replicate": replicate,
            "seed": seed,
            "metrics": metrics,
            "slices": slices,
            "delta_parent_vs_all_auprc": float(
                metrics["parent_vs_all"]["auprc"] - clean_metrics["parent_vs_all"]["auprc"]
            ),
            "flip_rate": float(np.mean(flips)),
            "mean_abs_score_change": float(np.mean(np.abs(delta_scores))),
        }
        condition_details.append(detail)
        condition_rows.append({
            "condition_id": condition_id,
            "magnitude_um": magnitude,
            "family": family,
            "replicate": replicate,
            "parent_vs_all_auprc": metrics["parent_vs_all"]["auprc"],
            "delta_parent_vs_all_auprc": detail["delta_parent_vs_all_auprc"],
            "parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"],
            "delta_parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"] - clean_metrics["parent_vs_normal"]["auprc"],
            "parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"],
            "delta_parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"] - clean_metrics["parent_vs_daughter"]["auprc"],
            "flip_rate": detail["flip_rate"],
            "mean_abs_score_change": detail["mean_abs_score_change"],
        })
        for local_index, (sample, score, delta, flip) in enumerate(
            zip(validation_samples, scores, delta_scores, flips)
        ):
            score_rows.append({
                "condition_id": condition_id,
                "magnitude_um": magnitude,
                "family": family,
                "replicate": replicate,
                "sample_id": sample.sample_id,
                "stem": sample.stem,
                "frame": sample.frame,
                "category": sample.category,
                "label": sample.label,
                "clean_score": float(clean_scores[local_index]),
                "jitter_score": float(score),
                "score_delta": float(delta),
                "prediction_flip": int(flip),
                "prev_error_dz_um": float(previous_error[local_index, 0]),
                "prev_error_dy_um": float(previous_error[local_index, 1]),
                "prev_error_dx_um": float(previous_error[local_index, 2]),
                "next_error_dz_um": float(next_error[local_index, 0]),
                "next_error_dy_um": float(next_error[local_index, 1]),
                "next_error_dx_um": float(next_error[local_index, 2]),
                "min_support": int(supports[local_index]),
                "fallback": int(fallbacks[local_index]),
                "max_padding_fraction": float(paddings[local_index]),
                "max_local_movement_um": float(movements[local_index]),
            })
        print(
            f"[{condition_index}/{len(conditions)}] {condition_id} "
            f"ap={metrics['parent_vs_all']['auprc']:.6f} "
            f"delta={detail['delta_parent_vs_all_auprc']:+.6f} flips={detail['flip_rate']:.4f}",
            flush=True,
        )

    assert clean_scores is not None and clean_metrics is not None and clean_threshold is not None
    grouped: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        if magnitude == 0:
            continue
        for family in args.families:
            rows = [
                row for row in condition_rows
                if float(row["magnitude_um"]) == magnitude and row["family"] == family
            ]
            if not rows:
                continue
            grouped.append({
                "magnitude_um": magnitude,
                "family": family,
                "replicates": len(rows),
                "parent_vs_all_auprc_mean": float(np.mean([row["parent_vs_all_auprc"] for row in rows])),
                "parent_vs_all_auprc_std": float(np.std([row["parent_vs_all_auprc"] for row in rows])),
                "delta_parent_vs_all_auprc_mean": float(np.mean([row["delta_parent_vs_all_auprc"] for row in rows])),
                "delta_parent_vs_normal_auprc_mean": float(np.mean([row["delta_parent_vs_normal_auprc"] for row in rows])),
                "delta_parent_vs_daughter_auprc_mean": float(np.mean([row["delta_parent_vs_daughter_auprc"] for row in rows])),
                "flip_rate_mean": float(np.mean([row["flip_rate"] for row in rows])),
                "mean_abs_score_change": float(np.mean([row["mean_abs_score_change"] for row in rows])),
            })

    observed_band = [row for row in grouped if 0.5 <= row["magnitude_um"] <= 2.0]
    ap_vulnerable = any(row["delta_parent_vs_all_auprc_mean"] <= -0.01 for row in observed_band)
    flip_vulnerable = any(row["magnitude_um"] >= 1.0 and row["flip_rate_mean"] >= 0.02 for row in observed_band)
    gate_passed = bool(ap_vulnerable or flip_vulnerable)

    harmful = sorted(
        (
            row for row in score_rows
            if float(row["magnitude_um"]) > 0
            and ((int(row["label"]) == 1 and float(row["score_delta"]) < 0)
                 or (int(row["label"]) == 0 and float(row["score_delta"]) > 0))
        ),
        key=lambda row: abs(float(row["score_delta"])),
        reverse=True,
    )[:100]

    write_csv(args.output_dir / "per_candidate_scores.csv", score_rows)
    write_csv(args.output_dir / "condition_metrics.csv", condition_rows)
    write_csv(args.output_dir / "grouped_robustness_curve.csv", grouped)
    write_csv(args.output_dir / "representative_harmful_score_changes.csv", harmful)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "division_registration_jitter_sensitivity_fold2",
        "control": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "reference_scores": str(args.reference_scores),
            "clean_metrics": clean_metrics,
            "clean_threshold": clean_threshold,
            "clean_score_reference_max_abs_delta": (
                None if (args.smoke or args.skip_reference_replay) else float(max(
                    abs(clean_scores[index] - reference_by_id[sample.sample_id])
                    for index, sample in enumerate(validation_samples)
                ))
            ),
        },
        "contract": {
            "fold": 2,
            "seed": args.seed,
            "magnitudes_um": magnitudes,
            "families": list(args.families),
            "replicates": args.replicates,
            "validation_samples": len(validation_samples),
            "positive_samples": int(labels.sum()),
            "raw_frame_rewarp_before_seven_channel_derivation": True,
            "candidate_center_fixed": True,
            "translation_only": True,
        },
        "slice_thresholds": slice_thresholds,
        "clean_slices": clean_slice_metrics,
        "unavailable_slices": {
            "freeze": "no freeze-state metadata in fixed 1,850-candidate artifact",
            "post_freeze": "no freeze-state metadata in fixed 1,850-candidate artifact",
            "biological_dense": "local support-high is reported only as a density proxy",
        },
        "condition_details": condition_details,
        "grouped_robustness_curve": grouped,
        "j1_gate": {
            "passed": gate_passed,
            "ap_rule": "mean Parent-vs-all AUPRC delta <= -0.01 at 0.5--2 um",
            "flip_rule": "mean flip rate >= 0.02 at 1--2 um",
            "ap_vulnerable": ap_vulnerable,
            "flip_vulnerable": flip_vulnerable,
            "decision": "launch matched J1 augmentation" if gate_passed else "do not launch J1",
        },
        "runtime_seconds": time.perf_counter() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary["j1_gate"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
