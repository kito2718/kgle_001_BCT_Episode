#!/usr/bin/env python3
"""Evaluate fixed Shift9 inference and a two-seed ensemble for Fold2 TH BiGRU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.ndimage import shift as image_shift
from sklearn.metrics import average_precision_score

from biohub_division_crop_classifier import SPACING_ZYX, atomic_json, metrics_with_slices, write_csv
from biohub_division_flow_aligned_diff_classifier import read_samples
from biohub_division_light_unet_multitask import amp_context, input_channels
from biohub_division_registration_jitter_sensitivity import condition_seed, registration_errors
from biohub_division_sliding_t3_mil import SlidingT3Residual
from biohub_division_temporal_sequence_screen import normalize


DEFAULT_SEEDS = (20260817, 20260818)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--aligned-t9", type=Path, required=True)
    parser.add_argument("--extra-aligned-crops", type=Path, required=True)
    parser.add_argument("--condition-dir-template", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--magnitude-um", type=float, default=0.5)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--shift-seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--max-reference-delta", type=float, default=0.005)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in text.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integers")
    return seeds


def shift_volume(volume: np.ndarray, vector_um: np.ndarray) -> np.ndarray:
    if np.allclose(vector_um, 0.0):
        return np.asarray(volume, dtype=np.float32).copy()
    return image_shift(
        np.asarray(volume, dtype=np.float32),
        shift=np.asarray(vector_um, dtype=np.float64) / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float16).astype(np.float32)


def shifted_five_frames(
    raw_t9: np.ndarray, previous_error_um: np.ndarray, next_error_um: np.ndarray
) -> torch.Tensor:
    """Perturb the two past and two future aligned frames around the fixed centre."""
    if raw_t9.shape[0] != 9:
        raise ValueError(f"expected T9 input, got {raw_t9.shape}")
    frames = np.asarray(raw_t9[2:7], dtype=np.float32).copy()
    for index in (0, 1):
        frames[index] = shift_volume(frames[index], previous_error_um)
    for index in (3, 4):
        frames[index] = shift_volume(frames[index], -np.asarray(next_error_um))
    return torch.from_numpy(frames)


def shifted_heatmap(
    heatmap_t3: np.ndarray, previous_error_um: np.ndarray, next_error_um: np.ndarray
) -> torch.Tensor:
    if heatmap_t3.shape[0] != 3:
        raise ValueError(f"expected heatmap T3 input, got {heatmap_t3.shape}")
    frames = np.asarray(heatmap_t3, dtype=np.float32).copy()
    frames[0] = shift_volume(frames[0], previous_error_um)
    frames[2] = shift_volume(frames[2], -np.asarray(next_error_um))
    return input_channels(frames)


def make_image(
    raw_t9: np.ndarray,
    heatmap_t3: np.ndarray,
    previous_error_um: np.ndarray,
    next_error_um: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    five_frames = shifted_five_frames(raw_t9, previous_error_um, next_error_um)
    raw_channels = normalize(
        torch.stack([input_channels(five_frames[index:index + 3]) for index in range(3)]),
        mean,
        std,
    )
    heatmap_channels = shifted_heatmap(heatmap_t3, previous_error_um, next_error_um)
    return torch.cat(
        (raw_channels, heatmap_channels.unsqueeze(0).expand(3, -1, -1, -1, -1)), dim=1
    )


def inference_condition(
    models: Sequence[torch.nn.Module],
    raw: np.ndarray,
    extra: np.ndarray,
    validation_indices: Sequence[int],
    previous_errors: np.ndarray,
    next_errors: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    amp: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    binary: list[list[np.ndarray]] = [[] for _ in models]
    count: list[list[np.ndarray]] = [[] for _ in models]
    offset: list[list[np.ndarray]] = [[] for _ in models]
    for model in models:
        model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation_indices), batch_size):
            stop = min(start + batch_size, len(validation_indices))
            images = torch.stack([
                make_image(
                    raw[int(validation_indices[position])],
                    extra[int(validation_indices[position])],
                    previous_errors[position],
                    next_errors[position],
                    mean,
                    std,
                )
                for position in range(start, stop)
            ]).cuda(non_blocking=True)
            for model_index, model in enumerate(models):
                with amp_context(amp):
                    outputs = model(images)
                binary[model_index].append(torch.sigmoid(outputs["binary"].float()).cpu().numpy())
                count[model_index].append(torch.softmax(outputs["count"].float(), dim=1).cpu().numpy())
                offset[model_index].append(outputs["offset"].float().cpu().numpy())
    return (
        [np.concatenate(parts) for parts in binary],
        [np.concatenate(parts) for parts in count],
        [np.concatenate(parts) for parts in offset],
    )


def model_view_values_at(
    condition_scores: Sequence[Sequence[np.ndarray]], position: int
) -> np.ndarray:
    """Return one candidate's probabilities across every model and TTA view."""
    values = [
        float(view_scores[position])
        for model_scores in condition_scores
        for view_scores in model_scores
    ]
    if not values:
        raise ValueError("condition score collection is empty")
    return np.asarray(values, dtype=np.float64)


def cluster_bootstrap(
    labels: np.ndarray,
    stems: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(stems)
    by_stem = {stem: np.flatnonzero(stems == stem) for stem in unique}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled_stems = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by_stem[stem] for stem in sampled_stems])
        if len(np.unique(labels[selected])) < 2:
            continue
        deltas.append(float(
            average_precision_score(labels[selected], candidate[selected])
            - average_precision_score(labels[selected], control[selected])
        ))
    values = np.asarray(deltas, dtype=np.float64)
    if not len(values):
        raise RuntimeError("bootstrap produced no valid replicates")
    return {
        "unit": "video_stem",
        "requested_replicates": replicates,
        "valid_replicates": int(len(values)),
        "delta_mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_delta_gt_zero": float(np.mean(values > 0)),
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.magnitude_um <= 0 or args.replicates <= 0 or args.batch_size <= 0:
        raise ValueError("magnitude, replicates, and batch size must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    samples = read_samples(args.samples)
    raw = np.load(args.aligned_t9, mmap_mode="r")
    extra = np.load(args.extra_aligned_crops, mmap_mode="r")
    if raw.ndim != 5 or raw.shape[0] != len(samples) or raw.shape[1] != 9:
        raise ValueError(f"T9/sample mismatch: {raw.shape}, {len(samples)}")
    if extra.shape != (len(samples), 3, *raw.shape[-3:]):
        raise ValueError(f"heatmap/sample mismatch: {extra.shape}")
    validation_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    if args.smoke:
        positives = [index for index in validation_indices if samples[index].label]
        daughters = [index for index in validation_indices if samples[index].category == "daughter"]
        normals = [index for index in validation_indices if samples[index].category == "normal"]
        validation_indices = positives[:4] + daughters[:4] + normals[:16]
        args.replicates = 1
        args.bootstrap_replicates = 20
    validation_samples = [samples[index] for index in validation_indices]

    models: list[torch.nn.Module] = []
    references: list[np.ndarray] = []
    checkpoint_provenance: list[dict[str, Any]] = []
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    for seed in seeds:
        condition = Path(args.condition_dir_template.format(seed=seed))
        checkpoint_path = condition / "arms/sliding_t3_bigru/checkpoint_best.pt"
        score_path = condition / "arms/sliding_t3_bigru/oof_sample_scores.csv"
        summary_path = condition / "arms/sliding_t3_bigru/summary.json"
        for path in (checkpoint_path, score_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text())
        current_mean = torch.as_tensor(summary["normalization"]["mean"], dtype=torch.float32)
        current_std = torch.as_tensor(summary["normalization"]["std"], dtype=torch.float32)
        if mean is None:
            mean, std = current_mean, current_std
        elif not torch.equal(mean, current_mean) or not torch.equal(std, current_std):
            raise RuntimeError("seed checkpoints have different input normalization")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        model = SlidingT3Residual("bigru", 14).cuda()
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model)
        reference_by_id = {row["sample_id"]: float(row["prediction"]) for row in read_rows(score_path)}
        expected_ids = [sample.sample_id for sample in validation_samples]
        if any(sample_id not in reference_by_id for sample_id in expected_ids):
            raise RuntimeError(f"reference score cohort missing rows for seed {seed}")
        references.append(np.asarray([reference_by_id[sample_id] for sample_id in expected_ids]))
        checkpoint_provenance.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": int(checkpoint.get("epoch", -1)),
        })
    assert mean is not None and std is not None

    sample_count = len(validation_indices)
    zeros = np.zeros((sample_count, 3), dtype=np.float64)
    exact_binary, exact_count, exact_offset = inference_condition(
        models, raw, extra, validation_indices, zeros, zeros, mean, std,
        args.batch_size, args.amp,
    )
    replay_deltas = [float(np.max(np.abs(actual - reference))) for actual, reference in zip(exact_binary, references)]
    if not args.smoke and max(replay_deltas) > args.max_reference_delta:
        raise RuntimeError(f"exact replay delta exceeds contract: {replay_deltas}")

    condition_scores: list[list[np.ndarray]] = [[score] for score in exact_binary]
    condition_ids = ["exact"]
    for family in ("common", "independent"):
        for replicate in range(args.replicates):
            seed = condition_seed(args.shift_seed, args.magnitude_um, family, replicate)
            previous, following = registration_errors(sample_count, args.magnitude_um, family, seed)
            scores, _, _ = inference_condition(
                models, raw, extra, validation_indices, previous, following, mean, std,
                args.batch_size, args.amp,
            )
            for model_index, values in enumerate(scores):
                condition_scores[model_index].append(values)
            condition_ids.append(f"{family}_r{replicate}")

    mean9_by_seed = [np.mean(np.stack(scores), axis=0) for scores in condition_scores]
    exact_ensemble = np.mean(np.stack(exact_binary), axis=0)
    shift9_ensemble = np.mean(np.stack(mean9_by_seed), axis=0)
    labels = np.asarray([sample.label for sample in validation_samples], dtype=np.int8)
    stems = np.asarray([sample.stem for sample in validation_samples])
    ordered = np.asarray(validation_indices, dtype=np.int64)
    exact_metrics = metrics_with_slices(samples, ordered, exact_ensemble)
    candidate_metrics = metrics_with_slices(samples, ordered, shift9_ensemble)
    deltas = {
        name: float(candidate_metrics[name]["auprc"] - exact_metrics[name]["auprc"])
        for name in ("parent_vs_all", "parent_vs_normal", "parent_vs_daughter")
    }
    bootstrap = cluster_bootstrap(
        labels, stems, exact_ensemble, shift9_ensemble,
        args.bootstrap_replicates, args.shift_seed,
    )
    gate = {
        "overall_gain_ge_0.005": deltas["parent_vs_all"] >= 0.005,
        "normal_drop_ge_minus_0.005": deltas["parent_vs_normal"] >= -0.005,
        "daughter_drop_ge_minus_0.005": deltas["parent_vs_daughter"] >= -0.005,
        "bootstrap_ci_lower_gt_zero": bootstrap["ci95"][0] > 0,
    }
    gate["passed"] = all(gate.values())

    rows: list[dict[str, Any]] = []
    for position, sample in enumerate(validation_samples):
        stacked = model_view_values_at(condition_scores, position)
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "stem": sample.stem,
            "frame": sample.frame,
            "category": sample.category,
            "label": int(sample.label),
            "exact_seed_ensemble": float(exact_ensemble[position]),
            "shift9_seed_ensemble": float(shift9_ensemble[position]),
            "delta": float(shift9_ensemble[position] - exact_ensemble[position]),
            "all_model_view_std": float(stacked.std()),
            "all_model_view_range": float(stacked.max() - stacked.min()),
        }
        for model_index, seed in enumerate(seeds):
            row[f"exact_seed{seed}"] = float(exact_binary[model_index][position])
            row[f"shift9_seed{seed}"] = float(mean9_by_seed[model_index][position])
        rows.append(row)
    write_csv(args.output_dir / "per_candidate_scores.csv", rows)

    per_seed = {}
    for model_index, seed in enumerate(seeds):
        per_seed[str(seed)] = {
            "exact": metrics_with_slices(samples, ordered, exact_binary[model_index]),
            "shift9": metrics_with_slices(samples, ordered, mean9_by_seed[model_index]),
            "exact_reference_max_abs_delta": replay_deltas[model_index],
        }
    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "division_bigru_hard10_heatmap_shift9_two_seed_ensemble_fold2",
        "fold": 2,
        "validation_samples": len(validation_samples),
        "positive_samples": int(labels.sum()),
        "seeds": list(seeds),
        "shift": {
            "magnitude_um": args.magnitude_um,
            "families": ["common", "independent"],
            "replicates_per_family": args.replicates,
            "conditions": condition_ids,
            "past_frames": [-2, -1],
            "fixed_centre_frame": 0,
            "future_frames": [1, 2],
            "raw_and_heatmap_shifted_together": True,
        },
        "policy": "probability mean over two fixed seeds and exact+8 shifted views; binary score only",
        "auxiliary_policy": {
            "count_and_offset": "unchanged exact-view outputs",
            "reason": "the intervention changes only the binary Division ranking score",
            "exact_seed_metrics": [
                json.loads((Path(args.condition_dir_template.format(seed=seed)) / "arms/sliding_t3_bigru/summary.json").read_text())["metrics"]["auxiliary"]
                for seed in seeds
            ],
        },
        "control": {"name": "two_seed_exact_probability_mean", "metrics": exact_metrics},
        "candidate": {"name": "two_seed_shift9_probability_mean", "metrics": candidate_metrics},
        "deltas": deltas,
        "bootstrap": bootstrap,
        "gate": gate,
        "per_seed": per_seed,
        "checkpoints": checkpoint_provenance,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    })
    (args.output_dir / "DONE").touch()
    print(json.dumps({"deltas": deltas, "gate": gate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
