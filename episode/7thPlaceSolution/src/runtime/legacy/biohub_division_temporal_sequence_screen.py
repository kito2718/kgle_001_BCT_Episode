#!/usr/bin/env python3
"""Matched T5 concat/sequence fusion and timing-label uncertainty screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from biohub_division_crop_classifier import (
    SmallDivisionCNN,
    amp_context,
    atomic_json,
    metrics_with_slices,
    worker_seed,
    write_csv,
)
from biohub_division_flow_aligned_diff_classifier import read_samples
from biohub_division_light_unet_multitask import (
    SpatialTarget,
    build_spatial_targets,
    input_channels,
    sampling_weights,
    seed_everything,
)
from biohub_division_small_cnn_multitask import (
    arm_losses,
    atomic_torch_save,
    ordered_offset_targets,
)


BASE_SPECS = (
    ("concat7_plain", "concat7"),
    ("concat13_plain", "concat13"),
    ("bigru_plain", "bigru"),
    ("bilstm_plain", "bilstm"),
)
TIMING_LOSSES = ("none", "hard_exact", "soft_pm1", "interval_mil_pm1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--aligned-t9", type=Path, required=True)
    parser.add_argument("--flow-alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--timing-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-scale-um", type=float, default=10.0)
    parser.add_argument("--max-node-match-distance-um", type=float, default=7.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--concat7-only",
        action="store_true",
        help="Train only the matched multi-head concat7 control checkpoint.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def concat13_channels(frames: torch.Tensor) -> torch.Tensor:
    """Five raw frames plus signed/absolute residuals against the centre."""
    if frames.shape[0] != 5:
        raise ValueError(f"expected five frames, got {tuple(frames.shape)}")
    centre = frames[2:3]
    signed = [
        (centre - frames[index:index + 1]) / math.sqrt(2.0)
        if index < 2
        else (frames[index:index + 1] - centre) / math.sqrt(2.0)
        for index in range(5)
        if index != 2
    ]
    return torch.cat((frames, *signed, *(value.abs() for value in signed)), dim=0)


def sequence_token_channels(frames: torch.Tensor) -> torch.Tensor:
    """Return ordered [T, raw/backward/forward/abs-backward/abs-forward]."""
    if frames.shape[0] != 5:
        raise ValueError(f"expected five frames, got {tuple(frames.shape)}")
    tokens = []
    zero = torch.zeros_like(frames[0:1])
    for index in range(5):
        backward = (
            (frames[index:index + 1] - frames[index - 1:index]) / math.sqrt(2.0)
            if index > 0 else zero
        )
        forward = (
            (frames[index + 1:index + 2] - frames[index:index + 1]) / math.sqrt(2.0)
            if index < 4 else zero
        )
        tokens.append(torch.cat((frames[index:index + 1], backward, forward, backward.abs(), forward.abs()), dim=0))
    return torch.stack(tokens, dim=0)


def make_input(raw_t9: torch.Tensor, architecture: str, shift: int = 0) -> torch.Tensor:
    start = 2 + int(shift)
    frames = raw_t9[start:start + 5]
    if frames.shape[0] != 5:
        raise ValueError(f"invalid temporal shift {shift} for {tuple(raw_t9.shape)}")
    if architecture == "concat7":
        return input_channels(frames[1:4])
    if architecture == "concat13":
        return concat13_channels(frames)
    if architecture in {"bigru", "bilstm"}:
        return sequence_token_channels(frames)
    raise ValueError(f"unknown architecture: {architecture}")


def estimate_channel_statistics(
    path: Path, train_indices: Sequence[int], architecture: str
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    channel_count = 5 if architecture in {"bigru", "bilstm"} else 7 if architecture == "concat7" else 13
    sums = np.zeros(channel_count, dtype=np.float64)
    squared = np.zeros(channel_count, dtype=np.float64)
    count = 0
    for index in train_indices:
        image = make_input(torch.from_numpy(np.asarray(array[int(index)], dtype=np.float32).copy()), architecture)
        if image.ndim == 5:
            image = image.permute(1, 0, 2, 3, 4).reshape(channel_count, -1)
        else:
            image = image.reshape(channel_count, -1)
        values = image.numpy()
        sums += values.sum(1, dtype=np.float64)
        squared += np.square(values, dtype=np.float64).sum(1, dtype=np.float64)
        count += values.shape[1]
    mean = sums / count
    variance = np.maximum(squared / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return {
        "mode": "train_channel_zscore",
        "architecture": architecture,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_sample_count": len(train_indices),
        "voxel_count_per_channel": int(count),
        "source": "unaugmented shift-zero aligned T9 training rows only",
    }


def normalize(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return (image - mean[:, None, None, None]) / std[:, None, None, None]
    if image.ndim == 5:
        return (image - mean[None, :, None, None, None]) / std[None, :, None, None, None]
    raise ValueError(tuple(image.shape))


class TemporalDataset(Dataset):
    def __init__(
        self,
        path: Path,
        samples,
        targets: Sequence[SpatialTarget],
        indices: Sequence[int],
        architecture: str,
        augment: bool,
        temporal_shift: bool,
        offset_scale_um: float,
        channel_mean: Sequence[float],
        channel_std: Sequence[float],
    ) -> None:
        self.path = path
        self.samples = samples
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.architecture = architecture
        self.augment = augment
        self.temporal_shift = temporal_shift
        self.offset_scale_um = offset_scale_um
        self.mean = torch.as_tensor(channel_mean, dtype=torch.float32)
        self.std = torch.as_tensor(channel_std, dtype=torch.float32)
        self._array = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        sample = self.samples[index]
        target = self.targets[index]
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        raw = torch.from_numpy(np.asarray(self._array[index], dtype=np.float32).copy())
        shift = int(torch.randint(-2, 3, ()).item()) if self.augment and self.temporal_shift else 0
        flips = [False, False, False]
        rotations = 0
        if self.augment:
            flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
            for axis, enabled in enumerate(flips):
                if enabled:
                    raw = torch.flip(raw, dims=(axis - 3,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                raw = torch.rot90(raw, rotations, dims=(-2, -1))
            raw = raw * (0.95 + 0.10 * torch.rand(())) + 0.05 * torch.randn(())
        image = normalize(make_input(raw, self.architecture, shift), self.mean, self.std)
        offsets, offset_valid = ordered_offset_targets(
            target.children_zyx_aligned,
            raw.shape[-3:],
            flips,
            rotations,
            target.heatmap_valid,
            self.offset_scale_um,
        )
        timing = 2 - shift if sample.label else 5
        return {
            "image": image,
            "binary": torch.tensor(float(sample.label), dtype=torch.float32),
            "count": torch.tensor(target.child_count_index, dtype=torch.long),
            "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
            "offset": offsets,
            "offset_valid": offset_valid,
            "timing": torch.tensor(timing, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class FeedbackHeads(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.binary = nn.Sequential(nn.Dropout(0.25), nn.Linear(feature_dim, 1))
        self.count = nn.Linear(feature_dim, 3)
        self.offset = nn.Linear(feature_dim, 6)
        self.timing = nn.Linear(feature_dim, 6)
        self.feedback = nn.Linear(9, 1)
        nn.init.zeros_(self.feedback.weight)
        nn.init.zeros_(self.feedback.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        count = self.count(features)
        offset = self.offset(features).reshape(-1, 2, 3)
        binary_pre_feedback = self.binary(features).squeeze(1)
        probabilities = torch.softmax(count.float(), dim=1).to(dtype=offset.dtype)
        presence = torch.stack((probabilities[:, 1] + probabilities[:, 2], probabilities[:, 2]), dim=1).unsqueeze(2)
        feedback = torch.cat((probabilities, (torch.tanh(offset) * presence).flatten(1)), dim=1).detach()
        feedback_correction = self.feedback(feedback).squeeze(1)
        binary = binary_pre_feedback + feedback_correction
        return {
            "binary": binary,
            "binary_pre_feedback": binary_pre_feedback,
            "feedback_correction": feedback_correction,
            "count": count,
            "offset": offset,
            "timing": self.timing(features),
        }


class ConcatModel(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.features = SmallDivisionCNN(channels).features
        self.heads = FeedbackHeads()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.features(image).flatten(1))


class SequenceModel(nn.Module):
    def __init__(self, recurrent: str) -> None:
        super().__init__()
        self.frame_encoder = SmallDivisionCNN(5).features
        recurrent_class = nn.GRU if recurrent == "bigru" else nn.LSTM
        self.recurrent = recurrent_class(128, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.heads = FeedbackHeads()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, channels, depth, height, width = image.shape
        encoded = self.frame_encoder(image.reshape(batch * frames, channels, depth, height, width)).flatten(1)
        sequence, _ = self.recurrent(encoded.reshape(batch, frames, 128))
        weights = torch.softmax(self.attention(sequence).squeeze(2), dim=1)
        features = torch.sum(sequence * weights.unsqueeze(2), dim=1)
        return self.heads(features)


def build_model(architecture: str) -> nn.Module:
    if architecture == "concat7":
        return ConcatModel(7)
    if architecture == "concat13":
        return ConcatModel(13)
    if architecture in {"bigru", "bilstm"}:
        return SequenceModel(architecture)
    raise ValueError(architecture)


def timing_auxiliary_loss(logits: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hard_exact":
        return nn.functional.cross_entropy(logits.float(), target)
    log_probabilities = nn.functional.log_softmax(logits.float(), dim=1)
    if mode == "soft_pm1":
        distribution = torch.zeros_like(log_probabilities)
        for row, value in enumerate(target.tolist()):
            if value == 5:
                distribution[row, 5] = 1.0
                continue
            neighbours = [candidate for candidate in (value - 1, value, value + 1) if 0 <= candidate <= 4]
            raw_weights = torch.tensor([0.25 if candidate != value else 0.5 for candidate in neighbours], device=logits.device)
            raw_weights = raw_weights / raw_weights.sum()
            distribution[row, neighbours] = raw_weights
        return -(distribution * log_probabilities).sum(1).mean()
    if mode == "interval_mil_pm1":
        losses = []
        for row, value in enumerate(target.tolist()):
            if value == 5:
                losses.append(-log_probabilities[row, 5])
            else:
                neighbours = [candidate for candidate in (value - 1, value, value + 1) if 0 <= candidate <= 4]
                losses.append(-torch.logsumexp(log_probabilities[row, neighbours], dim=0))
        return torch.stack(losses).mean()
    raise ValueError(mode)


def timing_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    classes = sorted(set(true.tolist()))
    recalls = []
    for value in classes:
        mask = true == value
        recalls.append(float(np.mean(predicted[mask] == true[mask])))
    positive = true != 5
    negative = ~positive
    return {
        "exact_accuracy": float(np.mean(predicted == true)),
        "balanced_accuracy_present_classes": float(np.mean(recalls)),
        "all_none_accuracy_baseline": float(np.mean(true == 5)),
        "positive_exact_accuracy": float(np.mean(predicted[positive] == true[positive])) if np.any(positive) else None,
        "positive_pm1_accuracy": float(np.mean(np.abs(predicted[positive] - true[positive]) <= 1)) if np.any(positive) else None,
        "positive_mae_frames": float(np.mean(np.abs(predicted[positive] - true[positive]))) if np.any(positive) else None,
        "negative_none_recall": float(np.mean(predicted[negative] == 5)) if np.any(negative) else None,
        "n": int(len(true)),
        "positive_n": int(np.count_nonzero(positive)),
    }


@dataclass(frozen=True)
class RunSpec:
    name: str
    architecture: str
    temporal_shift: bool = False
    timing_loss: str = "none"


def evaluate(model, loader, samples, device, amp: str, offset_scale_um: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    indices = []
    count_true = []
    count_pred = []
    count_valid = []
    offset_errors = []
    offset_valid = []
    timing_true = []
    timing_pred = []
    with torch.inference_mode():
        for raw_batch in loader:
            cpu_indices = raw_batch["index"].numpy()
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items() if key != "index"}
            with amp_context(amp):
                outputs = model(batch["image"])
            scores.append(torch.sigmoid(outputs["binary"].float()).cpu().numpy())
            indices.append(cpu_indices)
            count_true.append(batch["count"].cpu().numpy())
            count_pred.append(outputs["count"].argmax(1).cpu().numpy())
            count_valid.append(batch["count_valid"].bool().cpu().numpy())
            offset_errors.append((torch.abs(outputs["offset"].float() - batch["offset"]).mean(2) * offset_scale_um).cpu().numpy())
            offset_valid.append(batch["offset_valid"].bool().cpu().numpy())
            timing_true.append(batch["timing"].cpu().numpy())
            timing_pred.append(outputs["timing"].argmax(1).cpu().numpy())
    ordered = np.concatenate(indices)
    probabilities = np.concatenate(scores)
    metrics = metrics_with_slices(samples, ordered, probabilities)
    count_mask = np.concatenate(count_valid)
    offset_mask = np.concatenate(offset_valid)
    count_true_array = np.concatenate(count_true)
    count_pred_array = np.concatenate(count_pred)
    offset_array = np.concatenate(offset_errors)
    metrics["auxiliary"] = {
        "child_count_accuracy": float(np.mean(count_true_array[count_mask] == count_pred_array[count_mask])),
        "child_count_valid_n": int(np.count_nonzero(count_mask)),
        "successor_offset_mae_um": float(np.mean(offset_array[offset_mask])),
        "successor_offset_valid_n": int(np.count_nonzero(offset_mask)),
    }
    metrics["timing"] = timing_metrics(np.concatenate(timing_true), np.concatenate(timing_pred))
    return metrics, ordered, probabilities


def train_spec(
    spec: RunSpec,
    path: Path,
    samples,
    targets,
    train_indices,
    val_indices,
    statistics,
    args,
) -> dict[str, Any]:
    run_dir = args.output_dir / "arms" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (run_dir / "DONE").exists() and (run_dir / "summary.json").is_file():
        print(f"[{spec.name}] reusing completed arm", flush=True)
        return json.loads((run_dir / "summary.json").read_text())
    seed_everything(args.seed)
    train_dataset = TemporalDataset(path, samples, targets, train_indices, spec.architecture, True, spec.temporal_shift,
                                    args.offset_scale_um, statistics["mean"], statistics["std"])
    val_dataset = TemporalDataset(path, samples, targets, val_indices, spec.architecture, False, False,
                                  args.offset_scale_um, statistics["mean"], statistics["std"])
    weights, sampler_metadata = sampling_weights(samples, train_indices, "binary_balanced", frozenset())
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), max(args.samples_per_epoch, len(train_indices)),
                                      replacement=True, generator=sampler_generator),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed)
    device = torch.device("cuda")
    model = build_model(spec.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    best_ap, best_epoch, stale = -math.inf, -1, 0
    curve = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for raw_batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items() if key != "index"}
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp):
                outputs = model(batch["image"])
                loss, _ = arm_losses("feedback", outputs, batch, args.count_loss_weight, args.offset_loss_weight)
                if spec.timing_loss != "none":
                    loss = loss + args.timing_loss_weight * timing_auxiliary_loss(outputs["timing"], batch["timing"], spec.timing_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics, _, _ = evaluate(model, val_loader, samples, device, args.amp, args.offset_scale_um)
        ap = float(metrics["parent_vs_all"]["auprc"])
        curve.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auprc": ap,
            "val_parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"],
            "val_parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"],
            "timing_positive_pm1_accuracy": metrics["timing"]["positive_pm1_accuracy"],
        })
        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        atomic_torch_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_ap": best_ap, "best_epoch": best_epoch, "stale": stale, "curve": curve,
            "spec": asdict(spec),
        }, run_dir / "checkpoint_latest.pt")
        print(f"[{spec.name}] epoch={epoch:03d} ap={ap:.5f} best={best_ap:.5f}@{best_epoch}", flush=True)
        if epoch + 1 >= min(20, args.epochs) and stale >= args.patience:
            break
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, ordered, probabilities = evaluate(model, val_loader, samples, device, args.amp, args.offset_scale_um)
    write_csv(run_dir / "learning_curve.csv", curve)
    write_csv(run_dir / "oof_sample_scores.csv", [{
        "sample_id": samples[int(index)].sample_id,
        "label": samples[int(index)].label,
        "category": samples[int(index)].category,
        "prediction": float(probability),
    } for index, probability in zip(ordered, probabilities)])
    summary = {
        "name": spec.name,
        "architecture": spec.architecture,
        "temporal_shift": spec.temporal_shift,
        "timing_loss": spec.timing_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(curve),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampler": sampler_metadata,
        "normalization": statistics,
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").touch()
    (run_dir / "checkpoint_latest.pt").unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.samples_per_epoch = 96
        args.num_workers = 0
        args.patience = 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("arms", "config", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.control_dir / "cache/samples.csv")
    aligned = np.load(args.aligned_t9, mmap_mode="r")
    if aligned.ndim != 5 or aligned.shape[0] != len(samples) or aligned.shape[1] != 9:
        raise ValueError(f"expected aligned T9 rows matching samples, got {aligned.shape} and {len(samples)}")
    with args.flow_alignment.open(newline="") as handle:
        flow_rows = list(csv.DictReader(handle))
    if len(flow_rows) != len(samples) or [row["sample_id"] for row in flow_rows] != [sample.sample_id for sample in samples]:
        raise ValueError("flow-alignment rows do not match T9 sample order")
    targets, target_summary = build_spatial_targets(samples, flow_rows, args.train_dir, aligned.shape[-3:], args.max_node_match_distance_um)
    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    val_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    if args.smoke:
        selected = []
        for partition in (train_indices, val_indices):
            selected.extend([index for index in partition if samples[index].label][:4])
            selected.extend([index for index in partition if not samples[index].label][:12])
        train_indices = [index for index in selected if samples[index].split == "train"]
        val_indices = [index for index in selected if samples[index].split == "val"]
    statistic_architectures = ("concat7",) if args.concat7_only else ("concat7", "concat13", "bigru")
    statistics = {
        architecture: estimate_channel_statistics(args.aligned_t9, train_indices, architecture)
        for architecture in statistic_architectures
    }
    if not args.concat7_only:
        statistics["bilstm"] = statistics["bigru"]
    atomic_json(args.output_dir / "config/normalization.json", statistics)
    atomic_json(args.output_dir / "config/spatial_targets.json", target_summary)
    atomic_json(args.output_dir / "config/run.json", vars(args) | {"output_dir": str(args.output_dir), "control_dir": str(args.control_dir),
                                                                    "train_dir": str(args.train_dir), "aligned_t9": str(args.aligned_t9),
                                                                    "flow_alignment": str(args.flow_alignment)})
    base_specs = BASE_SPECS[:1] if args.concat7_only else BASE_SPECS
    base_results = [
        train_spec(RunSpec(name, architecture), args.aligned_t9, samples, targets, train_indices, val_indices,
                   statistics[architecture], args)
        for name, architecture in base_specs
    ]
    winner = None
    timing_results = []
    if not args.concat7_only:
        t5_results = [result for result in base_results if result["architecture"] != "concat7"]
        winner = max(t5_results, key=lambda result: float(result["metrics"]["parent_vs_all"]["auprc"]))["architecture"]
        timing_results = [
            train_spec(RunSpec(f"{winner}_timing_{mode}", winner, True, mode), args.aligned_t9, samples, targets,
                       train_indices, val_indices, statistics[winner], args)
            for mode in TIMING_LOSSES
        ]
    control_ap = float(base_results[0]["metrics"]["parent_vs_all"]["auprc"])
    all_results = base_results + timing_results
    final = {
        "schema_version": 1,
        "fold": args.fold,
        "seed": args.seed,
        "concat7_only": args.concat7_only,
        "matched_contract": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "sampler": "binary_balanced",
            "augmentation": "shared spatial/intensity on raw frames before residual derivation",
            "normalization": "train-channel zscore",
            "successor_feedback": True,
            "count_loss_weight": args.count_loss_weight,
            "offset_loss_weight": args.offset_loss_weight,
            "timing_loss_weight": args.timing_loss_weight,
        },
        "temporal_winner": winner,
        "results": {result["name"]: result for result in all_results},
        "parent_vs_all_delta_vs_concat7": {
            result["name"]: float(result["metrics"]["parent_vs_all"]["auprc"]) - control_ap for result in all_results
        },
    }
    atomic_json(args.output_dir / "summary.json", final)
    (args.output_dir / "DONE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
