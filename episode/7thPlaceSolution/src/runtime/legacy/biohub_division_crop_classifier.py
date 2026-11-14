#!/usr/bin/env python3
"""Candidate-centred 3D temporal classifier for Biohub division parents.

This is deliberately an oracle-centre experiment.  It asks whether local image
appearance around an annotated cell can rank a division parent above annotated
normal cells and division daughters.  It does not create tracking candidates or
lineage edges; those are follow-up stages if this image-only control succeeds.
"""

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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler


SPACING_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
PARENT_ROLES = {2, 4}
DAUGHTER_ROLES = {3, 4}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    case_id: str
    stem: str
    frame: int
    z: int
    y: int
    x: int
    role: int
    label: int
    category: str
    matched_parent_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dataset-dir", type=Path, required=True)
    parser.add_argument("--data-identifier", default="nnUNetPlans_3d_fullres")
    parser.add_argument("--role-lookup", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--crop-shape", default="12,48,48")
    parser.add_argument("--train-normal-per-parent", type=int, default=8)
    parser.add_argument("--val-normal-per-parent", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--modes", default="center,t3")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_triplet(text: str) -> tuple[int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive Z,Y,X triplet: {text}")
    return values


def case_parts(case_id: str) -> tuple[str, int]:
    stem, frame_text = case_id.rsplit("_t", 1)
    return stem, int(frame_text)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        # Metric rows can intentionally differ between a heuristic baseline
        # and a learned head.  Preserve first-seen column order while taking
        # the union so DictWriter does not fail after the scientific work has
        # completed.
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_sample(split: str, case_id: str, row: Sequence[int], matched_parent_id: str = "") -> Sample:
    z, y, x, role = (int(value) for value in row)
    stem, frame = case_parts(case_id)
    category = "parent" if role in PARENT_ROLES else "daughter" if role in DAUGHTER_ROLES else "normal"
    return Sample(
        sample_id=f"{split}__{case_id}__z{z:03d}_y{y:03d}_x{x:03d}_r{role}",
        split=split,
        case_id=case_id,
        stem=stem,
        frame=frame,
        z=z,
        y=y,
        x=x,
        role=role,
        label=int(role in PARENT_ROLES),
        category=category,
        matched_parent_id=matched_parent_id,
    )


def build_samples(
    roles: dict[str, list[list[int]]],
    case_ids: set[str],
    split: str,
    normal_per_parent: int,
    smoke: bool,
) -> list[Sample]:
    parents: list[Sample] = []
    daughters: list[Sample] = []
    normals_by_case: dict[str, list[Sequence[int]]] = {}
    for case_id in sorted(case_ids):
        rows = roles.get(case_id, [])
        normals_by_case[case_id] = [row for row in rows if int(row[3]) == 1]
        for row in rows:
            role = int(row[3])
            if role in PARENT_ROLES:
                parents.append(make_sample(split, case_id, row))
            elif role in DAUGHTER_ROLES:
                daughters.append(make_sample(split, case_id, row))
    if smoke:
        parents = parents[: min(6, len(parents))]
        parent_stems = {sample.stem for sample in parents}
        daughters = [sample for sample in daughters if sample.stem in parent_stems][:12]

    selected_normals: dict[str, Sample] = {}
    for parent in parents:
        coordinate = np.asarray((parent.z, parent.y, parent.x), dtype=np.float64)
        candidates = []
        for row in normals_by_case.get(parent.case_id, []):
            other = np.asarray(row[:3], dtype=np.float64)
            distance = float(np.linalg.norm((other - coordinate) * SPACING_ZYX))
            candidates.append((distance, tuple(int(value) for value in row)))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, row in candidates[:normal_per_parent]:
            sample = make_sample(split, parent.case_id, row, matched_parent_id=parent.sample_id)
            selected_normals.setdefault(sample.sample_id, sample)

    samples = parents + daughters + list(selected_normals.values())
    if not parents:
        raise RuntimeError(f"no parent samples in {split}")
    return sorted(samples, key=lambda sample: (sample.case_id, sample.z, sample.y, sample.x, sample.role))


def crop_lazy(array: Any, centre: Sequence[int], crop_shape: Sequence[int]) -> np.ndarray:
    centre = np.asarray(centre, dtype=int)
    crop_shape = np.asarray(crop_shape, dtype=int)
    low = centre - crop_shape // 2
    high = low + crop_shape
    spatial_shape = np.asarray(array.shape[1:], dtype=int)
    source_low = np.maximum(low, 0)
    source_high = np.minimum(high, spatial_shape)
    destination_low = source_low - low
    destination_high = destination_low + (source_high - source_low)
    output = np.zeros((int(array.shape[0]), *crop_shape.tolist()), dtype=np.float32)
    if np.all(source_high > source_low):
        source_slice = tuple(slice(int(a), int(b)) for a, b in zip(source_low, source_high))
        destination_slice = tuple(slice(int(a), int(b)) for a, b in zip(destination_low, destination_high))
        output[(slice(None), *destination_slice)] = np.asarray(array[(slice(None), *source_slice)], dtype=np.float32)
    return output


def build_crop_cache(
    samples: Sequence[Sample],
    dataset_dir: Path,
    cache_dir: Path,
    crop_shape: Sequence[int],
    temporal_offsets: Sequence[int],
) -> tuple[Path, Path]:
    crop_path = cache_dir / "crops.float16.npy"
    metadata_path = cache_dir / "samples.csv"
    config_path = cache_dir / "cache_config.json"
    signature_payload = {
        "sample_ids": [sample.sample_id for sample in samples],
        "crop_shape_zyx": list(crop_shape),
        "temporal_offsets": list(temporal_offsets),
        "dataset_dir": str(dataset_dir),
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    if crop_path.exists() and metadata_path.exists() and config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("signature") == signature:
            return crop_path, metadata_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = crop_path.with_name(crop_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    crops = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(samples), len(temporal_offsets), *crop_shape),
    )
    from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalThreeFrame import (
        BiohubTemporalFiveFrameDataset,
        BiohubTemporalThreeFrameDataset,
    )

    dataset_class = BiohubTemporalFiveFrameDataset if len(temporal_offsets) == 5 else BiohubTemporalThreeFrameDataset
    temporal_dataset = dataset_class(str(dataset_dir), identifiers=sorted({s.case_id for s in samples}))
    by_case: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_case[sample.case_id].append((index, sample))
    started = time.perf_counter()
    for case_index, case_id in enumerate(sorted(by_case), start=1):
        data, _, _, properties = temporal_dataset.load_case(case_id)
        origin = np.asarray([int(axis[0]) for axis in properties["bbox_used_for_cropping"]], dtype=int)
        for index, sample in by_case[case_id]:
            local = np.asarray((sample.z, sample.y, sample.x), dtype=int) - origin
            crops[index] = crop_lazy(data, local, crop_shape).astype(np.float16)
        if case_index % 25 == 0 or case_index == len(by_case):
            print(
                f"[cache] cases={case_index}/{len(by_case)} samples={sum(len(by_case[c]) for c in sorted(by_case)[:case_index])} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    crops.flush()
    del crops
    os.replace(temporary, crop_path)
    write_csv(metadata_path, [asdict(sample) for sample in samples], fieldnames=list(asdict(samples[0])))
    atomic_json(config_path, {**signature_payload, "signature": signature, "dtype": "float16"})
    return crop_path, metadata_path


def _neighbor_case_id(sample: Sample, offset: int) -> str:
    frame_text = sample.case_id.rsplit("_t", 1)[1]
    frame = max(0, sample.frame + offset)
    return f"{sample.stem}_t{frame:0{len(frame_text)}d}"


def _distances_um(rows: Sequence[Sequence[int]], center_zyx: np.ndarray, exclude_center: bool) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    coordinates = np.asarray([row[:3] for row in rows], dtype=np.float64)
    vectors = (coordinates - center_zyx[None]) * SPACING_ZYX[None]
    distances = np.linalg.norm(vectors, axis=1)
    if exclude_center:
        keep = distances > 1e-6
        distances, vectors = distances[keep], vectors[keep]
    order = np.argsort(distances, kind="stable")
    return distances[order], vectors[order]


def _pad_knn(distances: np.ndarray, count: int, fill: float = 30.0) -> list[float]:
    values = distances[:count].tolist()
    return [float(value) for value in values] + [fill] * (count - len(values))


def build_geometry_features(
    samples: Sequence[Sample],
    roles: dict[str, list[list[int]]],
    train_indices: Sequence[int],
    output_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Build rotation-invariant oracle candidate geometry and train-normalize it."""

    radii = (5.0, 7.0, 10.0, 15.0)
    names = [
        *(f"prev_knn_{i}_um" for i in range(1, 4)),
        *(f"current_knn_{i}_um" for i in range(1, 4)),
        *(f"next_knn_{i}_um" for i in range(1, 5)),
        *(f"{time_name}_count_r{radius:g}" for time_name in ("prev", "current", "next") for radius in radii),
        "next_d1_um",
        "next_d2_um",
        "next_d2_minus_d1_um",
        "next_pair_sister_um",
        "next_pair_midpoint_residual_um",
        "next_pair_angle_cosine",
        *(f"next_minus_current_count_r{radius:g}" for radius in radii),
    ]
    raw_rows: list[list[float]] = []
    for sample in samples:
        center = np.asarray((sample.z, sample.y, sample.x), dtype=np.float64)
        temporal = []
        for offset in (-1, 0, 1):
            case_id = _neighbor_case_id(sample, offset)
            distances, vectors = _distances_um(roles.get(case_id, []), center, exclude_center=offset == 0)
            temporal.append((distances, vectors))
        prev_distances, _ = temporal[0]
        current_distances, _ = temporal[1]
        next_distances, next_vectors = temporal[2]
        features = [
            *_pad_knn(prev_distances, 3),
            *_pad_knn(current_distances, 3),
            *_pad_knn(next_distances, 4),
        ]
        counts = [[float(np.count_nonzero(distances <= radius)) for radius in radii] for distances, _ in temporal]
        features.extend(value for time_counts in counts for value in time_counts)
        if len(next_distances) >= 2:
            first, second = next_vectors[0], next_vectors[1]
            d1, d2 = float(next_distances[0]), float(next_distances[1])
            sister = float(np.linalg.norm(first - second))
            midpoint = float(np.linalg.norm((first + second) / 2.0))
            denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        elif len(next_distances) == 1:
            d1, d2, sister, midpoint, cosine = float(next_distances[0]), 30.0, 30.0, 30.0, 0.0
        else:
            d1, d2, sister, midpoint, cosine = 30.0, 30.0, 30.0, 30.0, 0.0
        features.extend((d1, d2, d2 - d1, sister, midpoint, cosine))
        features.extend(next_count - current_count for next_count, current_count in zip(counts[2], counts[1]))
        raw_rows.append(features)

    raw = np.asarray(raw_rows, dtype=np.float32)
    if raw.shape[1] != len(names):
        raise RuntimeError(f"geometry feature mismatch: {raw.shape[1]} vs {len(names)}")
    train_raw = raw[np.asarray(train_indices, dtype=np.int64)]
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = ((raw - mean) / std).astype(np.float32)
    np.save(output_dir / "cache" / "geometry_raw.float32.npy", raw)
    np.save(output_dir / "cache" / "geometry_normalized.float32.npy", normalized)
    atomic_json(
        output_dir / "cache" / "geometry_config.json",
        {"feature_names": names, "train_mean": mean.tolist(), "train_std": std.tolist(), "oracle_source": "role_lookup annotated centres"},
    )
    return normalized, names


class CropDataset(Dataset):
    def __init__(
        self,
        crop_path: Path,
        samples: Sequence[Sample],
        indices: Sequence[int],
        mode: str,
        augment: bool,
        geometry: np.ndarray,
    ):
        self.crop_path = crop_path
        self.samples = samples
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.augment = augment
        self.geometry = geometry
        self._crops: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _array(self) -> np.ndarray:
        if self._crops is None:
            self._crops = np.load(self.crop_path, mmap_mode="r")
        return self._crops

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        sample = self.samples[global_index]
        crop = np.asarray(self._array()[global_index], dtype=np.float32).copy()
        center = crop.shape[0] // 2
        if self.mode == "center":
            crop = crop[center : center + 1]
        elif self.mode in {
            "t3", "t3_geometry", "t3_diff_signed", "t3_diff_signed_abs",
            "t3_global_diff_signed_abs", "t3_local_diff_signed", "t3_local_diff_signed_abs",
        }:
            crop = crop[center - 1 : center + 2]
        elif self.mode == "t5":
            crop = crop[center - 2 : center + 3]
        else:
            raise ValueError(f"unsupported crop mode: {self.mode}")
        tensor = torch.from_numpy(crop)
        if self.augment:
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(-1,))
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(-2,))
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(-3,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                tensor = torch.rot90(tensor, rotations, dims=(-2, -1))
            # One shared affine intensity perturbation preserves temporal differences.
            scale = 0.95 + 0.10 * torch.rand(())
            shift = 0.05 * torch.randn(())
            tensor = tensor * scale + shift
        if self.mode in {
            "t3_diff_signed", "t3_diff_signed_abs", "t3_global_diff_signed_abs",
            "t3_local_diff_signed", "t3_local_diff_signed_abs",
        }:
            backward = (tensor[1:2] - tensor[0:1]) / math.sqrt(2.0)
            forward = (tensor[2:3] - tensor[1:2]) / math.sqrt(2.0)
            channels = [tensor, backward, forward]
            if self.mode in {"t3_diff_signed_abs", "t3_global_diff_signed_abs", "t3_local_diff_signed_abs"}:
                channels.extend((backward.abs(), forward.abs()))
            tensor = torch.cat(channels, dim=0)
        geometry = torch.from_numpy(np.asarray(self.geometry[global_index], dtype=np.float32))
        return tensor, geometry, torch.tensor(float(sample.label)), global_index


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int, int]):
        super().__init__()
        padding = tuple(value // 2 for value in kernel)
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SmallDivisionCNN(nn.Module):
    def __init__(self, in_channels: int, geometry_dim: int = 0):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.features = nn.Sequential(
            ConvBlock(in_channels, 24, (1, 3, 3)),
            nn.MaxPool3d((1, 2, 2)),
            ConvBlock(24, 48, (3, 3, 3)),
            nn.MaxPool3d((2, 2, 2)),
            ConvBlock(48, 96, (3, 3, 3)),
            nn.MaxPool3d((2, 2, 2)),
            ConvBlock(96, 128, (3, 3, 3)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.geometry_encoder = (
            nn.Sequential(nn.Linear(geometry_dim, 64), nn.SiLU(inplace=True), nn.Dropout(0.15), nn.Linear(64, 32), nn.SiLU(inplace=True))
            if geometry_dim
            else None
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(128 + (32 if geometry_dim else 0), 1))

    def forward(self, inputs: torch.Tensor, geometry: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.features(inputs).flatten(1)
        if self.geometry_encoder is not None:
            if geometry is None or geometry.shape[1] != self.geometry_dim:
                raise ValueError(f"expected geometry [B,{self.geometry_dim}], got {None if geometry is None else tuple(geometry.shape)}")
            image_features = torch.cat((image_features, self.geometry_encoder(geometry)), dim=1)
        return self.classifier(image_features).squeeze(1)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(np.int64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked.astype(bool)].sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(bool)
    prevalence = float(labels.mean()) if len(labels) else None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    tp = np.cumsum(ranked)
    fp = np.cumsum(~ranked)
    fn = int(labels.sum()) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(labels.sum()), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1)) if len(f1) else 0
    recall_at_precision = {}
    for target_precision in (0.25, 0.50):
        eligible = np.flatnonzero(precision >= target_precision)
        recall_at_precision[str(target_precision)] = float(recall[eligible].max()) if len(eligible) else 0.0
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "prevalence": prevalence,
        "auprc": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "best_f1": float(f1[best_index]) if len(f1) else None,
        "best_threshold": float(scores[order[best_index]]) if len(f1) else None,
        "best_precision": float(precision[best_index]) if len(f1) else None,
        "best_recall": float(recall[best_index]) if len(f1) else None,
        "recall_at_precision": recall_at_precision,
    }


def metrics_with_slices(samples: Sequence[Sample], indices: Sequence[int], scores: np.ndarray) -> dict[str, Any]:
    selected = [samples[int(index)] for index in indices]
    labels = np.asarray([sample.label for sample in selected], dtype=bool)
    categories = np.asarray([sample.category for sample in selected])
    result = {"parent_vs_all": binary_metrics(labels, scores)}
    for name, negative_category in (("parent_vs_normal", "normal"), ("parent_vs_daughter", "daughter")):
        mask = (categories == "parent") | (categories == negative_category)
        result[name] = binary_metrics(labels[mask], scores[mask])
    return result


def amp_context(amp: str):
    if amp == "none":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, geometry, targets, sample_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            geometry = geometry.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_context(amp):
                logits = model(inputs, geometry)
                batch_losses = criterion(logits.float(), targets)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            losses.append(batch_losses.cpu().numpy())
            indices.append(sample_indices.numpy())
    return np.concatenate(scores), np.concatenate(losses), np.concatenate(indices)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class CyclicTargetMixtureSampler(Sampler[int]):
    """Sample the base mixture with replacement and cycle targeted rows exactly once."""

    def __init__(
        self,
        base_weights: torch.Tensor,
        targeted_positions: Sequence[int],
        num_samples: int,
        targeted_draws_per_epoch: int,
        generator: torch.Generator,
    ) -> None:
        self.base_weights = torch.as_tensor(base_weights, dtype=torch.double).clone()
        self.targeted_positions = torch.as_tensor(list(targeted_positions), dtype=torch.long)
        self.num_samples = int(num_samples)
        self.targeted_draws_per_epoch = int(targeted_draws_per_epoch)
        self.generator = generator
        if not 0 < self.targeted_draws_per_epoch < self.num_samples:
            raise ValueError(
                f"targeted draws must be in (0, num_samples): "
                f"{self.targeted_draws_per_epoch} vs {self.num_samples}"
            )
        if len(self.targeted_positions) == 0:
            raise ValueError("cyclic targeted sampling requires targeted positions")
        self.base_weights[self.targeted_positions] = 0.0
        if not float(self.base_weights.sum()) > 0:
            raise ValueError("cyclic targeted sampling has no non-target base mass")
        self._cycle = torch.empty(0, dtype=torch.long)
        self._cursor = 0

    def __len__(self) -> int:
        return self.num_samples

    def _take_targeted(self, count: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = count
        while remaining:
            if self._cursor >= len(self._cycle):
                permutation = torch.randperm(len(self.targeted_positions), generator=self.generator)
                self._cycle = self.targeted_positions[permutation]
                self._cursor = 0
            take = min(remaining, len(self._cycle) - self._cursor)
            chunks.append(self._cycle[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
        return torch.cat(chunks)

    def __iter__(self):
        targeted = self._take_targeted(self.targeted_draws_per_epoch)
        base = torch.multinomial(
            self.base_weights,
            self.num_samples - self.targeted_draws_per_epoch,
            replacement=True,
            generator=self.generator,
        )
        target_slots = torch.sort(
            torch.randperm(self.num_samples, generator=self.generator)[: self.targeted_draws_per_epoch]
        ).values
        target_mask = torch.zeros(self.num_samples, dtype=torch.bool)
        target_mask[target_slots] = True
        combined = torch.empty(self.num_samples, dtype=torch.long)
        combined[target_mask] = targeted
        combined[~target_mask] = base
        return iter(combined.tolist())


def train_mode(
    mode: str,
    crop_path: Path,
    samples: Sequence[Sample],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    geometry: np.ndarray,
    args: argparse.Namespace,
    human_train_indices: Sequence[int] = (),
    human_loss_weight: float = 0.0,
) -> dict[str, Any]:
    mode_dir = args.output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    # T3 and T3+geometry intentionally share image-backbone initialization,
    # sampler draws, and augmentation worker seeds for a paired ablation.
    mode_seed = args.seed + {
        "center": 0,
        "t3": 1,
        "t3_geometry": 1,
        "t3_diff_signed": 1,
        "t3_diff_signed_abs": 1,
        "t3_global_diff_signed_abs": 1,
        "t3_local_diff_signed": 1,
        "t3_local_diff_signed_abs": 1,
        "t5": 2,
    }[mode]
    random.seed(mode_seed)
    np.random.seed(mode_seed)
    torch.manual_seed(mode_seed)
    torch.cuda.manual_seed_all(mode_seed)
    train_dataset = CropDataset(crop_path, samples, train_indices, mode, augment=True, geometry=geometry)
    val_dataset = CropDataset(crop_path, samples, val_indices, mode, augment=False, geometry=geometry)
    labels = np.asarray([samples[int(index)].label for index in train_indices], dtype=np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    hard_negative_path = getattr(args, "official_hard_negative_pool", None)
    hard_negative_fraction = float(getattr(args, "official_hard_negative_fraction", 0.0))
    targeted_negative_path = getattr(args, "official_targeted_negative_pool", None)
    targeted_negative_fraction = float(getattr(args, "official_targeted_negative_fraction", 0.0))
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError(f"official hard-negative fraction must be in [0,1], got {hard_negative_fraction}")
    if bool(hard_negative_path) != bool(hard_negative_fraction > 0):
        raise ValueError(
            "--official-hard-negative-pool and positive --official-hard-negative-fraction must be supplied together"
        )
    if not 0.0 <= targeted_negative_fraction <= 1.0:
        raise ValueError(
            f"official targeted-negative fraction must be in [0,1], got {targeted_negative_fraction}"
        )
    if bool(targeted_negative_path) != bool(targeted_negative_fraction > 0):
        raise ValueError(
            "--official-targeted-negative-pool and positive "
            "--official-targeted-negative-fraction must be supplied together"
        )
    if hard_negative_fraction + targeted_negative_fraction > 1.0:
        raise ValueError(
            "hard and targeted negative fractions must sum to <= 1: "
            f"{hard_negative_fraction} + {targeted_negative_fraction}"
        )
    hard_negative_positions: list[int] = []
    targeted_negative_positions: list[int] = []
    regular_negative_positions: list[int] = []
    if hard_negative_path or targeted_negative_path:
        def read_pool_ids(path: Path | None, name: str) -> set[str]:
            if path is None:
                return set()
            path = Path(path)
            if not (path.parent / "PREFLIGHT_PASSED").exists():
                raise ValueError(f"{name} pool has no PREFLIGHT_PASSED marker: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                ids = {
                    row["canonical_sample_id"]
                    for row in csv.DictReader(handle)
                    if row.get("usage") == "train"
                }
            if not ids:
                raise ValueError(f"{name} pool has no train rows: {path}")
            return ids

        hard_negative_path = Path(hard_negative_path) if hard_negative_path else None
        targeted_negative_path = Path(targeted_negative_path) if targeted_negative_path else None
        hard_ids = read_pool_ids(hard_negative_path, "hard-negative")
        targeted_ids = read_pool_ids(targeted_negative_path, "targeted-negative")
        overlap = hard_ids & targeted_ids
        if overlap:
            raise ValueError(f"hard and targeted negative pools overlap: {sorted(overlap)[:10]}")
        for position, index in enumerate(train_indices):
            sample = samples[int(index)]
            _, separator, canonical = sample.sample_id.partition("__")
            if not separator:
                raise ValueError(f"unexpected official sample_id: {sample.sample_id}")
            if sample.label:
                continue
            if canonical in targeted_ids:
                targeted_negative_positions.append(position)
            elif canonical in hard_ids:
                hard_negative_positions.append(position)
            else:
                regular_negative_positions.append(position)
        for name, positions, expected_ids in (
            ("hard-negative", hard_negative_positions, hard_ids),
            ("targeted-negative", targeted_negative_positions, targeted_ids),
        ):
            matched_ids = {
                samples[int(train_indices[position])].sample_id.partition("__")[2]
                for position in positions
            }
            if matched_ids != expected_ids and not bool(getattr(args, "smoke", False)):
                missing = sorted(expected_ids - matched_ids)
                raise ValueError(f"{name} pool rows do not map to official train samples: {missing[:10]}")
        if hard_negative_fraction > 0 and not hard_negative_positions:
            raise ValueError("hard-negative fraction is positive but no samples matched")
        if targeted_negative_fraction > 0 and not targeted_negative_positions:
            raise ValueError("targeted-negative fraction is positive but no samples matched")
        if not regular_negative_positions:
            raise ValueError(
                "negative mixture requires regular negatives: "
                f"hard={len(hard_negative_positions)} targeted={len(targeted_negative_positions)}"
            )
        weights = np.zeros(len(labels), dtype=np.float64)
        weights[labels == 1] = 0.5 / max(positives, 1)
        if hard_negative_positions:
            weights[np.asarray(hard_negative_positions)] = (
                0.5 * hard_negative_fraction / len(hard_negative_positions)
            )
        if targeted_negative_positions:
            weights[np.asarray(targeted_negative_positions)] = (
                0.5 * targeted_negative_fraction / len(targeted_negative_positions)
            )
        weights[np.asarray(regular_negative_positions)] = (
            0.5 * (1.0 - hard_negative_fraction - targeted_negative_fraction)
            / len(regular_negative_positions)
        )
    else:
        weights = np.where(labels == 1, 0.5 / max(positives, 1), 0.5 / max(negatives, 1))
    sampler_generator = torch.Generator().manual_seed(mode_seed)
    loader_generator = torch.Generator().manual_seed(mode_seed + 1000)
    official_draws = max(args.samples_per_epoch, len(train_indices))
    targeted_sampling = str(getattr(args, "official_targeted_sampling", "replacement"))
    targeted_draws_per_epoch = 0
    targeted_cycle_epochs = 0
    if targeted_sampling == "cycle_without_replacement":
        if not targeted_negative_positions or targeted_negative_fraction <= 0:
            raise ValueError("cyclic targeted sampling requires a non-empty targeted pool and positive fraction")
        targeted_draws_per_epoch = max(
            1, int(round(official_draws * 0.5 * targeted_negative_fraction))
        )
        targeted_cycle_epochs = int(math.ceil(len(targeted_negative_positions) / targeted_draws_per_epoch))
        sampler = CyclicTargetMixtureSampler(
            torch.as_tensor(weights, dtype=torch.double),
            targeted_negative_positions,
            official_draws,
            targeted_draws_per_epoch,
            sampler_generator,
        )
    elif targeted_sampling == "replacement":
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=official_draws,
            replacement=True,
            generator=sampler_generator,
        )
    else:
        raise ValueError(f"unknown official targeted sampling mode: {targeted_sampling}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
    )
    human_indices = [int(index) for index in human_train_indices]
    if human_loss_weight < 0:
        raise ValueError(f"human_loss_weight must be non-negative, got {human_loss_weight}")
    if bool(human_indices) != bool(human_loss_weight > 0):
        raise ValueError(
            "human_train_indices and positive human_loss_weight must be supplied together: "
            f"indices={len(human_indices)} weight={human_loss_weight}"
        )
    human_loader: DataLoader | None = None
    human_draws = 0
    human_batch_weight = human_loss_weight
    human_schedule: set[int] = set()
    if human_indices:
        human_labels = np.asarray([samples[index].label for index in human_indices], dtype=np.int64)
        human_positives = int(human_labels.sum())
        human_negatives = len(human_labels) - human_positives
        allow_single_class = bool(getattr(args, "allow_single_class_human_sidecar", False))
        if (human_positives == 0 or human_negatives == 0) and not allow_single_class:
            raise ValueError(
                "human sidecar must contain positive and negative event rows: "
                f"positive={human_positives} negative={human_negatives}"
            )
        if human_positives and human_negatives:
            human_weights = np.where(
                human_labels == 1,
                0.5 / human_positives,
                0.5 / human_negatives,
            )
        else:
            # The caller must opt in explicitly. Uniform weights preserve the
            # intended one-class auxiliary objective without divide-by-zero or
            # fabricating a dummy label from official data.
            human_weights = np.full(len(human_labels), 1.0 / len(human_labels))
        human_dataset = CropDataset(
            crop_path, samples, human_indices, mode, augment=True, geometry=geometry
        )
        requested_human_draws = int(getattr(args, "human_draws_per_epoch", 0))
        human_draws = official_draws if requested_human_draws <= 0 else requested_human_draws
        human_replacement = not bool(getattr(args, "human_sampling_without_replacement", False))
        if not human_replacement and human_draws > len(human_indices):
            raise ValueError(
                "human draws without replacement exceed sidecar size: "
                f"draws={human_draws} rows={len(human_indices)}"
            )
        human_sampler = WeightedRandomSampler(
            torch.as_tensor(human_weights, dtype=torch.double),
            num_samples=human_draws,
            replacement=human_replacement,
            generator=torch.Generator().manual_seed(mode_seed + 2000),
        )
        human_loader = DataLoader(
            human_dataset,
            batch_size=args.batch_size,
            sampler=human_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=worker_seed,
            generator=torch.Generator().manual_seed(mode_seed + 3000),
        )
        official_batches = len(train_loader)
        human_batches = len(human_loader)
        if human_batches > official_batches:
            raise ValueError(
                f"human batches exceed official batches: {human_batches} > {official_batches}"
            )
        human_schedule = {
            min(official_batches - 1, int((index + 0.5) * official_batches / human_batches))
            for index in range(human_batches)
        }
        if len(human_schedule) != human_batches:
            raise RuntimeError(
                f"human batch schedule collision: batches={human_batches} schedule={sorted(human_schedule)}"
            )
        if bool(getattr(args, "human_epoch_weight_matched", False)):
            human_batch_weight = human_loss_weight * official_batches / human_batches
    device = torch.device("cuda")
    in_channels = {
        "center": 1,
        "t3": 3,
        "t3_geometry": 3,
        "t3_diff_signed": 5,
        "t3_diff_signed_abs": 7,
        "t3_global_diff_signed_abs": 7,
        "t3_local_diff_signed": 5,
        "t3_local_diff_signed_abs": 7,
        "t5": 5,
    }[mode]
    geometry_dim = geometry.shape[1] if mode == "t3_geometry" else 0
    model = SmallDivisionCNN(in_channels, geometry_dim=geometry_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    train_criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_ap = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    curves: list[dict[str, Any]] = []
    train_loss_rows: list[dict[str, Any]] = []
    val_loss_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_values: dict[int, list[tuple[float, float]]] = defaultdict(list)
        human_epoch_values: dict[int, list[tuple[float, float]]] = defaultdict(list)
        human_iterator = iter(human_loader) if human_loader is not None else None
        for batch_index, (inputs, batch_geometry, targets, sample_indices) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            batch_geometry = batch_geometry.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp):
                logits = model(inputs, batch_geometry)
                per_sample_loss = train_criterion(logits.float(), targets)
                official_loss = per_sample_loss.mean()
                human_per_sample_loss = None
                human_logits = None
                human_sample_indices = None
                if human_iterator is not None and batch_index in human_schedule:
                    try:
                        human_inputs, human_geometry, human_targets, human_sample_indices = next(human_iterator)
                    except StopIteration as error:
                        raise RuntimeError("human sampler exhausted before official sampler") from error
                    human_inputs = human_inputs.to(device, non_blocking=True)
                    human_geometry = human_geometry.to(device, non_blocking=True)
                    human_targets = human_targets.to(device, non_blocking=True)
                    human_logits = model(human_inputs, human_geometry)
                    human_per_sample_loss = train_criterion(human_logits.float(), human_targets)
                    loss = official_loss + human_batch_weight * human_per_sample_loss.mean()
                else:
                    loss = official_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
            loss_values = per_sample_loss.detach().cpu().numpy()
            for index, probability, sample_loss in zip(sample_indices.tolist(), probabilities, loss_values):
                epoch_values[int(index)].append((float(probability), float(sample_loss)))
            if human_per_sample_loss is not None and human_logits is not None and human_sample_indices is not None:
                human_probabilities = torch.sigmoid(human_logits.detach().float()).cpu().numpy()
                human_loss_values = human_per_sample_loss.detach().cpu().numpy()
                for index, probability, sample_loss in zip(
                    human_sample_indices.tolist(), human_probabilities, human_loss_values
                ):
                    human_epoch_values[int(index)].append((float(probability), float(sample_loss)))
        scheduler.step()
        for training_source, source_values, source_weight in (
            ("official", epoch_values, 1.0),
            ("human_sidecar", human_epoch_values, human_batch_weight),
        ):
            for index, values in source_values.items():
                sample = samples[index]
                train_loss_rows.append({
                    "epoch": epoch,
                    "split": "train" if training_source == "official" else "human_train",
                    "mode": mode,
                    "sample_id": sample.sample_id,
                    "stem": sample.stem,
                    "frame": sample.frame,
                    "category": sample.category,
                    "label": sample.label,
                    "prediction": float(np.mean([value[0] for value in values])),
                    "loss": float(np.mean([value[1] for value in values])),
                    "fold": args.fold,
                    "draws": len(values),
                    "training_source": training_source,
                    "loss_weight": source_weight,
                    "event_mask": 1,
                })

        val_scores, val_losses, ordered_val_indices = evaluate(model, val_loader, train_criterion, device, args.amp)
        val_metrics = metrics_with_slices(samples, ordered_val_indices, val_scores)
        val_ap = float(val_metrics["parent_vs_all"]["auprc"])
        official_train_mean_loss = float(
            np.mean([value[1] for values in epoch_values.values() for value in values])
        )
        human_train_mean_loss = (
            float(np.mean([value[1] for values in human_epoch_values.values() for value in values]))
            if human_epoch_values
            else None
        )
        epoch_human_coefficient = (
            human_batch_weight * len(human_schedule) / max(len(train_loader), 1)
            if human_loader is not None
            else 0.0
        )
        train_mean_loss = official_train_mean_loss + (
            epoch_human_coefficient * human_train_mean_loss if human_train_mean_loss is not None else 0.0
        )
        val_mean_loss = float(val_losses.mean())
        curves.append(
            {
                "epoch": epoch,
                "mode": mode,
                "train_loss": train_mean_loss,
                "official_train_loss": official_train_mean_loss,
                "human_train_loss": human_train_mean_loss,
                "human_loss_weight": human_loss_weight,
                "val_loss": val_mean_loss,
                "val_auprc": val_ap,
                "val_roc_auc": val_metrics["parent_vs_all"]["roc_auc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        for index, score, sample_loss in zip(ordered_val_indices, val_scores, val_losses):
            sample = samples[int(index)]
            val_loss_rows.append(
                {
                    "epoch": epoch,
                    "split": "val",
                    "mode": mode,
                    "sample_id": sample.sample_id,
                    "stem": sample.stem,
                    "frame": sample.frame,
                    "category": sample.category,
                    "label": sample.label,
                    "prediction": float(score),
                    "loss": float(sample_loss),
                    "fold": args.fold,
                    "draws": 1,
                    "training_source": "official_validation",
                    "loss_weight": 1.0,
                    "event_mask": 1,
                }
            )
        improved = val_ap > best_ap + 1e-8
        if improved:
            best_ap = val_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mode": mode, "metrics": val_metrics},
                mode_dir / "checkpoint_best.pt",
            )
        else:
            epochs_without_improvement += 1
        print(
            f"[{mode}] epoch={epoch:03d} train_loss={train_mean_loss:.5f} val_loss={val_mean_loss:.5f} "
            f"val_ap={val_ap:.5f} best={best_ap:.5f}@{best_epoch} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
        checkpoint_selection = getattr(args, "checkpoint_selection", "best_auprc")
        min_target_cycles = int(getattr(args, "official_targeted_min_cycles", 0))
        minimum_epochs = min(20, args.epochs)
        if targeted_sampling == "cycle_without_replacement" and min_target_cycles > 0:
            minimum_epochs = max(
                minimum_epochs,
                targeted_cycle_epochs * min_target_cycles + args.patience,
            )
        if (
            checkpoint_selection == "best_auprc"
            and epoch + 1 >= min(minimum_epochs, args.epochs)
            and epochs_without_improvement >= args.patience
        ):
            print(f"[{mode}] early stop at epoch {epoch}", flush=True)
            break

    torch.save({"model": model.state_dict(), "epoch": curves[-1]["epoch"], "mode": mode}, mode_dir / "checkpoint_final.pt")
    selected_checkpoint = (
        mode_dir / "checkpoint_final.pt"
        if getattr(args, "checkpoint_selection", "best_auprc") == "fixed_final"
        else mode_dir / "checkpoint_best.pt"
    )
    checkpoint = torch.load(selected_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_scores, final_losses, final_indices = evaluate(model, val_loader, train_criterion, device, args.amp)
    final_metrics = metrics_with_slices(samples, final_indices, final_scores)
    score_rows = []
    for index, score, sample_loss in zip(final_indices, final_scores, final_losses):
        sample = samples[int(index)]
        score_rows.append(
            {
                "exp_id": os.environ.get("BIOHUB_EXP_ID", args.output_dir.name),
                "split": "val",
                "mode": mode,
                "sample_id": sample.sample_id,
                "stem": sample.stem,
                "frame": sample.frame,
                "fold": args.fold,
                "category": sample.category,
                "label": sample.label,
                "prediction": float(score),
                "loss": float(sample_loss),
                "sample_score": float(score if sample.label else 1.0 - score),
                "training_source": "official_validation",
            }
        )
    write_csv(mode_dir / "learning_curve.csv", curves)
    write_csv(mode_dir / "oof_sample_scores.csv", score_rows)
    atomic_json(
        mode_dir / "summary.json",
        {
            "mode": mode,
            "best_epoch": best_epoch,
            "epochs_completed": len(curves),
            "runtime_seconds": time.perf_counter() - started,
            "train_samples": len(train_indices),
            "human_train_samples": len(human_indices),
            "human_loss_weight": human_loss_weight,
            "human_batch_weight": human_batch_weight,
            "official_draws_per_epoch": official_draws,
            "official_hard_negative_pool": str(hard_negative_path) if hard_negative_path else None,
            "official_hard_negative_fraction": hard_negative_fraction,
            "official_hard_negative_samples": len(hard_negative_positions),
            "official_targeted_negative_pool": str(targeted_negative_path) if targeted_negative_path else None,
            "official_targeted_negative_fraction": targeted_negative_fraction,
            "official_targeted_negative_samples": len(targeted_negative_positions),
            "official_targeted_sampling": targeted_sampling,
            "official_targeted_draws_per_epoch": targeted_draws_per_epoch,
            "official_targeted_cycle_epochs": targeted_cycle_epochs,
            "official_targeted_min_cycles": int(getattr(args, "official_targeted_min_cycles", 0)),
            "official_regular_negative_samples": (
                len(regular_negative_positions)
                if hard_negative_path or targeted_negative_path
                else negatives
            ),
            "human_draws_per_epoch": human_draws,
            "human_batches_per_epoch": len(human_schedule),
            "checkpoint_selection": getattr(args, "checkpoint_selection", "best_auprc"),
            "selected_checkpoint": str(selected_checkpoint),
            "val_samples": len(val_indices),
            "metrics": final_metrics,
        },
    )
    return {
        "mode": mode,
        "mode_seed": mode_seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(curves),
        "runtime_seconds": time.perf_counter() - started,
        "metrics": final_metrics,
        "train_loss_rows": train_loss_rows,
        "val_loss_rows": val_loss_rows,
        "score_rows": score_rows,
        "final_indices": final_indices,
        "final_scores": final_scores,
        "final_losses": final_losses,
        "human_train_samples": len(human_indices),
        "human_loss_weight": human_loss_weight,
    }


def plot_curves(output_dir: Path, results: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for result in results:
        rows = list(csv.DictReader((output_dir / result["mode"] / "learning_curve.csv").open()))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label=f"{result['mode']} train")
        axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], linestyle="--", label=f"{result['mode']} val")
        axes[1].plot(epochs, [float(row["val_auprc"]) for row in rows], label=result["mode"])
    axes[0].set_title("Binary cross entropy")
    axes[1].set_title("Parent vs all validation AUPRC")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "plots" / "division_classifier_learning_curves.png", dpi=170)
    plt.close(figure)


def save_reviews(
    crop_path: Path,
    samples: Sequence[Sample],
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crops = np.load(crop_path, mmap_mode="r")
    ranked = sorted(
        zip(result["final_indices"], result["final_scores"], result["final_losses"]),
        key=lambda item: float(item[2]),
    )
    for review_name, selected in (("best_loss_top10", ranked[:10]), ("worst_loss_top10", ranked[-10:][::-1])):
        root = output_dir / "sample_reviews" / review_name
        for rank, (index, score, loss) in enumerate(selected, start=1):
            sample = samples[int(index)]
            sample_dir = root / f"{rank:02d}_{sample.sample_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            crop = np.asarray(crops[int(index)], dtype=np.float32)
            channel_count = crop.shape[0]
            offsets = range(-(channel_count // 2), channel_count // 2 + 1)
            figure, axes = plt.subplots(1, channel_count, figsize=(3.3 * channel_count, 3.6))
            axes = np.atleast_1d(axes)
            for channel, (axis, offset) in enumerate(zip(axes, offsets)):
                mip = crop[channel].max(axis=0)
                low, high = np.percentile(mip, (1, 99.5))
                axis.imshow(mip, cmap="gray", vmin=low, vmax=max(high, low + 1e-6))
                axis.set_title("t" if offset == 0 else f"t{offset:+d}")
                axis.axis("off")
            figure.suptitle(
                f"{sample.category} label={sample.label} pred={float(score):.4f} loss={float(loss):.4f}"
            )
            figure.tight_layout()
            figure.savefig(sample_dir / "division_triplet.png", dpi=150)
            plt.close(figure)
            metadata = {**asdict(sample), "prediction": float(score), "loss": float(loss), "mode": result["mode"]}
            atomic_json(sample_dir / "metadata.json", metadata)
            (sample_dir / "report.txt").write_text(
                "Biohub division-parent candidate review.\n"
                f"case={sample.case_id} category={sample.category} label={sample.label} "
                f"prediction={float(score):.6f} loss={float(loss):.6f}\n"
            )


def update_manifest(output_dir: Path, summary: dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.yaml"
    if not manifest_path.exists():
        return
    try:
        import yaml
    except ImportError:
        return
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    manifest["status"] = "completed"
    manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["config_path"] = str(output_dir / "config" / "run_config.json")
    manifest["result_summary"] = summary.get("result_summary", "")
    manifest["metrics"] = {
        mode: payload["metrics"]["parent_vs_all"] for mode, payload in summary["modes"].items()
    }
    manifest["sample_loss_tracking"]["train_loss_by_epoch_path"] = "sample_losses/train_sample_loss_by_epoch.csv"
    manifest["sample_loss_tracking"]["valid_loss_by_epoch_path"] = "sample_losses/valid_sample_loss_by_epoch.csv"
    manifest["sample_loss_tracking"]["required_columns"] = [
        "epoch", "split", "mode", "sample_id", "stem", "frame", "category", "label", "prediction", "loss", "fold", "draws"
    ]
    manifest["sample_loss_tracking"]["final_validation_review"] = {
        "worst_loss_top_k": 10,
        "best_loss_top_k": 10,
        "worst_loss_dir": "sample_reviews/worst_loss_top10",
        "best_loss_dir": "sample_reviews/best_loss_top10",
        "representative_slice_rule": "Render Z-max projections for t-1, t, and t+1 at the annotated candidate centre.",
        "required_files_per_sample": ["division_triplet.png", "metadata.json", "report.txt"],
    }
    manifest["sample_score_tracking"]["oof_scores_path"] = "sample_scores/oof_sample_scores.csv"
    manifest["sample_score_tracking"]["metadata_path"] = "cache/samples.csv"
    manifest["sample_score_tracking"]["required_columns"] = [
        "exp_id", "split", "mode", "sample_id", "stem", "frame", "fold", "category", "label", "prediction", "loss", "sample_score"
    ]
    manifest["sample_score_tracking"]["delta_vs_parent"]["compare_to_parent"] = False
    manifest["gt_outcome_tracking"]["required_for_training"] = False
    manifest["gt_outcome_tracking"]["delta_vs_parent"]["compare_to_parent"] = False
    manifest["context_review"] = {
        "status": "reviewed",
        "paths": [
            "records/ideas/idea_20260713_image-assisted-division-detection.md",
            "records/summaries/paper_malin_mayor_2023_automated_reconstruction_whole_embryo_cell_lineages_sparse_annotations.md",
            "records/summaries/github_repo_royerlab_kaggle_cell_tracking_competition.md",
            "participant_signals/notable_comments.md",
        ],
        "prior_findings": "Dense Parent head remained at prevalence despite focused sampling/loss ablation.",
        "discussion_signals": "Rule-based tracking is strong; learned division evidence should be evaluated by a division-specific slice.",
        "host_signals": "No related host fact was recorded.",
        "participant_signals": "Strong participants explicitly asked for division Jaccard while warning that learned 3D models are expensive.",
        "integrated_summary": "Use a small candidate-centred temporal classifier as an image-only gate before graph integration.",
    }
    manifest["lineage"] = {
        "parent_exp_id": "biohub_20260718_t3_role_focused_ablation_v001",
        "relation_type": "derived_from",
        "copied_from": ["division_role_lookup.json", "Dataset701 fold 0 split"],
        "changed_paths": ["tools/biohub_division_crop_classifier.py"],
        "diff_summary": "Replace dense Parent/Daughter heatmap supervision with oracle-centre temporal crop classification.",
        "score_change_summary": "See metrics and summary.json.",
        "parent_git_commit": None,
        "child_git_commit": manifest.get("execution_env", {}).get("git_commit"),
        "code_snapshot_path": None,
        "patch_path": None,
    }
    manifest["next_action_candidate"] = (
        "If T3 exceeds the predeclared gate, add detected-centre jitter and daughter-pair geometry; otherwise audit label timing."
    )
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))


def main() -> int:
    args = parse_args()
    args.crop_shape = parse_triplet(args.crop_shape)
    args.modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if any(mode not in {
        "center", "t3", "t5", "t3_geometry", "t3_diff_signed", "t3_diff_signed_abs",
        "t3_global_diff_signed_abs", "t3_local_diff_signed", "t3_local_diff_signed_abs",
    } for mode in args.modes):
        raise ValueError(f"unsupported modes: {args.modes}")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 128)
        args.patience = args.epochs
        args.num_workers = min(args.num_workers, 2)
    if args.overwrite and args.output_dir.exists():
        manifest = args.output_dir / "manifest.yaml"
        preserved_manifest = manifest.read_bytes() if manifest.exists() else None
        shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True)
        if preserved_manifest is not None:
            manifest.write_bytes(preserved_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("config", "cache", "plots", "sample_losses", "sample_scores", "sample_reviews"):
        (args.output_dir / directory).mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    lookup = json.loads(args.role_lookup.read_text())
    roles = lookup.get("cases", lookup)
    splits = json.loads((args.preprocessed_dataset_dir / "splits_final.json").read_text())
    split = splits[args.fold]
    train_samples = build_samples(
        roles, set(split["train"]), "train", args.train_normal_per_parent, args.smoke
    )
    val_samples = build_samples(
        roles, set(split["val"]), "val", args.val_normal_per_parent, args.smoke
    )
    samples = train_samples + val_samples
    train_indices = list(range(len(train_samples)))
    val_indices = list(range(len(train_samples), len(samples)))
    counts = {
        partition: {
            category: sum(sample.split == partition and sample.category == category for sample in samples)
            for category in ("parent", "daughter", "normal")
        }
        for partition in ("train", "val")
    }
    dataset_dir = args.preprocessed_dataset_dir / args.data_identifier
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    config = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preprocessed_dataset_dir": str(args.preprocessed_dataset_dir),
        "data_identifier": args.data_identifier,
        "role_lookup": str(args.role_lookup),
        "fold": args.fold,
        "crop_shape_zyx": list(args.crop_shape),
        "crop_physical_extent_zyx_um": (np.asarray(args.crop_shape) * SPACING_ZYX).tolist(),
        "train_normal_per_parent": args.train_normal_per_parent,
        "val_normal_per_parent": args.val_normal_per_parent,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "modes": list(args.modes),
        "amp": args.amp,
        "smoke": args.smoke,
        "sample_counts": counts,
        "label_semantics": "positive=division parent at centre frame; daughter and matched annotated normal are negative",
    }
    atomic_json(args.output_dir / "config" / "run_config.json", config)
    print(json.dumps({"sample_counts": counts, "crop_shape": args.crop_shape}, indent=2), flush=True)

    temporal_offsets = (-2, -1, 0, 1, 2) if "t5" in args.modes else (-1, 0, 1)
    crop_path, _ = build_crop_cache(
        samples, dataset_dir, args.output_dir / "cache", args.crop_shape, temporal_offsets
    )
    geometry, geometry_names = build_geometry_features(samples, roles, train_indices, args.output_dir)
    config["temporal_offsets"] = list(temporal_offsets)
    config["geometry_feature_names"] = geometry_names
    atomic_json(args.output_dir / "config" / "run_config.json", config)
    results = [
        train_mode(mode, crop_path, samples, train_indices, val_indices, geometry, args)
        for mode in args.modes
    ]
    all_train_rows = [row for result in results for row in result["train_loss_rows"]]
    all_val_rows = [row for result in results for row in result["val_loss_rows"]]
    all_score_rows = [row for result in results for row in result["score_rows"]]
    write_csv(args.output_dir / "sample_losses" / "train_sample_loss_by_epoch.csv", all_train_rows)
    write_csv(args.output_dir / "sample_losses" / "valid_sample_loss_by_epoch.csv", all_val_rows)
    write_csv(args.output_dir / "sample_scores" / "oof_sample_scores.csv", all_score_rows)
    plot_curves(args.output_dir, results)
    t3_result = next((result for result in results if result["mode"] == "t3"), results[-1])
    review_result = max(results, key=lambda result: result["metrics"]["parent_vs_all"]["auprc"])
    save_reviews(crop_path, samples, review_result, args.output_dir)

    mode_summaries = {
        result["mode"]: {
            "mode_seed": result["mode_seed"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "runtime_seconds": result["runtime_seconds"],
            "metrics": result["metrics"],
        }
        for result in results
    }
    center_ap = mode_summaries.get("center", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    t3_ap = mode_summaries.get("t3", {}).get("metrics", {}).get("parent_vs_all", {}).get("auprc")
    prevalence = mode_summaries[t3_result["mode"]]["metrics"]["parent_vs_all"]["prevalence"]
    gate_passed = bool(
        t3_ap is not None
        and t3_ap >= 2 * prevalence
        and (center_ap is None or t3_ap - center_ap >= 0.05)
    )
    summary = {
        "sample_counts": counts,
        "modes": mode_summaries,
        "t3_minus_center_auprc": None if center_ap is None or t3_ap is None else t3_ap - center_ap,
        "predeclared_gate_passed": gate_passed,
        "result_summary": (
            f"T3 Parent-vs-all AUPRC={t3_ap:.4f}, centre={center_ap:.4f}, prevalence={prevalence:.4f}, "
            f"gate_passed={gate_passed}"
            if center_ap is not None and t3_ap is not None
            else "Division crop classifier completed; see mode metrics."
        ),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    update_manifest(args.output_dir, summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
