#!/usr/bin/env python3
"""Lightweight 3D U-Net ablation for Biohub Division classification.

The model keeps the existing local-flow-aligned seven-channel image input, but
adds spatial supervision from the true GEFF successors.  The binary and
multitask arms use exactly the same network and sampler; only the auxiliary
loss coefficients differ.
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
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from biohub_division_crop_classifier import (
    SPACING_ZYX,
    Sample,
    atomic_json,
    metrics_with_slices,
    worker_seed,
    write_csv,
)
from biohub_division_detected_geometry_classifier import load_peaks
from biohub_division_flow_aligned_diff_classifier import (
    SparseLocalFlow,
    build_aligned_caches,
    neighbor_case_id,
    read_samples,
)
from biohub_gt_supervision_exclusions import (
    contract_metadata,
    load_optional,
    resolve_configured_manifest,
)


ROLE_TO_INDEX = {"normal": 0, "daughter": 1, "parent": 2}


@dataclass(frozen=True)
class SpatialTarget:
    role_index: int
    child_count_index: int
    child_count_valid: bool
    heatmap_valid: bool
    node_distance_um: float
    children_zyx_aligned: tuple[tuple[float, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--arms", nargs="+",
        choices=("binary", "multitask", "nnunet_random", "nnunet_pretrained"),
        default=("binary", "multitask"),
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--heatmap-sigma-um", type=float, default=2.0)
    parser.add_argument(
        "--max-node-match-distance-um",
        type=float,
        default=7.0,
        help=(
            "Maximum detector-centre to same-role GEFF-node distance.  This must "
            "match the 7 um detector-to-role labelling gate used to build the cohort."
        ),
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=("current", "strong_all", "strong_targeted"),
        default="current",
    )
    parser.add_argument(
        "--sampler-profile",
        choices=("binary_balanced", "division_stratified"),
        default="binary_balanced",
    )
    parser.add_argument("--hard-negative-score-csv", type=Path)
    parser.add_argument("--hard-negative-quantile", type=float, default=0.75)
    parser.add_argument("--role-loss-weight", type=float, default=0.25)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--nnunet-batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pretrained-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--sigma-um", type=float, default=20.0)
    parser.add_argument("--local-radius-um", type=float, default=50.0)
    parser.add_argument("--exclude-radius-um", type=float, default=3.0)
    parser.add_argument("--residual-gate-um", type=float, default=7.0)
    parser.add_argument("--min-local-support", type=int, default=3)
    parser.add_argument("--max-local-support", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_heatmap(
    shape: Sequence[int],
    centres_zyx: Sequence[Sequence[float]],
    sigma_um: float,
) -> np.ndarray:
    """Render max-composed physical-space Gaussians at sub-voxel centres."""
    shape_array = np.asarray(shape, dtype=np.int64)
    heatmap = np.zeros(tuple(shape_array), dtype=np.float32)
    sigma_voxels = sigma_um / SPACING_ZYX
    radii = np.maximum(1, np.ceil(3.0 * sigma_voxels).astype(np.int64))
    for centre_value in centres_zyx:
        centre = np.asarray(centre_value, dtype=np.float64)
        low = np.maximum(np.floor(centre).astype(np.int64) - radii, 0)
        high = np.minimum(np.floor(centre).astype(np.int64) + radii + 2, shape_array)
        if np.any(high <= low):
            continue
        grids = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        squared = sum(
            np.square((axis.astype(np.float64) - centre[index]) * SPACING_ZYX[index])
            for index, axis in enumerate(grids)
        )
        patch = np.exp(-0.5 * squared / (sigma_um * sigma_um)).astype(np.float32)
        destination = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        heatmap[destination] = np.maximum(heatmap[destination], patch)
    return heatmap


def _graph_for_stem(train_dir: Path, stem: str):
    import geff

    path = train_dir / f"{stem}.geff"
    if not path.exists():
        raise FileNotFoundError(path)
    graph, _ = geff.read(
        str(path), node_props=["t", "z", "y", "x"], structure_validation=False
    )
    return graph


def graph_node_category(graph: Any, node: Any) -> str:
    if int(graph.out_degree(node)) >= 2:
        return "parent"
    is_daughter = any(int(graph.out_degree(parent)) >= 2 for parent in graph.predecessors(node))
    return "daughter" if is_daughter else "normal"


def build_spatial_targets(
    samples: Sequence[Sample],
    flow_rows: Sequence[dict[str, Any]],
    train_dir: Path,
    crop_shape: Sequence[int],
    max_match_distance_um: float = 7.0,
    gt_exclusion_manifest: Path | None = None,
) -> tuple[list[SpatialTarget], dict[str, Any]]:
    """Resolve each rounded role-lookup point to GEFF and follow true edges."""
    if len(samples) != len(flow_rows):
        raise ValueError(f"sample/flow row mismatch: {len(samples)} vs {len(flow_rows)}")
    graphs: dict[str, Any] = {}
    frame_nodes: dict[tuple[str, int], list[Any]] = {}
    targets: list[SpatialTarget] = []
    counts: Counter[str] = Counter()
    crop_center = np.asarray(crop_shape, dtype=np.float64) // 2
    gt_exclusion_manifest = resolve_configured_manifest(gt_exclusion_manifest)
    exclusions = load_optional(gt_exclusion_manifest)
    for sample, flow_row in zip(samples, flow_rows):
        if flow_row["sample_id"] != sample.sample_id:
            raise ValueError(f"flow row order mismatch for {sample.sample_id}")
        if sample.stem not in graphs:
            graphs[sample.stem] = _graph_for_stem(train_dir, sample.stem)
        graph = graphs[sample.stem]
        key = (sample.stem, sample.frame)
        if key not in frame_nodes:
            frame_nodes[key] = [
                node for node in graph.nodes if int(graph.nodes[node]["t"]) == sample.frame
            ]
        nodes = [
            node for node in frame_nodes[key]
            if graph_node_category(graph, node) == sample.category
        ]
        if not nodes:
            raise RuntimeError(
                f"no {sample.category} GT nodes at {sample.stem} frame {sample.frame}"
            )
        sample_zyx = np.asarray((sample.z, sample.y, sample.x), dtype=np.float64)
        node_positions = np.asarray(
            [[graph.nodes[node][axis] for axis in ("z", "y", "x")] for node in nodes],
            dtype=np.float64,
        )
        distances = np.linalg.norm((node_positions - sample_zyx[None]) * SPACING_ZYX[None], axis=1)
        best = int(np.argmin(distances))
        node = nodes[best]
        node_distance_um = float(distances[best])
        if node_distance_um > max_match_distance_um:
            raise RuntimeError(
                f"role point does not resolve to GEFF: {sample.sample_id}, distance={node_distance_um:.3f}um"
            )
        children = [
            child for child in graph.successors(node)
            if int(graph.nodes[child]["t"]) == sample.frame + 1
        ]
        temporal_target_excluded = bool(
            exclusions is not None
            and exclusions.has_excluded_outgoing(sample.stem, int(node), {
                int(node): [int(child) for child in graph.successors(node)]
            })
        )
        next_vector_um = np.asarray(
            [flow_row[f"next_local_d{axis}_um"] for axis in ("z", "y", "x")],
            dtype=np.float64,
        )
        aligned_centres = []
        for child in (() if temporal_target_excluded else children):
            child_zyx = np.asarray(
                [graph.nodes[child][axis] for axis in ("z", "y", "x")], dtype=np.float64
            )
            aligned = crop_center + (child_zyx - sample_zyx) - next_vector_um / SPACING_ZYX
            aligned_centres.append(tuple(float(value) for value in aligned))
        role_index = ROLE_TO_INDEX[sample.category]
        child_count = len(children)
        max_frame = max(int(graph.nodes[other]["t"]) for other in graph.nodes)
        target_known = sample.frame < max_frame
        count_valid = (
            target_known and not temporal_target_excluded and child_count in {0, 1, 2}
        )
        target = SpatialTarget(
            role_index=role_index,
            child_count_index=min(child_count, 2),
            child_count_valid=count_valid,
            heatmap_valid=target_known and not temporal_target_excluded,
            node_distance_um=node_distance_um,
            children_zyx_aligned=tuple(aligned_centres),
        )
        targets.append(target)
        counts[f"role_{sample.category}"] += 1
        counts[f"children_{child_count}"] += 1
        counts["heatmap_valid"] += int(target.heatmap_valid)
        counts["count_valid"] += int(target.child_count_valid)
        counts["gt_excluded_temporal_components"] += int(temporal_target_excluded)
    summary = {
        "counts": dict(sorted(counts.items())),
        "max_match_distance_um": max_match_distance_um,
        "max_node_match_distance_um": max(target.node_distance_um for target in targets),
        "mean_node_match_distance_um": float(np.mean([target.node_distance_um for target in targets])),
        "target_coordinate_system": "local-flow-aligned t+1 crop voxel zyx",
        "gt_exclusion_manifest": (
            str(gt_exclusion_manifest.resolve()) if gt_exclusion_manifest else None
        ),
        "gt_exclusion_contract": (
            contract_metadata(gt_exclusion_manifest) if gt_exclusion_manifest else None
        ),
    }
    return targets, summary


def input_channels(crop: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(crop, dtype=np.float32).copy())
    backward = (tensor[1:2] - tensor[0:1]) / math.sqrt(2.0)
    forward = (tensor[2:3] - tensor[1:2]) / math.sqrt(2.0)
    return torch.cat((tensor, backward, forward, backward.abs(), forward.abs()), dim=0)


class MultitaskCropDataset(Dataset):
    def __init__(
        self,
        crop_path: Path,
        samples: Sequence[Sample],
        targets: Sequence[SpatialTarget],
        indices: Sequence[int],
        heatmap_sigma_um: float,
        augment: bool,
        augmentation_profile: str = "current",
        hard_negative_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.crop_path = crop_path
        self.samples = samples
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.heatmap_sigma_um = heatmap_sigma_um
        self.augment = augment
        self.augmentation_profile = augmentation_profile
        self.hard_negative_ids = hard_negative_ids
        self._crops: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _array(self) -> np.ndarray:
        if self._crops is None:
            self._crops = np.load(self.crop_path, mmap_mode="r")
        return self._crops

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        sample = self.samples[index]
        target = self.targets[index]
        raw_image = torch.from_numpy(
            np.asarray(self._array()[index], dtype=np.float32).copy()
        )
        heatmap = torch.from_numpy(
            gaussian_heatmap(raw_image.shape[-3:], target.children_zyx_aligned, self.heatmap_sigma_um)
        ).unsqueeze(0)
        if self.augment:
            strong = should_use_strong_augmentation(
                self.augmentation_profile, sample, self.hard_negative_ids
            )
            if strong:
                image, heatmap = strong_augmentation(raw_image, heatmap)
            else:
                image = input_channels(raw_image)
                image, heatmap = current_augmentation(image, heatmap)
        else:
            image = input_channels(raw_image)
        return {
            "image": image,
            "binary": torch.tensor(float(sample.label), dtype=torch.float32),
            "role": torch.tensor(target.role_index, dtype=torch.long),
            "count": torch.tensor(target.child_count_index, dtype=torch.long),
            "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
            "heatmap": heatmap,
            "heatmap_valid": torch.tensor(float(target.heatmap_valid), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


def current_augmentation(
    image: torch.Tensor, heatmap: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-existing augmentation, kept byte-for-byte in the control path."""
    for dimension in (-1, -2, -3):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(dimension,))
            heatmap = torch.flip(heatmap, dims=(dimension,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
        heatmap = torch.rot90(heatmap, rotations, dims=(-2, -1))
    image = image * (0.95 + 0.10 * torch.rand(())) + 0.05 * torch.randn(())
    return image, heatmap


def strong_augmentation(
    raw_image: torch.Tensor, heatmap: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label-preserving spatial/intensity augmentation for three-frame crops.

    Spatial transforms are shared by the image and successor heatmap. Intensity
    transforms are applied to the three raw time frames before difference
    channels are derived, so the seven-channel algebra remains valid.
    """
    for dimension in (-1, -2, -3):
        if torch.rand(()) < 0.5:
            raw_image = torch.flip(raw_image, dims=(dimension,))
            heatmap = torch.flip(heatmap, dims=(dimension,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        raw_image = torch.rot90(raw_image, rotations, dims=(-2, -1))
        heatmap = torch.rot90(heatmap, rotations, dims=(-2, -1))

    # Output-to-input affine map. Keep z scaling fixed because the crop is thin
    # and physically anisotropic; allow only a small z translation.
    scale_y = float(torch.empty(()).uniform_(0.88, 1.12))
    scale_x = float(torch.empty(()).uniform_(0.88, 1.12))
    theta = torch.tensor(
        [
            [scale_x, 0.0, 0.0, float(torch.empty(()).uniform_(-0.12, 0.12))],
            [0.0, scale_y, 0.0, float(torch.empty(()).uniform_(-0.12, 0.12))],
            [0.0, 0.0, 1.0, float(torch.empty(()).uniform_(-0.08, 0.08))],
        ],
        dtype=raw_image.dtype,
    )
    image_batch = raw_image.unsqueeze(0)
    grid = nn.functional.affine_grid(theta.unsqueeze(0), image_batch.shape, align_corners=False)
    raw_image = nn.functional.grid_sample(
        image_batch, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    ).squeeze(0)
    heatmap = nn.functional.grid_sample(
        heatmap.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)

    mean = raw_image.mean()
    std = raw_image.std().clamp_min(1e-6)
    gamma = float(torch.empty(()).uniform_(0.80, 1.20))
    normalized = (raw_image - mean) / std
    raw_image = normalized.sign() * normalized.abs().pow(gamma) * std + mean
    raw_image = raw_image * float(torch.empty(()).uniform_(0.80, 1.20))
    raw_image = raw_image + std * float(torch.empty(()).uniform_(-0.10, 0.10))
    if torch.rand(()) < 0.50:
        raw_image = raw_image + torch.randn_like(raw_image) * std * float(
            torch.empty(()).uniform_(0.01, 0.05)
        )
    if torch.rand(()) < 0.25:
        raw_image = nn.functional.avg_pool3d(
            raw_image.unsqueeze(0), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        ).squeeze(0)
    return input_channels(raw_image), heatmap.clamp_(0.0, 1.0)


def should_use_strong_augmentation(
    profile: str, sample: Sample, hard_negative_ids: frozenset[str]
) -> bool:
    if profile == "strong_all":
        return True
    if profile == "strong_targeted":
        return sample.category in {"parent", "daughter"} or sample.sample_id in hard_negative_ids
    if profile != "current":
        raise ValueError(f"unknown augmentation profile: {profile}")
    return False


def load_hard_negative_ids(
    path: Path | None, quantile: float, mode: str = "t3_local_diff_signed_abs"
) -> tuple[frozenset[str], dict[str, Any]]:
    if path is None:
        return frozenset(), {"source": None, "normal_scored": 0, "selected": 0}
    if not 0.0 < quantile < 1.0:
        raise ValueError("hard-negative-quantile must be in (0, 1)")
    predictions: dict[str, list[tuple[int, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("split") == "train"
                and row.get("mode") == mode
                and row.get("category") == "normal"
                and int(row.get("label", 0)) == 0
            ):
                predictions.setdefault(row["sample_id"], []).append(
                    (int(row["epoch"]), float(row["prediction"]))
                )
    scores = {}
    for sample_id, values in predictions.items():
        latest = sorted(values)[-5:]
        scores[sample_id] = float(np.mean([value for _, value in latest]))
    if not scores:
        raise ValueError(f"no train normal scores found in {path}")
    threshold = float(np.quantile(list(scores.values()), quantile))
    selected = frozenset(sample_id for sample_id, score in scores.items() if score >= threshold)
    return selected, {
        "source": str(path),
        "selection": "mean prediction over each sample's latest five observed epochs",
        "quantile": quantile,
        "threshold": threshold,
        "normal_scored": len(scores),
        "selected": len(selected),
    }


def sampling_weights(
    samples: Sequence[Sample],
    indices: Sequence[int],
    profile: str,
    hard_negative_ids: frozenset[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if profile == "binary_balanced":
        labels = np.asarray([samples[index].label for index in indices], dtype=np.int64)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        weights = np.where(labels == 1, 0.5 / max(positives, 1), 0.5 / max(negatives, 1))
        return weights, {
            "profile": profile,
            "target_mass": {"parent": 0.5, "all_negative": 0.5},
            "counts": {"parent": positives, "all_negative": negatives},
        }
    if profile != "division_stratified":
        raise ValueError(f"unknown sampler profile: {profile}")
    groups = {"parent": [], "daughter": [], "hard_normal": [], "other_normal": []}
    for position, index in enumerate(indices):
        sample = samples[index]
        if sample.category == "parent":
            group = "parent"
        elif sample.category == "daughter":
            group = "daughter"
        elif sample.sample_id in hard_negative_ids:
            group = "hard_normal"
        else:
            group = "other_normal"
        groups[group].append(position)
    target_mass = {"parent": 0.50, "daughter": 0.125, "hard_normal": 0.125, "other_normal": 0.25}
    empty = [name for name, positions in groups.items() if not positions]
    if empty:
        raise ValueError(f"division_stratified sampler has empty groups: {empty}")
    weights = np.zeros(len(indices), dtype=np.float64)
    for name, positions in groups.items():
        weights[positions] = target_mass[name] / len(positions)
    return weights, {
        "profile": profile,
        "target_mass": target_mass,
        "counts": {name: len(positions) for name, positions in groups.items()},
    }


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first_kernel: tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, first_kernel, padding=tuple(k // 2 for k in first_kernel), bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class LightDivisionUNet(nn.Module):
    """Three-stage anisotropic U-Net with classification and spatial heads."""

    def __init__(self, in_channels: int = 7, base_channels: int = 16):
        super().__init__()
        b = base_channels
        self.enc0 = ResidualBlock(in_channels, b, first_kernel=(1, 3, 3))
        self.pool0 = nn.MaxPool3d((1, 2, 2))
        self.enc1 = ResidualBlock(b, 2 * b)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.bottleneck = ResidualBlock(2 * b, 4 * b)
        self.up1 = nn.ConvTranspose3d(4 * b, 2 * b, (2, 2, 2), stride=(2, 2, 2))
        self.dec1 = ResidualBlock(4 * b, 2 * b)
        self.up0 = nn.ConvTranspose3d(2 * b, b, (1, 2, 2), stride=(1, 2, 2))
        self.dec0 = ResidualBlock(2 * b, b, first_kernel=(1, 3, 3))
        self.heatmap_head = nn.Conv3d(b, 1, 1)
        self.class_pool = nn.AdaptiveAvgPool3d(1)
        self.class_shared = nn.Sequential(nn.Flatten(), nn.Dropout(0.20), nn.Linear(4 * b, 2 * b), nn.SiLU(inplace=True))
        self.binary_head = nn.Linear(2 * b, 1)
        self.role_head = nn.Linear(2 * b, 3)
        self.count_head = nn.Linear(2 * b, 3)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        e0 = self.enc0(inputs)
        e1 = self.enc1(self.pool0(e0))
        bottleneck = self.bottleneck(self.pool1(e1))
        d1 = self.dec1(torch.cat((self.up1(bottleneck), e1), dim=1))
        d0 = self.dec0(torch.cat((self.up0(d1), e0), dim=1))
        shared = self.class_shared(self.class_pool(bottleneck))
        return {
            "binary": self.binary_head(shared).squeeze(1),
            "role": self.role_head(shared),
            "count": self.count_head(shared),
            "heatmap": self.heatmap_head(d0),
        }


def load_pretrained_nnunet(model_dir: Path, fold: int, checkpoint_name: str) -> nn.Module:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=False, use_mirroring=False,
        perform_everything_on_device=False, device=torch.device("cuda"),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint_name
    )
    network = predictor.network
    predictor.network = None
    del predictor
    if not hasattr(network, "encoder") or not hasattr(network, "decoder"):
        raise TypeError(f"expected encoder-decoder nnU-Net, got {type(network)}")
    return network


def expand_first_conv3d_inputs(module: nn.Module, new_in_channels: int = 7) -> str:
    """Expand a detector stem while preserving its raw-image response.

    Temporal three-frame checkpoints already use the first three channels and
    are copied in place.  A single-frame checkpoint is mapped to channel 1,
    the centre frame in ``[t-1, t, t+1, signed diffs, absolute diffs]``.  Every
    newly introduced channel starts at zero so detector pretraining is not
    perturbed before fine-tuning.
    """
    # dynamic-network-architectures registers the same input convolution both
    # as ``block.conv`` and inside ``block.all_modules``.  named_children()
    # de-duplicates those aliases, so replacing only the first attribute leaves
    # forward() using the stale one-channel module.  Capture every parent first
    # and replace every reference to the selected Conv3d object.
    parents = list(module.named_modules(remove_duplicate=False))
    for parent_name, parent in parents:
        for child_name, child in parent._modules.items():
            if (
                isinstance(child, nn.Conv3d)
                and child.in_channels in {1, 3}
                and child.groups == 1
            ):
                replacement = nn.Conv3d(
                    new_in_channels, child.out_channels, child.kernel_size, child.stride,
                    child.padding, child.dilation, child.groups, child.bias is not None,
                    child.padding_mode,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    replacement.weight.zero_()
                    if child.in_channels == 1:
                        replacement.weight[:, 1:2].copy_(child.weight)
                    else:
                        replacement.weight[:, :3].copy_(child.weight)
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                aliases = 0
                for _, alias_parent in parents:
                    for alias_name, alias_child in list(alias_parent._modules.items()):
                        if alias_child is child:
                            alias_parent._modules[alias_name] = replacement
                            aliases += 1
                if not aliases:
                    raise RuntimeError("selected nnU-Net input Conv3d had no registered aliases")
                return f"{parent_name}.{child_name}".lstrip(".")
    raise RuntimeError("could not find a 1- or 3-channel nnU-Net input Conv3d")


def reset_network_parameters(module: nn.Module) -> int:
    reset = 0
    for child in module.modules():
        reset_parameters = getattr(child, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
            reset += 1
    if not reset:
        raise RuntimeError("nnU-Net random initialization reset no parameters")
    return reset


class NNUNetDivisionMultitask(nn.Module):
    """Use the native nnU-Net decoder spatial feature instead of pooled ROI features."""

    def __init__(self, network: nn.Module, head_seed: int):
        super().__init__()
        self.encoder = network.encoder
        self.decoder = network.decoder
        self.input_divisibility = tuple(
            math.prod(int(stride[axis]) for stride in self.encoder.strides)
            for axis in range(3)
        )
        bottleneck_channels = int(self.encoder.output_channels[-1])
        feature_channels = int(self.decoder.seg_layers[-1].in_channels)
        original_segmentation_head = self.decoder.seg_layers[-1]
        if not isinstance(original_segmentation_head, nn.Conv3d):
            raise TypeError(
                f"expected Conv3d final segmentation head, got {type(original_segmentation_head)}"
            )
        self.heatmap_head = nn.Conv3d(feature_channels, 1, 1).to(
            device=original_segmentation_head.weight.device,
            dtype=original_segmentation_head.weight.dtype,
        )
        with torch.no_grad():
            self.heatmap_head.weight.copy_(original_segmentation_head.weight[:1])
            if self.heatmap_head.bias is not None:
                if original_segmentation_head.bias is None:
                    self.heatmap_head.bias.zero_()
                else:
                    self.heatmap_head.bias.copy_(original_segmentation_head.bias[:1])
        # Make newly introduced classification heads paired between random and pretrained arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.class_shared = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.20),
                nn.Linear(bottleneck_channels, 128), nn.SiLU(inplace=True),
            )
            self.binary_head = nn.Linear(128, 1)
            self.role_head = nn.Linear(128, 3)
            self.count_head = nn.Linear(128, 3)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        original_shape = tuple(int(value) for value in inputs.shape[-3:])
        before_after = []
        for size, divisor in zip(original_shape, self.input_divisibility):
            total = (-size) % divisor
            before_after.append((total // 2, total - total // 2))
        z_pad, y_pad, x_pad = before_after
        padding = (x_pad[0], x_pad[1], y_pad[0], y_pad[1], z_pad[0], z_pad[1])
        if any(padding):
            inputs = nn.functional.pad(inputs, padding, mode="replicate")
        padded_shape = tuple(int(value) for value in inputs.shape[-3:])
        skips = self.encoder(inputs)
        decoded = skips[-1]
        for stage_index in range(len(self.decoder.stages)):
            decoded = self.decoder.transpconvs[stage_index](decoded)
            decoded = torch.cat((decoded, skips[-(stage_index + 2)]), dim=1)
            decoded = self.decoder.stages[stage_index](decoded)
        heatmap = self.heatmap_head(decoded)
        if heatmap.shape[-3:] != padded_shape:
            heatmap = nn.functional.interpolate(
                heatmap, size=padded_shape, mode="trilinear", align_corners=False
            )
        heatmap = heatmap[
            :, :,
            z_pad[0]:z_pad[0] + original_shape[0],
            y_pad[0]:y_pad[0] + original_shape[1],
            x_pad[0]:x_pad[0] + original_shape[2],
        ]
        shared = self.class_shared(skips[-1])
        return {
            "binary": self.binary_head(shared).squeeze(1),
            "role": self.role_head(shared),
            "count": self.count_head(shared),
            "heatmap": heatmap,
        }

    def pretrained_parameters(self) -> list[nn.Parameter]:
        # The nnU-Net decoder retains a reference to its encoder, so simply
        # concatenating encoder.parameters() and decoder.parameters() registers
        # the encoder twice in AdamW.  Preserve first-seen order while removing
        # identity duplicates.
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for group in (
            self.encoder.parameters(), self.decoder.parameters(), self.heatmap_head.parameters()
        ):
            for parameter in group:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def new_head_parameters(self) -> list[nn.Parameter]:
        return [
            *self.class_shared.parameters(), *self.binary_head.parameters(),
            *self.role_head.parameters(), *self.count_head.parameters(),
        ]


def build_model(arm: str, args: argparse.Namespace) -> tuple[nn.Module, int, dict[str, Any]]:
    if arm in {"binary", "multitask"}:
        model = LightDivisionUNet(7, args.base_channels)
        count = sum(parameter.numel() for parameter in model.parameters())
        if count >= 3_000_000:
            raise ValueError(f"model is not lightweight: {count:,} parameters")
        return model, count, {"architecture": "light_3d_unet", "initialization": "random"}
    if args.model_dir is None:
        raise ValueError(f"--model-dir is required for arm {arm}")
    network = load_pretrained_nnunet(args.model_dir, args.fold, args.checkpoint_name)
    reset_count = 0
    if arm == "nnunet_random":
        seed_everything(args.seed + 700)
        reset_count = reset_network_parameters(network)
    # Some dynamic-network-architectures containers rebuild child modules in
    # reset_parameters().  Expand the stem only after the random reset so the
    # seven-channel temporal input cannot silently revert to one channel.
    input_layer = "encoder.stem." + expand_first_conv3d_inputs(network.encoder.stem, 7)
    model = NNUNetDivisionMultitask(network, head_seed=args.seed + 701)
    count = sum(parameter.numel() for parameter in model.parameters())
    return model, count, {
        "architecture": type(network).__name__,
        "initialization": (
            "random" if arm == "nnunet_random" else "fold_specific_detector_pretrained"
        ),
        "expanded_input_layer": input_layer,
        "added_input_channels_initialization": "random" if arm == "nnunet_random" else "zeros",
        "reset_module_count": reset_count,
        "model_dir": str(args.model_dir),
        "checkpoint_name": args.checkpoint_name,
    }


def amp_context(name: str):
    if name == "none":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if name == "bf16" else torch.float16)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], multitask: bool, args: argparse.Namespace):
    binary_per_sample = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary"].float(), batch["binary"], reduction="none"
    )
    total = binary_per_sample.mean()
    components = {"binary": binary_per_sample.mean()}
    if multitask:
        role = nn.functional.cross_entropy(outputs["role"].float(), batch["role"])
        count_per_sample = nn.functional.cross_entropy(outputs["count"].float(), batch["count"], reduction="none")
        count = masked_mean(count_per_sample, batch["count_valid"])
        probability = torch.sigmoid(outputs["heatmap"].float())
        heatmap_per_voxel = torch.square(probability - batch["heatmap"]) * (1.0 + 10.0 * batch["heatmap"])
        heatmap_per_sample = heatmap_per_voxel.flatten(1).mean(1)
        heatmap = masked_mean(heatmap_per_sample, batch["heatmap_valid"])
        total = total + args.role_loss_weight * role + args.count_loss_weight * count + args.heatmap_loss_weight * heatmap
        components.update(role=role, count=count, heatmap=heatmap)
    return total, binary_per_sample, components


def role_macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for category in range(3):
        tp = np.count_nonzero((targets == category) & (predictions == category))
        fp = np.count_nonzero((targets != category) & (predictions == category))
        fn = np.count_nonzero((targets == category) & (predictions != category))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


def evaluate(model: nn.Module, loader: DataLoader, samples: Sequence[Sample], device: torch.device, amp: str):
    model.eval()
    binary_scores: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    role_predictions: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    count_predictions: list[np.ndarray] = []
    count_valids: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            cpu_indices = batch["index"].numpy()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "index"}
            with amp_context(amp):
                output = model(batch["image"])
            binary_scores.append(torch.sigmoid(output["binary"].float()).cpu().numpy())
            roles.append(batch["role"].cpu().numpy())
            role_predictions.append(output["role"].argmax(1).cpu().numpy())
            counts.append(batch["count"].cpu().numpy())
            count_predictions.append(output["count"].argmax(1).cpu().numpy())
            count_valids.append(batch["count_valid"].bool().cpu().numpy())
            indices.append(cpu_indices)
    ordered_indices = np.concatenate(indices)
    scores = np.concatenate(binary_scores)
    metrics = metrics_with_slices(samples, ordered_indices, scores)
    role_values = np.concatenate(roles)
    role_pred = np.concatenate(role_predictions)
    count_values = np.concatenate(counts)
    count_pred = np.concatenate(count_predictions)
    count_mask = np.concatenate(count_valids)
    metrics["auxiliary"] = {
        "role_accuracy": float(np.mean(role_values == role_pred)),
        "role_macro_f1": role_macro_f1(role_values, role_pred),
        "child_count_accuracy": float(np.mean(count_values[count_mask] == count_pred[count_mask])) if np.any(count_mask) else None,
        "child_count_valid_n": int(np.count_nonzero(count_mask)),
    }
    return scores, ordered_indices, metrics


def train_arm(
    arm: str,
    crop_path: Path,
    samples: Sequence[Sample],
    spatial_targets: Sequence[SpatialTarget],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    hard_negative_ids: frozenset[str],
    hard_negative_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    arm_dir = args.output_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_seed = args.seed
    seed_everything(arm_seed)
    train_dataset = MultitaskCropDataset(
        crop_path, samples, spatial_targets, train_indices, args.heatmap_sigma_um, True,
        args.augmentation_profile, hard_negative_ids,
    )
    val_dataset = MultitaskCropDataset(
        crop_path, samples, spatial_targets, val_indices, args.heatmap_sigma_um, False,
        "current", frozenset(),
    )
    weights, sampler_metadata = sampling_weights(
        samples, train_indices, args.sampler_profile, hard_negative_ids
    )
    draws = max(args.samples_per_epoch, len(train_indices))
    is_nnunet = arm.startswith("nnunet_")
    actual_batch_size = args.nnunet_batch_size if is_nnunet else args.batch_size
    if args.batch_size % actual_batch_size:
        raise ValueError(
            f"effective batch {args.batch_size} must be divisible by actual batch {actual_batch_size}"
        )
    accumulation_steps = args.batch_size // actual_batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=actual_batch_size,
        sampler=WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), draws, True, generator=torch.Generator().manual_seed(arm_seed)),
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(arm_seed + 1000),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=actual_batch_size * 2, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed,
    )
    device = torch.device("cuda")
    model, parameter_count, model_metadata = build_model(arm, args)
    model = model.to(device)
    if arm == "nnunet_pretrained":
        optimizer_groups = [
            {"params": model.new_head_parameters(), "lr": args.learning_rate},
            {"params": model.pretrained_parameters(), "lr": args.pretrained_learning_rate},
        ]
    else:
        optimizer_groups = [{"params": model.parameters(), "lr": args.learning_rate}]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)),
    )
    best_ap = -math.inf
    best_epoch = -1
    stale = 0
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    multitask = arm != "binary"
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch.pop("index")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with amp_context(args.amp):
                output = model(batch["image"])
                loss, _, components = losses(output, batch, multitask, args)
            (loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        _, _, metrics = evaluate(model, val_loader, samples, device, args.amp)
        ap = float(metrics["parent_vs_all"]["auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_auprc": ap,
            "val_parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"],
            "val_parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"],
            "role_macro_f1": metrics["auxiliary"]["role_macro_f1"],
            "child_count_accuracy": metrics["auxiliary"]["child_count_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "pretrained_learning_rate": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
        }
        curve.append(row)
        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, arm_dir / "checkpoint_best.pt")
        else:
            stale += 1
        print(
            f"[{arm}] epoch={epoch:03d} loss={row['train_loss']:.5f} val_ap={ap:.5f} "
            f"daughter_ap={row['val_parent_vs_daughter_auprc']:.5f} best={best_ap:.5f}@{best_epoch}",
            flush=True,
        )
        if epoch + 1 >= min(20, args.epochs) and stale >= args.patience:
            break
    checkpoint = torch.load(arm_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    scores, ordered_indices, final_metrics = evaluate(model, val_loader, samples, device, args.amp)
    score_rows = []
    for index, score in zip(ordered_indices, scores):
        sample = samples[int(index)]
        score_rows.append({
            "fold": args.fold, "arm": arm, "sample_id": sample.sample_id, "stem": sample.stem,
            "frame": sample.frame, "category": sample.category, "label": sample.label,
            "prediction": float(score),
        })
    write_csv(arm_dir / "learning_curve.csv", curve)
    write_csv(arm_dir / "oof_sample_scores.csv", score_rows)
    summary = {
        "arm": arm,
        "multitask": multitask,
        "parameter_count": parameter_count,
        "model": model_metadata,
        "actual_batch_size": actual_batch_size,
        "effective_batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "augmentation_profile": args.augmentation_profile,
        "sampler": sampler_metadata,
        "hard_negatives": hard_negative_metadata,
        "best_epoch": best_epoch,
        "epochs_completed": len(curve),
        "runtime_seconds": time.perf_counter() - started,
        "metrics": final_metrics,
    }
    atomic_json(arm_dir / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    nnunet_arms = [arm for arm in args.arms if arm.startswith("nnunet_")]
    if nnunet_arms and args.model_dir is None:
        raise ValueError(f"--model-dir is required for nnU-Net arms: {nnunet_arms}")
    if args.model_dir is not None and not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 96)
        args.num_workers = min(args.num_workers, 2)
        args.patience = args.epochs
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for directory in ("arms", "cache", "config", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sample_path = args.control_dir / "cache/samples.csv"
    raw_crop_path = args.control_dir / "cache/crops.float16.npy"
    samples = read_samples(sample_path)
    hard_negative_ids, hard_negative_metadata = load_hard_negative_ids(
        args.hard_negative_score_csv, args.hard_negative_quantile
    )
    if args.smoke:
        selected = []
        for split in ("train", "val"):
            partition = [index for index, sample in enumerate(samples) if sample.split == split]
            selected.extend([index for index in partition if samples[index].label][:6])
            if split == "train" and args.sampler_profile == "division_stratified":
                selected.extend([
                    index for index in partition if samples[index].category == "daughter"
                ][:6])
                selected.extend([
                    index for index in partition
                    if samples[index].category == "normal"
                    and samples[index].sample_id in hard_negative_ids
                ][:6])
                selected.extend([
                    index for index in partition
                    if samples[index].category == "normal"
                    and samples[index].sample_id not in hard_negative_ids
                ][:18])
            else:
                selected.extend([index for index in partition if not samples[index].label][:24])
        samples = [samples[index] for index in selected]
        source = np.load(raw_crop_path, mmap_mode="r")
        smoke_raw_path = args.output_dir / "cache/smoke_raw.float16.npy"
        smoke_raw = np.lib.format.open_memmap(smoke_raw_path, mode="w+", dtype=np.float16, shape=(len(selected), *source.shape[1:]))
        for output_index, source_index in enumerate(selected):
            smoke_raw[output_index] = source[source_index]
        smoke_raw.flush()
        del smoke_raw, source
        raw_crop_path = smoke_raw_path
    allowed_cases = set()
    for sample in samples:
        allowed_cases.update(
            case_id for case_id in (neighbor_case_id(sample, -1), sample.case_id, neighbor_case_id(sample, 1)) if case_id
        )
    peaks = load_peaks(args.detector_peaks, allowed_cases)
    flow = SparseLocalFlow(peaks, args)
    _, local_crop_path, flow_rows = build_aligned_caches(raw_crop_path, samples, flow, args.output_dir)
    write_csv(args.output_dir / "cache/flow_alignment.csv", flow_rows)
    write_csv(args.output_dir / "cache/samples.csv", [asdict(sample) for sample in samples])
    crop_shape = np.load(local_crop_path, mmap_mode="r").shape[-3:]
    spatial_targets, target_summary = build_spatial_targets(
        samples,
        flow_rows,
        args.train_dir,
        crop_shape,
        max_match_distance_um=args.max_node_match_distance_um,
    )
    atomic_json(args.output_dir / "config/spatial_target_summary.json", target_summary)
    target_rows = []
    for sample, target in zip(samples, spatial_targets):
        target_rows.append({
            "sample_id": sample.sample_id,
            "role_index": target.role_index,
            "child_count": len(target.children_zyx_aligned),
            "heatmap_valid": int(target.heatmap_valid),
            "count_valid": int(target.child_count_valid),
            "node_distance_um": target.node_distance_um,
            "children_zyx_aligned_json": json.dumps(target.children_zyx_aligned),
        })
    write_csv(args.output_dir / "config/spatial_targets.csv", target_rows)
    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    val_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    train_sample_ids = {samples[index].sample_id for index in train_indices}
    hard_negative_ids = frozenset(hard_negative_ids & train_sample_ids)
    hard_negative_metadata = {
        **hard_negative_metadata,
        "selected_present_in_current_train_split": len(hard_negative_ids),
    }
    if args.sampler_profile == "division_stratified" and not hard_negative_ids:
        raise ValueError("division_stratified sampling requires train hard negatives")
    summaries = [
        train_arm(
            arm, local_crop_path, samples, spatial_targets, train_indices, val_indices,
            hard_negative_ids, hard_negative_metadata, args,
        )
        for arm in args.arms
    ]
    summary = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold,
        "seed": args.seed,
        "control_dir": str(args.control_dir),
        "train_dir": str(args.train_dir),
        "input_mode": "t3_local_diff_signed_abs",
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "checkpoint_name": args.checkpoint_name if args.model_dir else None,
        "arms_requested": list(args.arms),
        "augmentation_profile": args.augmentation_profile,
        "sampler_profile": args.sampler_profile,
        "hard_negatives": hard_negative_metadata,
        "sample_count": len(samples),
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(val_indices),
        "spatial_targets": target_summary,
        "arms": {result["arm"]: result for result in summaries},
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "DONE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
