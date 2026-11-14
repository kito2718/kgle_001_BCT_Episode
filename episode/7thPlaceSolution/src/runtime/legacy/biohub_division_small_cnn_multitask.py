#!/usr/bin/env python3
"""Paired Small CNN successor-multitask screen for Biohub Division.

The image encoder and base binary classifier are identical in every arm.  The
only differences are successor-count loss, relative-successor-offset loss, and
an optional zero-initialized correction that feeds auxiliary predictions back
to the binary logit at inference.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import shift as image_shift
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from biohub_division_crop_classifier import (
    SPACING_ZYX,
    Sample,
    SmallDivisionCNN,
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
from biohub_division_light_unet_multitask import (
    SpatialTarget,
    amp_context,
    build_spatial_targets,
    input_channels,
    load_hard_negative_ids,
    masked_mean,
    sampling_weights,
    seed_everything,
    should_use_strong_augmentation,
)


ARMS = ("binary", "count_aux", "count_offset_aux", "feedback")
INPUT_NORMALIZATIONS = ("current", "train_channel_zscore")
AUGMENTATION_PROFILES = ("current", "strong_targeted", "algebra_repaired")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--detector-peaks", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument(
        "--input-normalization",
        choices=INPUT_NORMALIZATIONS,
        default="current",
        help=(
            "Optionally standardize the final seven image channels with "
            "statistics estimated only from the training partition."
        ),
    )
    parser.add_argument("--max-node-match-distance-um", type=float, default=7.0)
    parser.add_argument("--offset-scale-um", type=float, default=10.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--augmentation-profile",
        choices=AUGMENTATION_PROFILES,
        default="current",
    )
    parser.add_argument("--hard-negative-score-csv", type=Path)
    parser.add_argument(
        "--extra-aligned-crops",
        type=Path,
        help="Optional aligned T3 cache; its seven derived channels are appended to raw seven channels.",
    )
    parser.add_argument("--hard-negative-quantile", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--registration-jitter-probability", type=float, default=0.0)
    parser.add_argument("--registration-jitter-magnitude-um", type=float, default=0.0)
    parser.add_argument(
        "--paired-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Bernoulli-KL weight from a detached exact-view binary prediction "
            "to a paired jitter-view prediction. Supervised losses remain exact-only."
        ),
    )
    parser.add_argument(
        "--registration-jitter-family",
        choices=("common", "independent", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--registration-jitter-magnitude-distribution",
        choices=("fixed", "uniform_0_max"),
        default="fixed",
    )
    parser.add_argument(
        "--registration-jitter-application",
        choices=("aligned_two_pass", "raw_one_pass"),
        default="aligned_two_pass",
    )
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--sigma-um", type=float, default=20.0)
    parser.add_argument("--local-radius-um", type=float, default=50.0)
    parser.add_argument("--exclude-radius-um", type=float, default=3.0)
    parser.add_argument("--residual-gate-um", type=float, default=7.0)
    parser.add_argument("--min-local-support", type=int, default=3)
    parser.add_argument("--max-local-support", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="resume each incomplete arm from its rolling checkpoint_latest.pt",
    )
    return parser.parse_args()


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Replace one rolling checkpoint without exposing a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def training_resume_contract(
    arm: str,
    args: argparse.Namespace,
    samples: Sequence[Sample],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": arm,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "samples_per_epoch": int(args.samples_per_epoch),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "count_loss_weight": float(args.count_loss_weight),
        "offset_loss_weight": float(args.offset_loss_weight),
        "augmentation_profile": str(args.augmentation_profile),
        "registration_jitter_probability": float(args.registration_jitter_probability),
        "registration_jitter_magnitude_um": float(args.registration_jitter_magnitude_um),
        "registration_jitter_family": str(args.registration_jitter_family),
        "registration_jitter_magnitude_distribution": str(
            args.registration_jitter_magnitude_distribution
        ),
        "registration_jitter_application": str(args.registration_jitter_application),
        "paired_consistency_weight": float(args.paired_consistency_weight),
        "input_normalization": str(args.input_normalization),
        "extra_aligned_crops": str(args.extra_aligned_crops) if args.extra_aligned_crops else None,
        "sample_count": len(samples),
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
    }


def validate_resume_contract(
    saved: dict[str, Any], current: dict[str, Any]
) -> None:
    if saved != current:
        differences = {
            key: {"saved": saved.get(key), "current": current.get(key)}
            for key in sorted(set(saved) | set(current))
            if saved.get(key) != current.get(key)
        }
        raise RuntimeError(
            "refusing incompatible checkpoint resume: "
            + json.dumps(differences, sort_keys=True)
        )


def transform_points(
    points_zyx: torch.Tensor,
    shape_zyx: Sequence[int],
    flips_zyx: Sequence[bool],
    rotations_xy: int,
) -> torch.Tensor:
    """Apply the same flips and torch.rot90 transform used for image crops."""
    points = points_zyx.clone().to(dtype=torch.float32)
    shape = tuple(int(value) for value in shape_zyx)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected [N,3] points, got {tuple(points.shape)}")
    if len(flips_zyx) != 3:
        raise ValueError(f"expected 3 flip flags, got {flips_zyx}")
    for axis, enabled in enumerate(flips_zyx):
        if enabled:
            points[:, axis] = (shape[axis] - 1) - points[:, axis]
    rotations = int(rotations_xy) % 4
    for _ in range(rotations):
        y = points[:, 1].clone()
        x = points[:, 2].clone()
        points[:, 1] = (shape[2] - 1) - x
        points[:, 2] = y
        shape = (shape[0], shape[2], shape[1])
    return points


def ordered_offset_targets(
    children_zyx: Sequence[Sequence[float]],
    crop_shape_zyx: Sequence[int],
    flips_zyx: Sequence[bool],
    rotations_xy: int,
    target_valid: bool,
    offset_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two deterministic successor offsets and their validity mask."""
    if offset_scale_um <= 0:
        raise ValueError(f"offset_scale_um must be positive, got {offset_scale_um}")
    shape = tuple(int(value) for value in crop_shape_zyx)
    centre = torch.tensor([[float(value // 2) for value in shape]], dtype=torch.float32)
    children = torch.as_tensor(children_zyx, dtype=torch.float32).reshape(-1, 3)
    combined = torch.cat((centre, children), dim=0)
    transformed = transform_points(combined, shape, flips_zyx, rotations_xy)
    transformed_centre, transformed_children = transformed[:1], transformed[1:]
    if len(transformed_children):
        order = sorted(
            range(len(transformed_children)),
            key=lambda index: (
                float(transformed_children[index, 2]),
                float(transformed_children[index, 1]),
                float(transformed_children[index, 0]),
            ),
        )
        transformed_children = transformed_children[order]
    offsets = torch.zeros((2, 3), dtype=torch.float32)
    valid = torch.zeros(2, dtype=torch.float32)
    if target_valid:
        used = min(len(transformed_children), 2)
        if used:
            spacing = torch.as_tensor(SPACING_ZYX, dtype=torch.float32)
            offsets[:used] = (
                (transformed_children[:used] - transformed_centre) * spacing
            ) / float(offset_scale_um)
            valid[:used] = 1.0
    return offsets, valid


def affine_output_points(
    points_zyx: torch.Tensor,
    shape_zyx: Sequence[int],
    theta: torch.Tensor,
) -> torch.Tensor:
    """Map input voxel points to the output grid produced by affine_grid.

    ``theta`` follows PyTorch's output-to-input normalized-coordinate contract,
    so point targets use its inverse. This keeps successor offsets paired with
    the actual grid_sample image transform.
    """
    points = points_zyx.clone().to(dtype=torch.float32)
    shape = torch.as_tensor(shape_zyx, dtype=torch.float32, device=points.device)
    matrix = theta[:, :3].to(device=points.device, dtype=torch.float32)
    translation = theta[:, 3].to(device=points.device, dtype=torch.float32)
    # affine_grid coordinates are xyz, whereas target points are zyx.
    xyz = points[:, (2, 1, 0)]
    size_xyz = shape[[2, 1, 0]]
    normalized_input = 2.0 * (xyz + 0.5) / size_xyz - 1.0
    normalized_output = torch.linalg.solve(
        matrix, (normalized_input - translation).transpose(0, 1)
    ).transpose(0, 1)
    output_xyz = (normalized_output + 1.0) * size_xyz / 2.0 - 0.5
    return output_xyz[:, (2, 1, 0)]


def offsets_from_transformed_points(
    centre_zyx: torch.Tensor,
    children_zyx: torch.Tensor,
    crop_shape_zyx: Sequence[int],
    target_valid: bool,
    offset_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.zeros((2, 3), dtype=torch.float32)
    valid = torch.zeros(2, dtype=torch.float32)
    if not target_valid or not len(children_zyx):
        return offsets, valid
    shape = torch.as_tensor(crop_shape_zyx, dtype=torch.float32)
    inside = ((children_zyx >= 0.0) & (children_zyx < shape[None])).all(dim=1)
    children = children_zyx[inside]
    if not len(children):
        return offsets, valid
    order = sorted(
        range(len(children)),
        key=lambda index: (
            float(children[index, 2]),
            float(children[index, 1]),
            float(children[index, 0]),
        ),
    )
    children = children[order]
    used = min(len(children), 2)
    spacing = torch.as_tensor(SPACING_ZYX, dtype=torch.float32)
    offsets[:used] = (
        (children[:used] - centre_zyx[None]) * spacing
    ) / float(offset_scale_um)
    valid[:used] = 1.0
    return offsets, valid


def strong_successor_augmentation(
    raw_image: torch.Tensor,
    children_zyx: Sequence[Sequence[float]],
    target_valid: bool,
    offset_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strong paired transform for raw frames and continuous successor points."""
    shape = tuple(int(value) for value in raw_image.shape[-3:])
    if shape[1] != shape[2]:
        raise ValueError("strong XY rot90 currently requires a square crop")
    centre = torch.tensor(
        [[float(value // 2) for value in shape]], dtype=torch.float32
    )
    children = torch.as_tensor(children_zyx, dtype=torch.float32).reshape(-1, 3)
    points = torch.cat((centre, children), dim=0)
    flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
    for axis, enabled in enumerate(flips):
        if enabled:
            raw_image = torch.flip(raw_image, dims=(axis - 3,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        raw_image = torch.rot90(raw_image, rotations, dims=(-2, -1))
    points = transform_points(points, shape, flips, rotations)

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
    grid = nn.functional.affine_grid(
        theta.unsqueeze(0), raw_image.unsqueeze(0).shape, align_corners=False
    )
    raw_image = nn.functional.grid_sample(
        raw_image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).squeeze(0)
    points = affine_output_points(points, shape, theta)

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
            raw_image.unsqueeze(0),
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        ).squeeze(0)
    offsets, valid = offsets_from_transformed_points(
        points[0], points[1:], shape, target_valid, offset_scale_um
    )
    return input_channels(raw_image), offsets, valid


def estimate_input_channel_statistics(
    crop_path: Path,
    train_indices: Sequence[int],
) -> dict[str, Any]:
    """Estimate final seven-channel moments without reading validation rows."""
    crops = np.load(crop_path, mmap_mode="r")
    sums = np.zeros(7, dtype=np.float64)
    squared_sums = np.zeros(7, dtype=np.float64)
    voxel_count = 0
    for index in train_indices:
        channels = input_channels(
            np.asarray(crops[int(index)], dtype=np.float32)
        ).numpy()
        axes = tuple(range(1, channels.ndim))
        sums += channels.sum(axis=axes, dtype=np.float64)
        squared_sums += np.square(channels, dtype=np.float64).sum(
            axis=axes, dtype=np.float64
        )
        voxel_count += int(np.prod(channels.shape[1:]))
    if voxel_count <= 0:
        raise ValueError("cannot estimate channel statistics from an empty train partition")
    mean = sums / voxel_count
    variance = np.maximum(squared_sums / voxel_count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("non-finite seven-channel normalization statistics")
    return {
        "mode": "train_channel_zscore",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_sample_count": len(train_indices),
        "voxel_count_per_channel": voxel_count,
        "source": "unaugmented local-flow-aligned final seven-channel training crops only",
    }


def normalize_input_channels(
    image: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        if mean is not None or std is not None:
            raise ValueError("channel mean and std must either both be set or both be absent")
        return image
    normalized = (image[: len(mean)] - mean[:, None, None, None]) / std[:, None, None, None]
    if image.shape[0] == len(mean):
        return normalized
    if image.shape[0] < len(mean):
        raise ValueError(f"image has fewer channels than normalization statistics: {image.shape[0]} < {len(mean)}")
    return torch.cat((normalized, image[len(mean):]), dim=0)


def algebra_repaired_intensity_channels(
    raw: torch.Tensor, scale: torch.Tensor | float, shift: torch.Tensor | float
) -> torch.Tensor:
    """Apply one affine intensity transform to raw T3 before deriving differences."""
    return input_channels(raw * scale + shift)


def apply_registration_jitter_to_aligned_raw(
    raw: np.ndarray,
    children_zyx_aligned: Sequence[Sequence[float]],
    previous_error_um: np.ndarray,
    next_error_um: np.ndarray,
) -> tuple[np.ndarray, list[list[float]]]:
    """Perturb aligned raw neighbors before deriving seven channels.

    The central frame and candidate centre stay fixed. The t+1 successor target
    follows the same extra warp as the t+1 image so the auxiliary offset target
    remains in the actual augmented coordinate system.
    """
    if raw.shape[0] != 3:
        raise ValueError(f"expected aligned T3 raw crop, got {raw.shape}")
    previous_error_um = np.asarray(previous_error_um, dtype=np.float64)
    next_error_um = np.asarray(next_error_um, dtype=np.float64)
    if previous_error_um.shape != (3,) or next_error_um.shape != (3,):
        raise ValueError("registration errors must be physical ZYX triplets")
    output = np.asarray(raw, dtype=np.float32).copy()
    output[0] = image_shift(
        output[0], previous_error_um / SPACING_ZYX,
        order=1, mode="constant", cval=0.0, prefilter=False,
    )
    output[2] = image_shift(
        output[2], -next_error_um / SPACING_ZYX,
        order=1, mode="constant", cval=0.0, prefilter=False,
    )
    children = np.asarray(children_zyx_aligned, dtype=np.float64).reshape(-1, 3)
    if len(children):
        children = children - next_error_um[None] / SPACING_ZYX[None]
    return output, children.tolist()


def sample_registration_jitter_pair(
    maximum_um: float,
    family: str,
    magnitude_distribution: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample aligned-output-space residual shifts for the two neighbor frames."""
    if maximum_um <= 0:
        raise ValueError("registration jitter maximum must be positive")
    if family not in {"common", "independent"}:
        raise ValueError(f"unknown registration jitter family: {family}")
    if magnitude_distribution not in {"fixed", "uniform_0_max"}:
        raise ValueError(
            f"unknown registration jitter magnitude distribution: {magnitude_distribution}"
        )

    def draw() -> torch.Tensor:
        direction = torch.randn(3, dtype=torch.float64)
        direction /= direction.norm().clamp_min(1e-12)
        magnitude = torch.tensor(maximum_um, dtype=torch.float64)
        if magnitude_distribution == "uniform_0_max":
            magnitude *= torch.rand((), dtype=torch.float64)
        return direction * magnitude

    previous = draw()
    following = previous.clone() if family == "common" else draw()
    return previous, following


def apply_registration_jitter_to_unaligned_raw_one_pass(
    raw: np.ndarray,
    children_zyx_base_aligned: Sequence[Sequence[float]],
    previous_base_warp_um: np.ndarray,
    next_base_warp_um: np.ndarray,
    previous_error_um: np.ndarray,
    next_error_um: np.ndarray,
) -> tuple[np.ndarray, list[list[float]]]:
    """Apply base warp plus random residual once, in aligned output coordinates."""
    if raw.shape[0] != 3:
        raise ValueError(f"expected unaligned T3 raw crop, got {raw.shape}")
    vectors = [
        np.asarray(value, dtype=np.float64)
        for value in (
            previous_base_warp_um,
            next_base_warp_um,
            previous_error_um,
            next_error_um,
        )
    ]
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("base warps and registration errors must be physical ZYX triplets")
    previous_base, next_base, previous_error, next_error = vectors
    output = np.empty_like(raw, dtype=np.float32)
    output[0] = image_shift(
        np.asarray(raw[0], dtype=np.float32),
        (previous_base + previous_error) / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    output[1] = np.asarray(raw[1], dtype=np.float32)
    output[2] = image_shift(
        np.asarray(raw[2], dtype=np.float32),
        (next_base + next_error) / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    # Match J0/evaluation: only the final one-pass aligned raw is quantized.
    output = output.astype(np.float16).astype(np.float32)
    children = np.asarray(children_zyx_base_aligned, dtype=np.float64).reshape(-1, 3)
    if len(children):
        children = children + next_error[None] / SPACING_ZYX[None]
    return output, children.tolist()


def paired_input_channels_with_transform(
    exact_raw: np.ndarray,
    jitter_raw: np.ndarray,
    flips_zyx: Sequence[bool],
    rotations_xy: int,
    intensity_scale: torch.Tensor | float,
    intensity_shift: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one shared spatial/intensity transform to an exact/jitter pair."""
    if exact_raw.shape != jitter_raw.shape:
        raise ValueError(f"paired raw shape mismatch: {exact_raw.shape} != {jitter_raw.shape}")
    if len(flips_zyx) != 3:
        raise ValueError(f"expected three flip flags, got {flips_zyx}")
    exact = input_channels(exact_raw)
    jitter = input_channels(jitter_raw)
    for axis, enabled in enumerate(flips_zyx):
        if enabled:
            exact = torch.flip(exact, dims=(axis - 3,))
            jitter = torch.flip(jitter, dims=(axis - 3,))
    rotations = int(rotations_xy) % 4
    if rotations:
        exact = torch.rot90(exact, rotations, dims=(-2, -1))
        jitter = torch.rot90(jitter, rotations, dims=(-2, -1))
    exact = exact * intensity_scale + intensity_shift
    jitter = jitter * intensity_scale + intensity_shift
    return exact, jitter


def bernoulli_kl_exact_teacher(
    exact_logits: torch.Tensor, jitter_logits: torch.Tensor
) -> torch.Tensor:
    """KL(exact-detached || jitter), with no gradient through the exact teacher."""
    if exact_logits.shape != jitter_logits.shape:
        raise ValueError(
            f"paired binary logit shape mismatch: {exact_logits.shape} != {jitter_logits.shape}"
        )
    teacher = torch.sigmoid(exact_logits.float()).detach().clamp(1e-6, 1.0 - 1e-6)
    log_teacher = torch.log(teacher)
    log_teacher_negative = torch.log1p(-teacher)
    log_student = nn.functional.logsigmoid(jitter_logits.float())
    log_student_negative = nn.functional.logsigmoid(-jitter_logits.float())
    return torch.mean(
        teacher * (log_teacher - log_student)
        + (1.0 - teacher) * (log_teacher_negative - log_student_negative)
    )


class SuccessorCropDataset(Dataset):
    def __init__(
        self,
        crop_path: Path,
        samples: Sequence[Sample],
        targets: Sequence[SpatialTarget],
        indices: Sequence[int],
        augment: bool,
        offset_scale_um: float,
        augmentation_profile: str = "current",
        hard_negative_ids: frozenset[str] = frozenset(),
        channel_mean: Sequence[float] | None = None,
        channel_std: Sequence[float] | None = None,
        extra_crop_path: Path | None = None,
        registration_jitter_probability: float = 0.0,
        registration_jitter_magnitude_um: float = 0.0,
        registration_jitter_family: str = "mixed",
        registration_jitter_magnitude_distribution: str = "fixed",
        registration_jitter_application: str = "aligned_two_pass",
        registration_jitter_raw_crop_path: Path | None = None,
        registration_jitter_flow_rows: Sequence[dict[str, Any]] | None = None,
        paired_consistency: bool = False,
    ) -> None:
        self.crop_path = crop_path
        self.samples = samples
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.offset_scale_um = offset_scale_um
        self.augmentation_profile = augmentation_profile
        self.hard_negative_ids = hard_negative_ids
        self.channel_mean = (
            torch.as_tensor(channel_mean, dtype=torch.float32)
            if channel_mean is not None else None
        )
        self.channel_std = (
            torch.as_tensor(channel_std, dtype=torch.float32)
            if channel_std is not None else None
        )
        self.extra_crop_path = extra_crop_path
        self.registration_jitter_probability = registration_jitter_probability
        self.registration_jitter_magnitude_um = registration_jitter_magnitude_um
        self.registration_jitter_family = registration_jitter_family
        self.registration_jitter_magnitude_distribution = (
            registration_jitter_magnitude_distribution
        )
        self.registration_jitter_application = registration_jitter_application
        self.registration_jitter_raw_crop_path = registration_jitter_raw_crop_path
        self.registration_jitter_flow_rows = registration_jitter_flow_rows
        self.paired_consistency = paired_consistency
        self._registration_jitter_raw_crops: np.ndarray | None = None
        self._extra_crops: np.ndarray | None = None
        if (self.channel_mean is None) != (self.channel_std is None):
            raise ValueError("channel mean/std must be supplied together")
        if self.channel_mean is not None:
            if self.channel_mean.shape != (7,) or self.channel_std.shape != (7,):
                raise ValueError("expected seven channel means and standard deviations")
            if not torch.isfinite(self.channel_mean).all() or not torch.isfinite(self.channel_std).all():
                raise ValueError("channel normalization statistics must be finite")
            if not torch.all(self.channel_std > 0):
                raise ValueError("channel standard deviations must be positive")
        if not 0.0 <= self.registration_jitter_probability <= 1.0:
            raise ValueError("registration jitter probability must be in [0, 1]")
        if self.registration_jitter_magnitude_um < 0:
            raise ValueError("registration jitter magnitude must be nonnegative")
        if self.registration_jitter_probability > 0 and self.registration_jitter_magnitude_um <= 0:
            raise ValueError("positive jitter probability requires positive magnitude")
        if self.registration_jitter_family not in {"common", "independent", "mixed"}:
            raise ValueError(f"unknown registration jitter family: {self.registration_jitter_family}")
        if self.registration_jitter_magnitude_distribution not in {"fixed", "uniform_0_max"}:
            raise ValueError(
                "unknown registration jitter magnitude distribution: "
                f"{self.registration_jitter_magnitude_distribution}"
            )
        if self.registration_jitter_application not in {"aligned_two_pass", "raw_one_pass"}:
            raise ValueError(
                f"unknown registration jitter application: {self.registration_jitter_application}"
            )
        if self.registration_jitter_application == "raw_one_pass":
            if self.registration_jitter_raw_crop_path is None or self.registration_jitter_flow_rows is None:
                raise ValueError("raw_one_pass jitter requires raw crops and flow rows")
            if len(self.registration_jitter_flow_rows) != len(self.samples):
                raise ValueError("raw_one_pass flow rows must match sample count")
        if self.paired_consistency:
            if not self.augment:
                raise ValueError("paired consistency is training-only")
            if self.registration_jitter_application != "raw_one_pass":
                raise ValueError("paired consistency requires raw_one_pass jitter")
            if self.registration_jitter_probability <= 0:
                raise ValueError("paired consistency requires positive jitter probability")
            if self.augmentation_profile != "current":
                raise ValueError("paired consistency v1 supports augmentation_profile=current only")
            if self.extra_crop_path is not None:
                raise ValueError("paired consistency v1 does not support extra aligned crops")
        self._crops: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _array(self) -> np.ndarray:
        if self._crops is None:
            self._crops = np.load(self.crop_path, mmap_mode="r")
        return self._crops

    def _extra_array(self) -> np.ndarray:
        if self.extra_crop_path is None:
            raise RuntimeError("extra crop cache was not configured")
        if self._extra_crops is None:
            self._extra_crops = np.load(self.extra_crop_path, mmap_mode="r")
        return self._extra_crops

    def _registration_jitter_raw_array(self) -> np.ndarray:
        if self.registration_jitter_raw_crop_path is None:
            raise RuntimeError("registration jitter raw crop path was not configured")
        if self._registration_jitter_raw_crops is None:
            self._registration_jitter_raw_crops = np.load(
                self.registration_jitter_raw_crop_path, mmap_mode="r"
            )
        return self._registration_jitter_raw_crops

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        sample = self.samples[index]
        target = self.targets[index]
        raw = np.asarray(self._array()[index], dtype=np.float32).copy()
        children_zyx_aligned = target.children_zyx_aligned
        paired_raw: np.ndarray | None = None
        if (
            self.augment
            and self.registration_jitter_probability > 0
            and bool(torch.rand(()) < self.registration_jitter_probability)
        ):
            family = self.registration_jitter_family
            if family == "mixed":
                family = "common" if bool(torch.rand(()) < 0.5) else "independent"
            previous_error, next_error = sample_registration_jitter_pair(
                self.registration_jitter_magnitude_um,
                family,
                self.registration_jitter_magnitude_distribution,
            )
            if self.registration_jitter_application == "raw_one_pass":
                assert self.registration_jitter_flow_rows is not None
                flow_row = self.registration_jitter_flow_rows[index]
                previous_base_warp = np.asarray(
                    [flow_row[f"prev_local_d{axis}_um"] for axis in "zyx"],
                    dtype=np.float64,
                )
                # Stored next_local is forward t->t+1 motion; the output warp is its inverse.
                next_base_warp = -np.asarray(
                    [flow_row[f"next_local_d{axis}_um"] for axis in "zyx"],
                    dtype=np.float64,
                )
                jittered_raw, jittered_children = apply_registration_jitter_to_unaligned_raw_one_pass(
                    np.asarray(self._registration_jitter_raw_array()[index], dtype=np.float32),
                    children_zyx_aligned,
                    previous_base_warp,
                    next_base_warp,
                    previous_error.numpy(),
                    next_error.numpy(),
                )
            else:
                jittered_raw, jittered_children = apply_registration_jitter_to_aligned_raw(
                    raw,
                    children_zyx_aligned,
                    previous_error.numpy(),
                    next_error.numpy(),
                )
            if self.paired_consistency:
                paired_raw = jittered_raw
            else:
                raw, children_zyx_aligned = jittered_raw, jittered_children
        if self.paired_consistency and paired_raw is None:
            paired_raw = raw.copy()
        if self.augment and self.augmentation_profile == "strong_targeted" and should_use_strong_augmentation(
            self.augmentation_profile, sample, self.hard_negative_ids
        ):
            image, offsets, offset_valid = strong_successor_augmentation(
                torch.from_numpy(raw),
                children_zyx_aligned,
                target.heatmap_valid,
                self.offset_scale_um,
            )
            image = normalize_input_channels(
                image, self.channel_mean, self.channel_std
            )
            return {
                "image": image,
                "binary": torch.tensor(float(sample.label), dtype=torch.float32),
                "count": torch.tensor(target.child_count_index, dtype=torch.long),
                "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
                "offset": offsets,
                "offset_valid": offset_valid,
                "index": torch.tensor(index, dtype=torch.long),
            }
        image = input_channels(raw)
        paired_image: torch.Tensor | None = None
        flips = [False, False, False]
        rotations = 0
        if self.augment:
            flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
            if self.augmentation_profile == "algebra_repaired":
                raw_tensor = torch.from_numpy(raw)
                for axis, enabled in enumerate(flips):
                    if enabled:
                        raw_tensor = torch.flip(raw_tensor, dims=(axis - 3,))
            else:
                for axis, enabled in enumerate(flips):
                    if enabled:
                        image = torch.flip(image, dims=(axis - 3,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                if self.augmentation_profile == "algebra_repaired":
                    raw_tensor = torch.rot90(raw_tensor, rotations, dims=(-2, -1))
                else:
                    image = torch.rot90(image, rotations, dims=(-2, -1))
            scale = 0.95 + 0.10 * torch.rand(())
            shift = 0.05 * torch.randn(())
            if self.augmentation_profile == "algebra_repaired":
                image = algebra_repaired_intensity_channels(raw_tensor, scale, shift)
            elif paired_raw is not None:
                image, paired_image = paired_input_channels_with_transform(
                    raw, paired_raw, flips, rotations, scale, shift
                )
            else:
                image = image * scale + shift
        if self.extra_crop_path is not None:
            extra = input_channels(
                np.asarray(self._extra_array()[index], dtype=np.float32).copy()
            )
            for axis, enabled in enumerate(flips):
                if enabled:
                    extra = torch.flip(extra, dims=(axis - 3,))
            if rotations:
                extra = torch.rot90(extra, rotations, dims=(-2, -1))
            image = torch.cat((image, extra), dim=0)
        image = normalize_input_channels(image, self.channel_mean, self.channel_std)
        if paired_image is not None:
            paired_image = normalize_input_channels(
                paired_image, self.channel_mean, self.channel_std
            )
        offsets, offset_valid = ordered_offset_targets(
            children_zyx_aligned,
            raw.shape[-3:],
            flips,
            rotations,
            target.heatmap_valid,
            self.offset_scale_um,
        )
        result = {
            "image": image,
            "binary": torch.tensor(float(sample.label), dtype=torch.float32),
            "count": torch.tensor(target.child_count_index, dtype=torch.long),
            "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
            "offset": offsets,
            "offset_valid": offset_valid,
            "index": torch.tensor(index, dtype=torch.long),
        }
        if paired_image is not None:
            result["paired_image"] = paired_image
        return result


class SmallDivisionSuccessorCNN(nn.Module):
    """Existing SmallDivisionCNN plus paired auxiliary heads."""

    def __init__(self, arm: str, input_channel_count: int = 7):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        base = SmallDivisionCNN(input_channel_count, geometry_dim=0)
        self.arm = arm
        self.features = base.features
        self.binary_classifier = base.classifier
        self.count_head = nn.Linear(128, 3)
        self.offset_head = nn.Linear(128, 6)
        self.feedback_correction = nn.Linear(9, 1)
        nn.init.zeros_(self.feedback_correction.weight)
        nn.init.zeros_(self.feedback_correction.bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.features(image).flatten(1)
        count = self.count_head(shared)
        offset = self.offset_head(shared).reshape(-1, 2, 3)
        binary = self.binary_classifier(shared).squeeze(1)
        if self.arm == "feedback":
            probabilities = torch.softmax(count.float(), dim=1).to(dtype=offset.dtype)
            presence = torch.stack(
                (probabilities[:, 1] + probabilities[:, 2], probabilities[:, 2]), dim=1
            ).unsqueeze(2)
            feedback = torch.cat(
                (probabilities, (torch.tanh(offset) * presence).flatten(1)), dim=1
            ).detach()
            binary = binary + self.feedback_correction(feedback).squeeze(1)
        return {"binary": binary, "count": count, "offset": offset}


def arm_losses(
    arm: str,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_weight: float,
    offset_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    binary = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary"].float(), batch["binary"]
    )
    total = binary
    values = {"binary": float(binary.detach().cpu())}
    if arm != "binary":
        count_per_sample = nn.functional.cross_entropy(
            outputs["count"].float(), batch["count"], reduction="none"
        )
        count = masked_mean(count_per_sample, batch["count_valid"])
        total = total + count_weight * count
        values["count"] = float(count.detach().cpu())
    if arm in {"count_offset_aux", "feedback"}:
        per_coordinate = nn.functional.smooth_l1_loss(
            outputs["offset"].float(), batch["offset"], reduction="none", beta=0.1
        )
        per_slot = per_coordinate.mean(2)
        offset = masked_mean(per_slot, batch["offset_valid"])
        total = total + offset_weight * offset
        values["offset"] = float(offset.detach().cpu())
    return total, values


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    samples: Sequence[Sample],
    device: torch.device,
    amp: str,
    offset_scale_um: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    scores: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    count_true: list[np.ndarray] = []
    count_pred: list[np.ndarray] = []
    count_valid: list[np.ndarray] = []
    offset_errors: list[np.ndarray] = []
    offset_valid: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            cpu_indices = raw_batch["index"].numpy()
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
                if key != "index"
            }
            with amp_context(amp):
                outputs = model(batch["image"])
            scores.append(torch.sigmoid(outputs["binary"].float()).cpu().numpy())
            indices.append(cpu_indices)
            count_true.append(batch["count"].cpu().numpy())
            count_pred.append(outputs["count"].argmax(1).cpu().numpy())
            count_valid.append(batch["count_valid"].bool().cpu().numpy())
            error = torch.abs(outputs["offset"].float() - batch["offset"]).mean(2)
            offset_errors.append((error * offset_scale_um).cpu().numpy())
            offset_valid.append(batch["offset_valid"].bool().cpu().numpy())
    ordered_indices = np.concatenate(indices)
    probabilities = np.concatenate(scores)
    metrics = metrics_with_slices(samples, ordered_indices, probabilities)
    count_true_array = np.concatenate(count_true)
    count_pred_array = np.concatenate(count_pred)
    count_mask = np.concatenate(count_valid)
    offset_error_array = np.concatenate(offset_errors)
    offset_mask = np.concatenate(offset_valid)
    metrics["auxiliary"] = {
        "child_count_accuracy": (
            float(np.mean(count_true_array[count_mask] == count_pred_array[count_mask]))
            if np.any(count_mask) else None
        ),
        "child_count_valid_n": int(np.count_nonzero(count_mask)),
        "successor_offset_mae_um": (
            float(np.mean(offset_error_array[offset_mask])) if np.any(offset_mask) else None
        ),
        "successor_offset_valid_n": int(np.count_nonzero(offset_mask)),
    }
    return probabilities, ordered_indices, metrics


def train_arm(
    arm: str,
    crop_path: Path,
    samples: Sequence[Sample],
    targets: Sequence[SpatialTarget],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    args: argparse.Namespace,
    hard_negative_ids: frozenset[str],
    channel_mean: Sequence[float] | None,
    channel_std: Sequence[float] | None,
    extra_crop_path: Path | None = None,
    raw_crop_path: Path | None = None,
    flow_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    arm_dir = args.output_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_done = arm_dir / "DONE"
    arm_summary = arm_dir / "summary.json"
    if args.resume and arm_done.exists() and arm_summary.is_file():
        print(f"[{arm}] reusing completed arm", flush=True)
        return json.loads(arm_summary.read_text())
    seed_everything(args.seed)
    train_dataset = SuccessorCropDataset(
        crop_path,
        samples,
        targets,
        train_indices,
        True,
        args.offset_scale_um,
        args.augmentation_profile,
        hard_negative_ids,
        channel_mean,
        channel_std,
        extra_crop_path,
        args.registration_jitter_probability,
        args.registration_jitter_magnitude_um,
        args.registration_jitter_family,
        args.registration_jitter_magnitude_distribution,
        args.registration_jitter_application,
        raw_crop_path,
        flow_rows,
        args.paired_consistency_weight > 0,
    )
    val_dataset = SuccessorCropDataset(
        crop_path,
        samples,
        targets,
        val_indices,
        False,
        args.offset_scale_um,
        args.augmentation_profile,
        hard_negative_ids,
        channel_mean,
        channel_std,
        extra_crop_path,
        args.registration_jitter_probability,
        args.registration_jitter_magnitude_um,
        args.registration_jitter_family,
        args.registration_jitter_magnitude_distribution,
        args.registration_jitter_application,
        raw_crop_path,
        flow_rows,
        False,
    )
    weights, sampler_metadata = sampling_weights(
        samples, train_indices, "binary_balanced", frozenset()
    )
    draws = max(args.samples_per_epoch, len(train_indices))
    sampler_generator = torch.Generator().manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed + 1000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            draws,
            replacement=True,
            generator=sampler_generator,
        ),
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
    device = torch.device("cuda")
    model = SmallDivisionSuccessorCNN(arm, 14 if extra_crop_path is not None else 7).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_ap = -math.inf
    best_epoch = -1
    stale = 0
    curve: list[dict[str, Any]] = []
    start_epoch = 0
    runtime_offset = 0.0
    latest_path = arm_dir / "checkpoint_latest.pt"
    resume_contract = training_resume_contract(
        arm, args, samples, train_indices, val_indices
    )
    if args.resume and latest_path.is_file():
        checkpoint = torch.load(
            latest_path, map_location=device, weights_only=False
        )
        validate_resume_contract(checkpoint["resume_contract"], resume_contract)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ap = float(checkpoint["best_ap"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        curve = list(checkpoint["curve"])
        runtime_offset = float(checkpoint.get("runtime_seconds", 0.0))
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        sampler_generator.set_state(checkpoint["sampler_generator_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        print(
            f"[{arm}] resumed from epoch={start_epoch - 1:03d}; "
            f"next_epoch={start_epoch:03d} best={best_ap:.5f}@{best_epoch}",
            flush=True,
        )
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = []
        epoch_consistency = []
        for raw_batch in train_loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw_batch.items()
                if key != "index"
            }
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp):
                outputs = model(batch["image"])
                loss, _ = arm_losses(
                    arm,
                    outputs,
                    batch,
                    args.count_loss_weight,
                    args.offset_loss_weight,
                )
                if args.paired_consistency_weight > 0:
                    paired_outputs = model(batch["paired_image"])
                    consistency = bernoulli_kl_exact_teacher(
                        outputs["binary"], paired_outputs["binary"]
                    )
                    loss = loss + args.paired_consistency_weight * consistency
                    epoch_consistency.append(float(consistency.detach().cpu()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        scheduler.step()
        _, _, metrics = evaluate(
            model, val_loader, samples, device, args.amp, args.offset_scale_um
        )
        ap = float(metrics["parent_vs_all"]["auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_loss)),
            "val_auprc": ap,
            "val_parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"],
            "val_parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"],
            "count_accuracy": metrics["auxiliary"]["child_count_accuracy"],
            "offset_mae_um": metrics["auxiliary"]["successor_offset_mae_um"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "paired_consistency_kl": (
                float(np.mean(epoch_consistency)) if epoch_consistency else None
            ),
        }
        curve.append(row)
        if ap > best_ap + 1e-8:
            best_ap, best_epoch, stale = ap, epoch, 0
            atomic_torch_save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": metrics},
                arm_dir / "checkpoint_best.pt",
            )
        else:
            stale += 1
        atomic_torch_save(
            {
                "schema_version": 1,
                "resume_contract": resume_contract,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "stale": stale,
                "curve": curve,
                "runtime_seconds": runtime_offset + time.perf_counter() - started,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "sampler_generator_state": sampler_generator.get_state(),
                "loader_generator_state": loader_generator.get_state(),
            },
            latest_path,
        )
        print(
            f"[{arm}] epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_ap={ap:.5f} daughter_ap={row['val_parent_vs_daughter_auprc']:.5f} "
            f"best={best_ap:.5f}@{best_epoch}",
            flush=True,
        )
        if epoch + 1 >= min(20, args.epochs) and stale >= args.patience:
            break
    checkpoint = torch.load(
        arm_dir / "checkpoint_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    probabilities, ordered_indices, final_metrics = evaluate(
        model, val_loader, samples, device, args.amp, args.offset_scale_um
    )
    score_rows = []
    for index, probability in zip(ordered_indices, probabilities):
        sample = samples[int(index)]
        score_rows.append({
            "fold": args.fold,
            "arm": arm,
            "sample_id": sample.sample_id,
            "stem": sample.stem,
            "frame": sample.frame,
            "category": sample.category,
            "label": sample.label,
            "prediction": float(probability),
        })
    write_csv(arm_dir / "learning_curve.csv", curve)
    write_csv(arm_dir / "oof_sample_scores.csv", score_rows)
    summary = {
        "arm": arm,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(curve),
        "sampler": sampler_metadata,
        "metrics": final_metrics,
        "runtime_seconds": runtime_offset + time.perf_counter() - started,
        "input_normalization": args.input_normalization,
        "registration_jitter": {
            "probability": args.registration_jitter_probability,
            "magnitude_um": args.registration_jitter_magnitude_um,
            "family": args.registration_jitter_family,
            "magnitude_distribution": args.registration_jitter_magnitude_distribution,
            "application": args.registration_jitter_application,
        },
        "paired_consistency": {
            "weight": args.paired_consistency_weight,
            "teacher": "exact_binary_probability_stop_gradient",
            "student": "paired_jitter_binary_probability",
            "loss": "bernoulli_kl",
            "supervised_branches": "exact_only",
        },
    }
    atomic_json(arm_dir / "summary.json", summary)
    arm_done.touch()
    latest_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.registration_jitter_probability <= 1.0:
        raise ValueError("--registration-jitter-probability must be in [0, 1]")
    if args.registration_jitter_probability > 0 and args.registration_jitter_magnitude_um <= 0:
        raise ValueError("positive registration jitter probability requires positive magnitude")
    if args.paired_consistency_weight < 0:
        raise ValueError("--paired-consistency-weight must be nonnegative")
    if args.paired_consistency_weight > 0:
        if args.registration_jitter_probability <= 0:
            raise ValueError("paired consistency requires positive registration jitter probability")
        if args.registration_jitter_application != "raw_one_pass":
            raise ValueError("paired consistency requires --registration-jitter-application raw_one_pass")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 96)
        args.num_workers = 0
        args.patience = args.epochs
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for directory in ("arms", "cache", "config", "logs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples = read_samples(args.control_dir / "cache/samples.csv")
    raw_crop_path = args.control_dir / "cache/crops.float16.npy"
    if args.smoke:
        selected: list[int] = []
        for split in ("train", "val"):
            partition = [i for i, sample in enumerate(samples) if sample.split == split]
            selected.extend([i for i in partition if samples[i].label][:6])
            selected.extend([i for i in partition if not samples[i].label][:24])
        source = np.load(raw_crop_path, mmap_mode="r")
        smoke_path = args.output_dir / "cache/smoke_raw.float16.npy"
        smoke = np.lib.format.open_memmap(
            smoke_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(selected), *source.shape[1:]),
        )
        for output_index, source_index in enumerate(selected):
            smoke[output_index] = source[source_index]
        smoke.flush()
        del smoke, source
        samples = [samples[index] for index in selected]
        raw_crop_path = smoke_path
    allowed_cases = set()
    for sample in samples:
        allowed_cases.update(
            case_id
            for case_id in (
                neighbor_case_id(sample, -1),
                sample.case_id,
                neighbor_case_id(sample, 1),
            )
            if case_id
        )
    peaks = load_peaks(args.detector_peaks, allowed_cases)
    flow = SparseLocalFlow(peaks, args)
    local_crop_path = args.output_dir / "cache/local_aligned_crops.float16.npy"
    flow_path = args.output_dir / "cache/flow_alignment.csv"
    cached_samples_path = args.output_dir / "cache/samples.csv"
    cache_reusable = (
        args.resume
        and local_crop_path.is_file()
        and flow_path.is_file()
        and cached_samples_path.is_file()
    )
    if cache_reusable:
        cached_samples = read_samples(cached_samples_path)
        with flow_path.open(newline="") as handle:
            flow_rows = list(csv.DictReader(handle))
        if [sample.sample_id for sample in cached_samples] != [
            sample.sample_id for sample in samples
        ]:
            raise RuntimeError("refusing resume with a mismatched aligned-cache sample order")
        if len(flow_rows) != len(samples):
            raise RuntimeError("refusing resume with an incomplete flow-alignment cache")
        print(f"[resume] reusing aligned cache: {local_crop_path}", flush=True)
    else:
        _, local_crop_path, flow_rows = build_aligned_caches(
            raw_crop_path, samples, flow, args.output_dir, write_global=False
        )
        write_csv(flow_path, flow_rows)
        write_csv(cached_samples_path, [asdict(sample) for sample in samples])
    crop_shape = np.load(local_crop_path, mmap_mode="r").shape[-3:]
    targets, target_summary = build_spatial_targets(
        samples,
        flow_rows,
        args.train_dir,
        crop_shape,
        max_match_distance_um=args.max_node_match_distance_um,
    )
    atomic_json(args.output_dir / "config/spatial_target_summary.json", target_summary)
    hard_negative_ids, hard_negative_summary = load_hard_negative_ids(
        args.hard_negative_score_csv,
        args.hard_negative_quantile,
    )
    atomic_json(args.output_dir / "config/hard_negative_summary.json", hard_negative_summary)
    train_indices = [i for i, sample in enumerate(samples) if sample.split == "train"]
    val_indices = [i for i, sample in enumerate(samples) if sample.split == "val"]
    if args.input_normalization == "train_channel_zscore":
        channel_statistics = estimate_input_channel_statistics(
            local_crop_path, train_indices
        )
        channel_mean: Sequence[float] | None = channel_statistics["mean"]
        channel_std: Sequence[float] | None = channel_statistics["std"]
    else:
        channel_statistics = {
            "mode": "current",
            "mean": None,
            "std": None,
            "train_sample_count": len(train_indices),
            "voxel_count_per_channel": 0,
            "source": "no post-difference channel standardization",
        }
        channel_mean = None
        channel_std = None
    atomic_json(
        args.output_dir / "config/input_channel_normalization.json",
        channel_statistics,
    )
    extra_crop_path = args.extra_aligned_crops
    if extra_crop_path is not None:
        extra_shape = np.load(extra_crop_path, mmap_mode="r").shape
        local_shape = np.load(local_crop_path, mmap_mode="r").shape
        if extra_shape != local_shape:
            raise ValueError(f"extra aligned cache shape mismatch: {extra_shape} != {local_shape}")
    summaries = [
        train_arm(
            arm,
            local_crop_path,
            samples,
            targets,
            train_indices,
            val_indices,
            args,
            hard_negative_ids,
            channel_mean,
            channel_std,
            extra_crop_path,
            raw_crop_path,
            flow_rows,
        )
        for arm in args.arms
    ]
    control = summaries[0]["metrics"] if summaries[0]["arm"] == "binary" else None
    comparison = {}
    for summary in summaries:
        comparison[summary["arm"]] = {}
        for metric_slice in ("parent_vs_all", "parent_vs_normal", "parent_vs_daughter"):
            value = summary["metrics"][metric_slice]["auprc"]
            comparison[summary["arm"]][metric_slice] = {
                "auprc": value,
                "delta_vs_binary": (
                    value - control[metric_slice]["auprc"] if control is not None else None
                ),
            }
        comparison[summary["arm"]]["auxiliary"] = summary["metrics"]["auxiliary"]
    final = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold,
        "seed": args.seed,
        "arms": list(args.arms),
        "control_dir": str(args.control_dir),
        "input_mode": "t3_local_diff_signed_abs",
        "extra_aligned_crops": str(extra_crop_path) if extra_crop_path else None,
        "input_normalization": channel_statistics,
        "augmentation": args.augmentation_profile,
        "registration_jitter": {
            "probability": args.registration_jitter_probability,
            "magnitude_um": args.registration_jitter_magnitude_um,
            "family": args.registration_jitter_family,
            "magnitude_distribution": args.registration_jitter_magnitude_distribution,
            "application": args.registration_jitter_application,
            "central_frame_fixed": True,
            "successor_target_follows_next_frame_warp": True,
        },
        "paired_consistency": {
            "weight": args.paired_consistency_weight,
            "teacher": "exact_binary_probability_stop_gradient",
            "student": "paired_jitter_binary_probability",
            "loss": "bernoulli_kl",
            "supervised_branches": "exact_only",
        },
        "hard_negatives": hard_negative_summary,
        "sampler": "binary_balanced",
        "offset_scale_um": args.offset_scale_um,
        "count_loss_weight": args.count_loss_weight,
        "offset_loss_weight": args.offset_loss_weight,
        "sample_count": len(samples),
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(val_indices),
        "spatial_targets": target_summary,
        "comparison": comparison,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "summary.json", final)
    (args.output_dir / "DONE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
