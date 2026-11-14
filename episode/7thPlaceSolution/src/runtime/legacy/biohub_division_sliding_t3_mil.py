#!/usr/bin/env python3
"""Warm-started sliding-T3 temporal aggregation residual Division screen."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import shift as image_shift
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from biohub_division_crop_classifier import (
    SPACING_ZYX,
    SmallDivisionCNN,
    amp_context,
    atomic_json,
    binary_metrics,
    metrics_with_slices,
    worker_seed,
    write_csv,
)
from biohub_division_flow_aligned_diff_classifier import read_samples
from biohub_division_light_unet_multitask import (
    SpatialTarget,
    build_spatial_targets,
    input_channels,
    seed_everything,
)
from biohub_division_small_cnn_multitask import (
    arm_losses,
    atomic_torch_save,
    ordered_offset_targets,
    sample_registration_jitter_pair,
)
from biohub_division_small_cnn_multitask import (
    affine_output_points,
    offsets_from_transformed_points,
    transform_points,
)
from biohub_division_temporal_sequence_screen import (
    ConcatModel,
    FeedbackHeads,
    concat13_channels,
    estimate_channel_statistics,
    evaluate,
    normalize,
)


DEFAULT_ARMS = ("sliding_t3_mil", "sliding_t3_bigru")
SUPPORTED_MODES = ("center", "mil", "bigru", "lstm", "bilstm", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--aligned-t9", type=Path, required=True)
    parser.add_argument(
        "--extra-aligned-crops",
        type=Path,
        help=(
            "Optional aligned T3 OOF peak-raster cache [N,3,D,H,W]. Its seven "
            "derived channels are appended to every sliding raw T3 window."
        ),
    )
    parser.add_argument("--flow-alignment", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-warmstart-checkpoint",
        type=Path,
        help=(
            "Optional full SlidingT3Residual checkpoint produced by a train-only "
            "pretraining stage. Competition validation is still used only by this "
            "fine-tuning run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--freeze-base-epochs", type=int, default=5)
    parser.add_argument(
        "--freeze-encoder-only",
        action="store_true",
        help=(
            "During --freeze-base-epochs, freeze only the image encoder while "
            "keeping temporal layers and prediction heads trainable. The default "
            "preserves the historical encoder+heads freeze policy."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--offset-scale-um", type=float, default=10.0)
    parser.add_argument("--max-node-match-distance-um", type=float, default=7.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--augmentation-profile",
        choices=("current", "strong_targeted", "temporal_imaging_targeted"),
        default="current",
    )
    parser.add_argument("--registration-jitter-probability", type=float, default=0.0)
    parser.add_argument("--registration-jitter-magnitude-um", type=float, default=0.0)
    parser.add_argument(
        "--registration-jitter-family",
        choices=("common", "independent", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--registration-jitter-magnitude-distribution",
        choices=("fixed", "uniform_0_max"),
        default="uniform_0_max",
    )
    parser.add_argument("--official-hard-negative-pool", type=Path)
    parser.add_argument(
        "--official-hard-negative-fraction",
        type=float,
        default=0.0,
        help="Fraction within negative draws assigned to the OOF-hard pool.",
    )
    parser.add_argument(
        "--sampling-profile",
        choices=("binary_balanced", "parent_daughter_normal_2_1_1"),
        default="binary_balanced",
        help=(
            "binary_balanced preserves the historical 1:1 Parent/negative draw mass. "
            "parent_daughter_normal_2_1_1 uses explicit train-only Daughter-centred "
            "crops and assigns Parent/Daughter/Normal mass 0.50/0.25/0.25."
        ),
    )
    parser.add_argument("--daughter-centred-aligned-t9", type=Path)
    parser.add_argument("--daughter-centred-heatmap", type=Path)
    parser.add_argument("--daughter-centred-ledger", type=Path)
    parser.add_argument(
        "--ema-decays",
        default="",
        help="Comma-separated optimizer-step EMA decays, for example 0.995,0.999.",
    )
    parser.add_argument(
        "--extended-epoch-diagnostics",
        action="store_true",
        help=(
            "Persist raw/EMA train-validation loss components, category BCE, "
            "feedback logits, epoch OOF predictions, and trailing-3 AUPRC."
        ),
    )
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated sliding_t3_<mode> arms.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--binary-supervision",
        choices=("exact", "interval_mask_daughter"),
        default="exact",
        help=(
            "exact keeps the historical Parent-vs-all BCE. interval_mask_daughter "
            "uses a three-window positive bag for Parent, requires every window to "
            "be negative for normal, and masks Daughter from the binary loss because "
            "a one-frame boundary-label shift can make it observationally contradictory."
        ),
    )
    parser.add_argument(
        "--role-supervision",
        choices=("binary", "three_class"),
        default="binary",
        help=(
            "binary preserves the historical Parent-vs-all BCE. three_class replaces "
            "that BCE with Normal/Parent/Daughter cross-entropy and uses softmax "
            "P(Parent) as the inference score; successor auxiliaries are unchanged."
        ),
    )
    parser.add_argument("--paired-ranking-aligned-t9", type=Path)
    parser.add_argument("--paired-ranking-heatmap", type=Path)
    parser.add_argument("--paired-ranking-ledger", type=Path)
    parser.add_argument("--paired-ranking-weight", type=float, default=0.0)
    parser.add_argument("--paired-ranking-margin", type=float, default=0.5)
    parser.add_argument("--paired-ranking-batch-size", type=int, default=8)
    parser.add_argument("--event-ownership-factorization", action="store_true")
    parser.add_argument("--factorized-pair-weight", type=float, default=0.25)
    parser.add_argument("--reverse-ancestry-targets", type=Path)
    parser.add_argument("--reverse-ancestry-weight", type=float, default=0.0)
    parser.add_argument(
        "--visible-event-targets",
        type=Path,
        help="Per-sample complete/partial/not-visible event target CSV.",
    )
    parser.add_argument("--visible-event-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--input-channel-mask",
        choices=("none", "raw_diff_mean", "heatmap_diff_zero", "both_diff_off"),
        default="none",
        help=(
            "Training/evaluation ablation applied after raw-channel normalization and "
            "optional heatmap concatenation. Zero is the train mean for raw channels."
        ),
    )
    return parser.parse_args()


def sliding_t3_channels(raw_t9: torch.Tensor) -> torch.Tensor:
    """Return three overlapping T3 seven-channel tensors centred at -1/0/+1."""
    if raw_t9.shape[0] != 9:
        raise ValueError(f"expected T9, got {tuple(raw_t9.shape)}")
    frames = raw_t9[2:7]
    return torch.stack([input_channels(frames[index:index + 3]) for index in range(3)], dim=0)


def sliding_t5_channels(raw_t9: torch.Tensor) -> torch.Tensor:
    """Return three direct T5 13-channel tensors centred at -1/0/+1."""
    if raw_t9.shape[0] != 9:
        raise ValueError(f"expected T9, got {tuple(raw_t9.shape)}")
    return torch.stack(
        [concat13_channels(raw_t9[start:start + 5]) for start in (1, 2, 3)], dim=0
    )


def _shift_volume_aligned_space(volume: np.ndarray, vector_um: np.ndarray) -> np.ndarray:
    """Shift one already-aligned volume and preserve the persisted float16 contract."""
    vector = np.asarray(vector_um, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"registration error must be a ZYX triplet, got {vector.shape}")
    if np.allclose(vector, 0.0):
        return np.asarray(volume, dtype=np.float32).copy()
    shifted = image_shift(
        np.asarray(volume, dtype=np.float32),
        shift=vector / SPACING_ZYX,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return shifted.astype(np.float16).astype(np.float32)


def apply_shift9_training_jitter(
    raw_t9: np.ndarray,
    heatmap_t3: np.ndarray | None,
    children_zyx_aligned: Sequence[Sequence[float]],
    previous_error_um: np.ndarray,
    next_error_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, list[list[float]]]:
    """Apply the inference Shift9 geometry to one training example.

    Past/future frame groups share their residual error, the centre stays fixed,
    and raw/heatmap inputs move together before difference channels are derived.
    """
    raw = np.asarray(raw_t9, dtype=np.float32).copy()
    if raw.shape[0] != 9:
        raise ValueError(f"expected T9 input, got {raw.shape}")
    previous = np.asarray(previous_error_um, dtype=np.float64)
    following = np.asarray(next_error_um, dtype=np.float64)
    for frame in (2, 3):
        raw[frame] = _shift_volume_aligned_space(raw[frame], previous)
    for frame in (5, 6):
        raw[frame] = _shift_volume_aligned_space(raw[frame], -following)

    shifted_heatmap: np.ndarray | None = None
    if heatmap_t3 is not None:
        shifted_heatmap = np.asarray(heatmap_t3, dtype=np.float32).copy()
        if shifted_heatmap.shape[0] != 3:
            raise ValueError(f"expected heatmap T3 input, got {shifted_heatmap.shape}")
        shifted_heatmap[0] = _shift_volume_aligned_space(shifted_heatmap[0], previous)
        shifted_heatmap[2] = _shift_volume_aligned_space(shifted_heatmap[2], -following)

    children = np.asarray(children_zyx_aligned, dtype=np.float64).reshape(-1, 3)
    if len(children):
        children = children - following[None] / SPACING_ZYX[None]
    return raw, shifted_heatmap, children.tolist()


def canonical_sample_id(sample_id: str) -> str:
    """Strip the train__/val__ provenance prefix used by official samples."""
    _, separator, canonical = sample_id.partition("__")
    if not separator:
        raise ValueError(f"unexpected official sample_id: {sample_id}")
    return canonical


def apply_input_channel_mask(image: torch.Tensor, mask: str) -> torch.Tensor:
    """Apply a fixed, label-independent channel ablation to [W,C,D,H,W]."""
    if mask not in {"none", "raw_diff_mean", "heatmap_diff_zero", "both_diff_off"}:
        raise ValueError(mask)
    if mask == "none":
        return image
    result = image.clone()
    if mask in {"raw_diff_mean", "both_diff_off"}:
        result[:, 3:7] = 0.0
    if mask in {"heatmap_diff_zero", "both_diff_off"}:
        if result.shape[1] < 14:
            raise ValueError(f"{mask} requires the 14-channel heatmap input, got {result.shape[1]}")
        result[:, 10:14] = 0.0
    return result


def load_official_hard_negative_ids(path: Path | None) -> tuple[frozenset[str], dict[str, Any]]:
    if path is None:
        return frozenset(), {"source": None, "selected_train_ids": 0}
    if not path.is_file():
        raise FileNotFoundError(path)
    if not (path.parent / "PREFLIGHT_PASSED").exists():
        raise ValueError(f"hard-negative pool has no PREFLIGHT_PASSED marker: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        ids = frozenset(
            row["canonical_sample_id"]
            for row in csv.DictReader(handle)
            if row.get("usage") == "train"
        )
    if not ids:
        raise ValueError(f"hard-negative pool has no train rows: {path}")
    return ids, {"source": str(path), "selected_train_ids": len(ids), "usage": "train"}


def temporal_correlated_imaging_augmentation(
    raw: torch.Tensor,
    *,
    gain_slope: float | None = None,
    gain_curve: float | None = None,
    bias_slope: float | None = None,
    gamma_slope: float | None = None,
    noise_probability: float = 0.50,
    blur_probability: float = 0.25,
) -> torch.Tensor:
    """Add smooth acquisition drift and weak frame-specific noise/PSF variation.

    The transform operates on raw frames before signed/absolute temporal
    differences are derived.  Its low-order curves emulate photobleaching and
    illumination drift without independently recolouring every frame.  Camera
    noise and PSF variation remain frame-specific, as they are in acquisition.

    Explicit coefficients are accepted for deterministic unit tests.  Training
    callers leave them unset and sample conservative ranges.
    """
    if raw.ndim != 4 or raw.shape[0] < 3:
        raise ValueError(f"expected raw [T,D,H,W] with T>=3, got {tuple(raw.shape)}")
    if not 0.0 <= noise_probability <= 1.0:
        raise ValueError("noise_probability must be in [0,1]")
    if not 0.0 <= blur_probability <= 1.0:
        raise ValueError("blur_probability must be in [0,1]")
    if not torch.isfinite(raw).all():
        raise ValueError("raw temporal crop contains non-finite values")

    def sampled(value: float | None, low: float, high: float) -> float:
        return float(torch.empty((), device=raw.device).uniform_(low, high)) if value is None else float(value)

    gain_slope = sampled(gain_slope, -0.12, 0.12)
    gain_curve = sampled(gain_curve, -0.05, 0.05)
    bias_slope = sampled(bias_slope, -0.08, 0.08)
    gamma_slope = sampled(gamma_slope, -0.08, 0.08)

    positions = torch.linspace(-1.0, 1.0, raw.shape[0], dtype=raw.dtype, device=raw.device)
    centred_quadratic = positions.square() - positions.square().mean()
    gain = torch.exp(gain_slope * positions + gain_curve * centred_quadratic)
    gamma = (1.0 + gamma_slope * positions).clamp(0.75, 1.25)
    mean = raw.mean()
    std = raw.std().clamp_min(1e-6)
    normalized = (raw - mean) / std
    normalized = normalized.sign() * normalized.abs().pow(gamma[:, None, None, None])
    augmented = normalized * std * gain[:, None, None, None]
    augmented = augmented + mean + std * bias_slope * positions[:, None, None, None]

    frames: list[torch.Tensor] = []
    for frame in augmented:
        if torch.rand((), device=raw.device) < noise_probability:
            noise_scale = float(torch.empty((), device=raw.device).uniform_(0.005, 0.035))
            frame = frame + torch.randn_like(frame) * std * noise_scale
        if torch.rand((), device=raw.device) < blur_probability:
            # The crop is physically anisotropic, so blur XY only and avoid
            # inventing unsupported through-plane resolution.
            frame = nn.functional.avg_pool3d(
                frame[None, None], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
            )[0, 0]
        frames.append(frame)
    result = torch.stack(frames, dim=0)
    if not torch.isfinite(result).all():
        raise RuntimeError("temporal imaging augmentation produced non-finite values")
    return result


def strong_sliding_t3_augmentation(
    raw_t9: torch.Tensor,
    children_zyx: Sequence[Sequence[float]],
    target_valid: bool,
    offset_scale_um: float,
    temporal_imaging: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one strong transform to all five frames before deriving T3 windows."""
    raw = raw_t9[2:7]
    shape = tuple(int(value) for value in raw.shape[-3:])
    if shape[1] != shape[2]:
        raise ValueError("strong XY rot90 currently requires a square crop")
    centre = torch.tensor([[float(value // 2) for value in shape]], dtype=torch.float32)
    children = torch.as_tensor(children_zyx, dtype=torch.float32).reshape(-1, 3)
    points = torch.cat((centre, children), dim=0)
    flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
    for axis, enabled in enumerate(flips):
        if enabled:
            raw = torch.flip(raw, dims=(axis - 3,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        raw = torch.rot90(raw, rotations, dims=(-2, -1))
    points = transform_points(points, shape, flips, rotations)

    scale_y = float(torch.empty(()).uniform_(0.88, 1.12))
    scale_x = float(torch.empty(()).uniform_(0.88, 1.12))
    theta = torch.tensor(
        [
            [scale_x, 0.0, 0.0, float(torch.empty(()).uniform_(-0.12, 0.12))],
            [0.0, scale_y, 0.0, float(torch.empty(()).uniform_(-0.12, 0.12))],
            [0.0, 0.0, 1.0, float(torch.empty(()).uniform_(-0.08, 0.08))],
        ],
        dtype=raw.dtype,
    )
    grid = nn.functional.affine_grid(theta.unsqueeze(0), raw.unsqueeze(0).shape, align_corners=False)
    raw = nn.functional.grid_sample(
        raw.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=False
    ).squeeze(0)
    points = affine_output_points(points, shape, theta)

    mean = raw.mean()
    std = raw.std().clamp_min(1e-6)
    gamma = float(torch.empty(()).uniform_(0.80, 1.20))
    normalized = (raw - mean) / std
    raw = normalized.sign() * normalized.abs().pow(gamma) * std + mean
    raw = raw * float(torch.empty(()).uniform_(0.80, 1.20))
    raw = raw + std * float(torch.empty(()).uniform_(-0.10, 0.10))
    if torch.rand(()) < 0.50:
        raw = raw + torch.randn_like(raw) * std * float(torch.empty(()).uniform_(0.01, 0.05))
    if torch.rand(()) < 0.25:
        raw = nn.functional.avg_pool3d(
            raw.unsqueeze(0), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        ).squeeze(0)
    if temporal_imaging:
        raw = temporal_correlated_imaging_augmentation(raw)
    windows = torch.stack([input_channels(raw[index:index + 3]) for index in range(3)], dim=0)
    offsets, valid = offsets_from_transformed_points(
        points[0], points[1:], shape, target_valid, offset_scale_um
    )
    return windows, offsets, valid


def mixture_sampling_weights(
    samples,
    train_indices: Sequence[int],
    hard_ids: frozenset[str],
    hard_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError(f"hard-negative fraction must be in [0,1], got {hard_fraction}")
    labels = np.asarray([samples[int(index)].label for index in train_indices], dtype=np.int64)
    positive_positions = np.flatnonzero(labels == 1)
    hard_positions = np.asarray(
        [
            position
            for position, index in enumerate(train_indices)
            if not samples[int(index)].label
            and canonical_sample_id(samples[int(index)].sample_id) in hard_ids
        ],
        dtype=np.int64,
    )
    hard_position_set = set(hard_positions.tolist())
    regular_positions = np.asarray(
        [position for position in np.flatnonzero(labels == 0) if int(position) not in hard_position_set],
        dtype=np.int64,
    )
    if not len(positive_positions) or not len(regular_positions):
        raise ValueError("sampler requires positive and regular-negative samples")
    matched_ids = {
        canonical_sample_id(samples[int(train_indices[position])].sample_id)
        for position in hard_positions
    }
    if hard_ids and matched_ids != set(hard_ids):
        missing = sorted(set(hard_ids) - matched_ids)
        raise ValueError(f"hard-negative rows do not map to this train split: {missing[:10]}")
    if hard_fraction > 0 and not len(hard_positions):
        raise ValueError("positive hard-negative fraction requires matched hard negatives")
    weights = np.zeros(len(train_indices), dtype=np.float64)
    weights[positive_positions] = 0.5 / len(positive_positions)
    if hard_fraction > 0:
        weights[hard_positions] = 0.5 * hard_fraction / len(hard_positions)
        weights[regular_positions] = 0.5 * (1.0 - hard_fraction) / len(regular_positions)
    else:
        negative_positions = np.flatnonzero(labels == 0)
        weights[negative_positions] = 0.5 / len(negative_positions)
    return weights, {
        "profile": "binary_balanced_oof_hard_mixture" if hard_fraction > 0 else "binary_balanced",
        "positive_samples": int(len(positive_positions)),
        "hard_negative_samples": int(len(hard_positions)),
        "regular_negative_samples": int(len(regular_positions)),
        "hard_fraction_within_negative_draws": float(hard_fraction),
        "expected_positive_mass": float(weights[positive_positions].sum()),
        "expected_hard_negative_mass": float(weights[hard_positions].sum()) if len(hard_positions) else 0.0,
        "expected_regular_negative_mass": float(weights[regular_positions].sum()),
    }


class SlidingT3Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        samples,
        targets: Sequence[SpatialTarget],
        indices: Sequence[int],
        augment: bool,
        offset_scale_um: float,
        mean: Sequence[float],
        std: Sequence[float],
        augmentation_profile: str = "current",
        hard_negative_ids: frozenset[str] = frozenset(),
        extra_crop_path: Path | None = None,
        representation: str = "t3",
        registration_jitter_probability: float = 0.0,
        registration_jitter_magnitude_um: float = 0.0,
        registration_jitter_family: str = "mixed",
        registration_jitter_magnitude_distribution: str = "uniform_0_max",
        input_channel_mask: str = "none",
        visible_event_targets: Path | None = None,
    ) -> None:
        self.path = path
        self.samples = samples
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.offset_scale_um = offset_scale_um
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.augmentation_profile = augmentation_profile
        self.hard_negative_ids = hard_negative_ids
        self.extra_crop_path = extra_crop_path
        self.registration_jitter_probability = registration_jitter_probability
        self.registration_jitter_magnitude_um = registration_jitter_magnitude_um
        self.registration_jitter_family = registration_jitter_family
        self.registration_jitter_magnitude_distribution = registration_jitter_magnitude_distribution
        self.input_channel_mask = input_channel_mask
        self.visible_event_target = np.zeros(len(samples), dtype=np.float32)
        self.visible_event_valid = np.zeros(len(samples), dtype=np.float32)
        if visible_event_targets is not None:
            with visible_event_targets.open(newline="", encoding="utf-8") as handle:
                rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
            expected = {sample.sample_id for sample in samples}
            if set(rows) != expected:
                missing = sorted(expected - set(rows))
                extra = sorted(set(rows) - expected)
                raise ValueError(
                    f"visible event target/sample mismatch: missing={missing[:5]} extra={extra[:5]}"
                )
            for index, sample in enumerate(samples):
                row = rows[sample.sample_id]
                self.visible_event_target[index] = float(int(row["target"]))
                self.visible_event_valid[index] = float(int(row["valid"]))
        if not 0.0 <= registration_jitter_probability <= 1.0:
            raise ValueError("registration jitter probability must be in [0, 1]")
        if registration_jitter_magnitude_um < 0:
            raise ValueError("registration jitter magnitude must be nonnegative")
        if registration_jitter_probability > 0 and registration_jitter_magnitude_um <= 0:
            raise ValueError("positive jitter probability requires positive magnitude")
        if registration_jitter_family not in {"common", "independent", "mixed"}:
            raise ValueError(f"unknown registration jitter family: {registration_jitter_family}")
        if registration_jitter_magnitude_distribution not in {"fixed", "uniform_0_max"}:
            raise ValueError(
                "unknown registration jitter magnitude distribution: "
                f"{registration_jitter_magnitude_distribution}"
            )
        if input_channel_mask not in {"none", "raw_diff_mean", "heatmap_diff_zero", "both_diff_off"}:
            raise ValueError(input_channel_mask)
        if representation not in {"t3", "t5"}:
            raise ValueError(representation)
        self.representation = representation
        self._array = None
        self._extra_array = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        sample = self.samples[index]
        target = self.targets[index]
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        raw_array = np.asarray(self._array[index], dtype=np.float32).copy()
        extra_array = None
        if self.extra_crop_path is not None:
            if self._extra_array is None:
                self._extra_array = np.load(self.extra_crop_path, mmap_mode="r")
            extra_array = np.asarray(self._extra_array[index], dtype=np.float32).copy()
        children_zyx_aligned = target.children_zyx_aligned
        if self.augment and self.registration_jitter_probability > 0:
            # Jitter must not perturb the historical flip/rotation/intensity RNG
            # stream, so the matched exact continuation sees the same transforms.
            random_state = torch.random.get_rng_state()
            try:
                apply_jitter = bool(torch.rand(()) < self.registration_jitter_probability)
                if apply_jitter:
                    family = self.registration_jitter_family
                    if family == "mixed":
                        family = "common" if bool(torch.rand(()) < 0.5) else "independent"
                    previous_error, next_error = sample_registration_jitter_pair(
                        self.registration_jitter_magnitude_um,
                        family,
                        self.registration_jitter_magnitude_distribution,
                    )
                    raw_array, extra_array, children_zyx_aligned = apply_shift9_training_jitter(
                        raw_array,
                        extra_array,
                        children_zyx_aligned,
                        previous_error.numpy(),
                        next_error.numpy(),
                    )
            finally:
                torch.random.set_rng_state(random_state)
        raw = torch.from_numpy(raw_array)
        extra = torch.from_numpy(extra_array) if extra_array is not None else None
        targeted_profiles = {"strong_targeted", "temporal_imaging_targeted"}
        use_strong = (
            self.augment
            and self.augmentation_profile in targeted_profiles
            and (
                sample.category in {"parent", "daughter"}
                or canonical_sample_id(sample.sample_id) in self.hard_negative_ids
            )
        )
        if use_strong:
            if self.representation != "t3":
                raise ValueError("strong_targeted is not implemented for direct T5")
            if extra is not None:
                raise ValueError(
                    "strong_targeted with --extra-aligned-crops is not implemented; "
                    "the affine transform must be shared exactly"
                )
            image, offsets, offset_valid = strong_sliding_t3_augmentation(
                raw,
                children_zyx_aligned,
                target.heatmap_valid,
                self.offset_scale_um,
                temporal_imaging=self.augmentation_profile == "temporal_imaging_targeted",
            )
            image = apply_input_channel_mask(normalize(image, self.mean, self.std), self.input_channel_mask)
            return {
                "image": image,
                "binary": torch.tensor(float(sample.label), dtype=torch.float32),
                "category": torch.tensor(
                    {"normal": 0, "parent": 1, "daughter": 2}[sample.category],
                    dtype=torch.long,
                ),
                "count": torch.tensor(target.child_count_index, dtype=torch.long),
                "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
                "offset": offsets,
                "offset_valid": offset_valid,
                "timing": torch.tensor(2 if sample.label else 5, dtype=torch.long),
                "visible_event": torch.tensor(self.visible_event_target[index], dtype=torch.float32),
                "visible_event_valid": torch.tensor(self.visible_event_valid[index], dtype=torch.float32),
                "index": torch.tensor(index, dtype=torch.long),
            }
        flips = [False, False, False]
        rotations = 0
        if self.augment:
            flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
            for axis, enabled in enumerate(flips):
                if enabled:
                    raw = torch.flip(raw, dims=(axis - 3,))
                    if extra is not None:
                        extra = torch.flip(extra, dims=(axis - 3,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                raw = torch.rot90(raw, rotations, dims=(-2, -1))
                if extra is not None:
                    extra = torch.rot90(extra, rotations, dims=(-2, -1))
            # Shared raw-frame transform; difference channels are derived afterwards.
            raw = raw * (0.95 + 0.10 * torch.rand(())) + 0.05 * torch.randn(())
        image = normalize(
            sliding_t3_channels(raw) if self.representation == "t3" else sliding_t5_channels(raw),
            self.mean,
            self.std,
        )
        if extra is not None:
            if self.representation != "t3":
                raise ValueError("aligned heatmap concatenation is not implemented for direct T5")
            heatmap = input_channels(extra)
            if not torch.isfinite(heatmap).all():
                raise ValueError(f"non-finite heatmap channels at sample index {index}")
            image = torch.cat((image, heatmap.unsqueeze(0).expand(3, -1, -1, -1, -1)), dim=1)
        image = apply_input_channel_mask(image, self.input_channel_mask)
        offsets, offset_valid = ordered_offset_targets(
            children_zyx_aligned,
            raw.shape[-3:],
            flips,
            rotations,
            target.heatmap_valid,
            self.offset_scale_um,
        )
        return {
            "image": image,
            "binary": torch.tensor(float(sample.label), dtype=torch.float32),
            "category": torch.tensor(
                {"normal": 0, "parent": 1, "daughter": 2}[sample.category],
                dtype=torch.long,
            ),
            "count": torch.tensor(target.child_count_index, dtype=torch.long),
            "count_valid": torch.tensor(float(target.child_count_valid), dtype=torch.float32),
            "offset": offsets,
            "offset_valid": offset_valid,
            "timing": torch.tensor(2 if sample.label else 5, dtype=torch.long),
            "visible_event": torch.tensor(self.visible_event_target[index], dtype=torch.float32),
            "visible_event_valid": torch.tensor(self.visible_event_valid[index], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


class PairedRankingDataset(Dataset):
    """Train-only same-event Parent/Daughter views with shared augmentation."""

    def __init__(
        self,
        aligned_t9: Path,
        heatmap_t3: Path,
        ledger: Path,
        mean: Sequence[float],
        std: Sequence[float],
        augment: bool,
        reverse_ancestry_targets: Path | None = None,
        offset_scale_um: float = 10.0,
    ) -> None:
        self.aligned_t9 = aligned_t9
        self.heatmap_t3 = heatmap_t3
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.augment = augment
        self.offset_scale_um = float(offset_scale_um)
        if self.offset_scale_um <= 0:
            raise ValueError("offset_scale_um must be positive")
        with ledger.open(newline="", encoding="utf-8") as handle:
            self.pairs = list(csv.DictReader(handle))
        if not self.pairs:
            raise ValueError(f"paired-ranking ledger is empty: {ledger}")
        if any(row.get("split") != "train" for row in self.pairs):
            raise ValueError("paired-ranking ledger must contain train rows only")
        raw = np.load(aligned_t9, mmap_mode="r")
        heatmap = np.load(heatmap_t3, mmap_mode="r")
        if raw.ndim != 5 or raw.shape[1] != 9:
            raise ValueError(f"paired T9 cache must be [N,9,D,H,W], got {raw.shape}")
        if heatmap.shape != (raw.shape[0], 3, *raw.shape[-3:]):
            raise ValueError(f"paired heatmap mismatch: raw={raw.shape} heatmap={heatmap.shape}")
        maximum = max(
            max(int(row["parent_index"]), int(row["daughter_index"]))
            for row in self.pairs
        )
        if maximum >= raw.shape[0]:
            raise ValueError(f"paired ledger index {maximum} exceeds cache rows {raw.shape[0]}")
        self.reverse_targets: list[dict[str, str]] | None = None
        if reverse_ancestry_targets is not None:
            with reverse_ancestry_targets.open(newline="", encoding="utf-8") as handle:
                targets = list(csv.DictReader(handle))
            if len(targets) != len(self.pairs):
                raise ValueError(
                    f"reverse target/pair mismatch: {len(targets)} vs {len(self.pairs)}"
                )
            ordered = sorted(targets, key=lambda row: int(row["pair_index"]))
            if [int(row["pair_index"]) for row in ordered] != list(range(len(self.pairs))):
                raise ValueError("reverse ancestry pair_index must be contiguous and ordered")
            self.reverse_targets = ordered
        self._raw = None
        self._heatmap = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self._raw is None:
            self._raw = np.load(self.aligned_t9, mmap_mode="r")
            self._heatmap = np.load(self.heatmap_t3, mmap_mode="r")
        row = self.pairs[item]
        indices = (int(row["parent_index"]), int(row["daughter_index"]))
        raw_views = [torch.from_numpy(np.asarray(self._raw[index], dtype=np.float32).copy()) for index in indices]
        heatmap_views = [
            torch.from_numpy(np.asarray(self._heatmap[index], dtype=np.float32).copy())
            for index in indices
        ]
        if self.augment:
            flips = [bool(torch.rand(()) < 0.5) for _ in range(3)]
            rotations = int(torch.randint(0, 4, ()).item())
            scale = 0.95 + 0.10 * torch.rand(())
            shift = 0.05 * torch.randn(())
            for view in range(2):
                for axis, enabled in enumerate(flips):
                    if enabled:
                        raw_views[view] = torch.flip(raw_views[view], dims=(axis - 3,))
                        heatmap_views[view] = torch.flip(heatmap_views[view], dims=(axis - 3,))
                if rotations:
                    raw_views[view] = torch.rot90(raw_views[view], rotations, dims=(-2, -1))
                    heatmap_views[view] = torch.rot90(heatmap_views[view], rotations, dims=(-2, -1))
                raw_views[view] = raw_views[view] * scale + shift

        reverse_offset = torch.zeros((2, 3), dtype=torch.float32)
        reverse_valid = torch.zeros(2, dtype=torch.float32)
        if self.reverse_targets is not None:
            target = self.reverse_targets[item]
            vectors_um = torch.tensor(
                [
                    [float(target[f"aligned_previous_parent_d{axis}_um"]) for axis in ("z", "y", "x")],
                    [float(target[f"same_frame_sister_d{axis}_um"]) for axis in ("z", "y", "x")],
                ],
                dtype=torch.float32,
            )
            shape = tuple(int(value) for value in raw_views[0].shape[-3:])
            centre = torch.tensor([[float(value // 2) for value in shape]], dtype=torch.float32)
            spacing = torch.as_tensor(SPACING_ZYX, dtype=torch.float32)
            points = centre + vectors_um / spacing[None]
            transformed = transform_points(
                torch.cat((centre, points), dim=0), shape, flips if self.augment else (False,) * 3,
                rotations if self.augment else 0,
            )
            reverse_offset = (
                (transformed[1:] - transformed[:1]) * spacing[None]
            ) / self.offset_scale_um
            reverse_valid[:] = float(int(target["valid"]))

        images = []
        for raw, heatmap in zip(raw_views, heatmap_views):
            image = normalize(sliding_t3_channels(raw), self.mean, self.std)
            heatmap_channels = input_channels(heatmap)
            image = torch.cat(
                (image, heatmap_channels.unsqueeze(0).expand(3, -1, -1, -1, -1)),
                dim=1,
            )
            images.append(image)
        return {
            "parent_image": images[0],
            "daughter_image": images[1],
            "event_id": row["event_id"],
            "reverse_offset": reverse_offset,
            "reverse_valid": reverse_valid,
        }


class DaughterCentredPointwiseDataset(Dataset):
    """Expose train-only daughter views as binary-negative pointwise samples.

    These samples are generated from exact Parent->Daughter lineage edges.  They
    intentionally supervise only the Parent binary head; successor auxiliary
    labels are masked because the pair cache does not assert a complete
    Daughter->successor target contract.
    """

    def __init__(self, paired: PairedRankingDataset) -> None:
        self.paired = paired

    def __len__(self) -> int:
        return len(self.paired)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        pair = self.paired[item]
        return {
            "image": pair["daughter_image"],
            "binary": torch.tensor(0.0, dtype=torch.float32),
            "category": torch.tensor(2, dtype=torch.long),
            "count": torch.tensor(0, dtype=torch.long),
            "count_valid": torch.tensor(0.0, dtype=torch.float32),
            "offset": torch.zeros((2, 3), dtype=torch.float32),
            "offset_valid": torch.zeros(2, dtype=torch.float32),
            "timing": torch.tensor(5, dtype=torch.long),
            "visible_event": torch.tensor(0.0, dtype=torch.float32),
            "visible_event_valid": torch.tensor(0.0, dtype=torch.float32),
            # Negative values cannot collide with the main sample index space.
            "index": torch.tensor(-(item + 1), dtype=torch.long),
        }


class CombinedPointwiseDataset(Dataset):
    def __init__(self, main: Dataset, daughter: DaughterCentredPointwiseDataset) -> None:
        self.main = main
        self.daughter = daughter

    def __len__(self) -> int:
        return len(self.main) + len(self.daughter)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if item < len(self.main):
            return self.main[item]
        return self.daughter[item - len(self.main)]


def category_2_1_1_sampling_weights(
    samples,
    train_indices: Sequence[int],
    main_length: int,
    daughter_length: int,
    hard_ids: frozenset[str],
    hard_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parent/Daughter/Normal draw mass 0.50/0.25/0.25.

    Historical pointwise Daughter rows receive zero mass; the explicit
    Daughter-centred lineage cache is the sole Daughter source so the data
    construction remains auditable.
    """
    if daughter_length <= 0:
        raise ValueError("2:1:1 sampling requires Daughter-centred rows")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard-negative fraction must be in [0,1]")
    if main_length != len(train_indices):
        raise ValueError("main dataset/train index mismatch")
    parent = np.asarray(
        [position for position, index in enumerate(train_indices) if samples[int(index)].category == "parent"],
        dtype=np.int64,
    )
    normal = np.asarray(
        [position for position, index in enumerate(train_indices) if samples[int(index)].category == "normal"],
        dtype=np.int64,
    )
    ignored_historical_daughter = np.asarray(
        [position for position, index in enumerate(train_indices) if samples[int(index)].category == "daughter"],
        dtype=np.int64,
    )
    hard = np.asarray(
        [
            position
            for position in normal
            if canonical_sample_id(samples[int(train_indices[int(position)])].sample_id) in hard_ids
        ],
        dtype=np.int64,
    )
    hard_set = set(hard.tolist())
    regular_normal = np.asarray(
        [position for position in normal if int(position) not in hard_set], dtype=np.int64
    )
    if not len(parent) or not len(regular_normal):
        raise ValueError("2:1:1 sampler requires Parent and regular Normal rows")
    if hard_fraction > 0 and not len(hard):
        raise ValueError("positive hard-negative fraction requires matched hard Normal rows")
    weights = np.zeros(main_length + daughter_length, dtype=np.float64)
    weights[parent] = 0.50 / len(parent)
    daughter_positions = np.arange(main_length, main_length + daughter_length, dtype=np.int64)
    weights[daughter_positions] = 0.25 / daughter_length
    if hard_fraction > 0:
        weights[hard] = 0.25 * hard_fraction / len(hard)
        weights[regular_normal] = 0.25 * (1.0 - hard_fraction) / len(regular_normal)
    else:
        weights[normal] = 0.25 / len(normal)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError(f"2:1:1 sampler mass does not sum to one: {weights.sum()}")
    return weights, {
        "profile": "parent_daughter_normal_2_1_1",
        "parent_samples": int(len(parent)),
        "daughter_centred_samples": int(daughter_length),
        "normal_samples": int(len(normal)),
        "ignored_historical_daughter_samples": int(len(ignored_historical_daughter)),
        "hard_normal_samples": int(len(hard)),
        "regular_normal_samples": int(len(regular_normal)),
        "expected_parent_mass": float(weights[parent].sum()),
        "expected_daughter_mass": float(weights[daughter_positions].sum()),
        "expected_normal_mass": float(weights[normal].sum()),
        "hard_fraction_within_normal_draws": float(hard_fraction),
        "daughter_auxiliary_targets": "count/offset/timing masked; binary Parent-negative only",
    }


class CenterControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = ConcatModel(7)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.base(image[:, 1, :7])


class SlidingT3Residual(nn.Module):
    def __init__(
        self,
        mode: str,
        in_channels: int = 7,
        event_ownership_factorization: bool = False,
        visible_event_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(mode)
        self.mode = mode
        if in_channels < 7:
            raise ValueError(f"in_channels must be at least 7, got {in_channels}")
        self.in_channels = in_channels
        self.event_ownership_factorization = event_ownership_factorization
        self.visible_event_auxiliary = visible_event_auxiliary
        if event_ownership_factorization and visible_event_auxiliary:
            raise ValueError("factorized and visible-event-only heads are separate ablations")
        self.encoder = SmallDivisionCNN(in_channels).features
        self.heads = FeedbackHeads()
        if event_ownership_factorization or visible_event_auxiliary:
            self.event_head = nn.Linear(128, 1)
            nn.init.zeros_(self.event_head.weight)
            nn.init.zeros_(self.event_head.bias)
        if event_ownership_factorization:
            self.reverse_ancestry_head = nn.Linear(128, 6)
            nn.init.constant_(self.event_head.bias, math.log(99.0))
            nn.init.zeros_(self.reverse_ancestry_head.weight)
            nn.init.zeros_(self.reverse_ancestry_head.bias)
        # The explicit-role residual starts as an exact decomposition of the
        # historical binary probability.  With Normal/Daughter logits both
        # -log(2), Parent logit equal to the legacy binary logit, softmax
        # P(Parent) == sigmoid(binary).  The zero-initialized residual can then
        # learn anchor ownership without creating an epoch-zero score shift.
        self.role_residual = nn.Linear(128, 3)
        nn.init.zeros_(self.role_residual.weight)
        nn.init.zeros_(self.role_residual.bias)
        self.position = nn.Parameter(torch.zeros(3, 128))
        if mode == "center":
            # Matched non-temporal arm for the C0/H factorial.  It consumes
            # only the centre T3 window; with 14 channels this tests the
            # aligned heatmap prior without adding a sliding-window temporal
            # aggregator.  Extra-channel kernels start at zero in
            # load_control_state, preserving epoch-zero equivalence.
            pass
        elif mode == "mil":
            self.attn_v = nn.Linear(128, 64)
            self.attn_u = nn.Linear(128, 64)
            self.attn_score = nn.Linear(64, 1)
        elif mode == "bigru":
            self.recurrent = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        elif mode == "lstm":
            # Use the final state so the one-way model has observed all three
            # ordered windows. This intentionally tests directionality rather
            # than pretending the centre state can see the future window.
            self.recurrent = nn.LSTM(128, 128, batch_first=True)
        elif mode == "bilstm":
            self.recurrent = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(128))
        self.residual = nn.Linear(128, 128)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, windows, channels, depth, height, width = image.shape
        tokens = self.encoder(image.reshape(batch * windows, channels, depth, height, width)).flatten(1)
        tokens = tokens.reshape(batch, windows, 128)
        centre = tokens[:, 1]
        positioned = tokens + self.position.unsqueeze(0)
        if self.mode == "center":
            outputs = self.heads(centre)
            outputs = self.attach_factorized_heads(outputs, centre)
            outputs["role"] = self.role_logits(outputs["binary"], centre)
            outputs["window_binary"] = self.parent_logits_from_features(
                tokens.reshape(-1, 128)
            ).reshape(batch, windows)
            outputs["attention"] = torch.zeros(
                (batch, 3), dtype=tokens.dtype, device=tokens.device
            )
            return outputs
        if self.mode == "mil":
            gated = torch.tanh(self.attn_v(positioned)) * torch.sigmoid(self.attn_u(positioned))
            weights = torch.softmax(self.attn_score(gated).squeeze(2), dim=1)
            context = torch.sum(tokens * weights.unsqueeze(2), dim=1)
        elif self.mode in {"bigru", "bilstm"}:
            sequence, _ = self.recurrent(positioned)
            context = sequence[:, 1]
            contextual_tokens = tokens + self.residual(sequence - tokens)
            weights = torch.zeros((batch, 3), dtype=tokens.dtype, device=tokens.device)
        elif self.mode == "lstm":
            sequence, _ = self.recurrent(positioned)
            context = sequence[:, -1]
            weights = torch.zeros((batch, 3), dtype=tokens.dtype, device=tokens.device)
        else:
            context = self.temporal(positioned)[:, 1]
            weights = torch.zeros((batch, 3), dtype=tokens.dtype, device=tokens.device)
        final_features = centre + self.residual(context - centre)
        outputs = self.heads(final_features)
        outputs = self.attach_factorized_heads(outputs, final_features)
        outputs["role"] = self.role_logits(outputs["binary"], final_features)
        if self.mode not in {"bigru", "bilstm"}:
            contextual_tokens = tokens
        outputs["window_binary"] = self.parent_logits_from_features(
            contextual_tokens.reshape(-1, 128)
        ).reshape(batch, windows)
        outputs["attention"] = weights
        return outputs

    @staticmethod
    def probability_product_logit(event_logit: torch.Tensor, ownership_logit: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(event_logit.float()) * torch.sigmoid(ownership_logit.float())
        return torch.logit(probability.clamp(1e-6, 1.0 - 1e-6)).to(dtype=ownership_logit.dtype)

    def attach_factorized_heads(
        self, outputs: dict[str, torch.Tensor], features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.visible_event_auxiliary:
            outputs["visible_event"] = self.event_head(features).squeeze(1)
        if not self.event_ownership_factorization:
            return outputs
        ownership = outputs["binary"]
        event = self.event_head(features).squeeze(1)
        outputs["ownership"] = ownership
        outputs["event"] = event
        outputs["reverse_ancestry"] = self.reverse_ancestry_head(features).reshape(-1, 2, 3)
        outputs["binary"] = self.probability_product_logit(event, ownership)
        return outputs

    def parent_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        ownership = self.heads.binary(features).squeeze(1)
        if not self.event_ownership_factorization:
            return ownership
        event = self.event_head(features).squeeze(1)
        return self.probability_product_logit(event, ownership)

    def role_logits(self, binary: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        base = torch.stack(
            (
                torch.full_like(binary, -math.log(2.0)),
                binary,
                torch.full_like(binary, -math.log(2.0)),
            ),
            dim=1,
        )
        return base + self.role_residual(features)

    def load_control_state(
        self,
        state: dict[str, torch.Tensor],
        source_channel_targets: Sequence[int] | None = None,
    ) -> None:
        encoder = {key.removeprefix("features."): value for key, value in state.items() if key.startswith("features.")}
        heads = {key.removeprefix("heads."): value for key, value in state.items() if key.startswith("heads.")}
        if self.in_channels > 7:
            first_key = "0.block.0.weight"
            source = encoder[first_key]
            target = self.encoder.state_dict()[first_key]
            if source.shape[1] != 7 or target.shape[1] != self.in_channels:
                raise ValueError(
                    f"unexpected first-convolution shapes: source={tuple(source.shape)} "
                    f"target={tuple(target.shape)}"
                )
            expanded = torch.zeros_like(target)
            targets = tuple(source_channel_targets or range(7))
            if len(targets) != 7 or len(set(targets)) != 7 or min(targets) < 0 or max(targets) >= self.in_channels:
                raise ValueError(f"invalid source channel targets: {targets}")
            for source_index, target_index in enumerate(targets):
                expanded[:, target_index].copy_(source[:, source_index])
            encoder[first_key] = expanded
        self.encoder.load_state_dict(encoder, strict=True)
        self.heads.load_state_dict(heads, strict=True)


def make_loaders(path, samples, targets, train_indices, val_indices, stats, hard_ids, args, representation="t3"):
    # Older auxiliary callers predate the optional aligned-heatmap input.
    # Treat an absent attribute exactly like the CLI default (disabled).
    extra_aligned_crops = getattr(args, "extra_aligned_crops", None)
    train = SlidingT3Dataset(
        path, samples, targets, train_indices, True, args.offset_scale_um, stats["mean"], stats["std"],
        args.augmentation_profile, hard_ids, extra_aligned_crops, representation,
        getattr(args, "registration_jitter_probability", 0.0),
        getattr(args, "registration_jitter_magnitude_um", 0.0),
        getattr(args, "registration_jitter_family", "mixed"),
        getattr(args, "registration_jitter_magnitude_distribution", "uniform_0_max"),
        getattr(args, "input_channel_mask", "none"),
        getattr(args, "visible_event_targets", None),
    )
    val = SlidingT3Dataset(
        path, samples, targets, val_indices, False, args.offset_scale_um, stats["mean"], stats["std"],
        "current", hard_ids, extra_aligned_crops, representation,
        0.0, 0.0, "mixed", "uniform_0_max", getattr(args, "input_channel_mask", "none"),
        getattr(args, "visible_event_targets", None),
    )
    if getattr(args, "sampling_profile", "binary_balanced") == "parent_daughter_normal_2_1_1":
        daughter_paths = (
            getattr(args, "daughter_centred_aligned_t9", None),
            getattr(args, "daughter_centred_heatmap", None),
            getattr(args, "daughter_centred_ledger", None),
        )
        if not all(daughter_paths):
            raise ValueError("2:1:1 sampling requires all Daughter-centred cache paths")
        if representation != "t3" or extra_aligned_crops is None:
            raise ValueError("Daughter-centred sampling requires T3 plus aligned heatmap input")
        daughter = DaughterCentredPointwiseDataset(
            PairedRankingDataset(
                daughter_paths[0], daughter_paths[1], daughter_paths[2],
                stats["mean"], stats["std"], augment=True,
                offset_scale_um=args.offset_scale_um,
            )
        )
        main_length = len(train)
        train = CombinedPointwiseDataset(train, daughter)
        weights, sampler_metadata = category_2_1_1_sampling_weights(
            samples, train_indices, main_length, len(daughter), hard_ids,
            args.official_hard_negative_fraction,
        )
    else:
        weights, sampler_metadata = mixture_sampling_weights(
            samples, train_indices, hard_ids, args.official_hard_negative_fraction
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), max(args.samples_per_epoch, len(train_indices)),
                                      replacement=True, generator=generator),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
    )
    val_loader = DataLoader(val, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed)
    return train_loader, val_loader, sampler_metadata


def make_diagnostic_loaders(
    path, samples, targets, train_indices, val_indices, stats, hard_ids, args,
    representation="t3",
) -> tuple[DataLoader, DataLoader]:
    """Deterministic, no-augmentation loaders for epoch loss diagnostics."""
    extra_aligned_crops = getattr(args, "extra_aligned_crops", None)
    common = dict(
        offset_scale_um=args.offset_scale_um,
        mean=stats["mean"], std=stats["std"], hard_negative_ids=hard_ids,
        extra_crop_path=extra_aligned_crops, representation=representation,
        input_channel_mask=getattr(args, "input_channel_mask", "none"),
        visible_event_targets=getattr(args, "visible_event_targets", None),
    )
    train: Dataset = SlidingT3Dataset(
        path, samples, targets, train_indices, augment=False,
        augmentation_profile="current", **common,
    )
    if getattr(args, "sampling_profile", "binary_balanced") == "parent_daughter_normal_2_1_1":
        paired = PairedRankingDataset(
            args.daughter_centred_aligned_t9, args.daughter_centred_heatmap,
            args.daughter_centred_ledger, stats["mean"], stats["std"], augment=False,
            offset_scale_um=args.offset_scale_um,
        )
        train = CombinedPointwiseDataset(train, DaughterCentredPointwiseDataset(paired))
    val = SlidingT3Dataset(
        path, samples, targets, val_indices, augment=False,
        augmentation_profile="current", **common,
    )
    loader_args = dict(
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed,
    )
    return DataLoader(train, **loader_args), DataLoader(val, **loader_args)


def make_paired_ranking_loader(args: argparse.Namespace, stats: dict[str, Any]) -> DataLoader | None:
    paths = (
        args.paired_ranking_aligned_t9,
        args.paired_ranking_heatmap,
        args.paired_ranking_ledger,
    )
    if not any(paths):
        if args.paired_ranking_weight != 0 or args.event_ownership_factorization:
            raise ValueError("paired supervision requires all pair artifacts")
        return None
    if not all(paths):
        raise ValueError("paired ranking requires aligned T9, heatmap, and ledger together")
    if args.paired_ranking_weight <= 0 and not args.event_ownership_factorization:
        raise ValueError("pair artifacts require paired ranking or event-ownership factorization")
    dataset = PairedRankingDataset(
        args.paired_ranking_aligned_t9,
        args.paired_ranking_heatmap,
        args.paired_ranking_ledger,
        stats["mean"],
        stats["std"],
        augment=True,
        reverse_ancestry_targets=args.reverse_ancestry_targets,
        offset_scale_um=args.offset_scale_um,
    )
    generator = torch.Generator().manual_seed(args.seed + 7919)
    return DataLoader(
        dataset,
        batch_size=args.paired_ranking_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
    )


def paired_ranking_loss(
    parent_logits: torch.Tensor,
    daughter_logits: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if parent_logits.shape != daughter_logits.shape or parent_logits.ndim != 1:
        raise ValueError(
            f"paired logits must be matching vectors, got {parent_logits.shape}/{daughter_logits.shape}"
        )
    difference = parent_logits.float() - daughter_logits.float()
    loss = nn.functional.softplus(float(margin) - difference).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "accuracy": float((difference > 0).float().mean().detach().cpu()),
        "margin_satisfied": float((difference >= margin).float().mean().detach().cpu()),
        "mean_logit_margin": float(difference.mean().detach().cpu()),
    }


def role_ownership_logit(role_logits: torch.Tensor) -> torch.Tensor:
    """Return the Parent-vs-Daughter log-odds implied by a 3-class role head."""
    if role_logits.ndim != 2 or role_logits.shape[1] != 3:
        raise ValueError(f"role logits must have shape [B, 3], got {role_logits.shape}")
    # category order is Normal=0, Parent=1, Daughter=2.  The difference is
    # log(P(Parent) / P(Daughter)); the Normal logit cancels exactly.
    return role_logits[:, 1].float() - role_logits[:, 2].float()


def factorized_event_ownership_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_weight: float,
    offset_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Separate Division-event evidence from conditional Parent ownership."""
    if "event" not in outputs or "ownership" not in outputs:
        raise ValueError("factorized outputs are missing event/ownership logits")
    event_target = (batch["category"] != 0).float()
    event = nn.functional.binary_cross_entropy_with_logits(
        outputs["event"].float(), event_target
    )
    ownership_mask = batch["category"] != 0
    ownership_target = (batch["category"][ownership_mask] == 1).float()
    ownership = nn.functional.binary_cross_entropy_with_logits(
        outputs["ownership"][ownership_mask].float(), ownership_target
    )
    legacy_total, values = arm_losses(
        "feedback", outputs, batch, count_weight, offset_weight
    )
    removed_parent_bce = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary"].float(), batch["binary"]
    )
    total = event + ownership + (legacy_total - removed_parent_bce)
    values.pop("binary", None)
    values.update(
        event_membership=float(event.detach().cpu()),
        conditional_ownership=float(ownership.detach().cpu()),
        legacy_parent_bce_removed=float(removed_parent_bce.detach().cpu()),
    )
    return total, values


def visible_event_auxiliary_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked complete-visible event BCE without changing the Parent score."""
    if "visible_event" not in outputs:
        raise ValueError("visible-event auxiliary output is missing")
    valid = batch["visible_event_valid"].float()
    target = batch["visible_event"].float()
    per_sample = nn.functional.binary_cross_entropy_with_logits(
        outputs["visible_event"].float(), target, reduction="none"
    )
    denominator = valid.sum().clamp_min(1.0)
    loss = (per_sample * valid).sum() / denominator
    return loss, {
        "loss": float(loss.detach().cpu()),
        "valid": float(valid.sum().detach().cpu()),
        "positive": float((target * valid).sum().detach().cpu()),
    }


def factorized_pair_losses(
    parent_outputs: dict[str, torch.Tensor],
    daughter_outputs: dict[str, torch.Tensor],
    reverse_offset: torch.Tensor,
    reverse_valid: torch.Tensor,
    reverse_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ones = torch.ones_like(parent_outputs["event"], dtype=torch.float32)
    event = 0.5 * (
        nn.functional.binary_cross_entropy_with_logits(parent_outputs["event"].float(), ones)
        + nn.functional.binary_cross_entropy_with_logits(daughter_outputs["event"].float(), ones)
    )
    ownership = 0.5 * (
        nn.functional.binary_cross_entropy_with_logits(parent_outputs["ownership"].float(), ones)
        + nn.functional.binary_cross_entropy_with_logits(
            daughter_outputs["ownership"].float(), torch.zeros_like(ones)
        )
    )
    per_slot = nn.functional.smooth_l1_loss(
        daughter_outputs["reverse_ancestry"].float(), reverse_offset.float(),
        reduction="none", beta=0.1,
    ).mean(2)
    denominator = reverse_valid.float().sum().clamp_min(1.0)
    reverse = (per_slot * reverse_valid.float()).sum() / denominator
    total = event + ownership + float(reverse_weight) * reverse
    return total, {
        "event": float(event.detach().cpu()),
        "ownership": float(ownership.detach().cpu()),
        "reverse": float(reverse.detach().cpu()),
        "reverse_mae_um": float(
            (
                torch.abs(daughter_outputs["reverse_ancestry"].float() - reverse_offset.float()).mean(2)
                * reverse_valid.float()
            ).sum().div(denominator).mul(10.0).detach().cpu()
        ),
    }


def initial_equivalence(
    control, model, loader, device, source_channel_indices: Sequence[int] = tuple(range(7))
) -> dict[str, float]:
    raw = next(iter(loader))
    image = raw["image"].to(device)
    control.eval()
    model.eval()
    with torch.inference_mode():
        # Match the temporal model's packed B*3 encoder batch. On CUDA, cuDNN
        # may select a different convolution kernel for B and B*3, producing
        # harmless ~1e-4 rounding deltas even when the copied weights are bit
        # identical and the zero-initialized residual makes the models
        # mathematically equivalent.
        batch, windows, channels, depth, height, width = image.shape
        source_image = image[:, :, tuple(source_channel_indices)]
        features = control.base.features(
            source_image.reshape(batch * windows, 7, depth, height, width)
        ).flatten(1)
        expected = control.base.heads(features.reshape(batch, windows, 128)[:, 1])
        actual = model(image)
    exact_names = ("count", "offset", "timing")
    diffs = {
        name: float(torch.max(torch.abs(expected[name].float() - actual[name].float())).cpu())
        for name in exact_names
    }
    if model.event_ownership_factorization:
        # The legacy Parent logit is reinterpreted as P(Parent | event) and
        # must remain exactly preserved.  The final Parent score is
        # P(event)*P(Parent|event); its zero-initialized event prior is 0.99,
        # so probability-space drift is intentionally bounded by 0.01 rather
        # than requiring logit equality.
        diffs["ownership"] = float(
            torch.max(torch.abs(expected["binary"].float() - actual["ownership"].float())).cpu()
        )
        diffs["gated_parent_probability"] = float(
            torch.max(
                torch.abs(
                    torch.sigmoid(expected["binary"].float())
                    - torch.sigmoid(actual["binary"].float())
                )
            ).cpu()
        )
        exact_max = max(value for key, value in diffs.items() if key != "gated_parent_probability")
        failed = exact_max > 1e-4 or diffs["gated_parent_probability"] > 0.01001
    else:
        diffs["binary"] = float(
            torch.max(torch.abs(expected["binary"].float() - actual["binary"].float())).cpu()
        )
        failed = max(diffs.values()) > 1e-4
    if failed:
        raise RuntimeError(f"warm-start equivalence failed: {diffs}")
    return diffs


def interval_bag_logit(window_binary: torch.Tensor) -> torch.Tensor:
    """Smooth set score: equal window logits preserve their original value."""
    if window_binary.ndim != 2 or window_binary.shape[1] != 3:
        raise ValueError(f"expected [B,3] window logits, got {tuple(window_binary.shape)}")
    return torch.logsumexp(window_binary.float(), dim=1) - math.log(window_binary.shape[1])


def uncertainty_aware_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_weight: float,
    offset_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Interval Parent/normal BCE plus unchanged successor auxiliary losses.

    Category codes are normal=0, parent=1, daughter=2. Daughter rows still train
    child count/offset heads, but do not impose a contradictory exact-frame binary
    target while the annotated boundary may be shifted by one frame.
    """
    category = batch["category"]
    supervised = category != 2
    if torch.any(supervised):
        target = (category[supervised] == 1).float()
        bag = interval_bag_logit(outputs["window_binary"])[supervised]
        interval = nn.functional.binary_cross_entropy_with_logits(bag, target)
    else:
        # Rare all-Daughter batches still train successor auxiliaries while the
        # zero term keeps the graph valid for the common optimizer path.
        interval = outputs["window_binary"].sum() * 0.0
    legacy_total, values = arm_losses(
        "feedback", outputs, batch, count_weight, offset_weight
    )
    exact = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary"].float(), batch["binary"]
    )
    total = interval + (legacy_total - exact)
    values["binary_interval"] = float(interval.detach().cpu())
    values["binary_exact_removed"] = float(exact.detach().cpu())
    values["binary_supervised_fraction"] = float(supervised.float().mean().detach().cpu())
    values.pop("binary", None)
    return total, values


def three_class_role_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_weight: float,
    offset_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace legacy binary BCE with explicit Normal/Parent/Daughter CE."""
    legacy_total, values = arm_losses(
        "feedback", outputs, batch, count_weight, offset_weight
    )
    binary = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary"].float(), batch["binary"]
    )
    role = nn.functional.cross_entropy(outputs["role"].float(), batch["category"])
    total = role + (legacy_total - binary)
    values.pop("binary", None)
    values["role_three_class"] = float(role.detach().cpu())
    values["binary_removed"] = float(binary.detach().cpu())
    return total, values


def _role_classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    names = ("normal", "parent", "daughter")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    f1_values = []
    recall = {}
    for code, name in enumerate(names):
        tp = int(confusion[code, code])
        fp = int(confusion[:, code].sum() - tp)
        fn = int(confusion[code, :].sum() - tp)
        f1_values.append(2.0 * tp / max(2 * tp + fp + fn, 1))
        recall[name] = tp / max(int(confusion[code, :].sum()), 1)
    return {
        "class_order": list(names),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class_f1": {name: float(value) for name, value in zip(names, f1_values)},
        "per_class_recall": recall,
    }


def evaluate_three_class(
    model, loader, samples, device, amp: str, offset_scale_um: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate softmax P(Parent), explicit role assignment, and aux heads."""
    model.eval()
    probabilities, indices, role_targets = [], [], []
    count_true, count_pred, count_valid = [], [], []
    offset_errors, offset_valid = [], []
    with torch.inference_mode():
        for raw in loader:
            cpu_indices = raw["index"].numpy()
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw.items()
                if key != "index"
            }
            with amp_context(amp):
                outputs = model(batch["image"])
            probabilities.append(torch.softmax(outputs["role"].float(), dim=1).cpu().numpy())
            indices.append(cpu_indices)
            role_targets.append(batch["category"].cpu().numpy())
            count_true.append(batch["count"].cpu().numpy())
            count_pred.append(outputs["count"].argmax(1).cpu().numpy())
            count_valid.append(batch["count_valid"].bool().cpu().numpy())
            error = torch.abs(outputs["offset"].float() - batch["offset"]).mean(2)
            offset_errors.append((error * offset_scale_um).cpu().numpy())
            offset_valid.append(batch["offset_valid"].bool().cpu().numpy())
    ordered = np.concatenate(indices)
    role_probabilities = np.concatenate(probabilities)
    targets = np.concatenate(role_targets)
    parent_probabilities = role_probabilities[:, 1]
    metrics = metrics_with_slices(samples, ordered, parent_probabilities)
    metrics["role_three_class"] = _role_classification_metrics(
        targets, role_probabilities.argmax(1)
    )
    metrics["role_probability_mean_by_true_class"] = {
        name: {
            predicted_name: float(role_probabilities[targets == code, predicted_code].mean())
            for predicted_code, predicted_name in enumerate(("normal", "parent", "daughter"))
        }
        for code, name in enumerate(("normal", "parent", "daughter"))
    }
    count_mask = np.concatenate(count_valid)
    offset_mask = np.concatenate(offset_valid)
    count_true_array = np.concatenate(count_true)
    count_pred_array = np.concatenate(count_pred)
    offset_array = np.concatenate(offset_errors)
    metrics["auxiliary"] = {
        "child_count_accuracy": (
            float(np.mean(count_true_array[count_mask] == count_pred_array[count_mask]))
            if np.any(count_mask) else None
        ),
        "child_count_valid_n": int(np.count_nonzero(count_mask)),
        "successor_offset_mae_um": (
            float(np.mean(offset_array[offset_mask])) if np.any(offset_mask) else None
        ),
        "successor_offset_valid_n": int(np.count_nonzero(offset_mask)),
    }
    return metrics, ordered, parent_probabilities, role_probabilities


def evaluate_factorized_heads(
    model, loader, samples, device, amp: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    event_scores: list[np.ndarray] = []
    ownership_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for raw in loader:
            cpu_indices = raw["index"].numpy()
            image = raw["image"].to(device, non_blocking=True)
            with amp_context(amp):
                outputs = model(image)
            indices.append(cpu_indices)
            event_scores.append(torch.sigmoid(outputs["event"].float()).cpu().numpy())
            ownership_scores.append(torch.sigmoid(outputs["ownership"].float()).cpu().numpy())
    ordered = np.concatenate(indices)
    event = np.concatenate(event_scores)
    ownership = np.concatenate(ownership_scores)
    selected = [samples[int(index)] for index in ordered]
    categories = np.asarray([sample.category for sample in selected])
    event_target = categories != "normal"
    ownership_mask = categories != "normal"
    ownership_target = categories[ownership_mask] == "parent"
    metrics = {
        "event_parent_or_daughter_vs_normal": binary_metrics(event_target, event),
        "ownership_parent_vs_daughter": binary_metrics(
            ownership_target, ownership[ownership_mask]
        ),
        "probability_mean_by_category": {
            category: {
                "event": float(event[categories == category].mean()),
                "ownership": float(ownership[categories == category].mean()),
            }
            for category in ("parent", "daughter", "normal")
        },
    }
    return metrics, ordered, event, ownership


def evaluate_visible_event_head(model, loader, samples, device, amp: str):
    model.eval()
    indices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for raw in loader:
            indices.append(raw["index"].numpy())
            targets.append(raw["visible_event"].numpy())
            valid.append(raw["visible_event_valid"].numpy())
            with amp_context(amp):
                outputs = model(raw["image"].to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(outputs["visible_event"].float()).cpu().numpy())
    ordered = np.concatenate(indices)
    target = np.concatenate(targets).astype(np.int64)
    mask = np.concatenate(valid).astype(bool)
    probability = np.concatenate(probabilities)
    if not np.any(mask) or len(np.unique(target[mask])) != 2:
        raise ValueError("visible-event validation requires both valid classes")
    categories = np.asarray([samples[int(index)].category for index in ordered])
    metrics = {
        "complete_visible_vs_not_visible": binary_metrics(target[mask], probability[mask]),
        "valid_n": int(np.count_nonzero(mask)),
        "masked_partial_n": int(np.count_nonzero(~mask)),
        "positive_n": int(np.count_nonzero(target[mask] == 1)),
        "by_category": {},
    }
    for category in ("parent", "daughter", "normal"):
        selected = mask & (categories == category)
        if np.count_nonzero(selected) and len(np.unique(target[selected])) == 2:
            metrics["by_category"][category] = binary_metrics(
                target[selected], probability[selected]
            )
        else:
            metrics["by_category"][category] = {
                "n": int(np.count_nonzero(selected)), "single_class": True
            }
    return metrics, ordered, target, mask, probability


def evaluate_interval(
    model, loader, samples, device, amp: str, offset_scale_um: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate event-bag scores, exact-centre diagnostics, and aux heads."""
    model.eval()
    bag_scores, central_scores, indices, offsets = [], [], [], []
    count_true, count_pred, count_valid = [], [], []
    offset_errors, offset_valid = [], []
    with torch.inference_mode():
        for raw in loader:
            cpu_indices = raw["index"].numpy()
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw.items()
                if key != "index"
            }
            with amp_context(amp):
                outputs = model(batch["image"])
            window = outputs["window_binary"].float()
            bag_scores.append(torch.sigmoid(interval_bag_logit(window)).cpu().numpy())
            central_scores.append(torch.sigmoid(outputs["binary"].float()).cpu().numpy())
            offsets.append((window.argmax(1) - 1).cpu().numpy())
            indices.append(cpu_indices)
            count_true.append(batch["count"].cpu().numpy())
            count_pred.append(outputs["count"].argmax(1).cpu().numpy())
            count_valid.append(batch["count_valid"].bool().cpu().numpy())
            error = torch.abs(outputs["offset"].float() - batch["offset"]).mean(2)
            offset_errors.append((error * offset_scale_um).cpu().numpy())
            offset_valid.append(batch["offset_valid"].bool().cpu().numpy())
    ordered = np.concatenate(indices)
    bag_probabilities = np.concatenate(bag_scores)
    central_probabilities = np.concatenate(central_scores)
    best_offsets = np.concatenate(offsets)
    metrics = metrics_with_slices(samples, ordered, bag_probabilities)
    metrics["central_exact_diagnostic"] = metrics_with_slices(
        samples, ordered, central_probabilities
    )
    categories = np.asarray([samples[int(index)].category for index in ordered])
    for category in ("parent", "daughter", "normal"):
        mask = categories == category
        metrics.setdefault("event_bag_score_by_category", {})[category] = {
            "n": int(mask.sum()),
            "mean": float(bag_probabilities[mask].mean()) if np.any(mask) else None,
            "argmax_offset_counts": {
                str(position): int(np.count_nonzero(best_offsets[mask] == position))
                for position in (-1, 0, 1)
            },
        }
    count_mask = np.concatenate(count_valid)
    offset_mask = np.concatenate(offset_valid)
    count_true_array = np.concatenate(count_true)
    count_pred_array = np.concatenate(count_pred)
    offset_array = np.concatenate(offset_errors)
    metrics["auxiliary"] = {
        "child_count_accuracy": (
            float(np.mean(count_true_array[count_mask] == count_pred_array[count_mask]))
            if np.any(count_mask) else None
        ),
        "child_count_valid_n": int(np.count_nonzero(count_mask)),
        "successor_offset_mae_um": (
            float(np.mean(offset_array[offset_mask])) if np.any(offset_mask) else None
        ),
        "successor_offset_valid_n": int(np.count_nonzero(offset_mask)),
    }
    return metrics, ordered, bag_probabilities, central_probabilities, best_offsets


def attention_audit(model, loader, samples, device, amp: str) -> dict[str, Any] | None:
    if model.mode != "mil":
        return None
    positive, negative = [], []
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            indices = raw["index"].numpy()
            image = raw["image"].to(device, non_blocking=True)
            with amp_context(amp):
                weights = model(image)["attention"].float().cpu().numpy()
            for index, row in zip(indices, weights):
                (positive if samples[int(index)].label else negative).append(row)
    return {
        "positions": [-1, 0, 1],
        "positive_mean": np.mean(positive, axis=0).tolist(),
        "negative_mean": np.mean(negative, axis=0).tolist(),
        "positive_n": len(positive),
        "negative_n": len(negative),
    }


def set_fine_tune_trainability(
    model: SlidingT3Residual,
    epoch: int,
    freeze_base_epochs: int,
    freeze_encoder_only: bool,
) -> dict[str, bool]:
    """Apply the explicit fine-tuning freeze policy and report its state."""
    encoder_frozen = epoch < freeze_base_epochs
    heads_frozen = encoder_frozen and not freeze_encoder_only
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not encoder_frozen)
    for parameter in model.heads.parameters():
        parameter.requires_grad_(not heads_frozen)
    return {"encoder_frozen": encoder_frozen, "heads_frozen": heads_frozen}


def checkpoint_selection_value(
    metrics: dict[str, Any], args: argparse.Namespace,
) -> tuple[float, str]:
    """Return the scalar maximized for checkpoint selection.

    Narrow experiment modules may replace this hook when their added head has a
    validation objective that is more appropriate than Parent AP.  The default
    preserves the historical Current-TH contract exactly.
    """
    if args.binary_supervision == "interval_mask_daughter":
        return float(metrics["parent_vs_normal"]["auprc"]), "parent_vs_normal_auprc"
    return float(metrics["parent_vs_all"]["auprc"]), "parent_vs_all_auprc"


def validation_curve_fields(
    metrics: dict[str, Any], args: argparse.Namespace,
) -> dict[str, float | int | None]:
    """Optional experiment-specific validation fields persisted each epoch."""
    return {}


def validation_epoch_artifacts(
    run_dir: Path,
    epoch: int,
    ordered: np.ndarray,
    probabilities: np.ndarray,
    samples: Sequence[Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Optional experiment hook for per-epoch validation diagnostics."""
    return None


def parse_ema_decays(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    decays = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(set(decays)) != len(decays):
        raise ValueError(f"duplicate EMA decay: {decays}")
    if any(not 0.0 < decay < 1.0 for decay in decays):
        raise ValueError(f"EMA decays must be in (0,1): {decays}")
    return decays


def ema_name(decay: float) -> str:
    return f"ema_{decay:.6f}".rstrip("0").rstrip(".").replace(".", "p")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_parameters = dict(model.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.mul_(self.decay).add_(source_parameters[name].detach(), alpha=1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(source_buffers[name].detach())


def _safe_binary_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    if not len(target) or len(np.unique(target)) != 2:
        return {"n": int(len(target)), "single_class": True, "auprc": None}
    result = binary_metrics(target.astype(np.int64), score.astype(np.float64))
    result["n"] = int(len(target))
    return result


def evaluate_loss_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    count_weight: float,
    offset_weight: float,
) -> dict[str, Any]:
    """Evaluate loss components, role slices, and pre/post-feedback logits."""
    model.eval()
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    pre_logits: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    post_logits: list[np.ndarray] = []
    binary_sum = 0.0
    binary_n = 0
    count_sum = 0.0
    count_n = 0.0
    offset_sum = 0.0
    offset_n = 0.0
    category_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    category_n = {0: 0, 1: 0, 2: 0}
    with torch.inference_mode():
        for raw in loader:
            cpu_indices = raw["index"].numpy()
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in raw.items()
                if key != "index"
            }
            with amp_context(amp):
                outputs = model(batch["image"])
            per_binary = nn.functional.binary_cross_entropy_with_logits(
                outputs["binary"].float(), batch["binary"], reduction="none"
            )
            per_count = nn.functional.cross_entropy(
                outputs["count"].float(), batch["count"], reduction="none"
            )
            per_offset = nn.functional.smooth_l1_loss(
                outputs["offset"].float(), batch["offset"], reduction="none", beta=0.1
            ).mean(2)
            binary_sum += float(per_binary.sum().cpu())
            binary_n += int(per_binary.numel())
            count_sum += float((per_count * batch["count_valid"]).sum().cpu())
            count_n += float(batch["count_valid"].sum().cpu())
            offset_sum += float((per_offset * batch["offset_valid"]).sum().cpu())
            offset_n += float(batch["offset_valid"].sum().cpu())
            for code in (0, 1, 2):
                selected = batch["category"] == code
                category_sum[code] += float(per_binary[selected].sum().cpu())
                category_n[code] += int(selected.sum().cpu())
            indices.append(cpu_indices)
            labels.append(batch["binary"].cpu().numpy())
            categories.append(batch["category"].cpu().numpy())
            before = outputs.get("binary_pre_feedback", outputs["binary"]).float()
            correction = outputs.get(
                "feedback_correction", torch.zeros_like(outputs["binary"])
            ).float()
            pre_logits.append(before.cpu().numpy())
            corrections.append(correction.cpu().numpy())
            post_logits.append(outputs["binary"].float().cpu().numpy())
    ordered = np.concatenate(indices)
    target = np.concatenate(labels).astype(np.int64)
    category = np.concatenate(categories).astype(np.int64)
    pre = np.concatenate(pre_logits)
    correction = np.concatenate(corrections)
    post = np.concatenate(post_logits)
    probability = 1.0 / (1.0 + np.exp(-np.clip(post, -60.0, 60.0)))
    count_loss = count_sum / max(count_n, 1.0)
    offset_loss = offset_sum / max(offset_n, 1.0)
    binary_loss = binary_sum / max(binary_n, 1)
    role_names = {0: "normal", 1: "parent", 2: "daughter"}
    metrics = {
        "loss": {
            "binary_bce": binary_loss,
            "count_ce": count_loss,
            "offset_smooth_l1": offset_loss,
            "weighted_total": binary_loss + count_weight * count_loss + offset_weight * offset_loss,
            "count_valid_n": count_n,
            "offset_valid_n": offset_n,
        },
        "bce_by_category": {
            role_names[code]: (
                category_sum[code] / category_n[code] if category_n[code] else None
            )
            for code in (0, 1, 2)
        },
        "category_n": {role_names[code]: category_n[code] for code in (0, 1, 2)},
        "parent_vs_all": _safe_binary_metrics(target, probability),
        "parent_vs_normal": _safe_binary_metrics(
            target[np.isin(category, (0, 1))], probability[np.isin(category, (0, 1))]
        ),
        "parent_vs_daughter": _safe_binary_metrics(
            target[np.isin(category, (1, 2))], probability[np.isin(category, (1, 2))]
        ),
        "ordered": ordered,
        "target": target,
        "category": category,
        "pre_feedback_logit": pre,
        "feedback_correction": correction,
        "post_feedback_logit": post,
        "probability": probability,
    }
    return metrics


def diagnostic_curve_fields(
    prefix: str, split: str, diagnostics: dict[str, Any]
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for name in ("binary_bce", "count_ce", "offset_smooth_l1", "weighted_total"):
        result[f"{prefix}_{split}_{name}"] = diagnostics["loss"][name]
    for category in ("parent", "daughter", "normal"):
        result[f"{prefix}_{split}_{category}_bce"] = diagnostics["bce_by_category"][category]
    for group in ("parent_vs_all", "parent_vs_normal", "parent_vs_daughter"):
        result[f"{prefix}_{split}_{group}_auprc"] = diagnostics[group].get("auprc")
    return result


def write_epoch_oof_diagnostics(
    run_dir: Path,
    epoch: int,
    diagnostics_by_model: dict[str, dict[str, Any]],
    samples: Sequence[Any],
) -> None:
    names = list(diagnostics_by_model)
    reference = diagnostics_by_model[names[0]]
    ordered = reference["ordered"]
    if np.any(ordered < 0):
        raise ValueError("validation OOF cannot contain synthetic Daughter indices")
    rows = []
    for position, index in enumerate(ordered):
        row: dict[str, Any] = {
            "epoch": epoch,
            "sample_id": samples[int(index)].sample_id,
            "label": samples[int(index)].label,
            "category": samples[int(index)].category,
        }
        for name in names:
            diagnostic = diagnostics_by_model[name]
            if not np.array_equal(ordered, diagnostic["ordered"]):
                raise RuntimeError(f"OOF order mismatch for {name}")
            row.update({
                f"{name}_pre_feedback_logit": float(diagnostic["pre_feedback_logit"][position]),
                f"{name}_feedback_correction": float(diagnostic["feedback_correction"][position]),
                f"{name}_post_feedback_logit": float(diagnostic["post_feedback_logit"][position]),
                f"{name}_probability": float(diagnostic["probability"][position]),
            })
        rows.append(row)
    destination = run_dir / "epoch_oof" / f"epoch_{epoch:03d}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(destination, rows)


def post_training_export(
    model: SlidingT3Residual,
    path: Path,
    samples: Sequence[Any],
    targets: Sequence[Any],
    stats: dict[str, Any],
    hard_ids: frozenset[str],
    args: argparse.Namespace,
    representation: str,
    device: torch.device,
) -> None:
    """Optional experiment-specific export after the selected checkpoint loads."""
    return None


def train_arm(name, control_state, control, path, samples, targets, train_indices, val_indices, stats, hard_ids, args):
    run_dir = args.output_dir / "arms" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (run_dir / "DONE").exists() and (run_dir / "summary.json").is_file():
        return json.loads((run_dir / "summary.json").read_text())
    seed_everything(args.seed)
    if name.startswith("sliding_t3_"):
        representation = "t3"
        mode = name.removeprefix("sliding_t3_")
        input_channel_count = 14 if args.extra_aligned_crops is not None else 7
        source_channel_indices = tuple(range(7))
    elif name == "sliding_t5_center":
        representation = "t5"
        mode = "center"
        input_channel_count = 13
        source_channel_indices = (1, 2, 3, 6, 7, 10, 11)
        if args.extra_aligned_crops is not None:
            raise ValueError("direct T5 does not support --extra-aligned-crops")
    else:
        raise ValueError(f"unsupported arm {name!r}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported arm {name!r}; expected sliding_t3_<mode> for {SUPPORTED_MODES}")
    if args.binary_supervision == "interval_mask_daughter" and mode not in {"center", "bigru"}:
        raise ValueError("interval_mask_daughter is intentionally limited to center/bigru arms")
    model = SlidingT3Residual(
        mode,
        input_channel_count,
        args.event_ownership_factorization,
        args.visible_event_aux_weight > 0,
    )
    model.load_control_state(control_state, source_channel_indices)
    model_warmstart = None
    if args.model_warmstart_checkpoint is not None:
        model_warmstart = torch.load(
            args.model_warmstart_checkpoint, map_location="cpu", weights_only=False
        )
        if model_warmstart.get("mode") not in (None, mode):
            raise ValueError(
                "model warm-start mode mismatch: "
                f"{model_warmstart.get('mode')!r} != {mode!r}"
            )
        incompatible = model.load_state_dict(model_warmstart["model"], strict=False)
        allowed_missing = {"role_residual.weight", "role_residual.bias"}
        if hasattr(model, "tplus2_offset_head"):
            allowed_missing.update(
                {"tplus2_offset_head.weight", "tplus2_offset_head.bias"}
            )
        if args.event_ownership_factorization:
            allowed_missing.update(
                {
                    "event_head.weight", "event_head.bias",
                    "reverse_ancestry_head.weight", "reverse_ancestry_head.bias",
                }
            )
        if args.visible_event_aux_weight > 0:
            allowed_missing.update({"event_head.weight", "event_head.bias"})
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "model warm-start state mismatch: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
    device = torch.device("cuda")
    model = model.to(device)
    train_loader, val_loader, sampler_metadata = make_loaders(
        path, samples, targets, train_indices, val_indices, stats, hard_ids, args, representation
    )
    diagnostic_loaders = (
        make_diagnostic_loaders(
            path, samples, targets, train_indices, val_indices, stats, hard_ids, args,
            representation,
        )
        if args.extended_epoch_diagnostics else None
    )
    if args.paired_ranking_aligned_t9 is not None and (representation != "t3" or mode != "bigru"):
        raise ValueError("paired ranking is intentionally limited to the current sliding-T3 BiGRU")
    pair_loader = make_paired_ranking_loader(args, stats)
    audit = (
        initial_equivalence(control, model, val_loader, device, source_channel_indices)
        if model_warmstart is None
        else {"skipped_after_full_model_warmstart": True}
    )
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.heads.parameters())
    base_parameters = encoder_parameters + head_parameters
    temporal_parameters = [parameter for name_, parameter in model.named_parameters()
                           if not name_.startswith("encoder.") and not name_.startswith("heads.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": temporal_parameters, "lr": args.learning_rate},
            {"params": base_parameters, "lr": args.learning_rate * args.base_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.005)
    ema_trackers = {
        ema_name(decay): ModelEMA(model, decay) for decay in parse_ema_decays(args.ema_decays)
    }
    if ema_trackers and not args.extended_epoch_diagnostics:
        raise ValueError("EMA tracking requires --extended-epoch-diagnostics")
    start_epoch, best_ap, best_epoch, stale, curve = 0, -math.inf, -1, 0, []
    ema_best = {
        name_: {"auprc": -math.inf, "epoch": -1} for name_ in ema_trackers
    }
    latest = run_dir / "checkpoint_latest.pt"
    if args.resume and latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ap, best_epoch, stale = checkpoint["best_ap"], checkpoint["best_epoch"], checkpoint["stale"]
        curve = checkpoint["curve"]
        stored_ema = checkpoint.get("ema_models", {})
        if set(stored_ema) != set(ema_trackers):
            raise ValueError(
                f"EMA resume contract mismatch: stored={sorted(stored_ema)} "
                f"requested={sorted(ema_trackers)}"
            )
        for name_, tracker in ema_trackers.items():
            tracker.model.load_state_dict(stored_ema[name_])
        ema_best = checkpoint.get("ema_best", ema_best)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        freeze_state = set_fine_tune_trainability(
            model, epoch, args.freeze_base_epochs, args.freeze_encoder_only
        )
        frozen = freeze_state["encoder_frozen"]
        model.train()
        losses = []
        pair_losses: list[float] = []
        pair_accuracies: list[float] = []
        pair_margin_satisfied: list[float] = []
        pair_logit_margins: list[float] = []
        factorized_pair_losses_epoch: list[float] = []
        reverse_mae_epoch: list[float] = []
        visible_event_losses_epoch: list[float] = []
        pair_iterator = iter(pair_loader) if pair_loader is not None else None
        for raw in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items() if key != "index"}
            paired = None
            if pair_iterator is not None:
                try:
                    paired = next(pair_iterator)
                except StopIteration:
                    pair_iterator = iter(pair_loader)
                    paired = next(pair_iterator)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.amp):
                outputs = model(batch["image"])
                if args.event_ownership_factorization:
                    loss, _ = factorized_event_ownership_losses(
                        outputs, batch, args.count_loss_weight, args.offset_loss_weight
                    )
                elif args.role_supervision == "three_class":
                    loss, _ = three_class_role_losses(
                        outputs, batch, args.count_loss_weight, args.offset_loss_weight
                    )
                elif args.binary_supervision == "interval_mask_daughter":
                    loss, _ = uncertainty_aware_losses(
                        outputs, batch, args.count_loss_weight, args.offset_loss_weight
                    )
                else:
                    loss, _ = arm_losses(
                        "feedback", outputs, batch, args.count_loss_weight, args.offset_loss_weight
                    )
                if args.visible_event_aux_weight > 0:
                    visible_loss, _ = visible_event_auxiliary_loss(outputs, batch)
                    loss = loss + args.visible_event_aux_weight * visible_loss
                if paired is not None:
                    parent_outputs = model(paired["parent_image"].to(device, non_blocking=True))
                    daughter_outputs = model(paired["daughter_image"].to(device, non_blocking=True))
                    if args.event_ownership_factorization:
                        factorized_pair, factorized_values = factorized_pair_losses(
                            parent_outputs,
                            daughter_outputs,
                            paired["reverse_offset"].to(device, non_blocking=True),
                            paired["reverse_valid"].to(device, non_blocking=True),
                            args.reverse_ancestry_weight,
                        )
                        loss = loss + args.factorized_pair_weight * factorized_pair
                    else:
                        if args.role_supervision == "three_class":
                            parent_ranking_logits = role_ownership_logit(parent_outputs["role"])
                            daughter_ranking_logits = role_ownership_logit(daughter_outputs["role"])
                        else:
                            parent_ranking_logits = parent_outputs["binary"]
                            daughter_ranking_logits = daughter_outputs["binary"]
                        ranking, ranking_values = paired_ranking_loss(
                            parent_ranking_logits,
                            daughter_ranking_logits,
                            args.paired_ranking_margin,
                        )
                        loss = loss + args.paired_ranking_weight * ranking
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                for tracker in ema_trackers.values():
                    tracker.update(model)
            elif not (mode == "center" and frozen):
                raise RuntimeError(
                    f"loss unexpectedly has no autograd graph: mode={mode} frozen={frozen}"
                )
            losses.append(float(loss.detach().cpu()))
            if args.visible_event_aux_weight > 0:
                visible_event_losses_epoch.append(float(visible_loss.detach().cpu()))
            if paired is not None:
                if args.event_ownership_factorization:
                    factorized_pair_losses_epoch.append(float(factorized_pair.detach().cpu()))
                    reverse_mae_epoch.append(factorized_values["reverse_mae_um"])
                else:
                    pair_losses.append(ranking_values["loss"])
                    pair_accuracies.append(ranking_values["accuracy"])
                    pair_margin_satisfied.append(ranking_values["margin_satisfied"])
                    pair_logit_margins.append(ranking_values["mean_logit_margin"])
        scheduler.step()
        if args.role_supervision == "three_class":
            metrics, validation_order, validation_probabilities, _ = evaluate_three_class(
                model, val_loader, samples, device, args.amp, args.offset_scale_um
            )
            ap = float(metrics["parent_vs_all"]["auprc"])
        elif args.binary_supervision == "interval_mask_daughter":
            metrics, validation_order, validation_probabilities, _, _ = evaluate_interval(
                model, val_loader, samples, device, args.amp, args.offset_scale_um
            )
            ap = float(metrics["parent_vs_normal"]["auprc"])
        else:
            metrics, validation_order, validation_probabilities = evaluate(
                model, val_loader, samples, device, args.amp, args.offset_scale_um
            )
            ap = float(metrics["parent_vs_all"]["auprc"])
        validation_epoch_artifacts(
            run_dir, epoch, validation_order, validation_probabilities,
            samples, metrics, args,
        )
        selection_value, selection_name = checkpoint_selection_value(metrics, args)
        curve_row = {"epoch": epoch, "base_frozen": frozen,
                      "encoder_frozen": freeze_state["encoder_frozen"],
                      "heads_frozen": freeze_state["heads_frozen"],
                      "train_loss": float(np.mean(losses)), "val_auprc": ap,
                      "checkpoint_selection_value": selection_value,
                      "checkpoint_selection_name": selection_name,
                      "val_parent_vs_normal_auprc": metrics["parent_vs_normal"]["auprc"],
                      "val_parent_vs_daughter_auprc": metrics["parent_vs_daughter"]["auprc"],
                      "paired_ranking_loss": float(np.mean(pair_losses)) if pair_losses else None,
                      "paired_ranking_accuracy": float(np.mean(pair_accuracies)) if pair_accuracies else None,
                      "paired_margin_satisfied": float(np.mean(pair_margin_satisfied)) if pair_margin_satisfied else None,
                      "paired_mean_logit_margin": float(np.mean(pair_logit_margins)) if pair_logit_margins else None,
                      "factorized_pair_loss": float(np.mean(factorized_pair_losses_epoch)) if factorized_pair_losses_epoch else None,
                      "reverse_ancestry_train_mae_um": float(np.mean(reverse_mae_epoch)) if reverse_mae_epoch else None,
                      "visible_event_aux_loss": float(np.mean(visible_event_losses_epoch)) if visible_event_losses_epoch else None}
        curve_row.update(validation_curve_fields(metrics, args))
        if diagnostic_loaders is not None:
            train_diagnostic_loader, val_diagnostic_loader = diagnostic_loaders
            diagnostics_by_model: dict[str, dict[str, dict[str, Any]]] = {}
            diagnostic_models = {"raw": model}
            diagnostic_models.update(
                {name_: tracker.model for name_, tracker in ema_trackers.items()}
            )
            for diagnostic_name, diagnostic_model in diagnostic_models.items():
                train_diagnostics = evaluate_loss_diagnostics(
                    diagnostic_model, train_diagnostic_loader, device, args.amp,
                    args.count_loss_weight, args.offset_loss_weight,
                )
                val_diagnostics = evaluate_loss_diagnostics(
                    diagnostic_model, val_diagnostic_loader, device, args.amp,
                    args.count_loss_weight, args.offset_loss_weight,
                )
                diagnostics_by_model[diagnostic_name] = {
                    "train": train_diagnostics, "val": val_diagnostics
                }
                curve_row.update(
                    diagnostic_curve_fields(diagnostic_name, "train", train_diagnostics)
                )
                curve_row.update(
                    diagnostic_curve_fields(diagnostic_name, "val", val_diagnostics)
                )
                auprc_key = f"{diagnostic_name}_val_parent_vs_all_auprc"
                history = [
                    float(row[auprc_key]) for row in curve[-2:]
                    if row.get(auprc_key) is not None
                ]
                current_auprc = float(curve_row[auprc_key])
                curve_row[f"{diagnostic_name}_val_parent_vs_all_auprc_ma3"] = float(
                    np.mean(history + [current_auprc])
                )
                if diagnostic_name != "raw" and current_auprc > ema_best[diagnostic_name]["auprc"] + 1e-8:
                    ema_best[diagnostic_name] = {"auprc": current_auprc, "epoch": epoch}
                    atomic_torch_save(
                        {
                            "model": diagnostic_model.state_dict(), "epoch": epoch,
                            "diagnostics": {
                                "train": {
                                    key: value for key, value in train_diagnostics.items()
                                    if key not in {
                                        "ordered", "target", "category", "pre_feedback_logit",
                                        "feedback_correction", "post_feedback_logit", "probability",
                                    }
                                },
                                "val": {
                                    key: value for key, value in val_diagnostics.items()
                                    if key not in {
                                        "ordered", "target", "category", "pre_feedback_logit",
                                        "feedback_correction", "post_feedback_logit", "probability",
                                    }
                                },
                            },
                        },
                        run_dir / f"checkpoint_best_{diagnostic_name}.pt",
                    )
            write_epoch_oof_diagnostics(
                run_dir, epoch,
                {name_: payload["val"] for name_, payload in diagnostics_by_model.items()},
                samples,
            )
        curve.append(curve_row)
        if selection_value > best_ap + 1e-8:
            best_ap, best_epoch, stale = selection_value, epoch, 0
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "epoch": epoch, "best_ap": best_ap, "best_epoch": best_epoch, "stale": stale, "curve": curve,
                           "ema_models": {name_: tracker.model.state_dict() for name_, tracker in ema_trackers.items()},
                           "ema_best": ema_best}, latest)
        print(
            f"[{name}] epoch={epoch:03d} frozen={int(frozen)} ap={ap:.5f} "
            f"selection={selection_name}:{selection_value:.5f} best={best_ap:.5f}@{best_epoch}",
            flush=True,
        )
        if epoch + 1 >= min(20, args.epochs) and stale >= args.patience:
            break
    best = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    post_training_export(
        model, path, samples, targets, stats, hard_ids, args, representation, device
    )
    role_probabilities = None
    if args.role_supervision == "three_class":
        metrics, ordered, probabilities, role_probabilities = evaluate_three_class(
            model, val_loader, samples, device, args.amp, args.offset_scale_um
        )
        central_probabilities = probabilities
        best_offsets = np.zeros_like(ordered)
    elif args.binary_supervision == "interval_mask_daughter":
        metrics, ordered, probabilities, central_probabilities, best_offsets = evaluate_interval(
            model, val_loader, samples, device, args.amp, args.offset_scale_um
        )
    else:
        metrics, ordered, probabilities = evaluate(
            model, val_loader, samples, device, args.amp, args.offset_scale_um
        )
        central_probabilities = probabilities
        best_offsets = np.zeros_like(ordered)
    event_probabilities = None
    ownership_probabilities = None
    visible_event_targets = None
    visible_event_valid = None
    visible_event_probabilities = None
    if args.event_ownership_factorization:
        factorized_metrics, factorized_order, event_probabilities, ownership_probabilities = (
            evaluate_factorized_heads(model, val_loader, samples, device, args.amp)
        )
        if not np.array_equal(ordered, factorized_order):
            raise RuntimeError("factorized evaluation order mismatch")
        metrics["factorized_heads"] = factorized_metrics
    if args.visible_event_aux_weight > 0:
        visible_metrics, visible_order, visible_event_targets, visible_event_valid, visible_event_probabilities = (
            evaluate_visible_event_head(model, val_loader, samples, device, args.amp)
        )
        if not np.array_equal(ordered, visible_order):
            raise RuntimeError("visible-event evaluation order mismatch")
        metrics["visible_event_auxiliary"] = visible_metrics
    attention = attention_audit(model, val_loader, samples, device, args.amp)
    write_csv(run_dir / "learning_curve.csv", curve)
    score_rows = []
    for row_number, (index, probability, central, offset) in enumerate(zip(
        ordered, probabilities, central_probabilities, best_offsets
    )):
        row = {
            "sample_id": samples[int(index)].sample_id,
            "label": samples[int(index)].label,
            "category": samples[int(index)].category,
            "prediction": float(probability),
            "event_bag_prediction": float(probability),
            "central_exact_prediction": float(central),
            "event_argmax_offset": int(offset),
        }
        if role_probabilities is not None:
            row.update(
                normal_probability=float(role_probabilities[row_number, 0]),
                parent_probability=float(role_probabilities[row_number, 1]),
                daughter_probability=float(role_probabilities[row_number, 2]),
                predicted_role=("normal", "parent", "daughter")[
                    int(role_probabilities[row_number].argmax())
                ],
            )
        if event_probabilities is not None and ownership_probabilities is not None:
            row.update(
                event_probability=float(event_probabilities[row_number]),
                ownership_probability=float(ownership_probabilities[row_number]),
                gated_parent_probability=float(probability),
            )
        if visible_event_probabilities is not None:
            row.update(
                visible_event_target=int(visible_event_targets[row_number]),
                visible_event_valid=int(visible_event_valid[row_number]),
                visible_event_probability=float(visible_event_probabilities[row_number]),
            )
        score_rows.append(row)
    write_csv(run_dir / "oof_sample_scores.csv", score_rows)
    summary = {"name": name, "mode": mode, "best_epoch": best_epoch, "epochs_completed": len(curve),
               "representation": representation,
               "initial_equivalence_max_abs": audit, "sampler": sampler_metadata, "normalization": stats,
               "model_warmstart_checkpoint": (
                   str(args.model_warmstart_checkpoint)
                   if args.model_warmstart_checkpoint is not None else None
               ),
               "augmentation_profile": args.augmentation_profile,
               "freeze_policy": (
                   "encoder_only" if args.freeze_encoder_only else "encoder_and_heads"
               ),
               "freeze_base_epochs": args.freeze_base_epochs,
               "sampling_profile": args.sampling_profile,
               "ema": {
                   "decays": list(parse_ema_decays(args.ema_decays)),
                   "update_unit": "optimizer_step",
                   "initialization": "matched Current-TH raw checkpoint",
                   "checkpoint_selection_primary": "raw Parent-vs-all AUPRC",
                   "best": ema_best,
               } if ema_trackers else None,
               "extended_epoch_diagnostics": bool(args.extended_epoch_diagnostics),
               "binary_supervision": args.binary_supervision,
               "role_supervision": args.role_supervision,
               "paired_ranking": (
                   {
                       "weight": args.paired_ranking_weight,
                       "margin_logit": args.paired_ranking_margin,
                       "pair_rows": len(pair_loader.dataset),
                       "pair_batch_size": args.paired_ranking_batch_size,
                       "score_space": (
                           "role_logit_parent_minus_daughter"
                           if args.role_supervision == "three_class"
                           else "legacy_binary_parent_logit"
                       ),
                       "pair_batches_per_main_epoch": len(train_loader),
                       "shared_augmentation": True,
                       "train_only": True,
                       "aligned_t9": str(args.paired_ranking_aligned_t9),
                       "heatmap": str(args.paired_ranking_heatmap),
                       "ledger": str(args.paired_ranking_ledger),
                   }
                   if pair_loader is not None and args.paired_ranking_weight > 0 else None
               ),
               "event_ownership_factorization": (
                   {
                       "event_target": "Parent or Daughter vs Normal",
                       "ownership_target": "Parent vs Daughter, Normal masked",
                       "inference_score": "P(event) * P(Parent | event)",
                       "factorized_pair_weight": args.factorized_pair_weight,
                       "reverse_ancestry_weight": args.reverse_ancestry_weight,
                       "reverse_ancestry_targets": (
                           str(args.reverse_ancestry_targets)
                           if args.reverse_ancestry_targets is not None else None
                       ),
                   }
                   if args.event_ownership_factorization else None
               ),
               "visible_event_auxiliary": (
                   {
                       "target": "complete visible Parent+two-Daughter tuple in aligned crop",
                       "partial": "masked from auxiliary loss",
                       "weight": args.visible_event_aux_weight,
                       "targets": str(args.visible_event_targets),
                       "inference_score": "unchanged legacy binary Parent logit",
                   }
                   if args.visible_event_aux_weight > 0 else None
               ),
               "checkpoint_selection_metric": (
                   "softmax_parent_probability_parent_vs_all_auprc"
                   if args.role_supervision == "three_class"
                   else (
                       "event_bag_parent_vs_normal_auprc"
                       if args.binary_supervision == "interval_mask_daughter"
                       else "exact_parent_vs_all_auprc"
                   )
               ),
               "input_channel_count": input_channel_count,
               "input_channel_mask": args.input_channel_mask,
               "extra_aligned_crops": str(args.extra_aligned_crops) if args.extra_aligned_crops else None,
               "metrics": metrics, "attention": attention, "runtime_seconds": time.perf_counter() - started,
               "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").touch()
    latest.unlink(missing_ok=True)
    return summary


def main() -> int:
    args = parse_args()
    ema_decays = parse_ema_decays(args.ema_decays)
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    if not arms:
        raise ValueError("at least one --arms entry is required")
    invalid = [
        arm for arm in arms
        if not (
            (arm.startswith("sliding_t3_") and arm.removeprefix("sliding_t3_") in SUPPORTED_MODES)
            or arm == "sliding_t5_center"
        )
    ]
    if invalid:
        raise ValueError(f"unsupported arms: {invalid}; modes={SUPPORTED_MODES}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"duplicate arms: {arms}")
    if args.role_supervision == "three_class" and args.binary_supervision != "exact":
        raise ValueError("three_class role supervision requires --binary-supervision exact")
    if args.paired_ranking_weight < 0:
        raise ValueError("--paired-ranking-weight must be nonnegative")
    if args.paired_ranking_margin < 0:
        raise ValueError("--paired-ranking-margin must be nonnegative")
    if args.paired_ranking_batch_size <= 0:
        raise ValueError("--paired-ranking-batch-size must be positive")
    if args.factorized_pair_weight < 0 or args.reverse_ancestry_weight < 0:
        raise ValueError("factorized pair/reverse weights must be nonnegative")
    if args.visible_event_aux_weight < 0:
        raise ValueError("visible-event auxiliary weight must be nonnegative")
    if args.visible_event_aux_weight > 0 and args.visible_event_targets is None:
        raise ValueError("positive visible-event auxiliary weight requires target CSV")
    if args.visible_event_targets is not None and args.visible_event_aux_weight <= 0:
        raise ValueError("visible-event targets require a positive auxiliary weight")
    if args.visible_event_aux_weight > 0 and args.event_ownership_factorization:
        raise ValueError("visible-event-only and event-ownership factorization are separate ablations")
    if args.event_ownership_factorization:
        if args.role_supervision != "binary" or args.binary_supervision != "exact":
            raise ValueError("event-ownership factorization requires binary/exact base contract")
        if args.paired_ranking_weight != 0:
            raise ValueError("factorization and paired ranking are separate ablations")
        if args.factorized_pair_weight <= 0:
            raise ValueError("factorization requires positive --factorized-pair-weight")
        if args.reverse_ancestry_weight > 0 and args.reverse_ancestry_targets is None:
            raise ValueError("positive reverse ancestry weight requires target CSV")
    elif args.reverse_ancestry_targets is not None or args.reverse_ancestry_weight != 0:
        raise ValueError("reverse ancestry is only valid with event-ownership factorization")
    if args.binary_supervision == "interval_mask_daughter":
        invalid_interval = [
            arm for arm in arms
            if not (
                (arm.startswith("sliding_t3_") and arm.removeprefix("sliding_t3_") in {"center", "bigru"})
                or arm == "sliding_t5_center"
            )
        ]
        if invalid_interval:
            raise ValueError(f"interval supervision only supports center/bigru: {invalid_interval}")
    if args.smoke:
        args.epochs = 2
        args.freeze_base_epochs = 1
        args.samples_per_epoch = 96
        args.num_workers = 0
        args.patience = 2
    if not 0.0 <= args.registration_jitter_probability <= 1.0:
        raise ValueError("--registration-jitter-probability must be in [0, 1]")
    if args.registration_jitter_magnitude_um < 0:
        raise ValueError("--registration-jitter-magnitude-um must be nonnegative")
    if args.registration_jitter_probability > 0 and args.registration_jitter_magnitude_um <= 0:
        raise ValueError("positive registration jitter probability requires positive magnitude")
    daughter_paths = (
        args.daughter_centred_aligned_t9,
        args.daughter_centred_heatmap,
        args.daughter_centred_ledger,
    )
    if args.sampling_profile == "parent_daughter_normal_2_1_1" and not all(daughter_paths):
        raise ValueError("2:1:1 sampling requires all Daughter-centred cache paths")
    if args.sampling_profile == "binary_balanced" and any(daughter_paths) and not all(daughter_paths):
        raise ValueError("partial Daughter-centred cache paths are not allowed")
    if ema_decays and not args.extended_epoch_diagnostics:
        raise ValueError("--ema-decays requires --extended-epoch-diagnostics")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("arms", "config", "logs"):
        (args.output_dir / name).mkdir(exist_ok=True)
    samples = read_samples(args.control_dir / "cache/samples.csv")
    hard_ids, hard_metadata = load_official_hard_negative_ids(args.official_hard_negative_pool)
    if args.official_hard_negative_fraction > 0 and not hard_ids:
        raise ValueError("--official-hard-negative-fraction requires --official-hard-negative-pool")
    targeted_profiles = {"strong_targeted", "temporal_imaging_targeted"}
    if args.augmentation_profile in targeted_profiles and not hard_ids:
        raise ValueError(f"{args.augmentation_profile} requires --official-hard-negative-pool")
    if args.augmentation_profile in targeted_profiles and args.extra_aligned_crops is not None:
        raise ValueError(
            f"{args.augmentation_profile} cannot currently be combined with --extra-aligned-crops"
        )
    aligned = np.load(args.aligned_t9, mmap_mode="r")
    if aligned.ndim != 5 or aligned.shape[0] != len(samples) or aligned.shape[1] != 9:
        raise ValueError(f"T9/sample mismatch: {aligned.shape}, {len(samples)}")
    if args.extra_aligned_crops is not None:
        extra = np.load(args.extra_aligned_crops, mmap_mode="r")
        expected = (len(samples), 3, *aligned.shape[-3:])
        if extra.shape != expected:
            raise ValueError(f"extra aligned crop mismatch: expected {expected}, got {extra.shape}")
    with args.flow_alignment.open(newline="") as handle:
        flow_rows = list(csv.DictReader(handle))
    if len(flow_rows) != len(samples):
        raise ValueError("flow alignment/sample mismatch")
    targets, target_summary = build_spatial_targets(samples, flow_rows, args.train_dir, aligned.shape[-3:],
                                                     args.max_node_match_distance_um)
    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    val_indices = [index for index, sample in enumerate(samples) if sample.split == "val"]
    if args.smoke:
        chosen = []
        for split_name, partition in (("train", train_indices), ("val", val_indices)):
            chosen.extend([index for index in partition if samples[index].label][:4])
            chosen.extend([index for index in partition if samples[index].category == "daughter"][:4])
            if split_name == "train":
                chosen.extend([
                    index for index in partition
                    if not samples[index].label
                    and canonical_sample_id(samples[index].sample_id) in hard_ids
                ][:4])
            chosen.extend([
                index for index in partition
                if not samples[index].label
                and canonical_sample_id(samples[index].sample_id) not in hard_ids
            ][:12])
        train_indices = [index for index in chosen if samples[index].split == "train"]
        val_indices = [index for index in chosen if samples[index].split == "val"]
        hard_ids = frozenset(
            canonical_sample_id(samples[index].sample_id)
            for index in train_indices
            if not samples[index].label
            and canonical_sample_id(samples[index].sample_id) in hard_ids
        )
    stats_by_representation = {
        "t3": estimate_channel_statistics(args.aligned_t9, train_indices, "concat7"),
        "t5": estimate_channel_statistics(args.aligned_t9, train_indices, "concat13"),
    }
    stats = stats_by_representation["t3"]
    checkpoint = torch.load(args.warmstart_checkpoint, map_location="cpu", weights_only=False)
    control_state = checkpoint["model"]
    control = CenterControl()
    control.base.load_state_dict(control_state, strict=True)
    device = torch.device("cuda")
    control = control.to(device)
    _, val_loader, _ = make_loaders(
        args.aligned_t9, samples, targets, train_indices, val_indices, stats, hard_ids, args
    )
    control_metrics, control_order, control_probabilities = evaluate(control, val_loader, samples, device, args.amp,
                                                                      args.offset_scale_um)
    control_dir = args.output_dir / "arms/center_checkpoint"
    control_dir.mkdir(parents=True, exist_ok=True)
    write_csv(control_dir / "oof_sample_scores.csv", [{"sample_id": samples[int(index)].sample_id,
                                                        "label": samples[int(index)].label,
                                                        "category": samples[int(index)].category,
                                                        "prediction": float(probability)}
                                                       for index, probability in zip(control_order, control_probabilities)])
    control_summary = {"name": "center_checkpoint", "metrics": control_metrics,
                       "warmstart_epoch": checkpoint.get("epoch"), "normalization": stats}
    atomic_json(control_dir / "summary.json", control_summary)
    (control_dir / "DONE").touch()
    results = [
        train_arm(
            name, control_state, control, args.aligned_t9, samples, targets, train_indices,
            val_indices, stats_by_representation["t5" if name == "sliding_t5_center" else "t3"],
            hard_ids, args,
        )
        for name in arms
    ]
    control_ap = float(control_metrics["parent_vs_all"]["auprc"])
    if args.binary_supervision == "interval_mask_daughter":
        comparison_metric = "parent_vs_normal"
        comparison_reference = results[0]
        reference_ap = float(comparison_reference["metrics"][comparison_metric]["auprc"])
    else:
        comparison_metric = "parent_vs_all"
        comparison_reference = control_summary
        reference_ap = control_ap
    summary = {
        "contract": {"fold": args.fold, "seed": args.seed, "windows": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                     "arms": list(arms),
                     "sampling_profile": args.sampling_profile,
                     "daughter_centred_sampling": (
                         {
                             "aligned_t9": str(args.daughter_centred_aligned_t9),
                             "heatmap": str(args.daughter_centred_heatmap),
                             "ledger": str(args.daughter_centred_ledger),
                             "mass": {"parent": 0.50, "daughter": 0.25, "normal": 0.25},
                             "synthetic_daughter_auxiliary": "masked",
                         }
                         if args.sampling_profile == "parent_daughter_normal_2_1_1" else None
                     ),
                     "epoch_diagnostics": (
                         {
                             "ema_decays": list(ema_decays),
                             "loss_components": ["binary_bce", "count_ce", "offset_smooth_l1"],
                             "category_bce": ["parent", "daughter", "normal"],
                             "feedback_logits": ["pre", "correction", "post"],
                             "epoch_oof": True,
                             "parent_vs_all_auprc_trailing_window": 3,
                         }
                         if args.extended_epoch_diagnostics else None
                     ),
                     "binary_supervision": args.binary_supervision,
                     "role_supervision": args.role_supervision,
                     "role_contract": (
                         {
                             "class_order": ["normal", "parent", "daughter"],
                             "training_loss": "unweighted three-class cross-entropy replacing binary BCE",
                             "inference_score": "softmax P(Parent)",
                             "epoch_zero_equivalence": (
                                 "Normal and Daughter base logits are -log(2), Parent base logit is the "
                                 "historical binary logit, and the role residual is zero-initialized"
                             ),
                             "sampling": "unchanged historical binary-balanced hard-negative-10-percent mixture",
                         }
                         if args.role_supervision == "three_class" else None
                     ),
                     "paired_ranking_contract": (
                         {
                             "loss": "softplus(margin - (Parent logit - Daughter logit))",
                             "weight": args.paired_ranking_weight,
                             "margin_logit": args.paired_ranking_margin,
                             "pair_batch_size": args.paired_ranking_batch_size,
                             "pointwise_sampler_unchanged": True,
                             "pair_source": "train-only exact GEFF parent-to-daughter edges",
                             "shared_pair_augmentation": True,
                         }
                         if args.paired_ranking_weight > 0 else None
                     ),
                     "visible_event_auxiliary_contract": (
                         {
                             "targets": str(args.visible_event_targets),
                             "loss": "masked BCE complete-visible vs not-visible",
                             "weight": args.visible_event_aux_weight,
                             "partial": "ignored",
                             "legacy_parent_score_unchanged": True,
                         }
                         if args.visible_event_aux_weight > 0 else None
                     ),
                     "interval_contract": (
                         {
                             "parent": "positive bag over offsets -1/0/+1",
                             "normal": "negative bag over offsets -1/0/+1",
                             "daughter": "binary loss masked; auxiliary successor targets retained",
                             "selection_metric": "event-bag Parent-vs-normal AUPRC",
                         }
                         if args.binary_supervision == "interval_mask_daughter" else None
                     ),
                     "successor_feedback": True, "warmstart_checkpoint": str(args.warmstart_checkpoint),
                     "augmentation_profile": args.augmentation_profile,
                     "input_channel_mask": args.input_channel_mask,
                     "registration_jitter": {
                         "probability": args.registration_jitter_probability,
                         "maximum_um": args.registration_jitter_magnitude_um,
                         "family": args.registration_jitter_family,
                         "magnitude_distribution": args.registration_jitter_magnitude_distribution,
                         "application": "aligned_output_space_matching_inference_shift9",
                         "raw_and_heatmap_shifted_together": bool(args.extra_aligned_crops),
                     },
                     "representations": {
                         "sliding_t3_center": "three shared T3 encodes centred -1/0/+1; each encode has 7 channels",
                         "sliding_t5_center": "three shared direct T5 encodes centred -1/0/+1; each encode has 13 channels",
                     },
                     "extra_aligned_crops": str(args.extra_aligned_crops) if args.extra_aligned_crops else None,
                     "heatmap_fusion": (
                         "same aligned T3 seven-channel heatmap prior appended to each raw sliding window; "
                         "raw channels z-scored, heatmap channels kept at native finite scale"
                         if args.extra_aligned_crops else None
                     ),
                     "hard_negative_fraction_within_negative_draws": args.official_hard_negative_fraction,
                     "hard_negative_pool": hard_metadata},
        "control": control_summary,
        "arms": results,
        "comparison": {
            "metric": (
                f"softmax_parent_probability_{comparison_metric}_auprc"
                if args.role_supervision == "three_class"
                else (
                    f"event_bag_{comparison_metric}_auprc"
                    if args.binary_supervision != "exact"
                    else f"exact_{comparison_metric}_auprc"
                )
            ),
            "reference": comparison_reference["name"],
        },
        "delta_vs_control": {
            result["name"]: float(result["metrics"][comparison_metric]["auprc"]) - reference_ap
            for result in results
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "config/run.json", {key: str(value) if isinstance(value, Path) else value
                                                       for key, value in vars(args).items()})
    atomic_json(args.output_dir / "config/targets.json", target_summary)
    (args.output_dir / "DONE").touch()
    print(json.dumps(summary["delta_vs_control"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
