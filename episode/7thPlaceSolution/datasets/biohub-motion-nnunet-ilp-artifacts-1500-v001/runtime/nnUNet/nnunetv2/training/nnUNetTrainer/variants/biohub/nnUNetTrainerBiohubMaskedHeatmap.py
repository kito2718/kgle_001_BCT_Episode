"""nnU-Net v2 trainers for sparse Biohub cell-centre heatmaps."""

from __future__ import annotations

import csv
import math
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

try:
    from torch import autocast
except ImportError:  # pragma: no cover - old torch fallback
    from torch.cuda.amp import autocast


class MaskedGaussianHeatmapLoss(nn.Module):
    """Build a Gaussian from centre labels and score only non-ignore voxels."""

    def __init__(
        self,
        spacing_zyx_um: Sequence[float],
        sigma_um: float = 2.0,
        truncate_sigma: float = 3.0,
        mode: str = "mse",
        positive_weight: float = 16.0,
        topk_fraction: float = 0.2,
        smoothing: float = 0.01,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        ignore_label: int = 2,
        target_stride_zyx: Sequence[int] = (1, 1, 1),
        valid_radius_um: float | None = None,
        target_composition: str = "sum_clip",
        center_weight: float = 1.0,
        center_radius_um: float = 1.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.spacing = tuple(float(v) for v in spacing_zyx_um)
        self.sigma_um = float(sigma_um)
        self.truncate_sigma = float(truncate_sigma)
        self.mode = mode
        self.positive_weight = float(positive_weight)
        self.topk_fraction = float(topk_fraction)
        if not 0 < self.topk_fraction <= 1:
            raise ValueError(f"topk_fraction must be in (0, 1], got {self.topk_fraction}")
        self.smoothing = float(smoothing)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.ignore_label = int(ignore_label)
        self.target_stride = tuple(int(v) for v in target_stride_zyx)
        if len(self.target_stride) != 3 or any(v < 1 for v in self.target_stride):
            raise ValueError(f"target_stride_zyx must contain three positive integers: {self.target_stride}")
        self.valid_radius_um = None if valid_radius_um is None else float(valid_radius_um)
        if self.target_stride != (1, 1, 1) and self.valid_radius_um is None:
            raise ValueError("valid_radius_um is required for a lower-resolution target")
        self.target_composition = str(target_composition)
        if self.target_composition not in {"sum_clip", "max"}:
            raise ValueError(
                "target_composition must be 'sum_clip' or 'max', "
                f"got {self.target_composition!r}"
            )
        self.center_weight = float(center_weight)
        self.center_radius_um = float(center_radius_um)
        if self.center_weight < 1:
            raise ValueError(f"center_weight must be >= 1, got {self.center_weight}")
        if self.center_radius_um < 0:
            raise ValueError(
                f"center_radius_um must be >= 0, got {self.center_radius_um}"
            )
        self.eps = float(eps)
        self._kernel_cache: dict[tuple, torch.Tensor] = {}
        self._log_axis_kernel_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}
        self._valid_kernel_cache: dict[tuple, torch.Tensor] = {}
        self._center_kernel_cache: dict[tuple, torch.Tensor] = {}

    def _kernel(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, self.spacing, self.sigma_um, self.truncate_sigma)
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        axes = []
        for spacing in self.spacing:
            radius = max(1, int(math.ceil(self.truncate_sigma * self.sigma_um / spacing)))
            axes.append(torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * spacing)
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        kernel = torch.exp(-(zz.square() + yy.square() + xx.square()) / (2 * self.sigma_um**2))
        kernel = (kernel / kernel.max()).unsqueeze(0).unsqueeze(0)
        self._kernel_cache[key] = kernel
        return kernel

    def _log_axis_kernels(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Return separable log-Gaussian kernels for exact max composition."""
        key = (device.type, device.index, self.spacing, self.sigma_um, self.truncate_sigma)
        cached = self._log_axis_kernel_cache.get(key)
        if cached is not None:
            return cached
        axes = []
        for spacing in self.spacing:
            radius = max(1, int(math.ceil(self.truncate_sigma * self.sigma_um / spacing)))
            coordinates = torch.arange(
                -radius, radius + 1, device=device, dtype=torch.float32
            ) * spacing
            axes.append(-coordinates.square() / (2 * self.sigma_um**2))
        result = tuple(axes)
        self._log_axis_kernel_cache[key] = result
        return result

    @staticmethod
    def _max_plus_axis(
        values: torch.Tensor,
        log_weights: torch.Tensor,
        spatial_axis: int,
    ) -> torch.Tensor:
        """One-dimensional max-plus convolution on Z, Y, or X."""
        dimension = spatial_axis + 2
        size = int(values.shape[dimension])
        radius = int(log_weights.numel() // 2)
        padding = [0, 0, 0, 0, 0, 0]
        padding_offset = 2 * (2 - spatial_axis)
        padding[padding_offset] = radius
        padding[padding_offset + 1] = radius
        padded = F.pad(values, tuple(padding), mode="constant", value=-float("inf"))
        output = torch.full_like(values, -float("inf"))
        for offset, weight in enumerate(log_weights):
            shifted = padded.narrow(dimension, offset, size)
            output = torch.maximum(output, shifted + weight)
        return output

    def _max_composed_heatmap(self, centres: torch.Tensor) -> torch.Tensor:
        """Render the exact voxel-wise max of all truncated instance Gaussians."""
        log_heatmap = torch.where(
            centres.gt(0),
            torch.zeros_like(centres),
            torch.full_like(centres, -float("inf")),
        )
        for spatial_axis, log_weights in enumerate(self._log_axis_kernels(centres.device)):
            log_heatmap = self._max_plus_axis(log_heatmap, log_weights, spatial_axis)
        return torch.exp(log_heatmap)

    def _valid_kernel(self, device: torch.device) -> torch.Tensor:
        if self.valid_radius_um is None:
            raise RuntimeError("valid_radius_um is not configured")
        key = (device.type, device.index, self.spacing, self.valid_radius_um)
        cached = self._valid_kernel_cache.get(key)
        if cached is not None:
            return cached
        axes = []
        for spacing in self.spacing:
            radius = max(1, int(math.ceil(self.valid_radius_um / spacing)))
            axes.append(torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * spacing)
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        kernel = (
            zz.square() + yy.square() + xx.square() <= self.valid_radius_um**2
        ).float().unsqueeze(0).unsqueeze(0)
        self._valid_kernel_cache[key] = kernel
        return kernel

    def _center_kernel(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, self.spacing, self.center_radius_um)
        cached = self._center_kernel_cache.get(key)
        if cached is not None:
            return cached
        if self.center_radius_um == 0:
            kernel = torch.ones((1, 1, 1, 1, 1), device=device, dtype=torch.float32)
        else:
            axes = []
            for spacing in self.spacing:
                radius = max(1, int(math.ceil(self.center_radius_um / spacing)))
                axes.append(
                    torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
                    * spacing
                )
            zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
            kernel = (
                zz.square() + yy.square() + xx.square() <= self.center_radius_um**2
            ).float().unsqueeze(0).unsqueeze(0)
        self._center_kernel_cache[key] = kernel
        return kernel

    def _prepare_centres_and_valid(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim != 5 or target.shape[1] != 1:
            raise ValueError(f"expected target [B,1,Z,Y,X], got {tuple(target.shape)}")
        labels = target[:, :1]
        centres = labels.eq(1).float()
        if self.target_stride == (1, 1, 1):
            valid = labels.ne(self.ignore_label)
        else:
            # Quantize transformed centre points directly onto the output grid.
            # This is not a resize of the discrete supervision mask. Gaussian
            # and valid support are rendered again in physical units below.
            centres = F.max_pool3d(
                centres,
                kernel_size=self.target_stride,
                stride=self.target_stride,
            )
            valid_kernel = self._valid_kernel(target.device)
            valid_padding = tuple(int(size // 2) for size in valid_kernel.shape[2:])
            valid = F.conv3d(centres, valid_kernel, padding=valid_padding).gt(0)
        return centres, valid

    def _build_target_components(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centres, valid = self._prepare_centres_and_valid(target)
        if self.target_composition == "max":
            heatmap = self._max_composed_heatmap(centres)
        else:
            kernel = self._kernel(target.device)
            padding = tuple(int(size // 2) for size in kernel.shape[2:])
            heatmap = F.conv3d(centres, kernel, padding=padding).clamp_(0, 1)
        heatmap = heatmap * valid.float()
        return heatmap, valid, centres

    def build_target(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heatmap, valid, _ = self._build_target_components(target)
        return heatmap, valid

    def build_center_weights(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centres, valid = self._prepare_centres_and_valid(target)
        if self.center_weight == 1:
            return torch.ones_like(centres), valid
        kernel = self._center_kernel(target.device)
        padding = tuple(int(size // 2) for size in kernel.shape[2:])
        center_region = F.conv3d(centres, kernel, padding=padding).gt(0) & valid
        weights = 1 + (self.center_weight - 1) * center_region.float()
        return weights, valid

    def _masked_mean(self, values: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None = None):
        valid_float = valid.float()
        if weights is not None:
            valid_float = valid_float * weights
        return (values * valid_float).sum() / valid_float.sum().clamp_min(1.0)

    def _masked_mean_per_sample(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_float = valid.float()
        if weights is not None:
            valid_float = valid_float * weights
        dims = tuple(range(1, values.ndim))
        return (values * valid_float).sum(dims) / valid_float.sum(dims).clamp_min(1.0)

    def sample_statistics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample loss, quantized centre count, and mean centre score."""
        if logits.shape[1] != 2:
            raise ValueError(f"Biohub trainer expects 2 logits, got {logits.shape[1]}")
        heatmap, valid, centres = self._build_target_components(target)
        foreground_logit = (logits[:, 1:2] - logits[:, 0:1]).float()
        probability = torch.sigmoid(foreground_logit)
        squared_error = (probability - heatmap).square()
        if self.mode == "mse":
            sample_loss = self._masked_mean_per_sample(squared_error, valid)
        elif self.mode == "weighted_mse":
            weights = 1 + (self.positive_weight - 1) * heatmap
            sample_loss = self._masked_mean_per_sample(squared_error, valid, weights)
        elif self.mode == "center_weighted_mse":
            center_weights, _ = self.build_center_weights(target)
            sample_loss = self._masked_mean_per_sample(
                squared_error, valid, center_weights
            )
        else:
            raise RuntimeError(
                f"sample_statistics is implemented only for MSE-family losses, got {self.mode}"
            )
        dims = tuple(range(1, centres.ndim))
        centre_count = centres.sum(dims)
        mean_center_probability = (probability * centres).sum(dims) / centre_count.clamp_min(1)
        return sample_loss, centre_count, mean_center_probability

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 2:
            raise ValueError(f"Biohub trainer expects 2 logits, got {logits.shape[1]}")
        heatmap, valid = self.build_target(target)
        foreground_logit = (logits[:, 1:2] - logits[:, 0:1]).float()
        probability = torch.sigmoid(foreground_logit)

        if self.mode == "mse":
            return self._masked_mean((probability - heatmap).square(), valid)
        if self.mode == "weighted_mse":
            weights = 1 + (self.positive_weight - 1) * heatmap
            return self._masked_mean((probability - heatmap).square(), valid, weights)
        if self.mode == "center_weighted_mse":
            center_weights, _ = self.build_center_weights(target)
            return self._masked_mean(
                (probability - heatmap).square(), valid, center_weights
            )

        bce_target = heatmap
        if self.mode == "label_smoothing_bce":
            bce_target = heatmap * (1 - self.smoothing) + 0.5 * self.smoothing
        if self.mode == "smooth_bce":
            # Match BYU 1st place's SmoothBCE target transform:
            # 0 -> smoothing and 1 -> 1 - smoothing.
            bce_target = heatmap * (1 - self.smoothing) + (1 - heatmap) * self.smoothing
        bce = F.binary_cross_entropy_with_logits(foreground_logit, bce_target, reduction="none")
        if self.mode == "topk_bce":
            valid_bce = bce.masked_select(valid)
            if valid_bce.numel() == 0:
                return bce.sum() * 0
            k = max(1, int(valid_bce.numel() * self.topk_fraction))
            return torch.topk(valid_bce, k=k, sorted=False).values.mean()
        if self.mode == "weighted_bce":
            weights = 1 + (self.positive_weight - 1) * heatmap
            return self._masked_mean(bce, valid, weights)
        if self.mode == "label_smoothing_bce":
            return self._masked_mean(bce, valid)
        if self.mode == "smooth_bce":
            return self._masked_mean(bce, valid)

        masked_probability = probability * valid.float()
        masked_target = heatmap * valid.float()
        dims = tuple(range(2, probability.ndim))
        intersection = (masked_probability * masked_target).sum(dims)
        if self.mode == "bce_dice":
            dice = (2 * intersection + self.eps) / (
                masked_probability.sum(dims) + masked_target.sum(dims) + self.eps
            )
            return self._masked_mean(bce, valid) + (1 - dice).mean()
        if self.mode == "bce_tversky":
            fp = (masked_probability * (1 - masked_target)).sum(dims)
            fn = ((1 - masked_probability) * masked_target).sum(dims)
            tversky = (intersection + self.eps) / (
                intersection + self.tversky_alpha * fp + self.tversky_beta * fn + self.eps
            )
            return self._masked_mean(bce, valid) + (1 - tversky).mean()
        raise ValueError(f"unknown loss mode: {self.mode}")


class RepelCodingMaskedHeatmapLoss(MaskedGaussianHeatmapLoss):
    """Paper-formula Repel Coding rendered in 3D physical coordinates.

    Liang et al. (2019) define D' = d1 * (1 + d1 / d2)^2 and
    C = 1 / (1 + alpha * D') for D' < r, otherwise zero. Here d1 and d2
    are measured in physical micrometres on the output grid. The sparse
    valid mask and MSE reduction remain identical to the Biohub control.
    """

    def __init__(
        self,
        *args,
        repel_alpha_per_um: float,
        repel_support_um: float,
        distance_chunk_size: int = 8192,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.repel_alpha_per_um = float(repel_alpha_per_um)
        self.repel_support_um = float(repel_support_um)
        self.distance_chunk_size = int(distance_chunk_size)
        if self.mode != "mse":
            raise ValueError("Repel Coding target currently requires mode='mse'")
        if self.repel_alpha_per_um <= 0:
            raise ValueError(
                "repel_alpha_per_um must be positive, "
                f"got {self.repel_alpha_per_um}"
            )
        if self.repel_support_um <= 0:
            raise ValueError(
                f"repel_support_um must be positive, got {self.repel_support_um}"
            )
        if self.distance_chunk_size < 1:
            raise ValueError(
                f"distance_chunk_size must be positive, got {self.distance_chunk_size}"
            )

    def _repel_heatmap(
        self,
        centres: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        heatmap = torch.zeros_like(centres)
        spacing = torch.as_tensor(
            self.spacing,
            device=centres.device,
            dtype=torch.float32,
        )
        for batch_index in range(int(centres.shape[0])):
            centre_indices = torch.nonzero(
                centres[batch_index, 0].gt(0),
                as_tuple=False,
            )
            valid_mask = valid[batch_index, 0]
            valid_indices = torch.nonzero(valid_mask, as_tuple=False)
            if centre_indices.numel() == 0 or valid_indices.numel() == 0:
                continue
            centre_coordinates = centre_indices.float() * spacing
            valid_coordinates = valid_indices.float() * spacing
            encoded_chunks = []
            for start in range(0, int(valid_coordinates.shape[0]), self.distance_chunk_size):
                query = valid_coordinates[start : start + self.distance_chunk_size]
                # cdist is target construction, not a learned operation. Keep it
                # in float32 even when the surrounding train step uses AMP.
                with autocast(centres.device.type, enabled=False):
                    distances = torch.cdist(query.float(), centre_coordinates.float())
                    if int(centre_coordinates.shape[0]) == 1:
                        repel_distance = distances[:, 0]
                    else:
                        nearest_two = torch.topk(
                            distances,
                            k=2,
                            dim=1,
                            largest=False,
                            sorted=True,
                        ).values
                        d1 = nearest_two[:, 0]
                        d2 = nearest_two[:, 1]
                        repel_distance = d1 * (
                            1 + d1 / d2.clamp_min(self.eps)
                        ).square()
                    encoded = torch.where(
                        repel_distance < self.repel_support_um,
                        1 / (1 + self.repel_alpha_per_um * repel_distance),
                        torch.zeros_like(repel_distance),
                    )
                encoded_chunks.append(encoded)
            encoded_values = torch.cat(encoded_chunks)
            heatmap[
                batch_index,
                0,
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ] = encoded_values
        return heatmap

    def _build_target_components(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centres, valid = self._prepare_centres_and_valid(target)
        heatmap = self._repel_heatmap(centres, valid)
        return heatmap, valid, centres


class ClosePairValleyMaskedGaussianHeatmapLoss(MaskedGaussianHeatmapLoss):
    """Add a sparse-label-safe ranking constraint for annotated close pairs.

    This is a cheap persistence-inspired surrogate, not a full persistent
    homology loss. For nearby annotated centres, it requires the weaker centre
    score to exceed the trilinearly sampled midpoint by a margin. It does not
    constrain unannotated regions or the global number of modes.
    """

    def __init__(
        self,
        *args,
        pair_weight: float = 0.1,
        pair_margin: float = 0.15,
        pair_min_distance_um: float = 2.5,
        pair_max_distance_um: float = 6.0,
        third_center_exclusion_um: float = 2.0,
        max_pairs_per_sample: int = 256,
        pair_peak_weight: float = 0.0,
        pair_peak_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pair_weight = float(pair_weight)
        self.pair_margin = float(pair_margin)
        self.pair_min_distance_um = float(pair_min_distance_um)
        self.pair_max_distance_um = float(pair_max_distance_um)
        self.third_center_exclusion_um = float(third_center_exclusion_um)
        self.max_pairs_per_sample = int(max_pairs_per_sample)
        self.pair_peak_weight = float(pair_peak_weight)
        self.pair_peak_gamma = float(pair_peak_gamma)
        if self.mode != "mse":
            raise ValueError("close-pair valley loss currently requires mode='mse'")
        if self.pair_weight < 0:
            raise ValueError(f"pair_weight must be non-negative, got {self.pair_weight}")
        if self.pair_margin < 0:
            raise ValueError(f"pair_margin must be non-negative, got {self.pair_margin}")
        if not 0 <= self.pair_min_distance_um < self.pair_max_distance_um:
            raise ValueError(
                "pair distances must satisfy 0 <= min < max, got "
                f"{self.pair_min_distance_um}, {self.pair_max_distance_um}"
            )
        if self.third_center_exclusion_um < 0:
            raise ValueError(
                "third_center_exclusion_um must be non-negative, got "
                f"{self.third_center_exclusion_um}"
            )
        if self.max_pairs_per_sample < 1:
            raise ValueError(
                f"max_pairs_per_sample must be positive, got {self.max_pairs_per_sample}"
            )
        if self.pair_peak_weight < 0:
            raise ValueError(
                f"pair_peak_weight must be non-negative, got {self.pair_peak_weight}"
            )
        if self.pair_peak_gamma < 0:
            raise ValueError(
                f"pair_peak_gamma must be non-negative, got {self.pair_peak_gamma}"
            )
        self.last_base_loss = 0.0
        self.last_pair_loss = 0.0
        self.last_pair_peak_loss = 0.0
        self.last_active_pairs = 0

    @staticmethod
    def _sample_zyx(
        volume: torch.Tensor,
        points_zyx: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Differentiably sample [1,1,Z,Y,X] at voxel-space Z/Y/X points."""
        if volume.shape[0] != 1 or volume.shape[1] != 1:
            raise ValueError(f"expected one [1,1,Z,Y,X] sample, got {tuple(volume.shape)}")
        spatial_shape = volume.shape[2:]
        normalized = []
        # grid_sample consumes X/Y/Z, whereas targets and spacing use Z/Y/X.
        for axis, size in zip((2, 1, 0), reversed(spatial_shape)):
            coordinate = points_zyx[:, axis]
            normalized.append(
                coordinate * (2.0 / (size - 1)) - 1.0
                if size > 1
                else torch.zeros_like(coordinate)
            )
        grid = torch.stack(normalized, dim=-1).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            volume,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(-1)

    def _candidate_pairs(self, centres: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return centre coordinates and sparse nearest-neighbour close pairs."""
        coordinates = torch.nonzero(centres, as_tuple=False).float()
        if coordinates.shape[0] < 2:
            return coordinates, torch.empty(
                (0, 2), dtype=torch.long, device=centres.device
            )

        spacing = coordinates.new_tensor(self.spacing)
        physical = coordinates * spacing
        distances = torch.cdist(physical, physical)
        diagonal = torch.eye(
            coordinates.shape[0], dtype=torch.bool, device=coordinates.device
        )
        distances_without_self = distances.masked_fill(diagonal, float("inf"))
        nearest = distances_without_self.argmin(dim=1)
        indices = torch.arange(coordinates.shape[0], device=coordinates.device)
        nearest_graph = nearest[:, None].eq(indices[None, :])
        nearest_graph = nearest_graph | nearest_graph.T
        eligible = (
            distances.ge(self.pair_min_distance_um)
            & distances.le(self.pair_max_distance_um)
            & nearest_graph
        )
        pairs = torch.nonzero(torch.triu(eligible, diagonal=1), as_tuple=False)
        if pairs.shape[0] == 0:
            return coordinates, pairs

        midpoints = (coordinates[pairs[:, 0]] + coordinates[pairs[:, 1]]) * 0.5
        if coordinates.shape[0] > 2 and self.third_center_exclusion_um > 0:
            midpoint_to_centres = torch.cdist(midpoints * spacing, physical)
            midpoint_to_centres[
                torch.arange(pairs.shape[0], device=pairs.device), pairs[:, 0]
            ] = float("inf")
            midpoint_to_centres[
                torch.arange(pairs.shape[0], device=pairs.device), pairs[:, 1]
            ] = float("inf")
            pairs = pairs[
                midpoint_to_centres.min(dim=1).values.ge(
                    self.third_center_exclusion_um
                )
            ]
        if pairs.shape[0] > self.max_pairs_per_sample:
            pair_distances = distances[pairs[:, 0], pairs[:, 1]]
            pairs = pairs[
                torch.topk(
                    pair_distances,
                    k=self.max_pairs_per_sample,
                    largest=False,
                    sorted=False,
                ).indices
            ]
        return coordinates, pairs

    def pair_terms_per_sample(
        self,
        probability: torch.Tensor,
        centres: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ranking loss, pair-centre focal MSE, and pair count per sample."""
        ranking_losses = []
        peak_losses = []
        sample_counts = []
        for sample_index in range(probability.shape[0]):
            coordinates, pairs = self._candidate_pairs(
                centres[sample_index, 0].gt(0)
            )
            if pairs.shape[0] == 0:
                zero = probability[sample_index].sum() * 0
                ranking_losses.append(zero)
                peak_losses.append(zero)
                sample_counts.append(
                    torch.zeros((), dtype=torch.long, device=probability.device)
                )
                continue

            first = coordinates[pairs[:, 0]].long()
            second = coordinates[pairs[:, 1]].long()
            first_probability = probability[
                sample_index, 0, first[:, 0], first[:, 1], first[:, 2]
            ]
            second_probability = probability[
                sample_index, 0, second[:, 0], second[:, 1], second[:, 2]
            ]
            midpoints = (
                coordinates[pairs[:, 0]] + coordinates[pairs[:, 1]]
            ) * 0.5
            midpoint_probability = self._sample_zyx(
                probability[sample_index : sample_index + 1],
                midpoints,
            )
            midpoint_valid = self._sample_zyx(
                valid[sample_index : sample_index + 1].float(),
                midpoints,
                mode="nearest",
            ).gt(0.5)
            pairs = pairs[midpoint_valid]
            first_probability = first_probability[midpoint_valid]
            second_probability = second_probability[midpoint_valid]
            midpoint_probability = midpoint_probability[midpoint_valid]
            ranking = F.relu(
                self.pair_margin
                + midpoint_probability
                - torch.minimum(first_probability, second_probability)
            )
            if ranking.numel() == 0:
                zero = probability[sample_index].sum() * 0
                ranking_losses.append(zero)
                peak_losses.append(zero)
            else:
                ranking_losses.append(ranking.mean())
                pair_centre_indices = torch.unique(pairs.reshape(-1))
                pair_centre_coordinates = coordinates[pair_centre_indices].long()
                pair_centre_probability = probability[
                    sample_index,
                    0,
                    pair_centre_coordinates[:, 0],
                    pair_centre_coordinates[:, 1],
                    pair_centre_coordinates[:, 2],
                ]
                centre_error = (1.0 - pair_centre_probability).clamp_min(0)
                peak_losses.append(
                    (
                        centre_error.pow(self.pair_peak_gamma)
                        * centre_error.square()
                    ).mean()
                )
            sample_counts.append(
                torch.as_tensor(
                    ranking.numel(), dtype=torch.long, device=probability.device
                )
            )
        return (
            torch.stack(ranking_losses),
            torch.stack(peak_losses),
            torch.stack(sample_counts),
        )

    def pair_loss_per_sample(
        self,
        probability: torch.Tensor,
        centres: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible view of ranking loss and active pair count."""
        ranking, _, counts = self.pair_terms_per_sample(
            probability, centres, valid
        )
        return ranking, counts

    def sample_statistics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[1] != 2:
            raise ValueError(f"Biohub trainer expects 2 logits, got {logits.shape[1]}")
        heatmap, valid, centres = self._build_target_components(target)
        foreground_logit = (logits[:, 1:2] - logits[:, 0:1]).float()
        probability = torch.sigmoid(foreground_logit)
        base_per_sample = self._masked_mean_per_sample(
            (probability - heatmap).square(), valid
        )
        pair_per_sample, peak_per_sample, _ = self.pair_terms_per_sample(
            probability, centres, valid
        )
        sample_loss = (
            base_per_sample
            + self.pair_weight * pair_per_sample
            + self.pair_peak_weight * peak_per_sample
        )
        dims = tuple(range(1, centres.ndim))
        centre_count = centres.sum(dims)
        mean_center_probability = (
            (probability * centres).sum(dims) / centre_count.clamp_min(1)
        )
        return sample_loss, centre_count, mean_center_probability

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 2:
            raise ValueError(f"Biohub trainer expects 2 logits, got {logits.shape[1]}")
        heatmap, valid, centres = self._build_target_components(target)
        foreground_logit = (logits[:, 1:2] - logits[:, 0:1]).float()
        probability = torch.sigmoid(foreground_logit)
        base_loss = self._masked_mean((probability - heatmap).square(), valid)
        pair_per_sample, peak_per_sample, pair_counts = self.pair_terms_per_sample(
            probability, centres, valid
        )
        active_samples = pair_counts.gt(0)
        pair_loss = (
            pair_per_sample[active_samples].mean()
            if active_samples.any()
            else probability.sum() * 0
        )
        pair_peak_loss = (
            peak_per_sample[active_samples].mean()
            if active_samples.any()
            else probability.sum() * 0
        )
        self.last_base_loss = float(base_loss.detach())
        self.last_pair_loss = float(pair_loss.detach())
        self.last_pair_peak_loss = float(pair_peak_loss.detach())
        self.last_active_pairs = int(pair_counts.sum().detach())
        return (
            base_loss
            + self.pair_weight * pair_loss
            + self.pair_peak_weight * pair_peak_loss
        )


class nnUNetTrainerBiohubMaskedHeatmapBase(nnUNetTrainer):
    loss_mode = "mse"
    positive_weight = 16.0
    target_composition = "sum_clip"
    center_weight = 1.0
    center_radius_um = 1.5

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Continuous Gaussian targets are generated at the full output resolution.
        self.enable_deep_supervision = False
        self.initial_lr = float(os.environ.get("BIOHUB_NNUNET_LR", "0.001"))
        self.num_epochs = int(os.environ.get("BIOHUB_NNUNET_EPOCHS", "1000"))
        self.batch_size = int(os.environ.get("BIOHUB_NNUNET_BATCH_SIZE", "4"))
        self.num_iterations_per_epoch = int(os.environ.get("BIOHUB_NNUNET_ITERS_PER_EPOCH", "125"))
        self.num_val_iterations_per_epoch = int(os.environ.get("BIOHUB_NNUNET_VAL_ITERS", "25"))
        self.validation_every = int(os.environ.get("BIOHUB_NNUNET_VAL_EVERY", "5"))
        self.oversample_foreground_percent = float(os.environ.get("BIOHUB_NNUNET_OVERSAMPLE", "0.5"))
        self.save_every = int(os.environ.get("BIOHUB_NNUNET_SAVE_EVERY", "10"))
        sample_loss_file = os.environ.get("BIOHUB_NNUNET_SAMPLE_LOSS_FILE", "").strip()
        self.sample_loss_file = Path(sample_loss_file) if sample_loss_file else None
        cudnn_benchmark = os.environ.get("BIOHUB_NNUNET_CUDNN_BENCHMARK", "true").lower()
        if cudnn_benchmark not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("BIOHUB_NNUNET_CUDNN_BENCHMARK must be a boolean")
        torch.backends.cudnn.benchmark = cudnn_benchmark in {"true", "1", "yes"}
        if self.batch_size <= 0:
            raise ValueError("BIOHUB_NNUNET_BATCH_SIZE must be positive")
        if self.validation_every <= 0:
            raise ValueError("BIOHUB_NNUNET_VAL_EVERY must be positive")
        if self.save_every <= 0:
            raise ValueError("BIOHUB_NNUNET_SAVE_EVERY must be positive")

    def _run_validation_this_epoch(self) -> bool:
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.validation_every == 0
            or self.current_epoch == self.num_epochs - 1
        )

    def _carry_forward_validation_metrics(self) -> None:
        """Keep nnU-Net's one-value-per-epoch logger aligned on skipped validation epochs."""
        previous_epoch = self.current_epoch - 1
        if previous_epoch < 0:
            raise RuntimeError("validation cannot be skipped before initial metrics exist")
        for key in ("val_losses", "dice_per_class_or_region", "mean_fg_dice", "ema_fg_dice"):
            self.logger.log(key, self.logger.get_value(key, step=previous_epoch), self.current_epoch)

    def run_training(self):
        """Run expensive validation only at the configured epoch cadence."""
        self.on_train_start()

        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            if self._run_validation_this_epoch():
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)
            else:
                self.print_to_log_file(
                    f"Skipping validation at epoch {self.current_epoch}; "
                    f"cadence={self.validation_every}"
                )
                self._carry_forward_validation_metrics()

            self.on_epoch_end()

        self.on_train_end()

    def save_checkpoint(self, filename: str) -> None:
        """Write checkpoints atomically so an interrupted save keeps the prior file."""
        if self.local_rank != 0 or self.disable_checkpointing:
            super().save_checkpoint(filename)
            return
        temporary = f"{filename}.tmp"
        super().save_checkpoint(temporary)
        os.replace(temporary, filename)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, amsgrad=True
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def _build_loss(self):
        spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        return MaskedGaussianHeatmapLoss(
            spacing_zyx_um=spacing,
            sigma_um=float(os.environ.get("BIOHUB_HEATMAP_SIGMA_UM", "2.0")),
            truncate_sigma=float(os.environ.get("BIOHUB_HEATMAP_TRUNCATE_SIGMA", "3.0")),
            mode=self.loss_mode,
            positive_weight=float(os.environ.get("BIOHUB_HEATMAP_POSITIVE_WEIGHT", str(self.positive_weight))),
            topk_fraction=float(os.environ.get("BIOHUB_HEATMAP_TOPK_FRACTION", "0.2")),
            smoothing=float(os.environ.get("BIOHUB_HEATMAP_LABEL_SMOOTHING", "0.01")),
            ignore_label=int(self.label_manager.ignore_label),
            target_composition=self.target_composition,
            center_weight=float(
                os.environ.get("BIOHUB_HEATMAP_CENTER_WEIGHT", str(self.center_weight))
            ),
            center_radius_um=float(
                os.environ.get(
                    "BIOHUB_HEATMAP_CENTER_RADIUS_UM", str(self.center_radius_um)
                )
            ),
        )

    def _append_validation_sample_losses(
        self,
        batch: dict,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if self.sample_loss_file is None or self.local_rank != 0:
            return
        keys = batch.get("keys")
        if keys is None:
            self.print_to_log_file(
                "BIOHUB_NNUNET_SAMPLE_LOSS_FILE is set but validation batch has no keys"
            )
            return
        losses, labels, predictions = self.loss.sample_statistics(output, target)
        sample_ids = [str(value) for value in list(keys)]
        if len(sample_ids) != int(losses.numel()):
            raise RuntimeError(
                f"validation key count {len(sample_ids)} does not match batch "
                f"size {int(losses.numel())}"
            )
        self.sample_loss_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.sample_loss_file.exists() or self.sample_loss_file.stat().st_size == 0
        )
        with self.sample_loss_file.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "epoch",
                    "split",
                    "sample_id",
                    "series_id",
                    "label",
                    "prediction",
                    "loss",
                    "fold",
                ),
            )
            if write_header:
                writer.writeheader()
            for sample_id, label, prediction, loss in zip(
                sample_ids,
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                losses.detach().cpu().tolist(),
            ):
                writer.writerow(
                    {
                        "epoch": int(self.current_epoch) + 1,
                        "split": "val",
                        "sample_id": sample_id,
                        "series_id": sample_id.rsplit("_t", 1)[0],
                        "label": int(label),
                        "prediction": float(prediction),
                        "loss": float(loss),
                        "fold": self.fold,
                    }
                )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)
        self._append_validation_sample_losses(batch, output, target)
        with torch.no_grad():
            gaussian, valid = self.loss.build_target(target)
            probability = torch.softmax(output.float(), dim=1)[:, 1:2]
            predicted = probability.gt(0.5) & valid
            expected = gaussian.gt(0.5) & valid
            axes = (0, 2, 3, 4)
            tp = (predicted & expected).sum(axes).detach().cpu().numpy().astype(np.float64)
            fp = (predicted & ~expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
            fn = (~predicted & expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}


class nnUNetTrainerBiohubMaskedHeatmapMSE(nnUNetTrainerBiohubMaskedHeatmapBase):
    """Default: masked Gaussian MSE."""


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAware(nnUNetTrainerBiohubMaskedHeatmapMSE):
    """MSE ablation with XY-only geometry for the 4:1 Z/XY spacing ratio.

    nnU-Net's dummy-2D path applies the spatial transform independently to
    each Z slice while keeping the same sampled transform across the stack.
    It also excludes Z from simulated low-resolution augmentation.  Mirroring
    is explicitly limited to Y/X.  Intensity augmentation and synchronized
    scaling otherwise remain identical to the MSE baseline.
    """

    biohub_mirror_axes = (1, 2)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = np.asarray(self.configuration_manager.patch_size, dtype=int)
        if patch_size.size != 3:
            raise RuntimeError("Biohub anisotropy-aware augmentation expects a 3D patch")

        rotation_for_DA = (-30.0 / 360.0 * 2.0 * np.pi, 30.0 / 360.0 * 2.0 * np.pi)
        do_dummy_2d_data_aug = True
        mirror_axes = self.biohub_mirror_axes

        # Only the Y/X plane is rotated. Keep Z at the requested patch depth
        # and compute the larger loader crop from the in-plane rotation/scale.
        max_angle = max(abs(rotation_for_DA[0]), abs(rotation_for_DA[1]))
        cos_angle, sin_angle = abs(np.cos(max_angle)), abs(np.sin(max_angle))
        initial_yx = np.ceil(
            np.asarray(
                [
                    patch_size[1] * cos_angle + patch_size[2] * sin_angle,
                    patch_size[1] * sin_angle + patch_size[2] * cos_angle,
                ]
            )
            / 0.85
        ).astype(int)
        initial_patch_size = np.asarray(
            [int(patch_size[0]), int(initial_yx[0]), int(initial_yx[1])], dtype=int
        )

        self.print_to_log_file("Biohub anisotropy-aware augmentation: XY-only spatial transform")
        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.print_to_log_file(f"training/inference mirror axes: {mirror_axes}")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlip(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAware
):
    """XY-only interpolation/resolution transforms with discrete Z/Y/X flips."""

    biohub_mirror_axes = (0, 1, 2)


class _TruncatedUNetDecoder(nn.Module):
    """A standard nnU-Net decoder with one or more highest-resolution stages removed."""

    def __init__(
        self,
        decoder: nn.Module,
        omitted_stride_zyx: Sequence[int],
        num_omitted_stages: int = 1,
    ):
        super().__init__()
        if num_omitted_stages < 1 or num_omitted_stages >= len(decoder.stages):
            raise ValueError(
                f"num_omitted_stages must be in [1, {len(decoder.stages) - 1}], "
                f"got {num_omitted_stages}"
            )
        self.deep_supervision = bool(decoder.deep_supervision)
        self.encoder = decoder.encoder
        self.num_classes = int(decoder.num_classes)
        self.omitted_stride_zyx = tuple(int(v) for v in omitted_stride_zyx)
        self.num_omitted_stages = int(num_omitted_stages)
        keep = len(decoder.stages) - self.num_omitted_stages
        self.stages = nn.ModuleList(list(decoder.stages[:keep]))
        self.transpconvs = nn.ModuleList(list(decoder.transpconvs[:keep]))
        self.seg_layers = nn.ModuleList(list(decoder.seg_layers[:keep]))

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for stage_index in range(len(self.stages)):
            x = self.transpconvs[stage_index](lres_input)
            x = torch.cat((x, skips[-(stage_index + 2)]), 1)
            x = self.stages[stage_index](x)
            if self.deep_supervision or stage_index == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[stage_index](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        current_size = list(input_size)
        for stride in self.encoder.strides[:-1]:
            current_size = [size // step for size, step in zip(current_size, stride)]
            skip_sizes.append(current_size)

        output = np.int64(0)
        for stage_index in range(len(self.stages)):
            stage_size = skip_sizes[-(stage_index + 1)]
            output += self.stages[stage_index].compute_conv_feature_map_size(stage_size)
            output += np.prod(
                [self.encoder.output_channels[-(stage_index + 2)], *stage_size],
                dtype=np.int64,
            )
            if self.deep_supervision or stage_index == len(self.stages) - 1:
                output += np.prod([self.num_classes, *stage_size], dtype=np.int64)
        return output


class _XYHalfOutputNetwork(nn.Module):
    """Run a truncated XY decoder and restore coordinates only for nnU-Net inference."""

    def __init__(
        self,
        network: nn.Module,
        omitted_stride_zyx: Sequence[int],
        num_omitted_stages: int = 1,
    ):
        super().__init__()
        self.encoder = network.encoder
        self.decoder = _TruncatedUNetDecoder(
            network.decoder,
            omitted_stride_zyx,
            num_omitted_stages=num_omitted_stages,
        )
        # A network built directly by nnU-Net's inference code must emit the
        # input patch shape. The trainer disables this flag for train/patch-val
        # so the loss is evaluated on the native low-resolution head.
        self.return_full_resolution = True

    def forward(self, x):
        logits = self.decoder(self.encoder(x))
        if self.return_full_resolution:
            if isinstance(logits, (tuple, list)):
                raise RuntimeError("XY-half Biohub trainer does not support deep-supervision inference")
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlip
):
    """Stop before the final (1,2,2) decoder stage and predict on an XY-half grid."""

    output_stride_zyx = (1, 2, 2)

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        if not hasattr(network, "decoder") or not hasattr(network, "encoder"):
            raise TypeError("XY-half trainer requires an encoder-decoder nnU-Net architecture")
        final_stride = tuple(int(v) for v in network.encoder.strides[1])
        expected_stride = nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf.output_stride_zyx
        if final_stride != expected_stride:
            raise ValueError(
                f"expected the omitted highest-resolution decoder stride to be {expected_stride}, "
                f"got {final_stride}"
            )
        return _XYHalfOutputNetwork(network, final_stride)

    def _set_full_resolution_inference(self, enabled: bool) -> None:
        network = self.network
        while True:
            if hasattr(network, "_orig_mod"):
                network = network._orig_mod
                continue
            if hasattr(network, "module"):
                network = network.module
                continue
            break
        if not isinstance(network, _XYHalfOutputNetwork):
            raise TypeError(f"unexpected XY-half network wrapper: {type(network).__name__}")
        network.return_full_resolution = bool(enabled)

    def initialize(self):
        super().initialize()
        self._set_full_resolution_inference(False)

    def perform_actual_validation(self, *args, **kwargs):
        self._set_full_resolution_inference(True)
        try:
            return super().perform_actual_validation(*args, **kwargs)
        finally:
            self._set_full_resolution_inference(False)

    def _build_loss(self):
        input_spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        output_spacing = tuple(
            spacing * stride for spacing, stride in zip(input_spacing, self.output_stride_zyx)
        )
        return MaskedGaussianHeatmapLoss(
            spacing_zyx_um=output_spacing,
            sigma_um=float(os.environ.get("BIOHUB_HEATMAP_SIGMA_UM", "2.0")),
            truncate_sigma=float(os.environ.get("BIOHUB_HEATMAP_TRUNCATE_SIGMA", "3.0")),
            mode=self.loss_mode,
            positive_weight=float(
                os.environ.get("BIOHUB_HEATMAP_POSITIVE_WEIGHT", str(self.positive_weight))
            ),
            topk_fraction=float(os.environ.get("BIOHUB_HEATMAP_TOPK_FRACTION", "0.2")),
            smoothing=float(os.environ.get("BIOHUB_HEATMAP_LABEL_SMOOTHING", "0.01")),
            ignore_label=int(self.label_manager.ignore_label),
            target_stride_zyx=self.output_stride_zyx,
            valid_radius_um=float(os.environ.get("BIOHUB_VALID_RADIUS_UM", "6.0")),
            target_composition=self.target_composition,
            center_weight=float(
                os.environ.get("BIOHUB_HEATMAP_CENTER_WEIGHT", str(self.center_weight))
            ),
            center_radius_um=float(
                os.environ.get(
                    "BIOHUB_HEATMAP_CENTER_RADIUS_UM", str(self.center_radius_um)
                )
            ),
        )


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxCompose(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Matched XY-half ablation using voxel-wise max for instance Gaussians."""

    target_composition = "max"


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfRepelCoding(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Matched XY-half ablation using paper-formula 3D Repel Coding."""

    def _build_loss(self):
        input_spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        output_spacing = tuple(
            spacing * stride
            for spacing, stride in zip(input_spacing, self.output_stride_zyx)
        )
        return RepelCodingMaskedHeatmapLoss(
            spacing_zyx_um=output_spacing,
            sigma_um=float(os.environ.get("BIOHUB_HEATMAP_SIGMA_UM", "2.0")),
            truncate_sigma=float(
                os.environ.get("BIOHUB_HEATMAP_TRUNCATE_SIGMA", "3.0")
            ),
            mode=self.loss_mode,
            ignore_label=int(self.label_manager.ignore_label),
            target_stride_zyx=self.output_stride_zyx,
            valid_radius_um=float(os.environ.get("BIOHUB_VALID_RADIUS_UM", "6.0")),
            repel_alpha_per_um=float(
                os.environ.get(
                    "BIOHUB_REPEL_ALPHA_PER_UM",
                    "0.42466090014400953",
                )
            ),
            repel_support_um=float(
                os.environ.get("BIOHUB_REPEL_SUPPORT_UM", "6.0")
            ),
            distance_chunk_size=int(
                os.environ.get("BIOHUB_REPEL_DISTANCE_CHUNK_SIZE", "8192")
            ),
        )


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfClosePairValley(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Sum+clip MSE plus annotated close-pair centre-vs-midpoint ranking."""

    close_pair_peak_weight = 0.0
    close_pair_peak_gamma = 2.0

    def _build_loss(self):
        input_spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        output_spacing = tuple(
            spacing * stride
            for spacing, stride in zip(input_spacing, self.output_stride_zyx)
        )
        return ClosePairValleyMaskedGaussianHeatmapLoss(
            spacing_zyx_um=output_spacing,
            sigma_um=float(os.environ.get("BIOHUB_HEATMAP_SIGMA_UM", "2.0")),
            truncate_sigma=float(
                os.environ.get("BIOHUB_HEATMAP_TRUNCATE_SIGMA", "3.0")
            ),
            mode="mse",
            ignore_label=int(self.label_manager.ignore_label),
            target_stride_zyx=self.output_stride_zyx,
            valid_radius_um=float(os.environ.get("BIOHUB_VALID_RADIUS_UM", "6.0")),
            target_composition="sum_clip",
            pair_weight=float(
                os.environ.get("BIOHUB_CLOSE_PAIR_LOSS_WEIGHT", "0.10")
            ),
            pair_margin=float(
                os.environ.get("BIOHUB_CLOSE_PAIR_MARGIN", "0.15")
            ),
            pair_min_distance_um=float(
                os.environ.get("BIOHUB_CLOSE_PAIR_MIN_DISTANCE_UM", "2.5")
            ),
            pair_max_distance_um=float(
                os.environ.get("BIOHUB_CLOSE_PAIR_MAX_DISTANCE_UM", "6.0")
            ),
            third_center_exclusion_um=float(
                os.environ.get(
                    "BIOHUB_CLOSE_PAIR_THIRD_CENTER_EXCLUSION_UM", "2.0"
                )
            ),
            max_pairs_per_sample=int(
                os.environ.get("BIOHUB_CLOSE_PAIR_MAX_PAIRS_PER_SAMPLE", "256")
            ),
            pair_peak_weight=float(
                os.environ.get(
                    "BIOHUB_CLOSE_PAIR_PEAK_WEIGHT",
                    str(self.close_pair_peak_weight),
                )
            ),
            pair_peak_gamma=float(
                os.environ.get(
                    "BIOHUB_CLOSE_PAIR_PEAK_GAMMA",
                    str(self.close_pair_peak_gamma),
                )
            ),
        )


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfClosePairValleyFocalPeak(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfClosePairValley
):
    """Valley ranking plus focal MSE on centres belonging to eligible pairs."""

    close_pair_peak_weight = 0.10
    close_pair_peak_gamma = 2.0


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxComposeRestart300(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxCompose
):
    """Matched 300-epoch LR-restart control initialized from max1000 weights."""


class _MaxComposeCenterWeightedMSEBase(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxCompose
):
    """Max target with extra MSE weight in a physical-radius GT-center region."""

    loss_mode = "center_weighted_mse"
    center_radius_um = 1.5


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxComposeCenterW2Restart300(
    _MaxComposeCenterWeightedMSEBase
):
    center_weight = 2.0


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxComposeCenterW4Restart300(
    _MaxComposeCenterWeightedMSEBase
):
    center_weight = 4.0


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfMaxComposeCenterW8Restart300(
    _MaxComposeCenterWeightedMSEBase
):
    center_weight = 8.0


class nnUNetTrainerBiohubMaskedHeatmapWeightedMSEAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Matched XY-half ablation changing only masked MSE to balanced weighted MSE."""

    loss_mode = "weighted_mse"


class nnUNetTrainerBiohubMaskedHeatmapTopK20BCEAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Matched XY-half ablation using BYU second-place-style TopK20 BCE."""

    loss_mode = "topk_bce"


class nnUNetTrainerBiohubMaskedHeatmapSmoothBCEAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Matched XY-half ablation using unweighted BYU-style SmoothBCE."""

    loss_mode = "smooth_bce"


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYQuarter(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Stop two decoder stages early and predict on an isotropic XY-quarter grid."""

    output_stride_zyx = (1, 4, 4)

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        if not hasattr(network, "decoder") or not hasattr(network, "encoder"):
            raise TypeError("XY-quarter trainer requires an encoder-decoder nnU-Net architecture")
        omitted_stages = 2
        omitted_strides = network.encoder.strides[1 : 1 + omitted_stages]
        cumulative_stride = tuple(
            int(np.prod([int(stride[axis]) for stride in omitted_strides]))
            for axis in range(3)
        )
        expected_stride = (
            nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYQuarter.output_stride_zyx
        )
        if cumulative_stride != expected_stride:
            raise ValueError(
                f"expected the two omitted highest-resolution decoder stages to have cumulative "
                f"stride {expected_stride}, got {cumulative_stride}"
            )
        return _XYHalfOutputNetwork(
            network,
            cumulative_stride,
            num_omitted_stages=omitted_stages,
        )


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYQuarterB2(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYQuarter
):
    """Isolated batch-2 rerun of the XY-quarter architecture.

    The implementation intentionally remains identical to XYQuarter; the
    distinct class name gives the matched batch-size ablation an independent
    nnU-Net results directory and prevents checkpoint contamination.
    """


class nnUNetTrainerBiohubMaskedHeatmapWeightedMSE(nnUNetTrainerBiohubMaskedHeatmapBase):
    """CZII 4th-place inspired positive-weighted MSE."""

    loss_mode = "weighted_mse"


class nnUNetTrainerBiohubMaskedHeatmapWeightedBCE(nnUNetTrainerBiohubMaskedHeatmapBase):
    """CZII 1st-place segmentation-loss analogue for a binary heatmap."""

    loss_mode = "weighted_bce"
    positive_weight = 256.0


class nnUNetTrainerBiohubMaskedHeatmapBCEPlusDice(nnUNetTrainerBiohubMaskedHeatmapBase):
    """CZII 2nd-place Dice/CE family adapted to a soft heatmap."""

    loss_mode = "bce_dice"


class nnUNetTrainerBiohubMaskedHeatmapBCEPlusTversky(nnUNetTrainerBiohubMaskedHeatmapBase):
    """CZII 2nd-place Tversky/CE family adapted to a soft heatmap."""

    loss_mode = "bce_tversky"


class nnUNetTrainerBiohubMaskedHeatmapLabelSmoothingBCE(nnUNetTrainerBiohubMaskedHeatmapBase):
    """CZII 5th-place label-smoothing CE adapted to a binary heatmap."""

    loss_mode = "label_smoothing_bce"


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfLong2000(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Isolated 2000-epoch duration ablation with immutable milestone checkpoints."""

    milestone_epochs = frozenset((500, 1000, 1500))

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        completed_epoch = int(self.current_epoch) + 1
        if os.path.basename(str(filename)) != "checkpoint_latest.pth":
            return
        if completed_epoch not in self.milestone_epochs:
            return
        destination = os.path.join(os.path.dirname(str(filename)), f"checkpoint_epoch{completed_epoch}.pth")
        temporary = destination + ".tmp"
        shutil.copy2(filename, temporary)
        os.replace(temporary, destination)
