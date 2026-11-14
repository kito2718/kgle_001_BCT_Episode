"""Three-channel Biohub heatmap trainer: cell, division parent, division daughter.

The output channels are independent logits. Parent and daughter are subsets of
the cell channel, so a softmax head would be incorrect.

Categorical target encoding (stored in the single nnU-Net label volume):

* 0: human-verified local background
* 1: human normal cell centre
* 2: human division-parent centre
* 3: human division-daughter centre
* 4: human centre that is both daughter and parent
* 5: pseudo cell centre with unknown division role
* 6: pseudo-local background (valid for cell, ignored for role heads)
* 255: unknown / ignore

The loss family and channel weights are environment-controlled so the model
plumbing can be prepared before the active loss ablation is decided.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.utilities.helpers import dummy_context

try:
    from torch import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast

try:
    from .nnUNetTrainerBiohubMaskedHeatmap import (
        nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
    )
except ImportError:  # pragma: no cover - supports standalone smoke imports
    from nnUNetTrainerBiohubMaskedHeatmap import (
        nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
    )


HUMAN_NORMAL_CELL = 1
HUMAN_DIVISION_PARENT = 2
HUMAN_DIVISION_DAUGHTER = 3
HUMAN_PARENT_AND_DAUGHTER = 4
PSEUDO_CELL_UNKNOWN_ROLE = 5
PSEUDO_LOCAL_BACKGROUND = 6


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class MaskedThreeChannelGaussianHeatmapLoss(nn.Module):
    """Render three Gaussian targets and apply channel-specific masked losses."""

    supported_modes = {
        "mse",
        "weighted_mse",
        "topk_bce",
        "weighted_bce",
        "label_smoothing_bce",
        "smooth_bce",
        "bce_dice",
        "bce_tversky",
    }

    def __init__(
        self,
        spacing_zyx_um: Sequence[float],
        sigma_um: float = 2.0,
        truncate_sigma: float = 3.0,
        cell_mode: str = "mse",
        role_mode: str = "mse",
        channel_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
        channel_positive_weights: Sequence[float] = (1.0, 1.0, 1.0),
        topk_fraction: float = 0.2,
        smoothing: float = 0.01,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        ignore_label: int = 255,
        target_stride_zyx: Sequence[int] = (1, 1, 1),
        valid_radius_um: float | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.spacing = tuple(float(v) for v in spacing_zyx_um)
        self.sigma_um = float(sigma_um)
        self.truncate_sigma = float(truncate_sigma)
        self.modes = (str(cell_mode), str(role_mode), str(role_mode))
        unknown_modes = set(self.modes) - self.supported_modes
        if unknown_modes:
            raise ValueError(f"unsupported loss modes: {sorted(unknown_modes)}")
        self.channel_loss_weights = tuple(float(v) for v in channel_loss_weights)
        self.channel_positive_weights = tuple(float(v) for v in channel_positive_weights)
        if len(self.channel_loss_weights) != 3 or any(v < 0 for v in self.channel_loss_weights):
            raise ValueError("channel_loss_weights must contain three non-negative values")
        if len(self.channel_positive_weights) != 3 or any(
            v <= 0 for v in self.channel_positive_weights
        ):
            raise ValueError("channel_positive_weights must contain three positive values")
        self.topk_fraction = float(topk_fraction)
        if not 0 < self.topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        self.smoothing = float(smoothing)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.ignore_label = int(ignore_label)
        self.target_stride = tuple(int(v) for v in target_stride_zyx)
        if len(self.target_stride) != 3 or any(v < 1 for v in self.target_stride):
            raise ValueError("target_stride_zyx must contain three positive integers")
        self.valid_radius_um = None if valid_radius_um is None else float(valid_radius_um)
        if self.target_stride != (1, 1, 1) and self.valid_radius_um is None:
            raise ValueError("valid_radius_um is required for a lower-resolution target")
        self.eps = float(eps)
        self._kernel_cache: dict[tuple, torch.Tensor] = {}
        self._valid_kernel_cache: dict[tuple, torch.Tensor] = {}
        self.last_channel_losses: tuple[torch.Tensor, ...] | None = None

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

    @staticmethod
    def _centre_channels(
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        human_cell = labels.ge(HUMAN_NORMAL_CELL) & labels.le(HUMAN_PARENT_AND_DAUGHTER)
        pseudo_cell = labels.eq(PSEUDO_CELL_UNKNOWN_ROLE)
        cell = human_cell | pseudo_cell
        parent = labels.eq(HUMAN_DIVISION_PARENT) | labels.eq(HUMAN_PARENT_AND_DAUGHTER)
        daughter = labels.eq(HUMAN_DIVISION_DAUGHTER) | labels.eq(HUMAN_PARENT_AND_DAUGHTER)
        centres = torch.cat((cell, parent, daughter), dim=1).float()
        return centres, human_cell.float(), pseudo_cell.float()

    def build_target(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim != 5 or target.shape[1] != 1:
            raise ValueError(f"expected categorical target [B,1,Z,Y,X], got {tuple(target.shape)}")
        labels = target[:, :1].round().long()
        centres, human_centres, pseudo_centres = self._centre_channels(labels)

        if self.target_stride == (1, 1, 1):
            cell_valid = labels.ge(0) & labels.le(PSEUDO_LOCAL_BACKGROUND)
            # Pseudo centres and pseudo-local background have unknown lineage role.
            role_valid = labels.ge(0) & labels.le(HUMAN_PARENT_AND_DAUGHTER)
        else:
            centres = F.max_pool3d(
                centres,
                kernel_size=self.target_stride,
                stride=self.target_stride,
            )
            human_centres = F.max_pool3d(
                human_centres,
                kernel_size=self.target_stride,
                stride=self.target_stride,
            )
            pseudo_centres = F.max_pool3d(
                pseudo_centres,
                kernel_size=self.target_stride,
                stride=self.target_stride,
            )
            valid_kernel = self._valid_kernel(target.device)
            valid_padding = tuple(int(size // 2) for size in valid_kernel.shape[2:])
            cell_valid = F.conv3d(
                centres[:, :1], valid_kernel, padding=valid_padding
            ).gt(0)
            role_valid = F.conv3d(
                human_centres, valid_kernel, padding=valid_padding
            ).gt(0)
            # A pseudo cell has unknown lineage status. Its entire local support
            # overrides overlapping human-valid role background and stays ignored.
            pseudo_role_ignore = F.conv3d(
                pseudo_centres, valid_kernel, padding=valid_padding
            ).gt(0)
            role_valid = role_valid & ~pseudo_role_ignore

        kernel = self._kernel(target.device).repeat(3, 1, 1, 1, 1)
        padding = tuple(int(size // 2) for size in kernel.shape[2:])
        heatmaps = F.conv3d(centres, kernel, padding=padding, groups=3).clamp_(0, 1)
        valid = torch.cat((cell_valid, role_valid, role_valid), dim=1)
        # The high-confidence core of a human parent/daughter target takes
        # precedence if a pseudo ignore ball overlaps it in a dense region.
        # Tiny rectangular-kernel tails do not turn unknown pseudo roles into
        # negative/role supervision.
        valid[:, 1:] = valid[:, 1:] | heatmaps[:, 1:].ge(0.5)
        heatmaps = heatmaps * valid.float()
        return heatmaps, valid

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective = valid.float()
        if weights is not None:
            effective = effective * weights
        return (values * effective).sum() / effective.sum().clamp_min(1.0)

    def _channel_loss(
        self,
        logit: torch.Tensor,
        heatmap: torch.Tensor,
        valid: torch.Tensor,
        mode: str,
        positive_weight: float,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logit)
        if mode == "mse":
            return self._masked_mean((probability - heatmap).square(), valid)
        if mode == "weighted_mse":
            weights = 1 + (positive_weight - 1) * heatmap
            return self._masked_mean((probability - heatmap).square(), valid, weights)

        bce_target = heatmap
        if mode == "label_smoothing_bce":
            bce_target = heatmap * (1 - self.smoothing) + 0.5 * self.smoothing
        elif mode == "smooth_bce":
            bce_target = heatmap * (1 - self.smoothing) + (1 - heatmap) * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logit, bce_target, reduction="none")
        if mode == "topk_bce":
            valid_bce = bce.masked_select(valid)
            if valid_bce.numel() == 0:
                return bce.sum() * 0
            k = max(1, int(valid_bce.numel() * self.topk_fraction))
            return torch.topk(valid_bce, k=k, sorted=False).values.mean()
        if mode == "weighted_bce":
            weights = 1 + (positive_weight - 1) * heatmap
            return self._masked_mean(bce, valid, weights)
        if mode in {"label_smoothing_bce", "smooth_bce"}:
            return self._masked_mean(bce, valid)

        masked_probability = probability * valid.float()
        masked_target = heatmap * valid.float()
        dims = tuple(range(2, probability.ndim))
        intersection = (masked_probability * masked_target).sum(dims)
        if mode == "bce_dice":
            dice = (2 * intersection + self.eps) / (
                masked_probability.sum(dims) + masked_target.sum(dims) + self.eps
            )
            return self._masked_mean(bce, valid) + (1 - dice).mean()
        if mode == "bce_tversky":
            fp = (masked_probability * (1 - masked_target)).sum(dims)
            fn = ((1 - masked_probability) * masked_target).sum(dims)
            tversky = (intersection + self.eps) / (
                intersection + self.tversky_alpha * fp + self.tversky_beta * fn + self.eps
            )
            return self._masked_mean(bce, valid) + (1 - tversky).mean()
        raise AssertionError(f"unhandled loss mode: {mode}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError(f"three-channel trainer expects [B,3,Z,Y,X], got {tuple(logits.shape)}")
        heatmaps, valid = self.build_target(target)
        if logits.shape[2:] != heatmaps.shape[2:]:
            raise ValueError(
                f"logit/target spatial shapes differ: {tuple(logits.shape)} vs {tuple(heatmaps.shape)}"
            )
        losses = tuple(
            self._channel_loss(
                logits[:, index : index + 1].float(),
                heatmaps[:, index : index + 1],
                valid[:, index : index + 1],
                self.modes[index],
                self.channel_positive_weights[index],
            )
            for index in range(3)
        )
        self.last_channel_losses = losses
        return sum(weight * loss for weight, loss in zip(self.channel_loss_weights, losses))


class nnUNetTrainerBiohubThreeChannelHeatmapAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """XY-half trainer with independent cell/parent/daughter heatmap logits."""

    output_channel_names = ("cell", "division_parent", "division_daughter")

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        del num_output_channels  # dataset categories encode masks; the model always emits 3 logits.
        return nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            3,
            enable_deep_supervision,
        )

    def _build_loss(self):
        input_spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        output_spacing = tuple(
            spacing * stride for spacing, stride in zip(input_spacing, self.output_stride_zyx)
        )
        cell_mode = os.environ.get("BIOHUB_3CH_CELL_LOSS_MODE", "mse")
        role_mode = os.environ.get("BIOHUB_3CH_ROLE_LOSS_MODE", cell_mode)
        return MaskedThreeChannelGaussianHeatmapLoss(
            spacing_zyx_um=output_spacing,
            sigma_um=_env_float("BIOHUB_HEATMAP_SIGMA_UM", 2.0),
            truncate_sigma=_env_float("BIOHUB_HEATMAP_TRUNCATE_SIGMA", 3.0),
            cell_mode=cell_mode,
            role_mode=role_mode,
            channel_loss_weights=(
                _env_float("BIOHUB_3CH_CELL_LOSS_WEIGHT", 1.0),
                _env_float("BIOHUB_3CH_PARENT_LOSS_WEIGHT", 1.0),
                _env_float("BIOHUB_3CH_DAUGHTER_LOSS_WEIGHT", 1.0),
            ),
            channel_positive_weights=(
                _env_float("BIOHUB_3CH_CELL_POSITIVE_WEIGHT", 1.0),
                _env_float("BIOHUB_3CH_PARENT_POSITIVE_WEIGHT", 1.0),
                _env_float("BIOHUB_3CH_DAUGHTER_POSITIVE_WEIGHT", 1.0),
            ),
            topk_fraction=_env_float("BIOHUB_HEATMAP_TOPK_FRACTION", 0.2),
            smoothing=_env_float("BIOHUB_HEATMAP_LABEL_SMOOTHING", 0.01),
            tversky_alpha=_env_float("BIOHUB_HEATMAP_TVERSKY_ALPHA", 0.3),
            tversky_beta=_env_float("BIOHUB_HEATMAP_TVERSKY_BETA", 0.7),
            ignore_label=int(self.label_manager.ignore_label),
            target_stride_zyx=self.output_stride_zyx,
            valid_radius_um=_env_float("BIOHUB_VALID_RADIUS_UM", 6.0),
        )

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)
        with torch.no_grad():
            heatmaps, valid = self.loss.build_target(target)
            probability = torch.sigmoid(output.float())
            predicted = probability.gt(0.5) & valid
            expected = heatmaps.gt(0.5) & valid
            axes = (0, 2, 3, 4)
            tp = (predicted & expected).sum(axes).detach().cpu().numpy().astype(np.float64)
            fp = (predicted & ~expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
            fn = (~predicted & expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}
