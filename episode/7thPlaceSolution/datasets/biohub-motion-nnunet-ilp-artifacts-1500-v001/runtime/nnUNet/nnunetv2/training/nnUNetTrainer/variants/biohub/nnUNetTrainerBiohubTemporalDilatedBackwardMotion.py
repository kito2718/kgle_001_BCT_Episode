"""Scratch T3 detector with random temporal dilation and backward motion head."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.helpers import dummy_context

try:
    from torch import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubMaskedHeatmap import (
    MaskedGaussianHeatmapLoss,
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalThreeFrame import (
    BiohubTemporalThreeFrameDataset,
    _LazyTemporalChannelStack,
)


class _TemporalConditionedStack:
    """Three image channels plus a constant normalized dilation channel."""

    def __init__(self, temporal: _LazyTemporalChannelStack, dilation: int):
        self.temporal = temporal
        self.dilation = int(dilation)
        self.shape = (4, *temporal.shape[1:])
        self.ndim = 4
        self.dtype = temporal.dtype

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (self.ndim - len(index))
        channel_item, spatial = index[0], index[1:]
        channels = np.atleast_1d(np.arange(4)[channel_item]).tolist()
        arrays = []
        for channel in channels:
            if channel < 3:
                value = np.asarray(self.temporal[(channel, *spatial)])
            else:
                reference = np.asarray(self.temporal[(1, *spatial)])
                value = np.full(reference.shape, self.dilation / 2.0, dtype=self.dtype)
            arrays.append(value[None])
        result = np.concatenate(arrays, axis=0)
        return result[0] if isinstance(channel_item, (int, np.integer)) else result

    def __array__(self, dtype=None):
        result = self[(slice(None),) * self.ndim]
        return result.astype(dtype, copy=False) if dtype is not None else result


class _LazyMotionSegmentation:
    """Centre labels plus paired transformed correspondence-id maps and d."""

    def __init__(self, base, crop_origin, rows, dilation: int):
        self.base = base
        self.crop_origin = tuple(int(value) for value in crop_origin)
        self.rows = tuple(tuple(int(value) for value in row) for row in rows)
        self.dilation = int(dilation)
        self.shape = (4, *tuple(int(value) for value in base.shape[1:]))
        self.ndim = 4
        self.dtype = np.int16

    @staticmethod
    def _mapped_coordinate(coordinate: int, item, size: int) -> int | None:
        if isinstance(item, (int, np.integer)):
            selected = int(item)
            if selected < 0:
                selected += size
            return 0 if coordinate == selected else None
        start, stop, step = item.indices(size)
        if step != 1:
            raise ValueError("motion target supports unit-step spatial slices only")
        return coordinate - start if start <= coordinate < stop else None

    def _empty_spatial(self, spatial):
        reference = np.asarray(self.base[(0, *spatial)])
        return np.zeros(reference.shape, dtype=self.dtype)

    def _id_map(self, spatial, current: bool):
        result = self._empty_spatial(spatial)
        offset = 0 if current else 3
        for correspondence_id, row in enumerate(self.rows, start=1):
            output = []
            selected = True
            for axis, (raw_coord, item, size) in enumerate(
                zip(row[offset : offset + 3], spatial, self.shape[1:])
            ):
                local_coord = int(raw_coord) - self.crop_origin[axis]
                mapped = self._mapped_coordinate(local_coord, item, size)
                if mapped is None:
                    selected = False
                    break
                output.append(mapped)
            if selected:
                result[tuple(output)] = correspondence_id
        return result

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (self.ndim - len(index))
        channel_item, spatial = index[0], index[1:]
        channels = np.atleast_1d(np.arange(4)[channel_item]).tolist()
        arrays = []
        for channel in channels:
            if channel == 0:
                value = np.asarray(self.base[(0, *spatial)], dtype=self.dtype)
            elif channel == 1:
                value = self._id_map(spatial, current=True)
            elif channel == 2:
                value = self._id_map(spatial, current=False)
            else:
                value = np.full(self._empty_spatial(spatial).shape, self.dilation, dtype=self.dtype)
            arrays.append(value[None])
        result = np.concatenate(arrays, axis=0)
        return result[0] if isinstance(channel_item, (int, np.integer)) else result

    def __array__(self, dtype=None):
        result = self[(slice(None),) * self.ndim]
        return result.astype(dtype, copy=False) if dtype is not None else result


class BiohubTemporalDilatedMotionDataset(BiohubTemporalThreeFrameDataset):
    """Random d=1/2 temporal inputs with sparse motion correspondences."""

    dilation_two_probability = 0.25
    segmentation_class = _LazyMotionSegmentation

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        lookup_path = os.environ.get("BIOHUB_BACKWARD_MOTION_LOOKUP", "").strip()
        if not lookup_path:
            raise RuntimeError("BIOHUB_BACKWARD_MOTION_LOOKUP is required")
        payload = json.loads(Path(lookup_path).read_text())
        self.motion_cases = payload.get("cases", payload)

    def _select_dilation(self) -> int:
        return 2 if np.random.random() < self.dilation_two_probability else 1

    def load_case(self, identifier):
        center_data, center_seg, seg_prev, properties = nnUNetDatasetBlosc2.load_case(
            self, identifier
        )
        dilation = self._select_dilation()
        temporal_data = []
        temporal_origins = []
        resolved_cases = []
        replicated = []
        for offset in (-dilation, 0, dilation):
            if offset == 0:
                neighbor_id, was_replicated = identifier, False
                data, neighbor_properties = center_data, properties
            else:
                neighbor_id, was_replicated = self._resolved_neighbor(identifier, offset)
                data, _, _, neighbor_properties = nnUNetDatasetBlosc2.load_case(
                    self, neighbor_id
                )
            temporal_data.append(data)
            temporal_origins.append(self._crop_origin(data, neighbor_properties))
            resolved_cases.append(neighbor_id)
            replicated.append(was_replicated)

        center_origin = self._crop_origin(center_data, properties)
        rows = self.motion_cases.get(identifier, {}).get(str(dilation), ())
        # Replicated/missing past input is unknown motion, not zero motion.
        if replicated[0]:
            rows = ()
        properties = dict(properties)
        properties["biohub_temporal_case_ids"] = resolved_cases
        properties["biohub_temporal_replicated"] = replicated
        properties["biohub_temporal_dilation"] = dilation
        properties["biohub_motion_correspondences"] = len(rows)
        temporal = _LazyTemporalChannelStack(temporal_data, temporal_origins)
        segmentation = self.segmentation_class(
            center_seg, center_origin, rows, dilation
        )
        return _TemporalConditionedStack(temporal, dilation), segmentation, seg_prev, properties


class BiohubTemporalDilatedMotionValidationDataset(BiohubTemporalDilatedMotionDataset):
    dilation_two_probability = 0.0


class JointHeatmapBackwardMotionLoss(nn.Module):
    """Heatmap loss plus transformed correspondence-based SmoothL1 motion."""

    def __init__(
        self,
        heatmap_loss: MaskedGaussianHeatmapLoss,
        input_spacing_zyx_um,
        output_spacing_zyx_um,
        motion_scale_um: float = 8.0,
        motion_radius_um: float = 2.0,
        motion_sigma_um: float = 1.0,
        motion_lambda: float = 0.1,
    ):
        super().__init__()
        self.heatmap_loss = heatmap_loss
        self.input_spacing = tuple(float(value) for value in input_spacing_zyx_um)
        self.output_spacing = tuple(float(value) for value in output_spacing_zyx_um)
        self.motion_scale_um = float(motion_scale_um)
        self.motion_radius_um = float(motion_radius_um)
        self.motion_sigma_um = float(motion_sigma_um)
        self.motion_lambda = float(motion_lambda)
        self.motion_lambda_scale = 0.0
        self.last_heatmap_loss = float("nan")
        self.last_motion_loss = float("nan")
        self.last_motion_voxels = 0

    def build_target(self, target):
        return self.heatmap_loss.build_target(target[:, :1])

    def sample_statistics(self, logits, target):
        return self.heatmap_loss.sample_statistics(logits[:, :2], target[:, :1])

    @staticmethod
    def _ancestor_coordinates(
        ancestor_ids: torch.Tensor, correspondence_id: int
    ) -> torch.Tensor:
        return torch.nonzero(
            ancestor_ids == correspondence_id, as_tuple=False
        )

    def _motion_target(self, target: torch.Tensor, output_shape, device):
        batch = target.shape[0]
        vector = torch.zeros((batch, 3, *output_shape), device=device, dtype=torch.float32)
        weight = torch.zeros((batch, 1, *output_shape), device=device, dtype=torch.float32)
        best_distance = torch.full(
            (batch, 1, *output_shape), float("inf"), device=device, dtype=torch.float32
        )
        input_shape = np.asarray(target.shape[-3:], dtype=np.float64)
        output_shape_array = np.asarray(output_shape, dtype=np.float64)
        scale_to_output = output_shape_array / input_shape
        out_spacing = np.asarray(self.output_spacing, dtype=np.float64)
        radii = np.maximum(1, np.ceil(self.motion_radius_um / out_spacing).astype(int))

        for batch_index in range(batch):
            current_ids = target[batch_index, 1].detach()
            ancestor_ids = target[batch_index, 2].detach()
            d_values = target[batch_index, 3]
            positive_d = d_values[d_values > 0]
            if positive_d.numel() == 0:
                continue
            dilation = max(1, int(torch.median(positive_d.float()).item()))
            ids = torch.unique(current_ids[current_ids > 0]).tolist()
            for correspondence_id in ids:
                current = torch.nonzero(current_ids == correspondence_id, as_tuple=False)
                ancestor = self._ancestor_coordinates(
                    ancestor_ids, int(correspondence_id)
                )
                if current.numel() == 0 or ancestor.numel() == 0:
                    continue
                current_full = current.float().mean(0)
                ancestor_full = ancestor.float().mean(0)
                velocity_um = (
                    (ancestor_full - current_full)
                    * torch.as_tensor(self.input_spacing, device=device)
                    / float(dilation)
                )
                current_output = current_full.cpu().numpy() * scale_to_output
                center = np.rint(current_output).astype(int)
                lower = np.maximum(center - radii, 0)
                upper = np.minimum(center + radii + 1, output_shape_array.astype(int))
                if np.any(upper <= lower):
                    continue
                zz, yy, xx = torch.meshgrid(
                    *[
                        torch.arange(int(lo), int(hi), device=device, dtype=torch.float32)
                        for lo, hi in zip(lower, upper)
                    ],
                    indexing="ij",
                )
                coords = torch.stack((zz, yy, xx))
                center_tensor = torch.as_tensor(current_output, device=device).view(3, 1, 1, 1)
                spacing_tensor = torch.as_tensor(self.output_spacing, device=device).view(3, 1, 1, 1)
                distance = torch.sqrt(((coords - center_tensor) * spacing_tensor).square().sum(0))
                local_valid = distance <= self.motion_radius_um
                slices = (
                    batch_index,
                    slice(None),
                    slice(int(lower[0]), int(upper[0])),
                    slice(int(lower[1]), int(upper[1])),
                    slice(int(lower[2]), int(upper[2])),
                )
                local_best = best_distance[slices]
                replace = local_valid.unsqueeze(0) & (distance.unsqueeze(0) < local_best)
                local_weight = torch.exp(
                    -distance.square() / (2 * self.motion_sigma_um**2)
                ).unsqueeze(0)
                best_distance[slices] = torch.where(replace, distance.unsqueeze(0), local_best)
                weight[slices] = torch.where(replace, local_weight, weight[slices])
                normalized_velocity = velocity_um / self.motion_scale_um
                vector[slices] = torch.where(
                    replace.expand(3, -1, -1, -1),
                    normalized_velocity.view(3, 1, 1, 1),
                    vector[slices],
                )
        return vector, weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 5:
            raise ValueError(f"expected 5 output channels (2 heatmap + 3 motion), got {logits.shape}")
        heatmap = self.heatmap_loss(logits[:, :2], target[:, :1])
        motion_target, motion_weight = self._motion_target(
            target, tuple(int(value) for value in logits.shape[-3:]), logits.device
        )
        difference = torch.nn.functional.smooth_l1_loss(
            logits[:, 2:5].float(), motion_target, reduction="none", beta=0.25
        )
        denominator = motion_weight.sum().clamp_min(1.0) * 3
        motion = (difference * motion_weight).sum() / denominator
        self.last_heatmap_loss = float(heatmap.detach())
        self.last_motion_loss = float(motion.detach())
        self.last_motion_voxels = int((motion_weight > 0).sum().detach())
        return heatmap + self.motion_lambda * self.motion_lambda_scale * motion


class nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Random-initialized four-input, five-output joint detector/motion model."""

    dataset_class = BiohubTemporalDilatedMotionDataset

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataset_class = BiohubTemporalDilatedMotionDataset

    def initialize(self):
        super().initialize()
        self.dataset_class = BiohubTemporalDilatedMotionDataset

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        common = dict(
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        return (
            BiohubTemporalDilatedMotionDataset(
                self.preprocessed_dataset_folder, tr_keys, **common
            ),
            BiohubTemporalDilatedMotionValidationDataset(
                self.preprocessed_dataset_folder, val_keys, **common
            ),
        )

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf.build_network_architecture(
            plans_manager, configuration_manager, 4, 5, enable_deep_supervision
        )

    def _build_loss(self):
        input_spacing = tuple(float(value) for value in self.configuration_manager.spacing)
        output_spacing = tuple(
            spacing * stride
            for spacing, stride in zip(input_spacing, self.output_stride_zyx)
        )
        target_composition = os.environ.get("BIOHUB_SELECTED_HEATMAP_COMPOSITION", "sum_clip")
        loss_mode = os.environ.get("BIOHUB_SELECTED_HEATMAP_LOSS", "mse")
        center_weight = float(os.environ.get("BIOHUB_SELECTED_CENTER_WEIGHT", "1"))
        heatmap = MaskedGaussianHeatmapLoss(
            spacing_zyx_um=output_spacing,
            sigma_um=float(os.environ.get("BIOHUB_HEATMAP_SIGMA_UM", "2.0")),
            truncate_sigma=float(os.environ.get("BIOHUB_HEATMAP_TRUNCATE_SIGMA", "3.0")),
            mode=loss_mode,
            ignore_label=int(self.label_manager.ignore_label),
            target_stride_zyx=self.output_stride_zyx,
            valid_radius_um=float(os.environ.get("BIOHUB_VALID_RADIUS_UM", "6.0")),
            target_composition=target_composition,
            center_weight=center_weight,
            center_radius_um=float(os.environ.get("BIOHUB_HEATMAP_CENTER_RADIUS_UM", "1.5")),
        )
        return JointHeatmapBackwardMotionLoss(
            heatmap,
            input_spacing,
            output_spacing,
            motion_scale_um=float(os.environ.get("BIOHUB_MOTION_SCALE_UM", "8.0")),
            motion_radius_um=float(os.environ.get("BIOHUB_MOTION_RADIUS_UM", "2.0")),
            motion_sigma_um=float(os.environ.get("BIOHUB_MOTION_SIGMA_UM", "1.0")),
            motion_lambda=float(os.environ.get("BIOHUB_MOTION_LAMBDA", "0.1")),
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.loss.motion_lambda_scale = min(1.0, max(0.0, (self.current_epoch + 1) / 50.0))

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.loss.motion_lambda_scale = min(1.0, max(0.0, (self.current_epoch + 1) / 50.0))

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)
        self._append_validation_sample_losses(batch, output, target)
        with torch.no_grad():
            gaussian, valid = self.loss.build_target(target)
            probability = torch.sigmoid(
                output[:, 1:2].float() - output[:, 0:1].float()
            )
            predicted = probability.gt(0.5) & valid
            expected = gaussian.gt(0.5) & valid
            axes = (0, 2, 3, 4)
            tp = (predicted & expected).sum(axes).detach().cpu().numpy().astype(np.float64)
            fp = (predicted & ~expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
            fn = (~predicted & expected & valid).sum(axes).detach().cpu().numpy().astype(np.float64)
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}

    def perform_actual_validation(self, *args, **kwargs):
        """Skip nnU-Net's categorical exporter for the joint 5-channel head.

        The stock predictor allocates its sliding-window accumulator from the
        two-class label manager, while this network intentionally emits two
        detector logits followed by three backward-motion regression channels.
        Full-volume detector and motion validation are performed by the
        dedicated Biohub evaluators, which split ``[:2]`` and ``[2:5]``.
        """
        self.print_to_log_file(
            "Skipping stock nnU-Net actual validation: joint 2-channel heatmap "
            "+ 3-channel motion output requires the dedicated Biohub evaluators."
        )
        return None


class nnUNetTrainerBiohubTemporalDilatedNoMotionXYHalf(
    nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalf
):
    """Matched T3-dilate architecture trained with heatmap loss only.

    The five-channel output architecture and temporal dataset remain identical
    to the motion-backprop arm. The launcher fixes ``BIOHUB_MOTION_LAMBDA=0``,
    so motion channels receive no loss and motion gradients cannot shape the
    shared backbone.
    """


class nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalfIter250(
    nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalf
):
    """Separate results namespace for the batch-2, 250-iteration rerun.

    Training behavior is otherwise identical to the original temporal
    dilated backward-motion trainer. The dedicated class name prevents the
    scratch rerun from reading or overwriting the completed 125-iteration
    checkpoint lineage.
    """
