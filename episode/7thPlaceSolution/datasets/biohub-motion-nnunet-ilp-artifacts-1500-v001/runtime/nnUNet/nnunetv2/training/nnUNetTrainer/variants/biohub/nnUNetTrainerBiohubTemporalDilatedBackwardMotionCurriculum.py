"""Corrected Biohub backward-motion trainer with an epoch curriculum."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunetv2.training.nnUNetTrainer.variants.biohub.biohub_temporal_dilation_curriculum import (
    SharedEpochDilationSampler,
    curriculum_probabilities,
)
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalDilatedBackwardMotion import (
    BiohubTemporalDilatedMotionDataset,
    JointHeatmapBackwardMotionLoss,
    _LazyMotionSegmentation,
    nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalf,
)
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalThreeFrame import (
    nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)


PACK_BASE = 64
PACK_OFFSET = PACK_BASE * PACK_BASE
MARKER_OFFSETS = (
    (0, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


class _PackedRadiusOneMotionSegmentation(_LazyMotionSegmentation):
    """Collision-safe shared-parent IDs with a one-pixel XY marker support."""

    @staticmethod
    def _packed_value(correspondence_ids: list[int]) -> int:
        if len(correspondence_ids) == 1:
            return correspondence_ids[0]
        if len(correspondence_ids) != 2:
            raise RuntimeError(
                "packed backward-motion target supports at most two daughters "
                f"per ancestor, got {len(correspondence_ids)}"
            )
        first, second = sorted(correspondence_ids)
        if first >= PACK_BASE or second >= PACK_BASE:
            raise RuntimeError(
                f"correspondence IDs must be below {PACK_BASE}, got {first}, {second}"
            )
        return PACK_OFFSET + first * PACK_BASE + second

    def _mapped_point(self, coordinate, spatial):
        output = []
        for axis, (raw_coord, item, size) in enumerate(
            zip(coordinate, spatial, self.shape[1:])
        ):
            local_coord = int(raw_coord) - self.crop_origin[axis]
            mapped = self._mapped_coordinate(local_coord, item, size)
            if mapped is None:
                return None
            output.append(mapped)
        return tuple(output)

    @staticmethod
    def _paint_nearest(
        result: np.ndarray,
        best_distance: np.ndarray,
        center: tuple[int, int, int],
        value: int,
    ) -> None:
        if result.ndim != 3:
            raise ValueError(
                f"packed motion marker expects a 3D spatial slice, got {result.shape}"
            )
        for dz, dy, dx in MARKER_OFFSETS:
            point = (center[0] + dz, center[1] + dy, center[2] + dx)
            if not all(0 <= point[axis] < result.shape[axis] for axis in range(3)):
                continue
            distance = dz * dz + dy * dy + dx * dx
            existing_distance = int(best_distance[point])
            existing_value = int(result[point])
            if distance < existing_distance or (
                distance == existing_distance
                and (existing_value == 0 or value < existing_value)
            ):
                result[point] = value
                best_distance[point] = distance

    def _id_map(self, spatial, current: bool):
        result = self._empty_spatial(spatial)
        best_distance = np.full(result.shape, 127, dtype=np.int8)
        if current:
            groups = [
                (row[:3], [correspondence_id])
                for correspondence_id, row in enumerate(self.rows, start=1)
            ]
        else:
            grouped: dict[tuple[int, int, int], list[int]] = {}
            for correspondence_id, row in enumerate(self.rows, start=1):
                grouped.setdefault(tuple(row[3:6]), []).append(correspondence_id)
            groups = sorted(grouped.items())

        for coordinate, correspondence_ids in groups:
            mapped = self._mapped_point(coordinate, spatial)
            if mapped is None:
                continue
            value = (
                correspondence_ids[0]
                if current
                else self._packed_value(correspondence_ids)
            )
            self._paint_nearest(result, best_distance, mapped, value)
        return result


class _LazyEpochTaggedSegmentation:
    """Optional fifth target channel used only by the production loader smoke."""

    def __init__(self, base, epoch: int):
        self.base = base
        self.epoch = int(epoch)
        self.shape = (int(base.shape[0]) + 1, *tuple(base.shape[1:]))
        self.ndim = int(base.ndim)
        self.dtype = np.int16

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (self.ndim - len(index))
        channel_item, spatial = index[0], index[1:]
        channels = np.atleast_1d(np.arange(self.shape[0])[channel_item]).tolist()
        arrays = []
        for channel in channels:
            if channel < self.base.shape[0]:
                value = np.asarray(self.base[(channel, *spatial)])
            else:
                reference = np.asarray(self.base[(0, *spatial)])
                value = np.full(
                    reference.shape, self.epoch + 1, dtype=self.dtype
                )
            arrays.append(value[None])
        result = np.concatenate(arrays, axis=0)
        return (
            result[0]
            if isinstance(channel_item, (int, np.integer))
            else result
        )

    def __array__(self, dtype=None):
        result = self[(slice(None),) * self.ndim]
        return result.astype(dtype, copy=False) if dtype is not None else result


class BiohubTemporalDilatedMotionCurriculumDataset(
    BiohubTemporalDilatedMotionDataset
):
    """Corrected d1/d2/d3 dataset controlled by a shared zero-based epoch."""

    segmentation_class = _PackedRadiusOneMotionSegmentation

    def __init__(
        self,
        folder: str,
        identifiers=None,
        folder_with_segs_from_previous_stage=None,
        initial_epoch: int = 0,
    ):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.dilation_sampler = SharedEpochDilationSampler(initial_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.dilation_sampler.set_epoch(epoch)

    def get_epoch(self) -> int:
        return self.dilation_sampler.get_epoch()

    def _select_dilation(self) -> int:
        return self.dilation_sampler.sample()

    def load_case(self, identifier):
        data, segmentation, seg_prev, properties = super().load_case(identifier)
        properties = dict(properties)
        properties["biohub_curriculum_epoch"] = self.get_epoch()
        if os.environ.get(
            "BIOHUB_CURRICULUM_DEBUG_EPOCH_CHANNEL", "0"
        ).lower() in {"1", "true", "yes"}:
            segmentation = _LazyEpochTaggedSegmentation(
                segmentation, self.get_epoch()
            )
        return data, segmentation, seg_prev, properties


class BiohubTemporalDilatedMotionFixedD1Dataset(
    BiohubTemporalDilatedMotionCurriculumDataset
):
    """Deterministic d1 validation view."""

    def _select_dilation(self) -> int:
        return 1


class _RestoreDilationConditionTransform(BasicTransform):
    """Restore semantic condition channel after all intensity transforms."""

    def get_parameters(self, **data_dict) -> dict:
        segmentation = data_dict.get("segmentation")
        if segmentation is None or int(segmentation.shape[0]) < 4:
            raise RuntimeError(
                "dilation condition restore requires four-channel segmentation"
            )
        dilation_layer = segmentation[3]
        if isinstance(dilation_layer, torch.Tensor):
            positive = dilation_layer[dilation_layer > 0]
            if positive.numel() == 0:
                raise RuntimeError("transformed target contains no positive dilation")
            dilation = int(torch.median(positive.float()).item())
        else:
            positive = np.asarray(dilation_layer)[np.asarray(dilation_layer) > 0]
            if positive.size == 0:
                raise RuntimeError("transformed target contains no positive dilation")
            dilation = int(np.median(positive))
        if dilation not in (1, 2, 3):
            raise RuntimeError(f"unexpected dilation {dilation}")
        return {"condition": dilation / 2.0}

    def _apply_to_image(self, image, *, condition: float):
        image[3].fill_(condition)
        return image

    def _apply_to_segmentation(self, segmentation, **params):
        return segmentation


def _share_gaussian_noise_field(transform) -> None:
    """Make Gaussian-noise pixels identical across the temporal channels."""
    if type(transform).__name__ == "GaussianNoiseTransform":

        def shared_noise(self, image: torch.Tensor, **params) -> torch.Tensor:
            if params["num_apply"] == 0 or params["sigmas"] is None:
                return image
            noise = torch.empty(
                image.shape[1:], device=image.device, dtype=image.dtype
            ).normal_(mean=0.0, std=float(params["sigmas"]))
            image[params["apply_idx"]] += noise
            return image

        transform._apply_to_image = MethodType(shared_noise, transform)
    if hasattr(transform, "transform"):
        _share_gaussian_noise_field(transform.transform)
    if hasattr(transform, "transforms"):
        for child in transform.transforms:
            _share_gaussian_noise_field(child)


class PackedJointHeatmapBackwardMotionLoss(JointHeatmapBackwardMotionLoss):
    """Decode one or two correspondence IDs from the ancestor label layer."""

    @staticmethod
    def _ancestor_coordinates(
        ancestor_ids: torch.Tensor, correspondence_id: int
    ) -> torch.Tensor:
        packed = ancestor_ids >= PACK_OFFSET
        code = torch.where(packed, ancestor_ids - PACK_OFFSET, 0)
        first = torch.div(code, PACK_BASE, rounding_mode="floor")
        second = torch.remainder(code, PACK_BASE)
        matches = (ancestor_ids == correspondence_id) | (
            packed & ((first == correspondence_id) | (second == correspondence_id))
        )
        return torch.nonzero(matches, as_tuple=False)


class nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250(
    nnUNetTrainerBiohubTemporalDilatedBackwardMotionXYHalf
):
    """Corrected four-input/five-output curriculum trainer for direct run C."""

    dataset_class = BiohubTemporalDilatedMotionCurriculumDataset
    milestone_epochs = frozenset((100, 300, 700))

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device=torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataset_class = BiohubTemporalDilatedMotionCurriculumDataset
        self._curriculum_train_dataset = None
        self._active_curriculum_probabilities = None
        self._epoch_dilation_counts: Counter[int] = Counter()
        self.num_iterations_per_epoch = int(
            os.environ.get("BIOHUB_NNUNET_ITERS_PER_EPOCH", "250")
        )

    def initialize(self):
        super().initialize()
        self.dataset_class = BiohubTemporalDilatedMotionCurriculumDataset

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        temporal_cls = (
            nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
        )
        temporal_cls._synchronize_channel_transforms(transforms)
        _share_gaussian_noise_field(transforms)
        if not hasattr(transforms, "transforms"):
            raise TypeError(
                "expected ComposeTransforms so condition restore can run last"
            )
        transforms.transforms.append(_RestoreDilationConditionTransform())
        return transforms

    @classmethod
    def get_validation_transforms(cls, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        if not hasattr(transforms, "transforms"):
            raise TypeError(
                "expected ComposeTransforms so condition restore can run last"
            )
        transforms.transforms.append(_RestoreDilationConditionTransform())
        return transforms

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        common = dict(
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        train_dataset = BiohubTemporalDilatedMotionCurriculumDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            initial_epoch=int(self.current_epoch),
            **common,
        )
        validation_dataset = BiohubTemporalDilatedMotionFixedD1Dataset(
            self.preprocessed_dataset_folder,
            val_keys,
            initial_epoch=0,
            **common,
        )
        self._curriculum_train_dataset = train_dataset
        self._active_curriculum_probabilities = curriculum_probabilities(
            int(self.current_epoch)
        ).as_tuple()
        return train_dataset, validation_dataset

    def _build_loss(self):
        original = super()._build_loss()
        return PackedJointHeatmapBackwardMotionLoss(
            original.heatmap_loss,
            original.input_spacing,
            original.output_spacing,
            motion_scale_um=original.motion_scale_um,
            motion_radius_um=original.motion_radius_um,
            motion_sigma_um=original.motion_sigma_um,
            motion_lambda=original.motion_lambda,
        )

    @staticmethod
    def _finish_augmenter(augmenter) -> None:
        if augmenter is not None and hasattr(augmenter, "_finish"):
            augmenter._finish()

    def apply_curriculum_epoch(
        self, epoch: int, *, rebuild_on_phase_change: bool = True
    ) -> bool:
        """Update worker-visible epoch and optionally flush a changed phase."""
        epoch = int(epoch)
        desired = curriculum_probabilities(epoch).as_tuple()
        phase_changed = (
            self._active_curriculum_probabilities is not None
            and desired != self._active_curriculum_probabilities
        )
        self.current_epoch = epoch
        if (
            phase_changed
            and rebuild_on_phase_change
            and hasattr(self, "dataloader_train")
        ):
            self.print_to_log_file(
                "Biohub dilation curriculum phase change: "
                f"{self._active_curriculum_probabilities} -> {desired}; "
                "rebuilding augmenters to flush prefetched batches"
            )
            self._finish_augmenter(self.dataloader_train)
            self._finish_augmenter(self.dataloader_val)
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        elif self._curriculum_train_dataset is not None:
            self._curriculum_train_dataset.set_epoch(epoch)
            self._active_curriculum_probabilities = desired
        return phase_changed

    def on_train_epoch_start(self):
        self.apply_curriculum_epoch(
            int(self.current_epoch), rebuild_on_phase_change=True
        )
        self._epoch_dilation_counts.clear()
        super().on_train_epoch_start()

    def train_step(self, batch: dict) -> dict:
        condition = batch["data"][:, 3, 0, 0, 0].detach().cpu()
        for value in torch.round(condition * 2).to(torch.int64).tolist():
            self._epoch_dilation_counts[int(value)] += 1
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        self.print_to_log_file(
            "Biohub observed dilation samples: "
            + json.dumps(
                {
                    f"d{dilation}": int(self._epoch_dilation_counts[dilation])
                    for dilation in (1, 2, 3)
                },
                sort_keys=True,
            )
        )
        return super().on_train_epoch_end(train_outputs)

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        completed_epoch = int(self.current_epoch) + 1
        if os.path.basename(str(filename)) != "checkpoint_latest.pth":
            return
        if completed_epoch not in self.milestone_epochs:
            return
        destination = os.path.join(
            os.path.dirname(str(filename)),
            f"checkpoint_epoch{completed_epoch}.pth",
        )
        temporary = destination + ".tmp"
        shutil.copy2(filename, temporary)
        os.replace(temporary, destination)


class nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250Long1500(
    nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250
):
    """Scratch 1,500-epoch budget-scaling arm with sparse milestones."""

    milestone_epochs = frozenset((1000, 1250))
