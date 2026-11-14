"""Three-frame input with independent cell/parent/daughter output logits."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubMaskedHeatmap import (
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalThreeFrame import (
    BiohubTemporalThreeFrameDataset,
    nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubThreeChannelHeatmap import (
    nnUNetTrainerBiohubThreeChannelHeatmapAnisotropyAwareZFlipXYHalf,
)


def _role_dataset_json(dataset_json: dict) -> dict:
    """Declare all latent role values and use the next contiguous value as ignore.

    nnU-Net requires ``ignore == max(non-ignore labels) + 1``. The dedicated
    three-channel dataset uses 255, but this lazy Dataset701 overlay exposes
    categorical role values 1..6, so its runtime metadata must use ignore 7.
    """
    result = copy.deepcopy(dataset_json)
    result["labels"] = {
        "background": 0,
        "human_normal_cell": 1,
        "human_division_parent": 2,
        "human_division_daughter": 3,
        "human_parent_and_daughter": 4,
        "pseudo_cell_unknown_role": 5,
        "pseudo_local_background": 6,
        "ignore": 7,
    }
    result["description"] = (
        str(result.get("description", ""))
        + "; lazy GEFF parent/daughter role overlay"
    )
    return result


class _LazyDivisionRoleSegmentation:
    """Remap Dataset701 ignore=2 to runtime ignore=7 and overlay roles lazily."""

    def __init__(self, base, properties: dict, global_role_rows):
        self.base = base
        self.shape = tuple(int(v) for v in base.shape)
        self.ndim = len(self.shape)
        self.dtype = np.uint8
        if self.ndim != 4 or self.shape[0] != 1:
            raise ValueError(f"expected one-channel CZYX segmentation, got {self.shape}")
        bbox = properties.get("bbox_used_for_cropping")
        if bbox is None or len(bbox) != 3:
            raise ValueError("division role overlay requires bbox_used_for_cropping")
        origin = tuple(int(axis[0]) for axis in bbox)
        merged = {}
        for z, y, x, role in global_role_rows:
            local = (int(z) - origin[0], int(y) - origin[1], int(x) - origin[2])
            if all(0 <= local[i] < self.shape[i + 1] for i in range(3)):
                merged[local] = int(role)
        self.local_roles = tuple((*coord, role) for coord, role in sorted(merged.items()))

    @staticmethod
    def _coordinate_in_index(coordinate: int, item, size: int):
        if isinstance(item, (int, np.integer)):
            selected = int(item)
            if selected < 0:
                selected += size
            return () if coordinate == selected else None
        start, stop, step = item.indices(size)
        if step != 1:
            raise ValueError("division role overlay only supports unit-step slices")
        if start <= coordinate < stop:
            return (coordinate - start,)
        return None

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (self.ndim - len(index))
        result = np.asarray(self.base[index]).copy()
        result[result == 2] = 7

        channel_item = index[0]
        selected_channels = np.atleast_1d(np.arange(1)[channel_item]).tolist()
        if 0 not in selected_channels:
            return result
        channel_prefix = () if isinstance(channel_item, (int, np.integer)) else (selected_channels.index(0),)
        for z, y, x, role in self.local_roles:
            output_index = list(channel_prefix)
            selected = True
            for coordinate, item, size in zip((z, y, x), index[1:], self.shape[1:]):
                mapped = self._coordinate_in_index(coordinate, item, size)
                if mapped is None:
                    selected = False
                    break
                output_index.extend(mapped)
            if selected:
                result[tuple(output_index)] = role
        return result

    def __array__(self, dtype=None):
        result = self[(slice(None),) * self.ndim]
        return result.astype(dtype, copy=False) if dtype is not None else result


class BiohubTemporalThreeFrameDivisionDataset(BiohubTemporalThreeFrameDataset):
    """Dataset701 temporal images with a GEFF role overlay on centre labels."""

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        lookup_path = os.environ.get("BIOHUB_DIVISION_ROLE_LOOKUP")
        if not lookup_path:
            raise RuntimeError("BIOHUB_DIVISION_ROLE_LOOKUP is required")
        payload = json.loads(Path(lookup_path).read_text())
        self._division_roles = payload.get("cases", payload)

    def load_case(self, identifier):
        data, seg, seg_prev, properties = super().load_case(identifier)
        role_seg = _LazyDivisionRoleSegmentation(
            seg, properties, self._division_roles.get(identifier, ())
        )
        properties = dict(properties)
        # Preprocessing stored the Dataset701 annotated-voxel key for labels
        # (-1, 0, 1). Runtime role metadata declares values 0..6, so nnU-Net's
        # dataloader requests (-1, 0, 1, 2, 3, 4, 5, 6). Re-key the same valid
        # locations and keep the existing (1,) cell locations for foreground
        # oversampling. No coordinate recomputation or image preprocessing is
        # required.
        class_locations = dict(properties.get("class_locations", {}))
        old_annotated_keys = [
            key
            for key in class_locations
            if isinstance(key, tuple) and len(key) > 0 and key[0] == -1
        ]
        if not old_annotated_keys:
            raise KeyError(f"{identifier}: no annotated class_locations key found")
        annotated_locations = class_locations[old_annotated_keys[0]]
        for key in old_annotated_keys:
            class_locations.pop(key)
        class_locations[(-1, 0, 1, 2, 3, 4, 5, 6)] = annotated_locations
        # The source Dataset701 preprocessing only knows the binary cell label,
        # so it cannot provide class locations for the lazily overlaid lineage
        # roles. Add exact local role centres here. This is essential for a
        # foreground crop to be able to target a parent or daughter explicitly.
        role_locations = {2: [], 3: [], 4: []}
        for z, y, x, role in role_seg.local_roles:
            if role in role_locations:
                role_locations[role].append((0, z, y, x))
        for role, locations in role_locations.items():
            if locations:
                class_locations[role] = np.asarray(locations, dtype=np.int64)
        properties["class_locations"] = class_locations
        properties["biohub_division_role_points"] = sum(
            len(locations) for locations in role_locations.values()
        )
        return data, role_seg, seg_prev, properties


class BiohubTemporalThreeFrameRoleFocusedDataset(BiohubTemporalThreeFrameDivisionDataset):
    """Training dataset whose forced foreground in positive cases is a role centre."""

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.role_positive_identifiers = tuple(
            identifier
            for identifier in self.identifiers
            if any(int(row[3]) in (2, 3, 4) for row in self._division_roles.get(identifier, ()))
        )
        if not self.role_positive_identifiers:
            raise RuntimeError("role-focused training split contains no parent/daughter cases")

    def load_case(self, identifier):
        data, seg, seg_prev, properties = super().load_case(identifier)
        if identifier in self.role_positive_identifiers:
            properties = dict(properties)
            class_locations = dict(properties["class_locations"])
            # The role-focused dataloader puts a positive case in the final
            # minibatch slot, which is forced-foreground. Removing the generic
            # cell key ensures get_bbox chooses 2/3/4 rather than a normal cell.
            class_locations.pop(1, None)
            properties["class_locations"] = class_locations
            properties["biohub_role_focused_foreground"] = True
        return data, seg, seg_prev, properties


class BiohubRoleFocusedDataLoader(nnUNetDataLoader):
    """Guarantee one role-positive, role-centred patch in every train batch."""

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.role_positive_identifiers = tuple(data.role_positive_identifiers)

    def get_indices(self):
        selected = list(super().get_indices())
        if selected:
            selected[-1] = np.random.choice(self.role_positive_identifiers)
        return selected

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        return sample_idx == self.batch_size - 1 or super()._oversample_last_XX_percent(sample_idx)


class _BiohubRoleFocusedSamplingMixin:
    """Use focused sampling for training while leaving validation untouched."""

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = BiohubTemporalThreeFrameRoleFocusedDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        dataset_val = BiohubTemporalThreeFrameDivisionDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = BiohubRoleFocusedDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerBiohubTemporal3FrameThreeChannelHeatmapAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Full-network fine-tuning trainer for temporal cell/division roles."""

    output_channel_names = ("cell", "division_parent", "division_daughter")
    dataset_class = BiohubTemporalThreeFrameDivisionDataset

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, _role_dataset_json(dataset_json), device)
        self.dataset_class = BiohubTemporalThreeFrameDivisionDataset

    def initialize(self):
        super().initialize()
        self.dataset_class = BiohubTemporalThreeFrameDivisionDataset

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf.build_network_architecture(
            plans_manager, configuration_manager, 3, 3, enable_deep_supervision
        )

    def _build_loss(self):
        return nnUNetTrainerBiohubThreeChannelHeatmapAnisotropyAwareZFlipXYHalf._build_loss(self)

    def validation_step(self, batch: dict) -> dict:
        return nnUNetTrainerBiohubThreeChannelHeatmapAnisotropyAwareZFlipXYHalf.validation_step(
            self, batch
        )

    def perform_actual_validation(self, *args, **kwargs):
        """Skip nnU-Net's categorical exporter for independent role logits.

        LabelManager sees latent categorical target values 0..6, whereas the
        network intentionally emits only three overlapping sigmoid maps. The
        stock exporter allocates seven channels and cannot represent this
        contract. Epoch validation above remains active; full-volume point and
        role metrics are run by the dedicated Biohub evaluator instead.
        """
        self.print_to_log_file(
            "Skipping stock nnU-Net actual validation: independent "
            "cell/parent/daughter logits require the dedicated Biohub evaluator."
        )
        return None


class nnUNetTrainerBiohubTemporal3FrameThreeChannelHeatmapHeadOnlyAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubTemporal3FrameThreeChannelHeatmapAnisotropyAwareZFlipXYHalf
):
    """Warm role slices while keeping backbone and exact cell slice fixed."""

    def _prepare_head_only_parameters(self):
        if getattr(self, "_biohub_head_only_prepared", False):
            return
        head_parameters = []
        for name, parameter in self.network.named_parameters():
            is_segmentation_head = "decoder.seg_layers" in name
            parameter.requires_grad_(is_segmentation_head)
            if is_segmentation_head:
                if parameter.shape[0] != 3:
                    raise ValueError(f"expected three-output segmentation head: {name} {parameter.shape}")

                def zero_cell_slice(gradient):
                    gradient = gradient.clone()
                    gradient[0].zero_()
                    return gradient

                parameter.register_hook(zero_cell_slice)
                head_parameters.append(parameter)
        if not head_parameters:
            raise RuntimeError("no decoder.seg_layers parameters found for head-only warmup")
        self._biohub_head_only_parameters = head_parameters
        self._biohub_head_only_prepared = True

    def configure_optimizers(self):
        self._prepare_head_only_parameters()
        # No weight decay: even a zero cell-slice gradient must leave that slice exact.
        optimizer = torch.optim.AdamW(
            self._biohub_head_only_parameters,
            lr=self.initial_lr,
            weight_decay=0.0,
            amsgrad=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


class nnUNetTrainerBiohubTemporal3FrameThreeChannelRoleFocusedMSEHeadOnlyAnisotropyAwareZFlipXYHalf(
    _BiohubRoleFocusedSamplingMixin,
    nnUNetTrainerBiohubTemporal3FrameThreeChannelHeatmapHeadOnlyAnisotropyAwareZFlipXYHalf,
):
    """Sampler-only ablation: one role-centred patch/batch with weighted MSE."""


class nnUNetTrainerBiohubTemporal3FrameThreeChannelRoleFocusedBCEHeadOnlyAnisotropyAwareZFlipXYHalf(
    _BiohubRoleFocusedSamplingMixin,
    nnUNetTrainerBiohubTemporal3FrameThreeChannelHeatmapHeadOnlyAnisotropyAwareZFlipXYHalf,
):
    """Focused sampler plus cell-local weighted BCE role supervision."""
