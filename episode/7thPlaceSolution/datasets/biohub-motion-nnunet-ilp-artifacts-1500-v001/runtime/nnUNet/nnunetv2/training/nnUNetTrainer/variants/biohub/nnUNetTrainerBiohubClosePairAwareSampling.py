"""Pair-aware crop sampling for the Biohub single-frame detector."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubMaskedHeatmap import (
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)


BIOHUB_CLOSE_PAIR_CLASS_KEY = 999


def choose_pair_aware_indices(
    base_indices,
    pair_positive_identifiers,
    target_fraction: float,
    random_state=np.random,
):
    """Replace independent minibatch slots with close-pair cases."""
    selected = list(base_indices)
    targeted = random_state.random(len(selected)) < float(target_fraction)
    for sample_index in np.flatnonzero(targeted):
        selected[int(sample_index)] = random_state.choice(pair_positive_identifiers)
    return selected, targeted.tolist()


class BiohubClosePairAwareDataset:
    """Attach train-only eligible-pair midpoints to preprocessed case metadata."""

    def __init__(
        self,
        base_dataset,
    ):
        self.base_dataset = base_dataset
        self.identifiers = base_dataset.identifiers
        lookup_path = os.environ.get("BIOHUB_CLOSE_PAIR_SAMPLING_LOOKUP")
        if not lookup_path:
            raise RuntimeError("BIOHUB_CLOSE_PAIR_SAMPLING_LOOKUP is required")
        payload = json.loads(Path(lookup_path).read_text())
        lookup_fold = int(payload.get("fold", -1))
        requested_fold = int(os.environ.get("BIOHUB_NNUNET_FOLD", "0"))
        if lookup_fold != requested_fold:
            raise ValueError(
                f"close-pair lookup fold {lookup_fold} does not match requested fold "
                f"{requested_fold}"
            )
        lookup_cases = payload.get("cases", {})
        identifier_set = set(self.identifiers)
        self._pair_midpoints = {
            identifier: rows
            for identifier, rows in lookup_cases.items()
            if identifier in identifier_set and rows
        }
        self.pair_positive_identifiers = tuple(sorted(self._pair_midpoints))
        if not self.pair_positive_identifiers:
            raise RuntimeError("training split contains no eligible close-pair cases")

    def load_case(self, identifier):
        data, seg, seg_prev, properties = self.base_dataset.load_case(identifier)
        midpoint_rows = self._pair_midpoints.get(identifier)
        if midpoint_rows:
            properties = dict(properties)
            class_locations = dict(properties["class_locations"])
            class_locations[BIOHUB_CLOSE_PAIR_CLASS_KEY] = np.asarray(
                [[0, *row["midpoint_zyx"]] for row in midpoint_rows],
                dtype=np.int64,
            )
            properties["class_locations"] = class_locations
            properties["biohub_close_pair_count"] = len(midpoint_rows)
        return data, seg, seg_prev, properties

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)


class BiohubClosePairAwareDataLoader(nnUNetDataLoader):
    """Use pair-midpoint crops for a controlled fraction of train samples."""

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.pair_positive_identifiers = tuple(data.pair_positive_identifiers)
        self.target_fraction = float(
            os.environ.get("BIOHUB_CLOSE_PAIR_SAMPLING_FRACTION", "0.10")
        )
        if not 0 < self.target_fraction < 1:
            raise ValueError(
                "BIOHUB_CLOSE_PAIR_SAMPLING_FRACTION must be in (0, 1), got "
                f"{self.target_fraction}"
            )
        self._pair_target_queue: list[bool] = []

    def get_indices(self):
        selected, targeted = choose_pair_aware_indices(
            super().get_indices(),
            self.pair_positive_identifiers,
            self.target_fraction,
        )
        self._pair_target_queue = targeted
        return selected

    def get_bbox(
        self,
        data_shape,
        force_fg,
        class_locations,
        overwrite_class=None,
        verbose=False,
    ):
        pair_targeted = bool(self._pair_target_queue.pop(0)) if self._pair_target_queue else False
        if pair_targeted:
            if BIOHUB_CLOSE_PAIR_CLASS_KEY not in class_locations:
                raise KeyError("targeted close-pair sample has no midpoint class locations")
            return super().get_bbox(
                data_shape,
                True,
                class_locations,
                overwrite_class=BIOHUB_CLOSE_PAIR_CLASS_KEY,
                verbose=verbose,
            )
        # The synthetic midpoint key must never compete with ordinary foreground
        # locations in standard samples.
        standard_locations = dict(class_locations)
        standard_locations.pop(BIOHUB_CLOSE_PAIR_CLASS_KEY, None)
        return super().get_bbox(
            data_shape,
            force_fg,
            standard_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


class nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalfPairAwareSampling(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Standard masked MSE with train-only close-pair midpoint crop sampling."""

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        base_dataset_tr = self.dataset_class(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        dataset_tr = BiohubClosePairAwareDataset(base_dataset_tr)
        dataset_val = self.dataset_class(
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
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = BiohubClosePairAwareDataLoader(
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
