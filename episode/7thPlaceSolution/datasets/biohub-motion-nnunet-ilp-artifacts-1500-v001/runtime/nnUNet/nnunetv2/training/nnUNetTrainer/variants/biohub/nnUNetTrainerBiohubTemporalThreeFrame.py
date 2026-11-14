"""Three-frame temporal-input variant of the Biohub XY-half detector.

The network receives ``[t-1, t, t+1]`` as three input channels and predicts
the existing sparse cell-centre target for the centre frame only.  Adjacent
preprocessed cases are opened lazily so a training crop reads only the same
spatial patch from each frame; no second 3-channel preprocessed dataset is
required.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubMaskedHeatmap import (
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf,
)


CASE_FRAME_RE = re.compile(r"^(?P<stem>.+)_t(?P<frame>\d+)$")


def temporal_neighbor_identifier(identifier: str, offset: int) -> str | None:
    """Return the zero-padded adjacent case identifier, if parsable."""

    match = CASE_FRAME_RE.match(identifier)
    if match is None:
        return None
    frame_text = match.group("frame")
    frame = int(frame_text) + int(offset)
    if frame < 0:
        return None
    return f"{match.group('stem')}_t{frame:0{len(frame_text)}d}"


class _LazyTemporalChannelStack:
    """CZYX-like view that concatenates aligned one-channel temporal arrays."""

    def __init__(self, arrays, crop_origins):
        self.arrays = tuple(arrays)
        self.crop_origins = tuple(tuple(int(v) for v in origin) for origin in crop_origins)
        if not self.arrays or len(self.arrays) % 2 != 1:
            raise ValueError(f"expected an odd non-zero number of temporal arrays, got {len(self.arrays)}")
        self.center_index = len(self.arrays) // 2
        shapes = [tuple(int(value) for value in array.shape) for array in self.arrays]
        if any(shape[0] != 1 or len(shape) != 4 for shape in shapes):
            raise ValueError(f"temporal arrays must have one-channel CZYX shape, got {shapes}")
        if len(self.crop_origins) != len(self.arrays) or any(len(origin) != 3 for origin in self.crop_origins):
            raise ValueError(f"expected one ZYX crop origin per temporal array, got {self.crop_origins}")
        self.shape = (len(self.arrays), *shapes[self.center_index][1:])
        self.ndim = len(self.shape)
        self.dtype = self.arrays[self.center_index].dtype

    def _aligned_crop(self, temporal_index: int, spatial_index) -> np.ndarray:
        array = self.arrays[temporal_index]
        center_origin = self.crop_origins[self.center_index]
        source_origin = self.crop_origins[temporal_index]
        source_shape = tuple(int(v) for v in array.shape[1:])
        source_slices = []
        padding = [(0, 0)]
        squeeze_axes = []
        output_shape = []

        for axis, (item, center_size, source_size) in enumerate(
            zip(spatial_index, self.shape[1:], source_shape)
        ):
            is_integer = isinstance(item, (int, np.integer))
            if is_integer:
                position = int(item)
                if position < 0:
                    position += center_size
                start, stop = position, position + 1
                squeeze_axes.append(axis + 1)
            else:
                start, stop, step = item.indices(center_size)
                if step != 1:
                    raise ValueError("temporal lazy stack only supports unit-step spatial slices")
            length = max(0, stop - start)
            output_shape.append(length)
            offset = center_origin[axis] - source_origin[axis]
            source_start, source_stop = start + offset, stop + offset
            valid_start, valid_stop = max(0, source_start), min(source_size, source_stop)
            pad_before = max(0, -source_start)
            pad_after = max(0, source_stop - source_size)
            source_slices.append(slice(valid_start, max(valid_start, valid_stop)))
            padding.append((pad_before, pad_after))

        if any(s.stop <= s.start for s in source_slices):
            result = np.zeros((1, *output_shape), dtype=self.dtype)
        else:
            result = np.asarray(array[(slice(None), *source_slices)])
            if any(before or after for before, after in padding):
                result = np.pad(result, padding, mode="constant", constant_values=0)
        if tuple(result.shape[1:]) != tuple(output_shape):
            raise RuntimeError(
                f"aligned temporal crop shape mismatch: got {result.shape}, expected {(1, *output_shape)}"
            )
        for axis in reversed(squeeze_axes):
            result = np.squeeze(result, axis=axis)
        return result

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        index = index + (slice(None),) * (self.ndim - len(index))
        channel_index, spatial_index = index[0], index[1:]
        selected = np.arange(len(self.arrays))[channel_index]
        selected = np.atleast_1d(selected).tolist()
        cropped = [self._aligned_crop(int(i), spatial_index) for i in selected]
        result = np.concatenate(cropped, axis=0)
        if isinstance(channel_index, (int, np.integer)):
            return result[0]
        return result

    def __array__(self, dtype=None):
        result = self[(slice(None),) * self.ndim]
        return result.astype(dtype, copy=False) if dtype is not None else result


class BiohubTemporalThreeFrameDataset(nnUNetDatasetBlosc2):
    """Load centre labels plus lazy previous/centre/next image channels."""

    temporal_offsets = (-1, 0, 1)

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self._available_cases = {
            path.name[:-5]
            for path in Path(self.source_folder).glob("*.b2nd")
            if not path.name.endswith("_seg.b2nd")
        }

    def _resolved_neighbor(self, identifier: str, offset: int) -> tuple[str, bool]:
        candidate = temporal_neighbor_identifier(identifier, offset)
        if candidate is None or candidate not in self._available_cases:
            return identifier, True
        return candidate, False

    @staticmethod
    def _crop_origin(data, properties) -> tuple[int, int, int]:
        bbox = properties.get("bbox_used_for_cropping")
        if bbox is None or len(bbox) != 3:
            raise ValueError("temporal alignment requires bbox_used_for_cropping in ZYX")
        before_resampling = properties.get("shape_after_cropping_and_before_resampling")
        data_shape = tuple(int(v) for v in data.shape[1:])
        if before_resampling is not None and tuple(int(v) for v in before_resampling) != data_shape:
            raise ValueError(
                "temporal crop-origin alignment currently requires no spatial resampling: "
                f"before_resampling={before_resampling}, data={data_shape}"
            )
        return tuple(int(axis_bbox[0]) for axis_bbox in bbox)

    def load_case(self, identifier):
        center_data, center_seg, seg_prev, properties = super().load_case(identifier)
        temporal_data = []
        temporal_origins = []
        resolved_cases = []
        replicated = []
        for offset in self.temporal_offsets:
            if offset == 0:
                neighbor_id, was_replicated = identifier, False
                data = center_data
                neighbor_properties = properties
            else:
                neighbor_id, was_replicated = self._resolved_neighbor(identifier, offset)
                data, _, _, neighbor_properties = super().load_case(neighbor_id)
            temporal_data.append(data)
            temporal_origins.append(self._crop_origin(data, neighbor_properties))
            resolved_cases.append(neighbor_id)
            replicated.append(was_replicated)

        # Do not mutate the cached/source property dictionary.
        properties = dict(properties)
        properties["biohub_temporal_case_ids"] = resolved_cases
        properties["biohub_temporal_replicated"] = replicated
        properties["biohub_temporal_crop_origins_zyx"] = temporal_origins
        return (
            _LazyTemporalChannelStack(temporal_data, temporal_origins),
            center_seg,
            seg_prev,
            properties,
        )


class BiohubTemporalFiveFrameDataset(BiohubTemporalThreeFrameDataset):
    """Load centre labels plus aligned ``t-2..t+2`` image channels."""

    temporal_offsets = (-2, -1, 0, 1, 2)


class nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf(
    nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf
):
    """Half-head MSE detector with previous/centre/next input channels."""

    biohub_temporal_three_frame = True
    dataset_class = BiohubTemporalThreeFrameDataset

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataset_class = BiohubTemporalThreeFrameDataset

    def initialize(self):
        # nnUNetTrainer.initialize resolves the on-disk dataset implementation
        # and overwrites dataset_class. Restore the temporal view afterwards,
        # before on_train_start creates the data loaders.
        super().initialize()
        self.dataset_class = BiohubTemporalThreeFrameDataset

    @staticmethod
    def _synchronize_channel_transforms(transform, random_parent=None):
        """Make one-channel intensity probabilities act jointly on all times.

        nnU-Net wraps channel transforms in RandomTransform and then sometimes
        applies an additional per-channel Bernoulli. For temporal channels, we
        fold the latter probability into the outer Bernoulli and apply the
        selected transform to all frames with shared sampled parameters. This
        preserves the single-channel marginal application probability.
        """

        if hasattr(transform, "synchronize_channels"):
            transform.synchronize_channels = True
        if hasattr(transform, "p_per_channel"):
            probability = float(transform.p_per_channel)
            if probability < 1:
                if random_parent is None or not hasattr(random_parent, "apply_probability"):
                    raise RuntimeError(
                        f"cannot synchronize {type(transform).__name__}: no outer RandomTransform"
                    )
                random_parent.apply_probability *= probability
                transform.p_per_channel = 1.0
        if hasattr(transform, "transform"):
            nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf._synchronize_channel_transforms(
                transform.transform, transform
            )
        if hasattr(transform, "transforms"):
            for child in transform.transforms:
                nnUNetTrainerBiohubTemporal3FrameMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf._synchronize_channel_transforms(
                    child, random_parent
                )

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        cls._synchronize_channel_transforms(transforms)
        return transforms

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        # Dataset701 remains one modality on disk. Temporal stacking happens in
        # BiohubTemporalThreeFrameDataset, so architecture creation must force
        # the corresponding three input channels in both training and inference.
        return nnUNetTrainerBiohubMaskedHeatmapMSEAnisotropyAwareZFlipXYHalf.build_network_architecture(
            plans_manager,
            configuration_manager,
            3,
            num_output_channels,
            enable_deep_supervision,
        )
