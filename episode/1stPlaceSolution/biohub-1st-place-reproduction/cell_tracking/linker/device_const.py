"""Small constant tensors (voxel sizes, volume extents) made once per device and dtype."""

from __future__ import annotations
import functools
import numpy as np
import torch


@functools.lru_cache(maxsize=256)
def _cached(values: tuple, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device).reshape(shape)


def device_const(values, dtype: torch.dtype, device) -> torch.Tensor:
    """``torch.as_tensor(values, dtype=dtype, device=device)``, cached."""
    arr = np.asarray(values, dtype=np.float64)
    return _cached(tuple(arr.ravel().tolist()), arr.shape, dtype, torch.device(device))
