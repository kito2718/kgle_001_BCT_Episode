"""Zarr movie access: per-movie normalization, optional mean-pool downsampling, grid coordinate maps."""
from pathlib import Path

import numpy as np
import torch
import zarr


def get_downsample(cfg):
    """data.downsample as an int array (z,y,x); [1,1,1] when absent."""
    return np.array(list(cfg.data.get("downsample", [1, 1, 1])), int)


def to_ds(coords, ds):
    """Original-voxel zyx -> downsampled-grid coords (block centers).
    Block b of factor d covers original voxels [d*b, d*b+d-1], center d*b+(d-1)/2."""
    return (np.asarray(coords, np.float64) - (np.asarray(ds) - 1) / 2) / np.asarray(ds)


def to_model_input(x, local_contrast):
    """(Z,Y,X) normalized frame -> (1,Z,Y,X) tensor the detector consumes."""
    if local_contrast:
        raise ValueError('local-contrast detector input is not supported')
    return torch.from_numpy(x)[None]


class CellStore:
    """Per-movie zarr volumes (opened lazily) with the image-statistics normalization.

    frame(sid, t) -> float32 (Z,Y,X), (raw - q0.001) / (q0.999 - q0.001) clipped at 0; with downsample > 1 the
    raw frame is mean-pooled per axis first (remainder voxels dropped). Stores over the same movies at
    different downsamples can share `raw_cache` ({(sid, t): raw frame}) so each zarr chunk is decompressed
    once; the caller owns its eviction.
    """

    def __init__(self, data_dir, ids, with_gt=False, downsample=(1, 1, 1), raw_cache=None):
        if with_gt:
            raise ValueError('ground truth is not read at inference')
        self.data_dir = Path(data_dir)
        self._raw = raw_cache
        self.ds = np.array(list(downsample), int)
        self.ids = list(ids)
        self.norm = {}        # sid -> (q0.001, q0.999) from the zarr attrs image_statistics
        self.shape = {}       # sid -> (T, Z, Y, X)
        self._arr = {}        # sid -> open zarr array
        for sid in self.ids:
            zg = zarr.open(str(self.data_dir / f"{sid}.zarr"), mode="r")
            q = dict(zg.attrs)["image_statistics"]["quantiles"]
            self.norm[sid] = (float(q["0.001"]), float(q["0.999"]))
            self.shape[sid] = tuple(zg["0"].shape)

    def frame(self, sid, t):
        raw = None if self._raw is None else self._raw.get((sid, t))
        if raw is None:
            arr = self._arr.get(sid)
            if arr is None:
                arr = zarr.open(str(self.data_dir / f"{sid}.zarr"), mode="r")["0"]
                self._arr[sid] = arr
            raw = arr[t]
            if self._raw is not None:
                self._raw[(sid, t)] = raw
        ql, qh = self.norm[sid]
        f = raw.astype(np.float32)
        if (self.ds > 1).any():
            dz, dy, dx = self.ds
            Z, Y, X = f.shape
            f = f[:Z // dz * dz, :Y // dy * dy, :X // dx * dx]
            f = f.reshape(Z // dz, dz, Y // dy, dy, X // dx, dx).mean(axis=(1, 3, 5))
        return np.clip((f - ql) / (qh - ql + 1e-6), 0, None)
