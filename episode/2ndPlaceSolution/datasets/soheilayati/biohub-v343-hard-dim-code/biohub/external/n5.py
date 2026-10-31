"""Small read-only N5 filesystem adapter for external image pilots.

The project only needs deterministic ROI reads from the downloaded Linajea N5
store. Keeping this reader local avoids adding a compiled z5py dependency to
the Kaggle environment.
"""

from __future__ import annotations

import bz2
import gzip
import itertools
import json
import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DTYPES = {
    "uint8": ">u1", "uint16": ">u2", "uint32": ">u4", "uint64": ">u8",
    "int8": ">i1", "int16": ">i2", "int32": ">i4", "int64": ">i8",
    "float32": ">f4", "float64": ">f8",
}


@dataclass(frozen=True)
class N5Dataset:
    """Read-only primitive N5 dataset with coordinates in N5 storage order."""

    path: Path
    dimensions: tuple[int, ...]
    block_size: tuple[int, ...]
    data_type: str
    compression: dict

    @classmethod
    def open(cls, path: str | Path) -> "N5Dataset":
        path = Path(path)
        attributes = json.loads((path / "attributes.json").read_text(encoding="utf-8"))
        required = {"dimensions", "blockSize", "dataType", "compression"}
        if missing := required - set(attributes):
            raise ValueError(f"N5 dataset is missing attributes {sorted(missing)}")
        data_type = str(attributes["dataType"])
        if data_type not in _DTYPES:
            raise ValueError(f"unsupported N5 data type {data_type!r}")
        dimensions = tuple(map(int, attributes["dimensions"]))
        block_size = tuple(map(int, attributes["blockSize"]))
        if len(dimensions) != len(block_size):
            raise ValueError("N5 dimensions and blockSize ranks differ")
        return cls(path, dimensions, block_size, data_type, attributes["compression"])

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    @property
    def native_dtype(self) -> np.dtype:
        return np.dtype(_DTYPES[self.data_type]).newbyteorder("=")

    def _decompress(self, payload: bytes) -> bytes:
        kind = self.compression["type"]
        if kind == "raw":
            return payload
        if kind == "gzip":
            return gzip.decompress(payload)
        if kind == "zlib":
            return zlib.decompress(payload)
        if kind == "bzip2":
            return bz2.decompress(payload)
        if kind == "xz":
            return lzma.decompress(payload)
        raise ValueError(f"unsupported N5 compression {kind!r}")

    def read_block(self, grid_position: tuple[int, ...]) -> np.ndarray | None:
        """Return one block with array axes reversed from N5 storage order."""
        if len(grid_position) != self.ndim or min(grid_position) < 0:
            raise ValueError(f"invalid N5 grid position {grid_position}")
        path = self.path.joinpath(*map(str, grid_position))
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            mode, rank = struct.unpack(">HH", stream.read(4))
            if rank != self.ndim:
                raise ValueError(f"N5 block rank {rank} does not match dataset rank {self.ndim}")
            shape = struct.unpack(f">{rank}I", stream.read(4 * rank))
            if mode == 1:
                stream.read(4)  # number of elements; shape remains authoritative here
            elif mode != 0:
                raise ValueError(f"unsupported N5 block mode {mode}")
            payload = self._decompress(stream.read())
        count = int(np.prod(shape, dtype=np.int64))
        values = np.frombuffer(payload, dtype=np.dtype(_DTYPES[self.data_type]), count=count)
        if len(values) != count:
            raise IOError(f"short N5 block {path}: expected {count} values, got {len(values)}")
        return values.reshape(tuple(reversed(shape))).astype(self.native_dtype, copy=False)

    def read_roi(
        self,
        start: tuple[int, ...],
        stop: tuple[int, ...],
    ) -> np.ndarray:
        """Read an N5-order half-open ROI; returned NumPy axes are reversed."""
        if len(start) != self.ndim or len(stop) != self.ndim:
            raise ValueError("N5 ROI rank mismatch")
        start_array = np.asarray(start, dtype=np.int64)
        stop_array = np.asarray(stop, dtype=np.int64)
        dimensions = np.asarray(self.dimensions, dtype=np.int64)
        block_size = np.asarray(self.block_size, dtype=np.int64)
        if np.any(start_array < 0) or np.any(stop_array <= start_array):
            raise ValueError(f"invalid N5 ROI {start}:{stop}")
        if np.any(stop_array > dimensions):
            raise IndexError(f"N5 ROI {start}:{stop} exceeds dimensions {self.dimensions}")
        output = np.zeros(tuple(reversed(stop_array - start_array)), dtype=self.native_dtype)
        first = start_array // block_size
        last = (stop_array - 1) // block_size
        ranges = [range(int(lo), int(hi) + 1) for lo, hi in zip(first, last)]
        for grid_position in itertools.product(*ranges):
            block = self.read_block(tuple(grid_position))
            if block is None:
                continue
            block_origin = np.asarray(grid_position) * block_size
            block_stop = block_origin + np.asarray(block.shape[::-1])
            overlap_start = np.maximum(start_array, block_origin)
            overlap_stop = np.minimum(stop_array, block_stop)
            source_start = overlap_start - block_origin
            source_stop = overlap_stop - block_origin
            target_start = overlap_start - start_array
            target_stop = overlap_stop - start_array
            source_slices = tuple(
                slice(int(lo), int(hi))
                for lo, hi in zip(source_start[::-1], source_stop[::-1]))
            target_slices = tuple(
                slice(int(lo), int(hi))
                for lo, hi in zip(target_start[::-1], target_stop[::-1]))
            output[target_slices] = block[source_slices]
        return output
