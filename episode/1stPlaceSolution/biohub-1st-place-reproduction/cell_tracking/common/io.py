"""IO / preproc (reused)."""

import json
from pathlib import Path
import numpy as np
from cell_tracking.common import config


def set_pool(p: int) -> None:
    """Set the XY pooling factor."""
    config.POOL = int(p)


def get_pool() -> int:
    return config.POOL


def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


_ZC = {}


def load_volume(zp, t, meta=None):
    try:
        import zarr

        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2

        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])


SEG_EXTS = (".npy", ".npz")
_SEG_ARR = "seg"


def seg_paths(seg_dir):
    """Sorted [(path, name, t)] for every seg_<name>_t<t>.{npy,npz} under seg_dir."""
    import glob as _glob
    import os

    found = {}
    for ext in reversed(SEG_EXTS):
        for p in _glob.glob(os.path.join(str(seg_dir), "seg_*_t*" + ext)):
            stem = os.path.basename(p)[len("seg_") : -len(ext)]
            if "_t" not in stem:
                continue
            name, t = stem.rsplit("_t", 1)
            if not t.isdigit():
                continue
            found[name, int(t)] = p
    return [(found[k], k[0], k[1]) for k in sorted(found)]


def seg_file(seg_dir, name, t):
    """Existing seg path for one frame in either format, or None."""
    import os

    for ext in SEG_EXTS:
        p = os.path.join(str(seg_dir), f"seg_{name}_t{t}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_seg(path):
    """(Z,Y,X) label volume from either format."""
    if str(path).endswith(".npz"):
        with np.load(path) as z:
            return z[_SEG_ARR]
    return np.load(path)


def seg_stats(path):
    """(nnz, unique label values) for a frame, without inflating the volume when possible."""
    if str(path).endswith(".npz"):
        with np.load(path) as z:
            if "nnz" in z.files and "ids" in z.files:
                return (int(z["nnz"]), z["ids"])
            arr = z[_SEG_ARR]
    else:
        arr = np.load(path, mmap_mode="r")
    return (int((arr > 0).sum()), np.unique(arr))


def save_seg(seg_dir, name, t, arr, compressed=True):
    """Write one frame's label volume; returns the path written."""
    import os

    d = str(seg_dir)
    if not compressed:
        p = os.path.join(d, f"seg_{name}_t{t}.npy")
        np.save(p, arr)
        return p
    p = os.path.join(d, f"seg_{name}_t{t}.npz")
    np.savez_compressed(p, **{_SEG_ARR: arr}, ids=np.unique(arr), nnz=np.int64((arr > 0).sum()))
    return p


def pool_xy(vol, f=None):
    """Mean block-pool Y,X by f (Z untouched)."""
    f = config.POOL if f is None else f
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f * f, X // f * f)
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def pool_labels(lbl, f=None):
    """Label-preserving XY downsample by f: strided subsample (NOT mean-pool, which would fabricate labels)."""
    f = config.POOL if f is None else f
    Z, Y, X = lbl.shape
    Y2, X2 = (Y // f * f, X // f * f)
    return np.ascontiguousarray(lbl[:, :Y2:f, :X2:f])


def unpool_labels(lbl, native_shape, f=None):
    """Nearest-neighbour XY upsample of a pooled label volume back to `native_shape`."""
    f = config.POOL if f is None else int(f)
    native_shape = tuple((int(s) for s in native_shape))
    lbl = np.asarray(lbl)
    if f <= 0:
        raise ValueError(f"pool factor must be >= 1, got {f}")
    if f == 1:
        out = np.zeros(native_shape, dtype=lbl.dtype)
        sl = tuple((slice(0, min(a, b)) for a, b in zip(native_shape, lbl.shape)))
        out[sl] = lbl[sl]
        return out
    Z, Y, X = native_shape
    Y2, X2 = (Y // f * f, X // f * f)
    want = (Z, Y2 // f, X2 // f)
    src = np.zeros(want, dtype=lbl.dtype)
    sl = tuple((slice(0, min(a, b)) for a, b in zip(want, lbl.shape)))
    src[sl] = lbl[sl]
    up = np.repeat(np.repeat(src, f, axis=1), f, axis=2)
    if up.shape == native_shape:
        return np.ascontiguousarray(up)
    out = np.zeros(native_shape, dtype=lbl.dtype)
    z = min(Z, up.shape[0])
    out[:z, :Y2, :X2] = up[:z, :Y2, :X2]
    return out


def normalize(p):
    p = np.asarray(p, np.float32)
    return apply_norm(p, *norm_bounds(p))


def norm_bounds(p):
    """``(lo, hi)`` of :func:`normalize`: the frame's p50 and p99.5 (see its comment)."""
    p = np.asarray(p, np.float32)
    return (float(np.percentile(p, 50.0)), float(np.percentile(p, 99.5)))


def apply_norm(p, lo, hi):
    """:func:`normalize` with precomputed bounds."""
    p = np.asarray(p, np.float32)
    return np.clip((p - lo) / (hi - lo + 1e-06), -0.5, 6.0).astype(np.float32)


def pooled_voxel_um():
    """Physical size (µm) of one POOLED voxel (Z,Y,X)."""
    return np.array([config.SCALE[0], config.SCALE[1] * config.POOL, config.SCALE[2] * config.POOL], dtype=np.float64)
