from __future__ import annotations
import numpy as np
from cell_tracking.common.config import SCALE as NATIVE_SCALE_UM
from cell_tracking.common.config import SOON_CROP_SIZE as CROP_SIZE


def crop_around(vol, zyx, size=CROP_SIZE, pad_value=0.0):
    """Box of ``size`` (Z,Y,X) centred on ``zyx``."""
    vol = np.asarray(vol)
    squeezed = vol.ndim == 3
    if squeezed:
        vol = vol[None]
    if vol.ndim != 4:
        raise ValueError(f"vol must be (Z,Y,X) or (C,Z,Y,X), got {vol.shape}")
    _c, Z, Y, X = vol.shape
    sz, sy, sx = (int(s) for s in size)
    cz, cy, cx = (int(round(float(v))) for v in zyx)
    z0, y0, x0 = (cz - sz // 2, cy - sy // 2, cx - sx // 2)
    out = np.full((vol.shape[0], sz, sy, sx), pad_value, dtype=np.float32)
    src_z0, dst_z0 = (max(0, z0), max(0, -z0))
    src_y0, dst_y0 = (max(0, y0), max(0, -y0))
    src_x0, dst_x0 = (max(0, x0), max(0, -x0))
    src_z1, dst_z1 = (min(Z, z0 + sz), dst_z0 + min(Z, z0 + sz) - src_z0)
    src_y1, dst_y1 = (min(Y, y0 + sy), dst_y0 + min(Y, y0 + sy) - src_y0)
    src_x1, dst_x1 = (min(X, x0 + sx), dst_x0 + min(X, x0 + sx) - src_x0)
    if src_z1 > src_z0 and src_y1 > src_y0 and (src_x1 > src_x0):
        out[:, dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol[:, src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out[0] if squeezed else out


def make_gaussian_channel(shape, centre, radius_um=1.625, scale_um=NATIVE_SCALE_UM):
    """``(1, Z, Y, X)`` float32 3D Gaussian heatmap isotropic in physical 3D µm space."""
    Z, Y, X = (int(s) for s in shape)
    cz, cy, cx = (float(c) for c in centre)
    scale = np.asarray(scale_um, dtype=np.float32).reshape(3)
    sigma_vox = float(radius_um) / scale
    sz, sy, sx = (float(sigma_vox[0]), float(sigma_vox[1]), float(sigma_vox[2]))
    z = np.arange(Z, dtype=np.float32)
    y = np.arange(Y, dtype=np.float32)
    x = np.arange(X, dtype=np.float32)
    gz = np.exp(-((z - cz) ** 2) / (2.0 * sz**2))
    gy = np.exp(-((y - cy) ** 2) / (2.0 * sy**2))
    gx = np.exp(-((x - cx) ** 2) / (2.0 * sx**2))
    g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return g[None, ...].astype(np.float32)


def native_crop_size(crop_size, xy_pool=1):
    """Native-voxel extent of a crop whose YX was average-pooled by ``xy_pool`` at cut time."""
    z, y, x = (int(v) for v in crop_size)
    p = int(xy_pool)
    return (z, y * p, x * p)


def pool_xy_blocks(arr, xy_pool):
    """Average ``(..., Y, X)`` over non-overlapping ``xy_pool`` × ``xy_pool`` blocks."""
    p = int(xy_pool)
    if p == 1:
        return arr
    a = np.asarray(arr, dtype=np.float32)
    y, x = (a.shape[-2], a.shape[-1])
    if y % p or x % p:
        raise ValueError(f"YX {(y, x)} not divisible by xy_pool={p}")
    return a.reshape(*a.shape[:-2], y // p, p, x // p, p).mean(axis=(-3, -1))


def pooled_gaussian_centre(shape, xy_pool=1):
    """Pooled-grid position of the native crop centre: a pooled voxel k covers native ``[P·k, P·k + P)``, so the native centre ``P·Y/2`` sits at ``Y/2 - (P-1)/(2P)``."""
    z, y, x = (float(v) for v in shape)
    p = float(xy_pool)
    shift = (p - 1.0) / (2.0 * p)
    return (z / 2.0, y / 2.0 - shift, x / 2.0 - shift)


def pooled_scale_um(xy_pool=1):
    """µm per voxel on a grid whose YX was pooled by ``xy_pool``."""
    s = np.asarray(NATIVE_SCALE_UM, dtype=np.float64).copy()
    s[1:] *= float(xy_pool)
    return tuple((float(v) for v in s))


def extract_crop(vol, zyx, size=CROP_SIZE, prev=None, nxt=None, radius_um=1.625):
    """Three intensity frames and a Gaussian channel at the query coordinate."""
    p = vol if prev is None else prev
    n = vol if nxt is None else nxt
    c_prev = np.asarray(crop_around(p, zyx, size=size), dtype=np.float32)
    c_curr = np.asarray(crop_around(vol, zyx, size=size), dtype=np.float32)
    c_nxt = np.asarray(crop_around(n, zyx, size=size), dtype=np.float32)
    stack = [c_prev, c_curr, c_nxt]
    res = np.concatenate(stack, axis=0) if c_curr.ndim == 4 else np.stack(stack, axis=0)
    Z, Y, X = tuple((int(s) for s in size))
    g = make_gaussian_channel((Z, Y, X), (Z / 2.0, Y / 2.0, X / 2.0), radius_um=radius_um)
    return np.concatenate([res, g], axis=0)


def as_zyx_list(val):
    """``[(z, y, x), ...]`` from JSON lists / tuples; empty for missing."""
    if not val:
        return []
    out = []
    for p in val:
        if p is None or len(p) != 3:
            raise ValueError(f"zyx entry must be length-3, got {p!r}")
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def crop_origin(query_zyx, size):
    """Top-left voxel of a ``crop_around`` box centred on ``query_zyx``. Matches ``crop_around``."""
    cz, cy, cx = (int(round(float(v))) for v in query_zyx)
    sz, sy, sx = (int(s) for s in size)
    return (cz - sz // 2, cy - sy // 2, cx - sx // 2)
