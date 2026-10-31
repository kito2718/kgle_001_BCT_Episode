"""Shared candidate support and masked-normalisation utilities for the edge linker."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from cell_tracking.linker.soon_cache import SOON_KEYS
from cell_tracking.linker.intensity import INTENSITY_KEYS
import torch
import torch.nn.functional as F
from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM
from cell_tracking.linker.features import GEOM_FEAT_DIM


def candidate_mask(
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    *,
    radius_um: float = 0.0,
    rank_k: int = 0,
    voxel_size: Sequence[float] | np.ndarray = NATIVE_VOXEL_UM,
) -> np.ndarray:
    """Return the local parent-to-child candidate mask."""
    src = np.asarray(coords_t, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(coords_t1, dtype=np.float64).reshape(-1, 3)
    radius = float(radius_um)
    k = int(rank_k)
    if radius < 0:
        raise ValueError(f"radius_um must be >= 0, got {radius_um}")
    if k < 0:
        raise ValueError(f"rank_k must be >= 0, got {rank_k}")
    n_src, n_dst = (len(src), len(dst))
    if n_src == 0 or n_dst == 0:
        return np.zeros((n_src, n_dst), dtype=bool)
    if radius <= 0 and k <= 0:
        return np.ones((n_src, n_dst), dtype=bool)
    vs = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    dist = np.linalg.norm((dst[None, :, :] - src[:, None, :]) * vs, axis=-1)
    keep = np.zeros((n_src, n_dst), dtype=bool)
    if radius > 0:
        keep |= dist <= radius
    if k > 0:
        kk = min(k, n_dst)
        order = np.argsort(dist, axis=1, kind="stable")[:, :kk]
        rows = np.repeat(np.arange(n_src), kk)
        keep[rows, order.reshape(-1)] = True
    return keep


def masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor | None, *, dim: int) -> torch.Tensor:
    """Log-softmax over entries selected by ``mask``."""
    if mask is None:
        return F.log_softmax(scores, dim=dim)
    if mask.shape != scores.shape:
        raise ValueError(f"support mask shape {tuple(mask.shape)} does not match scores {tuple(scores.shape)}")
    mask = torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
    neg_inf = torch.full_like(scores, float("-inf"))
    masked = scores.masked_fill(~mask, float("-inf"))
    valid = mask.any(dim=dim, keepdim=True)
    safe_masked = torch.where(valid, masked, torch.zeros_like(masked))
    lse = torch.logsumexp(safe_masked, dim=dim, keepdim=True)
    safe_lse = torch.where(valid, lse, torch.zeros_like(lse))
    out = masked - safe_lse
    return torch.where(mask & valid, out, neg_inf)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, *, dim: int) -> torch.Tensor:
    """Softmax over a support mask, returning zero for unsupported entries."""
    if mask is None:
        return torch.softmax(scores, dim=dim)
    return masked_log_softmax(scores, mask, dim=dim).exp()


def cache_mean_prob(cache: dict, *, include_geom: bool = False) -> np.ndarray:
    """Return the cache's detector-confidence feature without changing dtype."""
    feats = np.asarray(cache["feats"])
    trailing = 34 + (GEOM_FEAT_DIM if include_geom else 0)
    if feats.ndim != 2 or feats.shape[1] < trailing:
        raise ValueError(f"cannot locate mean_prob in a {feats.shape} feature block")
    if len(feats) == 0:
        return np.zeros(0, feats.dtype)
    return feats[:, feats.shape[1] - trailing + 1]


def prune_cache_frame(cache: dict, min_prob: float = 0.0, *, include_geom: bool = False) -> dict:
    """Drop cache nodes below ``min_prob`` while preserving cache metadata."""
    if float(min_prob) <= 0.0:
        return cache
    feats = np.asarray(cache["feats"])
    keep = cache_mean_prob(cache, include_geom=include_geom) >= float(min_prob)
    out = dict(cache)
    out["feats"] = feats[keep]
    out["coords_native"] = np.asarray(cache["coords_native"])[keep]
    out["volumes"] = np.asarray(cache["volumes"])[keep]
    for key in (*SOON_KEYS, *INTENSITY_KEYS):
        if key in out:
            out[key] = np.asarray(cache[key])[keep]
    return out
