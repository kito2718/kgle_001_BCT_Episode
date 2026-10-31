"""Node features from unetcellpose decoded instances."""

from __future__ import annotations
import numpy as np
from cell_tracking.common.io import pooled_voxel_um
from cell_tracking.common.detection_metrics import centroids_vox

NODE_FEAT_DIM = 82
POS_FEAT_DIM = 32
TIME_FEAT_DIM = 8
GEOM_FEAT_DIM = 13
_POS_EMBED_DIM = 8


def extract_pos_features(
    coords: np.ndarray, image_shape: tuple[int, ...], pos_embed_dim: int = _POS_EMBED_DIM
) -> np.ndarray:
    """Sinusoidal positional embeddings for node coordinates."""
    t, z, y, x = (coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def node_feat_dim(
    base: int = 24,
    include_time: bool = True,
    include_appearance: bool = True,
    include_shape: bool = True,
    include_geom: bool = False,
) -> int:
    return (
        appearance_feat_dim(base, include_appearance)
        + (2 if include_shape else 0)
        + pos_feat_dim(include_time)
        + (GEOM_FEAT_DIM if include_geom else 0)
    )


def appearance_feat_dim(base: int = 24, include_appearance: bool = True) -> int:
    return 2 * int(base) if include_appearance else 0


def pos_feat_dim(include_time: bool = True) -> int:
    return POS_FEAT_DIM if include_time else POS_FEAT_DIM - TIME_FEAT_DIM


def time_block_slice(width: int) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the 8 t_rel dims, or ``None`` if already dropped."""
    if int(width) == 39:
        return None
    if int(width) == 47:
        return (2, 10)
    raise ValueError(f"expected 47 cached or 39 spatial node features, got {width}")


def drop_time_columns(feats: np.ndarray) -> np.ndarray:
    """Slice the 8 constant t_rel dims off a cached node-feature matrix."""
    sl = time_block_slice(int(feats.shape[-1]))
    if sl is None:
        return feats
    start, end = sl
    return np.concatenate([feats[..., :start], feats[..., end:]], axis=-1)


def appearance_slice(base: int = 24) -> slice:
    """The ``d1`` appearance block: ``d1`` mean over mask, then ``d1`` at the centroid."""
    return slice(0, 2 * int(base))


def shape_slice(base: int = 24, include_appearance: bool = True) -> slice:
    """The two decode-derived dims: ``log(volume)`` then mean ``sigmoid(prob)``."""
    start = appearance_feat_dim(base, include_appearance)
    return slice(start, start + 2)


def point_node_features(
    coords_native: np.ndarray,
    *,
    window_shape: tuple[int, int, int, int],
    t_rel: float = 0.0,
    log_volume: float | np.ndarray = 0.0,
    mean_prob: float | np.ndarray = 0.0,
    include_time: bool = True,
    include_shape: bool = True,
    include_geom: bool = False,
    geom_fill: np.ndarray | None = None,
) -> np.ndarray:
    """Node features for arbitrary points, with no decode and no instance masks."""
    coords_native = np.asarray(coords_native, dtype=np.float64).reshape(-1, 3)
    n = len(coords_native)
    pos = position_features(coords_native, t_rel, window_shape, include_time=include_time)
    geom = None
    if include_geom:
        geom = (
            np.zeros((n, GEOM_FEAT_DIM), np.float64)
            if geom_fill is None
            else np.broadcast_to(np.asarray(geom_fill, np.float64), (n, GEOM_FEAT_DIM))
        )
    if not include_shape:
        out = pos if geom is None else np.concatenate([pos, geom], axis=1)
        return out.astype(np.float32)
    if n == 0:
        return np.zeros((0, 2 + pos.shape[1] + (GEOM_FEAT_DIM if include_geom else 0)), np.float32)
    blocks = [
        np.broadcast_to(np.asarray(log_volume, np.float64), (n,))[:, None],
        np.broadcast_to(np.asarray(mean_prob, np.float64), (n,))[:, None],
        pos,
    ]
    if geom is not None:
        blocks.append(geom)
    return np.concatenate(blocks, axis=1).astype(np.float32)


_AXIS_OFF, _AXIS_LEN = (7, 3)


def geom_slice(feat_dim: int) -> slice:
    """The geometry block, given a cache's total width."""
    return slice(int(feat_dim) - GEOM_FEAT_DIM, int(feat_dim))


def geom_axis_slice(feat_dim: int) -> slice:
    """Just the principal-axis unit vector, for the line-to-line pair terms."""
    start = int(feat_dim) - GEOM_FEAT_DIM + _AXIS_OFF
    return slice(start, start + _AXIS_LEN)


def instance_geom_features(vox, lo, hi, spacing=None) -> np.ndarray:
    """``(n, GEOM_FEAT_DIM)`` second-moment shape descriptors, in µm."""
    spacing = np.asarray(pooled_voxel_um() if spacing is None else spacing, np.float64)
    lo = np.asarray(lo, np.intp)
    n = len(lo)
    if n == 0:
        return np.zeros((0, GEOM_FEAT_DIM), np.float32)
    p = np.asarray(vox, np.float64) * spacing
    cnt = (np.asarray(hi, np.intp) - lo).astype(np.float64)
    mean = np.add.reduceat(p, lo, axis=0) / cnt[:, None]
    pr = np.stack(
        [
            p[:, 0] * p[:, 0],
            p[:, 0] * p[:, 1],
            p[:, 0] * p[:, 2],
            p[:, 1] * p[:, 1],
            p[:, 1] * p[:, 2],
            p[:, 2] * p[:, 2],
        ],
        axis=1,
    )
    m2 = np.add.reduceat(pr, lo, axis=0) / cnt[:, None]
    cov = np.empty((n, 3, 3), np.float64)
    for a, b, c in ((0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 1, 1), (4, 1, 2), (5, 2, 2)):
        cov[:, b, c] = cov[:, c, b] = m2[:, a] - mean[:, b] * mean[:, c]
    cov[:, 0, 0] += spacing[0] ** 2 / 12.0
    cov[:, 1, 1] += spacing[1] ** 2 / 12.0
    cov[:, 2, 2] += spacing[2] ** 2 / 12.0
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals[:, ::-1], 1e-12, None)
    evecs = evecs[:, :, ::-1]
    axes = np.sqrt(5.0 * evals)
    axis0 = evecs[:, :, 0]
    dom = np.abs(axis0).argmax(axis=1)
    sgn = np.sign(axis0[np.arange(n), dom])
    axis0 = axis0 * np.where(sgn == 0, 1.0, sgn)[:, None]
    bb_lo = np.minimum.reduceat(p, lo, axis=0)
    bb_hi = np.maximum.reduceat(p, lo, axis=0)
    extent_um = bb_hi - bb_lo + spacing
    vol_um3 = cnt * float(np.prod(spacing))
    bbox_fill = vol_um3 / np.maximum(np.prod(extent_um, axis=1), 1e-12)
    ellip = 4.0 / 3.0 * np.pi * np.prod(axes, axis=1)
    ellip_fill = vol_um3 / np.maximum(ellip, 1e-12)
    out = np.concatenate(
        [
            np.log(axes),
            axes[:, 1:2] / axes[:, 0:1],
            axes[:, 2:3] / axes[:, 0:1],
            bbox_fill[:, None],
            ellip_fill[:, None],
            axis0,
            np.log(extent_um),
        ],
        axis=1,
    )
    assert out.shape[1] == GEOM_FEAT_DIM
    return out.astype(np.float32)


def grouped_instance_slices(instances: np.ndarray):
    """``(ids, vox, lo, hi)`` for foreground voxels sorted by label."""
    instances = np.asarray(instances)
    ids = np.unique(instances)
    ids = ids[ids > 0]
    if len(ids) == 0:
        z = np.zeros(0, np.intp)
        return (ids, np.zeros((0, 3), np.intp), z, z)
    vox = np.argwhere(instances > 0)
    labs = instances[vox[:, 0], vox[:, 1], vox[:, 2]]
    order = np.argsort(labs, kind="stable")
    vox = vox[order]
    labs = labs[order]
    lo = np.searchsorted(labs, ids, side="left")
    hi = np.searchsorted(labs, ids, side="right")
    return (ids, vox, lo, hi)


def position_features(
    coords_native: np.ndarray, t_rel: float, window_shape: tuple[int, int, int, int], include_time: bool = True
) -> np.ndarray:
    """Sinusoidal position encoding; ``window_shape`` is ``(T, Z, Y, X)``."""
    coords_native = np.asarray(coords_native, dtype=np.float32)
    if len(coords_native) == 0:
        return np.zeros((0, pos_feat_dim(include_time)), np.float32)
    c = np.zeros((len(coords_native), 4), np.float32)
    c[:, 0] = t_rel
    c[:, 1:] = coords_native
    pos = extract_pos_features(c, window_shape)
    return pos if include_time else pos[:, TIME_FEAT_DIM:]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def instance_node_features(
    instances: np.ndarray,
    d1: np.ndarray | None,
    prob_logit: np.ndarray,
    *,
    coords_native: np.ndarray | None = None,
    pool: int = 2,
    t_rel: float = 0.0,
    window_shape: tuple[int, int, int, int],
    base: int = 24,
    include_time: bool = True,
    include_appearance: bool = True,
    include_shape: bool = True,
    include_geom: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-instance features on the pooled decode grid."""
    instances = np.asarray(instances, np.int32)
    prob_logit = np.asarray(prob_logit, np.float64)
    if prob_logit.ndim == 4:
        prob_logit = prob_logit[0]
    if include_appearance:
        if d1 is None:
            raise ValueError("include_appearance=True needs d1; got None")
        d1 = np.asarray(d1, np.float64)
        if d1.ndim != 4:
            raise ValueError(f"d1 must be (C,Z,Y,X), got shape {d1.shape}")
        if d1.shape[0] != base:
            raise ValueError(f"d1 has {d1.shape[0]} channels, expected base={base}")
    else:
        d1 = None
    ids, vox, lo, hi = grouped_instance_slices(instances)
    n = len(ids)
    if n == 0:
        return (
            ids,
            np.zeros((0, 3), np.float64),
            np.zeros(
                (0, node_feat_dim(base, include_time, include_appearance, include_shape, include_geom)), np.float32
            ),
            np.zeros(0, np.float64),
        )
    cent_pooled = centroids_vox(instances)
    if coords_native is None:
        from cell_tracking.detector.inference import pooled_to_native

        coords_native = pooled_to_native(cent_pooled, pool)
    else:
        coords_native = np.asarray(coords_native, dtype=np.float64)
    shape = prob_logit.shape
    lin = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)
    prob_gather = _sigmoid(prob_logit.ravel()[lin])
    mean_prob = np.zeros(n, np.float64)
    volumes = np.zeros(n, np.float64)
    blocks: list[np.ndarray] = []
    if d1 is not None:
        d1_gather = d1.reshape(d1.shape[0], -1)[:, lin].T
        d1_mean = np.zeros((n, base), np.float64)
        for j in range(n):
            d1_mean[j] = d1_gather[lo[j] : hi[j]].mean(axis=0)
        zz = np.clip(np.round(cent_pooled[:, 0]).astype(np.intp), 0, shape[0] - 1)
        yy = np.clip(np.round(cent_pooled[:, 1]).astype(np.intp), 0, shape[1] - 1)
        xx = np.clip(np.round(cent_pooled[:, 2]).astype(np.intp), 0, shape[2] - 1)
        blocks += [d1_mean, d1[:, zz, yy, xx].T]
    for j in range(n):
        mean_prob[j] = prob_gather[lo[j] : hi[j]].mean()
        volumes[j] = hi[j] - lo[j]
    if include_shape:
        blocks.append(np.log(np.maximum(volumes, 1.0))[:, None])
        blocks.append(mean_prob[:, None])
    blocks.append(position_features(coords_native, t_rel, window_shape, include_time=include_time))
    if include_geom:
        blocks.append(instance_geom_features(vox, lo, hi))
    feats = np.concatenate(blocks, axis=1).astype(np.float32)
    feat_dim = node_feat_dim(base, include_time, include_appearance, include_shape, include_geom)
    if feats.shape[1] != feat_dim:
        raise RuntimeError(f"expected {feat_dim} features, got {feats.shape[1]}")
    return (ids, coords_native, feats, volumes)
