"""HOCT raw fluorescence intensity features for edge linking and mitotic division."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

DEFAULT_NATIVE_SPACING = (1.625, 0.40625, 0.40625)
DEFAULT_RADIUS_UM = 3.5
INTENSITY_NODE_DIM = 4
INTENSITY_PAIR_DIM = 4
INTENSITY_FORK_DIM = 4
INTENSITY_KEYS = ("intensity_norm", "intensity_raw_mean", "intensity_raw_max", "intensity_raw_std", "intensity_raw_sum")
INTENSITY_NORM_KEY = INTENSITY_KEYS[0]
INTENSITY_RAW_SUM_KEY = INTENSITY_KEYS[4]


def intensity_frame_path(intensity_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(intensity_dir) / movie / f"frame_{int(frame):04d}_intensity.npz"


def read_intensity_frame(path: Path | str) -> dict[str, np.ndarray] | None:
    """Read one ``build_intensity_sidecar.py`` frame, or None if it is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        out = {
            "intensity_norm": np.asarray(data["norm_feats"], dtype=np.float32),
            "intensity_raw_mean": np.asarray(data["raw_mean"], dtype=np.float32),
            "intensity_raw_max": np.asarray(data["raw_max"], dtype=np.float32),
            "intensity_raw_std": np.asarray(data["raw_std"], dtype=np.float32),
            "intensity_raw_sum": np.asarray(data["raw_sum"], dtype=np.float32),
        }
        if "coords_native" in data.files:
            out["coords_native"] = np.asarray(data["coords_native"], dtype=np.float64)
        return out


def write_intensity_sidecar(
    path: Path | str,
    vol: np.ndarray,
    coords_native: np.ndarray,
    *,
    volumes: np.ndarray | None = None,
    frame: int | None = None,
    radius_um: float = DEFAULT_RADIUS_UM,
) -> Path:
    """Extract node intensities and write one sidecar frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_stats = extract_node_intensities(vol, coords_native, radius_um=radius_um, volumes=volumes)
    norm_feats = normalize_intensity_features(raw_stats)
    payload = {
        "raw_mean": np.asarray(raw_stats["mean"], dtype=np.float32),
        "raw_max": np.asarray(raw_stats["max"], dtype=np.float32),
        "raw_std": np.asarray(raw_stats["std"], dtype=np.float32),
        "raw_sum": np.asarray(raw_stats["sum"], dtype=np.float32),
        "norm_feats": np.asarray(norm_feats, dtype=np.float32),
        "coords_native": np.asarray(coords_native, dtype=np.float64),
    }
    if frame is not None:
        payload["frame"] = np.int32(frame)
    np.savez(path, **payload)
    return path


def write_instance_mask_intensity_sidecar(
    path: Path | str,
    vol: np.ndarray,
    instances: np.ndarray,
    *,
    coords_native: np.ndarray | None = None,
    pool: int = 2,
    frame: int | None = None,
    bg_percentile: float = 10.0,
) -> Path:
    """Extract exact-mask node intensities and write one sidecar frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_stats = extract_instance_mask_intensities(vol, instances, pool=pool, bg_percentile=bg_percentile)
    norm_feats = normalize_intensity_features(raw_stats)
    payload = {
        "raw_mean": np.asarray(raw_stats["mean"], dtype=np.float32),
        "raw_max": np.asarray(raw_stats["max"], dtype=np.float32),
        "raw_std": np.asarray(raw_stats["std"], dtype=np.float32),
        "raw_sum": np.asarray(raw_stats["sum"], dtype=np.float32),
        "norm_feats": np.asarray(norm_feats, dtype=np.float32),
    }
    if coords_native is not None:
        payload["coords_native"] = np.asarray(coords_native, dtype=np.float64)
    if frame is not None:
        payload["frame"] = np.int32(frame)
    np.savez(path, **payload)
    return path


ROW_IDENTITY_TOL_VOX = 0.5
ROW_IDENTITY_RECOVERY_TOL_VOX = 5.0
ROW_IDENTITY_MAX_AMBIGUOUS_FRAC = 0.1


def attach_intensity_from_dir(cache: dict, intensity_dir: Path | str, movie: str) -> dict:
    """Attach the intensity sidecar frame matching *cache*'s own frame index."""
    frame = int(np.asarray(cache["frame"]))
    path = intensity_frame_path(intensity_dir, movie, frame)
    got = read_intensity_frame(path)
    if got is None:
        raise FileNotFoundError(f"no intensity sidecar at {path}")
    cache_coords = np.asarray(cache["coords_native"], dtype=np.float64)
    n = int(cache_coords.shape[0])
    n_sidecar = int(got["intensity_norm"].shape[0])
    sidecar_coords = got.pop("coords_native", None)
    count_mismatch = n_sidecar != n
    if count_mismatch and sidecar_coords is None:
        raise ValueError(f"intensity sidecar {path} has {n_sidecar} detections, cache has {n}")
    if sidecar_coords is not None:
        sidecar_coords = np.asarray(sidecar_coords, dtype=np.float64)
        if count_mismatch:
            bad = max(n, n_sidecar)
            dist = None
        else:
            dist = np.linalg.norm(sidecar_coords - cache_coords, axis=1)
            bad = int((dist > ROW_IDENTITY_TOL_VOX).sum())
        if bad:
            got, n_ambiguous = _realign_by_nearest_neighbor(got, sidecar_coords, cache_coords)
            frac = n_ambiguous / n
            if frac > ROW_IDENTITY_MAX_AMBIGUOUS_FRAC:
                detail = (
                    f"{n_sidecar} sidecar detections vs {n} cache detections (count mismatch)"
                    if count_mismatch
                    else f"{bad}/{n} rows differed by more than {ROW_IDENTITY_TOL_VOX} voxels (worst: row {int(np.argmax(dist))}, {dist.max():.3f} vox)"
                )
                raise ValueError(
                    f"intensity sidecar {path} ROW IDENTITY mismatch against the cache it is being attached to: {detail}. Nearest-neighbor realignment still left {n_ambiguous}/{n} ({frac:.1%}) rows unmatched within {ROW_IDENTITY_RECOVERY_TOL_VOX} voxels, above the {ROW_IDENTITY_MAX_AMBIGUOUS_FRAC:.0%} recovery threshold -- this looks like a genuinely different detection set (e.g. a real decode non-determinism, not a tie-break reorder or an off-by-one detection), not safe to zero-fill."
                )
            import logging

            logging.getLogger(__name__).warning(
                "%s: %d sidecar vs %d cache detections%s; realigned via nearest-neighbor, %d row(s) zero-filled (no close match)",
                path,
                n_sidecar,
                n,
                " (count mismatch)" if count_mismatch else " reordered",
                n_ambiguous,
            )
    out = dict(cache)
    out.update(got)
    return out


def _realign_by_nearest_neighbor(
    got: dict[str, np.ndarray], sidecar_coords: np.ndarray, cache_coords: np.ndarray
) -> tuple[dict[str, np.ndarray], int]:
    n_cache = cache_coords.shape[0]
    n_sidecar = sidecar_coords.shape[0]
    tree = cKDTree(cache_coords)
    dist_nn, idx_nn = tree.query(sidecar_coords, k=1)
    order = np.argsort(dist_nn)
    perm = np.full(n_cache, -1, dtype=np.intp)
    used_cache = np.zeros(n_cache, dtype=bool)
    used_sidecar = np.zeros(n_sidecar, dtype=bool)
    match_dist = np.full(n_cache, np.inf)
    for j in order:
        i = int(idx_nn[j])
        if used_cache[i]:
            continue
        perm[i] = j
        used_cache[i] = True
        used_sidecar[j] = True
        match_dist[i] = dist_nn[j]
    leftover_cache = np.where(~used_cache)[0]
    leftover_sidecar = np.where(~used_sidecar)[0]
    k = min(len(leftover_cache), len(leftover_sidecar))
    if k:
        perm[leftover_cache[:k]] = leftover_sidecar[:k]
    still_unmatched = perm < 0
    perm[still_unmatched] = 0
    ambiguous = still_unmatched | (match_dist > ROW_IDENTITY_RECOVERY_TOL_VOX)
    realigned = {k_: np.asarray(v)[perm] for k_, v in got.items()}
    n_ambiguous = int(ambiguous.sum())
    if n_ambiguous:
        for arr in realigned.values():
            arr[ambiguous] = 0.0
    return (realigned, n_ambiguous)


def extract_instance_mask_intensities(
    vol: np.ndarray, instances: np.ndarray, *, pool: int = 2, bg_percentile: float = 10.0
) -> dict[str, np.ndarray]:
    """Extract exact nuclear intensity statistics inside each unetcellpose instance mask."""
    instances = np.asarray(instances, dtype=np.int32)
    vol = np.asarray(vol, dtype=np.float32)
    if instances.ndim != 3:
        raise ValueError(f"instances must be 3D (Z, Y, X), got shape {instances.shape}")
    if vol.shape != instances.shape:
        from cell_tracking.common.io import pool_xy

        vol = pool_xy(vol, f=int(pool))
        if vol.shape != instances.shape:
            raise ValueError(f"vol shape {vol.shape} does not match instances {instances.shape} after pool={pool}")
    from cell_tracking.linker.features import grouped_instance_slices

    ids, vox, lo, hi = grouped_instance_slices(instances)
    n = len(ids)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"mean": empty, "max": empty, "std": empty, "sum": empty}
    shape = instances.shape
    lin = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)
    raw_gather = vol.ravel()[lin].astype(np.float64)
    b_local = float(np.percentile(vol, bg_percentile)) if bg_percentile > 0 else 0.0
    sub_gather = np.maximum(0.0, raw_gather - b_local)
    means = np.zeros(n, dtype=np.float32)
    maxs = np.zeros(n, dtype=np.float32)
    stds = np.zeros(n, dtype=np.float32)
    sums = np.zeros(n, dtype=np.float32)
    for j in range(n):
        cell_vox = sub_gather[lo[j] : hi[j]]
        if len(cell_vox) > 0:
            means[j] = float(cell_vox.mean())
            maxs[j] = float(cell_vox.max())
            stds[j] = float(cell_vox.std())
            sums[j] = float(cell_vox.sum())
    return {"mean": means, "max": maxs, "std": stds, "sum": sums}


def extract_node_intensities(
    vol_3d: np.ndarray,
    coords_native: np.ndarray,
    *,
    spacing: tuple[float, float, float] = DEFAULT_NATIVE_SPACING,
    radius_um: float = DEFAULT_RADIUS_UM,
    volumes: np.ndarray | None = None,
    bg_percentile: float = 10.0,
) -> dict[str, np.ndarray]:
    """Extract raw nuclear intensity statistics for each centroid in a 3D volume."""
    coords = np.asarray(coords_native, dtype=np.float64)
    n = len(coords)
    if n == 0 or vol_3d.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"mean": empty, "max": empty, "std": empty, "sum": empty}
    Z, Y, X = vol_3d.shape
    coords_um = coords * np.asarray(spacing, dtype=np.float64)[None, :]
    if n > 1:
        tree = cKDTree(coords_um)
        nn_dists, _ = tree.query(coords_um, k=2)
        min_dist_um = nn_dists[:, 1]
        clamped_r_um = np.clip(0.45 * min_dist_um, 1.5, radius_um)
    else:
        clamped_r_um = np.full(n, radius_um, dtype=np.float64)
    if volumes is not None and len(volumes) == n and np.any(volumes > 0):
        v = np.asarray(volumes, dtype=np.float64)
        med_v = float(np.median(v[v > 0])) if np.any(v > 0) else 1.0
        v_factor = np.clip((v / max(med_v, 1.0)) ** (1.0 / 3.0), 0.75, 1.35)
        clamped_r_um = np.clip(clamped_r_um * v_factor, 1.5, radius_um)
    means = np.zeros(n, dtype=np.float32)
    maxs = np.zeros(n, dtype=np.float32)
    stds = np.zeros(n, dtype=np.float32)
    sums = np.zeros(n, dtype=np.float32)
    sp_z, sp_y, sp_x = spacing
    for i in range(n):
        cz, cy, cx = coords[i]
        r_um = clamped_r_um[i]
        rz = max(1.0, float(r_um / sp_z))
        ry = max(1.0, float(r_um / sp_y))
        rx = max(1.0, float(r_um / sp_x))
        rz_int = int(np.ceil(rz))
        ry_int = int(np.ceil(ry))
        rx_int = int(np.ceil(rx))
        pad_z = rz_int + 2
        pad_y = ry_int + 4
        pad_x = rx_int + 4
        zmin, zmax = (max(0, int(round(cz - pad_z))), min(Z, int(round(cz + pad_z + 1))))
        ymin, ymax = (max(0, int(round(cy - pad_y))), min(Y, int(round(cy + pad_y + 1))))
        xmin, xmax = (max(0, int(round(cx - pad_x))), min(X, int(round(cx + pad_x + 1))))
        if zmin >= zmax or ymin >= ymax or xmin >= xmax:
            continue
        box = vol_3d[zmin:zmax, ymin:ymax, xmin:xmax]
        b_local = float(np.percentile(box, bg_percentile)) if box.size > 0 else 0.0
        ezmin, ezmax = (max(0, int(round(cz - rz_int))), min(Z, int(round(cz + rz_int + 1))))
        eymin, eymax = (max(0, int(round(cy - ry_int))), min(Y, int(round(cy + ry_int + 1))))
        exmin, exmax = (max(0, int(round(cx - rx_int))), min(X, int(round(cx + rx_int + 1))))
        sub = vol_3d[ezmin:ezmax, eymin:eymax, exmin:exmax]
        zz, yy, xx = np.ogrid[ezmin:ezmax, eymin:eymax, exmin:exmax]
        mask = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        raw_vox = sub[mask].astype(np.float64)
        if len(raw_vox) > 0:
            sub_vox = np.maximum(0.0, raw_vox - b_local)
            means[i] = float(sub_vox.mean())
            maxs[i] = float(sub_vox.max())
            stds[i] = float(sub_vox.std())
            sums[i] = float(sub_vox.sum())
    return {"mean": means, "max": maxs, "std": stds, "sum": sums}


def normalize_intensity_features(
    raw_stats: dict[str, np.ndarray], frame_mean: float | None = None, frame_std: float | None = None
) -> np.ndarray:
    """Normalize raw intensity statistics to be scale-free and drift-invariant."""
    means = np.asarray(raw_stats["mean"], dtype=np.float32)
    maxs = np.asarray(raw_stats["max"], dtype=np.float32)
    stds = np.asarray(raw_stats["std"], dtype=np.float32)
    sums = np.asarray(raw_stats["sum"], dtype=np.float32)
    n = len(means)
    if n == 0:
        return np.zeros((0, INTENSITY_NODE_DIM), dtype=np.float32)
    pos_means = means[means > 0]
    f_mean = (
        float(np.median(pos_means))
        if frame_mean is None and len(pos_means) > 0
        else 0.0
        if frame_mean is None
        else float(frame_mean)
    )
    f_std = (
        float(np.std(pos_means))
        if frame_std is None and len(pos_means) > 0
        else 1.0
        if frame_std is None
        else float(frame_std)
    )
    f_std = max(f_std, 0.0001)
    pos_sums = sums[sums > 0]
    med_sum = float(np.median(pos_sums)) if len(pos_sums) > 0 else 1.0
    med_sum = max(med_sum, 1.0)
    norm_mean = (means - f_mean) / f_std
    norm_max = (maxs - f_mean) / f_std
    norm_std = stds / f_std
    log_sum_rel = np.log(np.maximum(sums, 1.0)) - np.log(med_sum)
    return np.stack([norm_mean, norm_max, norm_std, log_sum_rel], axis=-1)


def intensity_pair_features(intensity_t: torch.Tensor, intensity_t1: torch.Tensor) -> torch.Tensor:
    """Compute pairwise intensity continuity features for all (i, j) candidate edges."""
    mean_t = intensity_t[..., 0].unsqueeze(-1)
    mean_t1 = intensity_t1[..., 0].unsqueeze(-2)
    max_t = intensity_t[..., 1].unsqueeze(-1)
    max_t1 = intensity_t1[..., 1].unsqueeze(-2)
    lsum_t = intensity_t[..., 3].unsqueeze(-1)
    lsum_t1 = intensity_t1[..., 3].unsqueeze(-2)
    d_lsum_signed = lsum_t1 - lsum_t
    d_lsum_abs = d_lsum_signed.abs()
    d_mean = (mean_t1 - mean_t).abs()
    d_max = max_t1 - max_t
    return torch.stack([d_lsum_abs, d_lsum_signed, d_mean, d_max], dim=-1)


def intensity_fork_features(
    raw_sums_t: torch.Tensor,
    norm_feats_t: torch.Tensor,
    raw_sums_t1: torch.Tensor,
    norm_feats_t1: torch.Tensor,
    fork_pairs: torch.Tensor,
) -> torch.Tensor:
    """Compute swap-invariant intensity features for mitotic triplet candidates (parent, d1, d2)."""
    batch = fork_pairs.shape[0]
    batch_idx = torch.arange(batch, device=fork_pairs.device).view(batch, 1).expand(-1, fork_pairs.shape[1])
    src_i = fork_pairs[..., 0].long()
    child_a = fork_pairs[..., 1].long()
    child_b = fork_pairs[..., 2].long()
    p_sum = raw_sums_t[batch_idx, src_i].clamp(min=1.0)
    p_norm_mean = norm_feats_t[batch_idx, src_i, 0]
    p_norm_max = norm_feats_t[batch_idx, src_i, 1]
    ca_sum = raw_sums_t1[batch_idx, child_a].clamp(min=1.0)
    cb_sum = raw_sums_t1[batch_idx, child_b].clamp(min=1.0)
    ca_norm_mean = norm_feats_t1[batch_idx, child_a, 0]
    cb_norm_mean = norm_feats_t1[batch_idx, child_b, 0]
    sister_d_mean = (ca_norm_mean - cb_norm_mean).abs().unsqueeze(-1)
    sister_d_log_sum = (ca_sum.log() - cb_sum.log()).abs().unsqueeze(-1)
    intensity_conserv = ((ca_sum + cb_sum).log() - p_sum.log()).unsqueeze(-1)
    mother_condensation = (p_norm_max - p_norm_mean).unsqueeze(-1)
    return torch.cat([sister_d_mean, sister_d_log_sum, intensity_conserv, mother_condensation], dim=-1)
