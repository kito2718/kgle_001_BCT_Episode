"""Aux-sphere occupancy as a linker pair/fork skip."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM
from cell_tracking.linker.device_const import device_const
from cell_tracking.linker.intensity import ROW_IDENTITY_RECOVERY_TOL_VOX, ROW_IDENTITY_TOL_VOX

CROP_SIZE = (16, 48, 48)
AUX_GRID = (16, 24, 24)
OCC_PAIR_DIM = 6
OCC_FORK_DIM = 6
OCC_FORK_DAUGHTER_KEYS = ("occ_fork_next", "occ_fork_has_next", "occ_fork_prev", "occ_fork_has_prev")
OCC_DIST_SCALE = 10.0
OCC_MISSING_DIST = 2.0
OCC_LOG_EPS = 1e-06


def crop_origin(query_zyx, size=CROP_SIZE):
    """Top-left voxel of a crop box centred on ``query_zyx``."""
    cz, cy, cx = (int(round(float(v))) for v in query_zyx)
    sz, sy, sx = (int(s) for s in size)
    return (cz - sz // 2, cy - sy // 2, cx - sx // 2)


OCC_KEYS = (
    "occ_coords",
    "occ_peak_next",
    "occ_peak_prev",
    "occ_next_indptr",
    "occ_next_idx",
    "occ_next_val",
    "occ_prev_indptr",
    "occ_prev_idx",
    "occ_prev_val",
)


def occupancy_frame_path(occupancy_dir: Path | str, movie: str, frame: int) -> Path:
    return Path(occupancy_dir) / movie / f"frame_{int(frame):04d}_occupancy.npz"


def world_to_aux(world_zyx, query_zyx, crop_size=CROP_SIZE):
    """World native (z,y,x) → aux-grid continuous coords, or None if outside the crop."""
    oz, oy, ox = crop_origin(query_zyx, crop_size)
    lz = float(world_zyx[0]) - oz
    ly = float(world_zyx[1]) - oy
    lx = float(world_zyx[2]) - ox
    sz, sy, sx = (int(s) for s in crop_size)
    if not (0.0 <= lz < sz and 0.0 <= ly < sy and (0.0 <= lx < sx)):
        return None
    return (lz, ly / 2.0, lx / 2.0)


def sample_map(occ, aux_zyx):
    """Trilinear sample of a (Z,Y,X) map at continuous aux coords."""
    z, y, x = aux_zyx
    return float(map_coordinates(occ, [[z], [y], [x]], order=1, mode="nearest")[0])


def sample_map_many(occ, aux_zyx):
    """``(M,)`` trilinear samples; *aux_zyx* is ``(M, 3)``."""
    pts = np.asarray(aux_zyx, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(
        map_coordinates(occ, [pts[:, 0], pts[:, 1], pts[:, 2]], order=1, mode="nearest"), dtype=np.float32
    )


def peak_world(occ, query_zyx, crop_size=CROP_SIZE):
    """Argmax of occupancy, mapped back to native world zyx (centre of the 2×2 YX block)."""
    z, y, x = (int(v) for v in np.unravel_index(int(np.argmax(occ)), occ.shape))
    oz, oy, ox = crop_origin(query_zyx, crop_size)
    return (oz + z, oy + (y * 2 + 0.5), ox + (x * 2 + 0.5))


def in_crop_aux(query_zyx, dest_zyx, crop_size=CROP_SIZE):
    """Boolean mask and aux-grid coords for dest detections inside the query crop."""
    dest = np.asarray(dest_zyx, dtype=np.float64)
    if dest.size == 0:
        return (np.zeros((0,), dtype=bool), np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32))
    oz, oy, ox = crop_origin(query_zyx, crop_size)
    sz, sy, sx = (int(s) for s in crop_size)
    rel = dest - np.array((oz, oy, ox), dtype=np.float64)
    keep = (
        (rel[:, 0] >= 0.0)
        & (rel[:, 0] < sz)
        & (rel[:, 1] >= 0.0)
        & (rel[:, 1] < sy)
        & (rel[:, 2] >= 0.0)
        & (rel[:, 2] < sx)
    )
    idx = np.nonzero(keep)[0].astype(np.int32)
    aux = np.stack([rel[keep, 0], rel[keep, 1] / 2.0, rel[keep, 2] / 2.0], axis=1)
    return (keep, aux, idx)


def empty_csr(n: int) -> dict[str, np.ndarray]:
    n = int(n)
    z = np.zeros((0,), dtype=np.int32)
    return {"indptr": np.zeros((n + 1,), dtype=np.int32), "idx": z, "val": np.zeros((0,), dtype=np.float32)}


@np.errstate(invalid="ignore")
def csr_from_maps(maps, queries, dests, crop_size=CROP_SIZE):
    """Sample each query's aux map at in-crop *dests*; return CSR + peaks."""
    n = int(np.asarray(queries).shape[0])
    if n == 0:
        return (empty_csr(0), np.zeros((0, 3), dtype=np.float32))
    dst_lists: list[np.ndarray] = []
    val_lists: list[np.ndarray] = []
    peaks = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        peaks[i] = peak_world(maps[i], queries[i], crop_size)
        _keep, aux, idx = in_crop_aux(queries[i], dests, crop_size)
        if len(idx) == 0:
            dst_lists.append(np.zeros((0,), dtype=np.int32))
            val_lists.append(np.zeros((0,), dtype=np.float32))
            continue
        dst_lists.append(idx)
        val_lists.append(sample_map_many(maps[i], aux))
    return (lists_to_csr(n, dst_lists, val_lists), peaks)


def write_occupancy_from_maps(
    path: Path | str,
    maps: np.ndarray,
    queries: np.ndarray,
    dest_next: np.ndarray,
    dest_prev: np.ndarray,
    crop_size=CROP_SIZE,
    frame: int | None = None,
    store_maps: bool = True,
    maps2: np.ndarray | None = None,
    dest_next2: np.ndarray | None = None,
    dest_prev2: np.ndarray | None = None,
) -> Path:
    """Write one occupancy sidecar frame from sigmoid aux maps ``(N, 3, Z, Y, X)``."""
    queries = np.asarray(queries, dtype=np.float64)
    maps = np.asarray(maps)
    stride2 = maps2 is not None
    if queries.shape[0] == 0:
        zero3 = np.zeros((0, 3), dtype=np.float32)
        return write_occupancy_frame(
            path,
            coords_native=queries,
            peak_next_zyx=zero3,
            peak_prev_zyx=zero3,
            next_csr=empty_csr(0),
            prev_csr=empty_csr(0),
            frame=frame,
            peak_next2_zyx=zero3 if stride2 else None,
            peak_prev2_zyx=zero3 if stride2 else None,
            next2_csr=empty_csr(0) if stride2 else None,
            prev2_csr=empty_csr(0) if stride2 else None,
        )
    if maps.ndim != 5 or maps.shape[1] < 3:
        raise ValueError(f"aux maps must be (N, >=3, Z, Y, X), got {maps.shape}")
    next_csr, peak_next = csr_from_maps(maps[:, 2], queries, dest_next, crop_size)
    prev_csr, peak_prev = csr_from_maps(maps[:, 0], queries, dest_prev, crop_size)
    next_maps_fp16 = np.asarray(maps[:, 2], dtype=np.float16) if store_maps else None
    next2_csr = prev2_csr = peak_next2 = peak_prev2 = None
    if stride2:
        m2 = np.asarray(maps2)
        if m2.shape != maps.shape:
            raise ValueError(f"stride-2 aux maps {m2.shape} must match the stride-1 maps {maps.shape}")
        if dest_next2 is None or dest_prev2 is None:
            raise ValueError("maps2 needs dest_next2 and dest_prev2")
        next2_csr, peak_next2 = csr_from_maps(m2[:, 2], queries, dest_next2, crop_size)
        prev2_csr, peak_prev2 = csr_from_maps(m2[:, 0], queries, dest_prev2, crop_size)
    return write_occupancy_frame(
        path,
        coords_native=queries,
        peak_next_zyx=peak_next,
        peak_prev_zyx=peak_prev,
        next_csr=next_csr,
        prev_csr=prev_csr,
        frame=frame,
        next_maps_fp16=next_maps_fp16,
        peak_next2_zyx=peak_next2,
        peak_prev2_zyx=peak_prev2,
        next2_csr=next2_csr,
        prev2_csr=prev2_csr,
    )


def lists_to_csr(n: int, dst_lists: list[np.ndarray], val_lists: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Pack per-source destination lists into CSR."""
    n = int(n)
    if len(dst_lists) != n or len(val_lists) != n:
        raise ValueError(f"csr lists length {len(dst_lists)}/{len(val_lists)} != n={n}")
    counts = np.fromiter((len(a) for a in dst_lists), dtype=np.int32, count=n)
    indptr = np.zeros((n + 1,), dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    e = int(indptr[-1])
    idx = np.empty((e,), dtype=np.int32)
    val = np.empty((e,), dtype=np.float32)
    for i, (d, v) in enumerate(zip(dst_lists, val_lists)):
        lo, hi = (int(indptr[i]), int(indptr[i + 1]))
        if hi > lo:
            idx[lo:hi] = np.asarray(d, dtype=np.int32)
            val[lo:hi] = np.asarray(v, dtype=np.float32)
    return {"indptr": indptr, "idx": idx, "val": val}


def write_occupancy_frame(
    path: Path | str,
    *,
    coords_native: np.ndarray,
    peak_next_zyx: np.ndarray,
    peak_prev_zyx: np.ndarray,
    next_csr: dict[str, np.ndarray],
    prev_csr: dict[str, np.ndarray],
    frame: int | None = None,
    next_maps_fp16: np.ndarray | None = None,
    peak_next2_zyx: np.ndarray | None = None,
    peak_prev2_zyx: np.ndarray | None = None,
    next2_csr: dict[str, np.ndarray] | None = None,
    prev2_csr: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write one sidecar frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(coords_native).shape[0])
    payload = {
        "coords_native": np.asarray(coords_native, dtype=np.float64),
        "peak_next_zyx": np.asarray(peak_next_zyx, dtype=np.float32).reshape(n, 3),
        "peak_prev_zyx": np.asarray(peak_prev_zyx, dtype=np.float32).reshape(n, 3),
        "next_indptr": np.asarray(next_csr["indptr"], dtype=np.int32),
        "next_idx": np.asarray(next_csr["idx"], dtype=np.int32),
        "next_val": np.asarray(next_csr["val"], dtype=np.float32),
        "prev_indptr": np.asarray(prev_csr["indptr"], dtype=np.int32),
        "prev_idx": np.asarray(prev_csr["idx"], dtype=np.int32),
        "prev_val": np.asarray(prev_csr["val"], dtype=np.float32),
        "aux_grid": np.asarray(AUX_GRID, dtype=np.int32),
    }
    if next2_csr is not None or prev2_csr is not None:
        if next2_csr is None or prev2_csr is None:
            raise ValueError("the stride-2 horizon needs both next2_csr and prev2_csr")
        payload.update(
            {
                "peak_next2_zyx": np.asarray(peak_next2_zyx, dtype=np.float32).reshape(n, 3),
                "peak_prev2_zyx": np.asarray(peak_prev2_zyx, dtype=np.float32).reshape(n, 3),
                "next2_indptr": np.asarray(next2_csr["indptr"], dtype=np.int32),
                "next2_idx": np.asarray(next2_csr["idx"], dtype=np.int32),
                "next2_val": np.asarray(next2_csr["val"], dtype=np.float32),
                "prev2_indptr": np.asarray(prev2_csr["indptr"], dtype=np.int32),
                "prev2_idx": np.asarray(prev2_csr["idx"], dtype=np.int32),
                "prev2_val": np.asarray(prev2_csr["val"], dtype=np.float32),
            }
        )
    if frame is not None:
        payload["frame"] = np.int32(frame)
    if next_maps_fp16 is not None:
        nm = np.asarray(next_maps_fp16, dtype=np.float16)
        if nm.shape[0] != n:
            raise ValueError(f"next_maps_fp16 rows {nm.shape[0]} != n={n}")
        payload["next_maps_fp16"] = nm
        if nm.ndim >= 3:
            payload["aux_grid"] = np.asarray(nm.shape[-3:], dtype=np.int32)
    np.savez(path, **payload)
    return path


def read_occupancy_frame(path: Path | str) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        out = {
            "coords_native": np.asarray(data["coords_native"], dtype=np.float64),
            "peak_next_zyx": np.asarray(data["peak_next_zyx"], dtype=np.float32),
            "peak_prev_zyx": np.asarray(data["peak_prev_zyx"], dtype=np.float32),
            "next_indptr": np.asarray(data["next_indptr"], dtype=np.int32),
            "next_idx": np.asarray(data["next_idx"], dtype=np.int32),
            "next_val": np.asarray(data["next_val"], dtype=np.float32),
            "prev_indptr": np.asarray(data["prev_indptr"], dtype=np.int32),
            "prev_idx": np.asarray(data["prev_idx"], dtype=np.int32),
            "prev_val": np.asarray(data["prev_val"], dtype=np.float32),
        }
        if "next_maps_fp16" in data:
            out["next_maps_fp16"] = np.asarray(data["next_maps_fp16"], dtype=np.float16)
        if "next2_indptr" in data:
            out.update(
                {
                    "peak_next2_zyx": np.asarray(data["peak_next2_zyx"], dtype=np.float32),
                    "peak_prev2_zyx": np.asarray(data["peak_prev2_zyx"], dtype=np.float32),
                    "next2_indptr": np.asarray(data["next2_indptr"], dtype=np.int32),
                    "next2_idx": np.asarray(data["next2_idx"], dtype=np.int32),
                    "next2_val": np.asarray(data["next2_val"], dtype=np.float32),
                    "prev2_indptr": np.asarray(data["prev2_indptr"], dtype=np.int32),
                    "prev2_idx": np.asarray(data["prev2_idx"], dtype=np.int32),
                    "prev2_val": np.asarray(data["prev2_val"], dtype=np.float32),
                }
            )
        return out


def _occupancy_payload(got: dict[str, np.ndarray], path: Path, cache_coords: np.ndarray) -> dict[str, np.ndarray]:
    sidecar_coords = np.asarray(got["coords_native"], dtype=np.float64)
    cache_coords = np.asarray(cache_coords, dtype=np.float64)
    n = int(cache_coords.shape[0])
    n_sidecar = int(sidecar_coords.shape[0])
    if n_sidecar != n:
        raise ValueError(f"occupancy sidecar {path} has {n_sidecar} detections, cache has {n}")
    if n:
        dist = np.linalg.norm(sidecar_coords - cache_coords, axis=1)
        bad = int((dist > ROW_IDENTITY_TOL_VOX).sum())
        if bad:
            raise ValueError(
                f"occupancy sidecar {path} ROW IDENTITY mismatch against the cache: {bad}/{n} rows differed by more than {ROW_IDENTITY_TOL_VOX} voxels (worst: row {int(np.argmax(dist))}, {dist.max():.3f} vox)"
            )
    out = {
        "occ_coords": sidecar_coords,
        "occ_peak_next": got["peak_next_zyx"],
        "occ_peak_prev": got["peak_prev_zyx"],
        "occ_next_indptr": got["next_indptr"],
        "occ_next_idx": got["next_idx"],
        "occ_next_val": got["next_val"],
        "occ_prev_indptr": got["prev_indptr"],
        "occ_prev_idx": got["prev_idx"],
        "occ_prev_val": got["prev_val"],
    }
    if "next_maps_fp16" in got:
        out["occ_next_maps"] = np.asarray(got["next_maps_fp16"], dtype=np.float32)
    if "next2_indptr" in got:
        out.update(
            {
                "occ_peak_next2": got["peak_next2_zyx"],
                "occ_peak_prev2": got["peak_prev2_zyx"],
                "occ_next2_indptr": got["next2_indptr"],
                "occ_next2_idx": got["next2_idx"],
                "occ_next2_val": got["next2_val"],
                "occ_prev2_indptr": got["prev2_indptr"],
                "occ_prev2_idx": got["prev2_idx"],
                "occ_prev2_val": got["prev2_val"],
            }
        )
    return out


def attach_occupancy_from_dir(
    cache: dict, occupancy_dir: Path | str | list | tuple, movie: str, *, routing: str = "shared"
) -> dict:
    """Attach the occupancy sidecar frame matching *cache*'s own frame index."""
    dirs = list(occupancy_dir) if isinstance(occupancy_dir, (list, tuple)) else [occupancy_dir]
    if not dirs:
        raise ValueError("occupancy_dir is empty")
    routing = str(routing)
    if routing not in ("shared", "matched"):
        raise ValueError(f"routing must be 'shared' or 'matched', got {routing!r}")
    frame = int(np.asarray(cache["frame"]))
    if len(dirs) == 1:
        path = occupancy_frame_path(dirs[0], movie, frame)
        got = read_occupancy_frame(path)
        if got is None:
            raise FileNotFoundError(f"no occupancy sidecar at {path}")
        out = dict(cache)
        out.update(_occupancy_payload(got, path, cache["coords_native"]))
        return out
    if routing != "matched":
        raise ValueError("multiple occupancy dirs require routing='matched'")
    cache_coords = np.asarray(cache["coords_native"], dtype=np.float64)
    members = []
    for d in dirs:
        path = occupancy_frame_path(d, movie, frame)
        got = read_occupancy_frame(path)
        if got is None:
            raise FileNotFoundError(f"no occupancy sidecar at {path}")
        members.append(_occupancy_payload(got, path, cache_coords))
    out = dict(cache)
    out["occ_members"] = members
    return out


def _row_map(src_coords: np.ndarray, dst_coords: np.ndarray) -> np.ndarray:
    """Map each *src* row onto a *dst* row, or −1 if unmatched."""
    src = np.asarray(src_coords, dtype=np.float64)
    dst = np.asarray(dst_coords, dtype=np.float64)
    n_src = int(src.shape[0])
    n_dst = int(dst.shape[0])
    if n_src == 0:
        return np.zeros((0,), dtype=np.int32)
    if n_dst == 0:
        return np.full((n_src,), -1, dtype=np.int32)
    if n_src == n_dst:
        dist = np.linalg.norm(src - dst, axis=1)
        if not bool((dist > ROW_IDENTITY_TOL_VOX).any()):
            return np.arange(n_src, dtype=np.int32)
    tree = cKDTree(dst)
    d, j = tree.query(src)
    return np.where(
        np.asarray(d) <= ROW_IDENTITY_RECOVERY_TOL_VOX,
        np.asarray(j, dtype=np.int32),
        np.full((n_src,), -1, dtype=np.int32),
    )


def _scatter_csr(
    indptr: np.ndarray,
    idx: np.ndarray,
    val: np.ndarray,
    src_map: np.ndarray,
    dst_map: np.ndarray,
    n_src: int,
    n_dst: int,
) -> tuple[np.ndarray, np.ndarray]:
    occ = np.zeros((n_src, n_dst), dtype=np.float32)
    has = np.zeros((n_src, n_dst), dtype=np.float32)
    n_full = int(src_map.shape[0])
    indptr = np.asarray(indptr, dtype=np.int32)
    idx = np.asarray(idx, dtype=np.int32)
    val = np.asarray(val, dtype=np.float32)
    if int(indptr.shape[0]) != n_full + 1:
        raise ValueError(f"occupancy CSR indptr length {int(indptr.shape[0])} != n_full+1={n_full + 1}")
    n_dst_full = int(dst_map.shape[0])
    for i_full, i in enumerate(src_map.tolist()):
        if i < 0:
            continue
        lo, hi = (int(indptr[i_full]), int(indptr[i_full + 1]))
        for e in range(lo, hi):
            j_full = int(idx[e])
            if j_full < 0 or j_full >= n_dst_full:
                continue
            j = int(dst_map[j_full])
            if j < 0:
                continue
            occ[i, j] = float(val[e])
            has[i, j] = 1.0
    return (occ, has)


def empty_occupancy_arrays(n_t: int, n_t1: int) -> dict[str, np.ndarray]:
    """Zeros for a gap>1 pair or a missing neighbour: has=0, occupancy-blind."""
    n_t, n_t1 = (int(n_t), int(n_t1))
    return {
        "occ_next": np.zeros((n_t, n_t1), dtype=np.float32),
        "has_next": np.zeros((n_t, n_t1), dtype=np.float32),
        "occ_prev": np.zeros((n_t, n_t1), dtype=np.float32),
        "has_prev": np.zeros((n_t, n_t1), dtype=np.float32),
        "peak_next": np.zeros((n_t, 3), dtype=np.float32),
    }


def pad_occupancy_child_dim(arr: dict[str, np.ndarray], n_t1: int) -> dict[str, np.ndarray]:
    """Pad dense occupancy matrices after hardneg clones add children."""
    pad = int(n_t1) - int(arr["occ_next"].shape[1])
    if pad > 0:
        for key in ("occ_next", "has_next", "occ_prev", "has_prev"):
            arr[key] = np.pad(arr[key], ((0, 0), (0, pad)))
    elif pad < 0:
        raise ValueError(f"occupancy t+1 width {arr['occ_next'].shape[1]} exceeds coords_t1 {n_t1}")
    return arr


def fork_occupancy_from_pair(arr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename pair arrays to the fork-daughter keys occupancy_forward_kw emits."""
    return {
        "occ_fork_next": arr["occ_next"],
        "occ_fork_has_next": arr["has_next"],
        "occ_fork_prev": arr["occ_prev"],
        "occ_fork_has_prev": arr["has_prev"],
    }


def _sidecar_view(d: dict[str, np.ndarray], stride: str = "") -> dict[str, np.ndarray] | None:
    """Reader-named view of one sidecar horizon, or None when that horizon is absent."""
    if "occ_coords" in d:
        coords = d["occ_coords"]
        names = {
            "peak_next_zyx": f"occ_peak_next{stride}",
            "peak_prev_zyx": f"occ_peak_prev{stride}",
            "next_indptr": f"occ_next{stride}_indptr",
            "next_idx": f"occ_next{stride}_idx",
            "next_val": f"occ_next{stride}_val",
            "prev_indptr": f"occ_prev{stride}_indptr",
            "prev_idx": f"occ_prev{stride}_idx",
            "prev_val": f"occ_prev{stride}_val",
        }
        maps_key = "occ_next_maps"
    elif "peak_next_zyx" in d and "next_indptr" in d:
        coords = d["coords_native"]
        names = {
            "peak_next_zyx": f"peak_next{stride}_zyx",
            "peak_prev_zyx": f"peak_prev{stride}_zyx",
            "next_indptr": f"next{stride}_indptr",
            "next_idx": f"next{stride}_idx",
            "next_val": f"next{stride}_val",
            "prev_indptr": f"prev{stride}_indptr",
            "prev_idx": f"prev{stride}_idx",
            "prev_val": f"prev{stride}_val",
        }
        maps_key = "next_maps"
    else:
        raise ValueError(
            "occupancy sidecar dict needs either occ_* keys (attach_occupancy_from_dir) or peak_next_zyx/next_indptr (read_occupancy_frame)"
        )
    if stride and any((key not in d for key in names.values())):
        return None
    view = {"coords_native": coords}
    view.update({dst: d[src] for dst, src in names.items()})
    if not stride and maps_key in d:
        view["next_maps"] = d[maps_key]
    return view


def occupancy_pair_arrays(
    occ_t: dict[str, np.ndarray] | None,
    occ_t1: dict[str, np.ndarray] | None,
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    *,
    gap: int = 1,
) -> dict[str, np.ndarray]:
    """Dense (N_t, N_t1) occupancy samples for one frame pair."""
    coords_t = np.asarray(coords_t, dtype=np.float64)
    coords_t1 = np.asarray(coords_t1, dtype=np.float64)
    n_t = int(coords_t.shape[0])
    n_t1 = int(coords_t1.shape[0])
    if int(gap) not in (1, 2) or occ_t is None or occ_t1 is None:
        return empty_occupancy_arrays(n_t, n_t1)
    stride = "" if int(gap) == 1 else "2"
    s0, s1 = (_sidecar_view(occ_t, stride), _sidecar_view(occ_t1, stride))
    if s0 is None or s1 is None:
        return empty_occupancy_arrays(n_t, n_t1)
    src_map = _row_map(s0["coords_native"], coords_t)
    dst_map = _row_map(s1["coords_native"], coords_t1)
    occ_next, has_next = _scatter_csr(s0["next_indptr"], s0["next_idx"], s0["next_val"], src_map, dst_map, n_t, n_t1)
    occ_prev, has_prev = _scatter_csr(s1["prev_indptr"], s1["prev_idx"], s1["prev_val"], dst_map, src_map, n_t1, n_t)
    occ_prev = occ_prev.T
    has_prev = has_prev.T
    peak_next = np.zeros((n_t, 3), dtype=np.float32)
    peaks = np.asarray(s0["peak_next_zyx"], dtype=np.float32)
    for i_full, i in enumerate(src_map.tolist()):
        if i >= 0:
            peak_next[i] = peaks[i_full]
    next_maps = None
    raw_maps = s0.get("next_maps")
    if raw_maps is not None:
        raw_maps = np.asarray(raw_maps, dtype=np.float32)
        zyx = raw_maps.shape[1:]
        next_maps = np.zeros((n_t,) + zyx, dtype=np.float32)
        for i_full, i in enumerate(src_map.tolist()):
            if i >= 0:
                next_maps[i] = raw_maps[i_full]
    out = {
        "occ_next": occ_next,
        "has_next": has_next,
        "occ_prev": occ_prev,
        "has_prev": has_prev,
        "peak_next": peak_next,
    }
    if next_maps is not None:
        out["next_maps"] = next_maps
    return out


def occupancy_forward_kw(
    occ_t: dict[str, np.ndarray] | None,
    occ_t1: dict[str, np.ndarray] | None,
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    *,
    gap: int = 1,
    fork_occ_t: dict[str, np.ndarray] | None = None,
    fork_occ_t1: dict[str, np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """Torch tensors for ``model(..., occ_next=..., has_next=..., ...)``."""
    if isinstance(occ_t, dict) and "occ_members" in occ_t:
        members_t = occ_t["occ_members"]
        members_t1 = occ_t1["occ_members"] if isinstance(occ_t1, dict) else []
        if len(members_t) != len(members_t1):
            raise ValueError(f"occupancy member count mismatch: t has {len(members_t)}, t+1 has {len(members_t1)}")
        if fork_occ_t is not None or fork_occ_t1 is not None:
            raise ValueError("occupancy_fork_daughter is not defined for matched occupancy ensembles")
        parts = [
            occupancy_pair_arrays(members_t[k], members_t1[k], coords_t, coords_t1, gap=gap)
            for k in range(len(members_t))
        ]
        if not parts:
            arr = empty_occupancy_arrays(int(coords_t.shape[0]), int(coords_t1.shape[0]))
        else:
            arr = {key: np.stack([p[key] for p in parts], axis=0) for key in parts[0]}
        return {name: torch.from_numpy(value) for name, value in arr.items()}
    arr = occupancy_pair_arrays(occ_t, occ_t1, coords_t, coords_t1, gap=gap)
    out = {name: torch.from_numpy(value) for name, value in arr.items()}
    if fork_occ_t is not None or fork_occ_t1 is not None:
        fork = occupancy_pair_arrays(fork_occ_t, fork_occ_t1, coords_t, coords_t1, gap=gap)
        out.update({k: torch.from_numpy(v) for k, v in fork_occupancy_from_pair(fork).items()})
    return out


def occupancy_pair_features(
    occ_next: torch.Tensor,
    has_next: torch.Tensor,
    occ_prev: torch.Tensor,
    has_prev: torch.Tensor,
    peak_next: torch.Tensor,
    coords_t1: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    voxel_size=NATIVE_VOXEL_UM,
    dist_scale: float = OCC_DIST_SCALE,
    eps: float = OCC_LOG_EPS,
) -> torch.Tensor:
    """``(B, N_t, N_t1, OCC_PAIR_DIM)`` occupancy skip for every pair."""
    has_n = has_next.to(dtype=occ_next.dtype)
    has_p = has_prev.to(dtype=occ_prev.dtype)
    occ_n = occ_next * has_n
    occ_p = occ_prev * has_p
    cand = pair_mask.bool() & has_n.bool()
    filled = occ_n.masked_fill(~cand, 0.0)
    row_max, argmax = filled.max(dim=-1)
    second = filled.clone()
    second.scatter_(-1, argmax.unsqueeze(-1), 0.0)
    row_second = second.max(dim=-1).values
    n_t1 = occ_next.shape[-1]
    is_max = torch.arange(n_t1, device=occ_next.device).view(1, 1, n_t1) == argmax.unsqueeze(-1)
    distractor = torch.where(is_max, row_second.unsqueeze(-1), row_max.unsqueeze(-1))
    n_cand = cand.sum(dim=-1, keepdim=True)
    has_other = n_cand > 1
    log_contrast = torch.log(occ_n.clamp_min(eps) / distractor.clamp_min(eps))
    log_contrast = torch.where(has_other & has_n.bool(), log_contrast, torch.zeros_like(log_contrast))
    vs = device_const(voxel_size, occ_next.dtype, occ_next.device)
    peak_um = peak_next * vs
    child_um = coords_t1 * vs
    dist = (child_um.unsqueeze(1) - peak_um.unsqueeze(2)).norm(dim=-1) / float(dist_scale)
    dist = torch.where(has_n.bool(), dist, torch.full_like(dist, OCC_MISSING_DIST))
    return torch.stack([occ_n, log_contrast, dist, occ_p, has_n, has_p], dim=-1)


def _sample_next_maps_at_mid(
    next_maps: torch.Tensor, coords_t: torch.Tensor, coords_t1: torch.Tensor, fork_pairs: torch.Tensor
) -> torch.Tensor:
    """``(B, P)`` sample of each parent's t+1 map at the daughter midpoint."""
    import torch.nn.functional as F

    batch, n_pairs, _ = fork_pairs.shape
    device = next_maps.device
    batch_idx = torch.arange(batch, device=device).view(batch, 1).expand(-1, n_pairs)
    src = fork_pairs[..., 0].long().clamp(min=0)
    d1 = fork_pairs[..., 1].long().clamp(min=0)
    d2 = fork_pairs[..., 2].long().clamp(min=0)
    valid = (fork_pairs[..., 1] >= 0) & (fork_pairs[..., 2] >= 0)
    parent = coords_t[batch_idx, src]
    mid = (coords_t1[batch_idx, d1] + coords_t1[batch_idx, d2]) * 0.5
    z, y, x = (int(v) for v in next_maps.shape[-3:])
    sz, sy, sx = (float(z), float(y * 2), float(x * 2))
    oz = parent[..., 0] - sz // 2
    oy = parent[..., 1] - sy // 2
    ox = parent[..., 2] - sx // 2
    lz = mid[..., 0] - oz
    ly = mid[..., 1] - oy
    lx = mid[..., 2] - ox
    inside = (lz >= 0) & (lz < sz) & (ly >= 0) & (ly < sy) & (lx >= 0) & (lx < sx)
    az = lz
    ay = ly / 2.0
    ax = lx / 2.0
    maps = next_maps[batch_idx, src]
    flat = maps.reshape(batch * n_pairs, 1, z, y, x)

    def _norm(coord: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coord)
        return 2.0 * coord / float(size - 1) - 1.0

    grid = torch.stack([_norm(ax, x), _norm(ay, y), _norm(az, z)], dim=-1)
    grid = grid.view(batch * n_pairs, 1, 1, 1, 3)
    sampled = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = sampled.view(batch, n_pairs)
    mask = inside & valid
    return out * mask.to(dtype=out.dtype)


def occupancy_fork_features(
    occ_next: torch.Tensor,
    has_next: torch.Tensor,
    fork_pairs: torch.Tensor,
    *,
    coords_t: torch.Tensor | None = None,
    coords_t1: torch.Tensor | None = None,
    next_maps: torch.Tensor | None = None,
    occ_prev: torch.Tensor | None = None,
    has_prev: torch.Tensor | None = None,
    daughter_query: bool = False,
    eps: float = OCC_LOG_EPS,
) -> torch.Tensor:
    """``(B, P, OCC_FORK_DIM)`` fork occupancy skip for (parent, D1, D2)."""
    batch, n_pairs, _ = fork_pairs.shape
    batch_idx = torch.arange(batch, device=fork_pairs.device).view(batch, 1).expand(-1, n_pairs)
    src = fork_pairs[..., 0].long().clamp(min=0)
    d1 = fork_pairs[..., 1].long().clamp(min=0)
    d2 = fork_pairs[..., 2].long().clamp(min=0)
    valid = (fork_pairs[..., 1] >= 0) & (fork_pairs[..., 2] >= 0)
    o1 = occ_next[batch_idx, src, d1] * has_next[batch_idx, src, d1].to(occ_next.dtype)
    o2 = occ_next[batch_idx, src, d2] * has_next[batch_idx, src, d2].to(occ_next.dtype)
    o1 = o1 * valid.to(o1.dtype)
    o2 = o2 * valid.to(o2.dtype)
    lo = torch.minimum(o1, o2)
    if daughter_query:
        if occ_prev is None or has_prev is None:
            raise ValueError("daughter_query=True requires occ_prev and has_prev")
        p1 = occ_prev[batch_idx, src, d1] * has_prev[batch_idx, src, d1].to(occ_prev.dtype)
        p2 = occ_prev[batch_idx, src, d2] * has_prev[batch_idx, src, d2].to(occ_prev.dtype)
        p1 = p1 * valid.to(p1.dtype)
        p2 = p2 * valid.to(p2.dtype)
        has_both = has_prev[batch_idx, src, d1].bool() & has_prev[batch_idx, src, d2].bool() & valid
        lo_p = torch.minimum(p1, p2)
        ad_p = (p1 - p2).abs()
        mask = has_both.to(lo_p.dtype)
        return torch.stack([o1, o2, lo, (o1 - o2).abs(), lo_p * mask, ad_p * mask], dim=-1)
    o_mid = torch.zeros_like(o1)
    log_valley = torch.zeros_like(o1)
    if next_maps is not None and coords_t is not None and (coords_t1 is not None):
        o_mid = _sample_next_maps_at_mid(next_maps, coords_t, coords_t1, fork_pairs)
        has_both = has_next[batch_idx, src, d1].bool() & has_next[batch_idx, src, d2].bool() & valid
        log_valley = torch.log(lo.clamp_min(eps)) - torch.log(o_mid.clamp_min(eps))
        mask = has_both.to(o_mid.dtype)
        o_mid = o_mid * mask
        log_valley = log_valley * mask
    return torch.stack([o1, o2, lo, (o1 - o2).abs(), o_mid, log_valley], dim=-1)
