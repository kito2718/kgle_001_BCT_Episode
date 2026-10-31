"""Second-order motion features for the edge head."""

from __future__ import annotations
import numpy as np
from cell_tracking.common.config import SCALE as NATIVE_VOXEL_UM

MOTION_PAIR_NAMES = (
    "dist_um",
    "speed_in",
    "accel_um",
    "cos_motion",
    "has_motion",
    "rank_src_for_child",
    "rank_child_for_src",
)
MOTION_PAIR_DIM = len(MOTION_PAIR_NAMES)
FLOW_PAIR_NAMES = ("res_z", "res_y", "res_x", "res_um", "flow_um")
FLOW_PAIR_DIM = len(FLOW_PAIR_NAMES)
UM_DIST_PAIR_NAMES = ("centre_dist_um",)
UM_DIST_PAIR_DIM = len(UM_DIST_PAIR_NAMES)
DEFAULT_MAX_LINK_UM = 8.0


def nearest_neighbor_index(
    query_coords: np.ndarray,
    key_coords: np.ndarray | None,
    *,
    voxel_size=NATIVE_VOXEL_UM,
    max_um: float = DEFAULT_MAX_LINK_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest *key* index per query row."""
    query = np.asarray(query_coords, np.float64).reshape(-1, 3)
    n = len(query)
    idx = np.zeros(n, np.int64)
    has = np.zeros(n, bool)
    if key_coords is None or n == 0:
        return (idx, has)
    key = np.asarray(key_coords, np.float64).reshape(-1, 3)
    if len(key) == 0:
        return (idx, has)
    vs = np.asarray(voxel_size, np.float64)
    cur_um = query * vs
    key_um = key * vs
    d = np.linalg.norm(cur_um[:, None, :] - key_um[None, :, :], axis=-1)
    j = d.argmin(axis=1)
    dmin = d[np.arange(n), j]
    ok = dmin <= float(max_um)
    idx[ok] = j[ok]
    has[ok] = True
    return (idx, has)


def incoming_velocity(
    coords_t: np.ndarray,
    coords_prev: np.ndarray | None,
    *,
    voxel_size=NATIVE_VOXEL_UM,
    max_um: float = DEFAULT_MAX_LINK_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node incoming velocity in µm/frame, from the nearest ``t-1`` detection."""
    coords_t = np.asarray(coords_t, np.float64).reshape(-1, 3)
    n = len(coords_t)
    vel = np.zeros((n, 3), np.float64)
    idx, has = nearest_neighbor_index(coords_t, coords_prev, voxel_size=voxel_size, max_um=max_um)
    if not has.any():
        return (vel, has)
    vs = np.asarray(voxel_size, np.float64)
    prev = np.asarray(coords_prev, np.float64).reshape(-1, 3)
    vel[has] = coords_t[has] * vs - prev[idx[has]] * vs
    return (vel, has)


FLOW_KINDS = (
    "local",
    "global",
    "rigid",
    "affine",
    "local_mean",
    "local_affine",
    "local_affine_robust",
    "local_c2f",
    "local_c2f_s",
    "local_c2f_13",
)
AFFINE_MIN_ANCHORS = 12
C2F_GRID_UM = 2.0
C2F_MAX_SHIFT_UM = 30.0
C2F_MIN_POINTS = 5
C2F_S_BIN = 0.25
C2F_S_WINDOW = 4.0
C2F_FINE_UM = 13.0


def nucleus_spacing_um(pts_um: np.ndarray) -> float:
    """Median distance from each point to its nearest other point, in µm."""
    pts = np.asarray(pts_um, np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return float("nan")
    nn = np.empty(n, np.float64)
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        dist = np.linalg.norm(pts[lo:hi, None, :] - pts[None, :, :], axis=-1)
        dist[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        nn[lo:hi] = dist.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return float("nan")
    return float(np.median(nn))


def coarse_shift_um(
    a_um: np.ndarray, b_um: np.ndarray, *, max_shift_um: float = C2F_MAX_SHIFT_UM, grid_um: float = C2F_GRID_UM
) -> np.ndarray:
    """One global translation ``a → b`` in µm, with NO radius cap on the pairing."""
    a_um = np.asarray(a_um, np.float64).reshape(-1, 3)
    b_um = np.asarray(b_um, np.float64).reshape(-1, 3)
    if len(a_um) < C2F_MIN_POINTS or len(b_um) < C2F_MIN_POINTS:
        return np.zeros(3, np.float64)
    grid = float(grid_um)
    if grid <= 0:
        return np.zeros(3, np.float64)
    half = round(float(max_shift_um) / grid)
    n = 2 * half + 1
    hist = np.zeros(n**3, np.int64)
    for lo in range(0, len(a_um), 512):
        d = b_um[None, :, :] - a_um[lo : lo + 512, None, :]
        b_idx = np.rint(d.reshape(-1, 3) / grid).astype(np.int64) + half
        keep = ((b_idx >= 0) & (b_idx < n)).all(axis=1)
        if not keep.any():
            continue
        b_idx = b_idx[keep]
        flat = (b_idx[:, 0] * n + b_idx[:, 1]) * n + b_idx[:, 2]
        hist += np.bincount(flat, minlength=n**3)
    if not hist.any():
        return np.zeros(3, np.float64)
    peak = np.unravel_index(int(hist.argmax()), (n, n, n))
    sl = tuple((slice(max(p - 1, 0), min(p + 2, n)) for p in peak))
    block = hist.reshape(n, n, n)[sl].astype(np.float64)
    grids = np.meshgrid(*[np.arange(s.start, s.stop, dtype=np.float64) for s in sl], indexing="ij")
    total = block.sum()
    centre = np.array([float((g * block).sum() / total) for g in grids])
    return (centre - half) * grid


def mutual_nn_displacements(
    a_um: np.ndarray,
    b_um: np.ndarray,
    *,
    max_um: float = DEFAULT_MAX_LINK_UM,
    return_index: bool = False,
    return_target: bool = False,
) -> tuple:
    """``(anchors, displacements)`` for mutual-nearest-neighbour pairs, both in µm."""
    a_um = np.asarray(a_um, np.float64).reshape(-1, 3)
    b_um = np.asarray(b_um, np.float64).reshape(-1, 3)
    empty = (np.zeros((0, 3)), np.zeros((0, 3)))
    if return_index:
        empty = (*empty, np.zeros(0, np.int64))
    if return_target:
        empty = (*empty, np.zeros(0, np.int64))
    if len(a_um) == 0 or len(b_um) == 0:
        return empty
    d = np.linalg.norm(a_um[:, None, :] - b_um[None, :, :], axis=-1)
    i2j = d.argmin(axis=1)
    j2i = d.argmin(axis=0)
    idx = np.arange(len(a_um))
    keep = (j2i[i2j] == idx) & (d[idx, i2j] <= float(max_um))
    if not keep.any():
        return empty
    anchors, disp = (a_um[keep], b_um[i2j[keep]] - a_um[keep])
    out = (anchors, disp)
    if return_index:
        out = (*out, idx[keep])
    if return_target:
        out = (*out, i2j[keep])
    return out


def _weighted_median_vectors(disp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Component-wise weighted median of ``(N, 3)`` displacements."""
    disp = np.asarray(disp, np.float64).reshape(-1, 3)
    w = np.maximum(np.asarray(weights, np.float64).reshape(-1), 1e-08)
    if len(disp) == 0:
        return np.zeros(3, np.float64)
    if len(disp) == 1:
        return disp[0].copy()
    out = np.empty(3, np.float64)
    for c in range(3):
        order = np.argsort(disp[:, c], kind="mergesort")
        cw = np.cumsum(w[order])
        out[c] = disp[order[int(np.searchsorted(cw, 0.5 * cw[-1]))], c]
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid ``dst ≈ src @ R.T + t`` (R is 3×3, t is (3,))."""
    mu_s, mu_d = (src.mean(0), dst.mean(0))
    x, y = (src - mu_s, dst - mu_d)
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    return (r, t)


def residual_pair_um(
    coords_t: np.ndarray, coords_t1: np.ndarray, flow_t: np.ndarray, *, voxel_size=NATIVE_VOXEL_UM
) -> np.ndarray:
    """``(N_t, N_t1)`` ``||x_j − (x_i + flow_i)||`` in µm."""
    vs = np.asarray(voxel_size, np.float64)
    pred = np.asarray(coords_t, np.float64).reshape(-1, 3) * vs + np.asarray(flow_t, np.float64).reshape(-1, 3)
    tgt = np.asarray(coords_t1, np.float64).reshape(-1, 3) * vs
    if len(pred) == 0 or len(tgt) == 0:
        return np.zeros((len(pred), len(tgt)), np.float64)
    return np.linalg.norm(tgt[None, :, :] - pred[:, None, :], axis=-1)


def _knn_index(q_um: np.ndarray, anchors: np.ndarray, k: int) -> np.ndarray:
    """``(n_out, kk)`` indices of the *k* anchors nearest each query point."""
    d = np.linalg.norm(q_um[:, None, :] - anchors[None, :, :], axis=-1)
    kk = min(int(k), len(anchors))
    return np.argpartition(d, kk - 1, axis=1)[:, :kk]


def _solve_affine(nbr: np.ndarray, disp: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """``(n_out, 4, 3)`` coefficients of ``disp ≈ [nbr, 1] @ coef``, batched."""
    x = np.concatenate([nbr, np.ones((*nbr.shape[:2], 1))], axis=2)
    if w is not None:
        xw = x * w[..., None]
        return np.linalg.pinv(xw.transpose(0, 2, 1) @ x) @ (xw.transpose(0, 2, 1) @ disp)
    return np.linalg.pinv(x.transpose(0, 2, 1) @ x) @ (x.transpose(0, 2, 1) @ disp)


def _local_first_order(q_um: np.ndarray, anchors: np.ndarray, disp: np.ndarray, *, k: int, robust: bool) -> np.ndarray:
    """``(n_out, 3)`` displacement at each query from a per-neighbourhood affine fit."""
    nn = _knn_index(q_um, anchors, k)
    nbr = anchors[nn] - q_um[:, None, :]
    d = disp[nn]
    mean = d.mean(axis=1)
    if nn.shape[1] < AFFINE_MIN_ANCHORS:
        return mean
    coef = _solve_affine(nbr, d, None)
    if not robust:
        return coef[:, 3, :]
    resid = np.linalg.norm(d - np.concatenate([nbr, np.ones((*nbr.shape[:2], 1))], axis=2) @ coef, axis=2)
    scale = np.maximum(1.5 * np.median(resid, axis=1, keepdims=True), 1e-09)
    u = np.minimum(resid / scale, 1.0)
    w = (1.0 - u**2) ** 2
    rw = _solve_affine(nbr, d, w)
    enough = (w > 0).sum(axis=1) >= AFFINE_MIN_ANCHORS
    return np.where(enough[:, None], rw[:, 3, :], coef[:, 3, :])


def local_tissue_flow(
    coords_t: np.ndarray,
    coords_t1: np.ndarray,
    *,
    voxel_size=NATIVE_VOXEL_UM,
    k: int = 20,
    max_um: float = DEFAULT_MAX_LINK_UM,
    weights: np.ndarray | None = None,
    kind: str = "local",
    query_um: np.ndarray | None = None,
) -> np.ndarray:
    """``(N_t, 3)`` local bulk displacement in µm, estimated from detections alone."""
    if kind not in FLOW_KINDS:
        raise ValueError(f"kind must be one of {FLOW_KINDS}, got {kind!r}")
    coords_t = np.asarray(coords_t, np.float64).reshape(-1, 3)
    n = len(coords_t)
    vs = np.asarray(voxel_size, np.float64)
    a_um = coords_t * vs
    q_um = a_um if query_um is None else np.asarray(query_um, np.float64).reshape(-1, 3)
    n_out = len(q_um)
    if n == 0:
        return np.zeros((n_out, 3))
    w_src = None if weights is None else np.asarray(weights, np.float64).reshape(-1)
    if w_src is not None and len(w_src) != n:
        raise ValueError(f"weights length {len(w_src)} != n_t {n}")
    b_um = np.asarray(coords_t1, np.float64).reshape(-1, 3) * vs
    if kind in ("local_c2f", "local_c2f_s", "local_c2f_13"):
        fine_max = C2F_FINE_UM if kind == "local_c2f_13" else max_um
        if kind == "local_c2f_s":
            spacing = nucleus_spacing_um(a_um)
            if np.isfinite(spacing) and spacing > 0:
                shift = coarse_shift_um(a_um, b_um, max_shift_um=C2F_S_WINDOW * spacing, grid_um=C2F_S_BIN * spacing)
                fine_max = np.inf
            else:
                shift = coarse_shift_um(a_um, b_um)
        else:
            shift = coarse_shift_um(a_um, b_um)
        fine = local_tissue_flow(
            coords_t + shift / vs,
            coords_t1,
            voxel_size=voxel_size,
            k=k,
            max_um=fine_max,
            weights=weights,
            kind="local",
            query_um=None if query_um is None else q_um + shift,
        )
        return fine + shift
    anchors, disp, idx = mutual_nn_displacements(a_um, b_um, max_um=max_um, return_index=True)
    if len(anchors) == 0:
        return np.zeros((n_out, 3))

    def _global_vec() -> np.ndarray:
        if w_src is None:
            return np.median(disp, axis=0)
        return _weighted_median_vectors(disp, w_src[idx])

    if kind == "global" or (kind == "local" and len(anchors) < 5):
        return np.repeat(_global_vec()[None, :], n_out, axis=0)
    if kind == "rigid":
        if len(anchors) < 3:
            return np.repeat(_global_vec()[None, :], n_out, axis=0)
        rot, t = _kabsch(anchors, anchors + disp)
        return q_um @ rot.T + t - q_um
    if kind == "affine":
        if len(anchors) < 4:
            return np.repeat(_global_vec()[None, :], n_out, axis=0)
        dst = anchors + disp
        x = np.concatenate([anchors, np.ones((len(anchors), 1))], axis=1)
        m, _, _, _ = np.linalg.lstsq(x, dst, rcond=None)
        pred = np.concatenate([q_um, np.ones((n_out, 1))], axis=1) @ m
        return pred - q_um
    if kind == "local_mean":
        return disp[_knn_index(q_um, anchors, k)].mean(axis=1)
    if kind in ("local_affine", "local_affine_robust"):
        return _local_first_order(q_um, anchors, disp, k=k, robust=kind == "local_affine_robust")
    nn = _knn_index(q_um, anchors, k)
    if w_src is None:
        return np.median(disp[nn], axis=1)
    w_nn = w_src[idx][nn]
    return np.stack([_weighted_median_vectors(disp[nn[i]], w_nn[i]) for i in range(n_out)])


class TissueFlowAnchors:
    """One frame pair's mutual-NN anchor set, re-readable with anchors left out."""

    __slots__ = ("anchors_um", "disp", "src", "dst", "coords_um", "k")

    def __init__(
        self,
        coords_t: np.ndarray,
        coords_t1: np.ndarray,
        *,
        voxel_size=NATIVE_VOXEL_UM,
        k: int = 20,
        max_um: float = DEFAULT_MAX_LINK_UM,
    ) -> None:
        vs = np.asarray(voxel_size, np.float64)
        self.coords_um = np.asarray(coords_t, np.float64).reshape(-1, 3) * vs
        self.k = int(k)
        self.anchors_um, self.disp, self.src, self.dst = mutual_nn_displacements(
            self.coords_um,
            np.asarray(coords_t1, np.float64).reshape(-1, 3) * vs,
            max_um=max_um,
            return_index=True,
            return_target=True,
        )

    def local_flow(self, i: int, drop_t1=()) -> np.ndarray:
        """Local median displacement at node *i* of frame ``t``, in µm."""
        keep = self.src != int(i)
        if len(drop_t1):
            keep &= ~np.isin(self.dst, np.asarray(drop_t1, np.int64))
        disp = self.disp[keep]
        if len(disp) == 0:
            return np.zeros(3, np.float64)
        if len(disp) < 5:
            return np.median(disp, axis=0)
        d = np.linalg.norm(self.anchors_um[keep] - self.coords_um[int(i)], axis=1)
        kk = min(self.k, len(disp))
        nn = np.argpartition(d, kk - 1)[:kk]
        return np.median(disp[nn], axis=0)


def velocity_for_cached_frames(
    coords_by_frame: list[np.ndarray], *, voxel_size=NATIVE_VOXEL_UM, max_um: float = DEFAULT_MAX_LINK_UM
) -> list[np.ndarray]:
    """``incoming_velocity`` for every frame of a movie; frame 0 gets zeros/False."""
    out = []
    for t, coords in enumerate(coords_by_frame):
        vel, has = incoming_velocity(
            coords, coords_by_frame[t - 1] if t > 0 else None, voxel_size=voxel_size, max_um=max_um
        )
        out.append(np.concatenate([vel, has[:, None].astype(np.float64)], axis=1))
    return out
