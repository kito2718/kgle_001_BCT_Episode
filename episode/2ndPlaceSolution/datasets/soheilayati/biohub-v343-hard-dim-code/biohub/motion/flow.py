"""Frame-to-frame motion estimation, in micrometres — PLAN.md §4.1.

Two levels, both estimated from the point clouds alone so they can be used
inside an oracle without leaking ground truth:

  global   one robust translation per frame pair
  local    a smooth residual field, for tissue that does not move rigidly

Everything takes and returns **micrometres**. Voxels are anisotropic by 4x in z
(CLAUDE.md §7), so a nearest neighbour computed on raw voxel coordinates is
wrong by up to 4x along z and would silently favour the wrong parent.
"""

from __future__ import annotations

import numpy as np


def estimate_global_translation(
    a_um: np.ndarray,
    b_um: np.ndarray,
    gate_um: float = 12.0,
    n_iter: int = 8,
    tol_um: float = 0.02,
) -> np.ndarray:
    """Robust translation carrying `a` onto `b`, as a `(3,)` vector.

    Iterated closest point with a *median* update rather than a mean: at true
    density most nearest neighbours are the wrong cell, and a mean would be
    dragged by them. The median needs only half the pairs to be right.
    """
    from scipy.spatial import cKDTree

    shift = np.zeros(3)
    if len(a_um) < 3 or len(b_um) < 3:
        return shift

    tree = cKDTree(b_um)
    for _ in range(n_iter):
        d, j = tree.query(a_um + shift)
        m = d < gate_um
        if m.sum() < 3:
            break
        delta = np.median(b_um[j[m]] - (a_um[m] + shift), axis=0)
        shift = shift + delta
        if np.linalg.norm(delta) < tol_um:
            break
    return shift


def confident_correspondences(
    a_um: np.ndarray,
    b_um: np.ndarray,
    shift: np.ndarray,
    gate_um: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours after `shift`, within `gate_um`.

    Mutual-NN is the cheap proxy for "unambiguous": if `a` prefers `b` and `b`
    prefers `a` and they are close, the pair is usually real even in dense
    tissue. These anchor the local field.
    """
    from scipy.spatial import cKDTree

    if len(a_um) == 0 or len(b_um) == 0:
        return np.zeros(0, int), np.zeros(0, int)

    src = a_um + shift
    d_ab, j = cKDTree(b_um).query(src)
    _, i = cKDTree(src).query(b_um)

    ia = np.arange(len(src))
    mutual = (i[j] == ia) & (d_ab < gate_um)
    return ia[mutual], j[mutual]


def estimate_local_flow(
    a_um: np.ndarray,
    b_um: np.ndarray,
    shift: np.ndarray | None = None,
    k: int = 12,
    anchor_gate_um: float = 5.0,
    smooth_um: float = 40.0,
) -> np.ndarray:
    """Per-point displacement `(N, 3)` for every point of `a`.

    Global translation plus a smooth residual interpolated from confident
    correspondences. `smooth_um` sets how far an anchor's influence reaches; too
    small and the field memorises individual pairs instead of tissue motion.
    """
    from scipy.spatial import cKDTree

    if shift is None:
        shift = estimate_global_translation(a_um, b_um)
    flow = np.broadcast_to(shift, (len(a_um), 3)).copy()
    if len(a_um) == 0:
        return flow

    ia, jb = confident_correspondences(a_um, b_um, shift, anchor_gate_um)
    if len(ia) < k:
        return flow  # too few anchors to fit anything smooth; keep the global term

    anchors = a_um[ia]
    residual = b_um[jb] - (a_um[ia] + shift)

    kk = min(k, len(ia))
    d, idx = cKDTree(anchors).query(a_um, k=kk)
    d = np.atleast_2d(d.T).T
    idx = np.atleast_2d(idx.T).T

    w = np.exp(-0.5 * (d / smooth_um) ** 2)
    w_sum = w.sum(axis=1, keepdims=True)
    w = np.where(w_sum > 0, w / np.maximum(w_sum, 1e-12), 0.0)
    return flow + (w[..., None] * residual[idx]).sum(axis=1)
