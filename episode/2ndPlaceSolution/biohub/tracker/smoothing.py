"""Fork-guarded local line smoothing for final track coordinates."""

from __future__ import annotations

import numpy as np


def smooth_tracks(
    node_t: np.ndarray,
    node_zyx_um: np.ndarray,
    edge_pi: np.ndarray,
    edge_ci: np.ndarray,
    weight: float,
    radius: int = 2,
    degree: int = 1,
    distance_decay: float | None = None,
) -> np.ndarray:
    """Blend simple-chain interior nodes toward a local linear trajectory.

    Edges touching births, deaths, merges, or forks are excluded from the
    smoothing chain.  A node needs history and future inside that guarded
    chain, so division daughters and parents are never extrapolated across the
    topology change.  Coordinates are returned in micrometres; the caller owns
    voxel conversion and integer rounding.
    """
    t = np.asarray(node_t, dtype=np.int64)
    xyz = np.asarray(node_zyx_um, dtype=np.float64)
    pi = np.asarray(edge_pi, dtype=np.int64)
    ci = np.asarray(edge_ci, dtype=np.int64)
    if weight <= 0 or len(pi) == 0:
        return xyz.copy()

    n = len(t)
    indeg = np.zeros(n, dtype=np.int16)
    outdeg = np.zeros(n, dtype=np.int16)
    np.add.at(indeg, ci, 1)
    np.add.at(outdeg, pi, 1)
    simple = (indeg == 1) & (outdeg == 1)

    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    safe = simple[pi] & simple[ci]
    prev[ci[safe]] = pi[safe]
    nxt[pi[safe]] = ci[safe]

    out = xyz.copy()
    for node in np.flatnonzero((prev >= 0) & (nxt >= 0)):
        rows = [int(node)]
        cur = int(node)
        for _ in range(radius):
            cur = int(prev[cur])
            if cur < 0:
                break
            rows.append(cur)
        cur = int(node)
        for _ in range(radius):
            cur = int(nxt[cur])
            if cur < 0:
                break
            rows.append(cur)
        rows_a = np.asarray(rows, dtype=np.int64)
        tt = t[rows_a].astype(np.float64)
        if len(rows_a) < 3 or np.unique(tt).size < 3:
            continue
        # Centre time at the node: the requested fitted value is then the
        # intercept, and quadratic fits stay numerically stable near t=99.
        xx = tt - float(t[node])
        design = np.column_stack([xx ** power for power in range(degree + 1)])
        target = xyz[rows_a]
        if distance_decay is not None:
            root_w = np.sqrt(np.exp(-np.abs(xx) / float(distance_decay)))
            design = design * root_w[:, None]
            target = target * root_w[:, None]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = coef[0]
        out[node] = (1.0 - weight) * xyz[node] + weight * fitted
    return out
